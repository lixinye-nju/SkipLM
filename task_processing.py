"""
Task Processing Framework: A reusable producer-consumer pattern implementation
for parallel processing of tasks with high GPU utilization.
"""

import multiprocessing.synchronize
import os
import sys
import time
import signal
import queue
import logging
import multiprocessing

from datetime import datetime
from typing import Dict, List, Any, Union, Optional, Callable, Iterator, Tuple, TypeVar
from multiprocessing import Process, Event, Manager, Value, cpu_count, Queue as MPQueue
from multiprocessing.managers import SyncManager
from queue import Queue
from contextlib import contextmanager
from copy import deepcopy
from tqdm import tqdm

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


# Type variables for generic task and result
T = TypeVar('T')  # Task type
R = TypeVar('R')  # Result type

# Constants
DEFAULT_TIMEOUT = 300  # 5 minutes timeout for operations
MAX_RETRY_ATTEMPTS = 3  # Max retries for failed operations


@contextmanager
def timeout_handler(seconds: int = DEFAULT_TIMEOUT, error_message: str = "Operation timed out"):
    """Context manager to handle timeouts for operations."""
    def signal_handler(signum, frame):
        raise TimeoutError(error_message)
    
    # Save the original signal handler to restore it later
    original_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        # Restore original signal handler
        signal.signal(signal.SIGALRM, original_handler)


class ResultsManager:
    """Manages result collection, writing, and statistics."""
    
    def __init__(self, total_tasks: int, result_handler: Callable[[List[R]], None]):
        """
        Initialize the ResultsManager.
        
        Args:
            total_tasks: Total number of tasks to process
            result_handler: Function that handles batches of results (e.g., writing to file)
        """
        self.total_tasks = total_tasks
        self.result_handler = result_handler
        self.buffer = []
        manager = Manager()
        self.buffer_lock = manager.Lock()

        self.progress_bar = None
        self.should_stop = Event()
        self.batch_size = 16  # Default batch size for processing results
        
        # Initialize progress bar
        self._init_progress_bar()
    
    def _init_progress_bar(self):
        """Initialize the progress bar."""
        self.progress_bar = tqdm(
            total=self.total_tasks, 
            dynamic_ncols=True, 
            desc="Processing tasks",
            position=0
        )
        self._update_progress_bar(0)
    
    def _update_progress_bar(self, increment: int = 0):
        """Update the progress bar with current statistics."""
        if self.progress_bar is not None:
            if increment > 0:
                self.progress_bar.update(increment)
            
    
    def add_result(self, result: Union[R, List[R]], success: Optional[bool] = None):
        """
        Add a result or list of results to the buffer.
        
        Args:
            result: Single result or list of results
            success: Whether the result represents a success (True) or failure (False)
                     If None, doesn't update the pass counter
        """
        if result is None:
            return
            
        if not isinstance(result, list):
            result = [result]
        
        # Check stop signal before acquiring lock to prevent deadlock during shutdown
        if self.should_stop.is_set():
            return
            
        try:
            # For Manager locks, we can't use timeout parameters, so we'll use a simple acquire/release
            self.buffer_lock.acquire()
            try:
                self.buffer.extend(result)
                
                # Update progress bar
                self._update_progress_bar(len(result))
                
                # Process results if buffer is large enough
                if len(self.buffer) >= self.batch_size:
                    self._flush_buffer_internal()
            finally:
                self.buffer_lock.release()
        except Exception as e:
            print(f"Error adding result: {e}")
    
    def _flush_buffer_internal(self):
        """Process buffered results using the handler function (assumes lock is already held)."""
        if not self.buffer:
            return
                
        try:
            self.result_handler(self.buffer)
            self.buffer = []
        except Exception as e:
            logger.error(f"Error processing results: {e}")
            # Keep buffer to retry later
    
    def _flush_buffer(self):
        """Process buffered results using the handler function (acquires lock)."""
        try:
            # For Manager locks, we can't use timeout parameters
            self.buffer_lock.acquire()
            try:
                self._flush_buffer_internal()
            finally:
                self.buffer_lock.release()
        except Exception as e:
            logger.error(f"Error flushing buffer: {e}")
    
    def result_processor(self, input_queue: MPQueue):
        """Process that receives results from the queue and manages processing."""
        try:
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
    
            while not self.should_stop.is_set():
                try:
                    # Try to get a result with timeout
                    try:
                        result = input_queue.get(timeout=0.5)
                    except (EOFError, ConnectionError):
                        logger.info("Result queue connection lost. Shutting down result processor.")
                        break
                        
                    if result is None:  # Sentinel value
                        logger.info("Result processor received sentinel. Finishing up.")
                        break
                        
                    # Unpack the result and success flag if provided as tuple
                    if isinstance(result, tuple) and len(result) == 2:
                        result_data, success = result
                    else:
                        result_data, success = result, None
                        
                    self.add_result(result_data, success)
                    
                except queue.Empty:
                    # No data in queue, check if we should flush buffer
                    self._flush_buffer()
                except Exception as e:
                    logger.error(f"Error in result processor: {e}")
                    if "Broken pipe" in str(e) or "Connection reset" in str(e):
                        logger.info("Connection to main process lost. Exiting.")
                        break
                    
            # Final flush before exiting
            self._flush_buffer()
            logger.info("Result processor completed")
            
        except KeyboardInterrupt:
            logger.info("Result processor interrupted")
        finally:
            # Don't close the progress bar here, just mark it as completed
            # The main process will handle proper cleanup
            if self.progress_bar is not None:
                try:
                    # Just update the final state but don't close
                    self.progress_bar.refresh()
                except:
                    pass
    
    def stop(self):
        """Signal the manager to stop operations."""
        logger.info("Stopping results manager...")
        self.should_stop.set()
        
        try:
            self._flush_buffer()
        except:
            logger.error("Error during final buffer flush, but continuing shutdown")
            
        # Instead of closing the progress bar here, just disable updates
        if self.progress_bar is not None:
            try:
                self.progress_bar.disable = True  # This prevents updates without closing
            except:
                pass
                
        logger.info("Results manager stopped.")
    
    def __del__(self):
        """Ensure progress bar is closed when the object is garbage collected."""
        if hasattr(self, 'progress_bar') and self.progress_bar is not None:
            try:
                self.progress_bar.close()
            except:
                pass


class TaskWorker:
    """Worker process that continuously processes tasks from a task queue."""
    
    def __init__(
        self, 
        worker_id: int, 
        task_queue: Queue, 
        result_queue: Queue, 
        processor_initializer: Callable[[], Any],
        processor_args: Dict[str, Any],
        process_func: Callable[[Any, T], R],
        stop_event: multiprocessing.synchronize.Event
    ):
        """
        Initialize a task worker.
        
        Args:
            worker_id: Unique ID for this worker
            task_queue: Queue to fetch tasks from
            result_queue: Queue to put results into
            processor_initializer: Function to initialize the task processor
            processor_args: Arguments to pass to the processor initializer
            process_func: Function to process a single task
            stop_event: Event to signal when to stop processing
        """
        self.worker_id = worker_id
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.processor_initializer = processor_initializer
        self.processor_args = processor_args
        self.process_func = process_func
        self.stop_event = stop_event
        self.processor = None
        
    def initialize(self):
        """Initialize the task processor (called within the worker process)."""
        try:
            # Initialize processor with worker-specific arguments
            processor_args = deepcopy(self.processor_args)
            processor_args["worker_id"] = self.worker_id
            
            self.processor = self.processor_initializer(**processor_args)
            logger.info(f"Worker {self.worker_id} initialized")
        except Exception as e:
            logger.error(f"Error initializing worker {self.worker_id}: {e}")
            raise
        
    def run(self):
        """Main worker loop that processes tasks from the queue."""
        try:
            self.initialize()
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
    
            while not self.stop_event.is_set():
                try:
                    # Try to get a task with timeout
                    try:
                        task = self.task_queue.get(timeout=0.5)
                    except (EOFError, ConnectionError):
                        logger.info(f"Worker {self.worker_id}: Task queue connection lost")
                        break
                        
                    if task is None:  # Sentinel value
                        break
                        
                    # Process the task with retries
                    for attempt in range(MAX_RETRY_ATTEMPTS):
                        try:
                            # Add worker metadata and timestamp
                            if isinstance(task, dict):
                                task_copy = deepcopy(task)
                                task_copy["_worker_id"] = self.worker_id
                                task_copy["_timestamp"] = datetime.now().isoformat()
                            else:
                                task_copy = task
                                
                            # Process the task with timeout
                            with timeout_handler(DEFAULT_TIMEOUT, f"Task timed out"):
                                result = self.process_func(self.processor, task_copy)
                                
                            # Put result in queue
                            if result is not None:
                                try:
                                    # Use non-blocking put with timeout to prevent hanging
                                    self.result_queue.put(result, timeout=1)
                                except queue.Full:
                                    logger.warning(f"Worker {self.worker_id}: Result queue full, dropping result")
                                except (EOFError, ConnectionError, BrokenPipeError):
                                    logger.info(f"Worker {self.worker_id}: Result queue connection lost")
                                    return
                            break
                            
                        except Exception as e:
                            if attempt == MAX_RETRY_ATTEMPTS - 1:
                                logger.error(f"Worker {self.worker_id}: Failed task after {MAX_RETRY_ATTEMPTS} attempts: {e}")
                            else:
                                # Exponential backoff
                                delay = 2 ** attempt
                                logger.info(f"Worker {self.worker_id}: Retrying task in {delay}s (attempt {attempt+1})")
                                time.sleep(delay)
                    
                    # No task_done for multiprocessing.Queue
                    pass  # We're tracking completion elsewhere
                    
                except queue.Empty:
                    # No tasks available, check if we should stop
                    continue
                except Exception as e:
                    logger.error(f"Worker {self.worker_id} error: {e}")
                    if "Broken pipe" in str(e) or "Connection reset" in str(e):
                        logger.info(f"Worker {self.worker_id}: Connection to main process lost. Exiting.")
                        break
                    
        except KeyboardInterrupt:
            logger.info(f"Worker {self.worker_id} interrupted")
        except Exception as e:
            logger.error(f"Worker {self.worker_id} failed: {e}")
        finally:
            logger.info(f"Worker {self.worker_id} shutting down")


class CleanupException(Exception):
    """Raised when cleanup has been performed."""
    pass


class TaskProcessingEngine:
    """A reusable engine for parallel task processing using producer-consumer pattern."""
    
    def __init__(
        self,
        processor_initializer: Callable[[], Any],
        processor_args: Dict[str, Any],
        process_func: Callable[[Any, T], R],
        result_handler: Callable[[List[R]], None],
        num_workers: Optional[int] = None,
        task_queue_size: int = 10000
    ):
        """
        Initialize the task processing engine.
        
        Args:
            processor_initializer: Function to initialize the task processor in each worker
            processor_args: Arguments to pass to the processor initializer
            process_func: Function to process a single task
            result_handler: Function to handle batches of results
            num_workers: Number of worker processes (defaults to CPU count)
            task_queue_size: Maximum size of the task queue
        """
        self.processor_initializer = processor_initializer
        self.processor_args = processor_args
        self.process_func = process_func
        self.result_handler = result_handler
        self.num_workers = num_workers or min(4, cpu_count())
        self.task_queue_size = task_queue_size
        
        # Internal state
        self.workers = []
        self.task_queue = None
        self.result_queue = None
        self.results_manager = None
        self.stop_event = Event()
        self.result_process = None
        self.manager = None
        
        # Register signal handlers for clean shutdown - ONLY in the main process
        self._original_sigint_handler = signal.getsignal(signal.SIGINT)
        self._original_sigterm_handler = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _signal_handler(self, sig, frame):
        """Handle termination signals for clean shutdown."""
        logger.info(f"\nReceived signal {sig}. Initiating clean shutdown...")
        self.stop_event.set()
        
        # Only do cleanup if setup has already been done
        if self.workers or self.result_process:
            self._cleanup()
            
        # Restore original signal handlers and re-raise the signal
        signal.signal(signal.SIGINT, self._original_sigint_handler)
        signal.signal(signal.SIGTERM, self._original_sigterm_handler)
        
        raise CleanupException(f"Cleanup performed due to signal {sig}")

        
    def _setup(self, total_tasks: int):
        """Set up queues, workers, and result manager."""
        # Create a manager for sharing objects between processes
        self.manager = SyncManager()
        self.manager.start()
        
        # Create queues
        self.task_queue = MPQueue(maxsize=self.task_queue_size)
        self.result_queue = MPQueue()
        
        # Set up results manager
        self.results_manager = ResultsManager(total_tasks, self.result_handler)
        
        # Start result processor
        self.result_process = Process(
            target=self.results_manager.result_processor,
            args=(self.result_queue,),
            daemon=True,  # Make it a daemon so it exits when main process exits
        )
        self.result_process.start()
        
        # Create and start worker processes
        self.workers = []
        for i in range(self.num_workers):
            worker = TaskWorker(
                worker_id=i,
                task_queue=self.task_queue,
                result_queue=self.result_queue,
                processor_initializer=self.processor_initializer,
                processor_args=self.processor_args,
                process_func=self.process_func,
                stop_event=self.stop_event
            )
            
            p = Process(target=worker.run, 
                        daemon=True)  # Make it a daemon
            p.start()
            self.workers.append(p)
            
        logger.info(f"Started {len(self.workers)} worker processes")
    

    def _cleanup(self):
        """Clean up resources and stop workers."""
        # # Only perform cleanup once
        # if hasattr(self, '_cleanup_done') and self._cleanup_done:
        #     return
        
        logger.info("Cleaning up resources...")
        
        # Mark cleanup as in progress to prevent multiple cleanup attempts
        # self._cleanup_done = True
        
        # Signal stop to all components
        self.stop_event.set()
        
        try:
            # STEP 1: Signal workers to finish by sending sentinel values to task queue
            if self.task_queue:
                try:
                    # Add sentinel values without blocking
                    for _ in range(len(self.workers)):
                        try:
                            self.task_queue.put(None, block=False)
                        except queue.Full:
                            pass
                        except Exception as e:
                            logger.error(f"Error putting sentinel in task queue: {e}")
                except Exception as e:
                    logger.error(f"Error accessing task queue for sentinels: {e}")
            
            # STEP 2: Give workers a short time to exit gracefully
            timeout = 5  # seconds
            start_time = time.time()
            
            # Wait for workers with timeout
            running_workers = [w for w in self.workers if w.is_alive()]
            while running_workers and time.time() - start_time < timeout:
                time.sleep(0.1)
                running_workers = [w for w in self.workers if w.is_alive()]
            
            # STEP 3: Terminate any workers that didn't exit gracefully
            for i, w in enumerate(self.workers):
                if w.is_alive():
                    logger.info(f"Terminating worker {i}...")
                    try:
                        w.terminate()
                    except Exception as e:
                        logger.error(f"Error terminating worker {i}: {e}")
            
            # STEP 4: Signal result processor to finish by sending sentinel
            if self.result_queue:
                try:
                    self.result_queue.put(None, block=False)
                except Exception as e:
                    logger.error(f"Error sending sentinel to result queue: {e}")
            
            # STEP 5: Wait for result processor with short timeout
            if self.result_process and self.result_process.is_alive():
                self.result_process.join(timeout=1)
                
                if self.result_process.is_alive():
                    logger.info("Terminating result processor...")
                    try:
                        self.result_process.terminate()
                    except Exception as e:
                        logger.error(f"Error terminating result processor: {e}")
            
            # STEP 6: Stop the results manager (which handles the progress bar)
            if self.results_manager:
                try:
                    self.results_manager.stop()
                except Exception as e:
                    logger.error(f"Error stopping results manager: {e}")
            
            # STEP 7: Close queues
            if self.task_queue:
                try:
                    self.task_queue.close()
                except Exception as e:
                    logger.error(f"Error closing task queue: {e}")
                    
            if self.result_queue:
                try:
                    self.result_queue.close()
                except Exception as e:
                    logger.error(f"Error closing result queue: {e}")
            
            # STEP 8: Shutdown the manager
            if self.manager:
                try:
                    self.manager.shutdown()
                except Exception as e:
                    logger.error(f"Error shutting down manager: {e}")
                
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            
        # Reset state
        self.workers = []
        self.result_process = None
        self.task_queue = None
        self.result_queue = None
        self.manager = None
        
        # Clean up any remaining progress bar
        if self.results_manager and hasattr(self.results_manager, 'progress_bar') and self.results_manager.progress_bar:
            try:
                # Only do this as the final step of cleanup
                self.results_manager.progress_bar.close()
            except:
                pass
        
        self.results_manager = None
        
        logger.info("Cleanup complete.")
        return
   
    def process_tasks(self, tasks: List[T]) -> Dict[str, Any]:
        """
        Process a list of tasks using multiple workers.
        
        Args:
            tasks: List of tasks to process
            
        Returns:
            Dictionary with statistics about the processing
        """

        total_tasks = len(tasks)
        
        try:
            # Set up processing pipeline
            self._setup(total_tasks)
            
            # Enqueue tasks with progress reporting
            logger.info(f"Enqueueing {total_tasks} tasks...")

            # Queue all tasks

            for task in tasks:
                # Check if we should stop
                if self.stop_event.is_set():
                    logger.info("Stop event detected during task enqueuing, exiting early")
                    break
                    
                # Try to put task in queue, with periodic checks for interruption
                while not self.stop_event.is_set():
                    try:
                        self.task_queue.put(task, timeout=0.5)
                        # task_enqueue_progress.update(1)
                        break
                    except queue.Full:
                        # Queue is full, wait a bit
                        time.sleep(0.1)
                    except (EOFError, BrokenPipeError, ConnectionError):
                        logger.error("Task queue connection lost.")
                        self.stop_event.set()
                        break
            
            if not self.stop_event.is_set():
                logger.info("All tasks enqueued")
                
                # Add sentinel values to signal workers to exit
                for _ in range(self.num_workers):
                    try:
                        self.task_queue.put(None, timeout=0.5)
                    except queue.Full:
                        logger.warning("Warning: Task queue full, couldn't add all sentinel values")
                        break
                    except (EOFError, BrokenPipeError):
                        logger.error("Task queue connection lost during sentinel placement.")
                        break
                
                # Wait for all workers to finish
                worker_timeout = 30  # 30 seconds timeout
                start_time = time.time()
                active_workers = len([w for w in self.workers if w.is_alive()])
                
                while active_workers > 0 and time.time() - start_time < worker_timeout:
                    time.sleep(0.5)
                    active_workers = len([w for w in self.workers if w.is_alive()])
                    
                if active_workers > 0:
                    logger.warning(f"Worker timeout reached. {active_workers} workers still running.")
                else:
                    logger.info("All workers finished successfully.")
                    
                logger.info("Sending sentinel to result processor.")
                    
                # Signal result processor to finish
                try:
                    self.result_queue.put(None, timeout=0.5)
                except (queue.Full, EOFError, BrokenPipeError):
                    logger.error("Could not send sentinel to result processor.")
                
                # Wait for result processor with timeout
                result_timeout = 5  # 5 seconds timeout
                if self.result_process:
                    self.result_process.join(timeout=result_timeout)
                
                if self.result_process and self.result_process.is_alive():
                    logger.warning("Result processor didn't terminate in time, will be handled in cleanup.")
                else:
                    logger.info("Result processor completed successfully.")

        except CleanupException as e:
            logger.info(str(e))
        except Exception as e:
            logger.error(f"Error during processing: {e}")
        finally:
            # Only call cleanup if the stop event hasn't been set
            # (if it was set, cleanup was already called by the signal handler)
            if not self.stop_event.is_set():
                # Restore original signal handlers before cleanup
                signal.signal(signal.SIGINT, self._original_sigint_handler)
                signal.signal(signal.SIGTERM, self._original_sigterm_handler)
                # Clean up
                self._cleanup()
            
            return


# Example usage - will be executed if this file is run directly
if __name__ == "__main__":
    # Set up basic logging configuration
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - [%(process)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # This is a simple example that demonstrates the task processing framework
    
    # 1. Define your processor initialization function
    def init_processor(model_path=None, worker_id=None):
        # In a real application, this would load your model or initialize your processing pipeline
        logger.info(f"Initializing processor {worker_id} with model {model_path}")
        return {"model": "dummy_model", "worker_id": worker_id}
    
    # 2. Define your task processing function
    def process_task(processor, task):
        # Process a single task using the processor
        logger.info(f"Worker {processor['worker_id']} processing task {task}")
        # Simulate some work
        time.sleep(0.1)
        # Return result and success flag
        return {"result": f"Processed {task}", "task_id": task}, True
    
    # 3. Define your result handler
    def handle_results(results):
        # Handle a batch of results (e.g., write to file)
        logger.info(f"Handling batch of {len(results)} results")
        # In a real application, you might write results to a file or database
    
    # 4. Create and use the task processing engine
    engine = TaskProcessingEngine(
        processor_initializer=init_processor,
        processor_args={"model_path": "path/to/model"},
        process_func=process_task,
        result_handler=handle_results,
        num_workers=2,
        task_queue_size=50  # Small queue size for this example
    )
    
    # 5. Process tasks
    logger.info("Creating sample tasks...")
    tasks = [f"task_{i}" for i in range(20)]  # Small number of tasks for testing
    
    logger.info("Starting task processing...")
    engine.process_tasks(tasks)
    
    logger.info(f"Processing completed")