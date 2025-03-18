import sys
import os
import re
import argparse
import logging
import yaml
import json
import gzip
import time
import random
import hashlib
import copy
import argparse
from typing import List, Dict, Union, Optional, Iterable


def write_jsonl(filename: str, data: Iterable[dict], append: bool = False):
    """
    Writes an iterable of dictionaries to jsonl
    """
    if append:
        mode = 'ab'
    else:
        mode = 'wb'
    filename = os.path.expanduser(filename)

    with open(filename, mode) as fp:
        for x in data:
            fp.write((json.dumps(x) + "\n").encode('utf-8'))


def extract_code_fences(text):
    pattern = r"```(?:[\w]*?\n)?([\s\S]*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    return matches

