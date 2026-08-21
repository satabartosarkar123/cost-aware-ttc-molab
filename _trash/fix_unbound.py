"""
Fix UnboundLocalError in all wrapped cells.
Strategy: Remove duplicate imports from inside subroutines that are 
already imported globally in Cell 1. This prevents Python from 
treating them as local variables.
"""
import json
import re

def fix_notebook():
    with open('molab_run.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # These are the exact import lines from Cell 1 that get duplicated in later cells.
    # We strip them from inside subroutines since they're already global.
    CELL1_IMPORT_LINES = [
        "import os, subprocess, zipfile, shutil, sys, importlib.util",
        "import os, sys, json, time, subprocess, zipfile, shutil, sqlite3",
        "import importlib, importlib.util",
        "import math, random, re, io, csv, gc, traceback, glob",
        "import warnings, logging, copy, functools, itertools, hashlib",
        "import tempfile, textwrap, threading, inspect, operator",
        "import ast, enum, dataclasses, statistics, argparse, runpy",
        "from pathlib import Path",
        "from collections import Counter, defaultdict, OrderedDict",
        "from datetime import datetime, timezone, timedelta",
        "from typing import Any, Dict, List, Optional, Tuple, Union",
        # Also catch standalone versions
        "import os",
        "import sys",
        "import json",
        "import time",
        "import subprocess",
        "import zipfile",
        "import shutil",
        "import sqlite3",
        "import math",
        "import random",
        "import re",
        "import io",
        "import csv",
        "import gc",
        "import traceback",
        "import glob",
        "import warnings",
        "import logging",
        "import copy",
        "import functools",
        "import itertools",
        "import hashlib",
        "import tempfile",
        "import textwrap",
        "import threading",
        "import inspect",
        "import operator",
        "import ast",
        "import importlib",
    ]

    problem_cells = [32, 34, 36, 38, 40, 42, 43, 45, 47]
    fixed_count = 0

    for i in problem_cells:
        cell = nb['cells'][i]
        src = "".join(cell['source'])
        
        if f'def execute_cell_{i}()' not in src:
            print(f"Cell {i}: NOT wrapped, skipping")
            continue
        
        lines = src.split('\n')
        new_lines = []
        removed = []
        
        for line in lines:
            stripped = line.strip()
            # Check if this line is a duplicate import (already in Cell 1)
            if stripped in CELL1_IMPORT_LINES:
                removed.append(stripped)
                continue
            new_lines.append(line)
        
        if removed:
            new_src = '\n'.join(new_lines)
            cell['source'] = [l + '\n' for l in new_src.split('\n')]
            cell['source'][-1] = cell['source'][-1].rstrip('\n')
            fixed_count += 1
            print(f"Cell {i}: Removed {len(removed)} duplicate imports: {removed}")
        else:
            print(f"Cell {i}: No duplicate imports found")

    with open('molab_run.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    
    print(f"\nDone! Fixed {fixed_count} cells.")

if __name__ == "__main__":
    fix_notebook()
