# ── CELL 0: STANDARD Python IMPORTS ──
import os, sys, json, time, subprocess, zipfile, shutil, sqlite3
import importlib, importlib.util
import math, random, re, io, csv, gc, traceback, glob
import warnings, logging, copy, functools, itertools, hashlib
import tempfile, textwrap, threading, inspect, operator
import ast, enum, dataclasses, statistics, argparse, runpy
from pathlib import Path
from collections import Counter, defaultdict, OrderedDict
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

# Suppress SyntaxWarning globally (from eval() on LLM math output)
warnings.filterwarnings('ignore', category=SyntaxWarning)

print('All standard Python libraries loaded.')
