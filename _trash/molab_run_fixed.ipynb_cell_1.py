# ── STANDARD Python IMPORTS ──
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

# ── CODEBASE DOWNLOAD ──
HF_REPO = "Satabarto/Molab_Checkpoints_Cost_AWARE"
HF_TOKEN = "REDACTED"
ZIP_NAME = "Cost-Aware-Test-Time-upload.zip"

if Path(ZIP_NAME).exists():
    os.remove(ZIP_NAME)

downloaded = False

# 1. Primary: Hugging Face Hub (Fast & Reliable - 13.5MB)
try:
    print("Downloading project zip from HuggingFace...")
    subprocess.run("pip install -q huggingface_hub", shell=True)
    from huggingface_hub import hf_hub_download
    hf_path = hf_hub_download(repo_id=HF_REPO, filename=ZIP_NAME, repo_type="dataset", token=HF_TOKEN)
    shutil.copy(hf_path, ZIP_NAME)
    downloaded = True
    print("Downloaded from HuggingFace Hub successfully!")
except Exception as e:
    print(f"HF download failed: {e}")

# 2. Fallback: Google Drive (old zip, different structure)
if not downloaded or not Path(ZIP_NAME).exists():
    GDRIVE_FILE_ID = "1F7lBEOBDC9FNHyK-gjOLoEQ24SpzC0tr"
    subprocess.run("pip install -q gdown", shell=True)
    import gdown
    print("Downloading from Google Drive via gdown (fallback)...")
    gdown.download(id=GDRIVE_FILE_ID, output=ZIP_NAME, quiet=False)

if Path(ZIP_NAME).exists():
    print(f"Extracting {ZIP_NAME}...")
    with zipfile.ZipFile(ZIP_NAME, "r") as z:
        z.extractall(".")

    base_dir = Path(".")
    search_files = ["auto_backup.py", "requirements_molab.txt"]
    if all((base_dir / f).exists() for f in search_files):
        pass
    else:
        for d in sorted(base_dir.iterdir()):
            if d.is_dir() and all((d / f).exists() for f in search_files):
                base_dir = d
                break

    os.environ["NOTEBOOK_DIR"] = str(base_dir.resolve())
    sys.path.append(str(base_dir.resolve()))
    sys.path.insert(0, str((base_dir / "ttc-frugalreason-poc" / "experiment_fr").resolve()))
    print(f"NOTEBOOK_DIR = {os.environ['NOTEBOOK_DIR']}")

    # --- AUTO-FIX SCATTERED LOGS ---
    res_dir = base_dir / "results"
    (res_dir / "block_b_logs").mkdir(parents=True, exist_ok=True)
    (res_dir / "block_a_logs").mkdir(parents=True, exist_ok=True)
    
    for p in base_dir.rglob("*.jsonl"):
        if "block_b" in str(p) or "qwen15b" in p.name or "llama32" in p.name:
            target = res_dir / "block_b_logs" / p.name
            if p.is_file() and p.resolve() != target.resolve():
                try: shutil.copy2(p, target)
                except Exception: pass
        elif "block_a_logs" in p.parts:
            target = res_dir / "block_a_logs" / p.name
            if p.is_file() and p.resolve() != target.resolve():
                try: shutil.copy2(p, target)
                except Exception: pass
    print("Auto-aggregated log files into results/ directory structure.")
    # -------------------------------

else:
    print(f"ERROR: {ZIP_NAME} not found after download attempt.")

# ── THIRD PARTY & LOCAL IMPORTS ──
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import yaml
import sympy
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
from scipy.stats import norm, binomtest, pearsonr
from scipy.optimize import minimize_scalar
from tqdm import tqdm
from tabulate import tabulate
import psutil
import datasets

try:
    from core.ollama_client import OllamaClient
    from core.task_loader import load_all_tasks
    from core.parsers import get_parser
    from core.verifier import OutcomeVerifier
    from core.prompt_manager import get_prompt
    from strategies.frugal_reason_v3 import frugal_reason_v3_evaluate
    print("Local imports loaded successfully.")
except ImportError as e:
    print(f"Local imports failed: {e}")
