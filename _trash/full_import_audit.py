"""
Full audit: find every import used across the experiment code,
then patch the notebook's pip-install cell and third-party-imports cell
to include EVERYTHING.
"""
import json, os, re, glob

NB = "molab_run.ipynb"

# ── 1. Scan all .py files for imports ──────────────────────────────
dirs_to_scan = [
    "ttc-frugalreason-poc/experiment_fr",
    "ttc-task-poc/experiment",
    "rq2_part1",
]

third_party_imports = set()
for d in dirs_to_scan:
    for py in glob.glob(os.path.join(d, "**/*.py"), recursive=True):
        if ".venv" in py or "__pycache__" in py:
            continue
        try:
            with open(py, encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    m = re.match(r"^(?:from|import)\s+([a-zA-Z_][a-zA-Z0-9_]*)", line)
                    if m:
                        pkg = m.group(1)
                        # skip standard library
                        stdlib = {
                            "os", "sys", "json", "time", "subprocess", "zipfile",
                            "shutil", "sqlite3", "importlib", "math", "random",
                            "re", "io", "csv", "gc", "traceback", "glob",
                            "pathlib", "collections", "datetime", "typing",
                            "logging", "copy", "functools", "itertools",
                            "abc", "ast", "warnings", "contextlib", "hashlib",
                            "tempfile", "textwrap", "threading", "multiprocessing",
                            "string", "struct", "enum", "dataclasses", "statistics",
                            "unittest", "argparse", "configparser", "http",
                            "urllib", "socket", "signal", "inspect", "operator",
                            "concurrent", "queue", "heapq", "bisect", "decimal",
                            "fractions", "numbers", "cmath", "pprint",
                        }
                        if pkg not in stdlib:
                            third_party_imports.add(pkg)
        except:
            pass

# Also scan the notebook cells themselves
with open(NB, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell.get("source", []))
    for line in src.split("\n"):
        line = line.strip()
        m = re.match(r"^(?:from|import)\s+([a-zA-Z_][a-zA-Z0-9_]*)", line)
        if m:
            pkg = m.group(1)
            stdlib = {
                "os", "sys", "json", "time", "subprocess", "zipfile",
                "shutil", "sqlite3", "importlib", "math", "random",
                "re", "io", "csv", "gc", "traceback", "glob",
                "pathlib", "collections", "datetime", "typing",
                "logging", "copy", "functools", "itertools",
                "abc", "ast", "warnings", "contextlib", "hashlib",
                "tempfile", "textwrap", "threading", "multiprocessing",
                "string", "struct", "enum", "dataclasses", "statistics",
                "unittest", "argparse", "configparser", "http",
                "urllib", "socket", "signal", "inspect", "operator",
                "concurrent", "queue", "heapq", "bisect", "decimal",
                "fractions", "numbers", "cmath", "pprint", "runpy",
            }
            if pkg not in stdlib:
                third_party_imports.add(pkg)

print("=== ALL third-party imports found across codebase ===")
for p in sorted(third_party_imports):
    print(f"  {p}")

# ── 2. Map import names to pip package names ──────────────────────
# Some imports differ from pip names
import_to_pip = {
    "numpy": "numpy",
    "np": "numpy",
    "pandas": "pandas",
    "pd": "pandas",
    "matplotlib": "matplotlib",
    "plt": "matplotlib",
    "seaborn": "seaborn",
    "sns": "seaborn",
    "scipy": "scipy",
    "requests": "requests",
    "yaml": "pyyaml",
    "pyyaml": "pyyaml",
    "tqdm": "tqdm",
    "tabulate": "tabulate",
    "pynvml": "pynvml",
    "psutil": "psutil",
    "reportlab": "reportlab",
    "fpdf2": "fpdf2",
    "fpdf": "fpdf2",
    "paramiko": "paramiko",
    "huggingface_hub": "huggingface-hub",
    "datasets": "datasets",
    "sympy": "sympy",
    "PIL": "Pillow",
    "ollama": "ollama",
    "torch": "torch",
    "pyarrow": "pyarrow",
}

pip_packages = set()
for imp in third_party_imports:
    if imp in import_to_pip:
        pip_packages.add(import_to_pip[imp])

# Force-add critical ones
pip_packages.add("sympy")
pip_packages.add("numpy")
pip_packages.add("pandas")
pip_packages.add("scipy")
pip_packages.add("matplotlib")
pip_packages.add("seaborn")
pip_packages.add("requests")
pip_packages.add("tqdm")
pip_packages.add("tabulate")
pip_packages.add("pyyaml")
pip_packages.add("pynvml")
pip_packages.add("psutil")
pip_packages.add("huggingface-hub")
pip_packages.add("datasets")

# Remove torch (special install), Pillow (usually present)
pip_packages.discard("torch")

print("\n=== pip packages to install ===")
for p in sorted(pip_packages):
    print(f"  {p}")

# ── 3. Find the pip install cell (Cell 2 area - has HF_REPO) ─────
# and add sympy + any missing pip installs
pip_install_line = "pip install " + " ".join(sorted(pip_packages))

for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell.get("source", []))
    if "HF_REPO = " in src and "ZIP_NAME = " in src:
        # This is the download/setup cell. Add pip install at top
        lines = src.split("\n")
        # Check if there's already a pip install line
        has_pip = any("pip install" in l for l in lines)
        if has_pip:
            # Replace existing pip install line
            new_lines = []
            for l in lines:
                if "pip install" in l and "subprocess" not in l:
                    new_lines.append(f"!pip install -q {' '.join(sorted(pip_packages))}")
                else:
                    new_lines.append(l)
            nb["cells"][i]["source"] = [l + "\n" for l in new_lines]
            nb["cells"][i]["source"][-1] = nb["cells"][i]["source"][-1].rstrip("\n")
        else:
            # Prepend pip install
            install_line = f"!pip install -q {' '.join(sorted(pip_packages))}\n\n"
            nb["cells"][i]["source"] = [install_line] + cell["source"]
        print(f"\nPatched Cell {i} with pip install line")
        break

# ── 4. Patch Cell 3 (third-party imports) to include EVERYTHING ──
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell.get("source", []))
    if "CELL 3: THIRD-PARTY IMPORTS" in src:
        nb["cells"][i]["source"] = [
            "# ── CELL 3: THIRD-PARTY IMPORTS ──\n",
            "# Run this AFTER Cell 1 (pip install) and Cell 2 (Ollama) have completed.\n",
            "import warnings\n",
            "warnings.filterwarnings('ignore', category=SyntaxWarning)\n",
            "\n",
            "import numpy as np\n",
            "import pandas as pd\n",
            "import matplotlib\n",
            "matplotlib.use('Agg')  # headless\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "import requests\n",
            "import yaml\n",
            "import sympy\n",
            "from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application\n",
            "from scipy.stats import norm, binomtest\n",
            "from scipy.optimize import minimize_scalar\n",
            "from tqdm import tqdm\n",
            "from tabulate import tabulate\n",
            "import psutil\n",
            "try:\n",
            "    import pynvml\n",
            "except Exception:\n",
            "    pass\n",
            "try:\n",
            "    import huggingface_hub\n",
            "    from huggingface_hub import HfApi, hf_hub_download\n",
            "except Exception:\n",
            "    pass\n",
            "try:\n",
            "    import datasets\n",
            "except Exception:\n",
            "    pass\n",
            "print('All third-party libraries loaded successfully.')\n",
        ]
        print(f"Patched Cell {i} with comprehensive third-party imports")
        break

# ── 5. Also patch Cell 0 (stdlib) to include warnings suppression ──
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell.get("source", []))
    if "CELL 0: STANDARD Python IMPORTS" in src:
        nb["cells"][i]["source"] = [
            "# ── CELL 0: STANDARD Python IMPORTS ──\n",
            "import os, sys, json, time, subprocess, zipfile, shutil, sqlite3\n",
            "import importlib, importlib.util\n",
            "import math, random, re, io, csv, gc, traceback, glob\n",
            "import warnings, logging, copy, functools, itertools, hashlib\n",
            "import tempfile, textwrap, threading, inspect, operator\n",
            "import ast, enum, dataclasses, statistics, argparse, runpy\n",
            "from pathlib import Path\n",
            "from collections import Counter, defaultdict, OrderedDict\n",
            "from datetime import datetime, timezone, timedelta\n",
            "from typing import Any, Dict, List, Optional, Tuple, Union\n",
            "\n",
            "# Suppress SyntaxWarning globally (from eval() on LLM math output)\n",
            "warnings.filterwarnings('ignore', category=SyntaxWarning)\n",
            "\n",
            "print('All standard Python libraries loaded.')\n",
        ]
        print(f"Patched Cell {i} with comprehensive stdlib imports + warning suppression")
        break

# ── 6. Save ─────────────────────────────────────────────────────
with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("\n=== DONE. Notebook saved. ===")
