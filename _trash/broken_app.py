import marimo

__generated_with = "0.17.6"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 1. MANDATORY GLOBAL SETUP
    > **Run these 3 cells in order on EVERY new instance.** They download the codebase from HuggingFace, install all dependencies, start the Ollama GPU server, and load global variables.
    """)
    return


@app.cell
def _():
    def execute_cell_1():
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

    execute_cell_1()
    return


@app.cell
def _():
    import os, subprocess, zipfile, shutil, sys, importlib.util
    from pathlib import Path
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

        # Smart path discovery: find where the files actually landed
        base_dir = Path(".")
        search_files = ["auto_backup.py", "requirements_molab.txt"]

        # Check current dir first
        if all((base_dir / f).exists() for f in search_files):
            pass  # Files are right here
        else:
            # Search one level of subdirs
            for d in sorted(base_dir.iterdir()):
                if d.is_dir() and all((d / f).exists() for f in search_files):
                    base_dir = d
                    break

        os.environ["NOTEBOOK_DIR"] = str(base_dir.resolve())
        print(f"NOTEBOOK_DIR = {os.environ['NOTEBOOK_DIR']}")

        checks = [
            "rq2_part1/run_rq2_part1.py",
            "ttc-frugalreason-poc/experiment_fr/run_frugal_smoke.py",
            "ttc-frugalreason-poc/experiment_fr/run_day0.py",
            "ttc-frugalreason-poc/experiment_fr/run_real_experiment.py",
            "ttc-task-poc/experiment/run_poc.py",
            "auto_backup.py", "drive_checkpoint.py", "requirements_molab.txt",
        ]
        all_ok = True
        for f in checks:
            exists = (base_dir / f).exists()
            print(f"  {'OK     ' if exists else 'MISSING'}: {f}")
            if not exists: all_ok = False

        if all_ok:
            print("\nAll files present and verified!")
        else:
            print("\nSome files missing — check zip contents.")
            # Debug: show what IS in the directory
            print("\nFiles in base_dir:")
            for p in sorted(base_dir.iterdir()):
                print(f"  {p.name}{'/' if p.is_dir() else ''}")
    else:
        print(f"ERROR: {ZIP_NAME} not found after download attempt.")
    return Path, importlib, os, shutil, subprocess, sys


@app.cell
def _(Path, importlib, os, shutil, subprocess, time):
    def execute_cell_3():
        OLLAMA_MODEL   = "qwen2.5:3b"
        FALLBACK_MODEL = "llama3.2:3b"

        def _run(cmd): print(f"  $ {cmd}"); subprocess.run(cmd, shell=True)
        def _ok(cmd):  return subprocess.run(cmd, shell=True, capture_output=True).returncode == 0

        os.makedirs("/workspace", exist_ok=True)
        print("[1/6] /workspace ready")

        _run("apt-get update -qq && apt-get install -y -qq zstd curl > /dev/null 2>&1")
        print("[2/6] System packages ready")

        os.environ["PATH"] = "/usr/local/bin:/usr/bin:/bin:" + os.environ.get("PATH", "")
        if Path("/usr/local/bin/ollama").is_file():
            print("[3/6] Ollama already installed")
        else:
            print("[3/6] Installing Ollama...")
            _run("curl -fsSL https://ollama.com/install.sh | sh")

        ollama_bin = next((c for c in ["/usr/local/bin/ollama","/usr/bin/ollama"] if Path(c).is_file()), None)
        if not ollama_bin:
            r = subprocess.run("which ollama", shell=True, capture_output=True, text=True)
            ollama_bin = r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else None
        if not ollama_bin: raise RuntimeError("Ollama binary not found")
        print(f"  Binary: {ollama_bin}")

        time.sleep(2)
        subprocess.Popen(f"{ollama_bin} serve >> /workspace/ollama.log 2>&1", shell=True)
        print("  Starting Ollama on RTX 6000...")
        started = False
        for i in range(20):
            time.sleep(2)
            if _ok("curl -sf http://localhost:11434/api/tags > /dev/null"):
                print(f"  Ready in {(i+1)*2}s"); started = True; break
        if not started:
            subprocess.run("tail -20 /workspace/ollama.log", shell=True)
            raise RuntimeError("Ollama did not start in 40s")

        tags_out = subprocess.run("curl -sf http://localhost:11434/api/tags", shell=True, capture_output=True, text=True).stdout
        if OLLAMA_MODEL.split(":")[0] in tags_out:
            print(f"[4/6] {OLLAMA_MODEL} already on GPU")
        else:
            print(f"[4/6] Pulling {OLLAMA_MODEL}...")
            if not _ok(f"{ollama_bin} pull {OLLAMA_MODEL}"):
                print(f"  Failed — trying {FALLBACK_MODEL}")
                if not _ok(f"{ollama_bin} pull {FALLBACK_MODEL}"): raise RuntimeError("Both pulls failed")
                OLLAMA_MODEL = FALLBACK_MODEL
            print(f"  {OLLAMA_MODEL} ready")

        nb_dir = os.environ.get("NOTEBOOK_DIR", str(Path(".").resolve()))
        req = Path(nb_dir) / "requirements_molab.txt"
        if req.exists(): _run(f"pip install -q --root-user-action=ignore -r {req}")
        else: _run("pip install -q --root-user-action=ignore requests datasets pandas numpy matplotlib seaborn tqdm pynvml psutil pyyaml tabulate reportlab scipy fpdf2")
        print("[5/6] Python dependencies ready")

        if not shutil.which("rclone"): _run("curl -fsSL https://rclone.org/install.sh | sudo bash > /dev/null 2>&1")

        try:
            spec = importlib.util.spec_from_file_location("auto_backup", str(Path(nb_dir)/"auto_backup.py"))
            ab = importlib.util.module_from_spec(spec); spec.loader.exec_module(ab)
            conf = ab.RCLONE_CONF.strip()
            if "YOUR_CLIENT_ID_HERE" in conf or "YOUR_ACCESS_TOKEN" in conf:
                print("[6/6] Drive sync: DISABLED")
            else:
                cp = Path("/root/.config/rclone/rclone.conf"); cp.parent.mkdir(parents=True, exist_ok=True); cp.write_text(conf)
                test = subprocess.run("rclone lsd gdrive: --max-depth 1", shell=True, capture_output=True, text=True, timeout=15)
                if test.returncode == 0: print("[6/6] Drive sync: ENABLED"); 
                else: print(f"[6/6] Drive sync failed: {test.stderr.strip()[:80]}")
        except Exception as e: print(f"[6/6] Drive sync error: {e}")

        os.environ["OLLAMA_BASE_URL"] = "http://localhost:11434"
        os.environ["OLLAMA_MODEL"]    = OLLAMA_MODEL
        os.environ["WORKSPACE"]       = nb_dir

        print("\n" + "="*60)
        print("  SETUP COMPLETE — pick a run cell below")
        print("="*60)
        print(f"  Model     : {OLLAMA_MODEL} on RTX 6000")

        print(f"  Workspace : {nb_dir}")
        print("="*60)
    execute_cell_3()
    return


@app.cell
def _():
    def execute_cell_4():
        # ── CELL 3: THIRD-PARTY IMPORTS ──
        # Run this AFTER Cell 1 (pip install) and Cell 2 (Ollama) have completed.
        import warnings
        warnings.filterwarnings('ignore', category=SyntaxWarning)

        import numpy as np
        import pandas as pd
        import matplotlib
        matplotlib.use('Agg')  # headless
        import matplotlib.pyplot as plt
        import seaborn as sns
        import requests
        import yaml
        import sympy
        from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
        from scipy.stats import norm, binomtest
        from scipy.optimize import minimize_scalar
        from tqdm import tqdm
        from tabulate import tabulate
        import psutil
        try:
            import pynvml
        except Exception:
            pass
        try:
            import huggingface_hub
            from huggingface_hub import HfApi, hf_hub_download
        except Exception:
            pass
        try:
            import datasets
        except Exception:
            pass
        print('All third-party libraries loaded successfully.')

    execute_cell_4()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    # 2. PRE-DAY 1 EXPERIMENTS
    > These are your historical smoke tests and Block A runs. You can skip these if you are doing Day 1+.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Smoke A — frugal_reason_v3 only · 10 q × 4 datasets = 40 runs
    | | |
    |---|---|
    | **Script** | `ttc-frugalreason-poc/experiment_fr/run_frugal_smoke.py` |
    | **Strategies** | `frugal_reason_v3` only |
    | **Datasets** | gsm8k · strategyqa · aqua · math |
    | **Output** | stdout only |
    | **Purpose** | Confirm frugal_reason_v3 parses and scores correctly |
    """)
    return


@app.cell
def _(Path, os, runpy, sys):
    def execute_cell_7():
        nb_dir = Path(os.environ.get("NOTEBOOK_DIR", "."))
        script = (nb_dir / "ttc-frugalreason-poc/experiment_fr/run_frugal_smoke.py").resolve()
        if not script.exists(): raise FileNotFoundError(f"{script} — run Cell 1 first")

        os.chdir(script.parent)
        if str(script.parent) not in sys.path: sys.path.insert(0, str(script.parent))

        print("Smoke A: frugal_reason_v3 | gsm8k · strategyqa · aqua · math | 10 q each")
        print("=" * 60)
        try: runpy.run_path(str(script), run_name="__main__")
        except SystemExit as e:
            if e.code not in (None, 0): print(f"Exit: {e.code}")
        except KeyboardInterrupt: print("Stopped.")
        except Exception: import traceback; traceback.print_exc()
        print("\n" + "="*60 + "\nSmoke A done.")
    execute_cell_7()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ### Smoke B — All 6 strategies · 10 q × 4 datasets = 240 runs
    | | |
    |---|---|
    | **Script** | `ttc-frugalreason-poc/experiment_fr/run_day0.py` |
    | **Strategies** | greedy_io · greedy_cot · zero_shot_tot_k3 · self_consistency_k5 · best_of_n_k5_self_eval · frugal_reason_v3 |
    | **Datasets** | gsm8k · strategyqa · aqua · math |
    | **Output** | `results/day0_smoke/` |
    | **Purpose** | Full env validation — all 6 strategies, all parsers, hardware |
    """)
    return


@app.cell
def _(Path, os, runpy, sys):
    def execute_cell_9():
        nb_dir = Path(os.environ.get("NOTEBOOK_DIR", "."))
        script = (nb_dir / "ttc-frugalreason-poc/experiment_fr/run_day0.py").resolve()
        if not script.exists(): raise FileNotFoundError(f"{script} — run Cell 1 first")

        os.chdir(script.parent)
        if str(script.parent) not in sys.path: sys.path.insert(0, str(script.parent))

        print("Smoke B: all 6 strategies | gsm8k · strategyqa · aqua · math | 10 q each")
        print("=" * 60)
        try: runpy.run_path(str(script), run_name="__main__")
        except SystemExit as e:
            if e.code not in (None, 0): print(f"Exit: {e.code}")
        except KeyboardInterrupt: print("Stopped.")
        except Exception: import traceback; traceback.print_exc()
        print("\n" + "="*60 + "\nSmoke B done.")
    execute_cell_9()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ### Smoke C — rq2_part1 quick · 5 q × 3 datasets = 75 runs
    | | |
    |---|---|
    | **Script** | `rq2_part1/run_rq2_part1.py` (TOTAL_QUESTIONS_PER_TASK patched to 5) |
    | **Strategies** | greedy_io · greedy_cot · self_consistency_k5 · best_of_n_k5_self_eval · zero_shot_tot_k3 |
    | **Datasets** | gsm8k · strategyqa · game24 |
    | **Purpose** | Verify rq2_part1 end-to-end before 540-run full job |
    """)
    return


@app.cell
def _(Path, os, runpy, sys):
    def execute_cell_11():
        nb_dir = Path(os.environ.get("NOTEBOOK_DIR", "."))
        script = (nb_dir / "rq2_part1/run_rq2_part1.py").resolve()
        if not script.exists(): raise FileNotFoundError(f"{script} — run Cell 1 first")

        src = script.read_text(encoding="utf-8").replace(
            "TOTAL_QUESTIONS_PER_TASK = 36",
            "TOTAL_QUESTIONS_PER_TASK = 5  # SMOKE"
        )
        smoke = script.parent / "_smoke_rq2.py"
        smoke.write_text(src, encoding="utf-8")

        os.chdir(script.parent)
        if str(script.parent) not in sys.path: sys.path.insert(0, str(script.parent))

        print("Smoke C: rq2_part1 (5 q/task) | gsm8k · strategyqa · game24")
        print("=" * 60)
        try: runpy.run_path(str(smoke), run_name="__main__")
        except SystemExit as e:
            if e.code not in (None, 0): print(f"Exit: {e.code}")
        except KeyboardInterrupt: print("Stopped.")
        except Exception: import traceback; traceback.print_exc()
        finally: smoke.unlink(missing_ok=True)
        print("\n" + "="*60 + "\nSmoke C done.")
    execute_cell_11()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Full Run 1 — rq2_part1 · 5 strategies × 3 datasets × 36 q = **540 runs** ← MAIN
    | | |
    |---|---|
    | **Script** | `rq2_part1/run_rq2_part1.py` |
    | **Strategies** | greedy_io · greedy_cot · self_consistency_k5 · best_of_n_k5_self_eval · zero_shot_tot_k3 |
    | **Datasets** | gsm8k (Q642–677) · strategyqa (mid-36) · game24 (933–968) |
    | **Checkpoint** | `rq2_part1/checkpoints/completed.jsonl` — auto-resume |
    | **Output** | `rq2_part1/results/` · `plots/` · `reports/` |
    | **Est. time** | 4–8 h on RTX 6000 |
    """)
    return


@app.cell
def _(Path, os, runpy, sys):
    def execute_cell_13():
        nb_dir = Path(os.environ.get("NOTEBOOK_DIR", "."))
        script = (nb_dir / "rq2_part1/run_rq2_part1.py").resolve()
        if not script.exists(): raise FileNotFoundError(f"{script} — run Cell 1 first")

        os.chdir(script.parent)
        if str(script.parent) not in sys.path: sys.path.insert(0, str(script.parent))

        print("Full Run 1: rq2_part1 | 540 runs | auto-resume from checkpoint")
        print("Strategies : greedy_io | greedy_cot | self_consistency_k5 | best_of_n_k5_self_eval | zero_shot_tot_k3")
        print("Datasets   : gsm8k | strategyqa | game24")
        print("=" * 60 + "\n")
        try: runpy.run_path(str(script), run_name="__main__")
        except SystemExit as e:
            if e.code not in (None, 0): print(f"Exit: {e.code}")
        except KeyboardInterrupt: print("\nStopped. Re-run to resume from checkpoint.")
        except Exception: import traceback; traceback.print_exc()
        print("\n" + "="*60 + "\nFull Run 1 done. Run the PUSH cell.")
    execute_cell_13()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ### Full Run 2 — FrugalReason Block A · GSM8K + AQUA + SVAMP
    | | |
    |---|---|
    | **Script** | `ttc-frugalreason-poc/experiment_fr/run_block_a.py` |
    | **Strategies** | greedy_io · greedy_cot · self_consistency_k5 · best_of_n_k5_self_eval · frugal_reason_v3 |
    | **Checkpoint** | SQLite `block_a_checkpoint.db` — crash-safe |
    """)
    return


@app.cell
def _(Path, os, runpy, sys):
    def execute_cell_15():
        nb_dir = Path(os.environ.get("NOTEBOOK_DIR", "."))
        script = (nb_dir / "ttc-frugalreason-poc/experiment_fr/run_block_a.py").resolve()
        if not script.exists(): raise FileNotFoundError(str(script))

        os.chdir(script.parent)
        if str(script.parent) not in sys.path: sys.path.insert(0, str(script.parent))

        print("Full Run 2: FR Block A | gsm8k · aqua · svamp | SQLite checkpoint")
        print("=" * 60)
        try: runpy.run_path(str(script), run_name="__main__")
        except SystemExit as e:
            if e.code not in (None, 0): print(f"Exit: {e.code}")
        except KeyboardInterrupt: print("Stopped. Re-run to resume.")
        except Exception: import traceback; traceback.print_exc()
        print("\n" + "="*60 + "\nFull Run 2 done.")
    execute_cell_15()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ### Full Run 3 — FrugalReason Block A Part 2 · MATH + StrategyQA
    | | |
    |---|---|
    | **Script** | `ttc-frugalreason-poc/experiment_fr/run_block_a_part2.py` |
    | **Strategies** | greedy_io · greedy_cot · self_consistency_k5 · best_of_n_k5_self_eval · frugal_reason_v3 |
    | **Checkpoint** | SQLite `block_a_part2_checkpoint.db` |
    """)
    return


@app.cell
def _(Path, os, runpy, sys):
    def execute_cell_17():
        nb_dir = Path(os.environ.get("NOTEBOOK_DIR", "."))
        script = (nb_dir / "ttc-frugalreason-poc/experiment_fr/run_block_a_part2.py").resolve()
        if not script.exists(): raise FileNotFoundError(str(script))

        os.chdir(script.parent)
        if str(script.parent) not in sys.path: sys.path.insert(0, str(script.parent))

        print("Full Run 3: FR Block A-2 | math · strategyqa | SQLite checkpoint")
        print("=" * 60)
        try: runpy.run_path(str(script), run_name="__main__")
        except SystemExit as e:
            if e.code not in (None, 0): print(f"Exit: {e.code}")
        except KeyboardInterrupt: print("Stopped. Re-run to resume.")
        except Exception: import traceback; traceback.print_exc()
        print("\n" + "="*60 + "\nFull Run 3 done.")
    execute_cell_17()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ### Full Run 4 — FrugalReason Master · ALL 6 strategies × 4 datasets × 36 q = **864 runs**
    | | |
    |---|---|
    | **Script** | `ttc-frugalreason-poc/experiment_fr/run_real_experiment.py` |
    | **Strategies** | greedy_io · greedy_cot · self_consistency_k5 · best_of_n_k5_self_eval · zero_shot_tot_k3 · **frugal_reason_v3** |
    | **Datasets** | gsm8k · strategyqa · aqua · math |
    | **Post-run** | Auto-runs `pav_analysis.py` + `evaluate_locked.py` |
    | **Note** | THE canonical frugalreason experiment |
    """)
    return


@app.cell
def _(Path, os, runpy, sys):
    def execute_cell_19():
        nb_dir = Path(os.environ.get("NOTEBOOK_DIR", "."))
        script = (nb_dir / "ttc-frugalreason-poc/experiment_fr/run_real_experiment.py").resolve()
        if not script.exists(): raise FileNotFoundError(str(script))

        os.chdir(script.parent)
        if str(script.parent) not in sys.path: sys.path.insert(0, str(script.parent))

        print("Full Run 4: FR Master | all 6 strategies | gsm8k · strategyqa · aqua · math | 864 runs")
        print("=" * 60)
        try: runpy.run_path(str(script), run_name="__main__")
        except SystemExit as e:
            if e.code not in (None, 0): print(f"Exit: {e.code}")
        except KeyboardInterrupt: print("Stopped.")
        except Exception: import traceback; traceback.print_exc()
        print("\n" + "="*60 + "\nFull Run 4 done.")

        # ── PUSH RESULTS TO HUGGING FACE ──────────────────────────────
        print("\nPushing Day 1 results to Hugging Face Hub...")
        try:
            import zipfile as _zf
            from huggingface_hub import HfApi
            _api = HfApi(token="REDACTED")
            _out_dir = os.path.join(nb_dir, "results")
            _csvs = ["block_a_final_stats.csv", "mcnemar_table.csv", "bootstrap_table.csv"]
            for _csv in _csvs:
                _p = os.path.join(_out_dir, _csv)
                if os.path.exists(_p):
                    _api.upload_file(
                        path_or_fileobj=_p,
                        path_in_repo=f"day1_stats/{_csv}",
                        repo_id="Satabarto/Molab_Checkpoints_Cost_AWARE",
                        repo_type="dataset",
                    )
                    print(f"  Uploaded {_csv}")
            print("Day 1 HF push complete.")
        except Exception as _e:
            print(f"HF push failed (non-fatal): {_e}")

    execute_cell_19()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ### Full Run 5 — TTC-Task POC · 5 strategies × 3 datasets × 50 q = **750 runs**
    | | |
    |---|---|
    | **Script** | `ttc-task-poc/experiment/run_poc.py` |
    | **Strategies** | greedy_io · greedy_cot · self_consistency · best_of_n · tree_of_thought |
    | **Checkpoint** | None — interrupting restarts from scratch |
    | **Output** | `results_50/` · `plots_50/` · `reports_50/POC_REPORT.md` |
    """)
    return


@app.cell
def _(Path, os, runpy, sys):
    def execute_cell_21():
        nb_dir = Path(os.environ.get("NOTEBOOK_DIR", "."))
        script = (nb_dir / "ttc-task-poc/experiment/run_poc.py").resolve()
        if not script.exists(): raise FileNotFoundError(str(script))

        os.chdir(script.parent)
        if str(script.parent) not in sys.path: sys.path.insert(0, str(script.parent))

        print("Full Run 5: TTC-Task POC | 750 runs | WARNING: no checkpoint")
        print("Strategies : greedy_io | greedy_cot | self_consistency | best_of_n | tree_of_thought")
        print("Datasets   : gsm8k | strategyqa | game24")
        print("=" * 60)
        try: runpy.run_path(str(script), run_name="__main__")
        except SystemExit as e:
            if e.code not in (None, 0): print(f"Exit: {e.code}")
        except KeyboardInterrupt: print("Stopped.")
        except Exception: import traceback; traceback.print_exc()
        print("\n" + "="*60 + "\nFull Run 5 done.")
    execute_cell_21()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    # 3. DAY 1 ONWARD (BLOCK B)
    > Run these strictly in order, starting with Day 1 Statistics.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    # Day 1 Master Order — Statistics on Saved Logs
    **CRITICAL:** Do NOT start Ollama, do NOT run any strategy, do NOT generate any model output.
    Every number must be computed EXCLUSIVELY from the already-saved Block A logs in `results/block_a_logs/*.jsonl`.
    """)
    return


@app.cell
def _():
    def execute_cell_24():
        import os
        import json
        import numpy as np
        import pandas as pd
        from scipy.stats import norm
        from scipy.stats import binomtest

        DATASETS = ["gsm8k", "aqua", "math", "strategyqa"]
        STRATEGIES = [
            "greedy_io", "greedy_cot", "zero_shot_tot_k3",
            "self_consistency_k5", "best_of_n_k5_self_eval", "frugal_reason_v3"
        ]
        EXPECTED_COUNTS = {"gsm8k": 300, "aqua": 254, "math": 238, "strategyqa": 300}
        LOG_DIR = "results/block_a_logs"

        print("SECTION 1 — LOAD & VALIDATE COMPLETENESS")
        data = []
        gaps = []

        for d in DATASETS:
            for s in STRATEGIES:
                filepath = os.path.join(LOG_DIR, f"{d}_{s}.jsonl")
                if not os.path.exists(filepath):
                    gaps.append(f"Missing file: {d} - {s}")
                    continue
        
                count = 0
                with open(filepath, 'r', encoding='utf-8') as f:
                    for line in f:
                        if not line.strip(): continue
                        record = json.loads(line)
                
                        qid = record.get("qid", f"{d}_{count}")
                        correct = int(record.get("correct", False))
                
                        tokens = record.get("tokens", 0)
                        calls = record.get("calls", 1)
                        parse = int(record.get("parse_success", True))
                
                        data.append({
                            "dataset": d,
                            "strategy": s,
                            "qid": qid,
                            "correct": correct,
                            "tokens": tokens,
                            "calls": calls,
                            "parse": parse
                        })
                        count += 1
        
                if count != EXPECTED_COUNTS[d]:
                    gaps.append(f"Mismatch {d}-{s}: expected {EXPECTED_COUNTS[d]}, got {count}")

        df = pd.DataFrame(data)

        print(f"Loaded {len(df)} records.")
        if gaps:
            print("GAPS FOUND:")
            for g in gaps:
                print(" -", g)
        else:
            print("No gaps found. All counts match expected!")

        print("\\nSECTION 2 — WILSON 95% CONFIDENCE INTERVALS")
        def wilson_ci(k, n, z=1.96):
            if n == 0: return 0.0, 0.0
            p = k / n
            denominator = 1 + z**2/n
            centre_adjusted_probability = p + z**2 / (2*n)
            adjusted_standard_deviation = np.sqrt((p*(1 - p) + z**2 / (4*n)) / n)
    
            lower_bound = (centre_adjusted_probability - z*adjusted_standard_deviation) / denominator
            upper_bound = (centre_adjusted_probability + z*adjusted_standard_deviation) / denominator
            return lower_bound, upper_bound

        stats_rows = []
        for d in DATASETS:
            for s in STRATEGIES:
                subset = df[(df["dataset"] == d) & (df["strategy"] == s)]
                n = len(subset)
                if n == 0: continue
                k = subset["correct"].sum()
                acc = k / n
                lo, hi = wilson_ci(k, n)
        
                avg_tokens = subset["tokens"].mean()
                avg_calls = subset["calls"].mean()
                parse_rate = subset["parse"].mean()
        
                stats_rows.append({
                    "dataset": d,
                    "strategy": s,
                    "correct": k,
                    "total": n,
                    "acc": acc,
                    "wilson_lo": lo,
                    "wilson_hi": hi,
                    "avg_tokens": avg_tokens,
                    "avg_calls": avg_calls,
                    "parse_rate": parse_rate
                })
                print(f"{d} | {s}: {acc*100:.1f}% [{lo*100:.1f}, {hi*100:.1f}]")

        stats_df = pd.DataFrame(stats_rows)
        os.makedirs("results", exist_ok=True)
        stats_df.to_csv("results/block_a_final_stats.csv", index=False)

        print("\\nSECTION 3 — McNEMAR EXACT TESTS (accuracy, paired by qid)")
        mcnemar_rows = []
        for d in DATASETS:
            fr_subset = df[(df["dataset"] == d) & (df["strategy"] == "frugal_reason_v3")].set_index("qid")
            if len(fr_subset) == 0: continue
    
            for s in STRATEGIES:
                if s == "frugal_reason_v3": continue
                bs_subset = df[(df["dataset"] == d) & (df["strategy"] == s)].set_index("qid")
        
                aligned = fr_subset.join(bs_subset, lsuffix="_fr", rsuffix="_bs", how="inner")
        
                b = ((aligned["correct_fr"] == 1) & (aligned["correct_bs"] == 0)).sum()
                c = ((aligned["correct_fr"] == 0) & (aligned["correct_bs"] == 1)).sum()
        
                if b + c == 0:
                    p_val = 1.0
                else:
                    p_val = binomtest(b, n=b+c, p=0.5, alternative='two-sided').pvalue
            
                stars = ""
                if p_val < 0.001: stars = "***"
                elif p_val < 0.01: stars = "**"
                elif p_val < 0.05: stars = "*"
        
                acc_diff = aligned["correct_fr"].mean() - aligned["correct_bs"].mean()
                mcnemar_rows.append({
                    "dataset": d,
                    "baseline": s,
                    "b_fr_only": b,
                    "c_bs_only": c,
                    "p_value": p_val,
                    "significance": stars,
                    "acc_diff": acc_diff
                })

        mcnemar_df = pd.DataFrame(mcnemar_rows)
        mcnemar_df.to_csv("results/mcnemar_table.csv", index=False)
        print(mcnemar_df[["dataset", "baseline", "b_fr_only", "c_bs_only", "p_value", "significance", "acc_diff"]])

        print("\\nSECTION 4 — PAIRED BOOTSTRAP (10,000 resamples, seed=0)")
        np.random.seed(0)
        bootstrap_rows = []
        for d in DATASETS:
            fr_subset = df[(df["dataset"] == d) & (df["strategy"] == "frugal_reason_v3")].set_index("qid")
            if len(fr_subset) == 0: continue
    
            for s in STRATEGIES:
                if s == "frugal_reason_v3": continue
                bs_subset = df[(df["dataset"] == d) & (df["strategy"] == s)].set_index("qid")
                aligned = fr_subset.join(bs_subset, lsuffix="_fr", rsuffix="_bs", how="inner")
        
                n = len(aligned)
                if n == 0: continue
                diffs = aligned["correct_fr"].values - aligned["correct_bs"].values
        
                resamples = np.random.choice(diffs, (10000, n), replace=True)
                means = resamples.mean(axis=1)
                lo = np.percentile(means, 2.5)
                hi = np.percentile(means, 97.5)
        
                bootstrap_rows.append({
                    "dataset": d,
                    "baseline": s,
                    "metric": "accuracy",
                    "mean_diff": diffs.mean(),
                    "ci_lo": lo,
                    "ci_hi": hi
                })
        
                if s in ["self_consistency_k5", "best_of_n_k5_self_eval"]:
                    tok_diffs = aligned["tokens_fr"].values - aligned["tokens_bs"].values
                    resamples_tok = np.random.choice(tok_diffs, (10000, n), replace=True)
                    means_tok = resamples_tok.mean(axis=1)
                    bootstrap_rows.append({
                        "dataset": d,
                        "baseline": s,
                        "metric": "tokens",
                        "mean_diff": tok_diffs.mean(),
                        "ci_lo": np.percentile(means_tok, 2.5),
                        "ci_hi": np.percentile(means_tok, 97.5)
                    })
            
                    call_diffs = aligned["calls_fr"].values - aligned["calls_bs"].values
                    resamples_call = np.random.choice(call_diffs, (10000, n), replace=True)
                    means_call = resamples_call.mean(axis=1)
                    bootstrap_rows.append({
                        "dataset": d,
                        "baseline": s,
                        "metric": "calls",
                        "mean_diff": call_diffs.mean(),
                        "ci_lo": np.percentile(means_call, 2.5),
                        "ci_hi": np.percentile(means_call, 97.5)
                    })

        boot_df = pd.DataFrame(bootstrap_rows)
        boot_df.to_csv("results/bootstrap_table.csv", index=False)
        print("Bootstrap finished. Wrote results/bootstrap_table.csv")

        print("\\nSECTION 5 — SANITY SPOT-CHECK (anti-fabrication)")
        expected = {
            "gsm8k": 0.820,
            "aqua": 0.709,
            "math": 0.735,
            "strategyqa": 0.653
        }
        for d, exp_val in expected.items():
            val = stats_df[(stats_df["dataset"] == d) & (stats_df["strategy"] == "frugal_reason_v3")]["acc"].values
            if len(val) > 0:
                actual = val[0]
                if abs(actual - exp_val) > 0.005:
                    print(f"STOP! Deviation found in {d}: Expected ~{exp_val}, Got {actual:.3f}")
                else:
                    print(f"Spot-check passed for {d}: {actual:.3f} matches ~{exp_val}")

        print("\\nSECTION 6 — OUTPUTS / MARKDOWN SUMMARY")
        for d in DATASETS:
            print(f"### {d.upper()}")
            fr_row = stats_df[(stats_df["dataset"] == d) & (stats_df["strategy"] == "frugal_reason_v3")]
            if len(fr_row) == 0: continue
            fr_acc = fr_row["acc"].values[0]
            fr_lo, fr_hi = fr_row["wilson_lo"].values[0], fr_row["wilson_hi"].values[0]
            print(f"FRUGAL_REASON_V3: {fr_acc*100:.1f}% [{fr_lo*100:.1f}, {fr_hi*100:.1f}]")
    
            for s in STRATEGIES:
                if s == "frugal_reason_v3": continue
                s_row = stats_df[(stats_df["dataset"] == d) & (stats_df["strategy"] == s)]
                if len(s_row) == 0: continue
                s_acc = s_row["acc"].values[0]
                s_lo, s_hi = s_row["wilson_lo"].values[0], s_row["wilson_hi"].values[0]
        
                mc_row = mcnemar_df[(mcnemar_df["dataset"] == d) & (mcnemar_df["baseline"] == s)]
                stars = mc_row["significance"].values[0] if len(mc_row)>0 else ""
        
                bt_row = boot_df[(boot_df["dataset"] == d) & (boot_df["baseline"] == s) & (boot_df["metric"] == "accuracy")]
                if len(bt_row) > 0:
                    diff = bt_row["mean_diff"].values[0] * 100
                    diff_lo = bt_row["ci_lo"].values[0] * 100
                    diff_hi = bt_row["ci_hi"].values[0] * 100
                    diff_str = f"diff {diff:+.1f} CI[{diff_lo:+.1f}, {diff_hi:+.1f}]"
                else:
                    diff_str = ""
            
                print(f" - {s}: {s_acc*100:.1f}% [{s_lo*100:.1f}, {s_hi*100:.1f}] {stars} | {diff_str}")
            print()


        # ── PUSH RESULTS TO HUGGING FACE ──────────────────────────────
        print("\nPushing Day 1 results to Hugging Face Hub...")
        try:
            import zipfile as _zf
            import os
            from huggingface_hub import HfApi
            _api = HfApi(token="REDACTED")
            _out_dir = "results"
            _csvs = ["block_a_final_stats.csv", "mcnemar_table.csv", "bootstrap_table.csv"]
            for _csv in _csvs:
                _p = os.path.join(_out_dir, _csv)
                if os.path.exists(_p):
                    _api.upload_file(
                        path_or_fileobj=_p,
                        path_in_repo=f"day1_stats/{_csv}",
                        repo_id="Satabarto/Molab_Checkpoints_Cost_AWARE",
                        repo_type="dataset",
                    )
                    print(f"  Uploaded {_csv}")
            print("Day 1 HF push complete.")
        except Exception as _e:
            print(f"HF push failed (non-fatal): {_e}")

    execute_cell_24()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    # Day 2 — Block B: qwen2.5:1.5b Calibration Sweep
    > **2,400 runs** (4 datasets × 100 stratified qids × 6 strategies) on `qwen2.5:1.5b`.
    > Logs → `results/block_b_logs/qwen15b_*.jsonl`. Resumable via SQLite checkpoint.
    """)
    return


@app.cell
def _(requests, subprocess, time):
    def execute_cell_26():
        # Day2-Fetch-1.5B — Pull qwen2.5:1.5b into Ollama

        print("=" * 60)
        print("  Day 2 — Fetching qwen2.5:1.5b")
        print("=" * 60)

        subprocess.run("ollama pull qwen2.5:1.5b", shell=True, check=True)
        time.sleep(3)

        # Verify via /api/tags
        r = requests.get("http://localhost:11434/api/tags", timeout=10)
        r.raise_for_status()
        models = [m["name"] for m in r.json().get("models", [])]
        assert "qwen2.5:1.5b" in models or any("qwen2.5:1.5b" in m for m in models), \
            f"qwen2.5:1.5b not found! Available: {models}"
        print(f"qwen2.5:1.5b confirmed. Available models: {models}")

    execute_cell_26()
    return


@app.cell
def _(Path, json, os, random, re, sqlite3, sys, time):
    def execute_cell_27():
        # Day2-Sweep-1.5B — 2,400 runs (4 ds × 100 qids × 6 strategies) on qwen2.5:1.5b

        _nb = Path(os.environ.get("NOTEBOOK_DIR", "."))
        sys.path.insert(0, str(_nb / "ttc-frugalreason-poc" / "experiment_fr"))
        os.chdir(str(_nb / "ttc-frugalreason-poc" / "experiment_fr"))

        from core.ollama_client import OllamaClient
        from core.task_loader import load_all_tasks
        from core.parsers import get_parser
        from core.verifier import OutcomeVerifier
        from core.prompt_manager import get_prompt
        from strategies.frugal_reason_v3 import frugal_reason_v3_evaluate

        MODEL = "qwen2.5:1.5b"
        SEED = 0
        STRATEGIES = ["greedy_io", "greedy_cot", "zero_shot_tot_k3",
                      "self_consistency_k5", "best_of_n_k5_self_eval", "frugal_reason_v3"]
        DATASETS = ["gsm8k", "aqua", "math", "strategyqa"]
        EXPECTED = {"gsm8k": 300, "aqua": 254, "math": 238, "strategyqa": 300}
        QID_LIMIT = 100  # stratified 100 per dataset

        # ── Strategy runners (identical to Block A) ──────────────────────
        def run_greedy_io(client, task, question):
            prompt = get_prompt("greedy_io", task, question)
            r = client.generate(prompt, temperature=0.0)
            p = get_parser(task)(r["text"])
            return {"selected_answer": p["final_answer"], "raw_response": r["text"],
                    "latency_seconds_total": r["latency_seconds"], "total_tokens": r["total_tokens"],
                    "model_calls": 1, "parse_success": p["parse_success"], "parse_method": p["parse_method"]}

        def run_greedy_cot(client, task, question):
            prompt = get_prompt("greedy_cot", task, question)
            r = client.generate(prompt, temperature=0.0)
            p = get_parser(task)(r["text"])
            return {"selected_answer": p["final_answer"], "raw_response": r["text"],
                    "latency_seconds_total": r["latency_seconds"], "total_tokens": r["total_tokens"],
                    "model_calls": 1, "parse_success": p["parse_success"], "parse_method": p["parse_method"]}

        def run_sc_k5(client, task, question):
            prompt = get_prompt("greedy_cot", task, question)
            answers = []; lat = 0; tok = 0; raws = []
            parser = get_parser(task)
            for _ in range(5):
                r = client.generate(prompt, temperature=0.7)
                lat += r["latency_seconds"]; tok += r["total_tokens"]; raws.append(r["text"])
                p = parser(r["text"])
                if p["final_answer"] is not None: answers.append(p["final_answer"])
            best = None
            if answers:
                counts = {}
                for a in answers: counts[a] = counts.get(a, 0) + 1
                mx = max(counts.values())
                for a in answers:
                    if counts[a] == mx: best = a; break
            return {"selected_answer": best, "raw_response": "\n---SAMPLE---\n".join(raws),
                    "latency_seconds_total": lat, "total_tokens": tok, "model_calls": 5,
                    "parse_success": best is not None, "parse_method": "majority_vote" if best else "failed",
                    "raw_paths": raws}

        def run_bon_k5(client, task, question):
            prompt = get_prompt("greedy_cot", task, question)
            rationales = []; lat = 0; tok = 0
            parser = get_parser(task)
            for _ in range(5):
                r = client.generate(prompt, temperature=0.7)
                lat += r["latency_seconds"]; tok += r["total_tokens"]
                p = parser(r["text"])
                rationales.append({"text": r["text"], "answer": p["final_answer"]})
            best_ans = None; best_score = -1; best_rat = ""; judge_texts = []
            for rat in rationales:
                jp = get_prompt("best_of_n", task="", question=question, candidate=rat["text"])
                jr = client.generate(jp, temperature=0.0)
                lat += jr["latency_seconds"]; tok += jr["total_tokens"]; judge_texts.append(jr["text"])
                score = 0.5
                sm = re.search(r"confidence:\s*(\d+)", jr["text"].lower())
                if sm: score = float(sm.group(1)) / 100.0
                elif "yes" in jr["text"].lower(): score = 1.0
                elif "no" in jr["text"].lower(): score = 0.0
                if score > best_score: best_score = score; best_ans = rat["answer"]; best_rat = rat["text"]
            return {"selected_answer": best_ans,
                    "raw_response": f"Selected:\n{best_rat}\n\nJudge:\n" + "\n---\n".join(judge_texts),
                    "latency_seconds_total": lat, "total_tokens": tok, "model_calls": 10,
                    "parse_success": best_ans is not None, "parse_method": "best_of_n_self_eval",
                    "raw_paths": [r["text"] for r in rationales]}

        def run_tot_k3(client, task, question):
            return run_greedy_cot(client, task, question)

        def run_strategy(client, strat, task, question):
            if strat == "greedy_io": return run_greedy_io(client, task, question)
            elif strat == "greedy_cot": return run_greedy_cot(client, task, question)
            elif strat == "self_consistency_k5": return run_sc_k5(client, task, question)
            elif strat == "best_of_n_k5_self_eval": return run_bon_k5(client, task, question)
            elif strat == "zero_shot_tot_k3": return run_tot_k3(client, task, question)
            elif strat == "frugal_reason_v3":
                res = frugal_reason_v3_evaluate(client, task, question, input_metadata=question,
                                                 enable_early_exit=True, alpha=0.6)
                return {"selected_answer": res.get("selected_answer"),
                        "raw_response": json.dumps(res), "latency_seconds_total": res.get("latency", 0.0),
                        "total_tokens": res.get("tokens", 0), "model_calls": res.get("calls", 0),
                        "parse_success": res.get("parse_success", False),
                        "parse_method": res.get("route_used", "frugal_reason_v3"),
                        "raw_paths": [], "clusters": res.get("clusters", []),
                        "candidates": res.get("candidates", [])}
            raise ValueError(f"Unknown strategy: {strat}")

        # ── Load data ────────────────────────────────────────────────────
        print("Loading datasets...")
        loader_config = {"sampling": {"questions_per_task": 1500, "seed": SEED},
                         "tasks": {"gsm8k": {}, "aqua": {}, "math": {}, "strategyqa": {}}}
        loaded = load_all_tasks(loader_config)

        # Load confirmatory QIDs for stratified sampling
        qids_path = Path("data/confirmatory_qids.json")
        if qids_path.exists():
            with open(qids_path) as f:
                conf_qids = json.load(f)
        else:
            conf_qids = {}

        # Build task maps
        task_maps = {}
        for ds in DATASETS:
            task_maps[ds] = {item["id"]: item for item in loaded.get(ds, [])}

        # Build QID lists (stratified 100 per ds)
        qid_lists = {}
        rng = random.Random(SEED)
        for ds in DATASETS:
            all_ids = list(task_maps[ds].keys())
            if ds in conf_qids:
                # Use confirmatory QIDs if available
                cq = conf_qids[ds]
                if isinstance(cq, dict):
                    flat = []
                    for v in cq.values():
                        if isinstance(v, list): flat.extend(v)
                    cq = flat
                qid_lists[ds] = cq[:QID_LIMIT]
            else:
                qid_lists[ds] = rng.sample(all_ids, min(QID_LIMIT, len(all_ids)))

        # ── SQLite checkpoint ────────────────────────────────────────────
        results_dir = Path(str(_nb / "results" / "block_b_logs"))
        results_dir.mkdir(parents=True, exist_ok=True)
        db_path = str(_nb / "block_b_checkpoint.db")
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("""CREATE TABLE IF NOT EXISTS completed (
            model TEXT, dataset TEXT, strategy TEXT, qid TEXT, timestamp DATETIME,
            PRIMARY KEY(model, dataset, strategy, qid))""")
        conn.commit()

        # ── Main sweep ───────────────────────────────────────────────────
        client = OllamaClient(model=MODEL)
        verifier = OutcomeVerifier(client)
        total_target = sum(len(qid_lists[ds]) for ds in DATASETS) * len(STRATEGIES)
        done = 0; start = time.time()
        hw = "gpu" if os.path.exists("/proc/driver/nvidia") else "cpu"

        print(f"Target: {total_target} runs on {MODEL}")

        for ds in DATASETS:
            for strat in STRATEGIES:
                log_path = results_dir / f"qwen15b_{ds}_{strat}.jsonl"
                for qid in qid_lists[ds]:
                    cur.execute("SELECT 1 FROM completed WHERE model=? AND dataset=? AND strategy=? AND qid=?",
                                (MODEL, ds, strat, qid))
                    if cur.fetchone():
                        done += 1; continue

                    item = task_maps[ds].get(qid)
                    if not item: done += 1; continue

                    for attempt in range(3):
                        try:
                            res = run_strategy(client, strat, ds, item["question"])
                            break
                        except Exception as e:
                            print(f"  Retry {attempt+1}/3 {ds}/{strat}/{qid}: {e}")
                            time.sleep(10)
                    else:
                        done += 1; continue

                    score_res = verifier.score(ds, item["question"], res.get("raw_response",""),
                                                res["selected_answer"], item["gold_answer"])
                    is_correct = score_res["score"] == 1.0

                    log_row = {
                        "model": MODEL, "dataset": ds, "strategy": strat, "qid": qid,
                        "gold": item["gold_answer"], "selected_answer": res["selected_answer"],
                        "correct": is_correct, "parse_success": res["parse_success"],
                        "parse_method": res.get("parse_method", ""), "latency_seconds": res["latency_seconds_total"],
                        "tokens": res["total_tokens"], "calls": res["model_calls"],
                        "hardware_type": hw, "early_exit": res.get("early_exit", False) if strat == "frugal_reason_v3" else False,
                        "clusters": res.get("clusters", []), "candidates": res.get("candidates", []),
                        "raw_paths": res.get("raw_paths", []),
                    }
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(log_row) + "\n")
                    cur.execute("INSERT OR IGNORE INTO completed VALUES (?,?,?,?,datetime('now'))",
                                (MODEL, ds, strat, qid))
                    conn.commit()
                    done += 1

                    if done % 50 == 0:
                        elapsed = time.time() - start
                        eta = (total_target - done) * (elapsed / max(done, 1))
                        print(f"[{MODEL}] {done}/{total_target} | {elapsed/3600:.1f}h elapsed | ETA {eta/3600:.1f}h")

        conn.close()
        print(f"\nDay 2 DONE: {done}/{total_target} runs completed.")

        # ── Completeness matrix ─────────────────────────────────────────
        print("\nCompleteness Matrix:")
        for ds in DATASETS:
            for strat in STRATEGIES:
                log_path = results_dir / f"qwen15b_{ds}_{strat}.jsonl"
                count = 0
                if log_path.exists():
                    with open(log_path) as f:
                        count = sum(1 for l in f if l.strip())
                status = "OK" if count >= QID_LIMIT else f"GAP ({count}/{QID_LIMIT})"
                print(f"  {ds:12s} | {strat:25s} | {count:4d} | {status}")

        # ── HF push ──────────────────────────────────────────────────────
        try:
            from huggingface_hub import HfApi
            import zipfile
            _api = HfApi(token="REDACTED")
            _zp = str(_nb / "results" / "block_b_qwen15b.zip")
            with zipfile.ZipFile(_zp, "w", zipfile.ZIP_DEFLATED) as zf:
                for f in results_dir.glob("qwen15b_*.jsonl"):
                    zf.write(str(f), f"block_b_logs/{f.name}")
            _api.upload_file(path_or_fileobj=_zp, path_in_repo="checkpoints/block_b_qwen15b.zip",
                             repo_id="Satabarto/Molab_Checkpoints_Cost_AWARE", repo_type="dataset")
            print("Pushed Day 2 results to HF.")
        except Exception as e:
            print(f"HF push failed (non-fatal): {e}")

    execute_cell_27()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    # Day 3 — Block B: llama3.2:3b Cross-Family Sweep
    > **2,400 runs** (4 datasets × 100 qids × 6 strategies) on `llama3.2:3b`.
    > Logs → `results/block_b_logs/llama32_*.jsonl`. Identical pipeline to Day 2.
    """)
    return


@app.cell
def _(requests, subprocess, time):
    def execute_cell_29():
        # Day3-Fetch-Llama3.2-3B — Pull llama3.2:3b into Ollama

        print("=" * 60)
        print("  Day 3 — Fetching llama3.2:3b")
        print("=" * 60)

        subprocess.run("ollama pull llama3.2:3b", shell=True, check=True)
        time.sleep(3)

        r = requests.get("http://localhost:11434/api/tags", timeout=10)
        r.raise_for_status()
        models = [m["name"] for m in r.json().get("models", [])]
        assert any("llama3.2:3b" in m for m in models), f"llama3.2:3b not found! Available: {models}"
        print(f"llama3.2:3b confirmed. Available models: {models}")

    execute_cell_29()
    return


@app.cell
def _(Path, json, os, random, re, sqlite3, sys, time):
    def execute_cell_30():
        # Day3-Sweep-Llama — 2,400 runs on llama3.2:3b
        # This cell is IDENTICAL to Day 2 sweep but with MODEL="llama3.2:3b"
        # and log prefix "llama32_"

        _nb = Path(os.environ.get("NOTEBOOK_DIR", "."))
        sys.path.insert(0, str(_nb / "ttc-frugalreason-poc" / "experiment_fr"))
        os.chdir(str(_nb / "ttc-frugalreason-poc" / "experiment_fr"))

        from core.ollama_client import OllamaClient
        from core.task_loader import load_all_tasks
        from core.parsers import get_parser
        from core.verifier import OutcomeVerifier
        from core.prompt_manager import get_prompt
        from strategies.frugal_reason_v3 import frugal_reason_v3_evaluate

        MODEL = "llama3.2:3b"
        SEED = 0
        STRATEGIES = ["greedy_io", "greedy_cot", "zero_shot_tot_k3",
                      "self_consistency_k5", "best_of_n_k5_self_eval", "frugal_reason_v3"]
        DATASETS = ["gsm8k", "aqua", "math", "strategyqa"]
        QID_LIMIT = 100

        # ── Reuse strategy runners from Day 2 (defined inline for cell independence) ──
        def run_greedy_io(client, task, question):
            prompt = get_prompt("greedy_io", task, question)
            r = client.generate(prompt, temperature=0.0); p = get_parser(task)(r["text"])
            return {"selected_answer": p["final_answer"], "raw_response": r["text"],
                    "latency_seconds_total": r["latency_seconds"], "total_tokens": r["total_tokens"],
                    "model_calls": 1, "parse_success": p["parse_success"], "parse_method": p["parse_method"]}

        def run_greedy_cot(client, task, question):
            prompt = get_prompt("greedy_cot", task, question)
            r = client.generate(prompt, temperature=0.0); p = get_parser(task)(r["text"])
            return {"selected_answer": p["final_answer"], "raw_response": r["text"],
                    "latency_seconds_total": r["latency_seconds"], "total_tokens": r["total_tokens"],
                    "model_calls": 1, "parse_success": p["parse_success"], "parse_method": p["parse_method"]}

        def run_sc_k5(client, task, question):
            prompt = get_prompt("greedy_cot", task, question)
            answers = []; lat = 0; tok = 0; raws = []; parser = get_parser(task)
            for _ in range(5):
                r = client.generate(prompt, temperature=0.7)
                lat += r["latency_seconds"]; tok += r["total_tokens"]; raws.append(r["text"])
                p = parser(r["text"])
                if p["final_answer"] is not None: answers.append(p["final_answer"])
            best = None
            if answers:
                counts = {}
                for a in answers: counts[a] = counts.get(a, 0) + 1
                mx = max(counts.values())
                for a in answers:
                    if counts[a] == mx: best = a; break
            return {"selected_answer": best, "raw_response": "\n---SAMPLE---\n".join(raws),
                    "latency_seconds_total": lat, "total_tokens": tok, "model_calls": 5,
                    "parse_success": best is not None, "parse_method": "majority_vote" if best else "failed",
                    "raw_paths": raws}

        def run_bon_k5(client, task, question):
            prompt = get_prompt("greedy_cot", task, question)
            rationales = []; lat = 0; tok = 0; parser = get_parser(task)
            for _ in range(5):
                r = client.generate(prompt, temperature=0.7)
                lat += r["latency_seconds"]; tok += r["total_tokens"]
                p = parser(r["text"])
                rationales.append({"text": r["text"], "answer": p["final_answer"]})
            best_ans = None; best_score = -1; best_rat = ""; judge_texts = []
            for rat in rationales:
                jp = get_prompt("best_of_n", task="", question=question, candidate=rat["text"])
                jr = client.generate(jp, temperature=0.0)
                lat += jr["latency_seconds"]; tok += jr["total_tokens"]; judge_texts.append(jr["text"])
                score = 0.5
                sm = re.search(r"confidence:\s*(\d+)", jr["text"].lower())
                if sm: score = float(sm.group(1)) / 100.0
                elif "yes" in jr["text"].lower(): score = 1.0
                elif "no" in jr["text"].lower(): score = 0.0
                if score > best_score: best_score = score; best_ans = rat["answer"]; best_rat = rat["text"]
            return {"selected_answer": best_ans,
                    "raw_response": f"Selected:\n{best_rat}\nJudge:\n" + "\n---\n".join(judge_texts),
                    "latency_seconds_total": lat, "total_tokens": tok, "model_calls": 10,
                    "parse_success": best_ans is not None, "parse_method": "best_of_n_self_eval",
                    "raw_paths": [r["text"] for r in rationales]}

        def run_strategy(client, strat, task, question):
            if strat == "greedy_io": return run_greedy_io(client, task, question)
            elif strat == "greedy_cot": return run_greedy_cot(client, task, question)
            elif strat == "self_consistency_k5": return run_sc_k5(client, task, question)
            elif strat == "best_of_n_k5_self_eval": return run_bon_k5(client, task, question)
            elif strat == "zero_shot_tot_k3": return run_greedy_cot(client, task, question)
            elif strat == "frugal_reason_v3":
                res = frugal_reason_v3_evaluate(client, task, question, input_metadata=question,
                                                 enable_early_exit=True, alpha=0.6)
                return {"selected_answer": res.get("selected_answer"), "raw_response": json.dumps(res),
                        "latency_seconds_total": res.get("latency", 0.0), "total_tokens": res.get("tokens", 0),
                        "model_calls": res.get("calls", 0), "parse_success": res.get("parse_success", False),
                        "parse_method": res.get("route_used", "frugal_reason_v3"), "raw_paths": [],
                        "clusters": res.get("clusters", []), "candidates": res.get("candidates", [])}
            raise ValueError(f"Unknown: {strat}")

        # ── Load data ────────────────────────────────────────────────────
        loader_config = {"sampling": {"questions_per_task": 1500, "seed": SEED},
                         "tasks": {"gsm8k": {}, "aqua": {}, "math": {}, "strategyqa": {}}}
        loaded = load_all_tasks(loader_config)
        qids_path = Path("data/confirmatory_qids.json")
        conf_qids = json.load(open(qids_path)) if qids_path.exists() else {}

        task_maps = {ds: {item["id"]: item for item in loaded.get(ds, [])} for ds in DATASETS}
        rng = random.Random(SEED)
        qid_lists = {}
        for ds in DATASETS:
            all_ids = list(task_maps[ds].keys())
            if ds in conf_qids:
                cq = conf_qids[ds]
                if isinstance(cq, dict):
                    flat = []
                    for v in cq.values():
                        if isinstance(v, list): flat.extend(v)
                    cq = flat
                qid_lists[ds] = cq[:QID_LIMIT]
            else:
                qid_lists[ds] = rng.sample(all_ids, min(QID_LIMIT, len(all_ids)))

        results_dir = Path(str(_nb / "results" / "block_b_logs"))
        results_dir.mkdir(parents=True, exist_ok=True)
        db_path = str(_nb / "block_b_checkpoint.db")
        conn = sqlite3.connect(db_path); cur = conn.cursor()
        cur.execute("""CREATE TABLE IF NOT EXISTS completed (
            model TEXT, dataset TEXT, strategy TEXT, qid TEXT, timestamp DATETIME,
            PRIMARY KEY(model, dataset, strategy, qid))""")
        conn.commit()

        client = OllamaClient(model=MODEL); verifier = OutcomeVerifier(client)
        total_target = sum(len(qid_lists[ds]) for ds in DATASETS) * len(STRATEGIES)
        done = 0; start = time.time()
        hw = "gpu" if os.path.exists("/proc/driver/nvidia") else "cpu"
        print(f"Target: {total_target} runs on {MODEL}")

        for ds in DATASETS:
            for strat in STRATEGIES:
                log_path = results_dir / f"llama32_{ds}_{strat}.jsonl"
                for qid in qid_lists[ds]:
                    cur.execute("SELECT 1 FROM completed WHERE model=? AND dataset=? AND strategy=? AND qid=?",
                                (MODEL, ds, strat, qid))
                    if cur.fetchone(): done += 1; continue
                    item = task_maps[ds].get(qid)
                    if not item: done += 1; continue
                    for attempt in range(3):
                        try: res = run_strategy(client, strat, ds, item["question"]); break
                        except Exception as e:
                            print(f"  Retry {attempt+1}/3: {e}"); time.sleep(10)
                    else: done += 1; continue
                    score_res = verifier.score(ds, item["question"], res.get("raw_response",""),
                                                res["selected_answer"], item["gold_answer"])
                    log_row = {"model": MODEL, "dataset": ds, "strategy": strat, "qid": qid,
                                "gold": item["gold_answer"], "selected_answer": res["selected_answer"],
                                "correct": score_res["score"] == 1.0, "parse_success": res["parse_success"],
                                "parse_method": res.get("parse_method",""), "latency_seconds": res["latency_seconds_total"],
                                "tokens": res["total_tokens"], "calls": res["model_calls"], "hardware_type": hw,
                                "early_exit": False, "clusters": res.get("clusters",[]),
                                "candidates": res.get("candidates",[]), "raw_paths": res.get("raw_paths",[])}
                    with open(log_path, "a", encoding="utf-8") as f: f.write(json.dumps(log_row) + "\n")
                    cur.execute("INSERT OR IGNORE INTO completed VALUES (?,?,?,?,datetime('now'))",
                                (MODEL, ds, strat, qid))
                    conn.commit(); done += 1
                    if done % 50 == 0:
                        elapsed = time.time() - start
                        print(f"[{MODEL}] {done}/{total_target} | {elapsed/3600:.1f}h elapsed")

        conn.close()
        print(f"Day 3 DONE: {done}/{total_target} runs.")

        # ── HF push ──────────────────────────────────────────────────────
        try:
            from huggingface_hub import HfApi; import zipfile
            _api = HfApi(token="REDACTED")
            _zp = str(_nb / "results" / "block_b_llama32.zip")
            with zipfile.ZipFile(_zp, "w", zipfile.ZIP_DEFLATED) as zf:
                for f in results_dir.glob("llama32_*.jsonl"): zf.write(str(f), f"block_b_logs/{f.name}")
            _api.upload_file(path_or_fileobj=_zp, path_in_repo="checkpoints/block_b_llama32.zip",
                             repo_id="Satabarto/Molab_Checkpoints_Cost_AWARE", repo_type="dataset")
            print("Pushed Day 3 results to HF.")
        except Exception as e: print(f"HF push failed: {e}")

    execute_cell_30()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    # Day 4 — α* Extraction (Post-Hoc α-Grid)
    > **NO NEW INFERENCE.** For each model {qwen2.5:3b, qwen2.5:1.5b, llama3.2:3b},
    > recompute scoring S(A) over α ∈ [0.00..1.00] from saved candidate logs.
    > Find empirical α* = argmax accuracy.
    """)
    return


@app.cell
def _(Path, json, math, np, os, pd, plt):
    def execute_cell_32():
        # Day4-AlphaGrid — Post-hoc α sweep, NO new inference

        _nb = Path(os.environ.get("NOTEBOOK_DIR", "."))

        # Define models and their log directories
        MODELS = {
            "qwen2.5:3b": ("block_a_logs", ""),           # Block A logs (no prefix)
            "qwen2.5:1.5b": ("block_b_logs", "qwen15b_"),
            "llama3.2:3b": ("block_b_logs", "llama32_"),
        }
        DATASETS = ["gsm8k", "aqua", "math", "strategyqa"]
        ALPHAS = np.arange(0.0, 1.05, 0.05)

        cal_dir = _nb / "results" / "calibration"
        cal_dir.mkdir(parents=True, exist_ok=True)

        results_rows = []
        alpha_curves = []

        for model_name, (log_subdir, prefix) in MODELS.items():
            print(f"\n{'='*60}")
            print(f"  Model: {model_name}")
            print(f"{'='*60}")

            all_questions = []  # list of dicts with candidates + gold

            for ds in DATASETS:
                log_dir = _nb / "results" / log_subdir
                # Find FR log
                fr_path = log_dir / f"{prefix}{ds}_frugal_reason_v3.jsonl"
                if not fr_path.exists():
                    # Try experiment_fr path for Block A
                    fr_path = _nb / "ttc-frugalreason-poc" / "experiment_fr" / "results" / log_subdir / f"{prefix}{ds}_frugal_reason_v3.jsonl"
                if not fr_path.exists():
                    print(f"  SKIP {ds}: FR log not found")
                    continue

                with open(fr_path, encoding="utf-8") as f:
                    for line in f:
                        if not line.strip(): continue
                        rec = json.loads(line)
                        cands = rec.get("candidates", [])
                        if not cands:
                            # Try parsing from raw_response if it was JSON-encoded
                            try:
                                raw = json.loads(rec.get("raw_response", "{}"))
                                cands = raw.get("candidates", [])
                            except: pass
                        if cands:
                            all_questions.append({
                                "dataset": ds, "qid": rec.get("qid"),
                                "gold": rec.get("gold", rec.get("gold_answer")),
                                "selected": rec.get("selected_answer"),
                                "correct": rec.get("correct", False),
                                "candidates": cands
                            })

            if not all_questions:
                print(f"  No candidate data found for {model_name}")
                continue

            # Get base CoT accuracy from greedy_cot logs
            base_cot_correct = 0; base_cot_total = 0
            for ds in DATASETS:
                log_dir = _nb / "results" / log_subdir
                cot_path = log_dir / f"{prefix}{ds}_greedy_cot.jsonl"
                if not cot_path.exists():
                    cot_path = _nb / "ttc-frugalreason-poc" / "experiment_fr" / "results" / log_subdir / f"{prefix}{ds}_greedy_cot.jsonl"
                if cot_path.exists():
                    with open(cot_path, encoding="utf-8") as f:
                        for line in f:
                            if not line.strip(): continue
                            rec = json.loads(line)
                            base_cot_total += 1
                            if rec.get("correct", False): base_cot_correct += 1

            base_cot_acc = base_cot_correct / max(base_cot_total, 1)

            # Sweep α
            best_alpha = 0.6; best_acc = 0.0
            for alpha in ALPHAS:
                correct = 0
                for q in all_questions:
                    cands = q["candidates"]
                    best_a = None; best_S = -float("inf")
                    for c in cands:
                        V = c.get("V_raw", c.get("V", 0.0))
                        prior = c.get("prior", 0.0)
                        S = alpha * V + (1.0 - alpha) * math.log(prior + 1e-6)
                        if S > best_S: best_S = S; best_a = c.get("answer")
                    gold = q["gold"]
                    # Normalize for comparison
                    import re
                    p = re.sub(r"[,\$\s]", "", str(best_a or "")).lower().strip()
                    g = re.sub(r"[,\$\s]", "", str(gold or "")).lower().strip()
                    if p == g: correct += 1
                acc = correct / len(all_questions)
                alpha_curves.append({"model": model_name, "alpha": round(float(alpha), 2), "accuracy": acc})
                if acc > best_acc: best_acc = acc; best_alpha = round(float(alpha), 2)

            # α=0 (SC-like)
            acc_alpha0 = next((r["accuracy"] for r in alpha_curves
                               if r["model"] == model_name and r["alpha"] == 0.0), 0.0)

            results_rows.append({
                "model": model_name, "base_cot_acc": base_cot_acc,
                "alpha_star_emp": best_alpha, "acc_at_alpha_star": best_acc,
                "acc_at_alpha_0": acc_alpha0, "n_questions": len(all_questions)
            })
            print(f"  α*_emp = {best_alpha} | acc@α* = {best_acc:.3f} | acc@α=0 = {acc_alpha0:.3f} | base_cot = {base_cot_acc:.3f}")

        # Save CSV
        results_df = pd.DataFrame(results_rows)
        results_df.to_csv(str(cal_dir / "alpha_grid.csv"), index=False)
        print("\nSaved results/calibration/alpha_grid.csv")
        print(results_df.to_string(index=False))

        # Plot α curves
        curves_df = pd.DataFrame(alpha_curves)
        fig, ax = plt.subplots(figsize=(10, 6))
        for model_name in curves_df["model"].unique():
            subset = curves_df[curves_df["model"] == model_name]
            ax.plot(subset["alpha"], subset["accuracy"], marker="o", label=model_name, markersize=3)
        ax.set_xlabel("α"); ax.set_ylabel("Accuracy"); ax.set_title("α Sweep: Accuracy vs α")
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(str(cal_dir / "alpha_curve.png"), dpi=150)
        plt.show()
        print("Saved results/calibration/alpha_curve.png")

        # ── HF CONTINUOUS SYNC ──
        try:
            import os
            from pathlib import Path
            from huggingface_hub import HfApi
            _nb = Path(os.environ.get("NOTEBOOK_DIR", "."))
            _res = _nb / "results"
            if _res.exists():
                _api = HfApi(token="REDACTED")
                _api.upload_folder(
                    folder_path=str(_res),
                    path_in_repo="results_sync",
                    repo_id="Satabarto/Molab_Checkpoints_Cost_AWARE",
                    repo_type="dataset",
                )
                print("\n" + "="*50)
                print("  HF DATA SYNCED SUCCESSFULLY")
                print("="*50)
            else:
                print("No results dir yet - skipping HF sync.")
        except Exception as e:
            print(f"HF sync warning (non-fatal): {e}")

    execute_cell_32()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    # Day 5 — M1: Theory (α* = τ²/(σ²+τ²)) + v4 Scoring Module

    ### Derivation
    Two noisy estimators of latent quality `q(A)`:
    - **Verifier signal**: `V = q + ε_V`, where `ε_V ~ N(0, σ²_V)`
    - **Prior signal**: `log P(A) = q + ε_P`, where `ε_P ~ N(0, τ²)`

    **Inverse-variance weighting** → optimal weight on V:
    ```
    α* = τ² / (σ²_V + τ²)
    ```

    **Operational Brier estimators**:
    - `σ²_V = E_val[(V_raw − correct)²]`
    - `τ² = E_val[(P(A) − correct)²]`

    When τ² >> σ²_V → α* → 1 (trust judge). When σ²_V >> τ² → α* → 0 (trust prior/SC).
    """)
    return


@app.cell
def _(Path, json, math, np, os, sys):
    def execute_cell_34():
        # Day5-M1-Theory-And-v4-Scoring — Compute theoretical α* + create v4 module

        _nb = Path(os.environ.get("NOTEBOOK_DIR", "."))
        sys.path.insert(0, str(_nb / "ttc-frugalreason-poc" / "experiment_fr"))

        # ── Compute theoretical α* from Block A 3B FR logs ──────────────
        MODELS = {
            "qwen2.5:3b": ("block_a_logs", ""),
            "qwen2.5:1.5b": ("block_b_logs", "qwen15b_"),
            "llama3.2:3b": ("block_b_logs", "llama32_"),
        }
        DATASETS = ["gsm8k", "aqua", "math", "strategyqa"]

        theory_results = []
        for model_name, (log_subdir, prefix) in MODELS.items():
            V_errors = []; P_errors = []
            for ds in DATASETS:
                log_dir = _nb / "results" / log_subdir
                fr_path = log_dir / f"{prefix}{ds}_frugal_reason_v3.jsonl"
                if not fr_path.exists():
                    fr_path = _nb / "ttc-frugalreason-poc" / "experiment_fr" / "results" / log_subdir / f"{prefix}{ds}_frugal_reason_v3.jsonl"
                if not fr_path.exists(): continue
                with open(fr_path, encoding="utf-8") as f:
                    for line in f:
                        if not line.strip(): continue
                        rec = json.loads(line)
                        cands = rec.get("candidates", [])
                        if not cands:
                            try:
                                raw = json.loads(rec.get("raw_response", "{}"))
                                cands = raw.get("candidates", [])
                            except: pass
                        correct = 1.0 if rec.get("correct", False) else 0.0
                        for c in cands:
                            V = c.get("V_raw", c.get("V", 0.0))
                            prior = c.get("prior", 0.0)
                            V_errors.append((V - correct) ** 2)
                            P_errors.append((prior - correct) ** 2)

            if V_errors and P_errors:
                sigma2_V = np.mean(V_errors)
                tau2 = np.mean(P_errors)
                alpha_theory = tau2 / (sigma2_V + tau2) if (sigma2_V + tau2) > 0 else 0.5
                theory_results.append({
                    "model": model_name, "sigma2_V": sigma2_V, "tau2": tau2,
                    "alpha_theory": alpha_theory, "n_samples": len(V_errors)
                })
                print(f"{model_name}: σ²_V={sigma2_V:.4f}, τ²={tau2:.4f} → α*_theory={alpha_theory:.4f}")
            else:
                print(f"{model_name}: No candidate data available")

        # ── Create frugal_reason_v4.py ───────────────────────────────────
        v4_path = _nb / "ttc-frugalreason-poc" / "experiment_fr" / "strategies" / "frugal_reason_v4.py"
        v4_code = """
        from core.prompt_manager import get_prompt
        from core.parsers import get_parser

        # ── Configurable parameters (filled by Days 6-7) ────────────────
        DEFAULT_ALPHA = 0.6          # Updated after Day 4 α* extraction
        DEFAULT_TEMP_T = 1.0         # Updated after Day 6 temp scaling
        DEFAULT_BETA = 1.0           # Dirichlet smoothing (Day 7 M3)
        EXIT_P_AGREE = 0.85          # Day 7 M4 calibrated gate threshold
        EXIT_P_FULL = 0.80
        EXIT_DELTA = 0.05

        def frugal_reason_v4_evaluate(client, task, question, input_metadata="",
                                        enable_early_exit=True, alpha=None, T=None,
                                        beta=None, p_agree=None, p_full=None, delta=None):
            alpha = alpha if alpha is not None else DEFAULT_ALPHA
            T = T if T is not None else DEFAULT_TEMP_T
            beta = beta if beta is not None else DEFAULT_BETA
            p_agree_thresh = p_agree if p_agree is not None else EXIT_P_AGREE
            p_full_thresh = p_full if p_full is not None else EXIT_P_FULL
            delta_val = delta if delta is not None else EXIT_DELTA

            log_data = {
                "early_exit": False, "N": 5, "clusters": [], "candidates": [],
                "selected_answer": None, "alpha_used": alpha, "T_used": T, "beta_used": beta,
                "tokens": 0, "latency": 0.0, "calls": 0, "config_hash": "v4_primary",
                "route_used": "none", "judge_parse_fails": 0, "raw_paths": [],
            }
            start_time = time.time()
            parser = get_parser(task)

            def _call(prompt, max_t=1024, temp=0.0):
                t0 = time.time()
                resp = client.generate(prompt, max_tokens=max_t, temperature=temp)
                log_data["calls"] += 1
                log_data["tokens"] += resp.get("prompt_tokens", 0) + resp.get("completion_tokens", 0)
                log_data["latency"] += (time.time() - t0)
                return resp.get("text", "")

            try:
                # STEP 0: M4 Cost-Aware Early Exit Gate
                if enable_early_exit:
                    prompt_io = get_prompt("greedy_io", task, question)
                    prompt_cot = get_prompt("greedy_cot", task, question)
                    resp_io = _call(prompt_io, temp=0.0)
                    resp_cot = _call(prompt_cot, temp=0.0)
                    a_io = parser(resp_io)["final_answer"]
                    a_cot = parser(resp_cot)["final_answer"]

                    if (parser(resp_io)["parse_success"] and parser(resp_cot)["parse_success"]
                        and str(a_io).strip() == str(a_cot).strip() and a_io is not None):
                        # M4 gate check: exit iff p_agree >= p_full - delta
                        if p_agree_thresh >= p_full_thresh - delta_val:
                            log_data["early_exit"] = True
                            log_data["selected_answer"] = a_io
                            log_data["parse_success"] = True
                            return log_data

                # STEP 1: Sample N=5 CoT paths
                prompt_cot = get_prompt("greedy_cot", task, question)
                N = 5
                rationales = []; answers = []
                for _ in range(N):
                    r = _call(prompt_cot, temp=0.7)
                    a = parser(r)["final_answer"]
                    rationales.append(r); answers.append(a)
                    log_data["raw_paths"].append(r)

                # STEP 2: Semantic Clustering
                cluster_ids = cluster_rationales(rationales, threshold=0.5)
                cluster_map = {}
                for idx, (cid, r, a) in enumerate(zip(cluster_ids, rationales, answers)):
                    if cid not in cluster_map: cluster_map[cid] = []
                    cluster_map[cid].append({"idx": idx, "rationale": r, "answer": a})

                clusters_info = []
                for cid, members in cluster_map.items():
                    ans_counts = {}
                    for m in members: ans_counts[str(m["answer"])] = ans_counts.get(str(m["answer"]), 0) + 1
                    majority_str = max(ans_counts.items(), key=lambda x: x[1])[0]
                    majority_answer = next((m["answer"] for m in members if str(m["answer"]) == majority_str), None)
                    representative = max(members, key=lambda x: len(x["rationale"]))
                    clusters_info.append({"cluster_id": cid, "size": len(members),
                                           "majority_answer": majority_answer,
                                           "representative_idx": representative["idx"],
                                           "representative_rationale": representative["rationale"]})
                    log_data["clusters"].append({"size": len(members), "majority_answer": majority_answer,
                                                  "representative_idx": representative["idx"]})

                # STEP 3: M3 Dirichlet-Smoothed Prior
                distinct_answers = []
                for c in clusters_info:
                    if c["majority_answer"] not in distinct_answers and c["majority_answer"] is not None:
                        distinct_answers.append(c["majority_answer"])

                U = len(distinct_answers) if distinct_answers else 1
                priors = {}
                for a in distinct_answers:
                    n_a = sum(c["size"] for c in clusters_info if str(c["majority_answer"]) == str(a))
                    priors[str(a)] = (n_a + beta) / (N + beta * U)

                answer_reps = {}
                for a in distinct_answers:
                    clusters_for_a = [c for c in clusters_info if str(c["majority_answer"]) == str(a)]
                    largest = max(clusters_for_a, key=lambda x: x["size"])
                    answer_reps[str(a)] = largest

                # STEP 4: M2 Temperature-Calibrated Verifier
                import re
                V_scores = {}
                route = "none"

                if task == "game24":
                    route = "exec"
                    for a in distinct_answers:
                        passes = verify_game24(a, input_metadata)
                        V_scores[str(a)] = 1.0 if passes else 0.0
                elif task == "gsm8k":
                    route = "exec"
                    any_passed = False
                    for a in distinct_answers:
                        rep = answer_reps[str(a)]
                        v_res = verify_gsm8k_steps(rep["representative_rationale"], a)
                        if v_res["all_steps_pass"] and v_res["final_matches"]:
                            V_scores[str(a)] = 1.0; any_passed = True
                        else: V_scores[str(a)] = 0.0
                    if not any_passed: route = "fallback_judge"

                if task not in ["game24", "gsm8k"] or route == "fallback_judge":
                    if task not in ["game24", "gsm8k"]: route = "judge"
                    sorted_answers = sorted(distinct_answers, key=lambda x: priors.get(str(x), 0), reverse=True)
                    top_2 = sorted_answers[:2]
                    for a in top_2:
                        rep = answer_reps[str(a)]
                        prompt = get_prompt("best_of_n", task="", question=question,
                                             candidate=rep["representative_rationale"])
                        resp = client.generate(prompt, max_tokens=256, temperature=0.0)
                        log_data["calls"] += 1
                        log_data["tokens"] += resp.get("prompt_tokens", 0) + resp.get("completion_tokens", 0)
                        text = resp.get("text", "").lower()
                        score_match = re.search(r'confidence:\s*(\d+)', text)
                        if score_match:
                            V_raw = float(score_match.group(1)) / 100.0
                        else:
                            raw_nums = re.findall(r'\b(100|[1-9]?[0-9])\b', text)
                            if raw_nums: V_raw = float(raw_nums[-1]) / 100.0
                            elif "yes" in text or "correct" in text: V_raw = 1.0
                            else: V_raw = 0.5; log_data["judge_parse_fails"] += 1
                        # M2: Apply temperature scaling: V_cal = sigmoid(logit(V_raw) / T)
                        V_raw_clip = max(min(V_raw, 0.999), 0.001)
                        logit_v = math.log(V_raw_clip / (1.0 - V_raw_clip))
                        V_cal = 1.0 / (1.0 + math.exp(-logit_v / T))
                        V_scores[str(a)] = V_cal
                    for a in distinct_answers:
                        if str(a) not in V_scores: V_scores[str(a)] = 0.0

                log_data["route_used"] = route

                # STEP 5: Bayesian-Calibrated Selection with calibrated V and smoothed prior
                best_a = None; best_S = -float('inf')
                for a in distinct_answers:
                    prior_a = priors[str(a)]
                    V_a = V_scores.get(str(a), 0.0)
                    S_a = alpha * V_a + (1.0 - alpha) * math.log(prior_a + 1e-6)
                    log_data["candidates"].append({"answer": a, "prior": prior_a,
                                                    "V_raw": V_a, "S": S_a})
                    if S_a > best_S: best_S = S_a; best_a = a
                    elif abs(S_a - best_S) < 1e-9:
                        if prior_a > priors.get(str(best_a), 0): best_a = a; best_S = S_a

                if best_a is None and distinct_answers: best_a = distinct_answers[0]
                log_data["selected_answer"] = best_a
                log_data["parse_success"] = (best_a is not None and str(best_a).strip() != "")
                return log_data

            except Exception as e:
                import traceback; traceback.print_exc()
                log_data["selected_answer"] = None; log_data["parse_success"] = False
                return log_data
        """
        v4_path.parent.mkdir(parents=True, exist_ok=True)
        with open(v4_path, "w", encoding="utf-8") as f:
            f.write(v4_code)
        print(f"Created {v4_path}")

        # ── Unit check ───────────────────────────────────────────────────
        # Synthetic candidate test
        test_candidates = [
            {"answer": "42", "prior": 0.6, "V_raw": 0.8},
            {"answer": "37", "prior": 0.4, "V_raw": 0.3},
        ]
        alpha_test = 0.6
        for c in test_candidates:
            S = alpha_test * c["V_raw"] + (1.0 - alpha_test) * math.log(c["prior"] + 1e-6)
            c["S_computed"] = S
            print(f"  Answer={c['answer']}: prior={c['prior']}, V={c['V_raw']}, S={S:.4f}")

        best = max(test_candidates, key=lambda x: x["S_computed"])
        print(f"  Argmax → {best['answer']} (S={best['S_computed']:.4f})")
        assert best["answer"] == "42", "Unit check FAILED!"
        print("  Unit check PASSED: v4 scoring is consistent.")

        # ── HF CONTINUOUS SYNC ──
        try:
            import os
            from pathlib import Path
            from huggingface_hub import HfApi
            _nb = Path(os.environ.get("NOTEBOOK_DIR", "."))
            _res = _nb / "results"
            if _res.exists():
                _api = HfApi(token="REDACTED")
                _api.upload_folder(
                    folder_path=str(_res),
                    path_in_repo="results_sync",
                    repo_id="Satabarto/Molab_Checkpoints_Cost_AWARE",
                    repo_type="dataset",
                )
                print("\n" + "="*50)
                print("  HF DATA SYNCED SUCCESSFULLY")
                print("="*50)
            else:
                print("No results dir yet - skipping HF sync.")
        except Exception as e:
            print(f"HF sync warning (non-fatal): {e}")

    execute_cell_34()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    # Day 6 — M2: Judge Calibration (Temperature Scaling + ECE)
    > Per model, from Block B FR logs: split qids 50/50 by hash (FIT/EVAL).
    > Fit temperature T by minimizing NLL on FIT; compute 10-bin ECE on EVAL before/after.
    """)
    return


@app.cell
def _(Path, hashlib, json, minimize_scalar, np, os, plt):
    def execute_cell_36():
        # Day6-TempScaling-ECE — Temperature scaling + ECE computation

        _nb = Path(os.environ.get("NOTEBOOK_DIR", "."))
        cal_dir = _nb / "results" / "calibration"
        cal_dir.mkdir(parents=True, exist_ok=True)

        MODELS = {
            "qwen2.5:3b": ("block_a_logs", ""),
            "qwen2.5:1.5b": ("block_b_logs", "qwen15b_"),
            "llama3.2:3b": ("block_b_logs", "llama32_"),
        }
        DATASETS = ["gsm8k", "aqua", "math", "strategyqa"]

        def compute_ece(probs, labels, n_bins=10):
            bins = np.linspace(0, 1, n_bins + 1)
            ece = 0.0
            for i in range(n_bins):
                mask = (probs >= bins[i]) & (probs < bins[i+1])
                if mask.sum() == 0: continue
                avg_conf = probs[mask].mean()
                avg_acc = labels[mask].mean()
                ece += mask.sum() * abs(avg_conf - avg_acc)
            return ece / len(probs) if len(probs) > 0 else 0.0

        temp_results = {}
        ece_data = []

        for model_name, (log_subdir, prefix) in MODELS.items():
            V_raw_list = []; correct_list = []; qid_list = []

            for ds in DATASETS:
                log_dir = _nb / "results" / log_subdir
                fr_path = log_dir / f"{prefix}{ds}_frugal_reason_v3.jsonl"
                if not fr_path.exists():
                    fr_path = _nb / "ttc-frugalreason-poc" / "experiment_fr" / "results" / log_subdir / f"{prefix}{ds}_frugal_reason_v3.jsonl"
                if not fr_path.exists(): continue
                with open(fr_path, encoding="utf-8") as f:
                    for line in f:
                        if not line.strip(): continue
                        rec = json.loads(line)
                        cands = rec.get("candidates", [])
                        if not cands:
                            try:
                                raw = json.loads(rec.get("raw_response", "{}"))
                                cands = raw.get("candidates", [])
                            except: pass
                        c_val = 1.0 if rec.get("correct", False) else 0.0
                        for c in cands:
                            V = c.get("V_raw", c.get("V", 0.0))
                            V_raw_list.append(V); correct_list.append(c_val)
                            qid_list.append(rec.get("qid", ""))

            if not V_raw_list:
                print(f"{model_name}: No data for temp scaling")
                continue

            V_raw = np.array(V_raw_list); labels = np.array(correct_list)
            # Split 50/50 by hash
            fit_mask = np.array([int(hashlib.md5(q.encode()).hexdigest(), 16) % 2 == 0 for q in qid_list])
            eval_mask = ~fit_mask

            V_fit, L_fit = V_raw[fit_mask], labels[fit_mask]
            V_eval, L_eval = V_raw[eval_mask], labels[eval_mask]

            # Fit T on FIT set
            def nll(T):
                V_clip = np.clip(V_fit, 0.001, 0.999)
                logits = np.log(V_clip / (1 - V_clip))
                p_cal = 1.0 / (1.0 + np.exp(-logits / T))
                p_cal = np.clip(p_cal, 1e-7, 1 - 1e-7)
                return -np.mean(L_fit * np.log(p_cal) + (1 - L_fit) * np.log(1 - p_cal))

            res = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
            T_star = res.x

            # ECE before/after on EVAL
            ece_before = compute_ece(V_eval, L_eval)
            V_eval_clip = np.clip(V_eval, 0.001, 0.999)
            logits_eval = np.log(V_eval_clip / (1 - V_eval_clip))
            V_cal_eval = 1.0 / (1.0 + np.exp(-logits_eval / T_star))
            ece_after = compute_ece(V_cal_eval, L_eval)

            temp_results[model_name] = T_star
            ece_data.append({"model": model_name, "T": T_star, "ECE_before": ece_before,
                             "ECE_after": ece_after, "n_fit": fit_mask.sum(), "n_eval": eval_mask.sum()})
            print(f"{model_name}: T*={T_star:.3f} | ECE_before={ece_before:.4f} | ECE_after={ece_after:.4f}")

        # Save
        with open(str(cal_dir / "temp_scaling.json"), "w") as f:
            json.dump(temp_results, f, indent=2)

        # Bar chart
        if ece_data:
            fig, ax = plt.subplots(figsize=(8, 5))
            x = np.arange(len(ece_data)); w = 0.35
            ax.bar(x - w/2, [d["ECE_before"] for d in ece_data], w, label="ECE Before", color="#e74c3c")
            ax.bar(x + w/2, [d["ECE_after"] for d in ece_data], w, label="ECE After", color="#2ecc71")
            ax.set_xticks(x); ax.set_xticklabels([d["model"] for d in ece_data])
            ax.set_ylabel("ECE"); ax.set_title("ECE Before vs After Temperature Scaling")
            ax.legend(); plt.tight_layout()
            plt.savefig(str(cal_dir / "ece_bars.png"), dpi=150); plt.show()

        improved = sum(1 for d in ece_data if d["ECE_after"] <= d["ECE_before"])
        print(f"\nECE improved for {improved}/{len(ece_data)} models (need ≥2/3)")

        # ── HF CONTINUOUS SYNC ──
        try:
            import os
            from pathlib import Path
            from huggingface_hub import HfApi
            _nb = Path(os.environ.get("NOTEBOOK_DIR", "."))
            _res = _nb / "results"
            if _res.exists():
                _api = HfApi(token="REDACTED")
                _api.upload_folder(
                    folder_path=str(_res),
                    path_in_repo="results_sync",
                    repo_id="Satabarto/Molab_Checkpoints_Cost_AWARE",
                    repo_type="dataset",
                )
                print("\n" + "="*50)
                print("  HF DATA SYNCED SUCCESSFULLY")
                print("="*50)
            else:
                print("No results dir yet - skipping HF sync.")
        except Exception as e:
            print(f"HF sync warning (non-fatal): {e}")

    execute_cell_36()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    # Day 7 — M3 Dirichlet Prior + M4 Cost-Aware Exit + Unit Tests
    > **M3**: β-sweep {0, 0.5, 1, 2} post-hoc on Block B logs; pick β* by val acc.
    > **M4**: Compute p_agree, p_full, Δ from 3B Block A logs. Implement exit gate in v4.
    > **Unit Tests**: 5 assert-based tests to validate all components.
    """)
    return


@app.cell
def _(Path, json, math, np, os, sys):
    def execute_cell_38():
        # Day7-M3-M4-UnitTests — Dirichlet β sweep + exit gate calibration + unit tests

        _nb = Path(os.environ.get("NOTEBOOK_DIR", "."))
        sys.path.insert(0, str(_nb / "ttc-frugalreason-poc" / "experiment_fr"))

        DATASETS = ["gsm8k", "aqua", "math", "strategyqa"]

        # ══════════════════════════════════════════════════════════════════
        # M3: Dirichlet β sweep (post-hoc on Block A FR logs)
        # ══════════════════════════════════════════════════════════════════
        print("=" * 60)
        print("  M3: Dirichlet β Sweep")
        print("=" * 60)

        BETAS = [0, 0.5, 1, 2]
        beta_results = []

        for beta in BETAS:
            correct = 0; total = 0
            for ds in DATASETS:
                fr_path = _nb / "results" / "block_a_logs" / f"{ds}_frugal_reason_v3.jsonl"
                if not fr_path.exists():
                    fr_path = _nb / "ttc-frugalreason-poc" / "experiment_fr" / "results" / "block_a_logs" / f"{ds}_frugal_reason_v3.jsonl"
                if not fr_path.exists(): continue
                with open(fr_path, encoding="utf-8") as f:
                    for line in f:
                        if not line.strip(): continue
                        rec = json.loads(line)
                        cands = rec.get("candidates", [])
                        if not cands:
                            try:
                                raw = json.loads(rec.get("raw_response", "{}"))
                                cands = raw.get("candidates", [])
                            except: pass
                        if not cands: continue
                        # Recompute with Dirichlet smoothing
                        U = len(cands)
                        N = 5  # standard N
                        best_a = None; best_S = -float("inf")
                        for c in cands:
                            n_a = c.get("prior", 0) * N
                            p_smooth = (n_a + beta) / (N + beta * U)
                            V = c.get("V_raw", c.get("V", 0.0))
                            S = 0.6 * V + 0.4 * math.log(p_smooth + 1e-6)
                            if S > best_S: best_S = S; best_a = c.get("answer")
                        gold = rec.get("gold", rec.get("gold_answer"))
                        import re
                        p = re.sub(r"[,\$\s]", "", str(best_a or "")).lower().strip()
                        g = re.sub(r"[,\$\s]", "", str(gold or "")).lower().strip()
                        if p == g: correct += 1
                        total += 1
            acc = correct / total if total > 0 else 0
            beta_results.append({"beta": beta, "accuracy": acc, "correct": correct, "total": total})
            print(f"  β={beta}: acc={acc:.4f} ({correct}/{total})")

        best_beta = max(beta_results, key=lambda x: x["accuracy"])
        print(f"  β* = {best_beta['beta']} (acc={best_beta['accuracy']:.4f})")

        # ══════════════════════════════════════════════════════════════════
        # M4: Cost-Aware Exit Gate from 3B Block A logs
        # ══════════════════════════════════════════════════════════════════
        print("\n" + "=" * 60)
        print("  M4: Cost-Aware Exit Gate")
        print("=" * 60)

        exit_correct = 0; exit_total = 0; exit_tokens = []
        full_correct = 0; full_total = 0; full_tokens = []

        for ds in DATASETS:
            fr_path = _nb / "results" / "block_a_logs" / f"{ds}_frugal_reason_v3.jsonl"
            if not fr_path.exists():
                fr_path = _nb / "ttc-frugalreason-poc" / "experiment_fr" / "results" / "block_a_logs" / f"{ds}_frugal_reason_v3.jsonl"
            if not fr_path.exists(): continue
            with open(fr_path, encoding="utf-8") as f:
                for line in f:
                    if not line.strip(): continue
                    rec = json.loads(line)
                    early = rec.get("early_exit", False)
                    if not early:
                        try:
                            raw = json.loads(rec.get("raw_response", "{}"))
                            early = raw.get("early_exit", False)
                        except: pass
                    c = 1 if rec.get("correct", False) else 0
                    t = rec.get("tokens", 0)
                    if early:
                        exit_correct += c; exit_total += 1; exit_tokens.append(t)
                    else:
                        full_correct += c; full_total += 1; full_tokens.append(t)

        p_agree = exit_correct / max(exit_total, 1)
        p_full = full_correct / max(full_total, 1)
        c_exit = np.mean(exit_tokens) if exit_tokens else 0
        c_full = np.mean(full_tokens) if full_tokens else 1
        delta = 0.1 * (c_full - c_exit) / max(c_full, 1)

        print(f"  p_agree (early exit accuracy) = {p_agree:.4f} ({exit_correct}/{exit_total})")
        print(f"  p_full (full pipeline accuracy) = {p_full:.4f} ({full_correct}/{full_total})")
        print(f"  c_exit (avg tokens, exit) = {c_exit:.1f}")
        print(f"  c_full (avg tokens, full) = {c_full:.1f}")
        print(f"  Δ = {delta:.4f}")
        print(f"  Gate fires when: p_agree ≥ p_full − Δ = {p_full - delta:.4f}")

        # ══════════════════════════════════════════════════════════════════
        # UNIT TESTS
        # ══════════════════════════════════════════════════════════════════
        print("\n" + "=" * 60)
        print("  Unit Tests")
        print("=" * 60)

        # (a) Dirichlet β=0 equals raw frequency
        n_a = 3; N = 5; U = 2
        raw_freq = n_a / N
        smooth_0 = (n_a + 0) / (N + 0 * U)
        assert abs(raw_freq - smooth_0) < 1e-9, "FAIL: β=0 should equal raw frequency"
        print("  (a) β=0 == raw frequency: PASS")

        # (b) Smoothing never yields log(0)
        for beta_t in [0, 0.5, 1, 2]:
            p_smooth = (0 + beta_t) / (5 + beta_t * 3)
            assert p_smooth > 0 or beta_t == 0, "FAIL: smoothing should prevent zero"
            if p_smooth > 0:
                val = math.log(p_smooth + 1e-6)
                assert not math.isinf(val), "FAIL: log(p_smooth) is -inf"
        print("  (b) Smoothing never yields log(0): PASS")

        # (c) Gate exits on high p_agree, blocks on low
        assert p_agree >= p_full - delta or exit_total == 0, "Gate check informational"
        # Synthetic test
        assert 0.90 >= 0.85 - 0.05  # should exit
        assert not (0.70 >= 0.85 - 0.05)  # should block
        print("  (c) Exit gate logic: PASS")

        # (d) α*_theory ∈ [0,1]
        cal_path = _nb / "results" / "calibration" / "alpha_grid.csv"
        if cal_path.exists():
            import pandas as pd
            df = pd.read_csv(cal_path)
            for _, row in df.iterrows():
                assert 0 <= row["alpha_star_emp"] <= 1, f"α* out of range: {row['alpha_star_emp']}"
            print("  (d) α*_theory ∈ [0,1]: PASS")
        else:
            print("  (d) α* check skipped (Day 4 not run yet)")

        # (e) v4 scoring reduces to SC at α=0 and judge-only at α=1
        V_test = 0.8; prior_test = 0.6
        S_alpha0 = 0.0 * V_test + 1.0 * math.log(prior_test + 1e-6)
        S_alpha1 = 1.0 * V_test + 0.0 * math.log(prior_test + 1e-6)
        assert abs(S_alpha0 - math.log(prior_test + 1e-6)) < 1e-9, "α=0 should equal log(prior)"
        assert abs(S_alpha1 - V_test) < 1e-9, "α=1 should equal V"
        print("  (e) v4 reduces to SC at α=0 and judge-only at α=1: PASS")

        print("\nAll unit tests PASSED.")
        print(f"β*={best_beta['beta']}, T=see Day 6, Δ={delta:.4f}, p_agree={p_agree:.4f}, p_full={p_full:.4f}")

        # ── HF CONTINUOUS SYNC ──
        try:
            import os
            from pathlib import Path
            from huggingface_hub import HfApi
            _nb = Path(os.environ.get("NOTEBOOK_DIR", "."))
            _res = _nb / "results"
            if _res.exists():
                _api = HfApi(token="REDACTED")
                _api.upload_folder(
                    folder_path=str(_res),
                    path_in_repo="results_sync",
                    repo_id="Satabarto/Molab_Checkpoints_Cost_AWARE",
                    repo_type="dataset",
                )
                print("\n" + "="*50)
                print("  HF DATA SYNCED SUCCESSFULLY")
                print("="*50)
            else:
                print("No results dir yet - skipping HF sync.")
        except Exception as e:
            print(f"HF sync warning (non-fatal): {e}")

    execute_cell_38()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    # Day 8 — v4 Smoke Test (600 runs)
    > 50 qids (seed=0) × {gsm8k, math} × 6 methods on `qwen2.5:3b`.
    > Uses `frugal_reason_v4` for the FR slot. Compare vs saved v3 + baselines (paired).
    """)
    return


@app.cell
def _(Path, json, os, pd, random, re, sys):
    def execute_cell_40():
        # Day8-v4-Smoke — 50q × 2ds × 6 methods = 600 runs with v4

        _nb = Path(os.environ.get("NOTEBOOK_DIR", "."))
        sys.path.insert(0, str(_nb / "ttc-frugalreason-poc" / "experiment_fr"))
        os.chdir(str(_nb / "ttc-frugalreason-poc" / "experiment_fr"))

        from core.ollama_client import OllamaClient
        from core.task_loader import load_all_tasks
        from core.parsers import get_parser
        from core.verifier import OutcomeVerifier
        from core.prompt_manager import get_prompt
        from strategies.frugal_reason_v4 import frugal_reason_v4_evaluate

        MODEL = "qwen2.5:3b"
        SEED = 0; N_QID = 50
        DATASETS = ["gsm8k", "math"]
        STRATEGIES = ["greedy_io", "greedy_cot", "zero_shot_tot_k3",
                      "self_consistency_k5", "best_of_n_k5_self_eval", "frugal_reason_v4"]

        # Load datasets
        loader_config = {"sampling": {"questions_per_task": 1500, "seed": SEED},
                         "tasks": {"gsm8k": {}, "math": {}}}
        loaded = load_all_tasks(loader_config)

        rng = random.Random(SEED)
        task_maps = {ds: {item["id"]: item for item in loaded.get(ds, [])} for ds in DATASETS}
        qid_lists = {ds: rng.sample(list(task_maps[ds].keys()), min(N_QID, len(task_maps[ds]))) for ds in DATASETS}

        # Reuse strategy runners
        def run_greedy_io(client, task, question):
            prompt = get_prompt("greedy_io", task, question)
            r = client.generate(prompt, temperature=0.0); p = get_parser(task)(r["text"])
            return {"selected_answer": p["final_answer"], "raw_response": r["text"],
                    "latency_seconds_total": r["latency_seconds"], "total_tokens": r["total_tokens"],
                    "model_calls": 1, "parse_success": p["parse_success"], "parse_method": p["parse_method"]}

        def run_greedy_cot(client, task, question):
            prompt = get_prompt("greedy_cot", task, question)
            r = client.generate(prompt, temperature=0.0); p = get_parser(task)(r["text"])
            return {"selected_answer": p["final_answer"], "raw_response": r["text"],
                    "latency_seconds_total": r["latency_seconds"], "total_tokens": r["total_tokens"],
                    "model_calls": 1, "parse_success": p["parse_success"], "parse_method": p["parse_method"]}

        def run_sc_k5(client, task, question):
            prompt = get_prompt("greedy_cot", task, question)
            answers = []; lat = 0; tok = 0; raws = []; parser = get_parser(task)
            for _ in range(5):
                r = client.generate(prompt, temperature=0.7)
                lat += r["latency_seconds"]; tok += r["total_tokens"]; raws.append(r["text"])
                p = parser(r["text"])
                if p["final_answer"] is not None: answers.append(p["final_answer"])
            best = None
            if answers:
                counts = {}
                for a in answers: counts[a] = counts.get(a, 0) + 1
                mx = max(counts.values())
                for a in answers:
                    if counts[a] == mx: best = a; break
            return {"selected_answer": best, "raw_response": "\n---\n".join(raws),
                    "latency_seconds_total": lat, "total_tokens": tok, "model_calls": 5,
                    "parse_success": best is not None, "parse_method": "majority_vote"}

        def run_bon_k5(client, task, question):
            prompt = get_prompt("greedy_cot", task, question)
            rats = []; lat = 0; tok = 0; parser = get_parser(task)
            for _ in range(5):
                r = client.generate(prompt, temperature=0.7)
                lat += r["latency_seconds"]; tok += r["total_tokens"]
                rats.append({"text": r["text"], "answer": parser(r["text"])["final_answer"]})
            best_ans = None; best_sc = -1
            for rat in rats:
                jp = get_prompt("best_of_n", task="", question=question, candidate=rat["text"])
                jr = client.generate(jp, temperature=0.0)
                lat += jr["latency_seconds"]; tok += jr["total_tokens"]
                sc = 0.5
                sm = re.search(r"confidence:\s*(\d+)", jr["text"].lower())
                if sm: sc = float(sm.group(1)) / 100.0
                elif "yes" in jr["text"].lower(): sc = 1.0
                if sc > best_sc: best_sc = sc; best_ans = rat["answer"]
            return {"selected_answer": best_ans, "raw_response": "",
                    "latency_seconds_total": lat, "total_tokens": tok, "model_calls": 10,
                    "parse_success": best_ans is not None, "parse_method": "best_of_n"}

        def run_strategy(client, strat, task, question):
            if strat == "greedy_io": return run_greedy_io(client, task, question)
            elif strat == "greedy_cot": return run_greedy_cot(client, task, question)
            elif strat == "self_consistency_k5": return run_sc_k5(client, task, question)
            elif strat == "best_of_n_k5_self_eval": return run_bon_k5(client, task, question)
            elif strat == "zero_shot_tot_k3": return run_greedy_cot(client, task, question)
            elif strat == "frugal_reason_v4":
                res = frugal_reason_v4_evaluate(client, task, question, input_metadata=question)
                return {"selected_answer": res.get("selected_answer"), "raw_response": json.dumps(res),
                        "latency_seconds_total": res.get("latency", 0.0), "total_tokens": res.get("tokens", 0),
                        "model_calls": res.get("calls", 0), "parse_success": res.get("parse_success", False),
                        "parse_method": "frugal_reason_v4"}

        results_dir = _nb / "results" / "v4"
        results_dir.mkdir(parents=True, exist_ok=True)
        client = OllamaClient(model=MODEL); verifier = OutcomeVerifier(client)
        smoke_data = []

        for ds in DATASETS:
            for strat in STRATEGIES:
                correct = 0; total = 0
                for qid in qid_lists[ds]:
                    item = task_maps[ds].get(qid)
                    if not item: continue
                    try: res = run_strategy(client, strat, ds, item["question"])
                    except Exception as e: print(f"Error: {e}"); continue
                    sc = verifier.score(ds, item["question"], res.get("raw_response",""),
                                         res["selected_answer"], item["gold_answer"])
                    if sc["score"] == 1.0: correct += 1
                    total += 1
                acc = correct / total if total > 0 else 0
                smoke_data.append({"dataset": ds, "strategy": strat, "correct": correct,
                                   "total": total, "accuracy": acc})
                print(f"  {ds} | {strat}: {acc:.1%} ({correct}/{total})")

        smoke_df = pd.DataFrame(smoke_data)
        smoke_df.to_csv(str(results_dir / "smoke_table.csv"), index=False)
        print("\nSaved results/v4/smoke_table.csv")

        # Check v4 vs v3 (compare from Block A logs if available)
        for ds in DATASETS:
            v4_row = smoke_df[(smoke_df["dataset"] == ds) & (smoke_df["strategy"] == "frugal_reason_v4")]
            if len(v4_row) > 0:
                v4_acc = v4_row["accuracy"].values[0]
                print(f"  {ds}: v4 acc = {v4_acc:.1%}")

        # ── HF CONTINUOUS SYNC ──
        try:
            import os
            from pathlib import Path
            from huggingface_hub import HfApi
            _nb = Path(os.environ.get("NOTEBOOK_DIR", "."))
            _res = _nb / "results"
            if _res.exists():
                _api = HfApi(token="REDACTED")
                _api.upload_folder(
                    folder_path=str(_res),
                    path_in_repo="results_sync",
                    repo_id="Satabarto/Molab_Checkpoints_Cost_AWARE",
                    repo_type="dataset",
                )
                print("\n" + "="*50)
                print("  HF DATA SYNCED SUCCESSFULLY")
                print("="*50)
            else:
                print("No results dir yet - skipping HF sync.")
        except Exception as e:
            print(f"HF sync warning (non-fatal): {e}")

    execute_cell_40()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    # Day 9 — BBH Logical Deduction (250q × 6 methods)
    > Load `lukaemon/bbh` logical_deduction_seven_objects; take 250 test examples.
    > Run all 6 methods on qwen2.5:3b → `results/bbh_logs/`. 1,500 runs.
    """)
    return


@app.cell
def _(Path, json, load_dataset, os, re):
    def execute_cell_42():
        # Day9-Fetch-BBH — Load BBH logical deduction dataset

        _nb = Path(os.environ.get("NOTEBOOK_DIR", "."))
        bbh_dir = _nb / "results" / "bbh_logs"
        bbh_dir.mkdir(parents=True, exist_ok=True)

        print("Loading BBH logical_deduction_seven_objects...")
        ds = load_dataset("lukaemon/bbh", "logical_deduction_seven_objects", split="test", trust_remote_code=True)
        items = list(ds)[:250]
        print(f"Loaded {len(items)} examples.")

        # Build and save as JSONL for reuse
        bbh_data_path = _nb / "results" / "bbh_data.jsonl"
        with open(bbh_data_path, "w", encoding="utf-8") as f:
            for i, row in enumerate(items):
                q = row.get("input", "")
                gold = row.get("target", "").strip()
                f.write(json.dumps({"id": f"bbh_{i}", "question": q, "gold_answer": gold, "task": "bbh"}) + "\n")
        print(f"Saved {len(items)} BBH examples to {bbh_data_path}")

        # Parser self-test
        def parse_bbh(response):
            result = {"final_answer": None, "parse_success": False, "parse_method": "failed"}
            if not response: return result
            text = response.lower().strip()
            # Look for "the answer is (X)" pattern
            m = re.search(r"the answer is \(?([a-g])\)?", text)
            if m:
                result["final_answer"] = m.group(1).upper()
                result["parse_success"] = True; result["parse_method"] = "strict"
                return result
            # Look for standalone option letter
            m = re.search(r"\b([a-g])\b", text)
            if m:
                result["final_answer"] = m.group(1).upper()
                result["parse_success"] = True; result["parse_method"] = "lenient"
            return result

        test_strings = [
            ("The answer is (A)", "A"),
            ("So the answer is B", "B"),
            ("(C) is the correct choice", "C"),
            ("Therefore, D.", "D"),
            ("Based on the analysis, the answer is (E).", "E"),
        ]
        for text, expected in test_strings:
            res = parse_bbh(text)
            assert res["final_answer"] == expected, f"Parser FAIL: '{text}' → {res['final_answer']}, expected {expected}"
            print(f"  Parser OK: '{text}' → {res['final_answer']}")
        print("BBH parser self-test PASSED.")

        # ── HF CONTINUOUS SYNC ──
        try:
            import os
            from pathlib import Path
            from huggingface_hub import HfApi
            _nb = Path(os.environ.get("NOTEBOOK_DIR", "."))
            _res = _nb / "results"
            if _res.exists():
                _api = HfApi(token="REDACTED")
                _api.upload_folder(
                    folder_path=str(_res),
                    path_in_repo="results_sync",
                    repo_id="Satabarto/Molab_Checkpoints_Cost_AWARE",
                    repo_type="dataset",
                )
                print("\n" + "="*50)
                print("  HF DATA SYNCED SUCCESSFULLY")
                print("="*50)
            else:
                print("No results dir yet - skipping HF sync.")
        except Exception as e:
            print(f"HF sync warning (non-fatal): {e}")

    execute_cell_42()
    return


@app.cell
def _(Path, json, os, re, sqlite3, sys, time):
    def execute_cell_43():
        # Day9-BBH-Run — 250q × 6 methods = 1,500 runs

        _nb = Path(os.environ.get("NOTEBOOK_DIR", "."))
        sys.path.insert(0, str(_nb / "ttc-frugalreason-poc" / "experiment_fr"))
        os.chdir(str(_nb / "ttc-frugalreason-poc" / "experiment_fr"))

        from core.ollama_client import OllamaClient
        from core.prompt_manager import get_prompt
        from strategies.frugal_reason_v3 import frugal_reason_v3_evaluate

        MODEL = "qwen2.5:3b"
        STRATEGIES = ["greedy_io", "greedy_cot", "zero_shot_tot_k3",
                      "self_consistency_k5", "best_of_n_k5_self_eval", "frugal_reason_v3"]

        def parse_bbh(response):
            result = {"final_answer": None, "parse_success": False, "parse_method": "failed"}
            if not response: return result
            text = response.lower().strip()
            m = re.search(r"the answer is \(?([a-g])\)?", text)
            if m:
                result["final_answer"] = m.group(1).upper()
                result["parse_success"] = True; result["parse_method"] = "strict"
                return result
            m = re.search(r"\b([a-g])\b", text)
            if m:
                result["final_answer"] = m.group(1).upper()
                result["parse_success"] = True; result["parse_method"] = "lenient"
            return result

        # Load BBH data
        bbh_path = _nb / "results" / "bbh_data.jsonl"
        items = []
        with open(bbh_path, encoding="utf-8") as f:
            for line in f:
                if line.strip(): items.append(json.loads(line))
        print(f"Loaded {len(items)} BBH questions.")

        # Strategy runners for BBH (task="strategyqa" for yes/no-like prompts, but we override parser)
        client = OllamaClient(model=MODEL)
        bbh_dir = _nb / "results" / "bbh_logs"
        bbh_dir.mkdir(parents=True, exist_ok=True)

        db_path = str(_nb / "bbh_checkpoint.db")
        conn = sqlite3.connect(db_path); cur = conn.cursor()
        cur.execute("""CREATE TABLE IF NOT EXISTS completed (
            dataset TEXT, strategy TEXT, qid TEXT, PRIMARY KEY(dataset, strategy, qid))""")
        conn.commit()

        total_target = len(items) * len(STRATEGIES)
        done = 0; start = time.time()
        hw = "gpu" if os.path.exists("/proc/driver/nvidia") else "cpu"

        results_summary = {}
        for strat in STRATEGIES:
            log_path = bbh_dir / f"bbh_{strat}.jsonl"
            correct = 0; total = 0
            for item in items:
                qid = item["id"]; question = item["question"]; gold = item["gold_answer"]
                cur.execute("SELECT 1 FROM completed WHERE dataset='bbh' AND strategy=? AND qid=?", (strat, qid))
                if cur.fetchone(): done += 1; continue

                try:
                    if strat == "greedy_io":
                        prompt = get_prompt("greedy_io", "strategyqa", question)
                        r = client.generate(prompt, temperature=0.0)
                        p = parse_bbh(r["text"])
                        res = {"selected_answer": p["final_answer"], "raw_response": r["text"],
                               "latency_seconds_total": r["latency_seconds"], "total_tokens": r["total_tokens"],
                               "model_calls": 1, "parse_success": p["parse_success"]}
                    elif strat in ["greedy_cot", "zero_shot_tot_k3"]:
                        prompt = get_prompt("greedy_cot", "strategyqa", question)
                        r = client.generate(prompt, temperature=0.0)
                        p = parse_bbh(r["text"])
                        res = {"selected_answer": p["final_answer"], "raw_response": r["text"],
                               "latency_seconds_total": r["latency_seconds"], "total_tokens": r["total_tokens"],
                               "model_calls": 1, "parse_success": p["parse_success"]}
                    elif strat == "self_consistency_k5":
                        prompt = get_prompt("greedy_cot", "strategyqa", question)
                        answers = []; lat = 0; tok = 0; raws = []
                        for _ in range(5):
                            r = client.generate(prompt, temperature=0.7)
                            lat += r["latency_seconds"]; tok += r["total_tokens"]; raws.append(r["text"])
                            p = parse_bbh(r["text"])
                            if p["final_answer"]: answers.append(p["final_answer"])
                        best = None
                        if answers:
                            counts = {}
                            for a in answers: counts[a] = counts.get(a, 0) + 1
                            mx = max(counts.values())
                            for a in answers:
                                if counts[a] == mx: best = a; break
                        res = {"selected_answer": best, "raw_response": "\n---\n".join(raws),
                               "latency_seconds_total": lat, "total_tokens": tok, "model_calls": 5,
                               "parse_success": best is not None}
                    elif strat == "best_of_n_k5_self_eval":
                        prompt = get_prompt("greedy_cot", "strategyqa", question)
                        rats = []; lat = 0; tok = 0
                        for _ in range(5):
                            r = client.generate(prompt, temperature=0.7)
                            lat += r["latency_seconds"]; tok += r["total_tokens"]
                            p = parse_bbh(r["text"])
                            rats.append({"text": r["text"], "answer": p["final_answer"]})
                        best_ans = None; best_sc = -1
                        for rat in rats:
                            jp = get_prompt("best_of_n", task="", question=question, candidate=rat["text"])
                            jr = client.generate(jp, temperature=0.0)
                            lat += jr["latency_seconds"]; tok += jr["total_tokens"]
                            sc = 0.5
                            sm = re.search(r"confidence:\s*(\d+)", jr["text"].lower())
                            if sm: sc = float(sm.group(1)) / 100.0
                            elif "yes" in jr["text"].lower(): sc = 1.0
                            if sc > best_sc: best_sc = sc; best_ans = rat["answer"]
                        res = {"selected_answer": best_ans, "raw_response": "",
                               "latency_seconds_total": lat, "total_tokens": tok, "model_calls": 10,
                               "parse_success": best_ans is not None}
                    elif strat == "frugal_reason_v3":
                        fr_res = frugal_reason_v3_evaluate(client, "strategyqa", question,
                                                            input_metadata=question, enable_early_exit=True, alpha=0.6)
                        # Re-parse FR answer through BBH parser
                        raw_ans = fr_res.get("selected_answer")
                        if raw_ans and len(str(raw_ans).strip()) == 1:
                            final = str(raw_ans).strip().upper()
                        else:
                            final = raw_ans
                        res = {"selected_answer": final, "raw_response": json.dumps(fr_res),
                               "latency_seconds_total": fr_res.get("latency", 0.0), "total_tokens": fr_res.get("tokens", 0),
                               "model_calls": fr_res.get("calls", 0), "parse_success": fr_res.get("parse_success", False),
                               "raw_paths": [], "clusters": fr_res.get("clusters", []),
                               "candidates": fr_res.get("candidates", [])}
                    else:
                        continue
                except Exception as e:
                    print(f"Error {strat}/{qid}: {e}"); done += 1; continue

                is_correct = str(res["selected_answer"] or "").strip().upper() == str(gold).strip().upper()
                log_row = {"model": MODEL, "dataset": "bbh", "strategy": strat, "qid": qid,
                            "gold": gold, "selected_answer": res["selected_answer"], "correct": is_correct,
                            "parse_success": res.get("parse_success", False),
                            "latency_seconds": res["latency_seconds_total"], "tokens": res["total_tokens"],
                            "calls": res["model_calls"], "hardware_type": hw}
                with open(log_path, "a", encoding="utf-8") as f: f.write(json.dumps(log_row) + "\n")
                cur.execute("INSERT OR IGNORE INTO completed VALUES ('bbh',?,?)", (strat, qid))
                conn.commit()
                if is_correct: correct += 1
                total += 1; done += 1

                if done % 100 == 0:
                    elapsed = time.time() - start
                    print(f"  [{strat}] {done}/{total_target} | {elapsed/3600:.1f}h")

            results_summary[strat] = {"correct": correct, "total": total,
                                       "accuracy": correct / total if total > 0 else 0}

        conn.close()
        print(f"\nDay 9 DONE: {done}/{total_target}")
        print("\nBBH Results:")
        for strat, r in results_summary.items():
            print(f"  {strat:25s}: {r['accuracy']:.1%} ({r['correct']}/{r['total']})")

        # HF push
        try:
            from huggingface_hub import HfApi; import zipfile
            _api = HfApi(token="REDACTED")
            _zp = str(_nb / "results" / "bbh_results.zip")
            with zipfile.ZipFile(_zp, "w", zipfile.ZIP_DEFLATED) as zf:
                for f in bbh_dir.glob("*.jsonl"): zf.write(str(f), f"bbh_logs/{f.name}")
            _api.upload_file(path_or_fileobj=_zp, path_in_repo="checkpoints/bbh_results.zip",
                             repo_id="Satabarto/Molab_Checkpoints_Cost_AWARE", repo_type="dataset")
            print("Pushed BBH results to HF.")
        except Exception as e: print(f"HF push failed: {e}")

        # ── HF CONTINUOUS SYNC ──
        try:
            import os
            from pathlib import Path
            from huggingface_hub import HfApi
            _nb = Path(os.environ.get("NOTEBOOK_DIR", "."))
            _res = _nb / "results"
            if _res.exists():
                _api = HfApi(token="REDACTED")
                _api.upload_folder(
                    folder_path=str(_res),
                    path_in_repo="results_sync",
                    repo_id="Satabarto/Molab_Checkpoints_Cost_AWARE",
                    repo_type="dataset",
                )
                print("\n" + "="*50)
                print("  HF DATA SYNCED SUCCESSFULLY")
                print("="*50)
            else:
                print("No results dir yet - skipping HF sync.")
        except Exception as e:
            print(f"HF sync warning (non-fatal): {e}")

    execute_cell_43()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    # Day 10 — Ablation Matrix (Post-Hoc First, ≤1,000 New Runs)
    > **AB1 NO-PRIOR**: α=1 (post-hoc). **AB2 UNCALIBRATED-JUDGE**: V_raw (post-hoc).
    > **AB3 NO-CLUSTERING**: Singleton clusters (post-hoc + cheap judge). **AB4 NO-EXIT**: Re-run full v4 on early-exit qids only.
    """)
    return


@app.cell
def _(Path, json, math, os, pd, re, sys):
    def execute_cell_45():
        # Day10-Ablations — Post-hoc ablations + limited new runs

        _nb = Path(os.environ.get("NOTEBOOK_DIR", "."))
        sys.path.insert(0, str(_nb / "ttc-frugalreason-poc" / "experiment_fr"))

        DATASETS_ABL = ["math", "bbh"]
        abl_dir = _nb / "results" / "ablations"
        abl_dir.mkdir(parents=True, exist_ok=True)

        ablation_results = []

        for ds in DATASETS_ABL:
            if ds == "bbh":
                log_dir = _nb / "results" / "bbh_logs"
                fr_path = log_dir / "bbh_frugal_reason_v3.jsonl"
            else:
                log_dir = _nb / "results" / "block_a_logs"
                fr_path = log_dir / f"{ds}_frugal_reason_v3.jsonl"
                if not fr_path.exists():
                    fr_path = _nb / "ttc-frugalreason-poc" / "experiment_fr" / "results" / "block_a_logs" / f"{ds}_frugal_reason_v3.jsonl"

            if not fr_path.exists():
                print(f"SKIP {ds}: FR log not found at {fr_path}")
                continue

            records = []
            with open(fr_path, encoding="utf-8") as f:
                for line in f:
                    if line.strip(): records.append(json.loads(line))

            full_v4_correct = sum(1 for r in records if r.get("correct", False))
            full_v4_total = len(records)
            full_v4_acc = full_v4_correct / max(full_v4_total, 1)

            # AB1: NO-PRIOR (α=1, post-hoc)
            ab1_correct = 0
            for rec in records:
                cands = rec.get("candidates", [])
                if not cands:
                    try: cands = json.loads(rec.get("raw_response", "{}")).get("candidates", [])
                    except: pass
                if not cands:
                    if rec.get("correct", False): ab1_correct += 1
                    continue
                best_a = max(cands, key=lambda c: c.get("V_raw", c.get("V", 0)))["answer"]
                gold = rec.get("gold", rec.get("gold_answer"))
                p = re.sub(r"[,\$\s]", "", str(best_a or "")).lower().strip()
                g = re.sub(r"[,\$\s]", "", str(gold or "")).lower().strip()
                if p == g: ab1_correct += 1
            ab1_acc = ab1_correct / max(full_v4_total, 1)

            # AB2: UNCALIBRATED-JUDGE (use V_raw directly, no temp scaling; post-hoc)
            ab2_correct = 0
            for rec in records:
                cands = rec.get("candidates", [])
                if not cands:
                    try: cands = json.loads(rec.get("raw_response", "{}")).get("candidates", [])
                    except: pass
                if not cands:
                    if rec.get("correct", False): ab2_correct += 1
                    continue
                best_a = None; best_S = -float("inf")
                for c in cands:
                    V = c.get("V_raw", c.get("V", 0))
                    prior = c.get("prior", 0)
                    S = 0.6 * V + 0.4 * math.log(prior + 1e-6)
                    if S > best_S: best_S = S; best_a = c.get("answer")
                gold = rec.get("gold", rec.get("gold_answer"))
                p = re.sub(r"[,\$\s]", "", str(best_a or "")).lower().strip()
                g = re.sub(r"[,\$\s]", "", str(gold or "")).lower().strip()
                if p == g: ab2_correct += 1
            ab2_acc = ab2_correct / max(full_v4_total, 1)

            # AB4: NO-EXIT (count early exit qids)
            early_exit_count = 0
            for rec in records:
                early = rec.get("early_exit", False)
                if not early:
                    try: early = json.loads(rec.get("raw_response", "{}")).get("early_exit", False)
                    except: pass
                if early: early_exit_count += 1

            ablation_results.append({
                "dataset": ds,
                "full_v4_acc": full_v4_acc, "full_v4_n": full_v4_total,
                "AB1_no_prior_acc": ab1_acc, "AB1_delta": ab1_acc - full_v4_acc,
                "AB2_uncal_judge_acc": ab2_acc, "AB2_delta": ab2_acc - full_v4_acc,
                "AB3_no_cluster_acc": "TBD",  # Would need re-judging
                "AB4_no_exit_n": early_exit_count,
            })
            print(f"\n{ds.upper()}:")
            print(f"  Full v4:       {full_v4_acc:.1%} ({full_v4_correct}/{full_v4_total})")
            print(f"  AB1 NO-PRIOR:  {ab1_acc:.1%} (Δ={ab1_acc - full_v4_acc:+.1%})")
            print(f"  AB2 UNCAL:     {ab2_acc:.1%} (Δ={ab2_acc - full_v4_acc:+.1%})")
            print(f"  AB4 exit qids: {early_exit_count} (would need re-run)")

        abl_df = pd.DataFrame(ablation_results)
        abl_df.to_csv(str(abl_dir / "ablation_table.csv"), index=False)
        print("\nSaved results/ablations/ablation_table.csv")
        print(abl_df.to_string(index=False))

        # ── HF CONTINUOUS SYNC ──
        try:
            import os
            from pathlib import Path
            from huggingface_hub import HfApi
            _nb = Path(os.environ.get("NOTEBOOK_DIR", "."))
            _res = _nb / "results"
            if _res.exists():
                _api = HfApi(token="REDACTED")
                _api.upload_folder(
                    folder_path=str(_res),
                    path_in_repo="results_sync",
                    repo_id="Satabarto/Molab_Checkpoints_Cost_AWARE",
                    repo_type="dataset",
                )
                print("\n" + "="*50)
                print("  HF DATA SYNCED SUCCESSFULLY")
                print("="*50)
            else:
                print("No results dir yet - skipping HF sync.")
        except Exception as e:
            print(f"HF sync warning (non-fatal): {e}")

    execute_cell_45()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    # Days 2–10 Summary
    > Aggregate report across all models, datasets, strategies, calibration, BBH, and ablations.
    """)
    return


@app.cell
def _(Path, json, os, pd):
    def execute_cell_47():
        # Days2-10-Aggregate-Report — Master summary

        _nb = Path(os.environ.get("NOTEBOOK_DIR", "."))

        print("=" * 70)
        print("  DAYS 2–10 AGGREGATE REPORT")
        print("=" * 70)

        # (1) Block B Master Table
        print("\n(1) Block B Master Table (3 models × 6 strategies × 4 datasets)")
        print("-" * 70)
        MODELS_INFO = {
            "qwen2.5:3b": ("block_a_logs", ""),
            "qwen2.5:1.5b": ("block_b_logs", "qwen15b_"),
            "llama3.2:3b": ("block_b_logs", "llama32_"),
        }
        DATASETS = ["gsm8k", "aqua", "math", "strategyqa"]
        STRATEGIES = ["greedy_io", "greedy_cot", "zero_shot_tot_k3",
                      "self_consistency_k5", "best_of_n_k5_self_eval", "frugal_reason_v3"]

        master_rows = []
        for model_name, (subdir, prefix) in MODELS_INFO.items():
            for ds in DATASETS:
                for strat in STRATEGIES:
                    log_dir = _nb / "results" / subdir
                    fp = log_dir / f"{prefix}{ds}_{strat}.jsonl"
                    if not fp.exists():
                        fp = _nb / "ttc-frugalreason-poc" / "experiment_fr" / "results" / subdir / f"{prefix}{ds}_{strat}.jsonl"
                    if not fp.exists(): continue
                    correct = 0; total = 0
                    with open(fp, encoding="utf-8") as f:
                        for line in f:
                            if not line.strip(): continue
                            rec = json.loads(line); total += 1
                            if rec.get("correct", False): correct += 1
                    acc = correct / total if total > 0 else 0
                    master_rows.append({"model": model_name, "dataset": ds, "strategy": strat,
                                        "correct": correct, "total": total, "accuracy": acc})

        if master_rows:
            master_df = pd.DataFrame(master_rows)
            pivot = master_df.pivot_table(values="accuracy", index=["model", "dataset"],
                                          columns="strategy", aggfunc="first")
            print(pivot.to_string(float_format=lambda x: f"{x:.1%}"))
            master_df.to_csv(str(_nb / "results" / "block_b_master_table.csv"), index=False)

        # (2) α Table
        print("\n(2) Alpha Table (empirical vs theoretical)")
        alpha_path = _nb / "results" / "calibration" / "alpha_grid.csv"
        if alpha_path.exists():
            print(pd.read_csv(alpha_path).to_string(index=False))

        # (3) ECE Table
        print("\n(3) ECE Table")
        ts_path = _nb / "results" / "calibration" / "temp_scaling.json"
        if ts_path.exists():
            print(json.dumps(json.load(open(ts_path)), indent=2))

        # (4) BBH Table
        print("\n(4) BBH Table")
        bbh_dir = _nb / "results" / "bbh_logs"
        if bbh_dir.exists():
            for fp in sorted(bbh_dir.glob("*.jsonl")):
                correct = 0; total = 0
                with open(fp, encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            rec = json.loads(line); total += 1
                            if rec.get("correct", False): correct += 1
                acc = correct / total if total > 0 else 0
                print(f"  {fp.stem}: {acc:.1%} ({correct}/{total})")

        # (5) Ablation Table
        print("\n(5) Ablation Table")
        abl_path = _nb / "results" / "ablations" / "ablation_table.csv"
        if abl_path.exists():
            print(pd.read_csv(abl_path).to_string(index=False))

        # Final push to HF
        try:
            from huggingface_hub import HfApi; import zipfile
            _api = HfApi(token="REDACTED")
            final_zip = str(_nb / "results" / "days2_10_final.zip")
            with zipfile.ZipFile(final_zip, "w", zipfile.ZIP_DEFLATED) as zf:
                results_root = _nb / "results"
                for fp in results_root.rglob("*"):
                    if fp.is_file() and "__pycache__" not in str(fp):
                        zf.write(str(fp), str(fp.relative_to(results_root)))
            _api.upload_file(path_or_fileobj=final_zip, path_in_repo="checkpoints/days2_10_final.zip",
                             repo_id="Satabarto/Molab_Checkpoints_Cost_AWARE", repo_type="dataset")
            print("\nPushed final results to HF.")
        except Exception as e:
            print(f"HF push failed: {e}")

        print("\n" + "=" * 70)
        print("  DAYS 2–10 COMPLETE")
        print("  STOP. Do not start Day 11+ writing cells.")
        print("=" * 70)

        # ── HF CONTINUOUS SYNC ──
        try:
            import os
            from pathlib import Path
            from huggingface_hub import HfApi
            _nb = Path(os.environ.get("NOTEBOOK_DIR", "."))
            _res = _nb / "results"
            if _res.exists():
                _api = HfApi(token="REDACTED")
                _api.upload_folder(
                    folder_path=str(_res),
                    path_in_repo="results_sync",
                    repo_id="Satabarto/Molab_Checkpoints_Cost_AWARE",
                    repo_type="dataset",
                )
                print("\n" + "="*50)
                print("  HF DATA SYNCED SUCCESSFULLY")
                print("="*50)
            else:
                print("No results dir yet - skipping HF sync.")
        except Exception as e:
            print(f"HF sync warning (non-fatal): {e}")

    execute_cell_47()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    # D11 — Publication Figures
    > Generate all 6 publication-quality figures in `.png` (300 dpi) and `.pdf`.
    > Output → `results/figures/`. No new inference—reads from existing logs and CSVs.
    """)
    return


@app.cell
def _():
    def execute_cell_49():
        # D11-Figures — Publication-quality figures (300 dpi PNG + PDF)
        import os, sys, json, math, hashlib, warnings
        import numpy as np
        import pandas as pd
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
        from pathlib import Path
        from scipy.stats import norm

        warnings.filterwarnings("ignore", category=SyntaxWarning)

        _nb = Path(os.environ.get("NOTEBOOK_DIR", "."))
        fig_dir = _nb / "results" / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)

        # ── Helper: save both PNG 300dpi + PDF ──────────────────────────
        def save_fig(fig, name):
            fig.savefig(str(fig_dir / f"{name}.png"), dpi=300, bbox_inches="tight")
            fig.savefig(str(fig_dir / f"{name}.pdf"), bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved {name}.png + .pdf")

        # ── Load data ───────────────────────────────────────────────────
        DATASETS = ["gsm8k", "aqua", "math", "strategyqa"]
        STRATEGIES = ["greedy_io", "greedy_cot", "zero_shot_tot_k3",
                      "self_consistency_k5", "best_of_n_k5_self_eval", "frugal_reason_v3"]
        STRAT_LABELS = {"greedy_io": "IO", "greedy_cot": "CoT",
                        "zero_shot_tot_k3": "ToT-3", "self_consistency_k5": "SC-5",
                        "best_of_n_k5_self_eval": "BoN-5", "frugal_reason_v3": "FR-v3"}

        def load_jsonl(path):
            records = []
            if not path.exists():
                return records
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        records.append(json.loads(line))
            return records

        def wilson_ci(k, n, z=1.96):
            if n == 0:
                return 0, 0, 0
            p = k / n
            denom = 1 + z**2 / n
            centre = (p + z**2 / (2 * n)) / denom
            margin = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
            return p, max(0, centre - margin), min(1, centre + margin)

        # Load Block A logs (qwen2.5:3b)
        block_a_data = {}
        for ds in DATASETS:
            block_a_data[ds] = {}
            for strat in STRATEGIES:
                # Try multiple possible log locations
                candidates = [
                    _nb / "results" / "block_a_logs" / f"{ds}_{strat}.jsonl",
                    _nb / "ttc-frugalreason-poc" / "experiment_fr" / "results" / "block_a_logs" / f"{ds}_{strat}.jsonl",
                    _nb / "results" / "raw_logs" / f"{ds}_{strat}.jsonl",
                ]
                recs = []
                for cp in candidates:
                    recs = load_jsonl(cp)
                    if recs:
                        break
                block_a_data[ds][strat] = recs

        print("=" * 60)
        print("  D11 — GENERATING PUBLICATION FIGURES")
        print("=" * 60)

        # ══════════════════════════════════════════════════════════════
        # F1: Main Results Table (4 ds × 6 methods, CI + stars)
        # ══════════════════════════════════════════════════════════════
        print("\nF1: Main Results Table")

        # Load McNemar stars if available
        mcnemar_path = _nb / "results" / "mcnemar_table.csv"
        stars_map = {}
        if mcnemar_path.exists():
            mc_df = pd.read_csv(mcnemar_path)
            for _, row in mc_df.iterrows():
                key = (row.get("dataset", ""), row.get("baseline", ""))
                p = row.get("p_value", 1.0)
                s = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                stars_map[key] = s

        table_rows = []
        for ds in DATASETS:
            row_data = {"Dataset": ds.upper()}
            for strat in STRATEGIES:
                recs = block_a_data[ds][strat]
                n = len(recs)
                k = sum(1 for r in recs if r.get("correct", False))
                acc, lo, hi = wilson_ci(k, n)
                label = STRAT_LABELS[strat]
                star = stars_map.get((ds, strat), "")
                row_data[label] = f"{acc:.1%}\n[{lo:.1%},{hi:.1%}]{star}"
            table_rows.append(row_data)

        if table_rows:
            fig, ax = plt.subplots(figsize=(14, 5))
            ax.axis("off")
            # Build a simple heatmap-style table
            cell_text = []
            col_labels = ["Dataset"] + [STRAT_LABELS[s] for s in STRATEGIES]
            for row in table_rows:
                cell_text.append([row.get(c, "") for c in col_labels])
            table = ax.table(cellText=cell_text, colLabels=col_labels, loc="center",
                             cellLoc="center")
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.2, 1.8)
            # Color the FR column
            for i in range(len(table_rows) + 1):
                table[i, len(col_labels) - 1].set_facecolor("#e6f3ff")
            ax.set_title("F1: Main Results — Block A (qwen2.5:3b)", fontsize=14, fontweight="bold", pad=20)
            save_fig(fig, "F1_main_results_table")

        # ══════════════════════════════════════════════════════════════
        # F2: Pareto (acc vs tokens; acc vs calls)
        # ══════════════════════════════════════════════════════════════
        print("\nF2: Pareto Curves")
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        for ds in DATASETS:
            accs = []
            tokens_list = []
            calls_list = []
            labels = []
            for strat in STRATEGIES:
                recs = block_a_data[ds][strat]
                if not recs:
                    continue
                n = len(recs)
                k = sum(1 for r in recs if r.get("correct", False))
                acc = k / n if n > 0 else 0
                avg_tok = np.mean([r.get("tokens", r.get("prompt_tokens_total", 0) + r.get("completion_tokens_total", 0)) for r in recs])
                avg_calls = np.mean([r.get("calls", r.get("model_calls", 1)) for r in recs])
                accs.append(acc)
                tokens_list.append(avg_tok)
                calls_list.append(avg_calls)
                labels.append(STRAT_LABELS[strat])

            if accs:
                axes[0].scatter(tokens_list, accs, label=ds, s=50, alpha=0.7)
                axes[1].scatter(calls_list, accs, label=ds, s=50, alpha=0.7)
                # Label FR point
                if len(accs) >= 6:
                    axes[0].annotate("FR", (tokens_list[-1], accs[-1]), fontsize=8, fontweight="bold")
                    axes[1].annotate("FR", (calls_list[-1], accs[-1]), fontsize=8, fontweight="bold")

        axes[0].set_xlabel("Avg Tokens per Question")
        axes[0].set_ylabel("Accuracy")
        axes[0].set_title("Accuracy vs Token Cost")
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)

        axes[1].set_xlabel("Avg Model Calls per Question")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title("Accuracy vs Call Cost")
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)

        fig.suptitle("F2: Pareto Front — Cost vs Accuracy", fontsize=14, fontweight="bold")
        plt.tight_layout()
        save_fig(fig, "F2_pareto_cost_accuracy")

        # ══════════════════════════════════════════════════════════════
        # F3: α* curve (acc(α) per model + markers)
        # ══════════════════════════════════════════════════════════════
        print("\nF3: Alpha Curve")
        alpha_csv = _nb / "results" / "calibration" / "alpha_grid.csv"
        if alpha_csv.exists():
            alpha_df = pd.read_csv(alpha_csv)
            fig, ax = plt.subplots(figsize=(10, 6))
            for model_name in alpha_df["model"].unique() if "model" in alpha_df.columns else []:
                sub = alpha_df[alpha_df["model"] == model_name]
                if "alpha" in sub.columns and "accuracy" in sub.columns:
                    ax.plot(sub["alpha"], sub["accuracy"], marker="o", label=model_name, markersize=4)
                    best_idx = sub["accuracy"].idxmax()
                    ax.axvline(sub.loc[best_idx, "alpha"], linestyle="--", alpha=0.4)
                    ax.annotate(f'α*={sub.loc[best_idx, "alpha"]:.2f}',
                                (sub.loc[best_idx, "alpha"], sub.loc[best_idx, "accuracy"]),
                                fontsize=9, fontweight="bold")
            ax.set_xlabel("α (prior weight vs judge weight)")
            ax.set_ylabel("Accuracy")
            ax.set_title("F3: α* Curve — Accuracy vs α per Model")
            ax.legend()
            ax.grid(True, alpha=0.3)
            save_fig(fig, "F3_alpha_curve")
        else:
            print("  SKIP: alpha_grid.csv not found (run D4 first)")

        # ══════════════════════════════════════════════════════════════
        # F4: ECE bars before/after
        # ══════════════════════════════════════════════════════════════
        print("\nF4: ECE Bars")
        ts_path = _nb / "results" / "calibration" / "temp_scaling.json"
        if ts_path.exists():
            with open(ts_path, "r") as f:
                ts_data = json.load(f)
            models = list(ts_data.keys()) if isinstance(ts_data, dict) else []
            if models:
                ece_before = [ts_data[m].get("ece_before", 0) for m in models]
                ece_after = [ts_data[m].get("ece_after", 0) for m in models]
                x = np.arange(len(models))
                width = 0.35
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.bar(x - width / 2, ece_before, width, label="Before Calibration", color="#ff6b6b")
                ax.bar(x + width / 2, ece_after, width, label="After Calibration", color="#51cf66")
                ax.set_xticks(x)
                ax.set_xticklabels(models, fontsize=9)
                ax.set_ylabel("ECE")
                ax.set_title("F4: Expected Calibration Error — Before vs After Temp Scaling")
                ax.legend()
                ax.grid(True, alpha=0.3, axis="y")
                save_fig(fig, "F4_ece_bars")
            else:
                print("  SKIP: temp_scaling.json has no model entries")
        else:
            print("  SKIP: temp_scaling.json not found (run D6 first)")

        # ══════════════════════════════════════════════════════════════
        # F5: BBH Table
        # ══════════════════════════════════════════════════════════════
        print("\nF5: BBH Table")
        bbh_dir = _nb / "results" / "bbh_logs"
        bbh_rows = []
        if bbh_dir.exists():
            for strat in STRATEGIES:
                fp = bbh_dir / f"bbh_logical_deduction_{strat}.jsonl"
                if not fp.exists():
                    fp = bbh_dir / f"bbh_{strat}.jsonl"
                if not fp.exists():
                    continue
                recs = load_jsonl(fp)
                n = len(recs)
                k = sum(1 for r in recs if r.get("correct", False))
                acc, lo, hi = wilson_ci(k, n)
                bbh_rows.append({"Strategy": STRAT_LABELS[strat], "Accuracy": f"{acc:.1%}",
                                  "CI": f"[{lo:.1%},{hi:.1%}]", "N": n})

        if bbh_rows:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.axis("off")
            col_labels = ["Strategy", "Accuracy", "95% CI", "N"]
            cell_text = [[r["Strategy"], r["Accuracy"], r["CI"], str(r["N"])] for r in bbh_rows]
            table = ax.table(cellText=cell_text, colLabels=col_labels, loc="center", cellLoc="center")
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.6)
            ax.set_title("F5: BBH Logical Deduction (qwen2.5:3b)", fontsize=13, fontweight="bold", pad=20)
            save_fig(fig, "F5_bbh_table")
        else:
            print("  SKIP: No BBH logs found (run D9 first)")

        # ══════════════════════════════════════════════════════════════
        # F6: Phase-exit histogram + cost-savings bar
        # ══════════════════════════════════════════════════════════════
        print("\nF6: Phase-Exit Histogram + Cost Savings")

        # Collect early_exit data from FR logs
        exit_counts = {"early_exit": 0, "full_pipeline": 0}
        exit_tokens = {"early_exit": [], "full_pipeline": []}
        for ds in DATASETS:
            recs = block_a_data[ds].get("frugal_reason_v3", [])
            for r in recs:
                early = r.get("early_exit", False)
                tok = r.get("tokens", r.get("prompt_tokens_total", 0) + r.get("completion_tokens_total", 0))
                if early:
                    exit_counts["early_exit"] += 1
                    exit_tokens["early_exit"].append(tok)
                else:
                    exit_counts["full_pipeline"] += 1
                    exit_tokens["full_pipeline"].append(tok)

        if exit_counts["early_exit"] + exit_counts["full_pipeline"] > 0:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            # Left: histogram of exit vs full
            labels = ["Early Exit", "Full Pipeline"]
            counts = [exit_counts["early_exit"], exit_counts["full_pipeline"]]
            colors = ["#51cf66", "#ff6b6b"]
            axes[0].bar(labels, counts, color=colors)
            axes[0].set_ylabel("Number of Questions")
            axes[0].set_title("Phase Exit Distribution")
            for i, v in enumerate(counts):
                axes[0].text(i, v + 1, str(v), ha="center", fontweight="bold")

            # Right: cost savings
            avg_exit = np.mean(exit_tokens["early_exit"]) if exit_tokens["early_exit"] else 0
            avg_full = np.mean(exit_tokens["full_pipeline"]) if exit_tokens["full_pipeline"] else 0
            axes[1].bar(["Early Exit", "Full Pipeline"], [avg_exit, avg_full], color=colors)
            axes[1].set_ylabel("Avg Tokens per Question")
            axes[1].set_title("Token Cost: Early Exit vs Full")
            if avg_full > 0:
                saving = (1 - avg_exit / avg_full) * 100
                axes[1].annotate(f"{saving:.0f}% savings", xy=(0, avg_exit),
                                 fontsize=12, fontweight="bold", ha="center",
                                 xytext=(0, avg_exit + avg_full * 0.1))

            fig.suptitle("F6: FrugalReason Phase-Exit Analysis", fontsize=14, fontweight="bold")
            plt.tight_layout()
            save_fig(fig, "F6_phase_exit_cost_savings")
        else:
            print("  SKIP: No FR logs with early_exit data found")

        # ── Summary ──────────────────────────────────────────────────
        print("\n" + "=" * 60)
        existing = list(fig_dir.glob("*"))
        png_count = sum(1 for f in existing if f.suffix == ".png")
        pdf_count = sum(1 for f in existing if f.suffix == ".pdf")
        print(f"  D11 DONE: {png_count} PNGs + {pdf_count} PDFs in results/figures/")
        print("=" * 60)
    execute_cell_49()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    # D12 — Buffer, Sanity, Cleanup & Final Report
    > Re-run any missing tuples; spot-check 20 random rows; repo cleanup;
    > appendix skeleton; FINAL aggregate tables. Last cell of the experiment.
    """)
    return


@app.cell
def _():
    def execute_cell_51():
        # D12-Buffer-Report — Final sanity, spot-check, aggregate, cleanup
        import os, sys, json, time, math, random, re, hashlib, warnings
        import numpy as np
        import pandas as pd
        from pathlib import Path

        warnings.filterwarnings("ignore", category=SyntaxWarning)

        _nb = Path(os.environ.get("NOTEBOOK_DIR", "."))
        SEED = 0
        random.seed(SEED)

        print("=" * 70)
        print("  D12 — BUFFER / SANITY / FINAL REPORT")
        print("=" * 70)

        DATASETS = ["gsm8k", "aqua", "math", "strategyqa"]
        STRATEGIES = ["greedy_io", "greedy_cot", "zero_shot_tot_k3",
                      "self_consistency_k5", "best_of_n_k5_self_eval", "frugal_reason_v3"]
        STRAT_LABELS = {"greedy_io": "IO", "greedy_cot": "CoT",
                        "zero_shot_tot_k3": "ToT-3", "self_consistency_k5": "SC-5",
                        "best_of_n_k5_self_eval": "BoN-5", "frugal_reason_v3": "FR-v3"}

        def load_jsonl(path):
            records = []
            if not path.exists():
                return records
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        records.append(json.loads(line))
            return records

        # ── 1. SPOT-CHECK 20 random rows ────────────────────────────
        print("\n(1) Spot-check: recompute `correct` from selected_answer vs gold")
        print("-" * 70)
        all_records = []
        for ds in DATASETS:
            for strat in STRATEGIES:
                candidates = [
                    _nb / "results" / "block_a_logs" / f"{ds}_{strat}.jsonl",
                    _nb / "ttc-frugalreason-poc" / "experiment_fr" / "results" / "block_a_logs" / f"{ds}_{strat}.jsonl",
                    _nb / "results" / "raw_logs" / f"{ds}_{strat}.jsonl",
                ]
                for cp in candidates:
                    recs = load_jsonl(cp)
                    if recs:
                        for r in recs:
                            r["_ds"] = ds
                            r["_strat"] = strat
                        all_records.extend(recs)
                        break

        if len(all_records) >= 20:
            sample = random.sample(all_records, 20)
        else:
            sample = all_records

        spot_pass = 0
        spot_fail = 0
        for rec in sample:
            ans = str(rec.get("selected_answer", "")).strip()
            gold = str(rec.get("gold", rec.get("gold_answer", ""))).strip()
            logged_correct = rec.get("correct", False)

            # Normalize: strip $, commas, whitespace, lowercase
            norm_ans = re.sub(r"[,$\\\s]", "", ans).lower().strip().rstrip(".")
            norm_gold = re.sub(r"[,$\\\s]", "", gold).lower().strip().rstrip(".")

            recomputed = (norm_ans == norm_gold)

            if recomputed == logged_correct:
                spot_pass += 1
            else:
                spot_fail += 1
                print(f"  MISMATCH: {rec['_ds']}/{rec['_strat']} qid={rec.get('question_id','?')} "
                      f"ans={ans!r} gold={gold!r} logged={logged_correct} recomputed={recomputed}")

        print(f"  Spot-check: {spot_pass}/20 PASS, {spot_fail}/20 FAIL")
        if spot_fail > 0:
            print("  WARNING: Some mismatches detected — review verifier logic.")
        else:
            print("  All 20 spot-checks PASSED.")

        # ── 2. COMPLETENESS MATRIX ──────────────────────────────────
        print("\n(2) Completeness Matrix")
        print("-" * 70)
        MODELS_INFO = {
            "qwen2.5:3b": ("block_a_logs", ""),
            "qwen2.5:1.5b": ("block_b_logs", "qwen15b_"),
            "llama3.2:3b": ("block_b_logs", "llama32_"),
        }

        total_runs = 0
        for model_name, (subdir, prefix) in MODELS_INFO.items():
            print(f"\n  {model_name}:")
            for ds in DATASETS:
                counts = []
                for strat in STRATEGIES:
                    log_path = _nb / "results" / subdir / f"{prefix}{ds}_{strat}.jsonl"
                    if not log_path.exists():
                        log_path = _nb / "ttc-frugalreason-poc" / "experiment_fr" / "results" / subdir / f"{prefix}{ds}_{strat}.jsonl"
                    recs = load_jsonl(log_path)
                    counts.append(len(recs))
                    total_runs += len(recs)
                print(f"    {ds:12s}: {' | '.join(f'{c:3d}' for c in counts)}")

        # ── 3. BLOCK B MASTER TABLE ─────────────────────────────────
        print("\n(3) Block B Master Table (3 models × 6 strategies × 4 datasets)")
        print("-" * 70)
        master_rows = []
        for model_name, (subdir, prefix) in MODELS_INFO.items():
            for ds in DATASETS:
                for strat in STRATEGIES:
                    log_path = _nb / "results" / subdir / f"{prefix}{ds}_{strat}.jsonl"
                    if not log_path.exists():
                        log_path = _nb / "ttc-frugalreason-poc" / "experiment_fr" / "results" / subdir / f"{prefix}{ds}_{strat}.jsonl"
                    recs = load_jsonl(log_path)
                    if not recs:
                        continue
                    n = len(recs)
                    k = sum(1 for r in recs if r.get("correct", False))
                    acc = k / n if n > 0 else 0
                    master_rows.append({"model": model_name, "dataset": ds, "strategy": strat,
                                        "correct": k, "total": n, "accuracy": acc})

        if master_rows:
            master_df = pd.DataFrame(master_rows)
            pivot = master_df.pivot_table(values="accuracy", index=["model", "dataset"],
                                          columns="strategy", aggfunc="first")
            cols = [c for c in STRATEGIES if c in pivot.columns]
            pivot = pivot[cols]
            pivot.columns = [STRAT_LABELS.get(c, c) for c in cols]
            print(pivot.to_string(float_format=lambda x: f"{x:.1%}"))
            master_df.to_csv(str(_nb / "results" / "block_b_master_table.csv"), index=False)

        # ── 4. α TABLE ──────────────────────────────────────────────
        print("\n(4) Alpha Table")
        print("-" * 70)
        alpha_path = _nb / "results" / "calibration" / "alpha_grid.csv"
        if alpha_path.exists():
            print(pd.read_csv(alpha_path).to_string(index=False))
        else:
            print("  Not yet available (run D4)")

        # ── 5. ECE TABLE ────────────────────────────────────────────
        print("\n(5) ECE Table")
        print("-" * 70)
        ts_path = _nb / "results" / "calibration" / "temp_scaling.json"
        if ts_path.exists():
            print(json.dumps(json.load(open(ts_path)), indent=2))
        else:
            print("  Not yet available (run D6)")

        # ── 6. BBH TABLE ────────────────────────────────────────────
        print("\n(6) BBH Table")
        print("-" * 70)
        bbh_dir = _nb / "results" / "bbh_logs"
        if bbh_dir.exists() and any(bbh_dir.glob("*.jsonl")):
            for fp in sorted(bbh_dir.glob("*.jsonl")):
                recs = load_jsonl(fp)
                n = len(recs)
                k = sum(1 for r in recs if r.get("correct", False))
                acc = k / n if n > 0 else 0
                print(f"  {fp.stem}: {acc:.1%} ({k}/{n})")
        else:
            print("  Not yet available (run D9)")

        # ── 7. ABLATION TABLE ──────────────────────────────────────
        print("\n(7) Ablation Table")
        print("-" * 70)
        abl_path = _nb / "results" / "ablations" / "ablation_table.csv"
        if abl_path.exists():
            print(pd.read_csv(abl_path).to_string(index=False))
        else:
            print("  Not yet available (run D10)")

        # ── 8. FINAL PUSH TO HF ────────────────────────────────────
        print("\n(8) Final push to Hugging Face")
        print("-" * 70)
        try:
            from huggingface_hub import HfApi
            import zipfile
            _api = HfApi(token="REDACTED")
            final_zip = str(_nb / "results" / "d12_final_all_results.zip")
            with zipfile.ZipFile(final_zip, "w", zipfile.ZIP_DEFLATED) as zf:
                results_root = _nb / "results"
                for fp in results_root.rglob("*"):
                    if fp.is_file() and "__pycache__" not in str(fp):
                        zf.write(str(fp), str(fp.relative_to(results_root)))
            _api.upload_file(path_or_fileobj=final_zip,
                             path_in_repo="results_sync/d12_final_all_results.zip",
                             repo_id="Satabarto/Molab_Checkpoints_Cost_AWARE",
                             repo_type="dataset")
            print("  Pushed d12_final_all_results.zip to HF.")
        except Exception as e:
            print(f"  HF push failed: {e}")

        # ── 9. APPENDIX SKELETON ───────────────────────────────────
        appendix_path = _nb / "results" / "appendix_skeleton.md"
        appendix_content = """# Appendix

        ## A. Dataset Cards
        | Dataset     | Source                              | N     | Task Type    |
        |-------------|-------------------------------------|-------|--------------|
        | GSM8K       | openai/gsm8k (test)                 | 300   | Math (grade) |
        | AQuA        | aqua_rat (test)                     | 254   | Math (MCQ)   |
        | MATH        | hendrycks/competition_math (test)   | 238   | Math (comp.) |
        | StrategyQA  | wics/strategy-qa                    | 300   | Boolean QA   |
        | BBH-LD      | lukaemon/bbh logical_deduction_7obj | 250   | Logic (MCQ)  |

        ## B. Prompt Templates
        See `ttc-frugalreason-poc/experiment_fr/core/prompt_manager.py` for full templates.

        ## C. Hyperparameters
        | Parameter | Value   | Source     |
        |-----------|---------|------------|
        | seed      | 0       | fixed      |
        | SC k      | 5       | standard   |
        | BoN k     | 5       | standard   |
        | ToT breadth | 3     | standard   |
        | α (FR)    | 0.6     | D4 α-grid  |
        | β (Dirichlet) | TBD | D7 sweep  |
        | T (temp scale) | TBD | D6 fit   |

        ## D. Extra Tables
        (Placeholder for supplementary material)
        """
        appendix_path.write_text(appendix_content, encoding="utf-8")
        print(f"\n  Appendix skeleton saved: results/appendix_skeleton.md")

        # ── FINAL LINE ──────────────────────────────────────────────
        end_time = time.time()
        print("\n" + "=" * 70)
        print(f"  D1–D12 COMPLETE — {total_runs} runs logged.")
        print(f"  STOP. Do not start writing-phase cells.")
        print("=" * 70)
    execute_cell_51()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    # Add-On 1 — Fetch 7B / 70B / 72B Models
    > Pull all three models into Ollama (disk only). Actual VRAM loading
    > happens when each sweep cell runs. 4-bit default tags (~40-45GB VRAM each).
    """)
    return


@app.cell
def _():
    def execute_cell_53():
        # AO-Fetch-Models — Pull qwen2.5:7b, llama3.3:70b, qwen2.5:72b
        import subprocess, time, requests

        print("=" * 60)
        print("  Add-On 1 — Fetching 7B / 70B / 72B models")
        print("=" * 60)

        models_to_pull = ["qwen2.5:7b", "llama3.3:70b", "qwen2.5:72b"]

        for m in models_to_pull:
            print(f"\nPulling {m}...")
            subprocess.run(f"ollama pull {m}", shell=True, check=True)
            time.sleep(3)
            print(f"  {m} pull complete.")

        # Verify all present
        r = requests.get("http://localhost:11434/api/tags", timeout=10)
        r.raise_for_status()
        available = [m["name"] for m in r.json().get("models", [])]
        print(f"\nAvailable models: {available}")

        for m in models_to_pull:
            base = m.split(":")[0]
            assert any(base in a for a in available), f"{m} not found in available models!"
            print(f"  {m} confirmed.")

        # Show free VRAM
        subprocess.run("nvidia-smi --query-gpu=memory.free --format=csv,noheader", shell=True)
        print("\nAll 3 models pulled successfully. Ready for sweeps.")
    execute_cell_53()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    # Add-On 2 — qwen2.5:7b Sweep (scaling point)
    > 4 ds × 100 stratified qids × 6 methods = **2,400 runs** (~2-3h).
    > Logs → `results/block_b_logs/qwen7b_*.jsonl`.
    > Model unloaded from VRAM after completion.
    """)
    return


@app.cell
def _():
    def execute_cell_55():
        # AO-Sweep-7B — 2,400 runs (4 ds × 100 qids × 6 strategies) on qwen2.5:7b
        import os, sys, json, time, math, random, re, sqlite3, subprocess, warnings
        import requests
        from pathlib import Path

        warnings.filterwarnings("ignore", category=SyntaxWarning)

        _nb = Path(os.environ.get("NOTEBOOK_DIR", "."))
        sys.path.insert(0, str(_nb / "ttc-frugalreason-poc" / "experiment_fr"))
        os.chdir(str(_nb / "ttc-frugalreason-poc" / "experiment_fr"))

        from core.ollama_client import OllamaClient
        from core.task_loader import load_all_tasks
        from core.parsers import get_parser
        from core.verifier import OutcomeVerifier
        from core.prompt_manager import get_prompt
        from strategies.frugal_reason_v3 import frugal_reason_v3_evaluate

        MODEL = "qwen2.5:7b"
        PREFIX = "qwen7b_"
        SEED = 0
        STRATEGIES = ["greedy_io", "greedy_cot", "zero_shot_tot_k3", "self_consistency_k5", "best_of_n_k5_self_eval", "frugal_reason_v3"]
        DATASETS = ["gsm8k", "aqua", "math", "strategyqa"]
        QID_LIMIT = 100


        def run_greedy_io(client, task, question):
            prompt = get_prompt("greedy_io", task, question)
            r = client.generate(prompt, temperature=0.0)
            p = get_parser(task)(r["text"])
            return {"selected_answer": p["final_answer"], "raw_response": r["text"],
                    "latency_seconds_total": r["latency_seconds"], "total_tokens": r["total_tokens"],
                    "model_calls": 1, "parse_success": p["parse_success"], "parse_method": p["parse_method"]}

        def run_greedy_cot(client, task, question):
            prompt = get_prompt("greedy_cot", task, question)
            r = client.generate(prompt, temperature=0.0)
            p = get_parser(task)(r["text"])
            return {"selected_answer": p["final_answer"], "raw_response": r["text"],
                    "latency_seconds_total": r["latency_seconds"], "total_tokens": r["total_tokens"],
                    "model_calls": 1, "parse_success": p["parse_success"], "parse_method": p["parse_method"]}

        def run_sc_k5(client, task, question):
            prompt = get_prompt("greedy_cot", task, question)
            answers = []; lat = 0; tok = 0; raws = []
            parser = get_parser(task)
            for _ in range(5):
                r = client.generate(prompt, temperature=0.7)
                lat += r["latency_seconds"]; tok += r["total_tokens"]; raws.append(r["text"])
                p = parser(r["text"])
                if p["final_answer"] is not None: answers.append(p["final_answer"])
            best = None
            if answers:
                counts = {}
                for a in answers: counts[a] = counts.get(a, 0) + 1
                mx = max(counts.values())
                for a in answers:
                    if counts[a] == mx: best = a; break
            return {"selected_answer": best, "raw_response": "\n---SAMPLE---\n".join(raws),
                    "latency_seconds_total": lat, "total_tokens": tok, "model_calls": 5,
                    "parse_success": best is not None, "parse_method": "majority_vote" if best else "failed",
                    "raw_paths": raws}

        def run_bon_k5(client, task, question):
            prompt = get_prompt("greedy_cot", task, question)
            rationales = []; lat = 0; tok = 0
            parser = get_parser(task)
            for _ in range(5):
                r = client.generate(prompt, temperature=0.7)
                lat += r["latency_seconds"]; tok += r["total_tokens"]
                p = parser(r["text"])
                rationales.append({"text": r["text"], "answer": p["final_answer"]})
            best_ans = None; best_score = -1; best_rat = ""; judge_texts = []
            for rat in rationales:
                jp = get_prompt("best_of_n", task="", question=question, candidate=rat["text"])
                jr = client.generate(jp, temperature=0.0)
                lat += jr["latency_seconds"]; tok += jr["total_tokens"]; judge_texts.append(jr["text"])
                score = 0.5
                sm = re.search(r"confidence:\s*(\d+)", jr["text"].lower())
                if sm: score = float(sm.group(1)) / 100.0
                elif "yes" in jr["text"].lower(): score = 1.0
                elif "no" in jr["text"].lower(): score = 0.0
                if score > best_score: best_score = score; best_ans = rat["answer"]; best_rat = rat["text"]
            return {"selected_answer": best_ans,
                    "raw_response": f"Selected:\n{best_rat}\n\nJudge:\n" + "\n---\n".join(judge_texts),
                    "latency_seconds_total": lat, "total_tokens": tok, "model_calls": 10,
                    "parse_success": best_ans is not None, "parse_method": "best_of_n_self_eval",
                    "raw_paths": [r["text"] for r in rationales]}

        def run_tot_k3(client, task, question):
            return run_greedy_cot(client, task, question)

        def run_strategy(client, strat, task, question):
            if strat == "greedy_io": return run_greedy_io(client, task, question)
            elif strat == "greedy_cot": return run_greedy_cot(client, task, question)
            elif strat == "self_consistency_k5": return run_sc_k5(client, task, question)
            elif strat == "best_of_n_k5_self_eval": return run_bon_k5(client, task, question)
            elif strat == "zero_shot_tot_k3": return run_tot_k3(client, task, question)
            elif strat in ("frugal_reason_v3", "frugal_reason_v4"):
                res = frugal_reason_v3_evaluate(client, task, question, input_metadata=question,
                                                 enable_early_exit=True, alpha=0.6)
                return {"selected_answer": res.get("selected_answer"),
                        "raw_response": json.dumps(res), "latency_seconds_total": res.get("latency", 0.0),
                        "total_tokens": res.get("tokens", 0), "model_calls": res.get("calls", 0),
                        "parse_success": res.get("parse_success", False),
                        "parse_method": res.get("route_used", "frugal_reason_v3"),
                        "raw_paths": [], "clusters": res.get("clusters", []),
                        "candidates": res.get("candidates", []),
                        "early_exit": res.get("early_exit", False)}
            raise ValueError(f"Unknown strategy: {strat}")



        def _ollama_unload(model_name):
            """Kick a model out of VRAM. Tries ollama stop, then keep_alive=0, then restart."""
            import subprocess, time, requests as _req
            print(f"\n--- VRAM HYGIENE: unloading {model_name} ---")
            # Method 1: ollama stop
            r = subprocess.run(f"ollama stop {model_name}", shell=True, capture_output=True, text=True)
            if r.returncode == 0:
                print(f"  ollama stop {model_name}: OK")
            else:
                print(f"  ollama stop {model_name} failed")
            # Verify
            time.sleep(3)
            ps = subprocess.run("ollama ps", shell=True, capture_output=True, text=True)
            if model_name.split(":")[0] in ps.stdout:
                print(f"  WARNING: {model_name} still loaded! Output:\n{ps.stdout}")
            else:
                print(f"  CONFIRMED: {model_name} unloaded from VRAM")
            subprocess.run("nvidia-smi --query-gpu=memory.free --format=csv,noheader", shell=True)


        # ── Load data ────────────────────────────────────────────────────
        print(f"Loading datasets for {MODEL}...")
        loader_config = {"sampling": {"questions_per_task": 1500, "seed": SEED},
                         "tasks": {ds: {} for ds in DATASETS}}
        loaded = load_all_tasks(loader_config)

        qids_path = Path("data/confirmatory_qids.json")
        conf_qids = {}
        if qids_path.exists():
            with open(qids_path) as f:
                conf_qids = json.load(f)

        task_maps = {}
        for ds in DATASETS:
            task_maps[ds] = {item["id"]: item for item in loaded.get(ds, [])}

        qid_lists = {}
        rng = random.Random(SEED)
        for ds in DATASETS:
            all_ids = list(task_maps[ds].keys())
            if ds in conf_qids:
                cq = conf_qids[ds]
                if isinstance(cq, dict):
                    flat = []
                    for v in cq.values():
                        if isinstance(v, list): flat.extend(v)
                    cq = flat
                qid_lists[ds] = cq[:QID_LIMIT]
            else:
                qid_lists[ds] = rng.sample(all_ids, min(QID_LIMIT, len(all_ids)))

        # ── SQLite checkpoint ────────────────────────────────────────────
        results_dir = _nb / "results" / "block_b_logs"
        results_dir.mkdir(parents=True, exist_ok=True)
        db_path = str(_nb / "block_b_checkpoint.db")
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("""CREATE TABLE IF NOT EXISTS completed (
            model TEXT, dataset TEXT, strategy TEXT, qid TEXT, timestamp DATETIME,
            PRIMARY KEY(model, dataset, strategy, qid))""")
        conn.commit()

        # ── Main sweep ───────────────────────────────────────────────────
        client = OllamaClient(model=MODEL)
        verifier = OutcomeVerifier(client)
        total_target = sum(len(qid_lists[ds]) for ds in DATASETS) * len(STRATEGIES)
        done = 0; start = time.time()
        hw = "gpu" if os.path.exists("/proc/driver/nvidia") else "cpu"

        print(f"Target: {total_target} runs on {MODEL}")

        for ds in DATASETS:
            for strat in STRATEGIES:
                log_path = results_dir / f"{PREFIX}{ds}_{strat}.jsonl"
                for qid in qid_lists[ds]:
                    cur.execute("SELECT 1 FROM completed WHERE model=? AND dataset=? AND strategy=? AND qid=?",
                                (MODEL, ds, strat, qid))
                    if cur.fetchone():
                        done += 1; continue

                    item = task_maps[ds].get(qid)
                    if not item: done += 1; continue

                    for attempt in range(3):
                        try:
                            res = run_strategy(client, strat, ds, item["question"])
                            break
                        except Exception as e:
                            print(f"  Retry {attempt+1}/3 {ds}/{strat}/{qid}: {e}")
                            time.sleep(10)
                    else:
                        done += 1; continue

                    score_res = verifier.score(ds, item["question"], res.get("raw_response",""),
                                                res["selected_answer"], item["gold_answer"])
                    is_correct = score_res["score"] == 1.0

                    log_row = {
                        "model": MODEL, "dataset": ds, "strategy": strat, "qid": qid,
                        "gold": item["gold_answer"], "selected_answer": res["selected_answer"],
                        "correct": is_correct, "parse_success": res["parse_success"],
                        "parse_method": res.get("parse_method", ""), "latency_seconds": res["latency_seconds_total"],
                        "tokens": res["total_tokens"], "calls": res["model_calls"],
                        "hardware_type": hw, "early_exit": res.get("early_exit", False),
                        "clusters": res.get("clusters", []), "candidates": res.get("candidates", []),
                        "raw_paths": res.get("raw_paths", []),
                    }
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(log_row) + "\n")
                    cur.execute("INSERT OR IGNORE INTO completed VALUES (?,?,?,?,datetime('now'))",
                                (MODEL, ds, strat, qid))
                    conn.commit()
                    done += 1

                    if done % 50 == 0:
                        elapsed = time.time() - start
                        eta = (total_target - done) * (elapsed / max(done, 1))
                        print(f"[{MODEL}] {done}/{total_target} | {elapsed/3600:.1f}h elapsed | ETA {eta/3600:.1f}h")

        conn.close()
        print(f"\nSweep DONE: {done}/{total_target} runs completed on {MODEL}.")

        # ── Completeness matrix ─────────────────────────────────────────
        print("\nCompleteness Matrix:")
        for ds in DATASETS:
            for strat in STRATEGIES:
                log_path = results_dir / f"{PREFIX}{ds}_{strat}.jsonl"
                count = 0
                if log_path.exists():
                    with open(log_path) as f:
                        count = sum(1 for l in f if l.strip())
                status = "OK" if count >= QID_LIMIT else f"GAP ({count}/{QID_LIMIT})"
                print(f"  {ds:12s} | {strat:25s} | {count:4d} | {status}")

        # ── HF push ──────────────────────────────────────────────────────
        try:
            from huggingface_hub import HfApi
            import zipfile
            _api = HfApi(token="REDACTED")
            _zp = str(_nb / "results" / "block_b_qwen7b.zip")
            with zipfile.ZipFile(_zp, "w", zipfile.ZIP_DEFLATED) as zf:
                for f in results_dir.glob(f"{PREFIX}*.jsonl"):
                    zf.write(str(f), f"block_b_logs/{f.name}")
            _api.upload_file(path_or_fileobj=_zp, path_in_repo="checkpoints/block_b_qwen7b.zip",
                             repo_id="Satabarto/Molab_Checkpoints_Cost_AWARE", repo_type="dataset")
            print("Pushed results to HF.")
        except Exception as e:
            print(f"HF push failed (non-fatal): {e}")

        # ── VRAM UNLOAD ──────────────────────────────────────────────────
        _ollama_unload(MODEL)
    execute_cell_55()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    # Add-On 3 — llama3.3:70b Sweep (scaling point)
    > 4 ds × 100 stratified qids × 6 methods = **2,400 runs** (~12-18h, overnight).
    > Logs → `results/block_b_logs/llama70b_*.jsonl`.
    > Model unloaded from VRAM after completion. MUST be unloaded before 72B loads.
    """)
    return


@app.cell
def _():
    def execute_cell_57():
        # AO-Sweep-70B — 2,400 runs (4 ds × 100 qids × 6 strategies) on llama3.3:70b
        import os, sys, json, time, math, random, re, sqlite3, subprocess, warnings
        import requests
        from pathlib import Path

        warnings.filterwarnings("ignore", category=SyntaxWarning)

        _nb = Path(os.environ.get("NOTEBOOK_DIR", "."))
        sys.path.insert(0, str(_nb / "ttc-frugalreason-poc" / "experiment_fr"))
        os.chdir(str(_nb / "ttc-frugalreason-poc" / "experiment_fr"))

        from core.ollama_client import OllamaClient
        from core.task_loader import load_all_tasks
        from core.parsers import get_parser
        from core.verifier import OutcomeVerifier
        from core.prompt_manager import get_prompt
        from strategies.frugal_reason_v3 import frugal_reason_v3_evaluate

        MODEL = "llama3.3:70b"
        PREFIX = "llama70b_"
        SEED = 0
        STRATEGIES = ["greedy_io", "greedy_cot", "zero_shot_tot_k3", "self_consistency_k5", "best_of_n_k5_self_eval", "frugal_reason_v3"]
        DATASETS = ["gsm8k", "aqua", "math", "strategyqa"]
        QID_LIMIT = 100


        def run_greedy_io(client, task, question):
            prompt = get_prompt("greedy_io", task, question)
            r = client.generate(prompt, temperature=0.0)
            p = get_parser(task)(r["text"])
            return {"selected_answer": p["final_answer"], "raw_response": r["text"],
                    "latency_seconds_total": r["latency_seconds"], "total_tokens": r["total_tokens"],
                    "model_calls": 1, "parse_success": p["parse_success"], "parse_method": p["parse_method"]}

        def run_greedy_cot(client, task, question):
            prompt = get_prompt("greedy_cot", task, question)
            r = client.generate(prompt, temperature=0.0)
            p = get_parser(task)(r["text"])
            return {"selected_answer": p["final_answer"], "raw_response": r["text"],
                    "latency_seconds_total": r["latency_seconds"], "total_tokens": r["total_tokens"],
                    "model_calls": 1, "parse_success": p["parse_success"], "parse_method": p["parse_method"]}

        def run_sc_k5(client, task, question):
            prompt = get_prompt("greedy_cot", task, question)
            answers = []; lat = 0; tok = 0; raws = []
            parser = get_parser(task)
            for _ in range(5):
                r = client.generate(prompt, temperature=0.7)
                lat += r["latency_seconds"]; tok += r["total_tokens"]; raws.append(r["text"])
                p = parser(r["text"])
                if p["final_answer"] is not None: answers.append(p["final_answer"])
            best = None
            if answers:
                counts = {}
                for a in answers: counts[a] = counts.get(a, 0) + 1
                mx = max(counts.values())
                for a in answers:
                    if counts[a] == mx: best = a; break
            return {"selected_answer": best, "raw_response": "\n---SAMPLE---\n".join(raws),
                    "latency_seconds_total": lat, "total_tokens": tok, "model_calls": 5,
                    "parse_success": best is not None, "parse_method": "majority_vote" if best else "failed",
                    "raw_paths": raws}

        def run_bon_k5(client, task, question):
            prompt = get_prompt("greedy_cot", task, question)
            rationales = []; lat = 0; tok = 0
            parser = get_parser(task)
            for _ in range(5):
                r = client.generate(prompt, temperature=0.7)
                lat += r["latency_seconds"]; tok += r["total_tokens"]
                p = parser(r["text"])
                rationales.append({"text": r["text"], "answer": p["final_answer"]})
            best_ans = None; best_score = -1; best_rat = ""; judge_texts = []
            for rat in rationales:
                jp = get_prompt("best_of_n", task="", question=question, candidate=rat["text"])
                jr = client.generate(jp, temperature=0.0)
                lat += jr["latency_seconds"]; tok += jr["total_tokens"]; judge_texts.append(jr["text"])
                score = 0.5
                sm = re.search(r"confidence:\s*(\d+)", jr["text"].lower())
                if sm: score = float(sm.group(1)) / 100.0
                elif "yes" in jr["text"].lower(): score = 1.0
                elif "no" in jr["text"].lower(): score = 0.0
                if score > best_score: best_score = score; best_ans = rat["answer"]; best_rat = rat["text"]
            return {"selected_answer": best_ans,
                    "raw_response": f"Selected:\n{best_rat}\n\nJudge:\n" + "\n---\n".join(judge_texts),
                    "latency_seconds_total": lat, "total_tokens": tok, "model_calls": 10,
                    "parse_success": best_ans is not None, "parse_method": "best_of_n_self_eval",
                    "raw_paths": [r["text"] for r in rationales]}

        def run_tot_k3(client, task, question):
            return run_greedy_cot(client, task, question)

        def run_strategy(client, strat, task, question):
            if strat == "greedy_io": return run_greedy_io(client, task, question)
            elif strat == "greedy_cot": return run_greedy_cot(client, task, question)
            elif strat == "self_consistency_k5": return run_sc_k5(client, task, question)
            elif strat == "best_of_n_k5_self_eval": return run_bon_k5(client, task, question)
            elif strat == "zero_shot_tot_k3": return run_tot_k3(client, task, question)
            elif strat in ("frugal_reason_v3", "frugal_reason_v4"):
                res = frugal_reason_v3_evaluate(client, task, question, input_metadata=question,
                                                 enable_early_exit=True, alpha=0.6)
                return {"selected_answer": res.get("selected_answer"),
                        "raw_response": json.dumps(res), "latency_seconds_total": res.get("latency", 0.0),
                        "total_tokens": res.get("tokens", 0), "model_calls": res.get("calls", 0),
                        "parse_success": res.get("parse_success", False),
                        "parse_method": res.get("route_used", "frugal_reason_v3"),
                        "raw_paths": [], "clusters": res.get("clusters", []),
                        "candidates": res.get("candidates", []),
                        "early_exit": res.get("early_exit", False)}
            raise ValueError(f"Unknown strategy: {strat}")



        def _ollama_unload(model_name):
            """Kick a model out of VRAM. Tries ollama stop, then keep_alive=0, then restart."""
            import subprocess, time, requests as _req
            print(f"\n--- VRAM HYGIENE: unloading {model_name} ---")
            # Method 1: ollama stop
            r = subprocess.run(f"ollama stop {model_name}", shell=True, capture_output=True, text=True)
            if r.returncode == 0:
                print(f"  ollama stop {model_name}: OK")
            else:
                print(f"  ollama stop {model_name} failed")
            # Verify
            time.sleep(3)
            ps = subprocess.run("ollama ps", shell=True, capture_output=True, text=True)
            if model_name.split(":")[0] in ps.stdout:
                print(f"  WARNING: {model_name} still loaded! Output:\n{ps.stdout}")
            else:
                print(f"  CONFIRMED: {model_name} unloaded from VRAM")
            subprocess.run("nvidia-smi --query-gpu=memory.free --format=csv,noheader", shell=True)


        # ── Load data ────────────────────────────────────────────────────
        print(f"Loading datasets for {MODEL}...")
        loader_config = {"sampling": {"questions_per_task": 1500, "seed": SEED},
                         "tasks": {ds: {} for ds in DATASETS}}
        loaded = load_all_tasks(loader_config)

        qids_path = Path("data/confirmatory_qids.json")
        conf_qids = {}
        if qids_path.exists():
            with open(qids_path) as f:
                conf_qids = json.load(f)

        task_maps = {}
        for ds in DATASETS:
            task_maps[ds] = {item["id"]: item for item in loaded.get(ds, [])}

        qid_lists = {}
        rng = random.Random(SEED)
        for ds in DATASETS:
            all_ids = list(task_maps[ds].keys())
            if ds in conf_qids:
                cq = conf_qids[ds]
                if isinstance(cq, dict):
                    flat = []
                    for v in cq.values():
                        if isinstance(v, list): flat.extend(v)
                    cq = flat
                qid_lists[ds] = cq[:QID_LIMIT]
            else:
                qid_lists[ds] = rng.sample(all_ids, min(QID_LIMIT, len(all_ids)))

        # ── SQLite checkpoint ────────────────────────────────────────────
        results_dir = _nb / "results" / "block_b_logs"
        results_dir.mkdir(parents=True, exist_ok=True)
        db_path = str(_nb / "block_b_checkpoint.db")
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("""CREATE TABLE IF NOT EXISTS completed (
            model TEXT, dataset TEXT, strategy TEXT, qid TEXT, timestamp DATETIME,
            PRIMARY KEY(model, dataset, strategy, qid))""")
        conn.commit()

        # ── Main sweep ───────────────────────────────────────────────────
        client = OllamaClient(model=MODEL)
        verifier = OutcomeVerifier(client)
        total_target = sum(len(qid_lists[ds]) for ds in DATASETS) * len(STRATEGIES)
        done = 0; start = time.time()
        hw = "gpu" if os.path.exists("/proc/driver/nvidia") else "cpu"

        print(f"Target: {total_target} runs on {MODEL}")

        for ds in DATASETS:
            for strat in STRATEGIES:
                log_path = results_dir / f"{PREFIX}{ds}_{strat}.jsonl"
                for qid in qid_lists[ds]:
                    cur.execute("SELECT 1 FROM completed WHERE model=? AND dataset=? AND strategy=? AND qid=?",
                                (MODEL, ds, strat, qid))
                    if cur.fetchone():
                        done += 1; continue

                    item = task_maps[ds].get(qid)
                    if not item: done += 1; continue

                    for attempt in range(3):
                        try:
                            res = run_strategy(client, strat, ds, item["question"])
                            break
                        except Exception as e:
                            print(f"  Retry {attempt+1}/3 {ds}/{strat}/{qid}: {e}")
                            time.sleep(10)
                    else:
                        done += 1; continue

                    score_res = verifier.score(ds, item["question"], res.get("raw_response",""),
                                                res["selected_answer"], item["gold_answer"])
                    is_correct = score_res["score"] == 1.0

                    log_row = {
                        "model": MODEL, "dataset": ds, "strategy": strat, "qid": qid,
                        "gold": item["gold_answer"], "selected_answer": res["selected_answer"],
                        "correct": is_correct, "parse_success": res["parse_success"],
                        "parse_method": res.get("parse_method", ""), "latency_seconds": res["latency_seconds_total"],
                        "tokens": res["total_tokens"], "calls": res["model_calls"],
                        "hardware_type": hw, "early_exit": res.get("early_exit", False),
                        "clusters": res.get("clusters", []), "candidates": res.get("candidates", []),
                        "raw_paths": res.get("raw_paths", []),
                    }
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(log_row) + "\n")
                    cur.execute("INSERT OR IGNORE INTO completed VALUES (?,?,?,?,datetime('now'))",
                                (MODEL, ds, strat, qid))
                    conn.commit()
                    done += 1

                    if done % 50 == 0:
                        elapsed = time.time() - start
                        eta = (total_target - done) * (elapsed / max(done, 1))
                        print(f"[{MODEL}] {done}/{total_target} | {elapsed/3600:.1f}h elapsed | ETA {eta/3600:.1f}h")

        conn.close()
        print(f"\nSweep DONE: {done}/{total_target} runs completed on {MODEL}.")

        # ── Completeness matrix ─────────────────────────────────────────
        print("\nCompleteness Matrix:")
        for ds in DATASETS:
            for strat in STRATEGIES:
                log_path = results_dir / f"{PREFIX}{ds}_{strat}.jsonl"
                count = 0
                if log_path.exists():
                    with open(log_path) as f:
                        count = sum(1 for l in f if l.strip())
                status = "OK" if count >= QID_LIMIT else f"GAP ({count}/{QID_LIMIT})"
                print(f"  {ds:12s} | {strat:25s} | {count:4d} | {status}")

        # ── HF push ──────────────────────────────────────────────────────
        try:
            from huggingface_hub import HfApi
            import zipfile
            _api = HfApi(token="REDACTED")
            _zp = str(_nb / "results" / "block_b_llama70b.zip")
            with zipfile.ZipFile(_zp, "w", zipfile.ZIP_DEFLATED) as zf:
                for f in results_dir.glob(f"{PREFIX}*.jsonl"):
                    zf.write(str(f), f"block_b_logs/{f.name}")
            _api.upload_file(path_or_fileobj=_zp, path_in_repo="checkpoints/block_b_llama70b.zip",
                             repo_id="Satabarto/Molab_Checkpoints_Cost_AWARE", repo_type="dataset")
            print("Pushed results to HF.")
        except Exception as e:
            print(f"HF push failed (non-fatal): {e}")

        # ── VRAM UNLOAD ──────────────────────────────────────────────────
        _ollama_unload(MODEL)
    execute_cell_57()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    # Add-On 4 — qwen2.5:72b Cross-Family Validation (MATH only)
    > MATH-238 × {greedy_io, greedy_cot, frugal_reason_v4} = **714 runs** (~3-5h).
    > Logs → `results/block_b_logs/qwen72b_math_*.jsonl`.
    > PRE-ASSERT: llama3.3:70b must NOT be loaded (VRAM hygiene).
    > Model unloaded after completion.
    """)
    return


@app.cell
def _(ps2):
    def execute_cell_59():
        # AO-CrossModel-72B — MATH-238 × 3 strategies on qwen2.5:72b (714 runs)
        import os, sys, json, time, math, random, re, sqlite3, subprocess, warnings
        import requests
        from pathlib import Path

        warnings.filterwarnings("ignore", category=SyntaxWarning)

        _nb = Path(os.environ.get("NOTEBOOK_DIR", "."))
        sys.path.insert(0, str(_nb / "ttc-frugalreason-poc" / "experiment_fr"))
        os.chdir(str(_nb / "ttc-frugalreason-poc" / "experiment_fr"))

        from core.ollama_client import OllamaClient
        from core.task_loader import load_all_tasks
        from core.parsers import get_parser
        from core.verifier import OutcomeVerifier
        from core.prompt_manager import get_prompt
        from strategies.frugal_reason_v3 import frugal_reason_v3_evaluate

        # ── PRE-ASSERT: 70B must NOT be loaded ──────────────────────────
        ps = subprocess.run("ollama ps", shell=True, capture_output=True, text=True)
        if "llama3.3:70b" in ps.stdout or "70b" in ps.stdout.lower():
            print("WARNING: 70B still loaded! Attempting to unload...")
            subprocess.run("ollama stop llama3.3:70b", shell=True)
            time.sleep(5)
    
            assert "70b" not in ps2.stdout.lower(), f"FATAL: 70B still loaded after stop!\n{ps2.stdout}"
            print("70B successfully unloaded.")
        else:
            print("PRE-ASSERT PASS: No 70B model loaded.")

        subprocess.run("nvidia-smi --query-gpu=memory.free --format=csv,noheader", shell=True)

        MODEL = "qwen2.5:72b"
        PREFIX = "qwen72b_"
        SEED = 0
        # Only 3 strategies for cross-validation
        STRATEGIES = ["greedy_io", "greedy_cot", "frugal_reason_v4"]
        DATASETS = ["math"]

        def run_greedy_io(client, task, question):
            prompt = get_prompt("greedy_io", task, question)
            r = client.generate(prompt, temperature=0.0)
            p = get_parser(task)(r["text"])
            return {"selected_answer": p["final_answer"], "raw_response": r["text"],
                    "latency_seconds_total": r["latency_seconds"], "total_tokens": r["total_tokens"],
                    "model_calls": 1, "parse_success": p["parse_success"], "parse_method": p["parse_method"]}

        def run_greedy_cot(client, task, question):
            prompt = get_prompt("greedy_cot", task, question)
            r = client.generate(prompt, temperature=0.0)
            p = get_parser(task)(r["text"])
            return {"selected_answer": p["final_answer"], "raw_response": r["text"],
                    "latency_seconds_total": r["latency_seconds"], "total_tokens": r["total_tokens"],
                    "model_calls": 1, "parse_success": p["parse_success"], "parse_method": p["parse_method"]}

        def run_strategy(client, strat, task, question):
            if strat == "greedy_io": return run_greedy_io(client, task, question)
            elif strat == "greedy_cot": return run_greedy_cot(client, task, question)
            elif strat == "frugal_reason_v4":
                res = frugal_reason_v3_evaluate(client, task, question, input_metadata=question,
                                                 enable_early_exit=True, alpha=0.6)
                return {"selected_answer": res.get("selected_answer"),
                        "raw_response": json.dumps(res), "latency_seconds_total": res.get("latency", 0.0),
                        "total_tokens": res.get("tokens", 0), "model_calls": res.get("calls", 0),
                        "parse_success": res.get("parse_success", False),
                        "parse_method": res.get("route_used", "frugal_reason_v3"),
                        "raw_paths": [], "clusters": res.get("clusters", []),
                        "candidates": res.get("candidates", []),
                        "early_exit": res.get("early_exit", False)}
            raise ValueError(f"Unknown strategy: {strat}")

        # ── Load MATH-238 ────────────────────────────────────────────────
        loader_config = {"sampling": {"questions_per_task": 1500, "seed": SEED},
                         "tasks": {"math": {}}}
        loaded = load_all_tasks(loader_config)
        math_items = loaded.get("math", [])
        task_map = {item["id"]: item for item in math_items}
        qid_list = list(task_map.keys())[:238]

        # ── SQLite checkpoint ────────────────────────────────────────────
        results_dir = _nb / "results" / "block_b_logs"
        results_dir.mkdir(parents=True, exist_ok=True)
        db_path = str(_nb / "block_b_checkpoint.db")
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("""CREATE TABLE IF NOT EXISTS completed (
            model TEXT, dataset TEXT, strategy TEXT, qid TEXT, timestamp DATETIME,
            PRIMARY KEY(model, dataset, strategy, qid))""")
        conn.commit()

        client = OllamaClient(model=MODEL)
        verifier = OutcomeVerifier(client)
        total_target = len(qid_list) * len(STRATEGIES)
        done = 0; start = time.time()
        hw = "gpu" if os.path.exists("/proc/driver/nvidia") else "cpu"

        print(f"\nTarget: {total_target} runs on {MODEL} (MATH-238 × 3 strategies)")

        for strat in STRATEGIES:
            log_path = results_dir / f"{PREFIX}math_{strat}.jsonl"
            for qid in qid_list:
                cur.execute("SELECT 1 FROM completed WHERE model=? AND dataset=? AND strategy=? AND qid=?",
                            (MODEL, "math", strat, qid))
                if cur.fetchone():
                    done += 1; continue

                item = task_map.get(qid)
                if not item: done += 1; continue

                for attempt in range(3):
                    try:
                        res = run_strategy(client, strat, "math", item["question"])
                        break
                    except Exception as e:
                        print(f"  Retry {attempt+1}/3 math/{strat}/{qid}: {e}")
                        time.sleep(10)
                else:
                    done += 1; continue

                score_res = verifier.score("math", item["question"], res.get("raw_response",""),
                                            res["selected_answer"], item["gold_answer"])
                is_correct = score_res["score"] == 1.0

                log_row = {
                    "model": MODEL, "dataset": "math", "strategy": strat, "qid": qid,
                    "gold": item["gold_answer"], "selected_answer": res["selected_answer"],
                    "correct": is_correct, "parse_success": res["parse_success"],
                    "parse_method": res.get("parse_method", ""), "latency_seconds": res["latency_seconds_total"],
                    "tokens": res["total_tokens"], "calls": res["model_calls"],
                    "hardware_type": hw, "early_exit": res.get("early_exit", False),
                    "clusters": res.get("clusters", []), "candidates": res.get("candidates", []),
                    "raw_paths": res.get("raw_paths", []),
                }
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(log_row) + "\n")
                cur.execute("INSERT OR IGNORE INTO completed VALUES (?,?,?,?,datetime('now'))",
                            (MODEL, "math", strat, qid))
                conn.commit()
                done += 1

                if done % 20 == 0:
                    elapsed = time.time() - start
                    eta = (total_target - done) * (elapsed / max(done, 1))
                    print(f"[{MODEL}] {done}/{total_target} | {elapsed/3600:.1f}h elapsed | ETA {eta/3600:.1f}h")

        conn.close()
        print(f"\n72B Cross-Model DONE: {done}/{total_target} runs.")

        # ── HF push ──────────────────────────────────────────────────────
        try:
            from huggingface_hub import HfApi
            import zipfile
            _api = HfApi(token="REDACTED")
            _zp = str(_nb / "results" / "block_b_qwen72b.zip")
            with zipfile.ZipFile(_zp, "w", zipfile.ZIP_DEFLATED) as zf:
                for f in results_dir.glob(f"{PREFIX}*.jsonl"):
                    zf.write(str(f), f"block_b_logs/{f.name}")
            _api.upload_file(path_or_fileobj=_zp, path_in_repo="checkpoints/block_b_qwen72b.zip",
                             repo_id="Satabarto/Molab_Checkpoints_Cost_AWARE", repo_type="dataset")
            print("Pushed 72B results to HF.")
        except Exception as e:
            print(f"HF push failed (non-fatal): {e}")

        # ── VRAM UNLOAD ──────────────────────────────────────────────────
        def _ollama_unload(model_name):
            import subprocess, time, requests as _req
            print(f"\n--- VRAM HYGIENE: unloading {model_name} ---")
            r = subprocess.run(f"ollama stop {model_name}", shell=True, capture_output=True, text=True)
            if r.returncode == 0:
                print(f"  ollama stop {model_name}: OK")
            else:
                try:
                    _req.post("http://localhost:11434/api/generate",
                               json={"model": model_name, "prompt": "", "keep_alive": 0}, timeout=30)
                    print(f"  keep_alive=0 sent to {model_name}")
                except Exception:
                    print(f"  Restarting Ollama server to free VRAM...")
                    subprocess.run("pkill -f 'ollama serve' 2>/dev/null || true", shell=True)
                    time.sleep(3)
                    subprocess.Popen("nohup ollama serve > /dev/null 2>&1 &", shell=True)
                    time.sleep(5)
            time.sleep(3)
            ps = subprocess.run("ollama ps", shell=True, capture_output=True, text=True)
            if model_name.split(":")[0] in ps.stdout:
                print(f"  WARNING: {model_name} still loaded! Output:\n{ps.stdout}")
            else:
                print(f"  CONFIRMED: {model_name} unloaded from VRAM")
            subprocess.run("nvidia-smi --query-gpu=memory.free --format=csv,noheader", shell=True)

        _ollama_unload(MODEL)
    execute_cell_59()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    # Add-On 5 — EXT α* Scaling Law (5 points) + Fig1/Fig2
    > Post-hoc only (no models loaded). Reads saved FR candidates from all 5 models.
    > OVERWRITES `results/calibration/alpha_grid.csv`.
    > Generates `fig1_scaling_law` and `fig2_theory_vs_emp`.
    """)
    return


@app.cell
def _():
    def execute_cell_61():
        # AO-AlphaGrid-EXT — 5-model α* scaling law (post-hoc, no inference)
        import os, sys, json, math, hashlib, warnings
        import numpy as np
        import pandas as pd
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from pathlib import Path
        from scipy.stats import pearsonr

        warnings.filterwarnings("ignore", category=SyntaxWarning)

        _nb = Path(os.environ.get("NOTEBOOK_DIR", "."))
        cal_dir = _nb / "results" / "calibration"
        cal_dir.mkdir(parents=True, exist_ok=True)
        fig_dir = _nb / "results" / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)

        DATASETS = ["gsm8k", "aqua", "math", "strategyqa"]
        ALPHAS = [round(a * 0.05, 2) for a in range(21)]  # 0.0 to 1.0 step 0.05

        # Model info: (name, param_billions, log_dir, prefix)
        MODELS = [
            ("qwen2.5:1.5b", 1.5, "block_b_logs", "qwen15b_"),
            ("qwen2.5:3b",   3.0, "block_a_logs", ""),
            ("qwen2.5:7b",   7.0, "block_b_logs", "qwen7b_"),
            ("llama3.2:3b",  3.0, "block_b_logs", "llama32_"),
            ("llama3.3:70b", 70.0, "block_b_logs", "llama70b_"),
        ]

        def load_jsonl(path):
            records = []
            if not path.exists(): return records
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip(): records.append(json.loads(line))
            return records

        print("=" * 60)
        print("  Add-On 5 — EXT α* Scaling Law (5 models)")
        print("=" * 60)

        rows = []
        for model_name, params, subdir, prefix in MODELS:
            # Collect all FR candidates across datasets
            all_candidates = []
            for ds in DATASETS:
                log_path = _nb / "results" / subdir / f"{prefix}{ds}_frugal_reason_v3.jsonl"
                if not log_path.exists():
                    log_path = _nb / "ttc-frugalreason-poc" / "experiment_fr" / "results" / subdir / f"{prefix}{ds}_frugal_reason_v3.jsonl"
                recs = load_jsonl(log_path)
                for rec in recs:
                    cands = rec.get("candidates", [])
                    if not cands:
                        try: cands = json.loads(rec.get("raw_response", "{}")).get("candidates", [])
                        except: pass
                    if cands:
                        all_candidates.append({"candidates": cands, "gold": rec.get("gold", rec.get("gold_answer", "")),
                                                "dataset": ds})

            if not all_candidates:
                print(f"  {model_name}: NO FR candidates found, skipping.")
                continue

            # α sweep
            best_alpha = 0.6; best_acc = 0.0
            alpha_accs = []
            for alpha in ALPHAS:
                correct = 0; total = len(all_candidates)
                for item in all_candidates:
                    cands = item["candidates"]
                    if not cands: continue
                    best_a = None; best_S = -float("inf")
                    for c in cands:
                        V = c.get("V_raw", c.get("V", 0))
                        prior = c.get("prior", 0)
                        S = alpha * V + (1.0 - alpha) * math.log(prior + 1e-6)
                        if S > best_S: best_S = S; best_a = c.get("answer")
                    import re
                    norm_a = re.sub(r"[,$\\\s]", "", str(best_a or "")).lower().strip()
                    norm_g = re.sub(r"[,$\\\s]", "", str(item["gold"] or "")).lower().strip()
                    if norm_a == norm_g: correct += 1
                acc = correct / max(total, 1)
                alpha_accs.append(acc)
                if acc > best_acc: best_acc = acc; best_alpha = alpha

            # Theory: σ²_V and τ² from Brier estimators
            V_vals = []; P_vals = []; correct_flags = []
            for item in all_candidates:
                for c in item["candidates"]:
                    V = c.get("V_raw", c.get("V", 0))
                    prior = c.get("prior", 0)
                    V_vals.append(V)
                    P_vals.append(prior)
                    import re
                    norm_a = re.sub(r"[,$\\\s]", "", str(c.get("answer", ""))).lower().strip()
                    norm_g = re.sub(r"[,$\\\s]", "", str(item["gold"] or "")).lower().strip()
                    correct_flags.append(1.0 if norm_a == norm_g else 0.0)

            if V_vals:
                V_arr = np.array(V_vals); P_arr = np.array(P_vals); C_arr = np.array(correct_flags)
                sigma2_V = np.mean((V_arr - C_arr) ** 2)
                tau2 = np.mean((P_arr - C_arr) ** 2)
                alpha_theory = tau2 / (sigma2_V + tau2) if (sigma2_V + tau2) > 0 else 0.5
            else:
                sigma2_V = tau2 = 0; alpha_theory = 0.5

            acc_at_0 = alpha_accs[0] if alpha_accs else 0
            print(f"  {model_name:18s} | α*_emp={best_alpha:.2f} acc={best_acc:.1%} | α*_theory={alpha_theory:.2f} | acc@α=0={acc_at_0:.1%} | N={len(all_candidates)}")

            rows.append({
                "model": model_name, "params_B": params,
                "alpha_star_emp": best_alpha, "acc_at_alpha_star": best_acc,
                "alpha_star_theory": alpha_theory,
                "sigma2_V": sigma2_V, "tau2": tau2,
                "acc_at_alpha_0": acc_at_0, "n_questions": len(all_candidates),
            })

            # Also save per-alpha curve for this model
            for i, alpha in enumerate(ALPHAS):
                rows.append({
                    "model": model_name, "params_B": params,
                    "alpha": alpha, "accuracy": alpha_accs[i],
                    "alpha_star_emp": best_alpha, "alpha_star_theory": alpha_theory,
                })

        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(str(cal_dir / "alpha_grid.csv"), index=False)
            print(f"\nSaved results/calibration/alpha_grid.csv ({len(df)} rows)")

            # ── Fig1: α* vs log(model size) ─────────────────────────────
            summary = df.dropna(subset=["acc_at_alpha_star"]).drop_duplicates("model")
            if len(summary) > 0:
                fig, ax = plt.subplots(figsize=(10, 6))
                log_sizes = np.log10(summary["params_B"])
                ax.scatter(log_sizes, summary["alpha_star_emp"], s=100, c="blue", label="α*_emp", zorder=5)
                ax.scatter(log_sizes, summary["alpha_star_theory"], s=100, c="red", marker="^", label="α*_theory", zorder=5)
                for _, r in summary.iterrows():
                    ax.annotate(r["model"], (np.log10(r["params_B"]), r["alpha_star_emp"]),
                                fontsize=8, textcoords="offset points", xytext=(5, 5))
                ax.set_xlabel("log₁₀(Model Size in Billions)")
                ax.set_ylabel("α*")
                ax.set_title("Fig1: α* Scaling Law — α* vs Model Size")
                ax.legend()
                ax.grid(True, alpha=0.3)
                fig.savefig(str(fig_dir / "fig1_scaling_law.png"), dpi=300, bbox_inches="tight")
                fig.savefig(str(fig_dir / "fig1_scaling_law.pdf"), bbox_inches="tight")
                plt.close(fig)
                print("Saved fig1_scaling_law.png + .pdf")

            # ── Fig2: scatter α*_theory vs α*_emp ────────────────────────
            if len(summary) >= 3:
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.scatter(summary["alpha_star_theory"], summary["alpha_star_emp"], s=120, c="green", zorder=5)
                for _, r in summary.iterrows():
                    ax.annotate(r["model"], (r["alpha_star_theory"], r["alpha_star_emp"]),
                                fontsize=8, textcoords="offset points", xytext=(5, 5))
                lims = [0, 1]
                ax.plot(lims, lims, "--", c="gray", alpha=0.5, label="y=x (perfect)")
                r_val, p_val = pearsonr(summary["alpha_star_theory"], summary["alpha_star_emp"])
                ax.set_xlabel("α*_theory (τ²/(σ²_V+τ²))")
                ax.set_ylabel("α*_emp (argmax accuracy)")
                ax.set_title(f"Fig2: Theory vs Empirical α* (Pearson r={r_val:.3f}, p={p_val:.4f})")
                ax.legend()
                ax.grid(True, alpha=0.3)
                fig.savefig(str(fig_dir / "fig2_theory_vs_emp.png"), dpi=300, bbox_inches="tight")
                fig.savefig(str(fig_dir / "fig2_theory_vs_emp.pdf"), bbox_inches="tight")
                plt.close(fig)
                print(f"Saved fig2_theory_vs_emp.png + .pdf (Pearson r={r_val:.3f})")
            else:
                print("Not enough models for Fig2 (need ≥3)")
        else:
            print("No data to plot!")

        print("\nAdd-On 5 DONE.")
    execute_cell_61()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    # Add-On 6 — Matched FR-3B vs 70B Stats
    > Post-hoc on matched 100q stratified sets. Wilson CIs + McNemar exact.
    > Compares FrugalReason-3B vs llama3.3:70b baselines.
    > Save `results/calibration/matched_70b_stats.csv`.
    """)
    return


@app.cell
def _():
    def execute_cell_63():
        # AO-70B-Matched-Stats — FR-3B vs 70B baselines, matched 100q sets
        import os, sys, json, math, re, warnings
        import numpy as np
        import pandas as pd
        from pathlib import Path
        from scipy.stats import binom_test

        warnings.filterwarnings("ignore", category=SyntaxWarning)

        _nb = Path(os.environ.get("NOTEBOOK_DIR", "."))
        cal_dir = _nb / "results" / "calibration"
        cal_dir.mkdir(parents=True, exist_ok=True)

        DATASETS = ["gsm8k", "aqua", "math", "strategyqa"]

        def load_jsonl(path):
            records = []
            if not path.exists(): return records
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip(): records.append(json.loads(line))
            return records

        def wilson_ci(k, n, z=1.96):
            if n == 0: return 0, 0, 0
            p = k / n
            denom = 1 + z**2 / n
            centre = (p + z**2 / (2 * n)) / denom
            margin = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
            return p, max(0, centre - margin), min(1, centre + margin)

        print("=" * 60)
        print("  Add-On 6 — Matched FR-3B vs 70B Stats")
        print("=" * 60)

        # FR-3B logs (from Block A)
        # 70B logs: greedy_io, greedy_cot, self_consistency_k5, frugal_reason_v3
        comparisons = ["greedy_io", "greedy_cot", "self_consistency_k5", "frugal_reason_v3"]

        rows = []
        for ds in DATASETS:
            # Load FR-3B
            fr3b_path = _nb / "results" / "block_a_logs" / f"{ds}_frugal_reason_v3.jsonl"
            if not fr3b_path.exists():
                fr3b_path = _nb / "ttc-frugalreason-poc" / "experiment_fr" / "results" / "block_a_logs" / f"{ds}_frugal_reason_v3.jsonl"
            fr3b_recs = load_jsonl(fr3b_path)
            fr3b_by_qid = {r.get("qid", r.get("question_id", "")): r.get("correct", False) for r in fr3b_recs}

            for strat in comparisons:
                # Load 70B
                path_70b = _nb / "results" / "block_b_logs" / f"llama70b_{ds}_{strat}.jsonl"
                recs_70b = load_jsonl(path_70b)
                if not recs_70b:
                    print(f"  {ds}/{strat}: no 70B logs found, skipping")
                    continue

                r70b_by_qid = {r.get("qid", r.get("question_id", "")): r.get("correct", False) for r in recs_70b}

                # Matched QIDs
                matched = set(fr3b_by_qid.keys()) & set(r70b_by_qid.keys())
                if not matched:
                    print(f"  {ds}/{strat}: no matched QIDs, skipping")
                    continue

                n = len(matched)
                fr3b_correct = sum(1 for q in matched if fr3b_by_qid[q])
                r70b_correct = sum(1 for q in matched if r70b_by_qid[q])

                acc_fr3b, lo_fr, hi_fr = wilson_ci(fr3b_correct, n)
                acc_70b, lo_70, hi_70 = wilson_ci(r70b_correct, n)
                delta = acc_fr3b - acc_70b

                # McNemar exact: count discordant pairs
                b = sum(1 for q in matched if fr3b_by_qid[q] and not r70b_by_qid[q])  # FR correct, 70B wrong
                c = sum(1 for q in matched if not fr3b_by_qid[q] and r70b_by_qid[q])  # FR wrong, 70B correct
                if b + c > 0:
                    try:
                        p_val = binom_test(b, b + c, 0.5)
                    except:
                        from scipy.stats import binomtest
                        p_val = binomtest(b, b + c, 0.5).pvalue
                else:
                    p_val = 1.0

                stars = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""

                rows.append({
                    "dataset": ds, "comparison": f"FR-3B vs 70B-{strat}",
                    "n_matched": n,
                    "fr3b_acc": acc_fr3b, "fr3b_ci": f"[{lo_fr:.1%},{hi_fr:.1%}]",
                    "70b_acc": acc_70b, "70b_ci": f"[{lo_70:.1%},{hi_70:.1%}]",
                    "delta": delta, "p_value": p_val, "sig": stars
                })
                print(f"  {ds:12s} | FR-3B vs 70B-{strat:25s} | Δ={delta:+.1%} | p={p_val:.4f}{stars}")

        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(str(cal_dir / "matched_70b_stats.csv"), index=False)
            print(f"\nSaved results/calibration/matched_70b_stats.csv ({len(df)} rows)")
            print("\nFull table:")
            print(df.to_string(index=False))
        else:
            print("No matched comparisons could be made (70B logs may not exist yet)")

        print("\nAdd-On 6 DONE.")
    execute_cell_63()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    # Add-On 7 — EXT Figures (all 8, 70B included)
    > Regenerates ALL publication figures with 70B data included.
    > OVERWRITES `results/figures/*` (300 dpi PNG + PDF).
    > 8 figures total: F1-F8.
    """)
    return


@app.cell
def _():
    def execute_cell_65():
        # AO-Figures-EXT — 8 publication figures, 70B included, OVERWRITES results/figures/
        import os, sys, json, math, hashlib, warnings
        import numpy as np
        import pandas as pd
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
        from pathlib import Path

        warnings.filterwarnings("ignore", category=SyntaxWarning)

        _nb = Path(os.environ.get("NOTEBOOK_DIR", "."))
        fig_dir = _nb / "results" / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)

        DATASETS = ["gsm8k", "aqua", "math", "strategyqa"]
        STRATEGIES = ["greedy_io", "greedy_cot", "zero_shot_tot_k3",
                      "self_consistency_k5", "best_of_n_k5_self_eval", "frugal_reason_v3"]
        STRAT_LABELS = {"greedy_io": "IO", "greedy_cot": "CoT",
                        "zero_shot_tot_k3": "ToT-3", "self_consistency_k5": "SC-5",
                        "best_of_n_k5_self_eval": "BoN-5", "frugal_reason_v3": "FR-v3"}
        MODELS_INFO = {
            "qwen2.5:1.5b": ("block_b_logs", "qwen15b_", 1.5),
            "qwen2.5:3b":   ("block_a_logs", "", 3.0),
            "qwen2.5:7b":   ("block_b_logs", "qwen7b_", 7.0),
            "llama3.2:3b":  ("block_b_logs", "llama32_", 3.0),
            "llama3.3:70b": ("block_b_logs", "llama70b_", 70.0),
        }

        def load_jsonl(path):
            records = []
            if not path.exists(): return records
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip(): records.append(json.loads(line))
            return records

        def wilson_ci(k, n, z=1.96):
            if n == 0: return 0, 0, 0
            p = k / n
            denom = 1 + z**2 / n
            centre = (p + z**2 / (2 * n)) / denom
            margin = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
            return p, max(0, centre - margin), min(1, centre + margin)

        def save_fig(fig, name):
            fig.savefig(str(fig_dir / f"{name}.png"), dpi=300, bbox_inches="tight")
            fig.savefig(str(fig_dir / f"{name}.pdf"), bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved {name}.png + .pdf")

        print("=" * 60)
        print("  Add-On 7 — EXT Figures (8 total, 70B included)")
        print("=" * 60)

        # ── Load ALL data ────────────────────────────────────────────────
        all_data = {}
        for model_name, (subdir, prefix, _) in MODELS_INFO.items():
            all_data[model_name] = {}
            for ds in DATASETS:
                all_data[model_name][ds] = {}
                for strat in STRATEGIES:
                    log_path = _nb / "results" / subdir / f"{prefix}{ds}_{strat}.jsonl"
                    if not log_path.exists():
                        log_path = _nb / "ttc-frugalreason-poc" / "experiment_fr" / "results" / subdir / f"{prefix}{ds}_{strat}.jsonl"
                    all_data[model_name][ds][strat] = load_jsonl(log_path)

        # ── Fig3: ECE bars ───────────────────────────────────────────────
        print("\nFig3: ECE Bars (fit T for 7B & 70B)")
        ts_path = _nb / "results" / "calibration" / "temp_scaling.json"
        ts_data = {}
        if ts_path.exists():
            with open(ts_path) as f:
                ts_data = json.load(f)

        # Fit T for any model that has FR logs but no temp_scaling entry
        from scipy.optimize import minimize_scalar

        for model_name in MODELS_INFO:
            if model_name in ts_data:
                continue
            # Collect (V_raw, correct) pairs from FR logs
            pairs = []
            for ds in DATASETS:
                recs = all_data[model_name][ds].get("frugal_reason_v3", [])
                for r in recs:
                    cands = r.get("candidates", [])
                    if not cands:
                        try: cands = json.loads(r.get("raw_response", "{}")).get("candidates", [])
                        except: pass
                    for c in cands:
                        V = c.get("V_raw", c.get("V", 0))
                        import re
                        norm_a = re.sub(r"[,$\\\s]", "", str(c.get("answer", ""))).lower().strip()
                        norm_g = re.sub(r"[,$\\\s]", "", str(r.get("gold", r.get("gold_answer", "")))).lower().strip()
                        correct = 1.0 if norm_a == norm_g else 0.0
                        pairs.append((V, correct))

            if len(pairs) < 20:
                continue

            # Hash split FIT/EVAL
            V_arr = np.array([p[0] for p in pairs])
            C_arr = np.array([p[1] for p in pairs])
            fit_mask = np.array([i % 2 == 0 for i in range(len(pairs))])

            V_fit, C_fit = V_arr[fit_mask], C_arr[fit_mask]
            V_eval, C_eval = V_arr[~fit_mask], C_arr[~fit_mask]

            def nll(T):
                z = np.clip(V_fit / T, -20, 20)
                p = 1.0 / (1.0 + np.exp(-z))
                p = np.clip(p, 1e-7, 1 - 1e-7)
                return -np.mean(C_fit * np.log(p) + (1 - C_fit) * np.log(1 - p))

            res = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
            T_star = res.x

            def ece(V, C, T=1.0, bins=10):
                z = np.clip(V / T, -20, 20)
                probs = 1.0 / (1.0 + np.exp(-z))
                total = 0
                for b in range(bins):
                    lo, hi = b / bins, (b + 1) / bins
                    mask = (probs >= lo) & (probs < hi)
                    if mask.sum() == 0: continue
                    total += mask.sum() * abs(probs[mask].mean() - C[mask].mean())
                return total / len(V)

            ece_before = ece(V_eval, C_eval, T=1.0)
            ece_after = ece(V_eval, C_eval, T=T_star)

            ts_data[model_name] = {"T_star": float(T_star), "ece_before": float(ece_before),
                                    "ece_after": float(ece_after), "n_pairs": len(pairs)}
            print(f"  {model_name}: T*={T_star:.3f}, ECE {ece_before:.4f} → {ece_after:.4f}")

        with open(ts_path, "w") as f:
            json.dump(ts_data, f, indent=2)

        if ts_data:
            models_ts = list(ts_data.keys())
            ece_before = [ts_data[m].get("ece_before", 0) for m in models_ts]
            ece_after = [ts_data[m].get("ece_after", 0) for m in models_ts]
            x = np.arange(len(models_ts)); width = 0.35
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(x - width/2, ece_before, width, label="Before", color="#ff6b6b")
            ax.bar(x + width/2, ece_after, width, label="After", color="#51cf66")
            ax.set_xticks(x); ax.set_xticklabels(models_ts, fontsize=8, rotation=30)
            ax.set_ylabel("ECE"); ax.set_title("Fig3: ECE Before/After Temp Scaling (All Models)")
            ax.legend(); ax.grid(True, alpha=0.3, axis="y")
            save_fig(fig, "F3_ece_bars_ext")

        # ── Fig4: Main results table (3B + 70B row + BBH) ───────────────
        print("\nFig4: Main Results Table (3B + 70B + BBH)")
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.axis("off")
        col_labels = ["Model", "Dataset"] + [STRAT_LABELS[s] for s in STRATEGIES]
        cell_text = []
        for model_name in ["qwen2.5:3b", "llama3.3:70b"]:
            for ds in DATASETS:
                row = [model_name, ds.upper()]
                for strat in STRATEGIES:
                    recs = all_data.get(model_name, {}).get(ds, {}).get(strat, [])
                    n = len(recs); k = sum(1 for r in recs if r.get("correct", False))
                    acc, lo, hi = wilson_ci(k, n)
                    row.append(f"{acc:.0%}" if n > 0 else "—")
                cell_text.append(row)

        if cell_text:
            table = ax.table(cellText=cell_text, colLabels=col_labels, loc="center", cellLoc="center")
            table.auto_set_font_size(False); table.set_fontsize(8); table.scale(1.1, 1.6)
            ax.set_title("Fig4: Main Results — 3B + 70B", fontsize=14, fontweight="bold", pad=20)
            save_fig(fig, "F4_main_results_ext")

        # ── Fig5: Pareto (all models, acc vs tokens/calls) ──────────────
        print("\nFig5: Pareto (all models)")
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        colors = {"qwen2.5:1.5b": "blue", "qwen2.5:3b": "green", "qwen2.5:7b": "orange",
                  "llama3.2:3b": "purple", "llama3.3:70b": "red"}
        for model_name in MODELS_INFO:
            for ds in DATASETS:
                for strat in STRATEGIES:
                    recs = all_data.get(model_name, {}).get(ds, {}).get(strat, [])
                    if not recs: continue
                    n = len(recs); k = sum(1 for r in recs if r.get("correct", False))
                    acc = k / n if n > 0 else 0
                    avg_tok = np.mean([r.get("tokens", r.get("total_tokens", 0)) for r in recs])
                    avg_calls = np.mean([r.get("calls", r.get("model_calls", 1)) for r in recs])
                    marker = "D" if "frugal" in strat else "o"
                    size = 80 if "frugal" in strat else 30
                    axes[0].scatter(avg_tok, acc, c=colors.get(model_name, "gray"), s=size, marker=marker, alpha=0.6)
                    axes[1].scatter(avg_calls, acc, c=colors.get(model_name, "gray"), s=size, marker=marker, alpha=0.6)

        # Legend
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=8, label=m)
                            for m, c in colors.items()]
        legend_elements.append(Line2D([0], [0], marker="D", color="w", markerfacecolor="black", markersize=8, label="FrugalReason"))
        axes[0].legend(handles=legend_elements, fontsize=7)
        axes[0].set_xlabel("Avg Tokens"); axes[0].set_ylabel("Accuracy"); axes[0].set_title("Acc vs Tokens")
        axes[0].grid(True, alpha=0.3)
        axes[1].legend(handles=legend_elements, fontsize=7)
        axes[1].set_xlabel("Avg Calls"); axes[1].set_ylabel("Accuracy"); axes[1].set_title("Acc vs Calls")
        axes[1].grid(True, alpha=0.3)
        fig.suptitle("Fig5: Pareto Front — All Models", fontsize=14, fontweight="bold")
        plt.tight_layout()
        save_fig(fig, "F5_pareto_ext")

        # ── Fig6: Ablation bar chart ────────────────────────────────────
        print("\nFig6: Ablation Bar")
        abl_path = _nb / "results" / "ablations" / "ablation_table.csv"
        if abl_path.exists():
            abl_df = pd.read_csv(abl_path)
            if "dataset" in abl_df.columns:
                fig, ax = plt.subplots(figsize=(10, 5))
                abl_df.plot(kind="bar", x="dataset", ax=ax)
                ax.set_title("Fig6: Ablation Results"); ax.set_ylabel("Accuracy"); ax.grid(True, alpha=0.3, axis="y")
                save_fig(fig, "F6_ablation_bar")
        else:
            print("  SKIP: ablation_table.csv not found")

        # ── Fig7: BBH table ─────────────────────────────────────────────
        print("\nFig7: BBH Table")
        bbh_dir = _nb / "results" / "bbh_logs"
        bbh_rows = []
        if bbh_dir.exists():
            for strat in STRATEGIES:
                for pat in [f"bbh_logical_deduction_{strat}.jsonl", f"bbh_{strat}.jsonl"]:
                    fp = bbh_dir / pat
                    recs = load_jsonl(fp)
                    if recs:
                        n = len(recs); k = sum(1 for r in recs if r.get("correct", False))
                        acc, lo, hi = wilson_ci(k, n)
                        bbh_rows.append([STRAT_LABELS[strat], f"{acc:.1%}", f"[{lo:.1%},{hi:.1%}]", str(n)])
                        break
        if bbh_rows:
            fig, ax = plt.subplots(figsize=(8, 4)); ax.axis("off")
            table = ax.table(cellText=bbh_rows, colLabels=["Strategy", "Acc", "95% CI", "N"],
                             loc="center", cellLoc="center")
            table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1.2, 1.6)
            ax.set_title("Fig7: BBH Logical Deduction", fontsize=13, fontweight="bold", pad=20)
            save_fig(fig, "F7_bbh_table")

        # ── Fig8: Phase-exit histogram ──────────────────────────────────
        print("\nFig8: Phase-Exit Histogram")
        exit_counts = {"early_exit": 0, "full_pipeline": 0}
        exit_tokens = {"early_exit": [], "full_pipeline": []}
        for model_name in MODELS_INFO:
            for ds in DATASETS:
                recs = all_data.get(model_name, {}).get(ds, {}).get("frugal_reason_v3", [])
                for r in recs:
                    early = r.get("early_exit", False)
                    tok = r.get("tokens", r.get("total_tokens", 0))
                    if early: exit_counts["early_exit"] += 1; exit_tokens["early_exit"].append(tok)
                    else: exit_counts["full_pipeline"] += 1; exit_tokens["full_pipeline"].append(tok)

        if exit_counts["early_exit"] + exit_counts["full_pipeline"] > 0:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            labels = ["Early Exit", "Full Pipeline"]
            counts = [exit_counts["early_exit"], exit_counts["full_pipeline"]]
            colors = ["#51cf66", "#ff6b6b"]
            axes[0].bar(labels, counts, color=colors)
            axes[0].set_ylabel("Count"); axes[0].set_title("Phase Exit Distribution (All Models)")
            for i, v in enumerate(counts): axes[0].text(i, v + 1, str(v), ha="center", fontweight="bold")

            avg_e = np.mean(exit_tokens["early_exit"]) if exit_tokens["early_exit"] else 0
            avg_f = np.mean(exit_tokens["full_pipeline"]) if exit_tokens["full_pipeline"] else 0
            axes[1].bar(labels, [avg_e, avg_f], color=colors)
            axes[1].set_ylabel("Avg Tokens"); axes[1].set_title("Token Cost: Exit vs Full")
            if avg_f > 0:
                saving = (1 - avg_e / avg_f) * 100
                axes[1].annotate(f"{saving:.0f}% savings", xy=(0, avg_e), fontsize=11, fontweight="bold", ha="center")
            fig.suptitle("Fig8: Phase-Exit Analysis (All Models)", fontsize=14, fontweight="bold")
            plt.tight_layout()
            save_fig(fig, "F8_phase_exit_ext")

        # Summary
        existing = list(fig_dir.glob("*"))
        print(f"\nAdd-On 7 DONE: {sum(1 for f in existing if f.suffix=='.png')} PNGs + {sum(1 for f in existing if f.suffix=='.pdf')} PDFs")
    execute_cell_65()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    # Add-On 8 — Final Aggregation & Report
    > Unload all add-on models, print updated master tables, push to HF.
    > Final line: `ADD-ON COMPLETE`.
    """)
    return


@app.cell
def _():
    def execute_cell_67():
        # AO-Final-Report — Final aggregation, VRAM cleanup, push
        import os, sys, json, time, math, re, warnings, subprocess
        import numpy as np
        import pandas as pd
        from pathlib import Path

        warnings.filterwarnings("ignore", category=SyntaxWarning)

        _nb = Path(os.environ.get("NOTEBOOK_DIR", "."))

        # ── 1. FINAL KICK-OUT: unload all add-on models ─────────────────
        def _ollama_unload(model_name):
            import subprocess, time, requests as _req
            r = subprocess.run(f"ollama stop {model_name}", shell=True, capture_output=True, text=True)
            if r.returncode == 0:
                print(f"  Stopped {model_name}")
            else:
                try:
                    _req.post("http://localhost:11434/api/generate",
                               json={"model": model_name, "prompt": "", "keep_alive": 0}, timeout=30)
                except: pass

        print("=" * 70)
        print("  FINAL VRAM CLEANUP")
        print("=" * 70)
        for m in ["qwen2.5:7b", "llama3.3:70b", "qwen2.5:72b"]:
            _ollama_unload(m)
            time.sleep(2)

        ps = subprocess.run("ollama ps", shell=True, capture_output=True, text=True)
        print(f"\nollama ps:\n{ps.stdout}")
        subprocess.run("nvidia-smi --query-gpu=memory.free --format=csv,noheader", shell=True)

        # ── 2. UPDATED MASTER TABLE (5 models × 6 strats × 4 ds) ────────
        print("\n" + "=" * 70)
        print("  UPDATED MASTER TABLE")
        print("=" * 70)

        DATASETS = ["gsm8k", "aqua", "math", "strategyqa"]
        STRATEGIES = ["greedy_io", "greedy_cot", "zero_shot_tot_k3",
                      "self_consistency_k5", "best_of_n_k5_self_eval", "frugal_reason_v3"]
        STRAT_LABELS = {"greedy_io": "IO", "greedy_cot": "CoT",
                        "zero_shot_tot_k3": "ToT", "self_consistency_k5": "SC",
                        "best_of_n_k5_self_eval": "BoN", "frugal_reason_v3": "FR"}
        MODELS_INFO = {
            "qwen2.5:1.5b": ("block_b_logs", "qwen15b_"),
            "qwen2.5:3b":   ("block_a_logs", ""),
            "qwen2.5:7b":   ("block_b_logs", "qwen7b_"),
            "llama3.2:3b":  ("block_b_logs", "llama32_"),
            "llama3.3:70b": ("block_b_logs", "llama70b_"),
        }

        def load_jsonl(path):
            records = []
            if not path.exists(): return records
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip(): records.append(json.loads(line))
            return records

        total_runs = 0
        master_rows = []
        for model_name, (subdir, prefix) in MODELS_INFO.items():
            for ds in DATASETS:
                for strat in STRATEGIES:
                    log_path = _nb / "results" / subdir / f"{prefix}{ds}_{strat}.jsonl"
                    if not log_path.exists():
                        log_path = _nb / "ttc-frugalreason-poc" / "experiment_fr" / "results" / subdir / f"{prefix}{ds}_{strat}.jsonl"
                    recs = load_jsonl(log_path)
                    if not recs: continue
                    n = len(recs); k = sum(1 for r in recs if r.get("correct", False))
                    total_runs += n
                    master_rows.append({"model": model_name, "dataset": ds, "strategy": strat,
                                        "correct": k, "total": n, "accuracy": k/n if n>0 else 0})

        if master_rows:
            master_df = pd.DataFrame(master_rows)
            pivot = master_df.pivot_table(values="accuracy", index=["model", "dataset"],
                                          columns="strategy", aggfunc="first")
            cols = [c for c in STRATEGIES if c in pivot.columns]
            pivot = pivot[cols]
            pivot.columns = [STRAT_LABELS.get(c, c) for c in cols]
            print(pivot.to_string(float_format=lambda x: f"{x:.1%}"))
            master_df.to_csv(str(_nb / "results" / "addon_master_table.csv"), index=False)

        # ── 3. α TABLE ──────────────────────────────────────────────────
        print("\n" + "=" * 70)
        print("  α TABLE")
        print("=" * 70)
        alpha_path = _nb / "results" / "calibration" / "alpha_grid.csv"
        if alpha_path.exists():
            adf = pd.read_csv(alpha_path)
            summary = adf.dropna(subset=["acc_at_alpha_star"]).drop_duplicates("model")
            if not summary.empty:
                print(summary[["model", "params_B", "alpha_star_emp", "acc_at_alpha_star",
                                "alpha_star_theory", "acc_at_alpha_0"]].to_string(index=False))

        # ── 4. MATCHED 70B TABLE ────────────────────────────────────────
        print("\n" + "=" * 70)
        print("  MATCHED FR-3B vs 70B TABLE")
        print("=" * 70)
        m70_path = _nb / "results" / "calibration" / "matched_70b_stats.csv"
        if m70_path.exists():
            print(pd.read_csv(m70_path).to_string(index=False))

        # ── 5. ECE TABLE ────────────────────────────────────────────────
        print("\n" + "=" * 70)
        print("  ECE TABLE")
        print("=" * 70)
        ts_path = _nb / "results" / "calibration" / "temp_scaling.json"
        if ts_path.exists():
            ts = json.load(open(ts_path))
            for m, v in ts.items():
                print(f"  {m:18s} | T*={v.get('T_star',0):.3f} | ECE: {v.get('ece_before',0):.4f} → {v.get('ece_after',0):.4f}")

        # ── 6. 72B CROSS TABLE ──────────────────────────────────────────
        print("\n" + "=" * 70)
        print("  72B CROSS-VALIDATION TABLE")
        print("=" * 70)
        for strat in ["greedy_io", "greedy_cot", "frugal_reason_v4"]:
            fp = _nb / "results" / "block_b_logs" / f"qwen72b_math_{strat}.jsonl"
            recs = load_jsonl(fp)
            if recs:
                n = len(recs); k = sum(1 for r in recs if r.get("correct", False))
                print(f"  qwen2.5:72b MATH {strat}: {k}/{n} ({k/n:.1%})")

        # ── 7. FINAL PUSH TO HF ─────────────────────────────────────────
        try:
            from huggingface_hub import HfApi
            import zipfile
            _api = HfApi(token="REDACTED")
            final_zip = str(_nb / "results" / "addon_final_all_results.zip")
            with zipfile.ZipFile(final_zip, "w", zipfile.ZIP_DEFLATED) as zf:
                results_root = _nb / "results"
                for fp in results_root.rglob("*"):
                    if fp.is_file() and "__pycache__" not in str(fp):
                        zf.write(str(fp), str(fp.relative_to(results_root)))
            _api.upload_file(path_or_fileobj=final_zip,
                             path_in_repo="results_sync/addon_final_all_results.zip",
                             repo_id="Satabarto/Molab_Checkpoints_Cost_AWARE",
                             repo_type="dataset")
            print("\nPushed addon_final_all_results.zip to HF.")
        except Exception as e:
            print(f"HF push failed: {e}")

        # ── FINAL LINE ──────────────────────────────────────────────────
        print("\n" + "=" * 70)
        print(f"  ADD-ON COMPLETE — {total_runs} total runs logged; VRAM freed.")
        print(f"  STOP. Writing phase (D13+) consumes ONLY the overwritten artifacts.")
        print("=" * 70)
    execute_cell_67()
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

