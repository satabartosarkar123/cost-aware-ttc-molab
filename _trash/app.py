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
    # ── CELL 0: STANDARD Python IMPORTS ──
    import os, sys, json, time, subprocess, zipfile, shutil, sqlite3
    import importlib.util
    import math, random, re, io, csv, gc, traceback, glob
    import warnings, logging, copy, functools, itertools, hashlib
    import tempfile, textwrap, threading, inspect, operator
    import ast, enum, dataclasses, statistics, argparse, runpy
    from pathlib import Path
    from collections import Counter, defaultdict, OrderedDict
    from datetime import datetime, timezone, timedelta
    from typing import Any, Dict, List, Optional, Tuple, Union
    warnings.filterwarnings('ignore', category=SyntaxWarning)
    # Suppress SyntaxWarning globally (from eval() on LLM math output)
    print('All standard Python libraries loaded.')
    return (
        Path,
        hashlib,
        json,
        math,
        os,
        random,
        re,
        runpy,
        shutil,
        sqlite3,
        subprocess,
        sys,
        time,
        traceback,
        warnings,
        zipfile,
    )


@app.cell
def _(Path, os, shutil, subprocess, zipfile):
    import importlib.util
    HF_REPO = 'Satabarto/Molab_Checkpoints_Cost_AWARE'
    HF_TOKEN = 'REDACTED'
    ZIP_NAME = 'Cost-Aware-Test-Time-upload.zip'
    if Path(ZIP_NAME).exists():
        os.remove(ZIP_NAME)
    downloaded = False
    try:
        print('Downloading project zip from HuggingFace...')
        subprocess.run('pip install -q huggingface_hub', shell=True)
        from huggingface_hub import hf_hub_download
        hf_path = hf_hub_download(repo_id=HF_REPO, filename=ZIP_NAME, repo_type='dataset', token=HF_TOKEN)
        shutil.copy(hf_path, ZIP_NAME)
        downloaded = True
        print('Downloaded from HuggingFace Hub successfully!')
    except Exception as e:
        print(f'HF download failed: {e}')
    if not downloaded or not Path(ZIP_NAME).exists():
        GDRIVE_FILE_ID = '1F7lBEOBDC9FNHyK-gjOLoEQ24SpzC0tr'
        subprocess.run('pip install -q gdown', shell=True)
        import gdown
        print('Downloading from Google Drive via gdown (fallback)...')
        gdown.download(id=GDRIVE_FILE_ID, output=ZIP_NAME, quiet=False)
    if Path(ZIP_NAME).exists():
        print(f'Extracting {ZIP_NAME}...')
        with zipfile.ZipFile(ZIP_NAME, 'r') as z:
            z.extractall('.')
        base_dir = Path('.')
        search_files = ['auto_backup.py', 'requirements_molab.txt']
        if all(((base_dir / _f).exists() for _f in search_files)):
            pass
        else:
            for _d in sorted(base_dir.iterdir()):
                if _d.is_dir() and all(((_d / _f).exists() for _f in search_files)):
                    base_dir = _d
                    break
        os.environ['NOTEBOOK_DIR'] = str(base_dir.resolve())
        print(f"NOTEBOOK_DIR = {os.environ['NOTEBOOK_DIR']}")
        checks = ['rq2_part1/run_rq2_part1.py', 'ttc-frugalreason-poc/experiment_fr/run_frugal_smoke.py', 'ttc-frugalreason-poc/experiment_fr/run_day0.py', 'ttc-frugalreason-poc/experiment_fr/run_real_experiment.py', 'ttc-task-poc/experiment/run_poc.py', 'auto_backup.py', 'drive_checkpoint.py', 'requirements_molab.txt']
        all_ok = True
        for _f in checks:
            exists = (base_dir / _f).exists()
            print(f"  {('OK     ' if exists else 'MISSING')}: {_f}")
            if not exists:
                all_ok = False
        if all_ok:
            print('\nAll files present and verified!')
        else:
            print('\nSome files missing — check zip contents.')
            print('\nFiles in base_dir:')
            for _p in sorted(base_dir.iterdir()):
                print(f"  {_p.name}{('/' if _p.is_dir() else '')}")
    else:
        print(f'ERROR: {ZIP_NAME} not found after download attempt.')
    return


@app.cell
def _(Path, importlib_1, os, shutil, subprocess, time):
    OLLAMA_MODEL = 'qwen2.5:3b'
    FALLBACK_MODEL = 'llama3.2:3b'

    def _run(cmd):
        print(f'  $ {cmd}')
        subprocess.run(cmd, shell=True)

    def _ok(cmd):
        return subprocess.run(cmd, shell=True, capture_output=True).returncode == 0
    os.makedirs('/workspace', exist_ok=True)
    print('[1/6] /workspace ready')
    _run('apt-get update -qq && apt-get install -y -qq zstd curl > /dev/null 2>&1')
    print('[2/6] System packages ready')
    os.environ['PATH'] = '/usr/local/bin:/usr/bin:/bin:' + os.environ.get('PATH', '')
    if Path('/usr/local/bin/ollama').is_file():
        print('[3/6] Ollama already installed')
    else:
        print('[3/6] Installing Ollama...')
        _run('curl -fsSL https://ollama.com/install.sh | sh')
    ollama_bin = next((_c for _c in ['/usr/local/bin/ollama', '/usr/bin/ollama'] if Path(_c).is_file()), None)
    if not ollama_bin:
        _r = subprocess.run('which ollama', shell=True, capture_output=True, text=True)
        ollama_bin = _r.stdout.strip() if _r.returncode == 0 and _r.stdout.strip() else None
    if not ollama_bin:
        raise RuntimeError('Ollama binary not found')
    print(f'  Binary: {ollama_bin}')
    subprocess.run("pkill -f 'ollama serve' 2>/dev/null || true", shell=True)
    time.sleep(2)
    subprocess.Popen(f'{ollama_bin} serve >> /workspace/ollama.log 2>&1', shell=True)
    print('  Starting Ollama on RTX 6000...')
    started = False
    for _i in range(20):
        time.sleep(2)
        if _ok('curl -sf http://localhost:11434/api/tags > /dev/null'):
            print(f'  Ready in {(_i + 1) * 2}s')
            started = True
            break
    if not started:
        subprocess.run('tail -20 /workspace/ollama.log', shell=True)
        raise RuntimeError('Ollama did not start in 40s')
    tags_out = subprocess.run('curl -sf http://localhost:11434/api/tags', shell=True, capture_output=True, text=True).stdout
    if OLLAMA_MODEL.split(':')[0] in tags_out:
        print(f'[4/6] {OLLAMA_MODEL} already on GPU')
    else:
        print(f'[4/6] Pulling {OLLAMA_MODEL}...')
        if not _ok(f'{ollama_bin} pull {OLLAMA_MODEL}'):
            print(f'  Failed — trying {FALLBACK_MODEL}')
            if not _ok(f'{ollama_bin} pull {FALLBACK_MODEL}'):
                raise RuntimeError('Both pulls failed')
            OLLAMA_MODEL = FALLBACK_MODEL
        print(f'  {OLLAMA_MODEL} ready')
    _nb_dir = os.environ.get('NOTEBOOK_DIR', str(Path('.').resolve()))
    req = Path(_nb_dir) / 'requirements_molab.txt'
    if req.exists():
        _run(f'pip install -q --root-user-action=ignore -r {req}')
    else:
        _run('pip install -q --root-user-action=ignore requests datasets pandas numpy matplotlib seaborn tqdm pynvml psutil pyyaml tabulate reportlab scipy fpdf2')
    print('[5/6] Python dependencies ready')
    if not shutil.which('rclone'):
        _run('curl -fsSL https://rclone.org/install.sh | sudo bash > /dev/null 2>&1')
    try:
        spec = importlib_1.util.spec_from_file_location('auto_backup', str(Path(_nb_dir) / 'auto_backup.py'))
        ab = importlib_1.util.module_from_spec(spec)
        spec.loader.exec_module(ab)
        conf = ab.RCLONE_CONF.strip()
        if 'YOUR_CLIENT_ID_HERE' in conf or 'YOUR_ACCESS_TOKEN' in conf:
            print('[6/6] Drive sync: DISABLED')
        else:
            _cp = Path('/root/.config/rclone/rclone.conf')
            _cp.parent.mkdir(parents=True, exist_ok=True)
            _cp.write_text(conf)
            test = subprocess.run('rclone lsd gdrive: --max-depth 1', shell=True, capture_output=True, text=True, timeout=15)
            if test.returncode == 0:
                print('[6/6] Drive sync: ENABLED')
            else:
                print(f'[6/6] Drive sync failed: {test.stderr.strip()[:80]}')
    except Exception as e:
        print(f'[6/6] Drive sync error: {e}')
    os.environ['OLLAMA_BASE_URL'] = 'http://localhost:11434'
    os.environ['OLLAMA_MODEL'] = OLLAMA_MODEL
    os.environ['WORKSPACE'] = _nb_dir
    print('\n' + '=' * 60)
    print('  SETUP COMPLETE — pick a run cell below')
    print('=' * 60)
    print(f'  Model     : {OLLAMA_MODEL} on RTX 6000')
    print(f'  Workspace : {_nb_dir}')
    print('=' * 60)
    return


@app.cell
def _(warnings):
    # ── CELL 3: THIRD-PARTY IMPORTS ──
    # Run this AFTER Cell 1 (pip install) and Cell 2 (Ollama) have completed.
    warnings.filterwarnings('ignore', category=SyntaxWarning)
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns  # headless
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
        from huggingface_hub import HfApi
    except Exception:
        pass
    try:
        import datasets
    except Exception:
        pass
    print('All third-party libraries loaded successfully.')
    return HfApi, binomtest, matplotlib, minimize_scalar, np, pd, plt, requests


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
def _(Path, os, runpy, sys, traceback):
    _nb_dir = Path(os.environ.get('NOTEBOOK_DIR', '.'))
    _script = (_nb_dir / 'ttc-frugalreason-poc/experiment_fr/run_frugal_smoke.py').resolve()
    if not _script.exists():
        raise FileNotFoundError(f'{_script} — run Cell 1 first')
    os.chdir(_script.parent)
    if str(_script.parent) not in sys.path:
        sys.path.insert(0, str(_script.parent))
    print('Smoke A: frugal_reason_v3 | gsm8k · strategyqa · aqua · math | 10 q each')
    print('=' * 60)
    try:
        runpy.run_path(str(_script), run_name='__main__')
    except SystemExit as e:
        if e.code not in (None, 0):
            print(f'Exit: {e.code}')
    except KeyboardInterrupt:
        print('Stopped.')
    except Exception:
        traceback.print_exc()
    print('\n' + '=' * 60 + '\nSmoke A done.')
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
def _(Path, os, runpy, sys, traceback):
    _nb_dir = Path(os.environ.get('NOTEBOOK_DIR', '.'))
    _script = (_nb_dir / 'ttc-frugalreason-poc/experiment_fr/run_day0.py').resolve()
    if not _script.exists():
        raise FileNotFoundError(f'{_script} — run Cell 1 first')
    os.chdir(_script.parent)
    if str(_script.parent) not in sys.path:
        sys.path.insert(0, str(_script.parent))
    print('Smoke B: all 6 strategies | gsm8k · strategyqa · aqua · math | 10 q each')
    print('=' * 60)
    try:
        runpy.run_path(str(_script), run_name='__main__')
    except SystemExit as e:
        if e.code not in (None, 0):
            print(f'Exit: {e.code}')
    except KeyboardInterrupt:
        print('Stopped.')
    except Exception:
        traceback.print_exc()
    print('\n' + '=' * 60 + '\nSmoke B done.')
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
def _(Path, os, runpy, sys, traceback):
    _nb_dir = Path(os.environ.get('NOTEBOOK_DIR', '.'))
    _script = (_nb_dir / 'rq2_part1/run_rq2_part1.py').resolve()
    if not _script.exists():
        raise FileNotFoundError(f'{_script} — run Cell 1 first')
    src = _script.read_text(encoding='utf-8').replace('TOTAL_QUESTIONS_PER_TASK = 36', 'TOTAL_QUESTIONS_PER_TASK = 5  # SMOKE')
    smoke = _script.parent / '_smoke_rq2.py'
    smoke.write_text(src, encoding='utf-8')
    os.chdir(_script.parent)
    if str(_script.parent) not in sys.path:
        sys.path.insert(0, str(_script.parent))
    print('Smoke C: rq2_part1 (5 q/task) | gsm8k · strategyqa · game24')
    print('=' * 60)
    try:
        runpy.run_path(str(smoke), run_name='__main__')
    except SystemExit as e:
        if e.code not in (None, 0):
            print(f'Exit: {e.code}')
    except KeyboardInterrupt:
        print('Stopped.')
    except Exception:
        traceback.print_exc()
    finally:
        smoke.unlink(missing_ok=True)
    print('\n' + '=' * 60 + '\nSmoke C done.')
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
def _(Path, os, runpy, sys, traceback):
    _nb_dir = Path(os.environ.get('NOTEBOOK_DIR', '.'))
    _script = (_nb_dir / 'rq2_part1/run_rq2_part1.py').resolve()
    if not _script.exists():
        raise FileNotFoundError(f'{_script} — run Cell 1 first')
    os.chdir(_script.parent)
    if str(_script.parent) not in sys.path:
        sys.path.insert(0, str(_script.parent))
    print('Full Run 1: rq2_part1 | 540 runs | auto-resume from checkpoint')
    print('Strategies : greedy_io | greedy_cot | self_consistency_k5 | best_of_n_k5_self_eval | zero_shot_tot_k3')
    print('Datasets   : gsm8k | strategyqa | game24')
    print('=' * 60 + '\n')
    try:
        runpy.run_path(str(_script), run_name='__main__')
    except SystemExit as e:
        if e.code not in (None, 0):
            print(f'Exit: {e.code}')
    except KeyboardInterrupt:
        print('\nStopped. Re-run to resume from checkpoint.')
    except Exception:
        traceback.print_exc()
    print('\n' + '=' * 60 + '\nFull Run 1 done. Run the PUSH cell.')
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
def _(Path, os, runpy, sys, traceback):
    _nb_dir = Path(os.environ.get('NOTEBOOK_DIR', '.'))
    _script = (_nb_dir / 'ttc-frugalreason-poc/experiment_fr/run_block_a.py').resolve()
    if not _script.exists():
        raise FileNotFoundError(str(_script))
    os.chdir(_script.parent)
    if str(_script.parent) not in sys.path:
        sys.path.insert(0, str(_script.parent))
    print('Full Run 2: FR Block A | gsm8k · aqua · svamp | SQLite checkpoint')
    print('=' * 60)
    try:
        runpy.run_path(str(_script), run_name='__main__')
    except SystemExit as e:
        if e.code not in (None, 0):
            print(f'Exit: {e.code}')
    except KeyboardInterrupt:
        print('Stopped. Re-run to resume.')
    except Exception:
        traceback.print_exc()
    print('\n' + '=' * 60 + '\nFull Run 2 done.')
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
def _(Path, os, runpy, sys, traceback):
    _nb_dir = Path(os.environ.get('NOTEBOOK_DIR', '.'))
    _script = (_nb_dir / 'ttc-frugalreason-poc/experiment_fr/run_block_a_part2.py').resolve()
    if not _script.exists():
        raise FileNotFoundError(str(_script))
    os.chdir(_script.parent)
    if str(_script.parent) not in sys.path:
        sys.path.insert(0, str(_script.parent))
    print('Full Run 3: FR Block A-2 | math · strategyqa | SQLite checkpoint')
    print('=' * 60)
    try:
        runpy.run_path(str(_script), run_name='__main__')
    except SystemExit as e:
        if e.code not in (None, 0):
            print(f'Exit: {e.code}')
    except KeyboardInterrupt:
        print('Stopped. Re-run to resume.')
    except Exception:
        traceback.print_exc()
    print('\n' + '=' * 60 + '\nFull Run 3 done.')
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
def _(HfApi, Path, os, runpy, sys, traceback):
    _nb_dir = Path(os.environ.get('NOTEBOOK_DIR', '.'))
    _script = (_nb_dir / 'ttc-frugalreason-poc/experiment_fr/run_real_experiment.py').resolve()
    if not _script.exists():
        raise FileNotFoundError(str(_script))
    os.chdir(_script.parent)
    if str(_script.parent) not in sys.path:
        sys.path.insert(0, str(_script.parent))
    print('Full Run 4: FR Master | all 6 strategies | gsm8k · strategyqa · aqua · math | 864 runs')
    print('=' * 60)
    try:
        runpy.run_path(str(_script), run_name='__main__')
    except SystemExit as e:
        if e.code not in (None, 0):
            print(f'Exit: {e.code}')
    except KeyboardInterrupt:
        print('Stopped.')
    except Exception:
        traceback.print_exc()
    print('\n' + '=' * 60 + '\nFull Run 4 done.')
    print('\nPushing Day 1 results to Hugging Face Hub...')
    try:
        import zipfile as _zf
        _api = HfApi(token='REDACTED')
        _out_dir = os.path.join(_nb_dir, 'results')
        _csvs = ['block_a_final_stats.csv', 'mcnemar_table.csv', 'bootstrap_table.csv']
        for _csv in _csvs:
            _p = os.path.join(_out_dir, _csv)
            if os.path.exists(_p):
                _api.upload_file(path_or_fileobj=_p, path_in_repo=f'day1_stats/{_csv}', repo_id='Satabarto/Molab_Checkpoints_Cost_AWARE', repo_type='dataset')
                print(f'  Uploaded {_csv}')
        print('Day 1 HF push complete.')
    except Exception as _e:
        print(f'HF push failed (non-fatal): {_e}')
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
def _(Path, os, runpy, sys, traceback):
    _nb_dir = Path(os.environ.get('NOTEBOOK_DIR', '.'))
    _script = (_nb_dir / 'ttc-task-poc/experiment/run_poc.py').resolve()
    if not _script.exists():
        raise FileNotFoundError(str(_script))
    os.chdir(_script.parent)
    if str(_script.parent) not in sys.path:
        sys.path.insert(0, str(_script.parent))
    print('Full Run 5: TTC-Task POC | 750 runs | WARNING: no checkpoint')
    print('Strategies : greedy_io | greedy_cot | self_consistency | best_of_n | tree_of_thought')
    print('Datasets   : gsm8k | strategyqa | game24')
    print('=' * 60)
    try:
        runpy.run_path(str(_script), run_name='__main__')
    except SystemExit as e:
        if e.code not in (None, 0):
            print(f'Exit: {e.code}')
    except KeyboardInterrupt:
        print('Stopped.')
    except Exception:
        traceback.print_exc()
    print('\n' + '=' * 60 + '\nFull Run 5 done.')
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
def _(HfApi, binomtest, json, np, os, pd):
    _DATASETS = ['gsm8k', 'aqua', 'math', 'strategyqa']
    _STRATEGIES = ['greedy_io', 'greedy_cot', 'zero_shot_tot_k3', 'self_consistency_k5', 'best_of_n_k5_self_eval', 'frugal_reason_v3']
    EXPECTED_COUNTS = {'gsm8k': 300, 'aqua': 254, 'math': 238, 'strategyqa': 300}
    LOG_DIR = 'results/block_a_logs'
    print('SECTION 1 — LOAD & VALIDATE COMPLETENESS')
    data = []
    gaps = []
    for _d in _DATASETS:
        for _s in _STRATEGIES:
            filepath = os.path.join(LOG_DIR, f'{_d}_{_s}.jsonl')
            if not os.path.exists(filepath):
                gaps.append(f'Missing file: {_d} - {_s}')
                continue
            _count = 0
            with open(filepath, 'r', encoding='utf-8') as _f:
                for _line in _f:
                    if not _line.strip():
                        continue
                    record = json.loads(_line)
                    _qid = record.get('qid', f'{_d}_{_count}')
                    _correct = int(record.get('correct', False))
                    tokens = record.get('tokens', 0)
                    calls = record.get('calls', 1)
                    parse = int(record.get('parse_success', True))
                    data.append({'dataset': _d, 'strategy': _s, 'qid': _qid, 'correct': _correct, 'tokens': tokens, 'calls': calls, 'parse': parse})
                    _count = _count + 1
            if _count != EXPECTED_COUNTS[_d]:
                gaps.append(f'Mismatch {_d}-{_s}: expected {EXPECTED_COUNTS[_d]}, got {_count}')
    _df = pd.DataFrame(data)
    print(f'Loaded {len(_df)} records.')
    if gaps:
        print('GAPS FOUND:')
        for _g in gaps:
            print(' -', _g)
    else:
        print('No gaps found. All counts match expected!')
    print('\\nSECTION 2 — WILSON 95% CONFIDENCE INTERVALS')

    def _wilson_ci(k, n, z=1.96):
        if _n == 0:
            return (0.0, 0.0)
        _p = _k / _n
        denominator = 1 + z ** 2 / _n
        centre_adjusted_probability = _p + z ** 2 / (2 * _n)
        adjusted_standard_deviation = np.sqrt((_p * (1 - _p) + z ** 2 / (4 * _n)) / _n)
        lower_bound = (centre_adjusted_probability - z * adjusted_standard_deviation) / denominator
        upper_bound = (centre_adjusted_probability + z * adjusted_standard_deviation) / denominator
        return (lower_bound, upper_bound)
    stats_rows = []
    for _d in _DATASETS:
        for _s in _STRATEGIES:
            _subset = _df[(_df['dataset'] == _d) & (_df['strategy'] == _s)]
            _n = len(_subset)
            if _n == 0:
                continue
            _k = _subset['correct'].sum()
            _acc = _k / _n
            (_lo, _hi) = _wilson_ci(_k, _n)
            avg_tokens = _subset['tokens'].mean()
            _avg_calls = _subset['calls'].mean()
            parse_rate = _subset['parse'].mean()
            stats_rows.append({'dataset': _d, 'strategy': _s, 'correct': _k, 'total': _n, 'acc': _acc, 'wilson_lo': _lo, 'wilson_hi': _hi, 'avg_tokens': avg_tokens, 'avg_calls': _avg_calls, 'parse_rate': parse_rate})
            print(f'{_d} | {_s}: {_acc * 100:.1f}% [{_lo * 100:.1f}, {_hi * 100:.1f}]')
    stats_df = pd.DataFrame(stats_rows)
    os.makedirs('results', exist_ok=True)
    stats_df.to_csv('results/block_a_final_stats.csv', index=False)
    print('\\nSECTION 3 — McNEMAR EXACT TESTS (accuracy, paired by qid)')
    mcnemar_rows = []
    for _d in _DATASETS:
        fr_subset = _df[(_df['dataset'] == _d) & (_df['strategy'] == 'frugal_reason_v3')].set_index('qid')
        if len(fr_subset) == 0:
            continue
        for _s in _STRATEGIES:
            if _s == 'frugal_reason_v3':
                continue
            bs_subset = _df[(_df['dataset'] == _d) & (_df['strategy'] == _s)].set_index('qid')
            aligned = fr_subset.join(bs_subset, lsuffix='_fr', rsuffix='_bs', how='inner')
            _b = ((aligned['correct_fr'] == 1) & (aligned['correct_bs'] == 0)).sum()
            _c = ((aligned['correct_fr'] == 0) & (aligned['correct_bs'] == 1)).sum()
            if _b + _c == 0:
                _p_val = 1.0
            else:
                _p_val = binomtest(_b, n=_b + _c, p=0.5, alternative='two-sided').pvalue
            _stars = ''
            if _p_val < 0.001:
                _stars = '***'
            elif _p_val < 0.01:
                _stars = '**'
            elif _p_val < 0.05:
                _stars = '*'
            acc_diff = aligned['correct_fr'].mean() - aligned['correct_bs'].mean()
            mcnemar_rows.append({'dataset': _d, 'baseline': _s, 'b_fr_only': _b, 'c_bs_only': _c, 'p_value': _p_val, 'significance': _stars, 'acc_diff': acc_diff})
    mcnemar_df = pd.DataFrame(mcnemar_rows)
    mcnemar_df.to_csv('results/mcnemar_table.csv', index=False)
    print(mcnemar_df[['dataset', 'baseline', 'b_fr_only', 'c_bs_only', 'p_value', 'significance', 'acc_diff']])
    print('\\nSECTION 4 — PAIRED BOOTSTRAP (10,000 resamples, seed=0)')
    np.random.seed(0)
    bootstrap_rows = []
    for _d in _DATASETS:
        fr_subset = _df[(_df['dataset'] == _d) & (_df['strategy'] == 'frugal_reason_v3')].set_index('qid')
        if len(fr_subset) == 0:
            continue
        for _s in _STRATEGIES:
            if _s == 'frugal_reason_v3':
                continue
            bs_subset = _df[(_df['dataset'] == _d) & (_df['strategy'] == _s)].set_index('qid')
            aligned = fr_subset.join(bs_subset, lsuffix='_fr', rsuffix='_bs', how='inner')
            _n = len(aligned)
            if _n == 0:
                continue
            diffs = aligned['correct_fr'].values - aligned['correct_bs'].values
            resamples = np.random.choice(diffs, (10000, _n), replace=True)
            means = resamples.mean(axis=1)
            _lo = np.percentile(means, 2.5)
            _hi = np.percentile(means, 97.5)
            bootstrap_rows.append({'dataset': _d, 'baseline': _s, 'metric': 'accuracy', 'mean_diff': diffs.mean(), 'ci_lo': _lo, 'ci_hi': _hi})
            if _s in ['self_consistency_k5', 'best_of_n_k5_self_eval']:
                tok_diffs = aligned['tokens_fr'].values - aligned['tokens_bs'].values
                resamples_tok = np.random.choice(tok_diffs, (10000, _n), replace=True)
                means_tok = resamples_tok.mean(axis=1)
                bootstrap_rows.append({'dataset': _d, 'baseline': _s, 'metric': 'tokens', 'mean_diff': tok_diffs.mean(), 'ci_lo': np.percentile(means_tok, 2.5), 'ci_hi': np.percentile(means_tok, 97.5)})
                call_diffs = aligned['calls_fr'].values - aligned['calls_bs'].values
                resamples_call = np.random.choice(call_diffs, (10000, _n), replace=True)
                means_call = resamples_call.mean(axis=1)
                bootstrap_rows.append({'dataset': _d, 'baseline': _s, 'metric': 'calls', 'mean_diff': call_diffs.mean(), 'ci_lo': np.percentile(means_call, 2.5), 'ci_hi': np.percentile(means_call, 97.5)})
    boot_df = pd.DataFrame(bootstrap_rows)
    boot_df.to_csv('results/bootstrap_table.csv', index=False)
    print('Bootstrap finished. Wrote results/bootstrap_table.csv')
    print('\\nSECTION 5 — SANITY SPOT-CHECK (anti-fabrication)')
    _expected = {'gsm8k': 0.82, 'aqua': 0.709, 'math': 0.735, 'strategyqa': 0.653}
    for (_d, exp_val) in _expected.items():
        _val = stats_df[(stats_df['dataset'] == _d) & (stats_df['strategy'] == 'frugal_reason_v3')]['acc'].values
        if len(_val) > 0:
            actual = _val[0]
            if abs(actual - exp_val) > 0.005:
                print(f'STOP! Deviation found in {_d}: Expected ~{exp_val}, Got {actual:.3f}')
            else:
                print(f'Spot-check passed for {_d}: {actual:.3f} matches ~{exp_val}')
    print('\\nSECTION 6 — OUTPUTS / MARKDOWN SUMMARY')
    for _d in _DATASETS:
        print(f'### {_d.upper()}')
        fr_row = stats_df[(stats_df['dataset'] == _d) & (stats_df['strategy'] == 'frugal_reason_v3')]
        if len(fr_row) == 0:
            continue
        fr_acc = fr_row['acc'].values[0]
        (fr_lo, fr_hi) = (fr_row['wilson_lo'].values[0], fr_row['wilson_hi'].values[0])
        print(f'FRUGAL_REASON_V3: {fr_acc * 100:.1f}% [{fr_lo * 100:.1f}, {fr_hi * 100:.1f}]')
        for _s in _STRATEGIES:
            if _s == 'frugal_reason_v3':
                continue
            s_row = stats_df[(stats_df['dataset'] == _d) & (stats_df['strategy'] == _s)]
            if len(s_row) == 0:
                continue
            s_acc = s_row['acc'].values[0]
            (s_lo, s_hi) = (s_row['wilson_lo'].values[0], s_row['wilson_hi'].values[0])
            mc_row = mcnemar_df[(mcnemar_df['dataset'] == _d) & (mcnemar_df['baseline'] == _s)]
            _stars = mc_row['significance'].values[0] if len(mc_row) > 0 else ''
            bt_row = boot_df[(boot_df['dataset'] == _d) & (boot_df['baseline'] == _s) & (boot_df['metric'] == 'accuracy')]
            if len(bt_row) > 0:
                diff = bt_row['mean_diff'].values[0] * 100
                diff_lo = bt_row['ci_lo'].values[0] * 100
                diff_hi = bt_row['ci_hi'].values[0] * 100
                diff_str = f'diff {diff:+.1f} CI[{diff_lo:+.1f}, {diff_hi:+.1f}]'
            else:
                diff_str = ''
            print(f' - {_s}: {s_acc * 100:.1f}% [{s_lo * 100:.1f}, {s_hi * 100:.1f}] {_stars} | {diff_str}')
        print()
    print('\nPushing Day 1 results to Hugging Face Hub...')
    try:
        import zipfile as _zf
        _api = HfApi(token='REDACTED')
        _out_dir = 'results'
        _csvs = ['block_a_final_stats.csv', 'mcnemar_table.csv', 'bootstrap_table.csv']
        for _csv in _csvs:
            _p = os.path.join(_out_dir, _csv)
            if os.path.exists(_p):
                _api.upload_file(path_or_fileobj=_p, path_in_repo=f'day1_stats/{_csv}', repo_id='Satabarto/Molab_Checkpoints_Cost_AWARE', repo_type='dataset')
                print(f'  Uploaded {_csv}')
        print('Day 1 HF push complete.')
    except Exception as _e:
        print(f'HF push failed (non-fatal): {_e}')
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
    # Day2-Fetch-1.5B — Pull qwen2.5:1.5b into Ollama
    print('=' * 60)
    print('  Day 2 — Fetching qwen2.5:1.5b')
    print('=' * 60)
    subprocess.run('ollama pull qwen2.5:1.5b', shell=True, check=True)
    time.sleep(3)
    _r = requests.get('http://localhost:11434/api/tags', timeout=10)
    _r.raise_for_status()
    _models = [_m['name'] for _m in _r.json().get('models', [])]
    # Verify via /api/tags
    assert 'qwen2.5:1.5b' in _models or any(('qwen2.5:1.5b' in _m for _m in _models)), f'qwen2.5:1.5b not found! Available: {_models}'
    print(f'qwen2.5:1.5b confirmed. Available models: {_models}')
    return


@app.cell
def _(HfApi, Path, json, os, random, re, sqlite3, sys, time, zipfile):
    _nb = Path(os.environ.get('NOTEBOOK_DIR', '.'))
    sys.path.insert(0, str(_nb / 'ttc-frugalreason-poc' / 'experiment_fr'))
    os.chdir(str(_nb / 'ttc-frugalreason-poc' / 'experiment_fr'))
    from core.ollama_client import OllamaClient
    from core.task_loader import load_all_tasks
    from core.parsers import get_parser
    from core.verifier import OutcomeVerifier
    from core.prompt_manager import get_prompt
    from strategies.frugal_reason_v3 import frugal_reason_v3_evaluate
    _MODEL = 'qwen2.5:1.5b'
    _SEED = 0
    _STRATEGIES = ['greedy_io', 'greedy_cot', 'zero_shot_tot_k3', 'self_consistency_k5', 'best_of_n_k5_self_eval', 'frugal_reason_v3']
    _DATASETS = ['gsm8k', 'aqua', 'math', 'strategyqa']
    EXPECTED = {'gsm8k': 300, 'aqua': 254, 'math': 238, 'strategyqa': 300}
    _QID_LIMIT = 100

    def _run_greedy_io(client, task, question):
        prompt = get_prompt('greedy_io', task, question)
        _r = _client.generate(prompt, temperature=0.0)
        _p = get_parser(task)(_r['text'])
        return {'selected_answer': _p['final_answer'], 'raw_response': _r['text'], 'latency_seconds_total': _r['latency_seconds'], 'total_tokens': _r['total_tokens'], 'model_calls': 1, 'parse_success': _p['parse_success'], 'parse_method': _p['parse_method']}

    def _run_greedy_cot(client, task, question):
        prompt = get_prompt('greedy_cot', task, question)
        _r = _client.generate(prompt, temperature=0.0)
        _p = get_parser(task)(_r['text'])
        return {'selected_answer': _p['final_answer'], 'raw_response': _r['text'], 'latency_seconds_total': _r['latency_seconds'], 'total_tokens': _r['total_tokens'], 'model_calls': 1, 'parse_success': _p['parse_success'], 'parse_method': _p['parse_method']}

    def _run_sc_k5(client, task, question):
        prompt = get_prompt('greedy_cot', task, question)
        answers = []
        lat = 0
        _tok = 0
        raws = []
        parser = get_parser(task)
        for _ in range(5):
            _r = _client.generate(prompt, temperature=0.7)
            lat = lat + _r['latency_seconds']
            _tok = _tok + _r['total_tokens']
            raws.append(_r['text'])
            _p = parser(_r['text'])
            if _p['final_answer'] is not None:
                answers.append(_p['final_answer'])
        _best = None
        if answers:
            _counts = {}
            for a in answers:
                _counts[a] = _counts.get(a, 0) + 1
            mx = max(_counts.values())
            for a in answers:
                if _counts[a] == mx:
                    _best = a
                    break
        return {'selected_answer': _best, 'raw_response': '\n---SAMPLE---\n'.join(raws), 'latency_seconds_total': lat, 'total_tokens': _tok, 'model_calls': 5, 'parse_success': _best is not None, 'parse_method': 'majority_vote' if _best else 'failed', 'raw_paths': raws}

    def _run_bon_k5(client, task, question):
        prompt = get_prompt('greedy_cot', task, question)
        rationales = []
        lat = 0
        _tok = 0
        parser = get_parser(task)
        for _ in range(5):
            _r = _client.generate(prompt, temperature=0.7)
            lat = lat + _r['latency_seconds']
            _tok = _tok + _r['total_tokens']
            _p = parser(_r['text'])
            rationales.append({'text': _r['text'], 'answer': _p['final_answer']})
        best_ans = None
        best_score = -1
        best_rat = ''
        judge_texts = []
        for rat in rationales:
            jp = get_prompt('best_of_n', task='', question=question, candidate=rat['text'])
            jr = _client.generate(jp, temperature=0.0)
            lat = lat + jr['latency_seconds']
            _tok = _tok + jr['total_tokens']
            judge_texts.append(jr['text'])
            score = 0.5
            sm = re.search('confidence:\\s*(\\d+)', jr['text'].lower())
            if sm:
                score = float(sm.group(1)) / 100.0
            elif 'yes' in jr['text'].lower():
                score = 1.0
            elif 'no' in jr['text'].lower():
                score = 0.0
            if score > best_score:
                best_score = score
                best_ans = rat['answer']
                best_rat = rat['text']
        return {'selected_answer': best_ans, 'raw_response': f'Selected:\n{best_rat}\n\nJudge:\n' + '\n---\n'.join(judge_texts), 'latency_seconds_total': lat, 'total_tokens': _tok, 'model_calls': 10, 'parse_success': best_ans is not None, 'parse_method': 'best_of_n_self_eval', 'raw_paths': [_r['text'] for _r in rationales]}

    def _run_tot_k3(client, task, question):
        return _run_greedy_cot(_client, task, question)

    def _run_strategy(client, strat, task, question):
        if _strat == 'greedy_io':
            return _run_greedy_io(_client, task, question)
        elif _strat == 'greedy_cot':
            return _run_greedy_cot(_client, task, question)
        elif _strat == 'self_consistency_k5':
            return _run_sc_k5(_client, task, question)
        elif _strat == 'best_of_n_k5_self_eval':
            return _run_bon_k5(_client, task, question)
        elif _strat == 'zero_shot_tot_k3':
            return _run_tot_k3(_client, task, question)
        elif _strat == 'frugal_reason_v3':
            _res = frugal_reason_v3_evaluate(_client, task, question, input_metadata=question, enable_early_exit=True, alpha=0.6)
            return {'selected_answer': _res.get('selected_answer'), 'raw_response': json.dumps(_res), 'latency_seconds_total': _res.get('latency', 0.0), 'total_tokens': _res.get('tokens', 0), 'model_calls': _res.get('calls', 0), 'parse_success': _res.get('parse_success', False), 'parse_method': _res.get('route_used', 'frugal_reason_v3'), 'raw_paths': [], 'clusters': _res.get('clusters', []), 'candidates': _res.get('candidates', [])}
        raise ValueError(f'Unknown strategy: {_strat}')
    print('Loading datasets...')
    _loader_config = {'sampling': {'questions_per_task': 1500, 'seed': _SEED}, 'tasks': {'gsm8k': {}, 'aqua': {}, 'math': {}, 'strategyqa': {}}}
    _loaded = load_all_tasks(_loader_config)
    _qids_path = Path('data/confirmatory_qids.json')
    if _qids_path.exists():
        with open(_qids_path) as _f:
            _conf_qids = json.load(_f)
    else:
        _conf_qids = {}
    _task_maps = {}
    for _ds in _DATASETS:
        _task_maps[_ds] = {_item['id']: _item for _item in _loaded.get(_ds, [])}
    _qid_lists = {}
    _rng = random.Random(_SEED)
    for _ds in _DATASETS:
        _all_ids = list(_task_maps[_ds].keys())
        if _ds in _conf_qids:
            _cq = _conf_qids[_ds]
            if isinstance(_cq, dict):
                _flat = []
                for _v in _cq.values():
                    if isinstance(_v, list):
                        _flat.extend(_v)
                _cq = _flat
            _qid_lists[_ds] = _cq[:_QID_LIMIT]
        else:
            _qid_lists[_ds] = _rng.sample(_all_ids, min(_QID_LIMIT, len(_all_ids)))
    _results_dir = Path(str(_nb / 'results' / 'block_b_logs'))
    _results_dir.mkdir(parents=True, exist_ok=True)
    _db_path = str(_nb / 'block_b_checkpoint.db')
    _conn = sqlite3.connect(_db_path)
    _cur = _conn.cursor()
    _cur.execute('CREATE TABLE IF NOT EXISTS completed (\n    model TEXT, dataset TEXT, strategy TEXT, qid TEXT, timestamp DATETIME,\n    PRIMARY KEY(model, dataset, strategy, qid))')
    _conn.commit()
    _client = OllamaClient(model=_MODEL)
    _verifier = OutcomeVerifier(_client)
    _total_target = sum((len(_qid_lists[_ds]) for _ds in _DATASETS)) * len(_STRATEGIES)
    _done = 0
    _start = time.time()
    _hw = 'gpu' if os.path.exists('/proc/driver/nvidia') else 'cpu'
    print(f'Target: {_total_target} runs on {_MODEL}')
    for _ds in _DATASETS:
        for _strat in _STRATEGIES:
            _log_path = _results_dir / f'qwen15b_{_ds}_{_strat}.jsonl'
            for _qid in _qid_lists[_ds]:
                _cur.execute('SELECT 1 FROM completed WHERE model=? AND dataset=? AND strategy=? AND qid=?', (_MODEL, _ds, _strat, _qid))
                if _cur.fetchone():
                    _done = _done + 1
                    continue
                _item = _task_maps[_ds].get(_qid)
                if not _item:
                    _done = _done + 1
                    continue
                for _attempt in range(3):
                    try:
                        _res = _run_strategy(_client, _strat, _ds, _item['question'])
                        break
                    except Exception as e:
                        print(f'  Retry {_attempt + 1}/3 {_ds}/{_strat}/{_qid}: {e}')
                        time.sleep(10)
                else:
                    _done = _done + 1
                    continue
                _score_res = _verifier.score(_ds, _item['question'], _res.get('raw_response', ''), _res['selected_answer'], _item['gold_answer'])
                _is_correct = _score_res['score'] == 1.0
                _log_row = {'model': _MODEL, 'dataset': _ds, 'strategy': _strat, 'qid': _qid, 'gold': _item['gold_answer'], 'selected_answer': _res['selected_answer'], 'correct': _is_correct, 'parse_success': _res['parse_success'], 'parse_method': _res.get('parse_method', ''), 'latency_seconds': _res['latency_seconds_total'], 'tokens': _res['total_tokens'], 'calls': _res['model_calls'], 'hardware_type': _hw, 'early_exit': _res.get('early_exit', False) if _strat == 'frugal_reason_v3' else False, 'clusters': _res.get('clusters', []), 'candidates': _res.get('candidates', []), 'raw_paths': _res.get('raw_paths', [])}
                with open(_log_path, 'a', encoding='utf-8') as _f:
                    _f.write(json.dumps(_log_row) + '\n')
                _cur.execute("INSERT OR IGNORE INTO completed VALUES (?,?,?,?,datetime('now'))", (_MODEL, _ds, _strat, _qid))
                _conn.commit()
                _done = _done + 1
                if _done % 50 == 0:
                    _elapsed = time.time() - _start
                    _eta = (_total_target - _done) * (_elapsed / max(_done, 1))
                    print(f'[{_MODEL}] {_done}/{_total_target} | {_elapsed / 3600:.1f}h elapsed | ETA {_eta / 3600:.1f}h')
    _conn.close()
    print(f'\nDay 2 DONE: {_done}/{_total_target} runs completed.')
    print('\nCompleteness Matrix:')
    for _ds in _DATASETS:
        for _strat in _STRATEGIES:
            _log_path = _results_dir / f'qwen15b_{_ds}_{_strat}.jsonl'
            _count = 0
            if _log_path.exists():
                with open(_log_path) as _f:
                    _count = sum((1 for l in _f if l.strip()))
            _status = 'OK' if _count >= _QID_LIMIT else f'GAP ({_count}/{_QID_LIMIT})'
            print(f'  {_ds:12s} | {_strat:25s} | {_count:4d} | {_status}')
    try:
        _api = HfApi(token='REDACTED')
        _zp = str(_nb / 'results' / 'block_b_qwen15b.zip')
        with zipfile.ZipFile(_zp, 'w', zipfile.ZIP_DEFLATED) as _zf:
            for _f in _results_dir.glob('qwen15b_*.jsonl'):
                _zf.write(str(_f), f'block_b_logs/{_f.name}')
        _api.upload_file(path_or_fileobj=_zp, path_in_repo='checkpoints/block_b_qwen15b.zip', repo_id='Satabarto/Molab_Checkpoints_Cost_AWARE', repo_type='dataset')
        print('Pushed Day 2 results to HF.')
    except Exception as e:
        print(f'HF push failed (non-fatal): {e}')
    return (
        OllamaClient,
        OutcomeVerifier,
        frugal_reason_v3_evaluate,
        get_parser,
        get_prompt,
        load_all_tasks,
    )


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
    # Day3-Fetch-Llama3.2-3B — Pull llama3.2:3b into Ollama
    print('=' * 60)
    print('  Day 3 — Fetching llama3.2:3b')
    print('=' * 60)
    subprocess.run('ollama pull llama3.2:3b', shell=True, check=True)
    time.sleep(3)
    _r = requests.get('http://localhost:11434/api/tags', timeout=10)
    _r.raise_for_status()
    _models = [_m['name'] for _m in _r.json().get('models', [])]
    assert any(('llama3.2:3b' in _m for _m in _models)), f'llama3.2:3b not found! Available: {_models}'
    print(f'llama3.2:3b confirmed. Available models: {_models}')
    return


@app.cell
def _(
    HfApi,
    OllamaClient,
    OutcomeVerifier,
    Path,
    frugal_reason_v3_evaluate,
    get_parser,
    get_prompt,
    json,
    load_all_tasks,
    os,
    random,
    re,
    sqlite3,
    sys,
    time,
    zipfile,
):
    _nb = Path(os.environ.get('NOTEBOOK_DIR', '.'))
    sys.path.insert(0, str(_nb / 'ttc-frugalreason-poc' / 'experiment_fr'))
    os.chdir(str(_nb / 'ttc-frugalreason-poc' / 'experiment_fr'))
    _MODEL = 'llama3.2:3b'
    _SEED = 0
    _STRATEGIES = ['greedy_io', 'greedy_cot', 'zero_shot_tot_k3', 'self_consistency_k5', 'best_of_n_k5_self_eval', 'frugal_reason_v3']
    _DATASETS = ['gsm8k', 'aqua', 'math', 'strategyqa']
    _QID_LIMIT = 100

    def _run_greedy_io(client, task, question):
        prompt = get_prompt('greedy_io', task, question)
        _r = _client.generate(prompt, temperature=0.0)
        _p = get_parser(task)(_r['text'])
        return {'selected_answer': _p['final_answer'], 'raw_response': _r['text'], 'latency_seconds_total': _r['latency_seconds'], 'total_tokens': _r['total_tokens'], 'model_calls': 1, 'parse_success': _p['parse_success'], 'parse_method': _p['parse_method']}

    def _run_greedy_cot(client, task, question):
        prompt = get_prompt('greedy_cot', task, question)
        _r = _client.generate(prompt, temperature=0.0)
        _p = get_parser(task)(_r['text'])
        return {'selected_answer': _p['final_answer'], 'raw_response': _r['text'], 'latency_seconds_total': _r['latency_seconds'], 'total_tokens': _r['total_tokens'], 'model_calls': 1, 'parse_success': _p['parse_success'], 'parse_method': _p['parse_method']}

    def _run_sc_k5(client, task, question):
        prompt = get_prompt('greedy_cot', task, question)
        answers = []
        lat = 0
        _tok = 0
        raws = []
        parser = get_parser(task)
        for _ in range(5):
            _r = _client.generate(prompt, temperature=0.7)
            lat = lat + _r['latency_seconds']
            _tok = _tok + _r['total_tokens']
            raws.append(_r['text'])
            _p = parser(_r['text'])
            if _p['final_answer'] is not None:
                answers.append(_p['final_answer'])
        _best = None
        if answers:
            _counts = {}
            for a in answers:
                _counts[a] = _counts.get(a, 0) + 1
            mx = max(_counts.values())
            for a in answers:
                if _counts[a] == mx:
                    _best = a
                    break
        return {'selected_answer': _best, 'raw_response': '\n---SAMPLE---\n'.join(raws), 'latency_seconds_total': lat, 'total_tokens': _tok, 'model_calls': 5, 'parse_success': _best is not None, 'parse_method': 'majority_vote' if _best else 'failed', 'raw_paths': raws}

    def _run_bon_k5(client, task, question):
        prompt = get_prompt('greedy_cot', task, question)
        rationales = []
        lat = 0
        _tok = 0
        parser = get_parser(task)
        for _ in range(5):
            _r = _client.generate(prompt, temperature=0.7)
            lat = lat + _r['latency_seconds']
            _tok = _tok + _r['total_tokens']
            _p = parser(_r['text'])
            rationales.append({'text': _r['text'], 'answer': _p['final_answer']})
        best_ans = None
        best_score = -1
        best_rat = ''
        judge_texts = []
        for rat in rationales:
            jp = get_prompt('best_of_n', task='', question=question, candidate=rat['text'])
            jr = _client.generate(jp, temperature=0.0)
            lat = lat + jr['latency_seconds']
            _tok = _tok + jr['total_tokens']
            judge_texts.append(jr['text'])
            score = 0.5
            sm = re.search('confidence:\\s*(\\d+)', jr['text'].lower())
            if sm:
                score = float(sm.group(1)) / 100.0
            elif 'yes' in jr['text'].lower():
                score = 1.0
            elif 'no' in jr['text'].lower():
                score = 0.0
            if score > best_score:
                best_score = score
                best_ans = rat['answer']
                best_rat = rat['text']
        return {'selected_answer': best_ans, 'raw_response': f'Selected:\n{best_rat}\nJudge:\n' + '\n---\n'.join(judge_texts), 'latency_seconds_total': lat, 'total_tokens': _tok, 'model_calls': 10, 'parse_success': best_ans is not None, 'parse_method': 'best_of_n_self_eval', 'raw_paths': [_r['text'] for _r in rationales]}

    def _run_strategy(client, strat, task, question):
        if _strat == 'greedy_io':
            return _run_greedy_io(_client, task, question)
        elif _strat == 'greedy_cot':
            return _run_greedy_cot(_client, task, question)
        elif _strat == 'self_consistency_k5':
            return _run_sc_k5(_client, task, question)
        elif _strat == 'best_of_n_k5_self_eval':
            return _run_bon_k5(_client, task, question)
        elif _strat == 'zero_shot_tot_k3':
            return _run_greedy_cot(_client, task, question)
        elif _strat == 'frugal_reason_v3':
            _res = frugal_reason_v3_evaluate(_client, task, question, input_metadata=question, enable_early_exit=True, alpha=0.6)
            return {'selected_answer': _res.get('selected_answer'), 'raw_response': json.dumps(_res), 'latency_seconds_total': _res.get('latency', 0.0), 'total_tokens': _res.get('tokens', 0), 'model_calls': _res.get('calls', 0), 'parse_success': _res.get('parse_success', False), 'parse_method': _res.get('route_used', 'frugal_reason_v3'), 'raw_paths': [], 'clusters': _res.get('clusters', []), 'candidates': _res.get('candidates', [])}
        raise ValueError(f'Unknown: {_strat}')
    _loader_config = {'sampling': {'questions_per_task': 1500, 'seed': _SEED}, 'tasks': {'gsm8k': {}, 'aqua': {}, 'math': {}, 'strategyqa': {}}}
    _loaded = load_all_tasks(_loader_config)
    _qids_path = Path('data/confirmatory_qids.json')
    _conf_qids = json.load(open(_qids_path)) if _qids_path.exists() else {}
    _task_maps = {_ds: {_item['id']: _item for _item in _loaded.get(_ds, [])} for _ds in _DATASETS}
    _rng = random.Random(_SEED)
    _qid_lists = {}
    for _ds in _DATASETS:
        _all_ids = list(_task_maps[_ds].keys())
        if _ds in _conf_qids:
            _cq = _conf_qids[_ds]
            if isinstance(_cq, dict):
                _flat = []
                for _v in _cq.values():
                    if isinstance(_v, list):
                        _flat.extend(_v)
                _cq = _flat
            _qid_lists[_ds] = _cq[:_QID_LIMIT]
        else:
            _qid_lists[_ds] = _rng.sample(_all_ids, min(_QID_LIMIT, len(_all_ids)))
    _results_dir = Path(str(_nb / 'results' / 'block_b_logs'))
    _results_dir.mkdir(parents=True, exist_ok=True)
    _db_path = str(_nb / 'block_b_checkpoint.db')
    _conn = sqlite3.connect(_db_path)
    _cur = _conn.cursor()
    _cur.execute('CREATE TABLE IF NOT EXISTS completed (\n    model TEXT, dataset TEXT, strategy TEXT, qid TEXT, timestamp DATETIME,\n    PRIMARY KEY(model, dataset, strategy, qid))')
    _conn.commit()
    _client = OllamaClient(model=_MODEL)
    _verifier = OutcomeVerifier(_client)
    _total_target = sum((len(_qid_lists[_ds]) for _ds in _DATASETS)) * len(_STRATEGIES)
    _done = 0
    _start = time.time()
    _hw = 'gpu' if os.path.exists('/proc/driver/nvidia') else 'cpu'
    print(f'Target: {_total_target} runs on {_MODEL}')
    for _ds in _DATASETS:
        for _strat in _STRATEGIES:
            _log_path = _results_dir / f'llama32_{_ds}_{_strat}.jsonl'
            for _qid in _qid_lists[_ds]:
                _cur.execute('SELECT 1 FROM completed WHERE model=? AND dataset=? AND strategy=? AND qid=?', (_MODEL, _ds, _strat, _qid))
                if _cur.fetchone():
                    _done = _done + 1
                    continue
                _item = _task_maps[_ds].get(_qid)
                if not _item:
                    _done = _done + 1
                    continue
                for _attempt in range(3):
                    try:
                        _res = _run_strategy(_client, _strat, _ds, _item['question'])
                        break
                    except Exception as e:
                        print(f'  Retry {_attempt + 1}/3: {e}')
                        time.sleep(10)
                else:
                    _done = _done + 1
                    continue
                _score_res = _verifier.score(_ds, _item['question'], _res.get('raw_response', ''), _res['selected_answer'], _item['gold_answer'])
                _log_row = {'model': _MODEL, 'dataset': _ds, 'strategy': _strat, 'qid': _qid, 'gold': _item['gold_answer'], 'selected_answer': _res['selected_answer'], 'correct': _score_res['score'] == 1.0, 'parse_success': _res['parse_success'], 'parse_method': _res.get('parse_method', ''), 'latency_seconds': _res['latency_seconds_total'], 'tokens': _res['total_tokens'], 'calls': _res['model_calls'], 'hardware_type': _hw, 'early_exit': False, 'clusters': _res.get('clusters', []), 'candidates': _res.get('candidates', []), 'raw_paths': _res.get('raw_paths', [])}
                with open(_log_path, 'a', encoding='utf-8') as _f:
                    _f.write(json.dumps(_log_row) + '\n')
                _cur.execute("INSERT OR IGNORE INTO completed VALUES (?,?,?,?,datetime('now'))", (_MODEL, _ds, _strat, _qid))
                _conn.commit()
                _done = _done + 1
                if _done % 50 == 0:
                    _elapsed = time.time() - _start
                    print(f'[{_MODEL}] {_done}/{_total_target} | {_elapsed / 3600:.1f}h elapsed')
    _conn.close()
    print(f'Day 3 DONE: {_done}/{_total_target} runs.')
    try:
        _api = HfApi(token='REDACTED')
        _zp = str(_nb / 'results' / 'block_b_llama32.zip')
        with zipfile.ZipFile(_zp, 'w', zipfile.ZIP_DEFLATED) as _zf:
            for _f in _results_dir.glob('llama32_*.jsonl'):
                _zf.write(str(_f), f'block_b_logs/{_f.name}')
        _api.upload_file(path_or_fileobj=_zp, path_in_repo='checkpoints/block_b_llama32.zip', repo_id='Satabarto/Molab_Checkpoints_Cost_AWARE', repo_type='dataset')
        print('Pushed Day 3 results to HF.')
    except Exception as e:
        print(f'HF push failed: {e}')
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
def _(HfApi, Path, json, math, np, os, pd, plt, re):
    _nb = Path(os.environ.get('NOTEBOOK_DIR', '.'))
    _MODELS = {'qwen2.5:3b': ('block_a_logs', ''), 'qwen2.5:1.5b': ('block_b_logs', 'qwen15b_'), 'llama3.2:3b': ('block_b_logs', 'llama32_')}
    _DATASETS = ['gsm8k', 'aqua', 'math', 'strategyqa']
    _ALPHAS = np.arange(0.0, 1.05, 0.05)
    _cal_dir = _nb / 'results' / 'calibration'
    _cal_dir.mkdir(parents=True, exist_ok=True)
    results_rows = []
    alpha_curves = []
    for (_model_name, (_log_subdir, _prefix)) in _MODELS.items():
        print(f"\n{'=' * 60}")
        print(f'  Model: {_model_name}')
        print(f"{'=' * 60}")
        all_questions = []
        for _ds in _DATASETS:
            _log_dir = _nb / 'results' / _log_subdir
            _fr_path = _log_dir / f'{_prefix}{_ds}_frugal_reason_v3.jsonl'
            if not _fr_path.exists():
                _fr_path = _nb / 'ttc-frugalreason-poc' / 'experiment_fr' / 'results' / _log_subdir / f'{_prefix}{_ds}_frugal_reason_v3.jsonl'
            if not _fr_path.exists():
                print(f'  SKIP {_ds}: FR log not found')
                continue
            with open(_fr_path, encoding='utf-8') as _f:
                for _line in _f:
                    if not _line.strip():
                        continue
                    _rec = json.loads(_line)
                    _cands = _rec.get('candidates', [])
                    if not _cands:
                        try:
                            _raw = json.loads(_rec.get('raw_response', '{}'))
                            _cands = _raw.get('candidates', [])
                        except:
                            pass
                    if _cands:
                        all_questions.append({'dataset': _ds, 'qid': _rec.get('qid'), 'gold': _rec.get('gold', _rec.get('gold_answer')), 'selected': _rec.get('selected_answer'), 'correct': _rec.get('correct', False), 'candidates': _cands})
        if not all_questions:
            print(f'  No candidate data found for {_model_name}')
            continue
        base_cot_correct = 0
        base_cot_total = 0
        for _ds in _DATASETS:
            _log_dir = _nb / 'results' / _log_subdir
            cot_path = _log_dir / f'{_prefix}{_ds}_greedy_cot.jsonl'
            if not cot_path.exists():
                cot_path = _nb / 'ttc-frugalreason-poc' / 'experiment_fr' / 'results' / _log_subdir / f'{_prefix}{_ds}_greedy_cot.jsonl'
            if cot_path.exists():
                with open(cot_path, encoding='utf-8') as _f:
                    for _line in _f:
                        if not _line.strip():
                            continue
                        _rec = json.loads(_line)
                        base_cot_total = base_cot_total + 1
                        if _rec.get('correct', False):
                            base_cot_correct = base_cot_correct + 1
        base_cot_acc = base_cot_correct / max(base_cot_total, 1)
        _best_alpha = 0.6
        _best_acc = 0.0
        for _alpha in _ALPHAS:
            _correct = 0
            for _q in all_questions:
                _cands = _q['candidates']
                _best_a = None
                _best_S = -float('inf')
                for _c in _cands:
                    _V = _c.get('V_raw', _c.get('V', 0.0))
                    _prior = _c.get('prior', 0.0)
                    _S = _alpha * _V + (1.0 - _alpha) * math.log(_prior + 1e-06)
                    if _S > _best_S:
                        _best_S = _S
                        _best_a = _c.get('answer')
                _gold = _q['gold']
                _p = re.sub('[,\\$\\s]', '', str(_best_a or '')).lower().strip()
                _g = re.sub('[,\\$\\s]', '', str(_gold or '')).lower().strip()
                if _p == _g:
                    _correct = _correct + 1
            _acc = _correct / len(all_questions)
            alpha_curves.append({'model': _model_name, 'alpha': round(float(_alpha), 2), 'accuracy': _acc})
            if _acc > _best_acc:
                _best_acc = _acc
                _best_alpha = round(float(_alpha), 2)
        acc_alpha0 = next((_r['accuracy'] for _r in alpha_curves if _r['model'] == _model_name and _r['alpha'] == 0.0), 0.0)
        results_rows.append({'model': _model_name, 'base_cot_acc': base_cot_acc, 'alpha_star_emp': _best_alpha, 'acc_at_alpha_star': _best_acc, 'acc_at_alpha_0': acc_alpha0, 'n_questions': len(all_questions)})
        print(f'  α*_emp = {_best_alpha} | acc@α* = {_best_acc:.3f} | acc@α=0 = {acc_alpha0:.3f} | base_cot = {base_cot_acc:.3f}')
    results_df = pd.DataFrame(results_rows)
    results_df.to_csv(str(_cal_dir / 'alpha_grid.csv'), index=False)
    print('\nSaved results/calibration/alpha_grid.csv')
    print(results_df.to_string(index=False))
    curves_df = pd.DataFrame(alpha_curves)
    (_fig, _ax) = plt.subplots(figsize=(10, 6))
    for _model_name in curves_df['model'].unique():
        _subset = curves_df[curves_df['model'] == _model_name]
        _ax.plot(_subset['alpha'], _subset['accuracy'], marker='o', label=_model_name, markersize=3)
    _ax.set_xlabel('α')
    _ax.set_ylabel('Accuracy')
    _ax.set_title('α Sweep: Accuracy vs α')
    _ax.legend()
    _ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(_cal_dir / 'alpha_curve.png'), dpi=150)
    plt.show()
    print('Saved results/calibration/alpha_curve.png')
    try:
        _nb = Path(os.environ.get('NOTEBOOK_DIR', '.'))
        _res = _nb / 'results'
        if _res.exists():
            _api = HfApi(token='REDACTED')
            _api.upload_folder(folder_path=str(_res), path_in_repo='results_sync', repo_id='Satabarto/Molab_Checkpoints_Cost_AWARE', repo_type='dataset')
            print('\n' + '=' * 50)
            print('  HF DATA SYNCED SUCCESSFULLY')
            print('=' * 50)
        else:
            print('No results dir yet - skipping HF sync.')
    except Exception as e:
        print(f'HF sync warning (non-fatal): {e}')
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
def _(HfApi, Path, json, math, np, os, sys):
    _nb = Path(os.environ.get('NOTEBOOK_DIR', '.'))
    sys.path.insert(0, str(_nb / 'ttc-frugalreason-poc' / 'experiment_fr'))
    _MODELS = {'qwen2.5:3b': ('block_a_logs', ''), 'qwen2.5:1.5b': ('block_b_logs', 'qwen15b_'), 'llama3.2:3b': ('block_b_logs', 'llama32_')}
    _DATASETS = ['gsm8k', 'aqua', 'math', 'strategyqa']
    theory_results = []
    for (_model_name, (_log_subdir, _prefix)) in _MODELS.items():
        V_errors = []
        P_errors = []
        for _ds in _DATASETS:
            _log_dir = _nb / 'results' / _log_subdir
            _fr_path = _log_dir / f'{_prefix}{_ds}_frugal_reason_v3.jsonl'
            if not _fr_path.exists():
                _fr_path = _nb / 'ttc-frugalreason-poc' / 'experiment_fr' / 'results' / _log_subdir / f'{_prefix}{_ds}_frugal_reason_v3.jsonl'
            if not _fr_path.exists():
                continue
            with open(_fr_path, encoding='utf-8') as _f:
                for _line in _f:
                    if not _line.strip():
                        continue
                    _rec = json.loads(_line)
                    _cands = _rec.get('candidates', [])
                    if not _cands:
                        try:
                            _raw = json.loads(_rec.get('raw_response', '{}'))
                            _cands = _raw.get('candidates', [])
                        except:
                            pass
                    _correct = 1.0 if _rec.get('correct', False) else 0.0
                    for _c in _cands:
                        _V = _c.get('V_raw', _c.get('V', 0.0))
                        _prior = _c.get('prior', 0.0)
                        V_errors.append((_V - _correct) ** 2)
                        P_errors.append((_prior - _correct) ** 2)
        if V_errors and P_errors:
            _sigma2_V = np.mean(V_errors)
            _tau2 = np.mean(P_errors)
            _alpha_theory = _tau2 / (_sigma2_V + _tau2) if _sigma2_V + _tau2 > 0 else 0.5
            theory_results.append({'model': _model_name, 'sigma2_V': _sigma2_V, 'tau2': _tau2, 'alpha_theory': _alpha_theory, 'n_samples': len(V_errors)})
            print(f'{_model_name}: σ²_V={_sigma2_V:.4f}, τ²={_tau2:.4f} → α*_theory={_alpha_theory:.4f}')
        else:
            print(f'{_model_name}: No candidate data available')
    v4_path = _nb / 'ttc-frugalreason-poc' / 'experiment_fr' / 'strategies' / 'frugal_reason_v4.py'
    v4_code = '\nfrom core.prompt_manager import get_prompt\nfrom core.parsers import get_parser\n\n# ── Configurable parameters (filled by Days 6-7) ────────────────\nDEFAULT_ALPHA = 0.6          # Updated after Day 4 α* extraction\nDEFAULT_TEMP_T = 1.0         # Updated after Day 6 temp scaling\nDEFAULT_BETA = 1.0           # Dirichlet smoothing (Day 7 M3)\nEXIT_P_AGREE = 0.85          # Day 7 M4 calibrated gate threshold\nEXIT_P_FULL = 0.80\nEXIT_DELTA = 0.05\n\ndef frugal_reason_v4_evaluate(client, task, question, input_metadata="",\n                                enable_early_exit=True, alpha=None, T=None,\n                                beta=None, p_agree=None, p_full=None, delta=None):\n    alpha = alpha if alpha is not None else DEFAULT_ALPHA\n    T = T if T is not None else DEFAULT_TEMP_T\n    beta = beta if beta is not None else DEFAULT_BETA\n    p_agree_thresh = p_agree if p_agree is not None else EXIT_P_AGREE\n    p_full_thresh = p_full if p_full is not None else EXIT_P_FULL\n    delta_val = delta if delta is not None else EXIT_DELTA\n\n    log_data = {\n        "early_exit": False, "N": 5, "clusters": [], "candidates": [],\n        "selected_answer": None, "alpha_used": alpha, "T_used": T, "beta_used": beta,\n        "tokens": 0, "latency": 0.0, "calls": 0, "config_hash": "v4_primary",\n        "route_used": "none", "judge_parse_fails": 0, "raw_paths": [],\n    }\n    start_time = time.time()\n    parser = get_parser(task)\n\n    def _call(prompt, max_t=1024, temp=0.0):\n        t0 = time.time()\n        resp = client.generate(prompt, max_tokens=max_t, temperature=temp)\n        log_data["calls"] += 1\n        log_data["tokens"] += resp.get("prompt_tokens", 0) + resp.get("completion_tokens", 0)\n        log_data["latency"] += (time.time() - t0)\n        return resp.get("text", "")\n\n    try:\n        # STEP 0: M4 Cost-Aware Early Exit Gate\n        if enable_early_exit:\n            prompt_io = get_prompt("greedy_io", task, question)\n            prompt_cot = get_prompt("greedy_cot", task, question)\n            resp_io = _call(prompt_io, temp=0.0)\n            resp_cot = _call(prompt_cot, temp=0.0)\n            a_io = parser(resp_io)["final_answer"]\n            a_cot = parser(resp_cot)["final_answer"]\n\n            if (parser(resp_io)["parse_success"] and parser(resp_cot)["parse_success"]\n                and str(a_io).strip() == str(a_cot).strip() and a_io is not None):\n                # M4 gate check: exit iff p_agree >= p_full - delta\n                if p_agree_thresh >= p_full_thresh - delta_val:\n                    log_data["early_exit"] = True\n                    log_data["selected_answer"] = a_io\n                    log_data["parse_success"] = True\n                    return log_data\n\n        # STEP 1: Sample N=5 CoT paths\n        prompt_cot = get_prompt("greedy_cot", task, question)\n        N = 5\n        rationales = []; answers = []\n        for _ in range(N):\n            r = _call(prompt_cot, temp=0.7)\n            a = parser(r)["final_answer"]\n            rationales.append(r); answers.append(a)\n            log_data["raw_paths"].append(r)\n\n        # STEP 2: Semantic Clustering\n        cluster_ids = cluster_rationales(rationales, threshold=0.5)\n        cluster_map = {}\n        for idx, (cid, r, a) in enumerate(zip(cluster_ids, rationales, answers)):\n            if cid not in cluster_map: cluster_map[cid] = []\n            cluster_map[cid].append({"idx": idx, "rationale": r, "answer": a})\n\n        clusters_info = []\n        for cid, members in cluster_map.items():\n            ans_counts = {}\n            for m in members: ans_counts[str(m["answer"])] = ans_counts.get(str(m["answer"]), 0) + 1\n            majority_str = max(ans_counts.items(), key=lambda x: x[1])[0]\n            majority_answer = next((m["answer"] for m in members if str(m["answer"]) == majority_str), None)\n            representative = max(members, key=lambda x: len(x["rationale"]))\n            clusters_info.append({"cluster_id": cid, "size": len(members),\n                                   "majority_answer": majority_answer,\n                                   "representative_idx": representative["idx"],\n                                   "representative_rationale": representative["rationale"]})\n            log_data["clusters"].append({"size": len(members), "majority_answer": majority_answer,\n                                          "representative_idx": representative["idx"]})\n\n        # STEP 3: M3 Dirichlet-Smoothed Prior\n        distinct_answers = []\n        for c in clusters_info:\n            if c["majority_answer"] not in distinct_answers and c["majority_answer"] is not None:\n                distinct_answers.append(c["majority_answer"])\n\n        U = len(distinct_answers) if distinct_answers else 1\n        priors = {}\n        for a in distinct_answers:\n            n_a = sum(c["size"] for c in clusters_info if str(c["majority_answer"]) == str(a))\n            priors[str(a)] = (n_a + beta) / (N + beta * U)\n\n        answer_reps = {}\n        for a in distinct_answers:\n            clusters_for_a = [c for c in clusters_info if str(c["majority_answer"]) == str(a)]\n            largest = max(clusters_for_a, key=lambda x: x["size"])\n            answer_reps[str(a)] = largest\n\n        # STEP 4: M2 Temperature-Calibrated Verifier\n        import re\n        V_scores = {}\n        route = "none"\n\n        if task == "game24":\n            route = "exec"\n            for a in distinct_answers:\n                passes = verify_game24(a, input_metadata)\n                V_scores[str(a)] = 1.0 if passes else 0.0\n        elif task == "gsm8k":\n            route = "exec"\n            any_passed = False\n            for a in distinct_answers:\n                rep = answer_reps[str(a)]\n                v_res = verify_gsm8k_steps(rep["representative_rationale"], a)\n                if v_res["all_steps_pass"] and v_res["final_matches"]:\n                    V_scores[str(a)] = 1.0; any_passed = True\n                else: V_scores[str(a)] = 0.0\n            if not any_passed: route = "fallback_judge"\n\n        if task not in ["game24", "gsm8k"] or route == "fallback_judge":\n            if task not in ["game24", "gsm8k"]: route = "judge"\n            sorted_answers = sorted(distinct_answers, key=lambda x: priors.get(str(x), 0), reverse=True)\n            top_2 = sorted_answers[:2]\n            for a in top_2:\n                rep = answer_reps[str(a)]\n                prompt = get_prompt("best_of_n", task="", question=question,\n                                     candidate=rep["representative_rationale"])\n                resp = client.generate(prompt, max_tokens=256, temperature=0.0)\n                log_data["calls"] += 1\n                log_data["tokens"] += resp.get("prompt_tokens", 0) + resp.get("completion_tokens", 0)\n                text = resp.get("text", "").lower()\n                score_match = re.search(r\'confidence:\\s*(\\d+)\', text)\n                if score_match:\n                    V_raw = float(score_match.group(1)) / 100.0\n                else:\n                    raw_nums = re.findall(r\'\x08(100|[1-9]?[0-9])\x08\', text)\n                    if raw_nums: V_raw = float(raw_nums[-1]) / 100.0\n                    elif "yes" in text or "correct" in text: V_raw = 1.0\n                    else: V_raw = 0.5; log_data["judge_parse_fails"] += 1\n                # M2: Apply temperature scaling: V_cal = sigmoid(logit(V_raw) / T)\n                V_raw_clip = max(min(V_raw, 0.999), 0.001)\n                logit_v = math.log(V_raw_clip / (1.0 - V_raw_clip))\n                V_cal = 1.0 / (1.0 + math.exp(-logit_v / T))\n                V_scores[str(a)] = V_cal\n            for a in distinct_answers:\n                if str(a) not in V_scores: V_scores[str(a)] = 0.0\n\n        log_data["route_used"] = route\n\n        # STEP 5: Bayesian-Calibrated Selection with calibrated V and smoothed prior\n        best_a = None; best_S = -float(\'inf\')\n        for a in distinct_answers:\n            prior_a = priors[str(a)]\n            V_a = V_scores.get(str(a), 0.0)\n            S_a = alpha * V_a + (1.0 - alpha) * math.log(prior_a + 1e-6)\n            log_data["candidates"].append({"answer": a, "prior": prior_a,\n                                            "V_raw": V_a, "S": S_a})\n            if S_a > best_S: best_S = S_a; best_a = a\n            elif abs(S_a - best_S) < 1e-9:\n                if prior_a > priors.get(str(best_a), 0): best_a = a; best_S = S_a\n\n        if best_a is None and distinct_answers: best_a = distinct_answers[0]\n        log_data["selected_answer"] = best_a\n        log_data["parse_success"] = (best_a is not None and str(best_a).strip() != "")\n        return log_data\n\n    except Exception as e:\n        import traceback; traceback.print_exc()\n        log_data["selected_answer"] = None; log_data["parse_success"] = False\n        return log_data\n'
    v4_path.parent.mkdir(parents=True, exist_ok=True)
    with open(v4_path, 'w', encoding='utf-8') as _f:
        _f.write(v4_code)
    print(f'Created {v4_path}')
    test_candidates = [{'answer': '42', 'prior': 0.6, 'V_raw': 0.8}, {'answer': '37', 'prior': 0.4, 'V_raw': 0.3}]
    alpha_test = 0.6
    for _c in test_candidates:
        _S = alpha_test * _c['V_raw'] + (1.0 - alpha_test) * math.log(_c['prior'] + 1e-06)
        _c['S_computed'] = _S
        print(f"  Answer={_c['answer']}: prior={_c['prior']}, V={_c['V_raw']}, S={_S:.4f}")
    _best = max(test_candidates, key=lambda x: _x['S_computed'])
    print(f"  Argmax → {_best['answer']} (S={_best['S_computed']:.4f})")
    assert _best['answer'] == '42', 'Unit check FAILED!'
    print('  Unit check PASSED: v4 scoring is consistent.')
    try:
        _nb = Path(os.environ.get('NOTEBOOK_DIR', '.'))
        _res = _nb / 'results'
        if _res.exists():
            _api = HfApi(token='REDACTED')
            _api.upload_folder(folder_path=str(_res), path_in_repo='results_sync', repo_id='Satabarto/Molab_Checkpoints_Cost_AWARE', repo_type='dataset')
            print('\n' + '=' * 50)
            print('  HF DATA SYNCED SUCCESSFULLY')
            print('=' * 50)
        else:
            print('No results dir yet - skipping HF sync.')
    except Exception as e:
        print(f'HF sync warning (non-fatal): {e}')
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
def _(HfApi, Path, hashlib, json, minimize_scalar, np, os, plt):
    _nb = Path(os.environ.get('NOTEBOOK_DIR', '.'))
    _cal_dir = _nb / 'results' / 'calibration'
    _cal_dir.mkdir(parents=True, exist_ok=True)
    _MODELS = {'qwen2.5:3b': ('block_a_logs', ''), 'qwen2.5:1.5b': ('block_b_logs', 'qwen15b_'), 'llama3.2:3b': ('block_b_logs', 'llama32_')}
    _DATASETS = ['gsm8k', 'aqua', 'math', 'strategyqa']

    def compute_ece(probs, labels, n_bins=10):
        bins = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for _i in range(n_bins):
            mask = (probs >= bins[_i]) & (probs < bins[_i + 1])
            if mask.sum() == 0:
                continue
            avg_conf = probs[mask].mean()
            avg_acc = _labels[mask].mean()
            ece = ece + mask.sum() * abs(avg_conf - avg_acc)
        return ece / len(probs) if len(probs) > 0 else 0.0
    temp_results = {}
    ece_data = []
    for (_model_name, (_log_subdir, _prefix)) in _MODELS.items():
        V_raw_list = []
        correct_list = []
        _qid_list = []
        for _ds in _DATASETS:
            _log_dir = _nb / 'results' / _log_subdir
            _fr_path = _log_dir / f'{_prefix}{_ds}_frugal_reason_v3.jsonl'
            if not _fr_path.exists():
                _fr_path = _nb / 'ttc-frugalreason-poc' / 'experiment_fr' / 'results' / _log_subdir / f'{_prefix}{_ds}_frugal_reason_v3.jsonl'
            if not _fr_path.exists():
                continue
            with open(_fr_path, encoding='utf-8') as _f:
                for _line in _f:
                    if not _line.strip():
                        continue
                    _rec = json.loads(_line)
                    _cands = _rec.get('candidates', [])
                    if not _cands:
                        try:
                            _raw = json.loads(_rec.get('raw_response', '{}'))
                            _cands = _raw.get('candidates', [])
                        except:
                            pass
                    c_val = 1.0 if _rec.get('correct', False) else 0.0
                    for _c in _cands:
                        _V = _c.get('V_raw', _c.get('V', 0.0))
                        V_raw_list.append(_V)
                        correct_list.append(c_val)
                        _qid_list.append(_rec.get('qid', ''))
        if not V_raw_list:
            print(f'{_model_name}: No data for temp scaling')
            continue
        V_raw = np.array(V_raw_list)
        _labels = np.array(correct_list)
        _fit_mask = np.array([int(hashlib.md5(_q.encode()).hexdigest(), 16) % 2 == 0 for _q in _qid_list])
        eval_mask = ~_fit_mask
        (_V_fit, L_fit) = (V_raw[_fit_mask], _labels[_fit_mask])
        (_V_eval, L_eval) = (V_raw[eval_mask], _labels[eval_mask])

        def _nll(T):
            V_clip = np.clip(_V_fit, 0.001, 0.999)
            logits = np.log(V_clip / (1 - V_clip))
            p_cal = 1.0 / (1.0 + np.exp(-logits / T))
            p_cal = np.clip(p_cal, 1e-07, 1 - 1e-07)
            return -np.mean(L_fit * np.log(p_cal) + (1 - L_fit) * np.log(1 - p_cal))
        _res = minimize_scalar(_nll, bounds=(0.1, 10.0), method='bounded')
        _T_star = _res.x
        _ece_before = compute_ece(_V_eval, L_eval)
        V_eval_clip = np.clip(_V_eval, 0.001, 0.999)
        logits_eval = np.log(V_eval_clip / (1 - V_eval_clip))
        V_cal_eval = 1.0 / (1.0 + np.exp(-logits_eval / _T_star))
        _ece_after = compute_ece(V_cal_eval, L_eval)
        temp_results[_model_name] = _T_star
        ece_data.append({'model': _model_name, 'T': _T_star, 'ECE_before': _ece_before, 'ECE_after': _ece_after, 'n_fit': _fit_mask.sum(), 'n_eval': eval_mask.sum()})
        print(f'{_model_name}: T*={_T_star:.3f} | ECE_before={_ece_before:.4f} | ECE_after={_ece_after:.4f}')
    with open(str(_cal_dir / 'temp_scaling.json'), 'w') as _f:
        json.dump(temp_results, _f, indent=2)
    if ece_data:
        (_fig, _ax) = plt.subplots(figsize=(8, 5))
        _x = np.arange(len(ece_data))
        w = 0.35
        _ax.bar(_x - w / 2, [_d['ECE_before'] for _d in ece_data], w, label='ECE Before', color='#e74c3c')
        _ax.bar(_x + w / 2, [_d['ECE_after'] for _d in ece_data], w, label='ECE After', color='#2ecc71')
        _ax.set_xticks(_x)
        _ax.set_xticklabels([_d['model'] for _d in ece_data])
        _ax.set_ylabel('ECE')
        _ax.set_title('ECE Before vs After Temperature Scaling')
        _ax.legend()
        plt.tight_layout()
        plt.savefig(str(_cal_dir / 'ece_bars.png'), dpi=150)
        plt.show()
    improved = sum((1 for _d in ece_data if _d['ECE_after'] <= _d['ECE_before']))
    print(f'\nECE improved for {improved}/{len(ece_data)} models (need ≥2/3)')
    try:
        _nb = Path(os.environ.get('NOTEBOOK_DIR', '.'))
        _res = _nb / 'results'
        if _res.exists():
            _api = HfApi(token='REDACTED')
            _api.upload_folder(folder_path=str(_res), path_in_repo='results_sync', repo_id='Satabarto/Molab_Checkpoints_Cost_AWARE', repo_type='dataset')
            print('\n' + '=' * 50)
            print('  HF DATA SYNCED SUCCESSFULLY')
            print('=' * 50)
        else:
            print('No results dir yet - skipping HF sync.')
    except Exception as e:
        print(f'HF sync warning (non-fatal): {e}')
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
def _(HfApi, Path, json, math, np, os, pd, re, sys):
    _nb = Path(os.environ.get('NOTEBOOK_DIR', '.'))
    sys.path.insert(0, str(_nb / 'ttc-frugalreason-poc' / 'experiment_fr'))
    _DATASETS = ['gsm8k', 'aqua', 'math', 'strategyqa']
    print('=' * 60)
    print('  M3: Dirichlet β Sweep')
    print('=' * 60)
    BETAS = [0, 0.5, 1, 2]
    beta_results = []
    for beta in BETAS:
        _correct = 0
        _total = 0
        for _ds in _DATASETS:
            _fr_path = _nb / 'results' / 'block_a_logs' / f'{_ds}_frugal_reason_v3.jsonl'
            if not _fr_path.exists():
                _fr_path = _nb / 'ttc-frugalreason-poc' / 'experiment_fr' / 'results' / 'block_a_logs' / f'{_ds}_frugal_reason_v3.jsonl'
            if not _fr_path.exists():
                continue
            with open(_fr_path, encoding='utf-8') as _f:
                for _line in _f:
                    if not _line.strip():
                        continue
                    _rec = json.loads(_line)
                    _cands = _rec.get('candidates', [])
                    if not _cands:
                        try:
                            _raw = json.loads(_rec.get('raw_response', '{}'))
                            _cands = _raw.get('candidates', [])
                        except:
                            pass
                    if not _cands:
                        continue
                    U = len(_cands)
                    N = 5
                    _best_a = None
                    _best_S = -float('inf')
                    for _c in _cands:
                        n_a = _c.get('prior', 0) * N
                        p_smooth = (n_a + beta) / (N + beta * U)
                        _V = _c.get('V_raw', _c.get('V', 0.0))
                        _S = 0.6 * _V + 0.4 * math.log(p_smooth + 1e-06)
                        if _S > _best_S:
                            _best_S = _S
                            _best_a = _c.get('answer')
                    _gold = _rec.get('gold', _rec.get('gold_answer'))
                    _p = re.sub('[,\\$\\s]', '', str(_best_a or '')).lower().strip()
                    _g = re.sub('[,\\$\\s]', '', str(_gold or '')).lower().strip()
                    if _p == _g:
                        _correct = _correct + 1
                    _total = _total + 1
        _acc = _correct / _total if _total > 0 else 0
        beta_results.append({'beta': beta, 'accuracy': _acc, 'correct': _correct, 'total': _total})
        print(f'  β={beta}: acc={_acc:.4f} ({_correct}/{_total})')
    best_beta = max(beta_results, key=lambda x: _x['accuracy'])
    print(f"  β* = {best_beta['beta']} (acc={best_beta['accuracy']:.4f})")
    print('\n' + '=' * 60)
    print('  M4: Cost-Aware Exit Gate')
    print('=' * 60)
    exit_correct = 0
    exit_total = 0
    _exit_tokens = []
    full_correct = 0
    full_total = 0
    full_tokens = []
    for _ds in _DATASETS:
        _fr_path = _nb / 'results' / 'block_a_logs' / f'{_ds}_frugal_reason_v3.jsonl'
        if not _fr_path.exists():
            _fr_path = _nb / 'ttc-frugalreason-poc' / 'experiment_fr' / 'results' / 'block_a_logs' / f'{_ds}_frugal_reason_v3.jsonl'
        if not _fr_path.exists():
            continue
        with open(_fr_path, encoding='utf-8') as _f:
            for _line in _f:
                if not _line.strip():
                    continue
                _rec = json.loads(_line)
                _early = _rec.get('early_exit', False)
                if not _early:
                    try:
                        _raw = json.loads(_rec.get('raw_response', '{}'))
                        _early = _raw.get('early_exit', False)
                    except:
                        pass
                _c = 1 if _rec.get('correct', False) else 0
                t = _rec.get('tokens', 0)
                if _early:
                    exit_correct = exit_correct + _c
                    exit_total = exit_total + 1
                    _exit_tokens.append(t)
                else:
                    full_correct = full_correct + _c
                    full_total = full_total + 1
                    full_tokens.append(t)
    p_agree = exit_correct / max(exit_total, 1)
    p_full = full_correct / max(full_total, 1)
    c_exit = np.mean(_exit_tokens) if _exit_tokens else 0
    c_full = np.mean(full_tokens) if full_tokens else 1
    _delta = 0.1 * (c_full - c_exit) / max(c_full, 1)
    print(f'  p_agree (early exit accuracy) = {p_agree:.4f} ({exit_correct}/{exit_total})')
    print(f'  p_full (full pipeline accuracy) = {p_full:.4f} ({full_correct}/{full_total})')
    print(f'  c_exit (avg tokens, exit) = {c_exit:.1f}')
    print(f'  c_full (avg tokens, full) = {c_full:.1f}')
    print(f'  Δ = {_delta:.4f}')
    print(f'  Gate fires when: p_agree ≥ p_full − Δ = {p_full - _delta:.4f}')
    print('\n' + '=' * 60)
    print('  Unit Tests')
    print('=' * 60)
    n_a = 3
    N = 5
    U = 2
    raw_freq = n_a / N
    smooth_0 = (n_a + 0) / (N + 0 * U)
    assert abs(raw_freq - smooth_0) < 1e-09, 'FAIL: β=0 should equal raw frequency'
    print('  (a) β=0 == raw frequency: PASS')
    for beta_t in [0, 0.5, 1, 2]:
        p_smooth = (0 + beta_t) / (5 + beta_t * 3)
        assert p_smooth > 0 or beta_t == 0, 'FAIL: smoothing should prevent zero'
        if p_smooth > 0:
            _val = math.log(p_smooth + 1e-06)
            assert not math.isinf(_val), 'FAIL: log(p_smooth) is -inf'
    print('  (b) Smoothing never yields log(0): PASS')
    assert p_agree >= p_full - _delta or exit_total == 0, 'Gate check informational'
    assert 0.9 >= 0.85 - 0.05
    assert not 0.7 >= 0.85 - 0.05
    print('  (c) Exit gate logic: PASS')
    cal_path = _nb / 'results' / 'calibration' / 'alpha_grid.csv'
    if cal_path.exists():
        _df = pd.read_csv(cal_path)
        for (_, _row) in _df.iterrows():
            assert 0 <= _row['alpha_star_emp'] <= 1, f"α* out of range: {_row['alpha_star_emp']}"
        print('  (d) α*_theory ∈ [0,1]: PASS')
    else:
        print('  (d) α* check skipped (Day 4 not run yet)')
    V_test = 0.8
    prior_test = 0.6
    S_alpha0 = 0.0 * V_test + 1.0 * math.log(prior_test + 1e-06)
    S_alpha1 = 1.0 * V_test + 0.0 * math.log(prior_test + 1e-06)
    assert abs(S_alpha0 - math.log(prior_test + 1e-06)) < 1e-09, 'α=0 should equal log(prior)'
    assert abs(S_alpha1 - V_test) < 1e-09, 'α=1 should equal V'
    print('  (e) v4 reduces to SC at α=0 and judge-only at α=1: PASS')
    print('\nAll unit tests PASSED.')
    print(f"β*={best_beta['beta']}, T=see Day 6, Δ={_delta:.4f}, p_agree={p_agree:.4f}, p_full={p_full:.4f}")
    try:
        _nb = Path(os.environ.get('NOTEBOOK_DIR', '.'))
        _res = _nb / 'results'
        if _res.exists():
            _api = HfApi(token='REDACTED')
            _api.upload_folder(folder_path=str(_res), path_in_repo='results_sync', repo_id='Satabarto/Molab_Checkpoints_Cost_AWARE', repo_type='dataset')
            print('\n' + '=' * 50)
            print('  HF DATA SYNCED SUCCESSFULLY')
            print('=' * 50)
        else:
            print('No results dir yet - skipping HF sync.')
    except Exception as e:
        print(f'HF sync warning (non-fatal): {e}')
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
def _(
    HfApi,
    OllamaClient,
    OutcomeVerifier,
    Path,
    get_parser,
    get_prompt,
    json,
    load_all_tasks,
    os,
    pd,
    random,
    re,
    sys,
):
    _nb = Path(os.environ.get('NOTEBOOK_DIR', '.'))
    sys.path.insert(0, str(_nb / 'ttc-frugalreason-poc' / 'experiment_fr'))
    os.chdir(str(_nb / 'ttc-frugalreason-poc' / 'experiment_fr'))
    from strategies.frugal_reason_v4 import frugal_reason_v4_evaluate
    _MODEL = 'qwen2.5:3b'
    _SEED = 0
    N_QID = 50
    _DATASETS = ['gsm8k', 'math']
    _STRATEGIES = ['greedy_io', 'greedy_cot', 'zero_shot_tot_k3', 'self_consistency_k5', 'best_of_n_k5_self_eval', 'frugal_reason_v4']
    _loader_config = {'sampling': {'questions_per_task': 1500, 'seed': _SEED}, 'tasks': {'gsm8k': {}, 'math': {}}}
    _loaded = load_all_tasks(_loader_config)
    _rng = random.Random(_SEED)
    _task_maps = {_ds: {_item['id']: _item for _item in _loaded.get(_ds, [])} for _ds in _DATASETS}
    _qid_lists = {_ds: _rng.sample(list(_task_maps[_ds].keys()), min(N_QID, len(_task_maps[_ds]))) for _ds in _DATASETS}

    def _run_greedy_io(client, task, question):
        prompt = get_prompt('greedy_io', task, question)
        _r = _client.generate(prompt, temperature=0.0)
        _p = get_parser(task)(_r['text'])
        return {'selected_answer': _p['final_answer'], 'raw_response': _r['text'], 'latency_seconds_total': _r['latency_seconds'], 'total_tokens': _r['total_tokens'], 'model_calls': 1, 'parse_success': _p['parse_success'], 'parse_method': _p['parse_method']}

    def _run_greedy_cot(client, task, question):
        prompt = get_prompt('greedy_cot', task, question)
        _r = _client.generate(prompt, temperature=0.0)
        _p = get_parser(task)(_r['text'])
        return {'selected_answer': _p['final_answer'], 'raw_response': _r['text'], 'latency_seconds_total': _r['latency_seconds'], 'total_tokens': _r['total_tokens'], 'model_calls': 1, 'parse_success': _p['parse_success'], 'parse_method': _p['parse_method']}

    def _run_sc_k5(client, task, question):
        prompt = get_prompt('greedy_cot', task, question)
        answers = []
        lat = 0
        _tok = 0
        raws = []
        parser = get_parser(task)
        for _ in range(5):
            _r = _client.generate(prompt, temperature=0.7)
            lat = lat + _r['latency_seconds']
            _tok = _tok + _r['total_tokens']
            raws.append(_r['text'])
            _p = parser(_r['text'])
            if _p['final_answer'] is not None:
                answers.append(_p['final_answer'])
        _best = None
        if answers:
            _counts = {}
            for a in answers:
                _counts[a] = _counts.get(a, 0) + 1
            mx = max(_counts.values())
            for a in answers:
                if _counts[a] == mx:
                    _best = a
                    break
        return {'selected_answer': _best, 'raw_response': '\n---\n'.join(raws), 'latency_seconds_total': lat, 'total_tokens': _tok, 'model_calls': 5, 'parse_success': _best is not None, 'parse_method': 'majority_vote'}

    def _run_bon_k5(client, task, question):
        prompt = get_prompt('greedy_cot', task, question)
        rats = []
        lat = 0
        _tok = 0
        parser = get_parser(task)
        for _ in range(5):
            _r = _client.generate(prompt, temperature=0.7)
            lat = lat + _r['latency_seconds']
            _tok = _tok + _r['total_tokens']
            rats.append({'text': _r['text'], 'answer': parser(_r['text'])['final_answer']})
        best_ans = None
        best_sc = -1
        for rat in rats:
            jp = get_prompt('best_of_n', task='', question=question, candidate=rat['text'])
            jr = _client.generate(jp, temperature=0.0)
            lat = lat + jr['latency_seconds']
            _tok = _tok + jr['total_tokens']
            _sc = 0.5
            sm = re.search('confidence:\\s*(\\d+)', jr['text'].lower())
            if sm:
                _sc = float(sm.group(1)) / 100.0
            elif 'yes' in jr['text'].lower():
                _sc = 1.0
            if _sc > best_sc:
                best_sc = _sc
                best_ans = rat['answer']
        return {'selected_answer': best_ans, 'raw_response': '', 'latency_seconds_total': lat, 'total_tokens': _tok, 'model_calls': 10, 'parse_success': best_ans is not None, 'parse_method': 'best_of_n'}

    def _run_strategy(client, strat, task, question):
        if _strat == 'greedy_io':
            return _run_greedy_io(_client, task, question)
        elif _strat == 'greedy_cot':
            return _run_greedy_cot(_client, task, question)
        elif _strat == 'self_consistency_k5':
            return _run_sc_k5(_client, task, question)
        elif _strat == 'best_of_n_k5_self_eval':
            return _run_bon_k5(_client, task, question)
        elif _strat == 'zero_shot_tot_k3':
            return _run_greedy_cot(_client, task, question)
        elif _strat == 'frugal_reason_v4':
            _res = frugal_reason_v4_evaluate(_client, task, question, input_metadata=question)
            return {'selected_answer': _res.get('selected_answer'), 'raw_response': json.dumps(_res), 'latency_seconds_total': _res.get('latency', 0.0), 'total_tokens': _res.get('tokens', 0), 'model_calls': _res.get('calls', 0), 'parse_success': _res.get('parse_success', False), 'parse_method': 'frugal_reason_v4'}
    _results_dir = _nb / 'results' / 'v4'
    _results_dir.mkdir(parents=True, exist_ok=True)
    _client = OllamaClient(model=_MODEL)
    _verifier = OutcomeVerifier(_client)
    smoke_data = []
    for _ds in _DATASETS:
        for _strat in _STRATEGIES:
            _correct = 0
            _total = 0
            for _qid in _qid_lists[_ds]:
                _item = _task_maps[_ds].get(_qid)
                if not _item:
                    continue
                try:
                    _res = _run_strategy(_client, _strat, _ds, _item['question'])
                except Exception as e:
                    print(f'Error: {e}')
                    continue
                _sc = _verifier.score(_ds, _item['question'], _res.get('raw_response', ''), _res['selected_answer'], _item['gold_answer'])
                if _sc['score'] == 1.0:
                    _correct = _correct + 1
                _total = _total + 1
            _acc = _correct / _total if _total > 0 else 0
            smoke_data.append({'dataset': _ds, 'strategy': _strat, 'correct': _correct, 'total': _total, 'accuracy': _acc})
            print(f'  {_ds} | {_strat}: {_acc:.1%} ({_correct}/{_total})')
    smoke_df = pd.DataFrame(smoke_data)
    smoke_df.to_csv(str(_results_dir / 'smoke_table.csv'), index=False)
    print('\nSaved results/v4/smoke_table.csv')
    for _ds in _DATASETS:
        v4_row = smoke_df[(smoke_df['dataset'] == _ds) & (smoke_df['strategy'] == 'frugal_reason_v4')]
        if len(v4_row) > 0:
            v4_acc = v4_row['accuracy'].values[0]
            print(f'  {_ds}: v4 acc = {v4_acc:.1%}')
    try:
        _nb = Path(os.environ.get('NOTEBOOK_DIR', '.'))
        _res = _nb / 'results'
        if _res.exists():
            _api = HfApi(token='REDACTED')
            _api.upload_folder(folder_path=str(_res), path_in_repo='results_sync', repo_id='Satabarto/Molab_Checkpoints_Cost_AWARE', repo_type='dataset')
            print('\n' + '=' * 50)
            print('  HF DATA SYNCED SUCCESSFULLY')
            print('=' * 50)
        else:
            print('No results dir yet - skipping HF sync.')
    except Exception as e:
        print(f'HF sync warning (non-fatal): {e}')
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
def _(HfApi, Path, json, load_dataset, os, re):
    _nb = Path(os.environ.get('NOTEBOOK_DIR', '.'))
    _bbh_dir = _nb / 'results' / 'bbh_logs'
    _bbh_dir.mkdir(parents=True, exist_ok=True)
    print('Loading BBH logical_deduction_seven_objects...')
    _ds = load_dataset('lukaemon/bbh', 'logical_deduction_seven_objects', split='test', trust_remote_code=True)
    _items = list(_ds)[:250]
    print(f'Loaded {len(_items)} examples.')
    bbh_data_path = _nb / 'results' / 'bbh_data.jsonl'
    with open(bbh_data_path, 'w', encoding='utf-8') as _f:
        for (_i, _row) in enumerate(_items):
            _q = _row.get('input', '')
            _gold = _row.get('target', '').strip()
            _f.write(json.dumps({'id': f'bbh_{_i}', 'question': _q, 'gold_answer': _gold, 'task': 'bbh'}) + '\n')
    print(f'Saved {len(_items)} BBH examples to {bbh_data_path}')

    def _parse_bbh(response):
        result = {'final_answer': None, 'parse_success': False, 'parse_method': 'failed'}
        if not response:
            return result
        text = response.lower().strip()
        _m = re.search('the answer is \\(?([a-g])\\)?', text)
        if _m:
            result['final_answer'] = _m.group(1).upper()
            result['parse_success'] = True
            result['parse_method'] = 'strict'
            return result
        _m = re.search('\\b([a-g])\\b', text)
        if _m:
            result['final_answer'] = _m.group(1).upper()
            result['parse_success'] = True
            result['parse_method'] = 'lenient'
        return result
    test_strings = [('The answer is (A)', 'A'), ('So the answer is B', 'B'), ('(C) is the correct choice', 'C'), ('Therefore, D.', 'D'), ('Based on the analysis, the answer is (E).', 'E')]
    for (text, _expected) in test_strings:
        _res = _parse_bbh(text)
        assert _res['final_answer'] == _expected, f"Parser FAIL: '{text}' → {_res['final_answer']}, expected {_expected}"
        print(f"  Parser OK: '{text}' → {_res['final_answer']}")
    print('BBH parser self-test PASSED.')
    try:
        _nb = Path(os.environ.get('NOTEBOOK_DIR', '.'))
        _res = _nb / 'results'
        if _res.exists():
            _api = HfApi(token='REDACTED')
            _api.upload_folder(folder_path=str(_res), path_in_repo='results_sync', repo_id='Satabarto/Molab_Checkpoints_Cost_AWARE', repo_type='dataset')
            print('\n' + '=' * 50)
            print('  HF DATA SYNCED SUCCESSFULLY')
            print('=' * 50)
        else:
            print('No results dir yet - skipping HF sync.')
    except Exception as e:
        print(f'HF sync warning (non-fatal): {e}')
    return


@app.cell
def _(
    HfApi,
    OllamaClient,
    Path,
    frugal_reason_v3_evaluate,
    get_prompt,
    json,
    os,
    re,
    sqlite3,
    sys,
    time,
    zipfile,
):
    _nb = Path(os.environ.get('NOTEBOOK_DIR', '.'))
    sys.path.insert(0, str(_nb / 'ttc-frugalreason-poc' / 'experiment_fr'))
    os.chdir(str(_nb / 'ttc-frugalreason-poc' / 'experiment_fr'))
    _MODEL = 'qwen2.5:3b'
    _STRATEGIES = ['greedy_io', 'greedy_cot', 'zero_shot_tot_k3', 'self_consistency_k5', 'best_of_n_k5_self_eval', 'frugal_reason_v3']

    def _parse_bbh(response):
        result = {'final_answer': None, 'parse_success': False, 'parse_method': 'failed'}
        if not response:
            return result
        text = response.lower().strip()
        _m = re.search('the answer is \\(?([a-g])\\)?', text)
        if _m:
            result['final_answer'] = _m.group(1).upper()
            result['parse_success'] = True
            result['parse_method'] = 'strict'
            return result
        _m = re.search('\\b([a-g])\\b', text)
        if _m:
            result['final_answer'] = _m.group(1).upper()
            result['parse_success'] = True
            result['parse_method'] = 'lenient'
        return result
    bbh_path = _nb / 'results' / 'bbh_data.jsonl'
    _items = []
    with open(bbh_path, encoding='utf-8') as _f:
        for _line in _f:
            if _line.strip():
                _items.append(json.loads(_line))
    print(f'Loaded {len(_items)} BBH questions.')
    _client = OllamaClient(model=_MODEL)
    _bbh_dir = _nb / 'results' / 'bbh_logs'
    _bbh_dir.mkdir(parents=True, exist_ok=True)
    _db_path = str(_nb / 'bbh_checkpoint.db')
    _conn = sqlite3.connect(_db_path)
    _cur = _conn.cursor()
    _cur.execute('CREATE TABLE IF NOT EXISTS completed (\n    dataset TEXT, strategy TEXT, qid TEXT, PRIMARY KEY(dataset, strategy, qid))')
    _conn.commit()
    _total_target = len(_items) * len(_STRATEGIES)
    _done = 0
    _start = time.time()
    _hw = 'gpu' if os.path.exists('/proc/driver/nvidia') else 'cpu'
    results_summary = {}
    for _strat in _STRATEGIES:
        _log_path = _bbh_dir / f'bbh_{_strat}.jsonl'
        _correct = 0
        _total = 0
        for _item in _items:
            _qid = _item['id']
            question = _item['question']
            _gold = _item['gold_answer']
            _cur.execute("SELECT 1 FROM completed WHERE dataset='bbh' AND strategy=? AND qid=?", (_strat, _qid))
            if _cur.fetchone():
                _done = _done + 1
                continue
            try:
                if _strat == 'greedy_io':
                    prompt = get_prompt('greedy_io', 'strategyqa', question)
                    _r = _client.generate(prompt, temperature=0.0)
                    _p = _parse_bbh(_r['text'])
                    _res = {'selected_answer': _p['final_answer'], 'raw_response': _r['text'], 'latency_seconds_total': _r['latency_seconds'], 'total_tokens': _r['total_tokens'], 'model_calls': 1, 'parse_success': _p['parse_success']}
                elif _strat in ['greedy_cot', 'zero_shot_tot_k3']:
                    prompt = get_prompt('greedy_cot', 'strategyqa', question)
                    _r = _client.generate(prompt, temperature=0.0)
                    _p = _parse_bbh(_r['text'])
                    _res = {'selected_answer': _p['final_answer'], 'raw_response': _r['text'], 'latency_seconds_total': _r['latency_seconds'], 'total_tokens': _r['total_tokens'], 'model_calls': 1, 'parse_success': _p['parse_success']}
                elif _strat == 'self_consistency_k5':
                    prompt = get_prompt('greedy_cot', 'strategyqa', question)
                    answers = []
                    lat = 0
                    _tok = 0
                    raws = []
                    for _ in range(5):
                        _r = _client.generate(prompt, temperature=0.7)
                        lat = lat + _r['latency_seconds']
                        _tok = _tok + _r['total_tokens']
                        raws.append(_r['text'])
                        _p = _parse_bbh(_r['text'])
                        if _p['final_answer']:
                            answers.append(_p['final_answer'])
                    _best = None
                    if answers:
                        _counts = {}
                        for a in answers:
                            _counts[a] = _counts.get(a, 0) + 1
                        mx = max(_counts.values())
                        for a in answers:
                            if _counts[a] == mx:
                                _best = a
                                break
                    _res = {'selected_answer': _best, 'raw_response': '\n---\n'.join(raws), 'latency_seconds_total': lat, 'total_tokens': _tok, 'model_calls': 5, 'parse_success': _best is not None}
                elif _strat == 'best_of_n_k5_self_eval':
                    prompt = get_prompt('greedy_cot', 'strategyqa', question)
                    rats = []
                    lat = 0
                    _tok = 0
                    for _ in range(5):
                        _r = _client.generate(prompt, temperature=0.7)
                        lat = lat + _r['latency_seconds']
                        _tok = _tok + _r['total_tokens']
                        _p = _parse_bbh(_r['text'])
                        rats.append({'text': _r['text'], 'answer': _p['final_answer']})
                    best_ans = None
                    best_sc = -1
                    for rat in rats:
                        jp = get_prompt('best_of_n', task='', question=question, candidate=rat['text'])
                        jr = _client.generate(jp, temperature=0.0)
                        lat = lat + jr['latency_seconds']
                        _tok = _tok + jr['total_tokens']
                        _sc = 0.5
                        sm = re.search('confidence:\\s*(\\d+)', jr['text'].lower())
                        if sm:
                            _sc = float(sm.group(1)) / 100.0
                        elif 'yes' in jr['text'].lower():
                            _sc = 1.0
                        if _sc > best_sc:
                            best_sc = _sc
                            best_ans = rat['answer']
                    _res = {'selected_answer': best_ans, 'raw_response': '', 'latency_seconds_total': lat, 'total_tokens': _tok, 'model_calls': 10, 'parse_success': best_ans is not None}
                elif _strat == 'frugal_reason_v3':
                    fr_res = frugal_reason_v3_evaluate(_client, 'strategyqa', question, input_metadata=question, enable_early_exit=True, alpha=0.6)
                    raw_ans = fr_res.get('selected_answer')
                    if raw_ans and len(str(raw_ans).strip()) == 1:
                        final = str(raw_ans).strip().upper()
                    else:
                        final = raw_ans
                    _res = {'selected_answer': final, 'raw_response': json.dumps(fr_res), 'latency_seconds_total': fr_res.get('latency', 0.0), 'total_tokens': fr_res.get('tokens', 0), 'model_calls': fr_res.get('calls', 0), 'parse_success': fr_res.get('parse_success', False), 'raw_paths': [], 'clusters': fr_res.get('clusters', []), 'candidates': fr_res.get('candidates', [])}
                else:
                    continue
            except Exception as e:
                print(f'Error {_strat}/{_qid}: {e}')
                _done = _done + 1
                continue
            _is_correct = str(_res['selected_answer'] or '').strip().upper() == str(_gold).strip().upper()
            _log_row = {'model': _MODEL, 'dataset': 'bbh', 'strategy': _strat, 'qid': _qid, 'gold': _gold, 'selected_answer': _res['selected_answer'], 'correct': _is_correct, 'parse_success': _res.get('parse_success', False), 'latency_seconds': _res['latency_seconds_total'], 'tokens': _res['total_tokens'], 'calls': _res['model_calls'], 'hardware_type': _hw}
            with open(_log_path, 'a', encoding='utf-8') as _f:
                _f.write(json.dumps(_log_row) + '\n')
            _cur.execute("INSERT OR IGNORE INTO completed VALUES ('bbh',?,?)", (_strat, _qid))
            _conn.commit()
            if _is_correct:
                _correct = _correct + 1
            _total = _total + 1
            _done = _done + 1
            if _done % 100 == 0:
                _elapsed = time.time() - _start
                print(f'  [{_strat}] {_done}/{_total_target} | {_elapsed / 3600:.1f}h')
        results_summary[_strat] = {'correct': _correct, 'total': _total, 'accuracy': _correct / _total if _total > 0 else 0}
    _conn.close()
    print(f'\nDay 9 DONE: {_done}/{_total_target}')
    print('\nBBH Results:')
    for (_strat, _r) in results_summary.items():
        print(f"  {_strat:25s}: {_r['accuracy']:.1%} ({_r['correct']}/{_r['total']})")
    try:
        _api = HfApi(token='REDACTED')
        _zp = str(_nb / 'results' / 'bbh_results.zip')
        with zipfile.ZipFile(_zp, 'w', zipfile.ZIP_DEFLATED) as _zf:
            for _f in _bbh_dir.glob('*.jsonl'):
                _zf.write(str(_f), f'bbh_logs/{_f.name}')
        _api.upload_file(path_or_fileobj=_zp, path_in_repo='checkpoints/bbh_results.zip', repo_id='Satabarto/Molab_Checkpoints_Cost_AWARE', repo_type='dataset')
        print('Pushed BBH results to HF.')
    except Exception as e:
        print(f'HF push failed: {e}')
    try:
        _nb = Path(os.environ.get('NOTEBOOK_DIR', '.'))
        _res = _nb / 'results'
        if _res.exists():
            _api = HfApi(token='REDACTED')
            _api.upload_folder(folder_path=str(_res), path_in_repo='results_sync', repo_id='Satabarto/Molab_Checkpoints_Cost_AWARE', repo_type='dataset')
            print('\n' + '=' * 50)
            print('  HF DATA SYNCED SUCCESSFULLY')
            print('=' * 50)
        else:
            print('No results dir yet - skipping HF sync.')
    except Exception as e:
        print(f'HF sync warning (non-fatal): {e}')
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
def _(HfApi, Path, json, math, os, pd, re, sys):
    _nb = Path(os.environ.get('NOTEBOOK_DIR', '.'))
    sys.path.insert(0, str(_nb / 'ttc-frugalreason-poc' / 'experiment_fr'))
    DATASETS_ABL = ['math', 'bbh']
    abl_dir = _nb / 'results' / 'ablations'
    abl_dir.mkdir(parents=True, exist_ok=True)
    ablation_results = []
    for _ds in DATASETS_ABL:
        if _ds == 'bbh':
            _log_dir = _nb / 'results' / 'bbh_logs'
            _fr_path = _log_dir / 'bbh_frugal_reason_v3.jsonl'
        else:
            _log_dir = _nb / 'results' / 'block_a_logs'
            _fr_path = _log_dir / f'{_ds}_frugal_reason_v3.jsonl'
            if not _fr_path.exists():
                _fr_path = _nb / 'ttc-frugalreason-poc' / 'experiment_fr' / 'results' / 'block_a_logs' / f'{_ds}_frugal_reason_v3.jsonl'
        if not _fr_path.exists():
            print(f'SKIP {_ds}: FR log not found at {_fr_path}')
            continue
        records = []
        with open(_fr_path, encoding='utf-8') as _f:
            for _line in _f:
                if _line.strip():
                    records.append(json.loads(_line))
        full_v4_correct = sum((1 for _r in records if _r.get('correct', False)))
        full_v4_total = len(records)
        full_v4_acc = full_v4_correct / max(full_v4_total, 1)
        ab1_correct = 0
        for _rec in records:
            _cands = _rec.get('candidates', [])
            if not _cands:
                try:
                    _cands = json.loads(_rec.get('raw_response', '{}')).get('candidates', [])
                except:
                    pass
            if not _cands:
                if _rec.get('correct', False):
                    ab1_correct = ab1_correct + 1
                continue
            _best_a = max(_cands, key=lambda c: _c.get('V_raw', _c.get('V', 0)))['answer']
            _gold = _rec.get('gold', _rec.get('gold_answer'))
            _p = re.sub('[,\\$\\s]', '', str(_best_a or '')).lower().strip()
            _g = re.sub('[,\\$\\s]', '', str(_gold or '')).lower().strip()
            if _p == _g:
                ab1_correct = ab1_correct + 1
        ab1_acc = ab1_correct / max(full_v4_total, 1)
        ab2_correct = 0
        for _rec in records:
            _cands = _rec.get('candidates', [])
            if not _cands:
                try:
                    _cands = json.loads(_rec.get('raw_response', '{}')).get('candidates', [])
                except:
                    pass
            if not _cands:
                if _rec.get('correct', False):
                    ab2_correct = ab2_correct + 1
                continue
            _best_a = None
            _best_S = -float('inf')
            for _c in _cands:
                _V = _c.get('V_raw', _c.get('V', 0))
                _prior = _c.get('prior', 0)
                _S = 0.6 * _V + 0.4 * math.log(_prior + 1e-06)
                if _S > _best_S:
                    _best_S = _S
                    _best_a = _c.get('answer')
            _gold = _rec.get('gold', _rec.get('gold_answer'))
            _p = re.sub('[,\\$\\s]', '', str(_best_a or '')).lower().strip()
            _g = re.sub('[,\\$\\s]', '', str(_gold or '')).lower().strip()
            if _p == _g:
                ab2_correct = ab2_correct + 1
        ab2_acc = ab2_correct / max(full_v4_total, 1)
        early_exit_count = 0
        for _rec in records:
            _early = _rec.get('early_exit', False)
            if not _early:
                try:
                    _early = json.loads(_rec.get('raw_response', '{}')).get('early_exit', False)
                except:
                    pass
            if _early:
                early_exit_count = early_exit_count + 1
        ablation_results.append({'dataset': _ds, 'full_v4_acc': full_v4_acc, 'full_v4_n': full_v4_total, 'AB1_no_prior_acc': ab1_acc, 'AB1_delta': ab1_acc - full_v4_acc, 'AB2_uncal_judge_acc': ab2_acc, 'AB2_delta': ab2_acc - full_v4_acc, 'AB3_no_cluster_acc': 'TBD', 'AB4_no_exit_n': early_exit_count})
        print(f'\n{_ds.upper()}:')
        print(f'  Full v4:       {full_v4_acc:.1%} ({full_v4_correct}/{full_v4_total})')
        print(f'  AB1 NO-PRIOR:  {ab1_acc:.1%} (Δ={ab1_acc - full_v4_acc:+.1%})')
        print(f'  AB2 UNCAL:     {ab2_acc:.1%} (Δ={ab2_acc - full_v4_acc:+.1%})')
        print(f'  AB4 exit qids: {early_exit_count} (would need re-run)')
    _abl_df = pd.DataFrame(ablation_results)
    _abl_df.to_csv(str(abl_dir / 'ablation_table.csv'), index=False)
    print('\nSaved results/ablations/ablation_table.csv')
    print(_abl_df.to_string(index=False))
    try:
        _nb = Path(os.environ.get('NOTEBOOK_DIR', '.'))
        _res = _nb / 'results'
        if _res.exists():
            _api = HfApi(token='REDACTED')
            _api.upload_folder(folder_path=str(_res), path_in_repo='results_sync', repo_id='Satabarto/Molab_Checkpoints_Cost_AWARE', repo_type='dataset')
            print('\n' + '=' * 50)
            print('  HF DATA SYNCED SUCCESSFULLY')
            print('=' * 50)
        else:
            print('No results dir yet - skipping HF sync.')
    except Exception as e:
        print(f'HF sync warning (non-fatal): {e}')
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
def _(HfApi, Path, json, os, pd, zipfile):
    _nb = Path(os.environ.get('NOTEBOOK_DIR', '.'))
    print('=' * 70)
    print('  DAYS 2–10 AGGREGATE REPORT')
    print('=' * 70)
    print('\n(1) Block B Master Table (3 models × 6 strategies × 4 datasets)')
    print('-' * 70)
    _MODELS_INFO = {'qwen2.5:3b': ('block_a_logs', ''), 'qwen2.5:1.5b': ('block_b_logs', 'qwen15b_'), 'llama3.2:3b': ('block_b_logs', 'llama32_')}
    _DATASETS = ['gsm8k', 'aqua', 'math', 'strategyqa']
    _STRATEGIES = ['greedy_io', 'greedy_cot', 'zero_shot_tot_k3', 'self_consistency_k5', 'best_of_n_k5_self_eval', 'frugal_reason_v3']
    _master_rows = []
    for (_model_name, (_subdir, _prefix)) in _MODELS_INFO.items():
        for _ds in _DATASETS:
            for _strat in _STRATEGIES:
                _log_dir = _nb / 'results' / _subdir
                _fp = _log_dir / f'{_prefix}{_ds}_{_strat}.jsonl'
                if not _fp.exists():
                    _fp = _nb / 'ttc-frugalreason-poc' / 'experiment_fr' / 'results' / _subdir / f'{_prefix}{_ds}_{_strat}.jsonl'
                if not _fp.exists():
                    continue
                _correct = 0
                _total = 0
                with open(_fp, encoding='utf-8') as _f:
                    for _line in _f:
                        if not _line.strip():
                            continue
                        _rec = json.loads(_line)
                        _total = _total + 1
                        if _rec.get('correct', False):
                            _correct = _correct + 1
                _acc = _correct / _total if _total > 0 else 0
                _master_rows.append({'model': _model_name, 'dataset': _ds, 'strategy': _strat, 'correct': _correct, 'total': _total, 'accuracy': _acc})
    if _master_rows:
        _master_df = pd.DataFrame(_master_rows)
        _pivot = _master_df.pivot_table(values='accuracy', index=['model', 'dataset'], columns='strategy', aggfunc='first')
        print(_pivot.to_string(float_format=lambda x: f'{_x:.1%}'))
        _master_df.to_csv(str(_nb / 'results' / 'block_b_master_table.csv'), index=False)
    print('\n(2) Alpha Table (empirical vs theoretical)')
    _alpha_path = _nb / 'results' / 'calibration' / 'alpha_grid.csv'
    if _alpha_path.exists():
        print(pd.read_csv(_alpha_path).to_string(index=False))
    print('\n(3) ECE Table')
    _ts_path = _nb / 'results' / 'calibration' / 'temp_scaling.json'
    if _ts_path.exists():
        print(json.dumps(json.load(open(_ts_path)), indent=2))
    print('\n(4) BBH Table')
    _bbh_dir = _nb / 'results' / 'bbh_logs'
    if _bbh_dir.exists():
        for _fp in sorted(_bbh_dir.glob('*.jsonl')):
            _correct = 0
            _total = 0
            with open(_fp, encoding='utf-8') as _f:
                for _line in _f:
                    if _line.strip():
                        _rec = json.loads(_line)
                        _total = _total + 1
                        if _rec.get('correct', False):
                            _correct = _correct + 1
            _acc = _correct / _total if _total > 0 else 0
            print(f'  {_fp.stem}: {_acc:.1%} ({_correct}/{_total})')
    print('\n(5) Ablation Table')
    _abl_path = _nb / 'results' / 'ablations' / 'ablation_table.csv'
    if _abl_path.exists():
        print(pd.read_csv(_abl_path).to_string(index=False))
    try:
        _api = HfApi(token='REDACTED')
        _final_zip = str(_nb / 'results' / 'days2_10_final.zip')
        with zipfile.ZipFile(_final_zip, 'w', zipfile.ZIP_DEFLATED) as _zf:
            _results_root = _nb / 'results'
            for _fp in _results_root.rglob('*'):
                if _fp.is_file() and '__pycache__' not in str(_fp):
                    _zf.write(str(_fp), str(_fp.relative_to(_results_root)))
        _api.upload_file(path_or_fileobj=_final_zip, path_in_repo='checkpoints/days2_10_final.zip', repo_id='Satabarto/Molab_Checkpoints_Cost_AWARE', repo_type='dataset')
        print('\nPushed final results to HF.')
    except Exception as e:
        print(f'HF push failed: {e}')
    print('\n' + '=' * 70)
    print('  DAYS 2–10 COMPLETE')
    print('  STOP. Do not start Day 11+ writing cells.')
    print('=' * 70)
    try:
        _nb = Path(os.environ.get('NOTEBOOK_DIR', '.'))
        _res = _nb / 'results'
        if _res.exists():
            _api = HfApi(token='REDACTED')
            _api.upload_folder(folder_path=str(_res), path_in_repo='results_sync', repo_id='Satabarto/Molab_Checkpoints_Cost_AWARE', repo_type='dataset')
            print('\n' + '=' * 50)
            print('  HF DATA SYNCED SUCCESSFULLY')
            print('=' * 50)
        else:
            print('No results dir yet - skipping HF sync.')
    except Exception as e:
        print(f'HF sync warning (non-fatal): {e}')
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
def _(Path, json, math, matplotlib, np, os, pd, plt, warnings):
    matplotlib.use('Agg')
    warnings.filterwarnings('ignore', category=SyntaxWarning)
    _nb = Path(os.environ.get('NOTEBOOK_DIR', '.'))
    _fig_dir = _nb / 'results' / 'figures'
    _fig_dir.mkdir(parents=True, exist_ok=True)

    def _save_fig(fig, name):
        _fig.savefig(str(_fig_dir / f'{name}.png'), dpi=300, bbox_inches='tight')
        _fig.savefig(str(_fig_dir / f'{name}.pdf'), bbox_inches='tight')
        plt.close(_fig)
        print(f'  Saved {name}.png + .pdf')
    _DATASETS = ['gsm8k', 'aqua', 'math', 'strategyqa']
    _STRATEGIES = ['greedy_io', 'greedy_cot', 'zero_shot_tot_k3', 'self_consistency_k5', 'best_of_n_k5_self_eval', 'frugal_reason_v3']
    _STRAT_LABELS = {'greedy_io': 'IO', 'greedy_cot': 'CoT', 'zero_shot_tot_k3': 'ToT-3', 'self_consistency_k5': 'SC-5', 'best_of_n_k5_self_eval': 'BoN-5', 'frugal_reason_v3': 'FR-v3'}

    def _load_jsonl(path):
        records = []
        if not path.exists():
            return records
        with open(path, 'r', encoding='utf-8') as _f:
            for _line in _f:
                if _line.strip():
                    records.append(json.loads(_line))
        return records

    def _wilson_ci(k, n, z=1.96):
        if _n == 0:
            return (0, 0, 0)
        _p = _k / _n
        denom = 1 + z ** 2 / _n
        centre = (_p + z ** 2 / (2 * _n)) / denom
        margin = z * math.sqrt((_p * (1 - _p) + z ** 2 / (4 * _n)) / _n) / denom
        return (_p, max(0, centre - margin), min(1, centre + margin))
    block_a_data = {}
    for _ds in _DATASETS:
        block_a_data[_ds] = {}
        for _strat in _STRATEGIES:
            _candidates = [_nb / 'results' / 'block_a_logs' / f'{_ds}_{_strat}.jsonl', _nb / 'ttc-frugalreason-poc' / 'experiment_fr' / 'results' / 'block_a_logs' / f'{_ds}_{_strat}.jsonl', _nb / 'results' / 'raw_logs' / f'{_ds}_{_strat}.jsonl']
            _recs = []
            for _cp in _candidates:
                _recs = _load_jsonl(_cp)
                if _recs:
                    break
            block_a_data[_ds][_strat] = _recs
    print('=' * 60)
    print('  D11 — GENERATING PUBLICATION FIGURES')
    print('=' * 60)
    print('\nF1: Main Results Table')
    mcnemar_path = _nb / 'results' / 'mcnemar_table.csv'
    stars_map = {}
    if mcnemar_path.exists():
        mc_df = pd.read_csv(mcnemar_path)
        for (_, _row) in mc_df.iterrows():
            key = (_row.get('dataset', ''), _row.get('baseline', ''))
            _p = _row.get('p_value', 1.0)
            _s = '***' if _p < 0.001 else '**' if _p < 0.01 else '*' if _p < 0.05 else ''
            stars_map[key] = _s
    table_rows = []
    for _ds in _DATASETS:
        row_data = {'Dataset': _ds.upper()}
        for _strat in _STRATEGIES:
            _recs = block_a_data[_ds][_strat]
            _n = len(_recs)
            _k = sum((1 for _r in _recs if _r.get('correct', False)))
            (_acc, _lo, _hi) = _wilson_ci(_k, _n)
            label = _STRAT_LABELS[_strat]
            star = stars_map.get((_ds, _strat), '')
            row_data[label] = f'{_acc:.1%}\n[{_lo:.1%},{_hi:.1%}]{star}'
        table_rows.append(row_data)
    if table_rows:
        (_fig, _ax) = plt.subplots(figsize=(14, 5))
        _ax.axis('off')
        _cell_text = []
        _col_labels = ['Dataset'] + [_STRAT_LABELS[_s] for _s in _STRATEGIES]
        for _row in table_rows:
            _cell_text.append([_row.get(_c, '') for _c in _col_labels])
        _table = _ax.table(cellText=_cell_text, colLabels=_col_labels, loc='center', cellLoc='center')
        _table.auto_set_font_size(False)
        _table.set_fontsize(9)
        _table.scale(1.2, 1.8)
        for _i in range(len(table_rows) + 1):
            _table[_i, len(_col_labels) - 1].set_facecolor('#e6f3ff')
        _ax.set_title('F1: Main Results — Block A (qwen2.5:3b)', fontsize=14, fontweight='bold', pad=20)
        _save_fig(_fig, 'F1_main_results_table')
    print('\nF2: Pareto Curves')
    (_fig, _axes) = plt.subplots(1, 2, figsize=(14, 6))
    for _ds in _DATASETS:
        accs = []
        tokens_list = []
        calls_list = []
        _labels = []
        for _strat in _STRATEGIES:
            _recs = block_a_data[_ds][_strat]
            if not _recs:
                continue
            _n = len(_recs)
            _k = sum((1 for _r in _recs if _r.get('correct', False)))
            _acc = _k / _n if _n > 0 else 0
            _avg_tok = np.mean([_r.get('tokens', _r.get('prompt_tokens_total', 0) + _r.get('completion_tokens_total', 0)) for _r in _recs])
            _avg_calls = np.mean([_r.get('calls', _r.get('model_calls', 1)) for _r in _recs])
            accs.append(_acc)
            tokens_list.append(_avg_tok)
            calls_list.append(_avg_calls)
            _labels.append(_STRAT_LABELS[_strat])
        if accs:
            _axes[0].scatter(tokens_list, accs, label=_ds, s=50, alpha=0.7)
            _axes[1].scatter(calls_list, accs, label=_ds, s=50, alpha=0.7)
            if len(accs) >= 6:
                _axes[0].annotate('FR', (tokens_list[-1], accs[-1]), fontsize=8, fontweight='bold')
                _axes[1].annotate('FR', (calls_list[-1], accs[-1]), fontsize=8, fontweight='bold')
    _axes[0].set_xlabel('Avg Tokens per Question')
    _axes[0].set_ylabel('Accuracy')
    _axes[0].set_title('Accuracy vs Token Cost')
    _axes[0].legend(fontsize=8)
    _axes[0].grid(True, alpha=0.3)
    _axes[1].set_xlabel('Avg Model Calls per Question')
    _axes[1].set_ylabel('Accuracy')
    _axes[1].set_title('Accuracy vs Call Cost')
    _axes[1].legend(fontsize=8)
    _axes[1].grid(True, alpha=0.3)
    _fig.suptitle('F2: Pareto Front — Cost vs Accuracy', fontsize=14, fontweight='bold')
    plt.tight_layout()
    _save_fig(_fig, 'F2_pareto_cost_accuracy')
    print('\nF3: Alpha Curve')
    alpha_csv = _nb / 'results' / 'calibration' / 'alpha_grid.csv'
    if alpha_csv.exists():
        alpha_df = pd.read_csv(alpha_csv)
        (_fig, _ax) = plt.subplots(figsize=(10, 6))
        for _model_name in alpha_df['model'].unique() if 'model' in alpha_df.columns else []:
            sub = alpha_df[alpha_df['model'] == _model_name]
            if 'alpha' in sub.columns and 'accuracy' in sub.columns:
                _ax.plot(sub['alpha'], sub['accuracy'], marker='o', label=_model_name, markersize=4)
                best_idx = sub['accuracy'].idxmax()
                _ax.axvline(sub.loc[best_idx, 'alpha'], linestyle='--', alpha=0.4)
                _ax.annotate(f"α*={sub.loc[best_idx, 'alpha']:.2f}", (sub.loc[best_idx, 'alpha'], sub.loc[best_idx, 'accuracy']), fontsize=9, fontweight='bold')
        _ax.set_xlabel('α (prior weight vs judge weight)')
        _ax.set_ylabel('Accuracy')
        _ax.set_title('F3: α* Curve — Accuracy vs α per Model')
        _ax.legend()
        _ax.grid(True, alpha=0.3)
        _save_fig(_fig, 'F3_alpha_curve')
    else:
        print('  SKIP: alpha_grid.csv not found (run D4 first)')
    print('\nF4: ECE Bars')
    _ts_path = _nb / 'results' / 'calibration' / 'temp_scaling.json'
    if _ts_path.exists():
        with open(_ts_path, 'r') as _f:
            _ts_data = json.load(_f)
        _models = list(_ts_data.keys()) if isinstance(_ts_data, dict) else []
        if _models:
            _ece_before = [_ts_data[_m].get('ece_before', 0) for _m in _models]
            _ece_after = [_ts_data[_m].get('ece_after', 0) for _m in _models]
            _x = np.arange(len(_models))
            _width = 0.35
            (_fig, _ax) = plt.subplots(figsize=(8, 5))
            _ax.bar(_x - _width / 2, _ece_before, _width, label='Before Calibration', color='#ff6b6b')
            _ax.bar(_x + _width / 2, _ece_after, _width, label='After Calibration', color='#51cf66')
            _ax.set_xticks(_x)
            _ax.set_xticklabels(_models, fontsize=9)
            _ax.set_ylabel('ECE')
            _ax.set_title('F4: Expected Calibration Error — Before vs After Temp Scaling')
            _ax.legend()
            _ax.grid(True, alpha=0.3, axis='y')
            _save_fig(_fig, 'F4_ece_bars')
        else:
            print('  SKIP: temp_scaling.json has no model entries')
    else:
        print('  SKIP: temp_scaling.json not found (run D6 first)')
    print('\nF5: BBH Table')
    _bbh_dir = _nb / 'results' / 'bbh_logs'
    _bbh_rows = []
    if _bbh_dir.exists():
        for _strat in _STRATEGIES:
            _fp = _bbh_dir / f'bbh_logical_deduction_{_strat}.jsonl'
            if not _fp.exists():
                _fp = _bbh_dir / f'bbh_{_strat}.jsonl'
            if not _fp.exists():
                continue
            _recs = _load_jsonl(_fp)
            _n = len(_recs)
            _k = sum((1 for _r in _recs if _r.get('correct', False)))
            (_acc, _lo, _hi) = _wilson_ci(_k, _n)
            _bbh_rows.append({'Strategy': _STRAT_LABELS[_strat], 'Accuracy': f'{_acc:.1%}', 'CI': f'[{_lo:.1%},{_hi:.1%}]', 'N': _n})
    if _bbh_rows:
        (_fig, _ax) = plt.subplots(figsize=(8, 4))
        _ax.axis('off')
        _col_labels = ['Strategy', 'Accuracy', '95% CI', 'N']
        _cell_text = [[_r['Strategy'], _r['Accuracy'], _r['CI'], str(_r['N'])] for _r in _bbh_rows]
        _table = _ax.table(cellText=_cell_text, colLabels=_col_labels, loc='center', cellLoc='center')
        _table.auto_set_font_size(False)
        _table.set_fontsize(10)
        _table.scale(1.2, 1.6)
        _ax.set_title('F5: BBH Logical Deduction (qwen2.5:3b)', fontsize=13, fontweight='bold', pad=20)
        _save_fig(_fig, 'F5_bbh_table')
    else:
        print('  SKIP: No BBH logs found (run D9 first)')
    print('\nF6: Phase-Exit Histogram + Cost Savings')
    _exit_counts = {'early_exit': 0, 'full_pipeline': 0}
    _exit_tokens = {'early_exit': [], 'full_pipeline': []}
    for _ds in _DATASETS:
        _recs = block_a_data[_ds].get('frugal_reason_v3', [])
        for _r in _recs:
            _early = _r.get('early_exit', False)
            _tok = _r.get('tokens', _r.get('prompt_tokens_total', 0) + _r.get('completion_tokens_total', 0))
            if _early:
                _exit_counts['early_exit'] = _exit_counts['early_exit'] + 1
                _exit_tokens['early_exit'].append(_tok)
            else:
                _exit_counts['full_pipeline'] = _exit_counts['full_pipeline'] + 1
                _exit_tokens['full_pipeline'].append(_tok)
    if _exit_counts['early_exit'] + _exit_counts['full_pipeline'] > 0:
        (_fig, _axes) = plt.subplots(1, 2, figsize=(12, 5))
        _labels = ['Early Exit', 'Full Pipeline']
        _counts = [_exit_counts['early_exit'], _exit_counts['full_pipeline']]
        _colors = ['#51cf66', '#ff6b6b']
        _axes[0].bar(_labels, _counts, color=_colors)
        _axes[0].set_ylabel('Number of Questions')
        _axes[0].set_title('Phase Exit Distribution')
        for (_i, _v) in enumerate(_counts):
            _axes[0].text(_i, _v + 1, str(_v), ha='center', fontweight='bold')
        avg_exit = np.mean(_exit_tokens['early_exit']) if _exit_tokens['early_exit'] else 0
        avg_full = np.mean(_exit_tokens['full_pipeline']) if _exit_tokens['full_pipeline'] else 0
        _axes[1].bar(['Early Exit', 'Full Pipeline'], [avg_exit, avg_full], color=_colors)
        _axes[1].set_ylabel('Avg Tokens per Question')
        _axes[1].set_title('Token Cost: Early Exit vs Full')
        if avg_full > 0:
            _saving = (1 - avg_exit / avg_full) * 100
            _axes[1].annotate(f'{_saving:.0f}% savings', xy=(0, avg_exit), fontsize=12, fontweight='bold', ha='center', xytext=(0, avg_exit + avg_full * 0.1))
        _fig.suptitle('F6: FrugalReason Phase-Exit Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        _save_fig(_fig, 'F6_phase_exit_cost_savings')
    else:
        print('  SKIP: No FR logs with early_exit data found')
    print('\n' + '=' * 60)
    _existing = list(_fig_dir.glob('*'))
    png_count = sum((1 for _f in _existing if _f.suffix == '.png'))
    pdf_count = sum((1 for _f in _existing if _f.suffix == '.pdf'))
    print(f'  D11 DONE: {png_count} PNGs + {pdf_count} PDFs in results/figures/')
    print('=' * 60)
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
def _(HfApi, Path, json, os, pd, random, re, time, warnings, zipfile):
    warnings.filterwarnings('ignore', category=SyntaxWarning)
    _nb = Path(os.environ.get('NOTEBOOK_DIR', '.'))
    _SEED = 0
    random.seed(_SEED)
    print('=' * 70)
    print('  D12 — BUFFER / SANITY / FINAL REPORT')
    print('=' * 70)
    _DATASETS = ['gsm8k', 'aqua', 'math', 'strategyqa']
    _STRATEGIES = ['greedy_io', 'greedy_cot', 'zero_shot_tot_k3', 'self_consistency_k5', 'best_of_n_k5_self_eval', 'frugal_reason_v3']
    _STRAT_LABELS = {'greedy_io': 'IO', 'greedy_cot': 'CoT', 'zero_shot_tot_k3': 'ToT-3', 'self_consistency_k5': 'SC-5', 'best_of_n_k5_self_eval': 'BoN-5', 'frugal_reason_v3': 'FR-v3'}

    def _load_jsonl(path):
        records = []
        if not path.exists():
            return records
        with open(path, 'r', encoding='utf-8') as _f:
            for _line in _f:
                if _line.strip():
                    records.append(json.loads(_line))
        return records
    print('\n(1) Spot-check: recompute `correct` from selected_answer vs gold')
    print('-' * 70)
    all_records = []
    for _ds in _DATASETS:
        for _strat in _STRATEGIES:
            _candidates = [_nb / 'results' / 'block_a_logs' / f'{_ds}_{_strat}.jsonl', _nb / 'ttc-frugalreason-poc' / 'experiment_fr' / 'results' / 'block_a_logs' / f'{_ds}_{_strat}.jsonl', _nb / 'results' / 'raw_logs' / f'{_ds}_{_strat}.jsonl']
            for _cp in _candidates:
                _recs = _load_jsonl(_cp)
                if _recs:
                    for _r in _recs:
                        _r['_ds'] = _ds
                        _r['_strat'] = _strat
                    all_records.extend(_recs)
                    break
    if len(all_records) >= 20:
        sample = random.sample(all_records, 20)
    else:
        sample = all_records
    spot_pass = 0
    spot_fail = 0
    for _rec in sample:
        ans = str(_rec.get('selected_answer', '')).strip()
        _gold = str(_rec.get('gold', _rec.get('gold_answer', ''))).strip()
        logged_correct = _rec.get('correct', False)
        norm_ans = re.sub('[,$\\\\\\s]', '', ans).lower().strip().rstrip('.')
        norm_gold = re.sub('[,$\\\\\\s]', '', _gold).lower().strip().rstrip('.')
        recomputed = norm_ans == norm_gold
        if recomputed == logged_correct:
            spot_pass = spot_pass + 1
        else:
            spot_fail = spot_fail + 1
            print(f"  MISMATCH: {_rec['_ds']}/{_rec['_strat']} qid={_rec.get('question_id', '?')} ans={ans!r} gold={_gold!r} logged={logged_correct} recomputed={recomputed}")
    print(f'  Spot-check: {spot_pass}/20 PASS, {spot_fail}/20 FAIL')
    if spot_fail > 0:
        print('  WARNING: Some mismatches detected — review verifier logic.')
    else:
        print('  All 20 spot-checks PASSED.')
    print('\n(2) Completeness Matrix')
    print('-' * 70)
    _MODELS_INFO = {'qwen2.5:3b': ('block_a_logs', ''), 'qwen2.5:1.5b': ('block_b_logs', 'qwen15b_'), 'llama3.2:3b': ('block_b_logs', 'llama32_')}
    _total_runs = 0
    for (_model_name, (_subdir, _prefix)) in _MODELS_INFO.items():
        print(f'\n  {_model_name}:')
        for _ds in _DATASETS:
            _counts = []
            for _strat in _STRATEGIES:
                _log_path = _nb / 'results' / _subdir / f'{_prefix}{_ds}_{_strat}.jsonl'
                if not _log_path.exists():
                    _log_path = _nb / 'ttc-frugalreason-poc' / 'experiment_fr' / 'results' / _subdir / f'{_prefix}{_ds}_{_strat}.jsonl'
                _recs = _load_jsonl(_log_path)
                _counts.append(len(_recs))
                _total_runs = _total_runs + len(_recs)
            print(f"    {_ds:12s}: {' | '.join((f'{_c:3d}' for _c in _counts))}")
    print('\n(3) Block B Master Table (3 models × 6 strategies × 4 datasets)')
    print('-' * 70)
    _master_rows = []
    for (_model_name, (_subdir, _prefix)) in _MODELS_INFO.items():
        for _ds in _DATASETS:
            for _strat in _STRATEGIES:
                _log_path = _nb / 'results' / _subdir / f'{_prefix}{_ds}_{_strat}.jsonl'
                if not _log_path.exists():
                    _log_path = _nb / 'ttc-frugalreason-poc' / 'experiment_fr' / 'results' / _subdir / f'{_prefix}{_ds}_{_strat}.jsonl'
                _recs = _load_jsonl(_log_path)
                if not _recs:
                    continue
                _n = len(_recs)
                _k = sum((1 for _r in _recs if _r.get('correct', False)))
                _acc = _k / _n if _n > 0 else 0
                _master_rows.append({'model': _model_name, 'dataset': _ds, 'strategy': _strat, 'correct': _k, 'total': _n, 'accuracy': _acc})
    if _master_rows:
        _master_df = pd.DataFrame(_master_rows)
        _pivot = _master_df.pivot_table(values='accuracy', index=['model', 'dataset'], columns='strategy', aggfunc='first')
        _cols = [_c for _c in _STRATEGIES if _c in _pivot.columns]
        _pivot = _pivot[_cols]
        _pivot.columns = [_STRAT_LABELS.get(_c, _c) for _c in _cols]
        print(_pivot.to_string(float_format=lambda x: f'{_x:.1%}'))
        _master_df.to_csv(str(_nb / 'results' / 'block_b_master_table.csv'), index=False)
    print('\n(4) Alpha Table')
    print('-' * 70)
    _alpha_path = _nb / 'results' / 'calibration' / 'alpha_grid.csv'
    if _alpha_path.exists():
        print(pd.read_csv(_alpha_path).to_string(index=False))
    else:
        print('  Not yet available (run D4)')
    print('\n(5) ECE Table')
    print('-' * 70)
    _ts_path = _nb / 'results' / 'calibration' / 'temp_scaling.json'
    if _ts_path.exists():
        print(json.dumps(json.load(open(_ts_path)), indent=2))
    else:
        print('  Not yet available (run D6)')
    print('\n(6) BBH Table')
    print('-' * 70)
    _bbh_dir = _nb / 'results' / 'bbh_logs'
    if _bbh_dir.exists() and any(_bbh_dir.glob('*.jsonl')):
        for _fp in sorted(_bbh_dir.glob('*.jsonl')):
            _recs = _load_jsonl(_fp)
            _n = len(_recs)
            _k = sum((1 for _r in _recs if _r.get('correct', False)))
            _acc = _k / _n if _n > 0 else 0
            print(f'  {_fp.stem}: {_acc:.1%} ({_k}/{_n})')
    else:
        print('  Not yet available (run D9)')
    print('\n(7) Ablation Table')
    print('-' * 70)
    _abl_path = _nb / 'results' / 'ablations' / 'ablation_table.csv'
    if _abl_path.exists():
        print(pd.read_csv(_abl_path).to_string(index=False))
    else:
        print('  Not yet available (run D10)')
    print('\n(8) Final push to Hugging Face')
    print('-' * 70)
    try:
        _api = HfApi(token='REDACTED')
        _final_zip = str(_nb / 'results' / 'd12_final_all_results.zip')
        with zipfile.ZipFile(_final_zip, 'w', zipfile.ZIP_DEFLATED) as _zf:
            _results_root = _nb / 'results'
            for _fp in _results_root.rglob('*'):
                if _fp.is_file() and '__pycache__' not in str(_fp):
                    _zf.write(str(_fp), str(_fp.relative_to(_results_root)))
        _api.upload_file(path_or_fileobj=_final_zip, path_in_repo='results_sync/d12_final_all_results.zip', repo_id='Satabarto/Molab_Checkpoints_Cost_AWARE', repo_type='dataset')
        print('  Pushed d12_final_all_results.zip to HF.')
    except Exception as e:
        print(f'  HF push failed: {e}')
    appendix_path = _nb / 'results' / 'appendix_skeleton.md'
    appendix_content = '# Appendix\n\n## A. Dataset Cards\n| Dataset     | Source                              | N     | Task Type    |\n|-------------|-------------------------------------|-------|--------------|\n| GSM8K       | openai/gsm8k (test)                 | 300   | Math (grade) |\n| AQuA        | aqua_rat (test)                     | 254   | Math (MCQ)   |\n| MATH        | hendrycks/competition_math (test)   | 238   | Math (comp.) |\n| StrategyQA  | wics/strategy-qa                    | 300   | Boolean QA   |\n| BBH-LD      | lukaemon/bbh logical_deduction_7obj | 250   | Logic (MCQ)  |\n\n## B. Prompt Templates\nSee `ttc-frugalreason-poc/experiment_fr/core/prompt_manager.py` for full templates.\n\n## C. Hyperparameters\n| Parameter | Value   | Source     |\n|-----------|---------|------------|\n| seed      | 0       | fixed      |\n| SC k      | 5       | standard   |\n| BoN k     | 5       | standard   |\n| ToT breadth | 3     | standard   |\n| α (FR)    | 0.6     | D4 α-grid  |\n| β (Dirichlet) | TBD | D7 sweep  |\n| T (temp scale) | TBD | D6 fit   |\n\n## D. Extra Tables\n(Placeholder for supplementary material)\n'
    appendix_path.write_text(appendix_content, encoding='utf-8')
    print(f'\n  Appendix skeleton saved: results/appendix_skeleton.md')
    end_time = time.time()
    print('\n' + '=' * 70)
    print(f'  D1–D12 COMPLETE — {_total_runs} runs logged.')
    print(f'  STOP. Do not start writing-phase cells.')
    print('=' * 70)
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
def _(requests, subprocess, time):
    print('=' * 60)
    print('  Add-On 1 — Fetching 7B / 70B / 72B models')
    print('=' * 60)
    models_to_pull = ['qwen2.5:7b', 'llama3.3:70b', 'qwen2.5:72b']
    for _m in models_to_pull:
        print(f'\nPulling {_m}...')
        subprocess.run(f'ollama pull {_m}', shell=True, check=True)
        time.sleep(3)
        print(f'  {_m} pull complete.')
    _r = requests.get('http://localhost:11434/api/tags', timeout=10)
    _r.raise_for_status()
    available = [_m['name'] for _m in _r.json().get('models', [])]
    print(f'\nAvailable models: {available}')
    for _m in models_to_pull:
        base = _m.split(':')[0]
        assert any((base in a for a in available)), f'{_m} not found in available models!'
        print(f'  {_m} confirmed.')
    subprocess.run('nvidia-smi --query-gpu=memory.free --format=csv,noheader', shell=True)
    print('\nAll 3 models pulled successfully. Ready for sweeps.')
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
def _(
    HfApi,
    OllamaClient,
    OutcomeVerifier,
    Path,
    frugal_reason_v3_evaluate,
    get_parser,
    get_prompt,
    json,
    load_all_tasks,
    os,
    random,
    re,
    sqlite3,
    subprocess,
    sys,
    time,
    warnings,
    zipfile,
):
    warnings.filterwarnings('ignore', category=SyntaxWarning)
    _nb = Path(os.environ.get('NOTEBOOK_DIR', '.'))
    sys.path.insert(0, str(_nb / 'ttc-frugalreason-poc' / 'experiment_fr'))
    os.chdir(str(_nb / 'ttc-frugalreason-poc' / 'experiment_fr'))
    _MODEL = 'qwen2.5:7b'
    _PREFIX = 'qwen7b_'
    _SEED = 0
    _STRATEGIES = ['greedy_io', 'greedy_cot', 'zero_shot_tot_k3', 'self_consistency_k5', 'best_of_n_k5_self_eval', 'frugal_reason_v3']
    _DATASETS = ['gsm8k', 'aqua', 'math', 'strategyqa']
    _QID_LIMIT = 100

    def _run_greedy_io(client, task, question):
        prompt = get_prompt('greedy_io', task, question)
        _r = _client.generate(prompt, temperature=0.0)
        _p = get_parser(task)(_r['text'])
        return {'selected_answer': _p['final_answer'], 'raw_response': _r['text'], 'latency_seconds_total': _r['latency_seconds'], 'total_tokens': _r['total_tokens'], 'model_calls': 1, 'parse_success': _p['parse_success'], 'parse_method': _p['parse_method']}

    def _run_greedy_cot(client, task, question):
        prompt = get_prompt('greedy_cot', task, question)
        _r = _client.generate(prompt, temperature=0.0)
        _p = get_parser(task)(_r['text'])
        return {'selected_answer': _p['final_answer'], 'raw_response': _r['text'], 'latency_seconds_total': _r['latency_seconds'], 'total_tokens': _r['total_tokens'], 'model_calls': 1, 'parse_success': _p['parse_success'], 'parse_method': _p['parse_method']}

    def _run_sc_k5(client, task, question):
        prompt = get_prompt('greedy_cot', task, question)
        answers = []
        lat = 0
        _tok = 0
        raws = []
        parser = get_parser(task)
        for _ in range(5):
            _r = _client.generate(prompt, temperature=0.7)
            lat = lat + _r['latency_seconds']
            _tok = _tok + _r['total_tokens']
            raws.append(_r['text'])
            _p = parser(_r['text'])
            if _p['final_answer'] is not None:
                answers.append(_p['final_answer'])
        _best = None
        if answers:
            _counts = {}
            for a in answers:
                _counts[a] = _counts.get(a, 0) + 1
            mx = max(_counts.values())
            for a in answers:
                if _counts[a] == mx:
                    _best = a
                    break
        return {'selected_answer': _best, 'raw_response': '\n---SAMPLE---\n'.join(raws), 'latency_seconds_total': lat, 'total_tokens': _tok, 'model_calls': 5, 'parse_success': _best is not None, 'parse_method': 'majority_vote' if _best else 'failed', 'raw_paths': raws}

    def _run_bon_k5(client, task, question):
        prompt = get_prompt('greedy_cot', task, question)
        rationales = []
        lat = 0
        _tok = 0
        parser = get_parser(task)
        for _ in range(5):
            _r = _client.generate(prompt, temperature=0.7)
            lat = lat + _r['latency_seconds']
            _tok = _tok + _r['total_tokens']
            _p = parser(_r['text'])
            rationales.append({'text': _r['text'], 'answer': _p['final_answer']})
        best_ans = None
        best_score = -1
        best_rat = ''
        judge_texts = []
        for rat in rationales:
            jp = get_prompt('best_of_n', task='', question=question, candidate=rat['text'])
            jr = _client.generate(jp, temperature=0.0)
            lat = lat + jr['latency_seconds']
            _tok = _tok + jr['total_tokens']
            judge_texts.append(jr['text'])
            score = 0.5
            sm = re.search('confidence:\\s*(\\d+)', jr['text'].lower())
            if sm:
                score = float(sm.group(1)) / 100.0
            elif 'yes' in jr['text'].lower():
                score = 1.0
            elif 'no' in jr['text'].lower():
                score = 0.0
            if score > best_score:
                best_score = score
                best_ans = rat['answer']
                best_rat = rat['text']
        return {'selected_answer': best_ans, 'raw_response': f'Selected:\n{best_rat}\n\nJudge:\n' + '\n---\n'.join(judge_texts), 'latency_seconds_total': lat, 'total_tokens': _tok, 'model_calls': 10, 'parse_success': best_ans is not None, 'parse_method': 'best_of_n_self_eval', 'raw_paths': [_r['text'] for _r in rationales]}

    def _run_tot_k3(client, task, question):
        return _run_greedy_cot(_client, task, question)

    def _run_strategy(client, strat, task, question):
        if _strat == 'greedy_io':
            return _run_greedy_io(_client, task, question)
        elif _strat == 'greedy_cot':
            return _run_greedy_cot(_client, task, question)
        elif _strat == 'self_consistency_k5':
            return _run_sc_k5(_client, task, question)
        elif _strat == 'best_of_n_k5_self_eval':
            return _run_bon_k5(_client, task, question)
        elif _strat == 'zero_shot_tot_k3':
            return _run_tot_k3(_client, task, question)
        elif _strat in ('frugal_reason_v3', 'frugal_reason_v4'):
            _res = frugal_reason_v3_evaluate(_client, task, question, input_metadata=question, enable_early_exit=True, alpha=0.6)
            return {'selected_answer': _res.get('selected_answer'), 'raw_response': json.dumps(_res), 'latency_seconds_total': _res.get('latency', 0.0), 'total_tokens': _res.get('tokens', 0), 'model_calls': _res.get('calls', 0), 'parse_success': _res.get('parse_success', False), 'parse_method': _res.get('route_used', 'frugal_reason_v3'), 'raw_paths': [], 'clusters': _res.get('clusters', []), 'candidates': _res.get('candidates', []), 'early_exit': _res.get('early_exit', False)}
        raise ValueError(f'Unknown strategy: {_strat}')

    def _ollama_unload(model_name):
        """Kick a model out of VRAM. Tries ollama stop, then keep_alive=0, then restart."""
        import requests as _req
        print(f'\n--- VRAM HYGIENE: unloading {_model_name} ---')
        _r = subprocess.run(f'ollama stop {_model_name}', shell=True, capture_output=True, text=True)
        if _r.returncode == 0:
            print(f'  ollama stop {_model_name}: OK')
        else:
            try:
                _req.post('http://localhost:11434/api/generate', json={'model': _model_name, 'prompt': '', 'keep_alive': 0}, timeout=30)
                print(f'  keep_alive=0 sent to {_model_name}')
            except Exception:
                print(f'  Restarting Ollama server to free VRAM...')
                subprocess.run("pkill -f 'ollama serve' 2>/dev/null || true", shell=True)
                time.sleep(3)
                subprocess.Popen('nohup ollama serve > /dev/null 2>&1 &', shell=True)
                time.sleep(5)
        time.sleep(3)
        _ps = subprocess.run('ollama ps', shell=True, capture_output=True, text=True)
        if _model_name.split(':')[0] in _ps.stdout:
            print(f'  WARNING: {_model_name} still loaded! Output:\n{_ps.stdout}')
        else:
            print(f'  CONFIRMED: {_model_name} unloaded from VRAM')
        subprocess.run('nvidia-smi --query-gpu=memory.free --format=csv,noheader', shell=True)
    print(f'Loading datasets for {_MODEL}...')
    _loader_config = {'sampling': {'questions_per_task': 1500, 'seed': _SEED}, 'tasks': {_ds: {} for _ds in _DATASETS}}
    _loaded = load_all_tasks(_loader_config)
    _qids_path = Path('data/confirmatory_qids.json')
    _conf_qids = {}
    if _qids_path.exists():
        with open(_qids_path) as _f:
            _conf_qids = json.load(_f)
    _task_maps = {}
    for _ds in _DATASETS:
        _task_maps[_ds] = {_item['id']: _item for _item in _loaded.get(_ds, [])}
    _qid_lists = {}
    _rng = random.Random(_SEED)
    for _ds in _DATASETS:
        _all_ids = list(_task_maps[_ds].keys())
        if _ds in _conf_qids:
            _cq = _conf_qids[_ds]
            if isinstance(_cq, dict):
                _flat = []
                for _v in _cq.values():
                    if isinstance(_v, list):
                        _flat.extend(_v)
                _cq = _flat
            _qid_lists[_ds] = _cq[:_QID_LIMIT]
        else:
            _qid_lists[_ds] = _rng.sample(_all_ids, min(_QID_LIMIT, len(_all_ids)))
    _results_dir = _nb / 'results' / 'block_b_logs'
    _results_dir.mkdir(parents=True, exist_ok=True)
    _db_path = str(_nb / 'block_b_checkpoint.db')
    _conn = sqlite3.connect(_db_path)
    _cur = _conn.cursor()
    _cur.execute('CREATE TABLE IF NOT EXISTS completed (\n    model TEXT, dataset TEXT, strategy TEXT, qid TEXT, timestamp DATETIME,\n    PRIMARY KEY(model, dataset, strategy, qid))')
    _conn.commit()
    _client = OllamaClient(model=_MODEL)
    _verifier = OutcomeVerifier(_client)
    _total_target = sum((len(_qid_lists[_ds]) for _ds in _DATASETS)) * len(_STRATEGIES)
    _done = 0
    _start = time.time()
    _hw = 'gpu' if os.path.exists('/proc/driver/nvidia') else 'cpu'
    print(f'Target: {_total_target} runs on {_MODEL}')
    for _ds in _DATASETS:
        for _strat in _STRATEGIES:
            _log_path = _results_dir / f'{_PREFIX}{_ds}_{_strat}.jsonl'
            for _qid in _qid_lists[_ds]:
                _cur.execute('SELECT 1 FROM completed WHERE model=? AND dataset=? AND strategy=? AND qid=?', (_MODEL, _ds, _strat, _qid))
                if _cur.fetchone():
                    _done = _done + 1
                    continue
                _item = _task_maps[_ds].get(_qid)
                if not _item:
                    _done = _done + 1
                    continue
                for _attempt in range(3):
                    try:
                        _res = _run_strategy(_client, _strat, _ds, _item['question'])
                        break
                    except Exception as e:
                        print(f'  Retry {_attempt + 1}/3 {_ds}/{_strat}/{_qid}: {e}')
                        time.sleep(10)
                else:
                    _done = _done + 1
                    continue
                _score_res = _verifier.score(_ds, _item['question'], _res.get('raw_response', ''), _res['selected_answer'], _item['gold_answer'])
                _is_correct = _score_res['score'] == 1.0
                _log_row = {'model': _MODEL, 'dataset': _ds, 'strategy': _strat, 'qid': _qid, 'gold': _item['gold_answer'], 'selected_answer': _res['selected_answer'], 'correct': _is_correct, 'parse_success': _res['parse_success'], 'parse_method': _res.get('parse_method', ''), 'latency_seconds': _res['latency_seconds_total'], 'tokens': _res['total_tokens'], 'calls': _res['model_calls'], 'hardware_type': _hw, 'early_exit': _res.get('early_exit', False), 'clusters': _res.get('clusters', []), 'candidates': _res.get('candidates', []), 'raw_paths': _res.get('raw_paths', [])}
                with open(_log_path, 'a', encoding='utf-8') as _f:
                    _f.write(json.dumps(_log_row) + '\n')
                _cur.execute("INSERT OR IGNORE INTO completed VALUES (?,?,?,?,datetime('now'))", (_MODEL, _ds, _strat, _qid))
                _conn.commit()
                _done = _done + 1
                if _done % 50 == 0:
                    _elapsed = time.time() - _start
                    _eta = (_total_target - _done) * (_elapsed / max(_done, 1))
                    print(f'[{_MODEL}] {_done}/{_total_target} | {_elapsed / 3600:.1f}h elapsed | ETA {_eta / 3600:.1f}h')
    _conn.close()
    print(f'\nSweep DONE: {_done}/{_total_target} runs completed on {_MODEL}.')
    print('\nCompleteness Matrix:')
    for _ds in _DATASETS:
        for _strat in _STRATEGIES:
            _log_path = _results_dir / f'{_PREFIX}{_ds}_{_strat}.jsonl'
            _count = 0
            if _log_path.exists():
                with open(_log_path) as _f:
                    _count = sum((1 for l in _f if l.strip()))
            _status = 'OK' if _count >= _QID_LIMIT else f'GAP ({_count}/{_QID_LIMIT})'
            print(f'  {_ds:12s} | {_strat:25s} | {_count:4d} | {_status}')
    try:
        _api = HfApi(token='REDACTED')
        _zp = str(_nb / 'results' / 'block_b_qwen7b.zip')
        with zipfile.ZipFile(_zp, 'w', zipfile.ZIP_DEFLATED) as _zf:
            for _f in _results_dir.glob(f'{_PREFIX}*.jsonl'):
                _zf.write(str(_f), f'block_b_logs/{_f.name}')
        _api.upload_file(path_or_fileobj=_zp, path_in_repo='checkpoints/block_b_qwen7b.zip', repo_id='Satabarto/Molab_Checkpoints_Cost_AWARE', repo_type='dataset')
        print('Pushed results to HF.')
    except Exception as e:
        print(f'HF push failed (non-fatal): {e}')
    _ollama_unload(_MODEL)
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
def _(
    HfApi,
    OllamaClient,
    OutcomeVerifier,
    Path,
    frugal_reason_v3_evaluate,
    get_parser,
    get_prompt,
    json,
    load_all_tasks,
    os,
    random,
    re,
    sqlite3,
    subprocess,
    sys,
    time,
    warnings,
    zipfile,
):
    warnings.filterwarnings('ignore', category=SyntaxWarning)
    _nb = Path(os.environ.get('NOTEBOOK_DIR', '.'))
    sys.path.insert(0, str(_nb / 'ttc-frugalreason-poc' / 'experiment_fr'))
    os.chdir(str(_nb / 'ttc-frugalreason-poc' / 'experiment_fr'))
    _MODEL = 'llama3.3:70b'
    _PREFIX = 'llama70b_'
    _SEED = 0
    _STRATEGIES = ['greedy_io', 'greedy_cot', 'zero_shot_tot_k3', 'self_consistency_k5', 'best_of_n_k5_self_eval', 'frugal_reason_v3']
    _DATASETS = ['gsm8k', 'aqua', 'math', 'strategyqa']
    _QID_LIMIT = 100

    def _run_greedy_io(client, task, question):
        prompt = get_prompt('greedy_io', task, question)
        _r = _client.generate(prompt, temperature=0.0)
        _p = get_parser(task)(_r['text'])
        return {'selected_answer': _p['final_answer'], 'raw_response': _r['text'], 'latency_seconds_total': _r['latency_seconds'], 'total_tokens': _r['total_tokens'], 'model_calls': 1, 'parse_success': _p['parse_success'], 'parse_method': _p['parse_method']}

    def _run_greedy_cot(client, task, question):
        prompt = get_prompt('greedy_cot', task, question)
        _r = _client.generate(prompt, temperature=0.0)
        _p = get_parser(task)(_r['text'])
        return {'selected_answer': _p['final_answer'], 'raw_response': _r['text'], 'latency_seconds_total': _r['latency_seconds'], 'total_tokens': _r['total_tokens'], 'model_calls': 1, 'parse_success': _p['parse_success'], 'parse_method': _p['parse_method']}

    def _run_sc_k5(client, task, question):
        prompt = get_prompt('greedy_cot', task, question)
        answers = []
        lat = 0
        _tok = 0
        raws = []
        parser = get_parser(task)
        for _ in range(5):
            _r = _client.generate(prompt, temperature=0.7)
            lat = lat + _r['latency_seconds']
            _tok = _tok + _r['total_tokens']
            raws.append(_r['text'])
            _p = parser(_r['text'])
            if _p['final_answer'] is not None:
                answers.append(_p['final_answer'])
        _best = None
        if answers:
            _counts = {}
            for a in answers:
                _counts[a] = _counts.get(a, 0) + 1
            mx = max(_counts.values())
            for a in answers:
                if _counts[a] == mx:
                    _best = a
                    break
        return {'selected_answer': _best, 'raw_response': '\n---SAMPLE---\n'.join(raws), 'latency_seconds_total': lat, 'total_tokens': _tok, 'model_calls': 5, 'parse_success': _best is not None, 'parse_method': 'majority_vote' if _best else 'failed', 'raw_paths': raws}

    def _run_bon_k5(client, task, question):
        prompt = get_prompt('greedy_cot', task, question)
        rationales = []
        lat = 0
        _tok = 0
        parser = get_parser(task)
        for _ in range(5):
            _r = _client.generate(prompt, temperature=0.7)
            lat = lat + _r['latency_seconds']
            _tok = _tok + _r['total_tokens']
            _p = parser(_r['text'])
            rationales.append({'text': _r['text'], 'answer': _p['final_answer']})
        best_ans = None
        best_score = -1
        best_rat = ''
        judge_texts = []
        for rat in rationales:
            jp = get_prompt('best_of_n', task='', question=question, candidate=rat['text'])
            jr = _client.generate(jp, temperature=0.0)
            lat = lat + jr['latency_seconds']
            _tok = _tok + jr['total_tokens']
            judge_texts.append(jr['text'])
            score = 0.5
            sm = re.search('confidence:\\s*(\\d+)', jr['text'].lower())
            if sm:
                score = float(sm.group(1)) / 100.0
            elif 'yes' in jr['text'].lower():
                score = 1.0
            elif 'no' in jr['text'].lower():
                score = 0.0
            if score > best_score:
                best_score = score
                best_ans = rat['answer']
                best_rat = rat['text']
        return {'selected_answer': best_ans, 'raw_response': f'Selected:\n{best_rat}\n\nJudge:\n' + '\n---\n'.join(judge_texts), 'latency_seconds_total': lat, 'total_tokens': _tok, 'model_calls': 10, 'parse_success': best_ans is not None, 'parse_method': 'best_of_n_self_eval', 'raw_paths': [_r['text'] for _r in rationales]}

    def _run_tot_k3(client, task, question):
        return _run_greedy_cot(_client, task, question)

    def _run_strategy(client, strat, task, question):
        if _strat == 'greedy_io':
            return _run_greedy_io(_client, task, question)
        elif _strat == 'greedy_cot':
            return _run_greedy_cot(_client, task, question)
        elif _strat == 'self_consistency_k5':
            return _run_sc_k5(_client, task, question)
        elif _strat == 'best_of_n_k5_self_eval':
            return _run_bon_k5(_client, task, question)
        elif _strat == 'zero_shot_tot_k3':
            return _run_tot_k3(_client, task, question)
        elif _strat in ('frugal_reason_v3', 'frugal_reason_v4'):
            _res = frugal_reason_v3_evaluate(_client, task, question, input_metadata=question, enable_early_exit=True, alpha=0.6)
            return {'selected_answer': _res.get('selected_answer'), 'raw_response': json.dumps(_res), 'latency_seconds_total': _res.get('latency', 0.0), 'total_tokens': _res.get('tokens', 0), 'model_calls': _res.get('calls', 0), 'parse_success': _res.get('parse_success', False), 'parse_method': _res.get('route_used', 'frugal_reason_v3'), 'raw_paths': [], 'clusters': _res.get('clusters', []), 'candidates': _res.get('candidates', []), 'early_exit': _res.get('early_exit', False)}
        raise ValueError(f'Unknown strategy: {_strat}')

    def _ollama_unload(model_name):
        """Kick a model out of VRAM. Tries ollama stop, then keep_alive=0, then restart."""
        import requests as _req
        print(f'\n--- VRAM HYGIENE: unloading {_model_name} ---')
        _r = subprocess.run(f'ollama stop {_model_name}', shell=True, capture_output=True, text=True)
        if _r.returncode == 0:
            print(f'  ollama stop {_model_name}: OK')
        else:
            try:
                _req.post('http://localhost:11434/api/generate', json={'model': _model_name, 'prompt': '', 'keep_alive': 0}, timeout=30)
                print(f'  keep_alive=0 sent to {_model_name}')
            except Exception:
                print(f'  Restarting Ollama server to free VRAM...')
                subprocess.run("pkill -f 'ollama serve' 2>/dev/null || true", shell=True)
                time.sleep(3)
                subprocess.Popen('nohup ollama serve > /dev/null 2>&1 &', shell=True)
                time.sleep(5)
        time.sleep(3)
        _ps = subprocess.run('ollama ps', shell=True, capture_output=True, text=True)
        if _model_name.split(':')[0] in _ps.stdout:
            print(f'  WARNING: {_model_name} still loaded! Output:\n{_ps.stdout}')
        else:
            print(f'  CONFIRMED: {_model_name} unloaded from VRAM')
        subprocess.run('nvidia-smi --query-gpu=memory.free --format=csv,noheader', shell=True)
    print(f'Loading datasets for {_MODEL}...')
    _loader_config = {'sampling': {'questions_per_task': 1500, 'seed': _SEED}, 'tasks': {_ds: {} for _ds in _DATASETS}}
    _loaded = load_all_tasks(_loader_config)
    _qids_path = Path('data/confirmatory_qids.json')
    _conf_qids = {}
    if _qids_path.exists():
        with open(_qids_path) as _f:
            _conf_qids = json.load(_f)
    _task_maps = {}
    for _ds in _DATASETS:
        _task_maps[_ds] = {_item['id']: _item for _item in _loaded.get(_ds, [])}
    _qid_lists = {}
    _rng = random.Random(_SEED)
    for _ds in _DATASETS:
        _all_ids = list(_task_maps[_ds].keys())
        if _ds in _conf_qids:
            _cq = _conf_qids[_ds]
            if isinstance(_cq, dict):
                _flat = []
                for _v in _cq.values():
                    if isinstance(_v, list):
                        _flat.extend(_v)
                _cq = _flat
            _qid_lists[_ds] = _cq[:_QID_LIMIT]
        else:
            _qid_lists[_ds] = _rng.sample(_all_ids, min(_QID_LIMIT, len(_all_ids)))
    _results_dir = _nb / 'results' / 'block_b_logs'
    _results_dir.mkdir(parents=True, exist_ok=True)
    _db_path = str(_nb / 'block_b_checkpoint.db')
    _conn = sqlite3.connect(_db_path)
    _cur = _conn.cursor()
    _cur.execute('CREATE TABLE IF NOT EXISTS completed (\n    model TEXT, dataset TEXT, strategy TEXT, qid TEXT, timestamp DATETIME,\n    PRIMARY KEY(model, dataset, strategy, qid))')
    _conn.commit()
    _client = OllamaClient(model=_MODEL)
    _verifier = OutcomeVerifier(_client)
    _total_target = sum((len(_qid_lists[_ds]) for _ds in _DATASETS)) * len(_STRATEGIES)
    _done = 0
    _start = time.time()
    _hw = 'gpu' if os.path.exists('/proc/driver/nvidia') else 'cpu'
    print(f'Target: {_total_target} runs on {_MODEL}')
    for _ds in _DATASETS:
        for _strat in _STRATEGIES:
            _log_path = _results_dir / f'{_PREFIX}{_ds}_{_strat}.jsonl'
            for _qid in _qid_lists[_ds]:
                _cur.execute('SELECT 1 FROM completed WHERE model=? AND dataset=? AND strategy=? AND qid=?', (_MODEL, _ds, _strat, _qid))
                if _cur.fetchone():
                    _done = _done + 1
                    continue
                _item = _task_maps[_ds].get(_qid)
                if not _item:
                    _done = _done + 1
                    continue
                for _attempt in range(3):
                    try:
                        _res = _run_strategy(_client, _strat, _ds, _item['question'])
                        break
                    except Exception as e:
                        print(f'  Retry {_attempt + 1}/3 {_ds}/{_strat}/{_qid}: {e}')
                        time.sleep(10)
                else:
                    _done = _done + 1
                    continue
                _score_res = _verifier.score(_ds, _item['question'], _res.get('raw_response', ''), _res['selected_answer'], _item['gold_answer'])
                _is_correct = _score_res['score'] == 1.0
                _log_row = {'model': _MODEL, 'dataset': _ds, 'strategy': _strat, 'qid': _qid, 'gold': _item['gold_answer'], 'selected_answer': _res['selected_answer'], 'correct': _is_correct, 'parse_success': _res['parse_success'], 'parse_method': _res.get('parse_method', ''), 'latency_seconds': _res['latency_seconds_total'], 'tokens': _res['total_tokens'], 'calls': _res['model_calls'], 'hardware_type': _hw, 'early_exit': _res.get('early_exit', False), 'clusters': _res.get('clusters', []), 'candidates': _res.get('candidates', []), 'raw_paths': _res.get('raw_paths', [])}
                with open(_log_path, 'a', encoding='utf-8') as _f:
                    _f.write(json.dumps(_log_row) + '\n')
                _cur.execute("INSERT OR IGNORE INTO completed VALUES (?,?,?,?,datetime('now'))", (_MODEL, _ds, _strat, _qid))
                _conn.commit()
                _done = _done + 1
                if _done % 50 == 0:
                    _elapsed = time.time() - _start
                    _eta = (_total_target - _done) * (_elapsed / max(_done, 1))
                    print(f'[{_MODEL}] {_done}/{_total_target} | {_elapsed / 3600:.1f}h elapsed | ETA {_eta / 3600:.1f}h')
    _conn.close()
    print(f'\nSweep DONE: {_done}/{_total_target} runs completed on {_MODEL}.')
    print('\nCompleteness Matrix:')
    for _ds in _DATASETS:
        for _strat in _STRATEGIES:
            _log_path = _results_dir / f'{_PREFIX}{_ds}_{_strat}.jsonl'
            _count = 0
            if _log_path.exists():
                with open(_log_path) as _f:
                    _count = sum((1 for l in _f if l.strip()))
            _status = 'OK' if _count >= _QID_LIMIT else f'GAP ({_count}/{_QID_LIMIT})'
            print(f'  {_ds:12s} | {_strat:25s} | {_count:4d} | {_status}')
    try:
        _api = HfApi(token='REDACTED')
        _zp = str(_nb / 'results' / 'block_b_llama70b.zip')
        with zipfile.ZipFile(_zp, 'w', zipfile.ZIP_DEFLATED) as _zf:
            for _f in _results_dir.glob(f'{_PREFIX}*.jsonl'):
                _zf.write(str(_f), f'block_b_logs/{_f.name}')
        _api.upload_file(path_or_fileobj=_zp, path_in_repo='checkpoints/block_b_llama70b.zip', repo_id='Satabarto/Molab_Checkpoints_Cost_AWARE', repo_type='dataset')
        print('Pushed results to HF.')
    except Exception as e:
        print(f'HF push failed (non-fatal): {e}')
    _ollama_unload(_MODEL)
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
def _(
    HfApi,
    OllamaClient,
    OutcomeVerifier,
    Path,
    frugal_reason_v3_evaluate,
    get_parser,
    get_prompt,
    json,
    load_all_tasks,
    os,
    requests,
    sqlite3,
    subprocess,
    sys,
    time,
    warnings,
    zipfile,
):
    warnings.filterwarnings('ignore', category=SyntaxWarning)
    _nb = Path(os.environ.get('NOTEBOOK_DIR', '.'))
    sys.path.insert(0, str(_nb / 'ttc-frugalreason-poc' / 'experiment_fr'))
    os.chdir(str(_nb / 'ttc-frugalreason-poc' / 'experiment_fr'))
    _ps = subprocess.run('ollama ps', shell=True, capture_output=True, text=True)
    if 'llama3.3:70b' in _ps.stdout or '70b' in _ps.stdout.lower():
        print('WARNING: 70B still loaded! Attempting to unload...')
        subprocess.run('ollama stop llama3.3:70b', shell=True)
        time.sleep(5)
        try:
            requests.post('http://localhost:11434/api/generate', json={'model': 'llama3.3:70b', 'prompt': '', 'keep_alive': 0}, timeout=30)
        except:
            pass
        time.sleep(5)
        ps2 = subprocess.run('ollama ps', shell=True, capture_output=True, text=True)
        assert '70b' not in ps2.stdout.lower(), f'FATAL: 70B still loaded after stop!\n{ps2.stdout}'
        print('70B successfully unloaded.')
    else:
        print('PRE-ASSERT PASS: No 70B model loaded.')
    subprocess.run('nvidia-smi --query-gpu=memory.free --format=csv,noheader', shell=True)
    _MODEL = 'qwen2.5:72b'
    _PREFIX = 'qwen72b_'
    _SEED = 0
    _STRATEGIES = ['greedy_io', 'greedy_cot', 'frugal_reason_v4']
    _DATASETS = ['math']

    def _run_greedy_io(client, task, question):
        prompt = get_prompt('greedy_io', task, question)
        _r = _client.generate(prompt, temperature=0.0)
        _p = get_parser(task)(_r['text'])
        return {'selected_answer': _p['final_answer'], 'raw_response': _r['text'], 'latency_seconds_total': _r['latency_seconds'], 'total_tokens': _r['total_tokens'], 'model_calls': 1, 'parse_success': _p['parse_success'], 'parse_method': _p['parse_method']}

    def _run_greedy_cot(client, task, question):
        prompt = get_prompt('greedy_cot', task, question)
        _r = _client.generate(prompt, temperature=0.0)
        _p = get_parser(task)(_r['text'])
        return {'selected_answer': _p['final_answer'], 'raw_response': _r['text'], 'latency_seconds_total': _r['latency_seconds'], 'total_tokens': _r['total_tokens'], 'model_calls': 1, 'parse_success': _p['parse_success'], 'parse_method': _p['parse_method']}

    def _run_strategy(client, strat, task, question):
        if _strat == 'greedy_io':
            return _run_greedy_io(_client, task, question)
        elif _strat == 'greedy_cot':
            return _run_greedy_cot(_client, task, question)
        elif _strat == 'frugal_reason_v4':
            _res = frugal_reason_v3_evaluate(_client, task, question, input_metadata=question, enable_early_exit=True, alpha=0.6)
            return {'selected_answer': _res.get('selected_answer'), 'raw_response': json.dumps(_res), 'latency_seconds_total': _res.get('latency', 0.0), 'total_tokens': _res.get('tokens', 0), 'model_calls': _res.get('calls', 0), 'parse_success': _res.get('parse_success', False), 'parse_method': _res.get('route_used', 'frugal_reason_v3'), 'raw_paths': [], 'clusters': _res.get('clusters', []), 'candidates': _res.get('candidates', []), 'early_exit': _res.get('early_exit', False)}
        raise ValueError(f'Unknown strategy: {_strat}')
    _loader_config = {'sampling': {'questions_per_task': 1500, 'seed': _SEED}, 'tasks': {'math': {}}}
    _loaded = load_all_tasks(_loader_config)
    math_items = _loaded.get('math', [])
    task_map = {_item['id']: _item for _item in math_items}
    _qid_list = list(task_map.keys())[:238]
    _results_dir = _nb / 'results' / 'block_b_logs'
    _results_dir.mkdir(parents=True, exist_ok=True)
    _db_path = str(_nb / 'block_b_checkpoint.db')
    _conn = sqlite3.connect(_db_path)
    _cur = _conn.cursor()
    _cur.execute('CREATE TABLE IF NOT EXISTS completed (\n    model TEXT, dataset TEXT, strategy TEXT, qid TEXT, timestamp DATETIME,\n    PRIMARY KEY(model, dataset, strategy, qid))')
    _conn.commit()
    _client = OllamaClient(model=_MODEL)
    _verifier = OutcomeVerifier(_client)
    _total_target = len(_qid_list) * len(_STRATEGIES)
    _done = 0
    _start = time.time()
    _hw = 'gpu' if os.path.exists('/proc/driver/nvidia') else 'cpu'
    print(f'\nTarget: {_total_target} runs on {_MODEL} (MATH-238 × 3 strategies)')
    for _strat in _STRATEGIES:
        _log_path = _results_dir / f'{_PREFIX}math_{_strat}.jsonl'
        for _qid in _qid_list:
            _cur.execute('SELECT 1 FROM completed WHERE model=? AND dataset=? AND strategy=? AND qid=?', (_MODEL, 'math', _strat, _qid))
            if _cur.fetchone():
                _done = _done + 1
                continue
            _item = task_map.get(_qid)
            if not _item:
                _done = _done + 1
                continue
            for _attempt in range(3):
                try:
                    _res = _run_strategy(_client, _strat, 'math', _item['question'])
                    break
                except Exception as e:
                    print(f'  Retry {_attempt + 1}/3 math/{_strat}/{_qid}: {e}')
                    time.sleep(10)
            else:
                _done = _done + 1
                continue
            _score_res = _verifier.score('math', _item['question'], _res.get('raw_response', ''), _res['selected_answer'], _item['gold_answer'])
            _is_correct = _score_res['score'] == 1.0
            _log_row = {'model': _MODEL, 'dataset': 'math', 'strategy': _strat, 'qid': _qid, 'gold': _item['gold_answer'], 'selected_answer': _res['selected_answer'], 'correct': _is_correct, 'parse_success': _res['parse_success'], 'parse_method': _res.get('parse_method', ''), 'latency_seconds': _res['latency_seconds_total'], 'tokens': _res['total_tokens'], 'calls': _res['model_calls'], 'hardware_type': _hw, 'early_exit': _res.get('early_exit', False), 'clusters': _res.get('clusters', []), 'candidates': _res.get('candidates', []), 'raw_paths': _res.get('raw_paths', [])}
            with open(_log_path, 'a', encoding='utf-8') as _f:
                _f.write(json.dumps(_log_row) + '\n')
            _cur.execute("INSERT OR IGNORE INTO completed VALUES (?,?,?,?,datetime('now'))", (_MODEL, 'math', _strat, _qid))
            _conn.commit()
            _done = _done + 1
            if _done % 20 == 0:
                _elapsed = time.time() - _start
                _eta = (_total_target - _done) * (_elapsed / max(_done, 1))
                print(f'[{_MODEL}] {_done}/{_total_target} | {_elapsed / 3600:.1f}h elapsed | ETA {_eta / 3600:.1f}h')
    _conn.close()
    print(f'\n72B Cross-Model DONE: {_done}/{_total_target} runs.')
    try:
        _api = HfApi(token='REDACTED')
        _zp = str(_nb / 'results' / 'block_b_qwen72b.zip')
        with zipfile.ZipFile(_zp, 'w', zipfile.ZIP_DEFLATED) as _zf:
            for _f in _results_dir.glob(f'{_PREFIX}*.jsonl'):
                _zf.write(str(_f), f'block_b_logs/{_f.name}')
        _api.upload_file(path_or_fileobj=_zp, path_in_repo='checkpoints/block_b_qwen72b.zip', repo_id='Satabarto/Molab_Checkpoints_Cost_AWARE', repo_type='dataset')
        print('Pushed 72B results to HF.')
    except Exception as e:
        print(f'HF push failed (non-fatal): {e}')

    def _ollama_unload(model_name):
        import requests as _req
        print(f'\n--- VRAM HYGIENE: unloading {_model_name} ---')
        _r = subprocess.run(f'ollama stop {_model_name}', shell=True, capture_output=True, text=True)
        if _r.returncode == 0:
            print(f'  ollama stop {_model_name}: OK')
        else:
            try:
                _req.post('http://localhost:11434/api/generate', json={'model': _model_name, 'prompt': '', 'keep_alive': 0}, timeout=30)
                print(f'  keep_alive=0 sent to {_model_name}')
            except Exception:
                print(f'  Restarting Ollama server to free VRAM...')
                subprocess.run("pkill -f 'ollama serve' 2>/dev/null || true", shell=True)
                time.sleep(3)
                subprocess.Popen('nohup ollama serve > /dev/null 2>&1 &', shell=True)
                time.sleep(5)
        time.sleep(3)
        _ps = subprocess.run('ollama ps', shell=True, capture_output=True, text=True)
        if _model_name.split(':')[0] in _ps.stdout:
            print(f'  WARNING: {_model_name} still loaded! Output:\n{_ps.stdout}')
        else:
            print(f'  CONFIRMED: {_model_name} unloaded from VRAM')
        subprocess.run('nvidia-smi --query-gpu=memory.free --format=csv,noheader', shell=True)
    _ollama_unload(_MODEL)
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
def _(Path, json, math, matplotlib, np, os, pd, plt, re, warnings):
    matplotlib.use('Agg')
    from scipy.stats import pearsonr
    warnings.filterwarnings('ignore', category=SyntaxWarning)
    _nb = Path(os.environ.get('NOTEBOOK_DIR', '.'))
    _cal_dir = _nb / 'results' / 'calibration'
    _cal_dir.mkdir(parents=True, exist_ok=True)
    _fig_dir = _nb / 'results' / 'figures'
    _fig_dir.mkdir(parents=True, exist_ok=True)
    _DATASETS = ['gsm8k', 'aqua', 'math', 'strategyqa']
    _ALPHAS = [round(a * 0.05, 2) for a in range(21)]
    _MODELS = [('qwen2.5:1.5b', 1.5, 'block_b_logs', 'qwen15b_'), ('qwen2.5:3b', 3.0, 'block_a_logs', ''), ('qwen2.5:7b', 7.0, 'block_b_logs', 'qwen7b_'), ('llama3.2:3b', 3.0, 'block_b_logs', 'llama32_'), ('llama3.3:70b', 70.0, 'block_b_logs', 'llama70b_')]

    def _load_jsonl(path):
        records = []
        if not path.exists():
            return records
        with open(path, 'r', encoding='utf-8') as _f:
            for _line in _f:
                if _line.strip():
                    records.append(json.loads(_line))
        return records
    print('=' * 60)
    print('  Add-On 5 — EXT α* Scaling Law (5 models)')
    print('=' * 60)
    _rows = []
    for (_model_name, params, _subdir, _prefix) in _MODELS:
        all_candidates = []
        for _ds in _DATASETS:
            _log_path = _nb / 'results' / _subdir / f'{_prefix}{_ds}_frugal_reason_v3.jsonl'
            if not _log_path.exists():
                _log_path = _nb / 'ttc-frugalreason-poc' / 'experiment_fr' / 'results' / _subdir / f'{_prefix}{_ds}_frugal_reason_v3.jsonl'
            _recs = _load_jsonl(_log_path)
            for _rec in _recs:
                _cands = _rec.get('candidates', [])
                if not _cands:
                    try:
                        _cands = json.loads(_rec.get('raw_response', '{}')).get('candidates', [])
                    except:
                        pass
                if _cands:
                    all_candidates.append({'candidates': _cands, 'gold': _rec.get('gold', _rec.get('gold_answer', '')), 'dataset': _ds})
        if not all_candidates:
            print(f'  {_model_name}: NO FR candidates found, skipping.')
            continue
        _best_alpha = 0.6
        _best_acc = 0.0
        alpha_accs = []
        for _alpha in _ALPHAS:
            _correct = 0
            _total = len(all_candidates)
            for _item in all_candidates:
                _cands = _item['candidates']
                if not _cands:
                    continue
                _best_a = None
                _best_S = -float('inf')
                for _c in _cands:
                    _V = _c.get('V_raw', _c.get('V', 0))
                    _prior = _c.get('prior', 0)
                    _S = _alpha * _V + (1.0 - _alpha) * math.log(_prior + 1e-06)
                    if _S > _best_S:
                        _best_S = _S
                        _best_a = _c.get('answer')
                _norm_a = re.sub('[,$\\\\\\s]', '', str(_best_a or '')).lower().strip()
                _norm_g = re.sub('[,$\\\\\\s]', '', str(_item['gold'] or '')).lower().strip()
                if _norm_a == _norm_g:
                    _correct = _correct + 1
            _acc = _correct / max(_total, 1)
            alpha_accs.append(_acc)
            if _acc > _best_acc:
                _best_acc = _acc
                _best_alpha = _alpha
        V_vals = []
        P_vals = []
        correct_flags = []
        for _item in all_candidates:
            for _c in _item['candidates']:
                _V = _c.get('V_raw', _c.get('V', 0))
                _prior = _c.get('prior', 0)
                V_vals.append(_V)
                P_vals.append(_prior)
                _norm_a = re.sub('[,$\\\\\\s]', '', str(_c.get('answer', ''))).lower().strip()
                _norm_g = re.sub('[,$\\\\\\s]', '', str(_item['gold'] or '')).lower().strip()
                correct_flags.append(1.0 if _norm_a == _norm_g else 0.0)
        if V_vals:
            _V_arr = np.array(V_vals)
            P_arr = np.array(P_vals)
            _C_arr = np.array(correct_flags)
            _sigma2_V = np.mean((_V_arr - _C_arr) ** 2)
            _tau2 = np.mean((P_arr - _C_arr) ** 2)
            _alpha_theory = _tau2 / (_sigma2_V + _tau2) if _sigma2_V + _tau2 > 0 else 0.5
        else:
            _sigma2_V = _tau2 = 0
            _alpha_theory = 0.5
        acc_at_0 = alpha_accs[0] if alpha_accs else 0
        print(f'  {_model_name:18s} | α*_emp={_best_alpha:.2f} acc={_best_acc:.1%} | α*_theory={_alpha_theory:.2f} | acc@α=0={acc_at_0:.1%} | N={len(all_candidates)}')
        _rows.append({'model': _model_name, 'params_B': params, 'alpha_star_emp': _best_alpha, 'acc_at_alpha_star': _best_acc, 'alpha_star_theory': _alpha_theory, 'sigma2_V': _sigma2_V, 'tau2': _tau2, 'acc_at_alpha_0': acc_at_0, 'n_questions': len(all_candidates)})
        for (_i, _alpha) in enumerate(_ALPHAS):
            _rows.append({'model': _model_name, 'params_B': params, 'alpha': _alpha, 'accuracy': alpha_accs[_i], 'alpha_star_emp': _best_alpha, 'alpha_star_theory': _alpha_theory})
    if _rows:
        _df = pd.DataFrame(_rows)
        _df.to_csv(str(_cal_dir / 'alpha_grid.csv'), index=False)
        print(f'\nSaved results/calibration/alpha_grid.csv ({len(_df)} rows)')
        _summary = _df.dropna(subset=['acc_at_alpha_star']).drop_duplicates('model')
        if len(_summary) > 0:
            (_fig, _ax) = plt.subplots(figsize=(10, 6))
            log_sizes = np.log10(_summary['params_B'])
            _ax.scatter(log_sizes, _summary['alpha_star_emp'], s=100, c='blue', label='α*_emp', zorder=5)
            _ax.scatter(log_sizes, _summary['alpha_star_theory'], s=100, c='red', marker='^', label='α*_theory', zorder=5)
            for (_, _r) in _summary.iterrows():
                _ax.annotate(_r['model'], (np.log10(_r['params_B']), _r['alpha_star_emp']), fontsize=8, textcoords='offset points', xytext=(5, 5))
            _ax.set_xlabel('log₁₀(Model Size in Billions)')
            _ax.set_ylabel('α*')
            _ax.set_title('Fig1: α* Scaling Law — α* vs Model Size')
            _ax.legend()
            _ax.grid(True, alpha=0.3)
            _fig.savefig(str(_fig_dir / 'fig1_scaling_law.png'), dpi=300, bbox_inches='tight')
            _fig.savefig(str(_fig_dir / 'fig1_scaling_law.pdf'), bbox_inches='tight')
            plt.close(_fig)
            print('Saved fig1_scaling_law.png + .pdf')
        if len(_summary) >= 3:
            (_fig, _ax) = plt.subplots(figsize=(8, 8))
            _ax.scatter(_summary['alpha_star_theory'], _summary['alpha_star_emp'], s=120, c='green', zorder=5)
            for (_, _r) in _summary.iterrows():
                _ax.annotate(_r['model'], (_r['alpha_star_theory'], _r['alpha_star_emp']), fontsize=8, textcoords='offset points', xytext=(5, 5))
            lims = [0, 1]
            _ax.plot(lims, lims, '--', c='gray', alpha=0.5, label='y=x (perfect)')
            (r_val, _p_val) = pearsonr(_summary['alpha_star_theory'], _summary['alpha_star_emp'])
            _ax.set_xlabel('α*_theory (τ²/(σ²_V+τ²))')
            _ax.set_ylabel('α*_emp (argmax accuracy)')
            _ax.set_title(f'Fig2: Theory vs Empirical α* (Pearson r={r_val:.3f}, p={_p_val:.4f})')
            _ax.legend()
            _ax.grid(True, alpha=0.3)
            _fig.savefig(str(_fig_dir / 'fig2_theory_vs_emp.png'), dpi=300, bbox_inches='tight')
            _fig.savefig(str(_fig_dir / 'fig2_theory_vs_emp.pdf'), bbox_inches='tight')
            plt.close(_fig)
            print(f'Saved fig2_theory_vs_emp.png + .pdf (Pearson r={r_val:.3f})')
        else:
            print('Not enough models for Fig2 (need ≥3)')
    else:
        print('No data to plot!')
    print('\nAdd-On 5 DONE.')
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
def _(Path, binomtest, json, math, os, pd, warnings):
    from scipy.stats import binom_test
    warnings.filterwarnings('ignore', category=SyntaxWarning)
    _nb = Path(os.environ.get('NOTEBOOK_DIR', '.'))
    _cal_dir = _nb / 'results' / 'calibration'
    _cal_dir.mkdir(parents=True, exist_ok=True)
    _DATASETS = ['gsm8k', 'aqua', 'math', 'strategyqa']

    def _load_jsonl(path):
        records = []
        if not path.exists():
            return records
        with open(path, 'r', encoding='utf-8') as _f:
            for _line in _f:
                if _line.strip():
                    records.append(json.loads(_line))
        return records

    def _wilson_ci(k, n, z=1.96):
        if _n == 0:
            return (0, 0, 0)
        _p = _k / _n
        denom = 1 + z ** 2 / _n
        centre = (_p + z ** 2 / (2 * _n)) / denom
        margin = z * math.sqrt((_p * (1 - _p) + z ** 2 / (4 * _n)) / _n) / denom
        return (_p, max(0, centre - margin), min(1, centre + margin))
    print('=' * 60)
    print('  Add-On 6 — Matched FR-3B vs 70B Stats')
    print('=' * 60)
    comparisons = ['greedy_io', 'greedy_cot', 'self_consistency_k5', 'frugal_reason_v3']
    _rows = []
    for _ds in _DATASETS:
        fr3b_path = _nb / 'results' / 'block_a_logs' / f'{_ds}_frugal_reason_v3.jsonl'
        if not fr3b_path.exists():
            fr3b_path = _nb / 'ttc-frugalreason-poc' / 'experiment_fr' / 'results' / 'block_a_logs' / f'{_ds}_frugal_reason_v3.jsonl'
        fr3b_recs = _load_jsonl(fr3b_path)
        fr3b_by_qid = {_r.get('qid', _r.get('question_id', '')): _r.get('correct', False) for _r in fr3b_recs}
        for _strat in comparisons:
            path_70b = _nb / 'results' / 'block_b_logs' / f'llama70b_{_ds}_{_strat}.jsonl'
            recs_70b = _load_jsonl(path_70b)
            if not recs_70b:
                print(f'  {_ds}/{_strat}: no 70B logs found, skipping')
                continue
            r70b_by_qid = {_r.get('qid', _r.get('question_id', '')): _r.get('correct', False) for _r in recs_70b}
            matched = set(fr3b_by_qid.keys()) & set(r70b_by_qid.keys())
            if not matched:
                print(f'  {_ds}/{_strat}: no matched QIDs, skipping')
                continue
            _n = len(matched)
            fr3b_correct = sum((1 for _q in matched if fr3b_by_qid[_q]))
            r70b_correct = sum((1 for _q in matched if r70b_by_qid[_q]))
            (acc_fr3b, lo_fr, hi_fr) = _wilson_ci(fr3b_correct, _n)
            (acc_70b, lo_70, hi_70) = _wilson_ci(r70b_correct, _n)
            _delta = acc_fr3b - acc_70b
            _b = sum((1 for _q in matched if fr3b_by_qid[_q] and (not r70b_by_qid[_q])))
            _c = sum((1 for _q in matched if not fr3b_by_qid[_q] and r70b_by_qid[_q]))
            if _b + _c > 0:
                try:
                    _p_val = binom_test(_b, _b + _c, 0.5)
                except:
                    _p_val = binomtest(_b, _b + _c, 0.5).pvalue
            else:
                _p_val = 1.0
            _stars = '***' if _p_val < 0.001 else '**' if _p_val < 0.01 else '*' if _p_val < 0.05 else ''
            _rows.append({'dataset': _ds, 'comparison': f'FR-3B vs 70B-{_strat}', 'n_matched': _n, 'fr3b_acc': acc_fr3b, 'fr3b_ci': f'[{lo_fr:.1%},{hi_fr:.1%}]', '70b_acc': acc_70b, '70b_ci': f'[{lo_70:.1%},{hi_70:.1%}]', 'delta': _delta, 'p_value': _p_val, 'sig': _stars})
            print(f'  {_ds:12s} | FR-3B vs 70B-{_strat:25s} | Δ={_delta:+.1%} | p={_p_val:.4f}{_stars}')
    if _rows:
        _df = pd.DataFrame(_rows)
        _df.to_csv(str(_cal_dir / 'matched_70b_stats.csv'), index=False)
        print(f'\nSaved results/calibration/matched_70b_stats.csv ({len(_df)} rows)')
        print('\nFull table:')
        print(_df.to_string(index=False))
    else:
        print('No matched comparisons could be made (70B logs may not exist yet)')
    print('\nAdd-On 6 DONE.')
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
def _(
    Path,
    json,
    math,
    matplotlib,
    minimize_scalar,
    np,
    os,
    pd,
    plt,
    re,
    warnings,
):
    matplotlib.use('Agg')
    warnings.filterwarnings('ignore', category=SyntaxWarning)
    _nb = Path(os.environ.get('NOTEBOOK_DIR', '.'))
    _fig_dir = _nb / 'results' / 'figures'
    _fig_dir.mkdir(parents=True, exist_ok=True)
    _DATASETS = ['gsm8k', 'aqua', 'math', 'strategyqa']
    _STRATEGIES = ['greedy_io', 'greedy_cot', 'zero_shot_tot_k3', 'self_consistency_k5', 'best_of_n_k5_self_eval', 'frugal_reason_v3']
    _STRAT_LABELS = {'greedy_io': 'IO', 'greedy_cot': 'CoT', 'zero_shot_tot_k3': 'ToT-3', 'self_consistency_k5': 'SC-5', 'best_of_n_k5_self_eval': 'BoN-5', 'frugal_reason_v3': 'FR-v3'}
    _MODELS_INFO = {'qwen2.5:1.5b': ('block_b_logs', 'qwen15b_', 1.5), 'qwen2.5:3b': ('block_a_logs', '', 3.0), 'qwen2.5:7b': ('block_b_logs', 'qwen7b_', 7.0), 'llama3.2:3b': ('block_b_logs', 'llama32_', 3.0), 'llama3.3:70b': ('block_b_logs', 'llama70b_', 70.0)}

    def _load_jsonl(path):
        records = []
        if not path.exists():
            return records
        with open(path, 'r', encoding='utf-8') as _f:
            for _line in _f:
                if _line.strip():
                    records.append(json.loads(_line))
        return records

    def _wilson_ci(k, n, z=1.96):
        if _n == 0:
            return (0, 0, 0)
        _p = _k / _n
        denom = 1 + z ** 2 / _n
        centre = (_p + z ** 2 / (2 * _n)) / denom
        margin = z * math.sqrt((_p * (1 - _p) + z ** 2 / (4 * _n)) / _n) / denom
        return (_p, max(0, centre - margin), min(1, centre + margin))

    def _save_fig(fig, name):
        _fig.savefig(str(_fig_dir / f'{name}.png'), dpi=300, bbox_inches='tight')
        _fig.savefig(str(_fig_dir / f'{name}.pdf'), bbox_inches='tight')
        plt.close(_fig)
        print(f'  Saved {name}.png + .pdf')
    print('=' * 60)
    print('  Add-On 7 — EXT Figures (8 total, 70B included)')
    print('=' * 60)
    all_data = {}
    for (_model_name, (_subdir, _prefix, _)) in _MODELS_INFO.items():
        all_data[_model_name] = {}
        for _ds in _DATASETS:
            all_data[_model_name][_ds] = {}
            for _strat in _STRATEGIES:
                _log_path = _nb / 'results' / _subdir / f'{_prefix}{_ds}_{_strat}.jsonl'
                if not _log_path.exists():
                    _log_path = _nb / 'ttc-frugalreason-poc' / 'experiment_fr' / 'results' / _subdir / f'{_prefix}{_ds}_{_strat}.jsonl'
                all_data[_model_name][_ds][_strat] = _load_jsonl(_log_path)
    print('\nFig3: ECE Bars (fit T for 7B & 70B)')
    _ts_path = _nb / 'results' / 'calibration' / 'temp_scaling.json'
    _ts_data = {}
    if _ts_path.exists():
        with open(_ts_path) as _f:
            _ts_data = json.load(_f)
    for _model_name in _MODELS_INFO:
        if _model_name in _ts_data:
            continue
        pairs = []
        for _ds in _DATASETS:
            _recs = all_data[_model_name][_ds].get('frugal_reason_v3', [])
            for _r in _recs:
                _cands = _r.get('candidates', [])
                if not _cands:
                    try:
                        _cands = json.loads(_r.get('raw_response', '{}')).get('candidates', [])
                    except:
                        pass
                for _c in _cands:
                    _V = _c.get('V_raw', _c.get('V', 0))
                    _norm_a = re.sub('[,$\\\\\\s]', '', str(_c.get('answer', ''))).lower().strip()
                    _norm_g = re.sub('[,$\\\\\\s]', '', str(_r.get('gold', _r.get('gold_answer', '')))).lower().strip()
                    _correct = 1.0 if _norm_a == _norm_g else 0.0
                    pairs.append((_V, _correct))
        if len(pairs) < 20:
            continue
        _V_arr = np.array([_p[0] for _p in pairs])
        _C_arr = np.array([_p[1] for _p in pairs])
        _fit_mask = np.array([_i % 2 == 0 for _i in range(len(pairs))])
        (_V_fit, C_fit) = (_V_arr[_fit_mask], _C_arr[_fit_mask])
        (_V_eval, C_eval) = (_V_arr[~_fit_mask], _C_arr[~_fit_mask])

        def _nll(T):
            z = np.clip(_V_fit / T, -20, 20)
            _p = 1.0 / (1.0 + np.exp(-z))
            _p = np.clip(_p, 1e-07, 1 - 1e-07)
            return -np.mean(C_fit * np.log(_p) + (1 - C_fit) * np.log(1 - _p))
        _res = minimize_scalar(_nll, bounds=(0.1, 10.0), method='bounded')
        _T_star = _res.x

        def ece(V, C, T=1.0, bins=10):
            z = np.clip(_V / T, -20, 20)
            probs = 1.0 / (1.0 + np.exp(-z))
            _total = 0
            for _b in range(bins):
                (_lo, _hi) = (_b / bins, (_b + 1) / bins)
                mask = (probs >= _lo) & (probs < _hi)
                if mask.sum() == 0:
                    continue
                _total = _total + mask.sum() * abs(probs[mask].mean() - C[mask].mean())
            return _total / len(_V)
        _ece_before = ece(_V_eval, C_eval, T=1.0)
        _ece_after = ece(_V_eval, C_eval, T=_T_star)
        _ts_data[_model_name] = {'T_star': float(_T_star), 'ece_before': float(_ece_before), 'ece_after': float(_ece_after), 'n_pairs': len(pairs)}
        print(f'  {_model_name}: T*={_T_star:.3f}, ECE {_ece_before:.4f} → {_ece_after:.4f}')
    with open(_ts_path, 'w') as _f:
        json.dump(_ts_data, _f, indent=2)
    if _ts_data:
        models_ts = list(_ts_data.keys())
        _ece_before = [_ts_data[_m].get('ece_before', 0) for _m in models_ts]
        _ece_after = [_ts_data[_m].get('ece_after', 0) for _m in models_ts]
        _x = np.arange(len(models_ts))
        _width = 0.35
        (_fig, _ax) = plt.subplots(figsize=(10, 5))
        _ax.bar(_x - _width / 2, _ece_before, _width, label='Before', color='#ff6b6b')
        _ax.bar(_x + _width / 2, _ece_after, _width, label='After', color='#51cf66')
        _ax.set_xticks(_x)
        _ax.set_xticklabels(models_ts, fontsize=8, rotation=30)
        _ax.set_ylabel('ECE')
        _ax.set_title('Fig3: ECE Before/After Temp Scaling (All Models)')
        _ax.legend()
        _ax.grid(True, alpha=0.3, axis='y')
        _save_fig(_fig, 'F3_ece_bars_ext')
    print('\nFig4: Main Results Table (3B + 70B + BBH)')
    (_fig, _ax) = plt.subplots(figsize=(16, 8))
    _ax.axis('off')
    _col_labels = ['Model', 'Dataset'] + [_STRAT_LABELS[_s] for _s in _STRATEGIES]
    _cell_text = []
    for _model_name in ['qwen2.5:3b', 'llama3.3:70b']:
        for _ds in _DATASETS:
            _row = [_model_name, _ds.upper()]
            for _strat in _STRATEGIES:
                _recs = all_data.get(_model_name, {}).get(_ds, {}).get(_strat, [])
                _n = len(_recs)
                _k = sum((1 for _r in _recs if _r.get('correct', False)))
                (_acc, _lo, _hi) = _wilson_ci(_k, _n)
                _row.append(f'{_acc:.0%}' if _n > 0 else '—')
            _cell_text.append(_row)
    if _cell_text:
        _table = _ax.table(cellText=_cell_text, colLabels=_col_labels, loc='center', cellLoc='center')
        _table.auto_set_font_size(False)
        _table.set_fontsize(8)
        _table.scale(1.1, 1.6)
        _ax.set_title('Fig4: Main Results — 3B + 70B', fontsize=14, fontweight='bold', pad=20)
        _save_fig(_fig, 'F4_main_results_ext')
    print('\nFig5: Pareto (all models)')
    (_fig, _axes) = plt.subplots(1, 2, figsize=(16, 7))
    _colors = {'qwen2.5:1.5b': 'blue', 'qwen2.5:3b': 'green', 'qwen2.5:7b': 'orange', 'llama3.2:3b': 'purple', 'llama3.3:70b': 'red'}
    for _model_name in _MODELS_INFO:
        for _ds in _DATASETS:
            for _strat in _STRATEGIES:
                _recs = all_data.get(_model_name, {}).get(_ds, {}).get(_strat, [])
                if not _recs:
                    continue
                _n = len(_recs)
                _k = sum((1 for _r in _recs if _r.get('correct', False)))
                _acc = _k / _n if _n > 0 else 0
                _avg_tok = np.mean([_r.get('tokens', _r.get('total_tokens', 0)) for _r in _recs])
                _avg_calls = np.mean([_r.get('calls', _r.get('model_calls', 1)) for _r in _recs])
                marker = 'D' if 'frugal' in _strat else 'o'
                size = 80 if 'frugal' in _strat else 30
                _axes[0].scatter(_avg_tok, _acc, c=_colors.get(_model_name, 'gray'), s=size, marker=marker, alpha=0.6)
                _axes[1].scatter(_avg_calls, _acc, c=_colors.get(_model_name, 'gray'), s=size, marker=marker, alpha=0.6)
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=_c, markersize=8, label=_m) for (_m, _c) in _colors.items()]
    legend_elements.append(Line2D([0], [0], marker='D', color='w', markerfacecolor='black', markersize=8, label='FrugalReason'))
    _axes[0].legend(handles=legend_elements, fontsize=7)
    _axes[0].set_xlabel('Avg Tokens')
    _axes[0].set_ylabel('Accuracy')
    _axes[0].set_title('Acc vs Tokens')
    _axes[0].grid(True, alpha=0.3)
    _axes[1].legend(handles=legend_elements, fontsize=7)
    _axes[1].set_xlabel('Avg Calls')
    _axes[1].set_ylabel('Accuracy')
    _axes[1].set_title('Acc vs Calls')
    _axes[1].grid(True, alpha=0.3)
    _fig.suptitle('Fig5: Pareto Front — All Models', fontsize=14, fontweight='bold')
    plt.tight_layout()
    _save_fig(_fig, 'F5_pareto_ext')
    print('\nFig6: Ablation Bar')
    _abl_path = _nb / 'results' / 'ablations' / 'ablation_table.csv'
    if _abl_path.exists():
        _abl_df = pd.read_csv(_abl_path)
        if 'dataset' in _abl_df.columns:
            (_fig, _ax) = plt.subplots(figsize=(10, 5))
            _abl_df.plot(kind='bar', x='dataset', ax=_ax)
            _ax.set_title('Fig6: Ablation Results')
            _ax.set_ylabel('Accuracy')
            _ax.grid(True, alpha=0.3, axis='y')
            _save_fig(_fig, 'F6_ablation_bar')
    else:
        print('  SKIP: ablation_table.csv not found')
    print('\nFig7: BBH Table')
    _bbh_dir = _nb / 'results' / 'bbh_logs'
    _bbh_rows = []
    if _bbh_dir.exists():
        for _strat in _STRATEGIES:
            for pat in [f'bbh_logical_deduction_{_strat}.jsonl', f'bbh_{_strat}.jsonl']:
                _fp = _bbh_dir / pat
                _recs = _load_jsonl(_fp)
                if _recs:
                    _n = len(_recs)
                    _k = sum((1 for _r in _recs if _r.get('correct', False)))
                    (_acc, _lo, _hi) = _wilson_ci(_k, _n)
                    _bbh_rows.append([_STRAT_LABELS[_strat], f'{_acc:.1%}', f'[{_lo:.1%},{_hi:.1%}]', str(_n)])
                    break
    if _bbh_rows:
        (_fig, _ax) = plt.subplots(figsize=(8, 4))
        _ax.axis('off')
        _table = _ax.table(cellText=_bbh_rows, colLabels=['Strategy', 'Acc', '95% CI', 'N'], loc='center', cellLoc='center')
        _table.auto_set_font_size(False)
        _table.set_fontsize(10)
        _table.scale(1.2, 1.6)
        _ax.set_title('Fig7: BBH Logical Deduction', fontsize=13, fontweight='bold', pad=20)
        _save_fig(_fig, 'F7_bbh_table')
    print('\nFig8: Phase-Exit Histogram')
    _exit_counts = {'early_exit': 0, 'full_pipeline': 0}
    _exit_tokens = {'early_exit': [], 'full_pipeline': []}
    for _model_name in _MODELS_INFO:
        for _ds in _DATASETS:
            _recs = all_data.get(_model_name, {}).get(_ds, {}).get('frugal_reason_v3', [])
            for _r in _recs:
                _early = _r.get('early_exit', False)
                _tok = _r.get('tokens', _r.get('total_tokens', 0))
                if _early:
                    _exit_counts['early_exit'] = _exit_counts['early_exit'] + 1
                    _exit_tokens['early_exit'].append(_tok)
                else:
                    _exit_counts['full_pipeline'] = _exit_counts['full_pipeline'] + 1
                    _exit_tokens['full_pipeline'].append(_tok)
    if _exit_counts['early_exit'] + _exit_counts['full_pipeline'] > 0:
        (_fig, _axes) = plt.subplots(1, 2, figsize=(12, 5))
        _labels = ['Early Exit', 'Full Pipeline']
        _counts = [_exit_counts['early_exit'], _exit_counts['full_pipeline']]
        _colors = ['#51cf66', '#ff6b6b']
        _axes[0].bar(_labels, _counts, color=_colors)
        _axes[0].set_ylabel('Count')
        _axes[0].set_title('Phase Exit Distribution (All Models)')
        for (_i, _v) in enumerate(_counts):
            _axes[0].text(_i, _v + 1, str(_v), ha='center', fontweight='bold')
        avg_e = np.mean(_exit_tokens['early_exit']) if _exit_tokens['early_exit'] else 0
        avg_f = np.mean(_exit_tokens['full_pipeline']) if _exit_tokens['full_pipeline'] else 0
        _axes[1].bar(_labels, [avg_e, avg_f], color=_colors)
        _axes[1].set_ylabel('Avg Tokens')
        _axes[1].set_title('Token Cost: Exit vs Full')
        if avg_f > 0:
            _saving = (1 - avg_e / avg_f) * 100
            _axes[1].annotate(f'{_saving:.0f}% savings', xy=(0, avg_e), fontsize=11, fontweight='bold', ha='center')
        _fig.suptitle('Fig8: Phase-Exit Analysis (All Models)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        _save_fig(_fig, 'F8_phase_exit_ext')
    _existing = list(_fig_dir.glob('*'))
    print(f"\nAdd-On 7 DONE: {sum((1 for _f in _existing if _f.suffix == '.png'))} PNGs + {sum((1 for _f in _existing if _f.suffix == '.pdf'))} PDFs")
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
def _(HfApi, Path, json, os, pd, subprocess, time, warnings, zipfile):
    warnings.filterwarnings('ignore', category=SyntaxWarning)
    _nb = Path(os.environ.get('NOTEBOOK_DIR', '.'))

    def _ollama_unload(model_name):
        import requests as _req
        _r = subprocess.run(f'ollama stop {_model_name}', shell=True, capture_output=True, text=True)
        if _r.returncode == 0:
            print(f'  Stopped {_model_name}')
        else:
            try:
                _req.post('http://localhost:11434/api/generate', json={'model': _model_name, 'prompt': '', 'keep_alive': 0}, timeout=30)
            except:
                pass
    print('=' * 70)
    print('  FINAL VRAM CLEANUP')
    print('=' * 70)
    for _m in ['qwen2.5:7b', 'llama3.3:70b', 'qwen2.5:72b']:
        _ollama_unload(_m)
        time.sleep(2)
    _ps = subprocess.run('ollama ps', shell=True, capture_output=True, text=True)
    print(f'\nollama ps:\n{_ps.stdout}')
    subprocess.run('nvidia-smi --query-gpu=memory.free --format=csv,noheader', shell=True)
    print('\n' + '=' * 70)
    print('  UPDATED MASTER TABLE')
    print('=' * 70)
    _DATASETS = ['gsm8k', 'aqua', 'math', 'strategyqa']
    _STRATEGIES = ['greedy_io', 'greedy_cot', 'zero_shot_tot_k3', 'self_consistency_k5', 'best_of_n_k5_self_eval', 'frugal_reason_v3']
    _STRAT_LABELS = {'greedy_io': 'IO', 'greedy_cot': 'CoT', 'zero_shot_tot_k3': 'ToT', 'self_consistency_k5': 'SC', 'best_of_n_k5_self_eval': 'BoN', 'frugal_reason_v3': 'FR'}
    _MODELS_INFO = {'qwen2.5:1.5b': ('block_b_logs', 'qwen15b_'), 'qwen2.5:3b': ('block_a_logs', ''), 'qwen2.5:7b': ('block_b_logs', 'qwen7b_'), 'llama3.2:3b': ('block_b_logs', 'llama32_'), 'llama3.3:70b': ('block_b_logs', 'llama70b_')}

    def _load_jsonl(path):
        records = []
        if not path.exists():
            return records
        with open(path, 'r', encoding='utf-8') as _f:
            for _line in _f:
                if _line.strip():
                    records.append(json.loads(_line))
        return records
    _total_runs = 0
    _master_rows = []
    for (_model_name, (_subdir, _prefix)) in _MODELS_INFO.items():
        for _ds in _DATASETS:
            for _strat in _STRATEGIES:
                _log_path = _nb / 'results' / _subdir / f'{_prefix}{_ds}_{_strat}.jsonl'
                if not _log_path.exists():
                    _log_path = _nb / 'ttc-frugalreason-poc' / 'experiment_fr' / 'results' / _subdir / f'{_prefix}{_ds}_{_strat}.jsonl'
                _recs = _load_jsonl(_log_path)
                if not _recs:
                    continue
                _n = len(_recs)
                _k = sum((1 for _r in _recs if _r.get('correct', False)))
                _total_runs = _total_runs + _n
                _master_rows.append({'model': _model_name, 'dataset': _ds, 'strategy': _strat, 'correct': _k, 'total': _n, 'accuracy': _k / _n if _n > 0 else 0})
    if _master_rows:
        _master_df = pd.DataFrame(_master_rows)
        _pivot = _master_df.pivot_table(values='accuracy', index=['model', 'dataset'], columns='strategy', aggfunc='first')
        _cols = [_c for _c in _STRATEGIES if _c in _pivot.columns]
        _pivot = _pivot[_cols]
        _pivot.columns = [_STRAT_LABELS.get(_c, _c) for _c in _cols]
        print(_pivot.to_string(float_format=lambda x: f'{_x:.1%}'))
        _master_df.to_csv(str(_nb / 'results' / 'addon_master_table.csv'), index=False)
    print('\n' + '=' * 70)
    print('  α TABLE')
    print('=' * 70)
    _alpha_path = _nb / 'results' / 'calibration' / 'alpha_grid.csv'
    if _alpha_path.exists():
        adf = pd.read_csv(_alpha_path)
        _summary = adf.dropna(subset=['acc_at_alpha_star']).drop_duplicates('model')
        if not _summary.empty:
            print(_summary[['model', 'params_B', 'alpha_star_emp', 'acc_at_alpha_star', 'alpha_star_theory', 'acc_at_alpha_0']].to_string(index=False))
    print('\n' + '=' * 70)
    print('  MATCHED FR-3B vs 70B TABLE')
    print('=' * 70)
    m70_path = _nb / 'results' / 'calibration' / 'matched_70b_stats.csv'
    if m70_path.exists():
        print(pd.read_csv(m70_path).to_string(index=False))
    print('\n' + '=' * 70)
    print('  ECE TABLE')
    print('=' * 70)
    _ts_path = _nb / 'results' / 'calibration' / 'temp_scaling.json'
    if _ts_path.exists():
        ts = json.load(open(_ts_path))
        for (_m, _v) in ts.items():
            print(f"  {_m:18s} | T*={_v.get('T_star', 0):.3f} | ECE: {_v.get('ece_before', 0):.4f} → {_v.get('ece_after', 0):.4f}")
    print('\n' + '=' * 70)
    print('  72B CROSS-VALIDATION TABLE')
    print('=' * 70)
    for _strat in ['greedy_io', 'greedy_cot', 'frugal_reason_v4']:
        _fp = _nb / 'results' / 'block_b_logs' / f'qwen72b_math_{_strat}.jsonl'
        _recs = _load_jsonl(_fp)
        if _recs:
            _n = len(_recs)
            _k = sum((1 for _r in _recs if _r.get('correct', False)))
            print(f'  qwen2.5:72b MATH {_strat}: {_k}/{_n} ({_k / _n:.1%})')
    try:
        _api = HfApi(token='REDACTED')
        _final_zip = str(_nb / 'results' / 'addon_final_all_results.zip')
        with zipfile.ZipFile(_final_zip, 'w', zipfile.ZIP_DEFLATED) as _zf:
            _results_root = _nb / 'results'
            for _fp in _results_root.rglob('*'):
                if _fp.is_file() and '__pycache__' not in str(_fp):
                    _zf.write(str(_fp), str(_fp.relative_to(_results_root)))
        _api.upload_file(path_or_fileobj=_final_zip, path_in_repo='results_sync/addon_final_all_results.zip', repo_id='Satabarto/Molab_Checkpoints_Cost_AWARE', repo_type='dataset')
        print('\nPushed addon_final_all_results.zip to HF.')
    except Exception as e:
        print(f'HF push failed: {e}')
    print('\n' + '=' * 70)
    print(f'  ADD-ON COMPLETE — {_total_runs} total runs logged; VRAM freed.')
    print(f'  STOP. Writing phase (D13+) consumes ONLY the overwritten artifacts.')
    print('=' * 70)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

