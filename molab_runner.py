"""
molab_runner.py — Paste the section you want to run into a Molab cell.
Or run sections top to bottom in separate Molab cells.
Generated from molab_run.ipynb
"""

# ========================================================================
# CELL 1: ---
# ========================================================================
import subprocess, zipfile, os, sys
from pathlib import Path

GDRIVE_FILE_ID = "1F7lBEOBDC9FNHyK-gjOLoEQ24SpzC0tr"
ZIP_NAME = "Cost-Aware-Test-Time-upload.zip"

if not Path(ZIP_NAME).exists():
    print("Installing gdown...")
    subprocess.run("pip install -q gdown", shell=True)
    import gdown
    print(f"Downloading {ZIP_NAME} from Google Drive...")
    gdown.download(id=GDRIVE_FILE_ID, output=ZIP_NAME, quiet=False)
else:
    print(f"{ZIP_NAME} already exists, skipping download")

if Path(ZIP_NAME).exists():
    print(f"Extracting {ZIP_NAME}...")
    with zipfile.ZipFile(ZIP_NAME, "r") as z:
        z.extractall(".")
    os.environ["NOTEBOOK_DIR"] = str(Path(".").resolve())
    print(f"Extracted. NOTEBOOK_DIR = {os.environ['NOTEBOOK_DIR']}")
    checks = [
        "rq2_part1/run_rq2_part1.py",
        "ttc-frugalreason-poc/experiment_fr/run_frugal_smoke.py",
        "ttc-frugalreason-poc/experiment_fr/run_day0.py",
        "ttc-frugalreason-poc/experiment_fr/run_real_experiment.py",
        "ttc-task-poc/experiment/run_poc.py",
        "auto_backup.py", "drive_checkpoint.py", "requirements_molab.txt"
    ]
    all_ok = True
    for f in checks:
        exists = Path(f).exists()
        print(f"  {'OK     ' if exists else 'MISSING'}: {f}")
        if not exists: all_ok = False
    print("\nAll files present. Run Cell 2." if all_ok else "\nSome files missing.")
else:
    print(f"ERROR: {ZIP_NAME} not found. Make sure Drive file is shared publicly.")

# ========================================================================
# CELL 2: ---
# ========================================================================
import subprocess, os, sys, time, shutil, importlib.util
from pathlib import Path

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

subprocess.run("pkill -f 'ollama serve' 2>/dev/null || true", shell=True)
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
DRIVE_SYNC = False
try:
    spec = importlib.util.spec_from_file_location("auto_backup", str(Path(nb_dir)/"auto_backup.py"))
    ab = importlib.util.module_from_spec(spec); spec.loader.exec_module(ab)
    conf = ab.RCLONE_CONF.strip()
    if "YOUR_CLIENT_ID_HERE" in conf or "YOUR_ACCESS_TOKEN" in conf:
        print("[6/6] Drive sync: DISABLED")
    else:
        cp = Path("/root/.config/rclone/rclone.conf"); cp.parent.mkdir(parents=True, exist_ok=True); cp.write_text(conf)
        test = subprocess.run("rclone lsd gdrive: --max-depth 1", shell=True, capture_output=True, text=True, timeout=15)
        if test.returncode == 0: print("[6/6] Drive sync: ENABLED"); DRIVE_SYNC = True
        else: print(f"[6/6] Drive sync failed: {test.stderr.strip()[:80]}")
except Exception as e: print(f"[6/6] Drive sync error: {e}")

os.environ["OLLAMA_BASE_URL"] = "http://localhost:11434"
os.environ["OLLAMA_MODEL"]    = OLLAMA_MODEL
os.environ["WORKSPACE"]       = nb_dir

print("\n" + "="*60)
print("  SETUP COMPLETE — pick a run cell below")
print("="*60)
print(f"  Model     : {OLLAMA_MODEL} on RTX 6000")
print(f"  Drive sync: {'ENABLED' if DRIVE_SYNC else 'DISABLED'}")
print(f"  Workspace : {nb_dir}")
print("="*60)

# ========================================================================
# CELL 3: Smoke A — frugal_reason_v3 only · 10 q × 4 datasets = 40 runs
# ========================================================================
import os, sys, runpy
from pathlib import Path

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

# ========================================================================
# CELL 4: ---
# ========================================================================
import os, sys, runpy
from pathlib import Path

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

# ========================================================================
# CELL 5: ---
# ========================================================================
import os, sys, runpy
from pathlib import Path

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

# ========================================================================
# CELL 6: Full Run 1 — rq2_part1 · 5 strategies × 3 datasets × 36 q = **540 runs** ← MAIN
# ========================================================================
import os, sys, runpy
from pathlib import Path

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

# ========================================================================
# CELL 7: ---
# ========================================================================
import os, sys, runpy
from pathlib import Path

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

# ========================================================================
# CELL 8: ---
# ========================================================================
import os, sys, runpy
from pathlib import Path

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

# ========================================================================
# CELL 9: ---
# ========================================================================
import os, sys, runpy
from pathlib import Path

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

# ========================================================================
# CELL 10: ---
# ========================================================================
import os, sys, runpy
from pathlib import Path

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

# ========================================================================
# CELL 11: ---
# ========================================================================
import subprocess, os
from pathlib import Path
from datetime import datetime, timezone

# ── SET YOUR TOKEN ONCE ───────────────────────────────────────────
GITHUB_TOKEN = "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"  # PAT — repo scope   # paste your PAT here (github.com → Settings → Developer settings → PAT)
GITHUB_REPO  = "https://github.com/satabartosarkar123/cost-aware-ttc-molab.git"
# ─────────────────────────────────────────────────────────────────

token = GITHUB_TOKEN or os.environ.get("GITHUB_TOKEN", "")
nb_dir = Path(os.environ.get("NOTEBOOK_DIR", "."))

if not token:
    print("ERROR: Set GITHUB_TOKEN above — get it from github.com → Settings → Developer settings → Personal access tokens → Classic → repo scope")
else:
    auth_url = GITHUB_REPO.replace("https://", f"https://{token}@")

    def git(cmd): return subprocess.run(cmd, shell=True, cwd=str(nb_dir), capture_output=True, text=True)

    # Init repo if not already
    if not (nb_dir / ".git").exists():
        print("Initialising git repo...")
        git("git init -b main")
        git(f"git remote add origin {auth_url}")
        git('git config user.name "Molab Runner"')
        git('git config user.email "molab@runner.local"')
    else:
        git(f"git remote set-url origin {auth_url}")

    # Stage result dirs
    RESULT_DIRS = [
        "rq2_part1/results", "rq2_part1/checkpoints", "rq2_part1/reports", "rq2_part1/plots",
        "ttc-frugalreason-poc/experiment_fr/results", "ttc-frugalreason-poc/experiment_fr/reports",
        "ttc-task-poc/experiment/results_50", "ttc-task-poc/experiment/reports_50",
    ]
    staged = 0
    for d in RESULT_DIRS:
        if (nb_dir / d).exists():
            git(f"git add {d}")
            staged += 1

    if staged == 0:
        print("Nothing to push — no result directories found yet.")
    else:
        diff = git("git diff --cached --quiet")
        if diff.returncode == 0:
            print("No new changes since last push.")
        else:
            ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            msg = f"Molab results — {ts}"
            git(f'git commit -m "{msg}"')

            # Push — handle first push (no upstream yet)
            r = git("git push origin main")
            if r.returncode != 0:
                r2 = git("git push -u origin main")
                if r2.returncode != 0:
                    print(f"Push failed: {r2.stderr.strip()}")
                else:
                    print(f"Pushed: {msg}")
            else:
                print(f"Pushed: {msg}")

            print("\nOn your LOCAL PC, run:")
            print("  python sync_from_molab.py")
            print("or simply: git pull")

# ========================================================================
# CELL 12: ---
# ========================================================================
import json, os
from pathlib import Path
from collections import Counter

nb_dir = Path(os.environ.get("NOTEBOOK_DIR", "."))
ckpt = nb_dir / "rq2_part1/checkpoints/completed.jsonl"

if ckpt.exists():
    done = ckpt.read_text().strip().splitlines()
    total = 540
    pct = len(done) / total * 100
    filled = int(pct / 2)
    print(f"rq2_part1: {len(done)}/{total} ({pct:.1f}%)")
    print(f"  [{'#'*filled}{'.'*(50-filled)}]")
    if done:
        last = json.loads(done[-1])
        print(f"  Last: {last['task']} q{last['local_question_id']} {last['strategy']} [{last['status']}]")
    print(f"  By task    : {dict(Counter(json.loads(l)['task'] for l in done))}")
    print(f"  By strategy: {dict(Counter(json.loads(l)['strategy'] for l in done))}")
else:
    print("No checkpoint yet.")

prog = nb_dir / "rq2_part1/results/progress_log.jsonl"
if prog.exists():
    lines = prog.read_text().strip().splitlines()
    print("\nLast 3 progress entries:")
    for l in lines[-3:]:
        d = json.loads(l)
        print(f"  {d.get('task','?'):12s} q{str(d.get('local_question_id','?')):>2}  {d.get('percent_questions_completed_for_run','?')}%")
