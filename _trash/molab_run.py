"""
molab_run.py — Single-file launcher for Molab. Paste into one cell and run.

What it does (in order, fully automatic):
  1. Creates /workspace if missing
  2. Installs zstd + curl (needed by Ollama installer)
  3. Installs Ollama binary
  4. Starts Ollama server (uses Molab's RTX 6000 GPU)
  5. Pulls qwen2.5:3b onto the GPU
  6. Installs Python dependencies
  7. Configures rclone for Google Drive checkpointing (if RCLONE_CONF is set)
  8. Verifies everything is ready
  9. Runs the pipeline — output streams live in this cell

CHECKPOINTING:
  After every completed (task, question, strategy) the checkpoint is pushed to:
    gdrive:Molab_Backups/Cost-Aware-Test-Time/checkpoints/completed.jsonl
  On the next session, the pipeline pulls that file and skips done work.
  Edit RCLONE_CONF in auto_backup.py once to enable this.
  Without it, checkpoints are local only (lost when session ends).

HOW TO USE:
  1. Upload Cost-Aware-Test-Time-upload.zip to Molab
  2. Extract it in a cell:
       import zipfile
       with zipfile.ZipFile("Cost-Aware-Test-Time-upload.zip","r") as z:
           z.extractall(".")
  3. Run this file:
       exec(open("molab_run.py").read())
"""

import os, sys, subprocess, time, shutil
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
OLLAMA_MODEL  = "qwen2.5:3b"
FALLBACK_MODEL = "llama3.2:3b"
# Leave OLLAMA_BASE_URL empty — we run Ollama locally on the RTX 6000
OLLAMA_BASE_URL = "http://localhost:11434"
# ─────────────────────────────────────────────────────────────────────────────

def _run(cmd):
    """Run shell command, print it, stream output."""
    print(f"  $ {cmd}")
    subprocess.run(cmd, shell=True)

def _run_check(cmd):
    """Run shell command silently, return exit code."""
    return subprocess.run(cmd, shell=True, capture_output=True).returncode

def _section(n, title):
    print(f"\n{'='*60}\n  [{n}] {title}\n{'='*60}")

# ── Locate project root (where this script lives) ─────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE   = SCRIPT_DIR / "rq2_part1" / "run_rq2_part1.py"

print("=" * 60)
print("  MOLAB RUN — Cost-Aware-Test-Time")
print("=" * 60)
print(f"  Project root : {SCRIPT_DIR}")
print(f"  Pipeline     : {PIPELINE}")

if not PIPELINE.exists():
    print(f"\n  ERROR: Pipeline not found at {PIPELINE}")
    print("  Make sure Cost-Aware-Test-Time-upload.zip was extracted correctly.")
    sys.exit(1)
print("  ✅ Project structure OK")

# ── Step 1: Ensure /workspace exists ─────────────────────────────────────────
_section(1, "Workspace directory")
os.makedirs("/workspace", exist_ok=True)
print("  ✅ /workspace ready")

# ── Step 2: System packages ───────────────────────────────────────────────────
_section(2, "System packages (zstd, curl)")
_run("apt-get update -qq && apt-get install -y -qq zstd curl > /dev/null 2>&1")
print("  ✅ System packages ready")

# ── Step 3: Install Ollama ────────────────────────────────────────────────────
_section(3, "Ollama installation")
os.environ["PATH"] = "/usr/local/bin:/usr/bin:/bin:" + os.environ.get("PATH", "")

if _run_check("test -f /usr/local/bin/ollama") == 0:
    print("  Ollama already installed at /usr/local/bin/ollama")
else:
    print("  Installing Ollama...")
    _run("curl -fsSL https://ollama.com/install.sh | sh")

# Resolve binary — never fall back to bare string "ollama"
ollama_bin = None
for candidate in ["/usr/local/bin/ollama", "/usr/bin/ollama"]:
    if Path(candidate).is_file():
        ollama_bin = candidate
        break
if ollama_bin is None:
    r = subprocess.run("which ollama", shell=True, capture_output=True, text=True)
    ollama_bin = r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else None

if ollama_bin is None:
    print("  FATAL: Ollama binary not found after install.")
    print("  Check the install output above for errors.")
    sys.exit(1)
print(f"  ✅ Ollama binary: {ollama_bin}")

# ── Step 4: Start Ollama server ───────────────────────────────────────────────
_section(4, "Starting Ollama server (RTX 6000)")
_run_check("pkill -f 'ollama serve' 2>/dev/null")
time.sleep(2)

subprocess.Popen(
    f"{ollama_bin} serve >> /workspace/ollama.log 2>&1",
    shell=True
)

print("  Waiting for Ollama API to be ready...")
ollama_ready = False
for i in range(20):
    time.sleep(2)
    if _run_check("curl -sf http://localhost:11434/api/tags > /dev/null") == 0:
        print(f"  ✅ Ollama server ready (took {(i+1)*2}s)")
        ollama_ready = True
        break

if not ollama_ready:
    print("  ❌ Ollama did not start in 40s. Check /workspace/ollama.log")
    _run("tail -20 /workspace/ollama.log")
    sys.exit(1)

# ── Step 5: Pull model ────────────────────────────────────────────────────────
_section(5, f"Pulling model: {OLLAMA_MODEL}")

# Check if already pulled
r = subprocess.run(
    f"curl -sf http://localhost:11434/api/tags",
    shell=True, capture_output=True, text=True
)
already_have = OLLAMA_MODEL.split(":")[0] in r.stdout if r.returncode == 0 else False

if already_have:
    print(f"  {OLLAMA_MODEL} already available, skipping pull")
else:
    rc = _run_check(f"{ollama_bin} pull {OLLAMA_MODEL}")
    if rc != 0:
        print(f"  Preferred model failed, trying fallback: {FALLBACK_MODEL}")
        rc2 = _run_check(f"{ollama_bin} pull {FALLBACK_MODEL}")
        if rc2 != 0:
            print("  FATAL: Both model pulls failed.")
            sys.exit(1)
        OLLAMA_MODEL = FALLBACK_MODEL
        print(f"  ✅ Using fallback model: {OLLAMA_MODEL}")
    else:
        print(f"  ✅ {OLLAMA_MODEL} ready on GPU")

# ── Step 6: Python dependencies ───────────────────────────────────────────────
_section(6, "Python dependencies")
req = SCRIPT_DIR / "requirements_molab.txt"
if req.exists():
    _run(f"pip install -q --root-user-action=ignore -r {req}")
else:
    pkgs = (
        "requests datasets pandas numpy matplotlib seaborn tqdm "
        "pynvml psutil pyyaml tabulate reportlab scipy fpdf2"
    )
    _run(f"pip install -q --root-user-action=ignore {pkgs}")
print("  ✅ Dependencies ready")

# ── Step 7: rclone for Drive checkpointing ────────────────────────────────────
_section(7, "Google Drive checkpointing (rclone)")

if shutil.which("rclone") is None:
    print("  Installing rclone...")
    _run("curl -fsSL https://rclone.org/install.sh | sudo bash > /dev/null 2>&1")

# Load RCLONE_CONF from auto_backup.py
DRIVE_SYNC = False
try:
    import importlib.util
    sys.path.insert(0, str(SCRIPT_DIR))
    spec = importlib.util.spec_from_file_location(
        "auto_backup", SCRIPT_DIR / "auto_backup.py"
    )
    ab = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ab)
    rclone_conf = ab.RCLONE_CONF.strip()

    if "YOUR_CLIENT_ID_HERE" in rclone_conf or "YOUR_ACCESS_TOKEN" in rclone_conf:
        print("  ⚠️  RCLONE_CONF not filled in — Drive sync disabled.")
        print("      Checkpoints will be LOCAL ONLY (lost when session ends).")
        print("      To fix: paste your rclone.conf into RCLONE_CONF in auto_backup.py")
    else:
        conf_path = Path("/root/.config/rclone/rclone.conf")
        conf_path.parent.mkdir(parents=True, exist_ok=True)
        conf_path.write_text(rclone_conf, encoding="utf-8")
        # Quick test
        test = subprocess.run(
            "rclone lsd gdrive: --max-depth 1",
            shell=True, capture_output=True, text=True, timeout=15
        )
        if test.returncode == 0:
            print("  ✅ Google Drive connected — checkpoints will sync after every question")
            DRIVE_SYNC = True
        else:
            print(f"  ⚠️  rclone config found but Drive test failed: {test.stderr.strip()[:100]}")
            print("      Checkpoints will be LOCAL ONLY.")
except Exception as e:
    print(f"  ⚠️  rclone setup error: {e}")
    print("      Checkpoints will be LOCAL ONLY.")

# ── Step 8: Set environment variables ─────────────────────────────────────────
_section(8, "Environment")
os.environ["OLLAMA_BASE_URL"] = OLLAMA_BASE_URL
os.environ["OLLAMA_MODEL"]    = OLLAMA_MODEL
os.environ["WORKSPACE"]       = str(SCRIPT_DIR)
print(f"  OLLAMA_BASE_URL = {OLLAMA_BASE_URL}")
print(f"  OLLAMA_MODEL    = {OLLAMA_MODEL}")
print(f"  WORKSPACE       = {SCRIPT_DIR}")

# ── Step 9: Final pre-flight check ────────────────────────────────────────────
_section(9, "Pre-flight check")

import requests as _req
try:
    resp = _req.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
    models = [m["name"] for m in resp.json().get("models", [])]
    print(f"  Ollama   : ✅ running")
    print(f"  Models   : {models}")
    print(f"  Pipeline : {PIPELINE}")
    print(f"  Drive    : {'✅ enabled' if DRIVE_SYNC else '⚠️  local only'}")
except Exception as e:
    print(f"  ❌ Ollama not responding: {e}")
    sys.exit(1)

# ── Step 10: Launch pipeline ──────────────────────────────────────────────────
_section(10, "Launching pipeline — output streams below")
print("  Model is running on Molab's GPU (RTX 6000)")
print("  Ctrl+C to stop. Checkpoints saved — resume anytime.\n")
print("=" * 60 + "\n")

os.chdir(PIPELINE.parent)

import subprocess
try:
    subprocess.run([sys.executable, PIPELINE.name], check=True)
except subprocess.CalledProcessError as e:
    if e.returncode not in (None, 0):
        print(f"\nPipeline exited with code {e.returncode}")
except KeyboardInterrupt:
    print("\n\nStopped by user. Checkpoint saved — re-run molab_run.py to resume.")
except Exception as e:
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("  Pipeline complete.")
if DRIVE_SYNC:
    print("  All results checkpointed to Google Drive.")
else:
    print("  Results saved locally in rq2_part1/results/")
    print("  Configure RCLONE_CONF in auto_backup.py for Drive sync.")
print("=" * 60)
