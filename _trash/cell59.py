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
    ps2 = subprocess.run("ollama ps", shell=True, capture_output=True, text=True)
    assert "70b" not in ps2.stdout.lower(), f"FATAL: 70B still loaded after stop!\n{ps2.stdout}"
    print("70B successfully unloaded.")
else:
        print(f"  ollama stop failed")
    time.sleep(3)
    ps = subprocess.run("ollama ps", shell=True, capture_output=True, text=True)
    if model_name.split(":")[0] in ps.stdout:
        print(f"  WARNING: {model_name} still loaded! Output:\n{ps.stdout}")
    else:
        print(f"  CONFIRMED: {model_name} unloaded from VRAM")
    subprocess.run("nvidia-smi --query-gpu=memory.free --format=csv,noheader", shell=True)

_ollama_unload(MODEL)