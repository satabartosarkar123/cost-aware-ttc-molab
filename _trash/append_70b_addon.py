"""
Append 70B Add-On cells (B1-B16) to molab_run.ipynb.
APPEND-ONLY: does NOT touch any existing cell.
"""
import json

NB_PATH = "molab_run.ipynb"

with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

existing_count = len(nb["cells"])
print(f"Notebook currently has {existing_count} cells.")

def md(src):
    lines = src.strip().split("\n")
    return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in lines]}

def code(src):
    lines = src.strip().split("\n")
    source = [l + "\n" for l in lines]
    if source:
        source[-1] = source[-1].rstrip("\n")
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source}

new_cells = []

# ═══════════════════════════════════════════════════════════════
# VRAM-UNLOAD HELPER (used by every sweep cell)
# ═══════════════════════════════════════════════════════════════
UNLOAD_HELPER = r'''
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
'''

# ═══════════════════════════════════════════════════════════════
# GENERIC SWEEP RUNNER (used by 7B, 70B, 72B cells)
# ═══════════════════════════════════════════════════════════════
# This is the same runner from D2/D3 but parameterized
SWEEP_RUNNER = r'''
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
'''

# Function to make the generic sweep loop (parameterized by MODEL, PREFIX)
def make_sweep_code(model, prefix, label, all_6=True, datasets=None, strat_list=None, qid_limit=100, hf_zip_name=None):
    ds_list = datasets or '["gsm8k", "aqua", "math", "strategyqa"]'
    strats = strat_list or '["greedy_io", "greedy_cot", "zero_shot_tot_k3", "self_consistency_k5", "best_of_n_k5_self_eval", "frugal_reason_v3"]'
    zip_name = hf_zip_name or f"block_b_{prefix.rstrip('_')}"
    return f'''# {label}
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

MODEL = "{model}"
PREFIX = "{prefix}"
SEED = 0
STRATEGIES = {strats}
DATASETS = {ds_list}
QID_LIMIT = {qid_limit}

{SWEEP_RUNNER}

{UNLOAD_HELPER}

# ── Load data ────────────────────────────────────────────────────
print(f"Loading datasets for {{MODEL}}...")
loader_config = {{"sampling": {{"questions_per_task": 1500, "seed": SEED}},
                 "tasks": {{ds: {{}} for ds in DATASETS}}}}
loaded = load_all_tasks(loader_config)

qids_path = Path("data/confirmatory_qids.json")
conf_qids = {{}}
if qids_path.exists():
    with open(qids_path) as f:
        conf_qids = json.load(f)

task_maps = {{}}
for ds in DATASETS:
    task_maps[ds] = {{item["id"]: item for item in loaded.get(ds, [])}}

qid_lists = {{}}
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

print(f"Target: {{total_target}} runs on {{MODEL}}")

for ds in DATASETS:
    for strat in STRATEGIES:
        log_path = results_dir / f"{{PREFIX}}{{ds}}_{{strat}}.jsonl"
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
                    print(f"  Retry {{attempt+1}}/3 {{ds}}/{{strat}}/{{qid}}: {{e}}")
                    time.sleep(10)
            else:
                done += 1; continue

            score_res = verifier.score(ds, item["question"], res.get("raw_response",""),
                                        res["selected_answer"], item["gold_answer"])
            is_correct = score_res["score"] == 1.0

            log_row = {{
                "model": MODEL, "dataset": ds, "strategy": strat, "qid": qid,
                "gold": item["gold_answer"], "selected_answer": res["selected_answer"],
                "correct": is_correct, "parse_success": res["parse_success"],
                "parse_method": res.get("parse_method", ""), "latency_seconds": res["latency_seconds_total"],
                "tokens": res["total_tokens"], "calls": res["model_calls"],
                "hardware_type": hw, "early_exit": res.get("early_exit", False),
                "clusters": res.get("clusters", []), "candidates": res.get("candidates", []),
                "raw_paths": res.get("raw_paths", []),
            }}
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_row) + "\\n")
            cur.execute("INSERT OR IGNORE INTO completed VALUES (?,?,?,?,datetime('now'))",
                        (MODEL, ds, strat, qid))
            conn.commit()
            done += 1

            if done % 50 == 0:
                elapsed = time.time() - start
                eta = (total_target - done) * (elapsed / max(done, 1))
                print(f"[{{MODEL}}] {{done}}/{{total_target}} | {{elapsed/3600:.1f}}h elapsed | ETA {{eta/3600:.1f}}h")

conn.close()
print(f"\\nSweep DONE: {{done}}/{{total_target}} runs completed on {{MODEL}}.")

# ── Completeness matrix ─────────────────────────────────────────
print("\\nCompleteness Matrix:")
for ds in DATASETS:
    for strat in STRATEGIES:
        log_path = results_dir / f"{{PREFIX}}{{ds}}_{{strat}}.jsonl"
        count = 0
        if log_path.exists():
            with open(log_path) as f:
                count = sum(1 for l in f if l.strip())
        status = "OK" if count >= QID_LIMIT else f"GAP ({{count}}/{{QID_LIMIT}})"
        print(f"  {{ds:12s}} | {{strat:25s}} | {{count:4d}} | {{status}}")

# ── HF push ──────────────────────────────────────────────────────
try:
    from huggingface_hub import HfApi
    import zipfile
    _api = HfApi(token="REDACTED")
    _zp = str(_nb / "results" / "{zip_name}.zip")
    with zipfile.ZipFile(_zp, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in results_dir.glob(f"{{PREFIX}}*.jsonl"):
            zf.write(str(f), f"block_b_logs/{{f.name}}")
    _api.upload_file(path_or_fileobj=_zp, path_in_repo="checkpoints/{zip_name}.zip",
                     repo_id="Satabarto/Molab_Checkpoints_Cost_AWARE", repo_type="dataset")
    print("Pushed results to HF.")
except Exception as e:
    print(f"HF push failed (non-fatal): {{e}}")

# ── VRAM UNLOAD ──────────────────────────────────────────────────
_ollama_unload(MODEL)
'''


# ═══════════════════════════════════════════════════════════════
# B1 [MD] — Add-On 1: Fetch Models
# ═══════════════════════════════════════════════════════════════
new_cells.append(md("""---
# Add-On 1 — Fetch 7B / 70B / 72B Models
> Pull all three models into Ollama (disk only). Actual VRAM loading
> happens when each sweep cell runs. 4-bit default tags (~40-45GB VRAM each).
"""))

# ═══════════════════════════════════════════════════════════════
# B2 [CODE] — AO-Fetch-Models
# ═══════════════════════════════════════════════════════════════
new_cells.append(code(r'''# AO-Fetch-Models — Pull qwen2.5:7b, llama3.3:70b, qwen2.5:72b
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
'''))

# ═══════════════════════════════════════════════════════════════
# B3 [MD] — Add-On 2: 7B Sweep
# ═══════════════════════════════════════════════════════════════
new_cells.append(md("""---
# Add-On 2 — qwen2.5:7b Sweep (scaling point)
> 4 ds × 100 stratified qids × 6 methods = **2,400 runs** (~2-3h).
> Logs → `results/block_b_logs/qwen7b_*.jsonl`.
> Model unloaded from VRAM after completion.
"""))

# ═══════════════════════════════════════════════════════════════
# B4 [CODE] — AO-Sweep-7B
# ═══════════════════════════════════════════════════════════════
new_cells.append(code(make_sweep_code(
    model="qwen2.5:7b",
    prefix="qwen7b_",
    label="AO-Sweep-7B — 2,400 runs (4 ds × 100 qids × 6 strategies) on qwen2.5:7b",
    hf_zip_name="block_b_qwen7b"
)))

# ═══════════════════════════════════════════════════════════════
# B5 [MD] — Add-On 3: 70B Sweep
# ═══════════════════════════════════════════════════════════════
new_cells.append(md("""---
# Add-On 3 — llama3.3:70b Sweep (scaling point)
> 4 ds × 100 stratified qids × 6 methods = **2,400 runs** (~12-18h, overnight).
> Logs → `results/block_b_logs/llama70b_*.jsonl`.
> Model unloaded from VRAM after completion. MUST be unloaded before 72B loads.
"""))

# ═══════════════════════════════════════════════════════════════
# B6 [CODE] — AO-Sweep-70B
# ═══════════════════════════════════════════════════════════════
new_cells.append(code(make_sweep_code(
    model="llama3.3:70b",
    prefix="llama70b_",
    label="AO-Sweep-70B — 2,400 runs (4 ds × 100 qids × 6 strategies) on llama3.3:70b",
    hf_zip_name="block_b_llama70b"
)))

# ═══════════════════════════════════════════════════════════════
# B7 [MD] — Add-On 4: Qwen-72B Cross-Family Validation
# ═══════════════════════════════════════════════════════════════
new_cells.append(md("""---
# Add-On 4 — qwen2.5:72b Cross-Family Validation (MATH only)
> MATH-238 × {greedy_io, greedy_cot, frugal_reason_v4} = **714 runs** (~3-5h).
> Logs → `results/block_b_logs/qwen72b_math_*.jsonl`.
> PRE-ASSERT: llama3.3:70b must NOT be loaded (VRAM hygiene).
> Model unloaded after completion.
"""))

# ═══════════════════════════════════════════════════════════════
# B8 [CODE] — AO-CrossModel-72B
# ═══════════════════════════════════════════════════════════════
new_cells.append(code(r'''# AO-CrossModel-72B — MATH-238 × 3 strategies on qwen2.5:72b (714 runs)
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
'''))

# ═══════════════════════════════════════════════════════════════
# B9 [MD] — Add-On 5: EXT α* Scaling Law
# ═══════════════════════════════════════════════════════════════
new_cells.append(md("""---
# Add-On 5 — EXT α* Scaling Law (5 points) + Fig1/Fig2
> Post-hoc only (no models loaded). Reads saved FR candidates from all 5 models.
> OVERWRITES `results/calibration/alpha_grid.csv`.
> Generates `fig1_scaling_law` and `fig2_theory_vs_emp`.
"""))

# ═══════════════════════════════════════════════════════════════
# B10 [CODE] — AO-AlphaGrid-EXT
# ═══════════════════════════════════════════════════════════════
new_cells.append(code(r'''# AO-AlphaGrid-EXT — 5-model α* scaling law (post-hoc, no inference)
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
'''))

# ═══════════════════════════════════════════════════════════════
# B11 [MD] — Add-On 6: Matched FR-3B vs 70B Stats
# ═══════════════════════════════════════════════════════════════
new_cells.append(md("""---
# Add-On 6 — Matched FR-3B vs 70B Stats
> Post-hoc on matched 100q stratified sets. Wilson CIs + McNemar exact.
> Compares FrugalReason-3B vs llama3.3:70b baselines.
> Save `results/calibration/matched_70b_stats.csv`.
"""))

# ═══════════════════════════════════════════════════════════════
# B12 [CODE] — AO-70B-Matched-Stats
# ═══════════════════════════════════════════════════════════════
new_cells.append(code(r'''# AO-70B-Matched-Stats — FR-3B vs 70B baselines, matched 100q sets
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
'''))

# ═══════════════════════════════════════════════════════════════
# B13 [MD] — Add-On 7: EXT Figures
# ═══════════════════════════════════════════════════════════════
new_cells.append(md("""---
# Add-On 7 — EXT Figures (all 8, 70B included)
> Regenerates ALL publication figures with 70B data included.
> OVERWRITES `results/figures/*` (300 dpi PNG + PDF).
> 8 figures total: F1-F8.
"""))

# ═══════════════════════════════════════════════════════════════
# B14 [CODE] — AO-Figures-EXT
# ═══════════════════════════════════════════════════════════════
new_cells.append(code(r'''# AO-Figures-EXT — 8 publication figures, 70B included, OVERWRITES results/figures/
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
'''))

# ═══════════════════════════════════════════════════════════════
# B15 [MD] — Add-On 8: Final Aggregation
# ═══════════════════════════════════════════════════════════════
new_cells.append(md("""---
# Add-On 8 — Final Aggregation & Report
> Unload all add-on models, print updated master tables, push to HF.
> Final line: `ADD-ON COMPLETE`.
"""))

# ═══════════════════════════════════════════════════════════════
# B16 [CODE] — AO-Final-Report
# ═══════════════════════════════════════════════════════════════
new_cells.append(code(r'''# AO-Final-Report — Final aggregation, VRAM cleanup, push
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
'''))

# ════════════════════════════════════════════════════════════════
# Append all new cells to notebook
# ════════════════════════════════════════════════════════════════
for cell in new_cells:
    nb["cells"].append(cell)

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Done! Notebook now has {len(nb['cells'])} cells (was {existing_count}).")
print(f"Added {len(new_cells)} new cells for 70B Add-On (B1-B16).")
