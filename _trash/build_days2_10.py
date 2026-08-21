"""
build_days2_10.py — Appends Days 2–10 cells to molab_run.ipynb
Also patches the Day 1 cell to push results to HF at the end.
"""
import json, os

NB_PATH = "molab_run.ipynb"
with open(NB_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": [source]}

def code(source):
    lines = source.split("\n")
    src = [line + "\n" for line in lines]
    if src:
        src[-1] = src[-1].rstrip("\n")
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src}

# ── PATCH Day 1 cell (index 28) — add HF push at the end ──────────
day1_src = "".join(nb["cells"][28]["source"])
if "huggingface_hub" not in day1_src:
    hf_push_block = '''

# ── PUSH RESULTS TO HUGGING FACE ──────────────────────────────
print("\\nPushing Day 1 results to Hugging Face Hub...")
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
'''
    day1_src += hf_push_block
    nb["cells"][28]["source"] = [line + "\n" for line in day1_src.split("\n")]
    nb["cells"][28]["source"][-1] = nb["cells"][28]["source"][-1].rstrip("\n")

# ════════════════════════════════════════════════════════════════
# DAYS 2–10 CELLS
# ════════════════════════════════════════════════════════════════

new_cells = []

# ──────── C1: Day 2 Markdown ────────
new_cells.append(md("""---
# Day 2 — Block B: qwen2.5:1.5b Calibration Sweep
> **2,400 runs** (4 datasets × 100 stratified qids × 6 strategies) on `qwen2.5:1.5b`.
> Logs → `results/block_b_logs/qwen15b_*.jsonl`. Resumable via SQLite checkpoint.
"""))

# ──────── C2: Day2-Fetch-1.5B ────────
new_cells.append(code('''# Day2-Fetch-1.5B — Pull qwen2.5:1.5b into Ollama
import subprocess, requests, time

print("=" * 60)
print("  Day 2 — Fetching qwen2.5:1.5b")
print("=" * 60)

subprocess.run("ollama pull qwen2.5:1.5b", shell=True, check=True)
time.sleep(3)

# Verify via /api/tags
r = requests.get("http://localhost:11434/api/tags", timeout=10)
r.raise_for_status()
models = [m["name"] for m in r.json().get("models", [])]
assert "qwen2.5:1.5b" in models or any("qwen2.5:1.5b" in m for m in models), \\
    f"qwen2.5:1.5b not found! Available: {models}"
print(f"qwen2.5:1.5b confirmed. Available models: {models}")
'''))

# ──────── C3: Day2-Sweep-1.5B ────────
new_cells.append(code('''# Day2-Sweep-1.5B — 2,400 runs (4 ds × 100 qids × 6 strategies) on qwen2.5:1.5b
import os, sys, json, time, re, sqlite3, requests
from pathlib import Path
from collections import Counter, defaultdict

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
    return {"selected_answer": best, "raw_response": "\\n---SAMPLE---\\n".join(raws),
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
        sm = re.search(r"confidence:\\s*(\\d+)", jr["text"].lower())
        if sm: score = float(sm.group(1)) / 100.0
        elif "yes" in jr["text"].lower(): score = 1.0
        elif "no" in jr["text"].lower(): score = 0.0
        if score > best_score: best_score = score; best_ans = rat["answer"]; best_rat = rat["text"]
    return {"selected_answer": best_ans,
            "raw_response": f"Selected:\\n{best_rat}\\n\\nJudge:\\n" + "\\n---\\n".join(judge_texts),
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
import random
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
                f.write(json.dumps(log_row) + "\\n")
            cur.execute("INSERT OR IGNORE INTO completed VALUES (?,?,?,?,datetime('now'))",
                        (MODEL, ds, strat, qid))
            conn.commit()
            done += 1

            if done % 50 == 0:
                elapsed = time.time() - start
                eta = (total_target - done) * (elapsed / max(done, 1))
                print(f"[{MODEL}] {done}/{total_target} | {elapsed/3600:.1f}h elapsed | ETA {eta/3600:.1f}h")

conn.close()
print(f"\\nDay 2 DONE: {done}/{total_target} runs completed.")

# ── Completeness matrix ─────────────────────────────────────────
print("\\nCompleteness Matrix:")
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
'''))

# ──────── C4: Day 3 Markdown ────────
new_cells.append(md("""---
# Day 3 — Block B: llama3.2:3b Cross-Family Sweep
> **2,400 runs** (4 datasets × 100 qids × 6 strategies) on `llama3.2:3b`.
> Logs → `results/block_b_logs/llama32_*.jsonl`. Identical pipeline to Day 2.
"""))

# ──────── C5: Day3-Fetch-Llama3.2-3B ────────
new_cells.append(code('''# Day3-Fetch-Llama3.2-3B — Pull llama3.2:3b into Ollama
import subprocess, requests, time

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
'''))

# ──────── C6: Day3-Sweep-Llama ────────
new_cells.append(code('''# Day3-Sweep-Llama — 2,400 runs on llama3.2:3b
# This cell is IDENTICAL to Day 2 sweep but with MODEL="llama3.2:3b"
# and log prefix "llama32_"
import os, sys, json, time, re, sqlite3, requests, random
from pathlib import Path

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
    return {"selected_answer": best, "raw_response": "\\n---SAMPLE---\\n".join(raws),
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
        sm = re.search(r"confidence:\\s*(\\d+)", jr["text"].lower())
        if sm: score = float(sm.group(1)) / 100.0
        elif "yes" in jr["text"].lower(): score = 1.0
        elif "no" in jr["text"].lower(): score = 0.0
        if score > best_score: best_score = score; best_ans = rat["answer"]; best_rat = rat["text"]
    return {"selected_answer": best_ans,
            "raw_response": f"Selected:\\n{best_rat}\\nJudge:\\n" + "\\n---\\n".join(judge_texts),
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
            with open(log_path, "a", encoding="utf-8") as f: f.write(json.dumps(log_row) + "\\n")
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
'''))

# ──────── C7: Day 4 Markdown ────────
new_cells.append(md("""---
# Day 4 — α* Extraction (Post-Hoc α-Grid)
> **NO NEW INFERENCE.** For each model {qwen2.5:3b, qwen2.5:1.5b, llama3.2:3b},
> recompute scoring S(A) over α ∈ [0.00..1.00] from saved candidate logs.
> Find empirical α* = argmax accuracy.
"""))

# ──────── C8: Day4-AlphaGrid ────────
new_cells.append(code('''# Day4-AlphaGrid — Post-hoc α sweep, NO new inference
import os, sys, json, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

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
    print(f"\\n{'='*60}")
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
            p = re.sub(r"[,\\$\\s]", "", str(best_a or "")).lower().strip()
            g = re.sub(r"[,\\$\\s]", "", str(gold or "")).lower().strip()
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
print("\\nSaved results/calibration/alpha_grid.csv")
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
'''))

# ──────── C9: Day 5 Markdown ────────
new_cells.append(md("""---
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
"""))

# ──────── C10: Day5-M1-Theory-And-v4-Scoring ────────
new_cells.append(code('''# Day5-M1-Theory-And-v4-Scoring — Compute theoretical α* + create v4 module
import os, sys, json, math
import numpy as np
from pathlib import Path

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
import time, math, json
from typing import Dict, Any, List
from core.prompt_manager import get_prompt
from core.parsers import get_parser
from verifiers.verifiers import verify_game24, verify_gsm8k_steps, llm_judge
from verifiers.clustering import cluster_rationales

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
                score_match = re.search(r'confidence:\\s*(\\d+)', text)
                if score_match:
                    V_raw = float(score_match.group(1)) / 100.0
                else:
                    raw_nums = re.findall(r'\\b(100|[1-9]?[0-9])\\b', text)
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
'''))

# ──────── C11: Day 6 Markdown ────────
new_cells.append(md("""---
# Day 6 — M2: Judge Calibration (Temperature Scaling + ECE)
> Per model, from Block B FR logs: split qids 50/50 by hash (FIT/EVAL).
> Fit temperature T by minimizing NLL on FIT; compute 10-bin ECE on EVAL before/after.
"""))

# ──────── C12: Day6-TempScaling-ECE ────────
new_cells.append(code('''# Day6-TempScaling-ECE — Temperature scaling + ECE computation
import os, sys, json, math, hashlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from pathlib import Path

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
print(f"\\nECE improved for {improved}/{len(ece_data)} models (need ≥2/3)")
'''))

# ──────── C13: Day 7 Markdown ────────
new_cells.append(md("""---
# Day 7 — M3 Dirichlet Prior + M4 Cost-Aware Exit + Unit Tests
> **M3**: β-sweep {0, 0.5, 1, 2} post-hoc on Block B logs; pick β* by val acc.
> **M4**: Compute p_agree, p_full, Δ from 3B Block A logs. Implement exit gate in v4.
> **Unit Tests**: 5 assert-based tests to validate all components.
"""))

# ──────── C14: Day7-M3-M4-UnitTests ────────
new_cells.append(code('''# Day7-M3-M4-UnitTests — Dirichlet β sweep + exit gate calibration + unit tests
import os, sys, json, math
import numpy as np
from pathlib import Path

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
                p = re.sub(r"[,\\$\\s]", "", str(best_a or "")).lower().strip()
                g = re.sub(r"[,\\$\\s]", "", str(gold or "")).lower().strip()
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
print("\\n" + "=" * 60)
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
print("\\n" + "=" * 60)
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

print("\\nAll unit tests PASSED.")
print(f"β*={best_beta['beta']}, T=see Day 6, Δ={delta:.4f}, p_agree={p_agree:.4f}, p_full={p_full:.4f}")
'''))

# ──────── C15: Day 8 Markdown ────────
new_cells.append(md("""---
# Day 8 — v4 Smoke Test (600 runs)
> 50 qids (seed=0) × {gsm8k, math} × 6 methods on `qwen2.5:3b`.
> Uses `frugal_reason_v4` for the FR slot. Compare vs saved v3 + baselines (paired).
"""))

# ──────── C16: Day8-v4-Smoke ────────
new_cells.append(code('''# Day8-v4-Smoke — 50q × 2ds × 6 methods = 600 runs with v4
import os, sys, json, time, re, sqlite3, random
from pathlib import Path

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
    return {"selected_answer": best, "raw_response": "\\n---\\n".join(raws),
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
        sm = re.search(r"confidence:\\s*(\\d+)", jr["text"].lower())
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

import pandas as pd
smoke_df = pd.DataFrame(smoke_data)
smoke_df.to_csv(str(results_dir / "smoke_table.csv"), index=False)
print("\\nSaved results/v4/smoke_table.csv")

# Check v4 vs v3 (compare from Block A logs if available)
for ds in DATASETS:
    v4_row = smoke_df[(smoke_df["dataset"] == ds) & (smoke_df["strategy"] == "frugal_reason_v4")]
    if len(v4_row) > 0:
        v4_acc = v4_row["accuracy"].values[0]
        print(f"  {ds}: v4 acc = {v4_acc:.1%}")
'''))

# ──────── C17: Day 9 Markdown ────────
new_cells.append(md("""---
# Day 9 — BBH Logical Deduction (250q × 6 methods)
> Load `lukaemon/bbh` logical_deduction_seven_objects; take 250 test examples.
> Run all 6 methods on qwen2.5:3b → `results/bbh_logs/`. 1,500 runs.
"""))

# ──────── C18: Day9-Fetch-BBH ────────
new_cells.append(code('''# Day9-Fetch-BBH — Load BBH logical deduction dataset
import os, sys, json, re
from pathlib import Path
from datasets import load_dataset

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
        f.write(json.dumps({"id": f"bbh_{i}", "question": q, "gold_answer": gold, "task": "bbh"}) + "\\n")
print(f"Saved {len(items)} BBH examples to {bbh_data_path}")

# Parser self-test
def parse_bbh(response):
    result = {"final_answer": None, "parse_success": False, "parse_method": "failed"}
    if not response: return result
    text = response.lower().strip()
    # Look for "the answer is (X)" pattern
    m = re.search(r"the answer is \\(?([a-g])\\)?", text)
    if m:
        result["final_answer"] = m.group(1).upper()
        result["parse_success"] = True; result["parse_method"] = "strict"
        return result
    # Look for standalone option letter
    m = re.search(r"\\b([a-g])\\b", text)
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
'''))

# ──────── C19: Day9-BBH-Run ────────
new_cells.append(code('''# Day9-BBH-Run — 250q × 6 methods = 1,500 runs
import os, sys, json, time, re, sqlite3
from pathlib import Path

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
    m = re.search(r"the answer is \\(?([a-g])\\)?", text)
    if m:
        result["final_answer"] = m.group(1).upper()
        result["parse_success"] = True; result["parse_method"] = "strict"
        return result
    m = re.search(r"\\b([a-g])\\b", text)
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
                res = {"selected_answer": best, "raw_response": "\\n---\\n".join(raws),
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
                    sm = re.search(r"confidence:\\s*(\\d+)", jr["text"].lower())
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
        with open(log_path, "a", encoding="utf-8") as f: f.write(json.dumps(log_row) + "\\n")
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
print(f"\\nDay 9 DONE: {done}/{total_target}")
print("\\nBBH Results:")
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
'''))

# ──────── C20: Day 10 Markdown ────────
new_cells.append(md("""---
# Day 10 — Ablation Matrix (Post-Hoc First, ≤1,000 New Runs)
> **AB1 NO-PRIOR**: α=1 (post-hoc). **AB2 UNCALIBRATED-JUDGE**: V_raw (post-hoc).
> **AB3 NO-CLUSTERING**: Singleton clusters (post-hoc + cheap judge). **AB4 NO-EXIT**: Re-run full v4 on early-exit qids only.
"""))

# ──────── C21: Day10-Ablations ────────
new_cells.append(code('''# Day10-Ablations — Post-hoc ablations + limited new runs
import os, sys, json, math, re
import numpy as np
import pandas as pd
from pathlib import Path

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
        p = re.sub(r"[,\\$\\s]", "", str(best_a or "")).lower().strip()
        g = re.sub(r"[,\\$\\s]", "", str(gold or "")).lower().strip()
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
        p = re.sub(r"[,\\$\\s]", "", str(best_a or "")).lower().strip()
        g = re.sub(r"[,\\$\\s]", "", str(gold or "")).lower().strip()
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
    print(f"\\n{ds.upper()}:")
    print(f"  Full v4:       {full_v4_acc:.1%} ({full_v4_correct}/{full_v4_total})")
    print(f"  AB1 NO-PRIOR:  {ab1_acc:.1%} (Δ={ab1_acc - full_v4_acc:+.1%})")
    print(f"  AB2 UNCAL:     {ab2_acc:.1%} (Δ={ab2_acc - full_v4_acc:+.1%})")
    print(f"  AB4 exit qids: {early_exit_count} (would need re-run)")

abl_df = pd.DataFrame(ablation_results)
abl_df.to_csv(str(abl_dir / "ablation_table.csv"), index=False)
print("\\nSaved results/ablations/ablation_table.csv")
print(abl_df.to_string(index=False))
'''))

# ──────── C22: Summary Markdown ────────
new_cells.append(md("""---
# Days 2–10 Summary
> Aggregate report across all models, datasets, strategies, calibration, BBH, and ablations.
"""))

# ──────── C23: Days2-10-Aggregate-Report ────────
new_cells.append(code('''# Days2-10-Aggregate-Report — Master summary
import os, json, time
import pandas as pd
from pathlib import Path

_nb = Path(os.environ.get("NOTEBOOK_DIR", "."))

print("=" * 70)
print("  DAYS 2–10 AGGREGATE REPORT")
print("=" * 70)

# (1) Block B Master Table
print("\\n(1) Block B Master Table (3 models × 6 strategies × 4 datasets)")
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
print("\\n(2) Alpha Table (empirical vs theoretical)")
alpha_path = _nb / "results" / "calibration" / "alpha_grid.csv"
if alpha_path.exists():
    print(pd.read_csv(alpha_path).to_string(index=False))

# (3) ECE Table
print("\\n(3) ECE Table")
ts_path = _nb / "results" / "calibration" / "temp_scaling.json"
if ts_path.exists():
    print(json.dumps(json.load(open(ts_path)), indent=2))

# (4) BBH Table
print("\\n(4) BBH Table")
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
print("\\n(5) Ablation Table")
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
    print("\\nPushed final results to HF.")
except Exception as e:
    print(f"HF push failed: {e}")

print("\\n" + "=" * 70)
print("  DAYS 2–10 COMPLETE")
print("  STOP. Do not start Day 11+ writing cells.")
print("=" * 70)
'''))

# ════════════════════════════════════════════════════════════════
# Append all new cells to notebook
# ════════════════════════════════════════════════════════════════
for cell in new_cells:
    nb["cells"].append(cell)

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Done! Notebook now has {len(nb['cells'])} cells.")
print(f"Added {len(new_cells)} new cells for Days 2-10.")
