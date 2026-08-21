import json, os

NB_PATH = "molab_run.ipynb"
with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Find the end to append
markdown_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "---\n",
        "# Night Run — 70B-A Part 1 (MATH: greedy_cot, SC, FRv4)\n",
        "> **MATH 238** × {greedy_cot, SC, FRv4} on a 70B class model.\n",
        "> Expects a 48GB GPU (e.g. RTX 6000 Ada) to fit a 4-bit 70B model (~40GB VRAM).\n",
        "> Logs saved to `results/night_70b_logs/` and synced to Hugging Face."
    ]
}

code_source = """# Night Run - 70B-A Part 1
import os, sys, json, time, math, subprocess
from pathlib import Path

nb_dir = Path(os.environ.get("NOTEBOOK_DIR", "."))
os.chdir(nb_dir)
sys.path.insert(0, str(nb_dir))

# ── 1. Pull the 70B model ─────────────────────────────────────
# Change this if you prefer a different 70B (e.g., "qwen2.5:72b")
MODEL = "llama3.1:70b" 

print("=" * 60)
print(f"  Fetching {MODEL} (Warning: ~40GB download, takes time!)")
print("=" * 60)
ollama_bin = subprocess.run("which ollama", shell=True, capture_output=True, text=True).stdout.strip()
if not ollama_bin: ollama_bin = "/usr/local/bin/ollama"
subprocess.run(f"{ollama_bin} pull {MODEL}", shell=True)
print(f"{MODEL} ready.")

# ── 2. Run the 3 strategies on MATH (238 questions) ───────────
from ttc_frugalreason_poc.experiment_fr.core.task_loader import load_all_tasks
from ttc_frugalreason_poc.experiment_fr.core.ollama_client import OllamaClient
from ttc_frugalreason_poc.experiment_fr.core.verifier import OutcomeVerifier
from ttc_frugalreason_poc.experiment_fr.strategies.frugal_reason_v3 import frugal_reason_v3_evaluate

# Import baselines dynamically to avoid path issues
sys.path.insert(0, str(nb_dir / "ttc-frugalreason-poc/experiment_fr"))
import run_real_experiment as baselines

client = OllamaClient(model_name=MODEL)
verifier = OutcomeVerifier(ollama_client=client)

# Only load MATH, all 238 Qs (seed=0)
tasks = load_all_tasks({"sampling": {"questions_per_task": 238, "seed": 0}})
math_data = tasks.get("math", [])
if not math_data:
    raise RuntimeError("Failed to load MATH dataset.")

out_dir = nb_dir / "results" / "night_70b_logs"
out_dir.mkdir(parents=True, exist_ok=True)

strategies = [
    ("greedy_cot", baselines.run_greedy_cot),
    ("self_consistency_k5", baselines.run_sc_k5),
    ("frugal_reason_v4", lambda c, t, q: frugal_reason_v3_evaluate(c, t, q, input_metadata=q, enable_early_exit=True, alpha=0.6))
]

print("\\n" + "=" * 60)
print(f"  STARTING 70B NIGHT RUN on MATH (238 Qs)")
print("=" * 60)

import warnings
warnings.filterwarnings('ignore', category=SyntaxWarning)

for strat_name, strat_fn in strategies:
    out_file = out_dir / f"math_{strat_name}.jsonl"
    print(f"\\n--- {strat_name} ---")
    
    # Resume logic
    done_qids = set()
    if out_file.exists():
        with open(out_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    done_qids.add(json.loads(line).get("question_id"))
                    
    total = len(math_data)
    todo = [item for item in math_data if f"math_{math_data.index(item)}" not in done_qids]
    done = len(done_qids)
    
    if done >= total:
        print(f"Strategy {strat_name} already complete ({total}/{total}).")
        continue
        
    start_t = time.time()
    print(f"Resuming {strat_name} at {done}/{total}...")
    
    for item in todo:
        idx = math_data.index(item)
        q_id = f"math_{idx}"
        q_text = item["question"]
        gold = item["gold_answer"]
        
        try:
            res = strat_fn(client, "math", q_text)
            if "latency_seconds_total" not in res:
                res["latency_seconds_total"] = res.get("latency", 0.0)
                res["prompt_tokens_total"] = res.get("tokens", 0)
                res["completion_tokens_total"] = 0
                res["model_calls"] = res.get("calls", 0)
                
            ans = res.get("selected_answer", "")
            raw_out = res.get("raw_output", str(ans))
            eval_res = verifier.score("math", q_text, str(raw_out), str(ans), gold)
            is_correct = eval_res["score"] == 1.0
            
            log_record = {
                "strategy": strat_name, "task": "math", "question_id": q_id,
                "gold_answer": gold, "correct": is_correct,
            }
            log_record.update(res)
            
            with open(out_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_record) + "\\n")
                
            done += 1
            if done % 10 == 0:
                elapsed = time.time() - start_t
                eta = (elapsed / max(done - len(done_qids), 1)) * (total - done)
                print(f"  [{MODEL}] {strat_name} {done}/{total} | {elapsed/3600:.1f}h elapsed | ETA {eta/3600:.1f}h")
                
                # Push to HF every 10 Qs
                try:
                    from huggingface_hub import HfApi
                    _api = HfApi(token="REDACTED")
                    _api.upload_file(
                        path_or_fileobj=str(out_file),
                        path_in_repo=f"night_70b_logs/math_{strat_name}.jsonl",
                        repo_id="Satabarto/Molab_Checkpoints_Cost_AWARE",
                        repo_type="dataset",
                    )
                except: pass
                
        except Exception as e:
            print(f"Error on {q_id}: {e}")
            sys.exit(1)

print("\\nNIGHT RUN COMPLETE! All 3 strategies finished on 70B MATH.")
"""

# Fix sys.path syntax in code_source
code_source = code_source.replace("ttc_frugalreason_poc.experiment_fr", "ttc-frugalreason-poc.experiment_fr")
# Wait, python imports can't have hyphens. Let's just use sys.path correctly.

code_source = """# Night Run - 70B-A Part 1
import os, sys, json, time, math, subprocess
from pathlib import Path
import warnings

warnings.filterwarnings('ignore', category=SyntaxWarning)

nb_dir = Path(os.environ.get("NOTEBOOK_DIR", "."))
os.chdir(nb_dir)

sys.path.insert(0, str(nb_dir / "ttc-frugalreason-poc" / "experiment_fr"))
import run_real_experiment as baselines
from core.task_loader import load_all_tasks
from core.ollama_client import OllamaClient
from core.verifier import OutcomeVerifier
from strategies.frugal_reason_v3 import frugal_reason_v3_evaluate

# ── 1. Pull the 70B model ─────────────────────────────────────
# Change this if you prefer a different 70B (e.g., "qwen2.5:72b")
MODEL = "llama3.1:70b" 

print("=" * 60)
print(f"  Fetching {MODEL} (Warning: ~40GB download, takes time!)")
print("=" * 60)
ollama_bin = subprocess.run("which ollama", shell=True, capture_output=True, text=True).stdout.strip()
if not ollama_bin: ollama_bin = "/usr/local/bin/ollama"
subprocess.run(f"{ollama_bin} pull {MODEL}", shell=True)
print(f"{MODEL} ready.")

# ── 2. Run the 3 strategies on MATH (238 questions) ───────────

client = OllamaClient(model_name=MODEL)
verifier = OutcomeVerifier(ollama_client=client)

# Only load MATH, all 238 Qs (seed=0)
tasks = load_all_tasks({"sampling": {"questions_per_task": 238, "seed": 0}})
math_data = tasks.get("math", [])
if not math_data:
    raise RuntimeError("Failed to load MATH dataset.")

out_dir = nb_dir / "results" / "night_70b_logs"
out_dir.mkdir(parents=True, exist_ok=True)

strategies = [
    ("greedy_cot", baselines.run_greedy_cot),
    ("self_consistency_k5", baselines.run_sc_k5),
    ("frugal_reason_v4", lambda c, t, q: frugal_reason_v3_evaluate(c, t, q, input_metadata=q, enable_early_exit=True, alpha=0.6))
]

print("\\n" + "=" * 60)
print(f"  STARTING 70B NIGHT RUN on MATH (238 Qs)")
print("=" * 60)


for strat_name, strat_fn in strategies:
    out_file = out_dir / f"math_{strat_name}.jsonl"
    print(f"\\n--- {strat_name} ---")
    
    # Resume logic
    done_qids = set()
    if out_file.exists():
        with open(out_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    done_qids.add(json.loads(line).get("question_id"))
                    
    total = len(math_data)
    todo = [item for item in math_data if f"math_{math_data.index(item)}" not in done_qids]
    done = len(done_qids)
    
    if done >= total:
        print(f"Strategy {strat_name} already complete ({total}/{total}).")
        continue
        
    start_t = time.time()
    print(f"Resuming {strat_name} at {done}/{total}...")
    
    for item in todo:
        idx = math_data.index(item)
        q_id = f"math_{idx}"
        q_text = item["question"]
        gold = item["gold_answer"]
        
        try:
            res = strat_fn(client, "math", q_text)
            if "latency_seconds_total" not in res:
                res["latency_seconds_total"] = res.get("latency", 0.0)
                res["prompt_tokens_total"] = res.get("tokens", 0)
                res["completion_tokens_total"] = 0
                res["model_calls"] = res.get("calls", 0)
                
            ans = res.get("selected_answer", "")
            raw_out = res.get("raw_output", str(ans))
            eval_res = verifier.score("math", q_text, str(raw_out), str(ans), gold)
            is_correct = eval_res["score"] == 1.0
            
            log_record = {
                "strategy": strat_name, "task": "math", "question_id": q_id,
                "gold_answer": gold, "correct": is_correct,
            }
            log_record.update(res)
            
            with open(out_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_record) + "\\n")
                
            done += 1
            if done % 10 == 0:
                elapsed = time.time() - start_t
                eta = (elapsed / max(done - len(done_qids), 1)) * (total - done)
                print(f"  [{MODEL}] {strat_name} {done}/{total} | {elapsed/3600:.1f}h elapsed | ETA {eta/3600:.1f}h")
                
                # Push to HF every 10 Qs
                try:
                    from huggingface_hub import HfApi
                    _api = HfApi(token="REDACTED")
                    _api.upload_file(
                        path_or_fileobj=str(out_file),
                        path_in_repo=f"night_70b_logs/math_{strat_name}.jsonl",
                        repo_id="Satabarto/Molab_Checkpoints_Cost_AWARE",
                        repo_type="dataset",
                    )
                except: pass
                
        except Exception as e:
            print(f"Error on {q_id}: {e}")
            sys.exit(1)

print("\\nNIGHT RUN COMPLETE! All 3 strategies finished on 70B MATH.")
"""

code_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [line + "\n" for line in code_source.split("\n")]
}
if code_cell["source"]:
    code_cell["source"][-1] = code_cell["source"][-1].rstrip("\n")

nb["cells"].append(markdown_cell)
nb["cells"].append(code_cell)

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Appended Night Run 70B cell to molab_run.ipynb")
