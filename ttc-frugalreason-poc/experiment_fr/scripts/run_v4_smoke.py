import os
import sys
import json
import time
import argparse
import sqlite3
from pathlib import Path

# Add experiment_fr to path
base_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(base_dir))

from core.ollama_client import OllamaClient
from core.task_loader import load_all_tasks
from core.verifier import OutcomeVerifier

try:
    from strategies.frugal_reason_v4 import frugal_reason_v4_evaluate
except ImportError:
    print("Warning: frugal_reason_v4 not found. Make sure the dynamic cell generated it.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen2.5:3b")
    parser.add_argument("--dataset", type=str, default="math")
    parser.add_argument("--n", type=int, default=50)
    args = parser.parse_args()

    MODEL = args.model
    DS = args.dataset
    SEED = 42

    print(f"Loading {DS} for v4 Smoke Test...")
    loader_config = {"sampling": {"questions_per_task": 1500, "seed": SEED}, "tasks": {DS: {}}}
    loaded = load_all_tasks(loader_config)
    task_maps = {item["id"]: item for item in loaded.get(DS, [])}
    
    all_ids = list(task_maps.keys())
    import random
    rng = random.Random(SEED)
    qids = rng.sample(all_ids, min(args.n, len(all_ids)))

    results_dir = base_dir / "results" / "smoke_tests"
    results_dir.mkdir(parents=True, exist_ok=True)
    db_path = base_dir / f"v4_smoke_checkpoint_{MODEL.replace(':', '_')}.db"
    
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute('''CREATE TABLE IF NOT EXISTS completed 
                   (qid TEXT PRIMARY KEY)''')
    conn.commit()

    client = OllamaClient(model=MODEL)
    verifier = OutcomeVerifier(client)
    
    log_path = results_dir / f"{MODEL.replace(':', '_')}_{DS}_frugal_reason_v4.jsonl"
    done = 0
    
    for qid in qids:
        cur.execute("SELECT 1 FROM completed WHERE qid=?", (qid,))
        if cur.fetchone():
            done += 1; continue
            
        item = task_maps.get(qid)
        try:
            res = frugal_reason_v4_evaluate(client, DS, item["question"], enable_early_exit=True)
        except Exception as e:
            print(f"Failed {qid}: {e}")
            continue
            
        score_res = verifier.score(DS, item["question"], "", res.get("selected_answer"), item["gold_answer"])
        is_correct = score_res["score"] == 1.0
        
        log_row = {
            "model": MODEL, "dataset": DS, "strategy": "frugal_reason_v4", "qid": qid,
            "gold": item["gold_answer"], "selected_answer": res.get("selected_answer"),
            "correct": is_correct, "parse_success": res.get("parse_success", False),
            "latency_seconds": res.get("latency", 0.0),
            "tokens": res.get("tokens", 0), "calls": res.get("calls", 0),
            "early_exit": res.get("early_exit", False)
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_row) + "\n")
            
        cur.execute("INSERT INTO completed VALUES (?)", (qid,))
        conn.commit()
        done += 1
        print(f"[v4 Smoke] {done}/{len(qids)}")

    conn.close()
    print("v4 Smoke Test complete.")

if __name__ == "__main__":
    main()
