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
from strategies.frugal_reason_v3 import frugal_reason_v3_evaluate

def run_ablation(client, task, question, variant):
    # Base params
    alpha = 0.6
    enable_early_exit = True
    temp_scale = 2.0 # Assuming this was calibrated, default for uncalibrated is 1.0
    
    if variant == "AB1_NoPrior":
        alpha = 1.0
    elif variant == "AB2_Uncalibrated":
        # Pass a flag or hack it if the function doesn't support T directly. 
        # For this script we will assume we can pass a dummy kwargs or we just run baseline if unsupported.
        # Ideally the v3 function takes temperature_scaling.
        pass
    elif variant == "AB3_NoCluster":
        # Disable semantic clustering.
        pass
    elif variant == "AB4_NoExit":
        enable_early_exit = False
        
    res = frugal_reason_v3_evaluate(
        client, task, question, 
        input_metadata=question, 
        enable_early_exit=enable_early_exit, 
        alpha=alpha
    )
    
    return {
        "selected_answer": res.get("selected_answer"),
        "raw_response": json.dumps(res),
        "latency_seconds_total": res.get("latency", 0.0),
        "total_tokens": res.get("tokens", 0),
        "model_calls": res.get("calls", 0),
        "parse_success": res.get("parse_success", False),
        "parse_method": res.get("route_used", f"fr_ablation_{variant}"),
        "raw_paths": [],
        "clusters": res.get("clusters", []),
        "candidates": res.get("candidates", []),
        "early_exit": res.get("early_exit", False)
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen2.5:3b")
    parser.add_argument("--dataset", type=str, default="math")
    parser.add_argument("--n", type=int, default=100)
    args = parser.parse_args()

    MODEL = args.model
    DS = args.dataset
    SEED = 42

    print(f"Loading {DS}...")
    loader_config = {"sampling": {"questions_per_task": 1500, "seed": SEED}, "tasks": {DS: {}}}
    loaded = load_all_tasks(loader_config)
    task_maps = {item["id"]: item for item in loaded.get(DS, [])}
    
    all_ids = list(task_maps.keys())
    import random
    rng = random.Random(SEED)
    qids = rng.sample(all_ids, min(args.n, len(all_ids)))

    results_dir = base_dir / "results" / "ablations"
    results_dir.mkdir(parents=True, exist_ok=True)
    db_path = base_dir / f"ablation_checkpoint_{MODEL.replace(':', '_')}.db"
    
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute('''CREATE TABLE IF NOT EXISTS completed 
                   (variant TEXT, qid TEXT, PRIMARY KEY(variant, qid))''')
    conn.commit()

    client = OllamaClient(model=MODEL)
    verifier = OutcomeVerifier(client)
    
    variants = ["AB1_NoPrior", "AB4_NoExit"] # AB2 and AB3 might require direct modifications to the v3 script, sticking to what the API exposes.
    
    for variant in variants:
        log_path = results_dir / f"{MODEL.replace(':', '_')}_{DS}_{variant}.jsonl"
        done = 0
        for qid in qids:
            cur.execute("SELECT 1 FROM completed WHERE variant=? AND qid=?", (variant, qid))
            if cur.fetchone():
                done += 1; continue
                
            item = task_maps.get(qid)
            try:
                res = run_ablation(client, DS, item["question"], variant)
            except Exception as e:
                print(f"Failed {variant} {qid}: {e}")
                continue
                
            score_res = verifier.score(DS, item["question"], res.get("raw_response",""),
                                        res["selected_answer"], item["gold_answer"])
            is_correct = score_res["score"] == 1.0
            
            log_row = {
                "model": MODEL, "dataset": DS, "variant": variant, "qid": qid,
                "gold": item["gold_answer"], "selected_answer": res["selected_answer"],
                "correct": is_correct, "parse_success": res["parse_success"],
                "latency_seconds": res["latency_seconds_total"],
                "tokens": res["total_tokens"], "calls": res["model_calls"],
                "early_exit": res.get("early_exit", False)
            }
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_row) + "\n")
                
            cur.execute("INSERT INTO completed VALUES (?,?)", (variant, qid))
            conn.commit()
            done += 1
            print(f"[{variant}] {done}/{len(qids)}")

    conn.close()
    print("Ablations complete.")

if __name__ == "__main__":
    main()
