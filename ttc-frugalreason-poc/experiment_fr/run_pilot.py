import os
import json
import argparse
import subprocess
import hashlib
from pathlib import Path
from tqdm import tqdm

from core.ollama_client import OllamaClient
from core.task_loader import load_all_tasks
from core.verifier import OutcomeVerifier
from core.hardware_monitor import HardwareMonitor

# Strategies
from strategies.frugal_reason_v3 import frugal_reason_v3_evaluate

def greedy_cot_evaluate(client, task, question):
    from core.prompt_manager import get_prompt
    from core.parsers import get_parser
    prompt = get_prompt("greedy_cot", task, question)
    res = client.generate(prompt)
    txt = res.get("text", "")
    parser = get_parser(task)
    p_res = parser(txt)
    return {
        "final_answer": p_res.get("final_answer"),
        "raw_output": txt,
        "model_calls": 1,
        "prompt_tokens_total": res.get("prompt_tokens", 0),
        "completion_tokens_total": res.get("completion_tokens", 0),
        "latency_seconds_total": res.get("latency", 0),
        "parse_success": p_res.get("parse_success", False),
        "parse_method": p_res.get("parse_method", "failed"),
    }

def self_consistency_evaluate(client, task, question, k=5):
    from core.prompt_manager import get_prompt
    from core.parsers import get_parser
    from collections import Counter
    prompt = get_prompt("greedy_cot", task, question)
    
    answers = []
    total_pt = 0
    total_ct = 0
    total_lat = 0
    
    parser = get_parser(task)
    for _ in range(k):
        # We should use non-zero temperature for diversity in SC
        res = client.generate(prompt, temperature=0.7)
        txt = res.get("text", "")
        total_pt += res.get("prompt_tokens", 0)
        total_ct += res.get("completion_tokens", 0)
        total_lat += res.get("latency", 0)
        
        p_res = parser(txt)
        if p_res.get("parse_success"):
            answers.append(p_res.get("final_answer"))
            
    if not answers:
        final_ans = None
    else:
        final_ans = Counter(answers).most_common(1)[0][0]
        
    return {
        "final_answer": final_ans,
        "model_calls": k,
        "prompt_tokens_total": total_pt,
        "completion_tokens_total": total_ct,
        "latency_seconds_total": total_lat,
        "parse_success": final_ans is not None,
    }


def get_git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
    except Exception:
        return "unknown"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    
    BASE_DIR = Path(__file__).parent
    
    # Config
    config = {
        "sampling": {"questions_per_task": 50, "seed": args.seed},
        "tasks": {"gsm_hard": {}, "svamp": {}, "aqua": {}, "math": {}}
    }
    
    config_str = json.dumps(config, sort_keys=True)
    config_hash = hashlib.md5(config_str.encode()).hexdigest()
    git_hash = get_git_hash()
    
    print(f"Git Hash: {git_hash}")
    print(f"Config Hash: {config_hash}")
    
    tasks = load_all_tasks(config)
    verifier = OutcomeVerifier()
    client = OllamaClient()
    
    out_dir = BASE_DIR / "results" / "pilot_logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Save QIDs
    qids_map = {t: [item["id"] for item in items] for t, items in tasks.items()}
    with open(BASE_DIR / "data" / "pilot_qids.json", "w") as f:
        json.dump(qids_map, f, indent=2)
        
    strategies = ["greedy_cot", "self_consistency_k5", "frugal_reason_v3"]
    
    for strat in strategies:
        out_file = out_dir / f"{strat}_pilot.jsonl"
        
        completed = set()
        if out_file.exists():
            with open(out_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        d = json.loads(line)
                        if "question_id" in d and "task" in d:
                            completed.add((d["task"], d["question_id"]))
                    except: pass
                    
        # Write metadata header if new
        if not completed:
            with open(out_file, "a", encoding="utf-8") as f:
                f.write(json.dumps({"metadata": True, "git_hash": git_hash, "config_hash": config_hash, "strategy": strat}) + "\n")
                
        for task_name, items in tasks.items():
            for item in tqdm(items, desc=f"{strat} on {task_name}"):
                q_id = item["id"]
                if (task_name, q_id) in completed:
                    continue
                    
                q_text = item["question"]
                gold = item["gold_answer"]
                
                with HardwareMonitor() as hw:
                    if strat == "greedy_cot":
                        res = greedy_cot_evaluate(client, task_name, q_text)
                    elif strat == "self_consistency_k5":
                        res = self_consistency_evaluate(client, task_name, q_text, k=5)
                    elif strat == "frugal_reason_v3":
                        res = frugal_reason_v3_evaluate(client, task_name, q_text, enable_early_exit=True, alpha=0.6)
                        if "selected_answer" in res:
                            res["final_answer"] = res["selected_answer"]
                            
                hw_metrics = hw.metrics()
                ans = str(res.get("final_answer", ""))
                eval_res = verifier.score(task_name, q_text, ans, ans, gold)
                
                res.update({
                    "task": task_name,
                    "question_id": q_id,
                    "gold_answer": gold,
                    "correct": eval_res["score"] == 1.0,
                    "energy_joules": hw_metrics.get("energy_joules"),
                    "avg_cpu_percent": hw_metrics.get("avg_cpu_percent"),
                    "wall_seconds": hw_metrics.get("wall_seconds")
                })
                
                with open(out_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(res) + "\n")

if __name__ == "__main__":
    main()
