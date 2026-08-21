import os
import sys
import json
import time
import argparse
from pathlib import Path
from tqdm import tqdm
from collections import Counter
import psutil

from core.ollama_client import OllamaClient
from core.task_loader import load_all_tasks
from core.verifier import OutcomeVerifier
from core.prompt_manager import get_prompt
from core.parsers import get_parser
from strategies.frugal_reason_v3 import frugal_reason_v3_evaluate
from verifiers.verifiers import parse_judge_score

try:
    # pyrefly: ignore [missing-import]
    import pynvml
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    has_gpu = True
except:
    has_gpu = False

def get_energy():
    if has_gpu:
        return pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
    return 0.0

def run_greedy_cot(client, task, question):
    prompt = get_prompt("greedy_cot", task, question)
    r = client.generate(prompt)
    txt = r.get("text", "")
    parser = get_parser(task)
    p_res = parser(txt)
    return {
        "final_answer": p_res.get("final_answer"),
        "raw_output": txt,
        "model_calls": 1,
        "prompt_tokens_total": r.get("prompt_tokens", 0),
        "completion_tokens_total": r.get("completion_tokens", 0),
        "latency_seconds_total": r.get("latency", 0),
        "parse_success": p_res.get("parse_success", False),
    }

def run_sc_k5(client, task, question):
    prompt = get_prompt("greedy_cot", task, question)
    answers = []
    lat = 0; pt = 0; ct = 0
    parser = get_parser(task)
    for _ in range(5):
        r = client.generate(prompt, temperature=0.7)
        lat += r.get("latency", 0); pt += r.get("prompt_tokens", 0); ct += r.get("completion_tokens", 0)
        ans = parser(r.get("text", ""))["final_answer"]
        if ans is not None: answers.append(ans)
    ans = Counter(answers).most_common(1)[0][0] if answers else None
    return {
        "final_answer": ans,
        "latency_seconds_total": lat,
        "prompt_tokens_total": pt,
        "completion_tokens_total": ct,
        "model_calls": 5,
        "parse_success": ans is not None
    }

def run_bon_k5(client, task, question):
    prompt = get_prompt("greedy_cot", task, question)
    rationales = []
    lat = 0; pt = 0; ct = 0
    parser = get_parser(task)
    for _ in range(5):
        r = client.generate(prompt, temperature=0.7)
        lat += r.get("latency", 0); pt += r.get("prompt_tokens", 0); ct += r.get("completion_tokens", 0)
        ans = parser(r.get("text", ""))["final_answer"]
        rationales.append({"text": r.get("text", ""), "answer": ans})
    
    best_ans = None
    best_score = -1
    for r in rationales:
        jp = get_prompt("best_of_n", task="", question=question, candidate=r["text"])
        jr = client.generate(jp, temperature=0.0)
        lat += jr.get("latency", 0); pt += jr.get("prompt_tokens", 0); ct += jr.get("completion_tokens", 0)
        
        score = parse_judge_score(jr.get("text", ""))
        
        if score > best_score:
            best_score = score
            best_ans = r["answer"]
            
    return {
        "final_answer": best_ans,
        "latency_seconds_total": lat,
        "prompt_tokens_total": pt,
        "completion_tokens_total": ct,
        "model_calls": 10,
        "parse_success": best_ans is not None
    }

def main():
    import subprocess
    target_model = "qwen2.5:3b"
    
    # Load QIDs
    qids_path = Path("data/pilot_qids.json")
    with open(qids_path, "r", encoding="utf-8") as f:
        target_qids = json.load(f)
        
    allowed_datasets = ["gsm_hard", "math", "aqua", "svamp"]
    
    config = {
        "sampling": {"questions_per_task": 1000, "seed": 0},
        "tasks": {t: {} for t in allowed_datasets}
    }
    
    all_tasks = load_all_tasks(config)
    filtered_tasks = {}
    
    for t_name in allowed_datasets:
        if t_name in all_tasks:
            qids = target_qids.get(t_name, [])
            filtered = [item for item in all_tasks[t_name] if item["id"] in qids]
            filtered_tasks[t_name] = filtered
            
    out_dir = Path("results/strict_eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    client = OllamaClient()
    client.default_model = target_model
    verifier = OutcomeVerifier(ollama_client=client)
    
    strategies = [
        ("greedy_cot", run_greedy_cot),
        ("self_consistency_k5", run_sc_k5),
        ("best_of_n_k5_self_eval", run_bon_k5),
        ("frugal_reason_v3", lambda c, t, q: frugal_reason_v3_evaluate(c, t, q, enable_early_exit=True, alpha=0.6))
    ]
    
    for strat_name, strat_fn in strategies:
        out_file = out_dir / f"{strat_name}.jsonl"
        completed = set()
        
        if out_file.exists():
            with open(out_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        d = json.loads(line)
                        if "question_id" in d and "task" in d:
                            completed.add((d["task"], d["question_id"]))
                    except: pass
                    
        print(f"\n--- Running Strategy: {strat_name} ---")
        
        for task_name, items in filtered_tasks.items():
            for item in tqdm(items, desc=f"{strat_name} - {task_name}"):
                q_id = item["id"]
                if (task_name, q_id) in completed:
                    continue
                    
                q_text = item["question"]
                gold = item.get("gold_answer", item.get("answer", ""))
                
                c0 = psutil.cpu_percent()
                e0 = get_energy()
                t0 = time.time()
                
                res = strat_fn(client, task_name, q_text)
                
                t1 = time.time()
                e1 = get_energy()
                c1 = psutil.cpu_percent()
                
                if strat_name == "frugal_reason_v3":
                    ans = res.get("selected_answer", "")
                else:
                    ans = res.get("final_answer", "")
                
                eval_res = verifier.score(task_name, q_text, str(ans), str(ans), gold)
                is_correct = eval_res["score"] == 1.0
                
                log_record = {
                    "strategy": strat_name,
                    "task": task_name,
                    "question_id": q_id,
                    "gold_answer": gold,
                    "correct": is_correct,
                    "energy_joules": (e0 + e1) / 2.0 * (t1 - t0),
                    "cpu_percent": (c0 + c1) / 2.0,
                    "wall_seconds": (t1 - t0)
                }
                log_record.update(res)
                
                with open(out_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(log_record) + "\n")

if __name__ == "__main__":
    main()
