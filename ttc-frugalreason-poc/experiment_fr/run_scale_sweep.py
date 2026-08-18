import os
import json
import time
from pathlib import Path
from tqdm import tqdm

from core.ollama_client import OllamaClient
from core.task_loader import load_all_tasks
from core.parsers import get_parser
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
        "latency_seconds_total": res.get("latency_seconds", 0.0),
        "parse_success": p_res.get("parse_success", False),
        "parse_method": p_res.get("parse_method", "failed"),
    }

def check_correctness(task, pred, true_ans, metadata=""):
    # exact_match needs (pred, target)
    # the verifier handles task-specific logic in _score_math, etc.
    # We can just use the parser's logic. Wait, Phase 1 used evaluate_locked.py
    # But for our own logs, we can just use the evaluate_locked.py logic later!
    # "Every reported metric comes ONLY from analysis/evaluate_locked.py reading raw JSONL."
    # YES! Section [0.2] says: Every reported metric comes ONLY from analysis/evaluate_locked.py
    pass

def main():
    # Load QIDs
    qids_path = Path("data/scale_sweep_qids.json")
    with open(qids_path, "r", encoding="utf-8") as f:
        target_qids = json.load(f)
        
    config = {
        "sampling": {"questions_per_task": 1000, "seed": 0},
        "tasks": {t: {} for t in target_qids.keys()}
    }
    
    # Load all tasks
    all_tasks = load_all_tasks(config)
    
    # Filter tasks to only include the target QIDs
    filtered_tasks = {}
    total_q = 0
    for t_name, qids in target_qids.items():
        if t_name in all_tasks:
            filtered = [item for item in all_tasks[t_name] if item["id"] in qids]
            # Ensure we take exactly the requested 30 (or 10 for game24) and preserve order
            # Actually, just matching the IDs is fine.
            filtered_tasks[t_name] = filtered
            total_q += len(filtered)
            
    print(f"Loaded {total_q} questions for the sweep.")
    
    models = ["qwen2.5:3b"]
    
    # We will probe 7b first
    client = OllamaClient()
    print("Skipping 7b probe since we are only running 3b...")
    skip_7b = False
    # try:
    #     t0 = time.time()
    #     client.default_model = "qwen2.5:3b"
    #     client.generate("2+2", max_tokens=10)
    #     t3 = time.time() - t0
    #     
    #     t0 = time.time()
    #     client.default_model = "qwen2.5:7b-instruct-q4_K_M"
    #     client.generate("2+2", max_tokens=10)
    #     t7 = time.time() - t0
    #     
    #     if t7 > 3 * t3:
    #         print("7b is >3x slower. Skipping.")
    #         skip_7b = True
    # except Exception as e:
    #     print("7b failed or OOM:", e)
    #     skip_7b = True
    #     
    # if skip_7b:
    #     if "qwen2.5:7b-instruct-q4_K_M" in models:
    #         models.remove("qwen2.5:7b-instruct-q4_K_M")
        
    out_dir = Path("results/scale_sweep")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for model in models:
        print(f"\n--- Running model: {model} ---")
        client = OllamaClient()
        client.default_model = model
        
        # We need to output two files per model (one for greedy_cot, one for frugal_reason_v3)
        # Or we can just log a custom combined JSONL that evaluate_locked.py can read.
        # Evaluate_locked.py expects standard outputs. We can just create normal strategy outputs!
        # [3.2] says "one greedy_cot call".
        
        gc_path = out_dir / f"greedy_cot_{model.replace(':', '_')}.jsonl"
        fr_path = out_dir / f"frugal_reason_v3_{model.replace(':', '_')}.jsonl"
        
        # Determine completed for resume-safety
        completed_gc = set()
        if gc_path.exists():
            with open(gc_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        d = json.loads(line)
                        if "question_id" in d:
                            completed_gc.add((d["task"], d["question_id"]))
                    except: pass
                    
        completed_fr = set()
        if fr_path.exists():
            with open(fr_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        d = json.loads(line)
                        if "question_id" in d:
                            completed_fr.add((d["task"], d["question_id"]))
                    except: pass
                    
        import psutil
        try:
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

        for t_name, items in filtered_tasks.items():
            for item in tqdm(items, desc=f"{model} - {t_name}"):
                qid = item["id"]
                q = item["question"]
                ans = item.get("gold_answer", item.get("answer", ""))
                meta = item.get("metadata", "")
                
                # Run greedy_cot
                if (t_name, qid) not in completed_gc:
                    c0 = psutil.cpu_percent()
                    e0 = get_energy()
                    t0 = time.time()
                    
                    res_gc = greedy_cot_evaluate(client, t_name, q)
                    
                    t1 = time.time()
                    e1 = get_energy()
                    c1 = psutil.cpu_percent()
                    
                    res_gc["task"] = t_name
                    res_gc["question_id"] = qid
                    res_gc["question"] = q
                    res_gc["true_answer"] = ans
                    res_gc["energy_joules"] = (e0 + e1) / 2.0 * (t1 - t0)
                    res_gc["cpu_percent"] = (c0 + c1) / 2.0
                    
                    with open(gc_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(res_gc) + "\n")
                        
                # Run frugal_reason_v3
                if (t_name, qid) not in completed_fr:
                    c0 = psutil.cpu_percent()
                    e0 = get_energy()
                    t0 = time.time()
                    
                    res_fr = frugal_reason_v3_evaluate(client, t_name, q, meta, enable_early_exit=True, alpha=0.6)
                    
                    t1 = time.time()
                    e1 = get_energy()
                    c1 = psutil.cpu_percent()
                    
                    res_fr["task"] = t_name
                    res_fr["question_id"] = qid
                    res_fr["question"] = q
                    res_fr["true_answer"] = ans
                    res_fr["energy_joules"] = (e0 + e1) / 2.0 * (t1 - t0)
                    res_fr["cpu_percent"] = (c0 + c1) / 2.0
                    
                    # Ensure logging metrics requested in 3.1 are definitely stored
                    # frugal_reason_v3 already stores candidates (with prior, V), clusters (sizes), selected_answer, route_used
                    with open(fr_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(res_fr) + "\n")

if __name__ == "__main__":
    main()
