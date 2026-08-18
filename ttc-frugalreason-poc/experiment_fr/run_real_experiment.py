import os
import sys
import json
import time
import argparse
import requests
from pathlib import Path
from tqdm import tqdm
from collections import Counter

from core.ollama_client import OllamaClient
from core.task_loader import load_all_tasks
from core.verifier import OutcomeVerifier
from core.prompt_manager import get_prompt, get_tot_propose_prompt, get_tot_value_prompt
from core.parsers import get_parser
from strategies.frugal_reason_v3 import frugal_reason_v3_evaluate

def check_ollama_running(base_url="http://localhost:11434"):
    try:
        r = requests.get(f"{base_url}/api/tags", timeout=5)
        r.raise_for_status()
    except Exception as e:
        print(f"FATAL: Ollama is not running or unreachable at {base_url}. Error: {e}")
        print("Please run Ollama in a separate terminal and try again.")
        sys.exit(1)

def run_self_tests():
    p_gsm8k = get_parser('gsm8k')
    assert p_gsm8k("...The answer is 18.")["final_answer"] == "18", "Parser failed on '18.'"
    assert p_gsm8k("$18.")["final_answer"] == "18", "Parser failed on '$18.'"
    assert p_gsm8k("14,000.")["final_answer"] == "14000", "Parser failed on '14,000.'"
    
    p_game24 = get_parser('game24')
    # The prompt explicitly asks to ensure it parses "(10-4)*(13-9)=24"
    res = p_game24("(10-4)*(13-9)=24")
    assert res["parse_success"] is True, "Game24 parser failed"
    print("Self-tests passed.")

# Strategy Implementations
def run_greedy_io(client, task, question):
    prompt = get_prompt("greedy_io", task, question)
    r = client.generate(prompt, temperature=0.0)
    p = get_parser(task)(r["text"])
    return {"selected_answer": p["final_answer"], "raw_output": r["text"], "latency_seconds_total": r["latency_seconds"],
            "prompt_tokens_total": r["prompt_tokens"], "completion_tokens_total": r["completion_tokens"],
            "model_calls": 1, "route_used": "greedy_io"}

def run_greedy_cot(client, task, question):
    prompt = get_prompt("greedy_cot", task, question)
    r = client.generate(prompt, temperature=0.0)
    p = get_parser(task)(r["text"])
    return {"selected_answer": p["final_answer"], "raw_output": r["text"], "latency_seconds_total": r["latency_seconds"],
            "prompt_tokens_total": r["prompt_tokens"], "completion_tokens_total": r["completion_tokens"],
            "model_calls": 1, "route_used": "greedy_cot"}

def run_sc_k5(client, task, question):
    prompt = get_prompt("greedy_cot", task, question)
    answers = []
    lat = 0; pt = 0; ct = 0
    for _ in range(5):
        r = client.generate(prompt, temperature=0.7)
        lat += r["latency_seconds"]; pt += r["prompt_tokens"]; ct += r["completion_tokens"]
        ans = get_parser(task)(r["text"])["final_answer"]
        if ans is not None: answers.append(ans)
    ans = Counter(answers).most_common(1)[0][0] if answers else None
    return {"selected_answer": ans, "latency_seconds_total": lat, "prompt_tokens_total": pt, "completion_tokens_total": ct, "model_calls": 5, "route_used": "self_consistency_k5"}

def run_bon_k5(client, task, question):
    prompt = get_prompt("greedy_cot", task, question)
    rationales = []
    lat = 0; pt = 0; ct = 0
    for _ in range(5):
        r = client.generate(prompt, temperature=0.7)
        lat += r["latency_seconds"]; pt += r["prompt_tokens"]; ct += r["completion_tokens"]
        ans = get_parser(task)(r["text"])["final_answer"]
        rationales.append({"text": r["text"], "answer": ans})
    
    best_ans = None
    best_score = -1
    for r in rationales:
        jp = get_prompt("best_of_n", task="", question=question, candidate=r["text"])
        jr = client.generate(jp, temperature=0.0)
        lat += jr["latency_seconds"]; pt += jr["prompt_tokens"]; ct += jr["completion_tokens"]
        # rudimentary parse of judge score
        score = 0.5
        import re
        sm = re.search(r'confidence:\s*(\d+)', jr["text"].lower())
        if sm: score = float(sm.group(1)) / 100.0
        elif "yes" in jr["text"].lower(): score = 1.0
        elif "no" in jr["text"].lower(): score = 0.0
        
        if score > best_score:
            best_score = score
            best_ans = r["answer"]
            
    return {"selected_answer": best_ans, "latency_seconds_total": lat, "prompt_tokens_total": pt, "completion_tokens_total": ct, "model_calls": 10, "route_used": "best_of_n_k5"}

def run_tot_k3(client, task, question):
    if task != "game24": return run_greedy_cot(client, task, question) # fallback
    prompt = get_tot_propose_prompt(question, k=3)
    r = client.generate(prompt, temperature=0.7)
    lat = r["latency_seconds"]; pt = r["prompt_tokens"]; ct = r["completion_tokens"]
    calls = 1
    
    # simplified zero-shot ToT: evaluate the proposals
    proposals = r["text"].split('\n')
    best_ans = None
    best_score = -1
    for p in proposals:
        if not p.strip(): continue
        vp = get_tot_value_prompt(question, p)
        vr = client.generate(vp, temperature=0.0)
        lat += vr["latency_seconds"]; pt += vr["prompt_tokens"]; ct += vr["completion_tokens"]
        calls += 1
        
        score = 0.5
        vtxt = vr["text"].lower()
        if "sure" in vtxt: score = 1.0
        elif "likely" in vtxt: score = 0.8
        elif "impossible" in vtxt: score = 0.0
        
        if score > best_score:
            best_score = score
            best_ans = get_parser(task)(p)["final_answer"]
            
    return {"selected_answer": best_ans, "latency_seconds_total": lat, "prompt_tokens_total": pt, "completion_tokens_total": ct, "model_calls": calls, "route_used": "zero_shot_tot"}

def main():
    check_ollama_running()
    run_self_tests()
    
    out_dir = Path("results/raw_logs")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    client = OllamaClient()
    verifier = OutcomeVerifier(ollama_client=client)
    tasks = load_all_tasks({"sampling": {"questions_per_task": 36, "seed": 0}})
    
    strategies = [
        ("greedy_io", run_greedy_io),
        ("greedy_cot", run_greedy_cot),
        ("self_consistency_k5", run_sc_k5),
        ("best_of_n_k5_self_eval", run_bon_k5),
        ("zero_shot_tot_k3", run_tot_k3),
        ("frugal_reason_v3", lambda c, t, q: frugal_reason_v3_evaluate(c, t, q, input_metadata=q, enable_early_exit=True, alpha=0.6))
    ]
    
    print("=============================================")
    print("         REAL-RUN MASTER EXPERIMENT          ")
    print("=============================================")
    
    for strat_name, strat_fn in strategies:
        out_file = out_dir / f"{strat_name}_raw_seed0.jsonl"
        if out_file.exists():
            os.remove(out_file)
            
        print(f"\n--- Running Strategy: {strat_name} ---")
        
        total = 0
        correct_count = 0
        
        for task_name, items in tasks.items():
            for i, item in enumerate(tqdm(items, desc=f"{strat_name} - {task_name}")):
                q_id = f"{task_name}_{i}"
                q_text = item["question"]
                gold = item["gold_answer"]
                
                try:
                    res = strat_fn(client, task_name, q_text)
                    if "latency_seconds_total" not in res:
                        # Normalize v3 naming
                        res["latency_seconds_total"] = res.get("latency", 0.0)
                        res["prompt_tokens_total"] = res.get("tokens", 0)
                        res["completion_tokens_total"] = 0
                        res["model_calls"] = res.get("calls", 0)
                except Exception as e:
                    print(f"ERROR on {q_id}: {e}")
                    sys.exit(1) # STRICT RULE: STOP on failure
                    
                ans = res.get("selected_answer", "")
                raw_out = res.get("raw_output", str(ans))
                eval_res = verifier.score(task_name, q_text, str(raw_out), str(ans), gold)
                is_correct = eval_res["score"] == 1.0
                
                total += 1
                if is_correct: correct_count += 1
                
                log_record = {
                    "strategy": strat_name,
                    "task": task_name,
                    "question_id": q_id,
                    "gold_answer": gold,
                    "correct": is_correct,
                    "strict_answer": is_correct,
                    "lenient_answer": is_correct,
                }
                log_record.update(res)
                
                with open(out_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(log_record) + "\n")
                    
        print(f"{strat_name} Complete. Accuracy: {correct_count}/{total} ({(correct_count/total)*100:.1f}%)")
        
    print("\nExecuting Post-Hoc Analysis...")
    os.system("python analysis/pav_analysis.py")
    os.system("python analysis/evaluate_locked.py")
    print("Done! All artifacts generated.")

if __name__ == "__main__":
    main()
