import os
import sys
import json
import time
import argparse
import random
import sqlite3
import re
from pathlib import Path
from huggingface_hub import HfApi

# Add experiment_fr to path
base_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(base_dir))

from core.ollama_client import OllamaClient
from core.task_loader import load_all_tasks
from core.parsers import get_parser
from core.verifier import OutcomeVerifier
from core.prompt_manager import get_prompt
from verifiers.verifiers import parse_judge_score
from strategies.frugal_reason_v3 import frugal_reason_v3_evaluate
from strategies.tot import zero_shot_tot_k3

def run_greedy_io(client, task, question):
    prompt = get_prompt("greedy_io", task, question)
    r = client.generate(prompt, temperature=0.0)
    p = get_parser(task)(r.get("text", ""))
    return {"selected_answer": p["final_answer"], "raw_response": r.get("text", ""),
            "latency_seconds_total": r.get("latency_seconds", 0.0), "total_tokens": r.get("total_tokens", 0),
            "model_calls": 1, "parse_success": p["parse_success"], "parse_method": p["parse_method"]}

def run_greedy_cot(client, task, question):
    prompt = get_prompt("greedy_cot", task, question)
    r = client.generate(prompt, temperature=0.0)
    p = get_parser(task)(r.get("text", ""))
    return {"selected_answer": p["final_answer"], "raw_response": r.get("text", ""),
            "latency_seconds_total": r.get("latency_seconds", 0.0), "total_tokens": r.get("total_tokens", 0),
            "model_calls": 1, "parse_success": p["parse_success"], "parse_method": p["parse_method"]}

def run_sc_k5(client, task, question):
    prompt = get_prompt("greedy_cot", task, question)
    answers = []; lat = 0; tok = 0; raws = []
    parser = get_parser(task)
    for _ in range(5):
        r = client.generate(prompt, temperature=0.7)
        lat += r.get("latency_seconds", 0.0); tok += r.get("total_tokens", 0); raws.append(r.get("text", ""))
        p = parser(r.get("text", ""))
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
        lat += r.get("latency_seconds", 0.0); tok += r.get("total_tokens", 0)
        p = parser(r.get("text", ""))
        rationales.append({"text": r.get("text", ""), "answer": p["final_answer"]})
    best_ans = None; best_score = -1; best_rat = ""; judge_texts = []
    for rat in rationales:
        jp = get_prompt("best_of_n", task="", question=question, candidate=rat["text"])
        jr = client.generate(jp, temperature=0.0)
        lat += jr.get("latency_seconds", 0.0); tok += jr.get("total_tokens", 0); judge_texts.append(jr.get("text", ""))
        score = parse_judge_score(jr.get("text", ""))
        if score > best_score: best_score = score; best_ans = rat["answer"]; best_rat = rat["text"]
    return {"selected_answer": best_ans,
            "raw_response": f"Selected:\n{best_rat}\n\nJudge:\n" + "\n---\n".join(judge_texts),
            "latency_seconds_total": lat, "total_tokens": tok, "model_calls": 10,
            "parse_success": best_ans is not None, "parse_method": "best_of_n_self_eval",
            "raw_paths": [r["text"] for r in rationales]}

def run_strategy(client, strat, task, question):
    if strat == "greedy_io": return run_greedy_io(client, task, question)
    elif strat == "greedy_cot": return run_greedy_cot(client, task, question)
    elif strat == "self_consistency_k5": return run_sc_k5(client, task, question)
    elif strat == "best_of_n_k5_self_eval": return run_bon_k5(client, task, question)
    elif strat == "zero_shot_tot_k3": return zero_shot_tot_k3(client, task, question)
    elif strat == "frugal_reason_v3":
        res = frugal_reason_v3_evaluate(client, task, question, input_metadata=question, enable_early_exit=True, alpha=0.6)
        return {"selected_answer": res.get("selected_answer"),
                "raw_response": json.dumps(res), "latency_seconds_total": res.get("latency", 0.0),
                "total_tokens": res.get("tokens", 0), "model_calls": res.get("calls", 0),
                "parse_success": res.get("parse_success", False), "parse_method": res.get("route_used", "frugal_reason_v3"),
                "raw_paths": [], "clusters": res.get("clusters", []), "candidates": res.get("candidates", []),
                "early_exit": res.get("early_exit", False)}
    raise ValueError(f"Unknown strategy: {strat}")

def main():
    parser = argparse.ArgumentParser(description="Mega Sweep Runner for 45-Day Calendar")
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g. qwen2.5:70b)")
    parser.add_argument("--datasets", type=str, default="gsm8k,aqua,math,strategyqa")
    parser.add_argument("--strategies", type=str, default="greedy_io,greedy_cot,self_consistency_k5,best_of_n_k5_self_eval,zero_shot_tot_k3,frugal_reason_v3")
    parser.add_argument("--n", type=int, default=100, help="Number of questions per dataset")
    args = parser.parse_args()

    MODEL = args.model
    DATASETS = args.datasets.split(",")
    STRATEGIES = args.strategies.split(",")
    SEED = 0

    print("Loading datasets...")
    loader_config = {"sampling": {"questions_per_task": 1500, "seed": SEED},
                     "tasks": {ds: {} for ds in DATASETS}}
    loaded = load_all_tasks(loader_config)

    # Load confirmatory QIDs
    qids_path = base_dir.parent.parent / "data" / "confirmatory_qids.json"
    conf_qids = {}
    if qids_path.exists():
        with open(qids_path) as f: conf_qids = json.load(f)

    # Prepare specific lists
    qid_lists = {}
    rng = random.Random(SEED)
    for ds in DATASETS:
        task_items = {item["id"]: item for item in loaded.get(ds, [])}
        all_ids = list(task_items.keys())
        if ds in conf_qids:
            cq = conf_qids[ds]
            if isinstance(cq, dict):
                flat = []
                for v in cq.values():
                    if isinstance(v, list): flat.extend(v)
                cq = flat
            qid_lists[ds] = cq[:args.n]
        else:
            qid_lists[ds] = rng.sample(all_ids, min(args.n, len(all_ids)))

    # DB and Dirs
    results_dir = base_dir / "results" / f"mega_sweeps"
    results_dir.mkdir(parents=True, exist_ok=True)
    db_path = base_dir / f"mega_checkpoint_{MODEL.replace(':', '_')}.db"
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute('''CREATE TABLE IF NOT EXISTS completed 
                   (model TEXT, dataset TEXT, strategy TEXT, qid TEXT, PRIMARY KEY(model, dataset, strategy, qid))''')
    conn.commit()

    client = OllamaClient(model=MODEL)
    verifier = OutcomeVerifier(client)
    
    total_target = sum(len(qid_lists[ds]) for ds in DATASETS) * len(STRATEGIES)
    done = 0; start = time.time()
    hw = "gpu" if os.path.exists("/proc/driver/nvidia") else "cpu"
    safe_model_name = MODEL.replace(':', '_')

    print(f"Target: {total_target} runs on {MODEL}")

    for ds in DATASETS:
        task_maps = {item["id"]: item for item in loaded.get(ds, [])}
        for strat in STRATEGIES:
            log_path = results_dir / f"{safe_model_name}_{ds}_{strat}.jsonl"
            for qid in qid_lists[ds]:
                cur.execute("SELECT 1 FROM completed WHERE model=? AND dataset=? AND strategy=? AND qid=?",
                            (MODEL, ds, strat, qid))
                if cur.fetchone():
                    done += 1; continue

                item = task_maps.get(qid)
                if not item: done += 1; continue

                try:
                    res = run_strategy(client, strat, ds, item["question"])
                except Exception as e:
                    print(f"  Failed {ds}/{strat}/{qid}: {e}")
                    continue

                ans = res.get("selected_answer", res.get("final_answer", ""))
                score_res = verifier.score(ds, item["question"], res.get("raw_response",""),
                                            ans, item["gold_answer"])
                is_correct = score_res["score"] == 1.0

                log_row = {
                    "model": MODEL, "dataset": ds, "strategy": strat, "qid": qid,
                    "gold": item["gold_answer"], "selected_answer": ans,
                    "correct": is_correct, "parse_success": res["parse_success"],
                    "parse_method": res.get("parse_method", ""), "latency_seconds": res["latency_seconds_total"],
                    "tokens": res["total_tokens"], "calls": res["model_calls"],
                    "hardware_type": hw, "early_exit": res.get("early_exit", False),
                    "clusters": res.get("clusters", []), "candidates": res.get("candidates", []),
                    "raw_paths": res.get("raw_paths", []),
                }
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(log_row) + "\n")
                cur.execute("INSERT INTO completed VALUES (?,?,?,?)", (MODEL, ds, strat, qid))
                conn.commit()
                done += 1

                if done % 10 == 0:
                    elapsed = time.time() - start
                    eta = (total_target - done) * (elapsed / max(done, 1))
                    print(f"[{MODEL}] {done}/{total_target} | {elapsed/3600:.1f}h elapsed | ETA {eta/3600:.1f}h")

    conn.close()
    print(f"\nMEGA SWEEP DONE: {MODEL}")

if __name__ == "__main__":
    main()
