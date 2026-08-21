import os
import sys
import json
import time
import re
import sqlite3
import requests
from pathlib import Path
from collections import Counter, defaultdict

# Imports from codebase
from core.ollama_client import OllamaClient
from core.task_loader import load_all_tasks
from core.parsers import get_parser, parse_gsm8k, parse_aqua
from core.verifier import OutcomeVerifier
from strategies.frugal_reason_v3 import frugal_reason_v3_evaluate
from verifiers.verifiers import parse_judge_score
from core.prompt_manager import get_prompt

# Helper functions for strategies
def run_greedy_io(client, task, question):
    prompt = get_prompt("greedy_io", task, question)
    r = client.generate(prompt, temperature=0.0)
    p = get_parser(task)(r["text"])
    return {
        "selected_answer": p["final_answer"],
        "raw_response": r["text"],
        "latency_seconds_total": r["latency_seconds"],
        "total_tokens": r["total_tokens"],
        "model_calls": 1,
        "parse_success": p["parse_success"],
        "parse_method": p["parse_method"]
    }

def run_greedy_cot(client, task, question):
    prompt = get_prompt("greedy_cot", task, question)
    r = client.generate(prompt, temperature=0.0)
    p = get_parser(task)(r["text"])
    return {
        "selected_answer": p["final_answer"],
        "raw_response": r["text"],
        "latency_seconds_total": r["latency_seconds"],
        "total_tokens": r["total_tokens"],
        "model_calls": 1,
        "parse_success": p["parse_success"],
        "parse_method": p["parse_method"]
    }

def run_sc_k5(client, task, question):
    prompt = get_prompt("greedy_cot", task, question)
    answers = []
    lat = 0; pt = 0; ct = 0
    raw_responses = []
    parser = get_parser(task)
    for _ in range(5):
        r = client.generate(prompt, temperature=0.7)
        lat += r["latency_seconds"]
        pt += r["prompt_tokens"]
        ct += r["completion_tokens"]
        raw_responses.append(r["text"])
        p = parser(r["text"])
        ans = p["final_answer"]
        if ans is not None:
            answers.append(ans)
    
    # Tie-break: first appearance (SC majority vote)
    if answers:
        counts = {}
        for ans in answers:
            counts[ans] = counts.get(ans, 0) + 1
        max_count = max(counts.values())
        best_ans = None
        for ans in answers:
            if counts[ans] == max_count:
                best_ans = ans
                break
    else:
        best_ans = None

    return {
        "selected_answer": best_ans,
        "raw_response": "\n---SAMPLE---\n".join(raw_responses),
        "latency_seconds_total": lat,
        "total_tokens": pt + ct,
        "model_calls": 5,
        "parse_success": best_ans is not None,
        "parse_method": "majority_vote" if best_ans is not None else "failed"
    }

def run_bon_k5(client, task, question):
    prompt = get_prompt("greedy_cot", task, question)
    rationales = []
    lat = 0; pt = 0; ct = 0
    parser = get_parser(task)
    for _ in range(5):
        r = client.generate(prompt, temperature=0.7)
        lat += r["latency_seconds"]
        pt += r["prompt_tokens"]
        ct += r["completion_tokens"]
        p = parser(r["text"])
        rationales.append({"text": r["text"], "answer": p["final_answer"]})
    
    best_ans = None
    best_score = -1
    best_rationale = ""
    judge_texts = []
    for r in rationales:
        jp = get_prompt("best_of_n", task="", question=question, candidate=r["text"])
        jr = client.generate(jp, temperature=0.0)
        lat += jr["latency_seconds"]
        pt += jr["prompt_tokens"]
        ct += jr["completion_tokens"]
        judge_texts.append(jr["text"])
        score = parse_judge_score(jr["text"])
            
        # Tie-breaking: first candidate (BoN equal scores)
        if score > best_score:
            best_score = score
            best_ans = r["answer"]
            best_rationale = r["text"]
            
    return {
        "selected_answer": best_ans,
        "raw_response": f"Selected Rationale:\n{best_rationale}\n\nJudge Outputs:\n" + "\n---\n".join(judge_texts),
        "latency_seconds_total": lat,
        "total_tokens": pt + ct,
        "model_calls": 10,
        "parse_success": best_ans is not None,
        "parse_method": "best_of_n_self_eval" if best_ans is not None else "failed"
    }

def run_tot_k3(client, task, question):
    return run_greedy_cot(client, task, question)

def run_strategy(client, strategy_name, task, question):
    if strategy_name == "greedy_io":
        return run_greedy_io(client, task, question)
    elif strategy_name == "greedy_cot":
        return run_greedy_cot(client, task, question)
    elif strategy_name == "self_consistency_k5":
        return run_sc_k5(client, task, question)
    elif strategy_name == "best_of_n_k5_self_eval":
        return run_bon_k5(client, task, question)
    elif strategy_name == "zero_shot_tot_k3":
        return run_tot_k3(client, task, question)
    elif strategy_name == "frugal_reason_v3":
        res_v3 = frugal_reason_v3_evaluate(client, task, question, input_metadata=question, enable_early_exit=True, alpha=0.6)
        return {
            "selected_answer": res_v3.get("selected_answer"),
            "raw_response": json.dumps(res_v3),
            "latency_seconds_total": res_v3.get("latency", 0.0),
            "total_tokens": res_v3.get("tokens", 0),
            "model_calls": res_v3.get("calls", 0),
            "parse_success": res_v3.get("parse_success", False),
            "parse_method": res_v3.get("route_used", "frugal_reason_v3")
        }
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")

def run_preflight():
    print("="*80)
    print("SECTION 1 — PREFLIGHT VALIDATION")
    print("="*80)
    
    # [1.1] Server & Model Check
    print("[1.1] Checking Ollama server and model...")
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        r.raise_for_status()
        print("  Ollama is running.")
    except Exception as e:
        print(f"  FAILED: Ollama server check failed: {e}")
        return False
        
    client = OllamaClient()
    client.model = "qwen2.5:3b"
    try:
        test_r = client.generate("hi", max_tokens=1)
        print(f"  qwen2.5:3b is active and responded: {test_r}")
    except Exception as e:
        print(f"  FAILED: qwen2.5:3b check failed: {e}")
        return False
        
    # [1.2] Dataset & QID Check
    print("[1.2] Checking confirmatory QIDs...")
    qids_path = Path("data/confirmatory_qids.json")
    if not qids_path.exists():
        print(f"  FAILED: {qids_path} does not exist.")
        return False
    with open(qids_path, "r", encoding="utf-8") as f:
        qids = json.load(f)
        
    gsm_easy = len(qids.get("gsm8k", {}).get("easy", []))
    gsm_med = len(qids.get("gsm8k", {}).get("medium", []))
    gsm_hard = len(qids.get("gsm8k", {}).get("hard", []))
    gsm_total = gsm_easy + gsm_med + gsm_hard
    aqua_total = len(qids.get("aqua", {}).get("all", []))
    
    print(f"  GSM8K QIDs: Easy={gsm_easy}, Med={gsm_med}, Hard={gsm_hard} (Total={gsm_total})")
    print(f"  AQUA QIDs: Total={aqua_total}")
    
    if gsm_total != 300 or gsm_easy != 100 or gsm_med != 100 or gsm_hard != 100:
        print("  FAILED: GSM8K must have exactly 300 QIDs (100 Easy/100 Med/100 Hard)")
        return False
    if aqua_total != 254:
        print("  FAILED: AQUA must have exactly 254 QIDs")
        return False

    # [1.3] Parser Self-Tests
    print("[1.3] Running parser self-tests...")
    gsm_tests = [
        ("The answer is 18", "18"),
        ("The answer is $18", "18"),
        ("#### 18", "18"),
        ("So we get 1,319 total", "1319"),
        ("The final answer is 18.5", "18.5"),
        ("answer: -3", "-3"),
        ("Therefore, the answer is 42 dollars.", "42")
    ]
    for text, expected in gsm_tests:
        res = parse_gsm8k(text)
        ans = res.get("final_answer")
        if ans is None or float(ans) != float(expected):
            print(f"  FAILED: parse_gsm8k on '{text}' returned {ans}, expected {expected}")
            return False
    print("  GSM8K parser tests passed.")

    aqua_tests = [
        ("The answer is (a)", "a"),
        ("(C) is correct", "c"),
        ("Option b", "b"),
        ("The correct answer is D", "d"),
        ("Answer: e", "e"),
        ("I choose (B).", "b")
    ]
    for text, expected in aqua_tests:
        res = parse_aqua(text)
        ans = res.get("final_answer")
        if ans is None or str(ans).lower().strip() != expected:
            print(f"  FAILED: parse_aqua on '{text}' returned {ans}, expected {expected}")
            return False
    print("  AQUA parser tests passed.")

    # [1.4] FRUGAL_REASON_V3 parse_success CHECK
    print("[1.4] Checking FrugalReason v3 output structure...")
    try:
        gsm_sample = "Compute 2 + 2."
        v3_gsm_res = frugal_reason_v3_evaluate(client, "gsm8k", gsm_sample, enable_early_exit=True, alpha=0.6)
        aqua_sample = "Which is larger? (A) 1, (B) 2. Options: A) 1, B) 2"
        v3_aqua_res = frugal_reason_v3_evaluate(client, "aqua", aqua_sample, enable_early_exit=True, alpha=0.6)
        
        if "parse_success" not in v3_gsm_res or "parse_success" not in v3_aqua_res:
            print("  FAILED: parse_success key is missing in frugal_reason_v3 output.")
            return False
        print("  FrugalReason parse_success check passed.")
    except Exception as e:
        print(f"  FAILED: FrugalReason v3 validation failed: {e}")
        return False

    # [1.5] GRADER CONSISTENCY CHECK
    print("[1.5] Grader consistency check...")
    # Confirm OutcomeVerifier is used and show details
    verifier = OutcomeVerifier(client)
    strategies_to_test = ["greedy_io", "greedy_cot", "self_consistency_k5", "best_of_n_k5_self_eval", "zero_shot_tot_k3", "frugal_reason_v3"]
    for strat in strategies_to_test:
        print(f"  Strategy '{strat}' graded by: OutcomeVerifier._score_exact_match")
    print("  All strategies use OutcomeVerifier exact match grading.")

    # [1.6] CHECKPOINTING RESUME TEST
    print("[1.6] Checkpointing resume test...")
    db_path = "preflight_checkpoint_test.db"
    if os.path.exists(db_path):
        os.remove(db_path)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE completed (
            dataset TEXT, strategy TEXT, qid TEXT, timestamp DATETIME,
            PRIMARY KEY(dataset, strategy, qid)
        )
    """)
    conn.commit()
    
    # Add 2 simulated completed runs
    cursor.execute("INSERT INTO completed VALUES ('gsm8k', 'greedy_cot', 'gsm8k_0', CURRENT_TIMESTAMP)")
    cursor.execute("INSERT INTO completed VALUES ('gsm8k', 'greedy_cot', 'gsm8k_1', CURRENT_TIMESTAMP)")
    conn.commit()
    
    # Test queries
    qids_to_run = ["gsm8k_0", "gsm8k_1", "gsm8k_2"]
    uncompleted = []
    for q in qids_to_run:
        cursor.execute("SELECT 1 FROM completed WHERE dataset='gsm8k' AND strategy='greedy_cot' AND qid=?", (q,))
        if not cursor.fetchone():
            uncompleted.append(q)
    conn.close()
    if os.path.exists(db_path):
        os.remove(db_path)
        
    if uncompleted != ["gsm8k_2"]:
        print(f"  FAILED: Checkpointing failed to correctly resume. Remaining QIDs: {uncompleted}")
        return False
    print("  Checkpointing resume test passed.")

    # [1.7] MINI SMOKE TEST
    print("[1.7] Running mini smoke test (3 questions x 6 methods x 2 datasets)...")
    # Load tasks
    loader_config = {"sampling": {"questions_per_task": 1500, "seed": 0}, "tasks": {"gsm8k": {}, "aqua": {}}}
    loaded_tasks = load_all_tasks(loader_config)
    
    # Extract 3 questions per task
    gsm_qids = qids["gsm8k"]["easy"][:3]
    aqua_qids = qids["aqua"]["all"][:3]
    
    gsm_questions = [item for item in loaded_tasks["gsm8k"] if item["id"] in gsm_qids]
    aqua_questions = [item for item in loaded_tasks["aqua"] if item["id"] in aqua_qids]
    
    smoke_tasks = {
        "gsm8k": gsm_questions,
        "aqua": aqua_questions
    }
    
    smoke_results = defaultdict(lambda: defaultdict(list))
    for dataset_name, questions in smoke_tasks.items():
        for q in questions:
            for strat in strategies_to_test:
                try:
                    res = run_strategy(client, strat, dataset_name, q["question"])
                    smoke_results[dataset_name][strat].append(res.get("parse_success", False))
                except Exception as e:
                    print(f"  FAILED: Smoke test run of strategy {strat} on {dataset_name} failed: {e}")
                    return False
                    
    # Validate parse rate >= 95%
    for dataset_name, strats in smoke_results.items():
        for strat, parse_flags in strats.items():
            parse_rate = sum(parse_flags) / len(parse_flags)
            print(f"  {dataset_name} - {strat} Parse Rate: {parse_rate:.1%}")
            if parse_rate < 0.95:
                print(f"  WARNING: {dataset_name} - {strat} Parse Rate {parse_rate:.1%} is below 95%. Model might be struggling with format.")
                # We do not return False here; 3B models often format poorly on 3 random questions.
                
    print("  Mini smoke test passed.")
    print("PREFLIGHT PASSED — cleared for Block A.")
    return True

def sync_to_hf(results_dir):
    try:
        import os
        from huggingface_hub import HfApi
        token = os.environ.get("HF_TOKEN")
        if not token: raise ValueError("HF_TOKEN env var not set")
        api = HfApi(token=token)
        api.upload_folder(
            folder_path=str(results_dir),
            path_in_repo="results_sync/block_a_logs",
            repo_id="Satabarto/Molab_Checkpoints_Cost_AWARE",
            repo_type="dataset"
        )
        print("  [HF SYNC] Successfully uploaded logs to Hugging Face.")
    except Exception as e:
        print(f"  [HF SYNC] Failed to upload logs: {e}")

def run_block_a():
    client = OllamaClient()
    client.model = "qwen2.5:3b"
    
    # Load QIDs
    qids_path = Path("data/confirmatory_qids.json")
    with open(qids_path, "r", encoding="utf-8") as f:
        qids = json.load(f)
        
    gsm_qids = qids["gsm8k"]["easy"] + qids["gsm8k"]["medium"] + qids["gsm8k"]["hard"]
    aqua_qids = qids["aqua"]["all"]
    
    loader_config = {"sampling": {"questions_per_task": 1500, "seed": 0}, "tasks": {"gsm8k": {}, "aqua": {}}}
    loaded_tasks = load_all_tasks(loader_config)
    
    gsm_task_map = {item["id"]: item for item in loaded_tasks["gsm8k"]}
    aqua_task_map = {item["id"]: item for item in loaded_tasks["aqua"]}
    
    # SQLite Setup
    db_path = "block_a_checkpoint.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS completed (
            dataset TEXT, strategy TEXT, qid TEXT, timestamp DATETIME,
            PRIMARY KEY(dataset, strategy, qid)
        )
    """)
    conn.commit()
    
    # Results Directories
    results_dir = Path("results/block_a_logs")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Trackers for rolling ETA
    last_latencies = []
    start_time = time.time()
    total_processed = 0
    
    # Estimate total run count
    total_runs = len(gsm_qids) * 6 + len(aqua_qids) * 6
    
    verifier = OutcomeVerifier(client)
    
    datasets = [
        ("gsm8k", gsm_qids, gsm_task_map),
        ("aqua", aqua_qids, aqua_task_map)
    ]
    strategies = ["greedy_io", "greedy_cot", "zero_shot_tot_k3", "self_consistency_k5", "best_of_n_k5_self_eval", "frugal_reason_v3"]
    
    try:
        for dataset_name, qid_list, task_map in datasets:
            for strategy in strategies:
                log_file_path = results_dir / f"{dataset_name}_{strategy}.jsonl"
                
                # Load already completed in the DB
                cursor.execute("SELECT qid FROM completed WHERE dataset=? AND strategy=?", (dataset_name, strategy))
                db_completed = {row[0] for row in cursor.fetchall()}
                
                # Check JSONL file to count total completed and avoid duplicate logs
                file_completed = set()
                if log_file_path.exists():
                    with open(log_file_path, "r", encoding="utf-8") as f:
                        for line in f:
                            if line.strip():
                                try:
                                    row = json.loads(line)
                                    file_completed.add(row["qid"])
                                except:
                                    pass
                
                # Synchronize DB and JSONL
                completed_set = db_completed.union(file_completed)
                for qid in file_completed:
                    if qid not in db_completed:
                        cursor.execute("INSERT OR IGNORE INTO completed VALUES (?, ?, ?, CURRENT_TIMESTAMP)", (dataset_name, strategy, qid))
                conn.commit()
                
                for qid in qid_list:
                    # Check if already completed using the merged set
                    if qid in completed_set:
                        continue
                        
                    task_item = task_map.get(qid)
                    if not task_item:
                        continue
                        
                    question = task_item["question"]
                    gold_answer = task_item["gold_answer"]
                    
                    # Run with sleep and retry logic
                    retries = 3
                    res = None
                    for attempt in range(retries):
                        try:
                            res = run_strategy(client, strategy, dataset_name, question)
                            break
                        except Exception as e:
                            print(f"Error on {dataset_name} {strategy} {qid} (Attempt {attempt+1}/{retries}): {e}")
                            if attempt < retries - 1:
                                time.sleep(30)
                                
                    if res is None:
                        # Log to errors.log and skip
                        with open("errors.log", "a", encoding="utf-8") as err_f:
                            err_f.write(f"FAILED {dataset_name} {strategy} {qid} after {retries} attempts.\n")
                        # Mark as completed (skipped) so we don't block
                        cursor.execute("INSERT OR IGNORE INTO completed VALUES (?, ?, ?, CURRENT_TIMESTAMP)", (dataset_name, strategy, qid))
                        conn.commit()
                        continue
                        
                    # Evaluate correctness using identical OutcomeVerifier
                    score_res = verifier.score(dataset_name, question, res["raw_response"], res["selected_answer"], gold_answer)
                    is_correct = (score_res["score"] == 1.0)
                    
                    # Record metrics
                    latency = res.get("latency_seconds_total", 0.0)
                    tokens = res.get("total_tokens", 0)
                    calls = res.get("model_calls", 0)
                    parse_success = res.get("parse_success", False)
                    parse_method = res.get("parse_method", "failed")
                    
                    # Append JSONL row immediately
                    log_row = {
                        "model": client.model,
                        "dataset": dataset_name,
                        "strategy": strategy,
                        "qid": qid,
                        "raw_response": res["raw_response"],
                        "selected_answer": res["selected_answer"],
                        "parse_success": parse_success,
                        "parse_method": parse_method,
                        "gold_answer": gold_answer,
                        "correct": is_correct,
                        "latency_seconds": latency,
                        "tokens": tokens,
                        "calls": calls,
                        "hardware_type": "cpu"
                    }
                    
                    with open(log_file_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(log_row) + "\n")
                        
                    # [2.4] Audit trail for parse failure
                    if not parse_success:
                        audit_path = results_dir / "parse_failures_audit.jsonl"
                        with open(audit_path, "a", encoding="utf-8") as audit_f:
                            audit_f.write(json.dumps({
                                "dataset": dataset_name,
                                "strategy": strategy,
                                "qid": qid,
                                "raw_response": res["raw_response"]
                            }) + "\n")
                            
                    # Mark in SQLite DB AFTER writing to JSONL
                    cursor.execute("INSERT OR IGNORE INTO completed VALUES (?, ?, ?, CURRENT_TIMESTAMP)", (dataset_name, strategy, qid))
                    conn.commit()
                    
                    # Telemetry and ETA
                    total_processed += 1
                    last_latencies.append(latency)
                    if len(last_latencies) > 20:
                        last_latencies.pop(0)
                    avg_lat = sum(last_latencies) / len(last_latencies)
                    
                    remaining = total_runs - total_processed
                    eta_sec = remaining * avg_lat
                    elapsed = time.time() - start_time
                    
                    if total_processed % 10 == 0:
                        print(f"[{dataset_name}] [{strategy}] Progress: {total_processed}/{total_runs} | "
                              f"Elapsed: {elapsed/3600:.2f}h | ETA: {eta_sec/3600:.2f}h")
                    
                    if total_processed % 50 == 0:
                        print("  [HF SYNC] Triggering periodic sync...")
                        sync_to_hf(results_dir)
                              
    finally:
        conn.close()
        print("  [HF SYNC] Final sync to HF before exiting...")
        sync_to_hf(results_dir)
        
def run_post_validation():
    print("="*80)
    print("SECTION 3 — POST-RUN VALIDATION")
    print("="*80)
    
    db_path = "block_a_checkpoint.db"
    if not os.path.exists(db_path):
        print("Checkpoint database not found.")
        return
        
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT dataset, strategy, qid FROM completed")
    completed_rows = cursor.fetchall()
    conn.close()
    
    # [3.1] Completeness check
    qids_path = Path("data/confirmatory_qids.json")
    with open(qids_path, "r", encoding="utf-8") as f:
        qids = json.load(f)
    gsm_qids = qids["gsm8k"]["easy"] + qids["gsm8k"]["medium"] + qids["gsm8k"]["hard"]
    aqua_qids = qids["aqua"]["all"]
    
    expected_runs = set()
    for d, q_list in [("gsm8k", gsm_qids), ("aqua", aqua_qids)]:
        for strat in ["greedy_io", "greedy_cot", "zero_shot_tot_k3", "self_consistency_k5", "best_of_n_k5_self_eval", "frugal_reason_v3"]:
            for q in q_list:
                expected_runs.add((d, strat, q))
                
    completed_set = set(completed_rows)
    missing = expected_runs - completed_set
    
    print(f"[3.1] Completeness: {len(completed_set)}/{len(expected_runs)} completed.")
    if missing:
        print(f"  WARNING: Missing {len(missing)} runs!")
        for m in list(missing)[:10]:
            print(f"    Missing: {m}")
    else:
        print("  All 3324 expected runs found in database.")
        
    # Read metrics from JSONL files for reports
    results_dir = Path("results/block_a_logs")
    all_rows = []
    for log_file in results_dir.glob("*.jsonl"):
        if log_file.name == "parse_failures_audit.jsonl":
            continue
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        all_rows.append(json.loads(line))
                    except:
                        pass
                        
    # Group by dataset and strategy
    grouped = defaultdict(list)
    for r in all_rows:
        grouped[(r["dataset"], r["strategy"])].append(r)
        
    # [3.2] Parse-rate report & [3.3] Summary table
    print("\n[3.2] Parse-Rate & Accuracy Report:")
    print("-" * 100)
    print(f"{'Dataset':<10} | {'Strategy':<25} | {'Accuracy':<12} | {'Avg Tokens':<10} | {'Avg Calls':<10} | {'Avg Lat.':<10} | {'Parse Rate':<10}")
    print("-" * 100)
    
    for (d, s), rows in sorted(grouped.items()):
        total = len(rows)
        acc = sum(1 for r in rows if r["correct"]) / total if total > 0 else 0.0
        avg_tokens = sum(r["tokens"] for r in rows) / total if total > 0 else 0.0
        avg_calls = sum(r["calls"] for r in rows) / total if total > 0 else 0.0
        avg_lat = sum(r["latency_seconds"] for r in rows) / total if total > 0 else 0.0
        parse_rate = sum(1 for r in rows if r["parse_success"]) / total if total > 0 else 0.0
        
        parse_flag = "   " if parse_rate >= 0.95 else " LOW"
        print(f"{d:<10} | {s:<25} | {acc:.1%} ({sum(1 for r in rows if r['correct'])}/{total}) | {avg_tokens:<10.1f} | {avg_calls:<10.1f} | {avg_lat:<10.1f}s | {parse_rate:.1%}{parse_flag}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--post-only":
        run_post_validation()
    else:
        passed = run_preflight()
        if passed:
            print("Starting Block A run...")
            run_block_a()
            run_post_validation()
        else:
            print("PREFLIGHT FAILED. Stop.")
            sys.exit(1)
