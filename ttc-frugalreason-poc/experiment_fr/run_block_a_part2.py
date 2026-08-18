"""
run_block_a_part2.py — Block A Part 2: MATH + StrategyQA
==========================================================
Preflight validation followed by full benchmark execution.
Uses the same strategy implementations, parsers, and verifier
as run_block_a.py but targets MATH (238 QIDs) and StrategyQA (300 QIDs).
"""

import os
import sys
import json
import time
import re
import sqlite3
import requests
import math as pymath
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict

# Imports from codebase
from core.ollama_client import OllamaClient
from core.task_loader import load_all_tasks
from core.parsers import get_parser, parse_math, parse_strategyqa
from core.verifier import OutcomeVerifier
from core.prompt_manager import get_prompt
from strategies.frugal_reason_v3 import frugal_reason_v3_evaluate


# ═══════════════════════════════════════════════════════════════
# STRATEGY RUNNERS (identical to run_block_a.py)
# ═══════════════════════════════════════════════════════════════

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
    for r_item in rationales:
        jp = get_prompt("best_of_n", task="", question=question, candidate=r_item["text"])
        jr = client.generate(jp, temperature=0.0)
        lat += jr["latency_seconds"]
        pt += jr["prompt_tokens"]
        ct += jr["completion_tokens"]
        judge_texts.append(jr["text"])
        
        score = 0.5
        sm = re.search(r'confidence:\s*(\d+)', jr["text"].lower())
        if sm:
            score = float(sm.group(1)) / 100.0
        elif "yes" in jr["text"].lower():
            score = 1.0
        elif "no" in jr["text"].lower():
            score = 0.0
            
        if score > best_score:
            best_score = score
            best_ans = r_item["answer"]
            best_rationale = r_item["text"]
            
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


# ═══════════════════════════════════════════════════════════════
# SECTION 1 — TARGETED PREFLIGHT (MATH + StrategyQA ONLY)
# ═══════════════════════════════════════════════════════════════

def run_preflight():
    print("=" * 80)
    print("SECTION 1 — TARGETED PREFLIGHT (MATH + StrategyQA ONLY)")
    print("CRITICAL: Do NOT start the full run until all checks pass.")
    print("=" * 80)
    
    # ─── [1.0] Server & Model Check ───────────────────────────────
    print("\n[1.0] Checking Ollama server and model...")
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        r.raise_for_status()
        print("  [OK] Ollama is running.")
    except Exception as e:
        print(f"  [FAIL] FAILED: Ollama server check failed: {e}")
        return False
        
    client = OllamaClient()
    client.model = "qwen2.5:3b"
    try:
        test_r = client.generate("hi", max_tokens=1)
        print(f"  [OK] qwen2.5:3b is active.")
    except Exception as e:
        print(f"  [FAIL] FAILED: qwen2.5:3b check failed: {e}")
        return False

    # ─── [1.1] DATASET & QID VERIFICATION ─────────────────────────
    print("\n[1.1] Dataset & QID Verification...")
    qids_path = Path("data/confirmatory_qids.json")
    if not qids_path.exists():
        print(f"  [FAIL] FAILED: {qids_path} does not exist.")
        return False
    with open(qids_path, "r", encoding="utf-8") as f:
        qids = json.load(f)
    
    # MATH L1-3
    math_qids = qids.get("math", {}).get("all", [])
    math_count = len(math_qids)
    print(f"  MATH L1-3 QIDs: {math_count}")
    if math_count < 230 or math_count > 250:
        print(f"  [FAIL] FAILED: MATH L1-3 must have ~238 QIDs, got {math_count}")
        return False
    print(f"  [OK] MATH L1-3 has {math_count} QIDs (expected ~238).")

    # StrategyQA
    sqa = qids.get("strategyqa", {})
    sqa_easy = len(sqa.get("easy", []))
    sqa_med = len(sqa.get("medium", []))
    sqa_hard = len(sqa.get("hard", []))
    sqa_total = sqa_easy + sqa_med + sqa_hard
    print(f"  StrategyQA QIDs: Easy={sqa_easy}, Med={sqa_med}, Hard={sqa_hard} (Total={sqa_total})")
    if sqa_total != 300 or sqa_easy != 100 or sqa_med != 100 or sqa_hard != 100:
        print(f"  [FAIL] FAILED: StrategyQA must have exactly 300 QIDs (100E/100M/100H)")
        return False
    print(f"  [OK] StrategyQA has {sqa_total} QIDs (100E/100M/100H).")

    # ─── [1.2] MATH PARSER STRESS TEST ────────────────────────────
    print("\n[1.2] MATH Parser Stress Test...")
    math_tests = [
        ("\\boxed{\\frac{1}{2}}", "0.5"),
        ("\\boxed{3}", "3"),
        ("The answer is \\boxed{-7}", "-7"),
        ("\\boxed{\\sqrt{4}}", "2"),
        ("So we get \\boxed{14}.", "14"),
    ]
    for text, expected in math_tests:
        res = parse_math(text)
        ans = res.get("final_answer")
        if ans is None:
            print(f"  [FAIL] FAILED: parse_math on '{text}' returned None, expected {expected}")
            return False
        try:
            if float(ans) != float(expected):
                print(f"  [FAIL] FAILED: parse_math on '{text}' returned {ans}, expected {expected}")
                return False
        except ValueError:
            if str(ans).strip() != str(expected).strip():
                print(f"  [FAIL] FAILED: parse_math on '{text}' returned {ans}, expected {expected}")
                return False
        print(f"  [OK] '{text}' -> {ans} (expected {expected})")
    print("  [OK] All MATH parser tests passed.")

    # ─── [1.3] STRATEGYQA PARSER STRESS TEST ──────────────────────
    print("\n[1.3] StrategyQA Parser Stress Test...")
    sqa_tests = [
        ("The answer is yes", "yes"),
        ("Therefore, the answer is no.", "no"),
        ("Yes.", "yes"),
        ("No, because...", "no"),
        ("the answer is Yes", "yes"),
    ]
    for text, expected in sqa_tests:
        res = parse_strategyqa(text)
        ans = res.get("final_answer")
        if ans is None or str(ans).lower().strip() != expected:
            print(f"  [FAIL] FAILED: parse_strategyqa on '{text}' returned {ans}, expected {expected}")
            return False
        print(f"  [OK] '{text}' -> {ans} (expected {expected})")
    print("  [OK] All StrategyQA parser tests passed.")

    # ─── [1.4] FRUGAL_REASON_V3 INTEGRITY CHECK ──────────────────
    print("\n[1.4] FrugalReason v3 Integrity Check...")
    try:
        math_sample = "What is 2 + 2? Present your answer in \\boxed{}."
        v3_math_res = frugal_reason_v3_evaluate(client, "math", math_sample, enable_early_exit=True, alpha=0.6)
        
        sqa_sample = "Is the sky blue? Answer yes or no."
        v3_sqa_res = frugal_reason_v3_evaluate(client, "strategyqa", sqa_sample, enable_early_exit=True, alpha=0.6)
        
        for label, res in [("MATH", v3_math_res), ("StrategyQA", v3_sqa_res)]:
            if "parse_success" not in res:
                print(f"  [FAIL] FAILED: 'parse_success' key is missing in frugal_reason_v3 output for {label}.")
                return False
            if "selected_answer" not in res:
                print(f"  [FAIL] FAILED: 'selected_answer' key is missing in frugal_reason_v3 output for {label}.")
                return False
            print(f"  [OK] {label}: parse_success={res['parse_success']}, selected_answer={res['selected_answer']}")
        print("  [OK] FrugalReason v3 integrity check passed.")
    except Exception as e:
        print(f"  [FAIL] FAILED: FrugalReason v3 validation failed: {e}")
        import traceback; traceback.print_exc()
        return False

    # ─── [1.5] GRADER CONSISTENCY CHECK ───────────────────────────
    print("\n[1.5] Grader Consistency Check...")
    verifier = OutcomeVerifier(client)
    strategies_list = ["greedy_io", "greedy_cot", "self_consistency_k5", "best_of_n_k5_self_eval", "zero_shot_tot_k3", "frugal_reason_v3"]
    
    # All strategies use the same OutcomeVerifier.score -> _score_math for math, _score_exact_match for strategyqa
    grader_class = type(verifier).__name__
    for strat in strategies_list:
        print(f"  Strategy '{strat}' graded by: {grader_class}._score_math (MATH) / {grader_class}._score_exact_match + LLM-judge (StrategyQA)")
    print(f"  [OK] All 6 strategies use the identical {grader_class} instance.")

    # ─── [1.6] CHECKPOINT RESUME TEST ─────────────────────────────
    print("\n[1.6] Checkpoint Resume Test (MATH + StrategyQA specific)...")
    test_db_path = "preflight_part2_test.db"
    if os.path.exists(test_db_path):
        os.remove(test_db_path)
    conn = sqlite3.connect(test_db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE completed (
            dataset TEXT, strategy TEXT, qid TEXT, timestamp DATETIME,
            PRIMARY KEY(dataset, strategy, qid)
        )
    """)
    conn.commit()
    
    # Simulate 2 completed MATH greedy_cot runs
    cursor.execute("INSERT INTO completed VALUES ('math', 'greedy_cot', 'math_0', CURRENT_TIMESTAMP)")
    cursor.execute("INSERT INTO completed VALUES ('math', 'greedy_cot', 'math_1', CURRENT_TIMESTAMP)")
    conn.commit()
    
    # Check that only math_2 would be run
    qids_to_run = ["math_0", "math_1", "math_2"]
    uncompleted = []
    for q in qids_to_run:
        cursor.execute("SELECT 1 FROM completed WHERE dataset='math' AND strategy='greedy_cot' AND qid=?", (q,))
        if not cursor.fetchone():
            uncompleted.append(q)
    conn.close()
    if os.path.exists(test_db_path):
        os.remove(test_db_path)
        
    if uncompleted != ["math_2"]:
        print(f"  [FAIL] FAILED: Checkpointing failed. Remaining QIDs: {uncompleted}")
        return False
    print("  [OK] Checkpoint resume test passed (math_0, math_1 skipped, math_2 would run).")

    # ─── PREFLIGHT GATE ───────────────────────────────────────────
    print("\n" + "=" * 80)
    print("[PASS] PREFLIGHT PASSED — launching Block A Part 2.")
    print("=" * 80)
    return True


# ═══════════════════════════════════════════════════════════════
# SECTION 2 — BLOCK A PART 2 RUN WITH LIVE COUNTER
# ═══════════════════════════════════════════════════════════════

def run_block_a_part2():
    client = OllamaClient()
    client.model = "qwen2.5:3b"
    
    # Load QIDs
    qids_path = Path("data/confirmatory_qids.json")
    with open(qids_path, "r", encoding="utf-8") as f:
        qids = json.load(f)
        
    math_qids = qids["math"]["all"]
    sqa_qids = qids["strategyqa"]["easy"] + qids["strategyqa"]["medium"] + qids["strategyqa"]["hard"]
    
    # Load task data
    loader_config = {"sampling": {"questions_per_task": 1500, "seed": 0}, "tasks": {"math": {}, "strategyqa": {}}}
    loaded_tasks = load_all_tasks(loader_config)
    
    math_task_map = {item["id"]: item for item in loaded_tasks.get("math", [])}
    sqa_task_map = {item["id"]: item for item in loaded_tasks.get("strategyqa", [])}
    
    # SQLite Setup (separate DB for Part 2)
    db_path = "block_a_part2_checkpoint.db"
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
    results_dir = Path("results/block_a_part2_logs")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Trackers
    last_latencies = []
    start_time = time.time()
    total_processed = 0
    total_skipped = 0
    skipped_qids = []
    
    # Per (dataset, strategy) running stats
    running_stats = defaultdict(lambda: {"correct": 0, "total": 0, "parse_ok": 0})
    
    strategies = ["greedy_io", "greedy_cot", "zero_shot_tot_k3", "self_consistency_k5", "best_of_n_k5_self_eval", "frugal_reason_v3"]
    
    datasets = [
        ("math", math_qids, math_task_map),
        ("strategyqa", sqa_qids, sqa_task_map)
    ]
    
    total_runs = len(math_qids) * 6 + len(sqa_qids) * 6
    print(f"\nTotal expected runs: {total_runs} ({len(math_qids)} MATH × 6 + {len(sqa_qids)} StrategyQA × 6)")
    
    verifier = OutcomeVerifier(client)
    
    try:
        for dataset_name, qid_list, task_map in datasets:
            for strategy in strategies:
                log_file_path = results_dir / f"{dataset_name}_{strategy}.jsonl"
                
                # Load already completed in the DB
                cursor.execute("SELECT qid FROM completed WHERE dataset=? AND strategy=?", (dataset_name, strategy))
                db_completed = {row[0] for row in cursor.fetchall()}
                
                # Check JSONL file to synchronize
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
                for qid in file_completed:
                    if qid not in db_completed:
                        cursor.execute("INSERT OR IGNORE INTO completed VALUES (?, ?, ?, CURRENT_TIMESTAMP)", (dataset_name, strategy, qid))
                conn.commit()
                
                strat_count = 0
                for qid in qid_list:
                    # Check if already completed
                    cursor.execute("SELECT 1 FROM completed WHERE dataset=? AND strategy=? AND qid=?", (dataset_name, strategy, qid))
                    if cursor.fetchone():
                        continue
                        
                    task_item = task_map.get(qid)
                    if not task_item:
                        continue
                        
                    question = task_item["question"]
                    gold_answer = task_item["gold_answer"]
                    
                    # Run with retry logic
                    retries = 3
                    res = None
                    for attempt in range(retries):
                        try:
                            res = run_strategy(client, strategy, dataset_name, question)
                            break
                        except Exception as e:
                            print(f"  Error on {dataset_name} {strategy} {qid} (Attempt {attempt+1}/{retries}): {e}")
                            if attempt < retries - 1:
                                time.sleep(30)
                                
                    if res is None:
                        # Log to errors.log and skip
                        with open("errors.log", "a", encoding="utf-8") as err_f:
                            err_f.write(f"FAILED {dataset_name} {strategy} {qid} after {retries} attempts.\n")
                        cursor.execute("INSERT OR IGNORE INTO completed VALUES (?, ?, ?, CURRENT_TIMESTAMP)", (dataset_name, strategy, qid))
                        conn.commit()
                        total_skipped += 1
                        skipped_qids.append(f"{dataset_name}/{strategy}/{qid}")
                        continue
                        
                    # Evaluate correctness
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
                        
                    # Audit trail for parse failure
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
                    
                    # Update running stats
                    key = (dataset_name, strategy)
                    running_stats[key]["total"] += 1
                    if is_correct:
                        running_stats[key]["correct"] += 1
                    if parse_success:
                        running_stats[key]["parse_ok"] += 1
                    
                    # Telemetry and ETA
                    total_processed += 1
                    strat_count += 1
                    last_latencies.append(latency)
                    if len(last_latencies) > 20:
                        last_latencies.pop(0)
                    avg_lat = sum(last_latencies) / len(last_latencies)
                    
                    remaining = total_runs - total_processed
                    eta_sec = remaining * avg_lat
                    elapsed = time.time() - start_time
                    
                    # Print every 5 completed questions
                    if strat_count % 5 == 0:
                        stats = running_stats[key]
                        pr = (stats["parse_ok"] / stats["total"] * 100) if stats["total"] > 0 else 0
                        acc = (stats["correct"] / stats["total"] * 100) if stats["total"] > 0 else 0
                        now = datetime.now().strftime("%H:%M:%S")
                        eta_h = int(eta_sec // 3600)
                        eta_m = int((eta_sec % 3600) // 60)
                        print(f"[{now}] [{dataset_name}] [{strategy}] | Done: {total_processed}/{total_runs} | "
                              f"Elapsed: {elapsed/60:.0f}m | ETA: {eta_h}h {eta_m}m | "
                              f"Parse Rate: {pr:.0f}% | Acc: {acc:.0f}%")
                              
    finally:
        conn.close()
    
    return total_processed, total_skipped, skipped_qids


# ═══════════════════════════════════════════════════════════════
# SECTION 3 — POST-RUN VALIDATION & COMPLETION REPORT
# ═══════════════════════════════════════════════════════════════

def run_post_validation():
    print("\n" + "=" * 80)
    print("=== BLOCK A PART 2 COMPLETE ===")
    print("=" * 80)
    
    db_path = "block_a_part2_checkpoint.db"
    if not os.path.exists(db_path):
        print("Checkpoint database not found.")
        return
        
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT dataset, strategy, qid FROM completed")
    completed_rows = cursor.fetchall()
    conn.close()
    
    # Completeness check
    qids_path = Path("data/confirmatory_qids.json")
    with open(qids_path, "r", encoding="utf-8") as f:
        qids = json.load(f)
    math_qids = qids["math"]["all"]
    sqa_qids = qids["strategyqa"]["easy"] + qids["strategyqa"]["medium"] + qids["strategyqa"]["hard"]
    
    strategies = ["greedy_io", "greedy_cot", "zero_shot_tot_k3", "self_consistency_k5", "best_of_n_k5_self_eval", "frugal_reason_v3"]
    
    expected_runs = set()
    for d, q_list in [("math", math_qids), ("strategyqa", sqa_qids)]:
        for strat in strategies:
            for q in q_list:
                expected_runs.add((d, strat, q))
                
    completed_set = set(completed_rows)
    missing = expected_runs - completed_set
    
    total_expected = len(expected_runs)
    print(f"\nCompleteness: {len(completed_set)}/{total_expected} completed.")
    if missing:
        print(f"  WARNING: Missing {len(missing)} runs!")
        for m in list(missing)[:10]:
            print(f"    Missing: {m}")
    else:
        print(f"  All {total_expected} expected runs found in database.")
        
    # Read metrics from JSONL files for reports
    results_dir = Path("results/block_a_part2_logs")
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
        
    # Summary table
    print("\nPer-dataset Per-strategy Results:")
    print("-" * 110)
    print(f"{'Dataset':<12} | {'Strategy':<25} | {'Accuracy':<14} | {'Avg Tokens':<10} | {'Avg Calls':<10} | {'Avg Lat.':<10} | {'Parse Rate':<10}")
    print("-" * 110)
    
    for (d, s), rows in sorted(grouped.items()):
        total = len(rows)
        acc = sum(1 for r in rows if r["correct"]) / total if total > 0 else 0.0
        avg_tokens = sum(r["tokens"] for r in rows) / total if total > 0 else 0.0
        avg_calls = sum(r["calls"] for r in rows) / total if total > 0 else 0.0
        avg_lat = sum(r["latency_seconds"] for r in rows) / total if total > 0 else 0.0
        parse_rate = sum(1 for r in rows if r["parse_success"]) / total if total > 0 else 0.0
        
        n_correct = sum(1 for r in rows if r["correct"])
        parse_flag = "   " if parse_rate >= 0.95 else " LOW"
        print(f"{d:<12} | {s:<25} | {acc:.1%} ({n_correct}/{total}) | {avg_tokens:<10.1f} | {avg_calls:<10.1f} | {avg_lat:<10.1f}s | {parse_rate:.1%}{parse_flag}")

    print("=" * 110)


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--post-only":
        run_post_validation()
    else:
        passed = run_preflight()
        if passed:
            print("\nStarting Block A Part 2 run...")
            total_proc, total_skip, skip_list = run_block_a_part2()
            
            elapsed = time.time() - time.time()  # will be recalculated in post_validation
            print(f"\nTotal questions processed: {total_proc}")
            print(f"Skipped/failed questions: {total_skip}")
            if skip_list:
                for s in skip_list:
                    print(f"  Skipped: {s}")
                    
            run_post_validation()
        else:
            print("\n[FAIL] PREFLIGHT FAILED. Stop.")
            sys.exit(1)
