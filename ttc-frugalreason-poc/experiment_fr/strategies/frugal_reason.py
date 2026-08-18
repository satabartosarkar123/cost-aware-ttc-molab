import time
import json
from typing import Dict, Any, List
from core.prompt_manager import get_prompt
from core.parsers import get_parser
from verifiers.verifiers import verify_game24, verify_gsm8k_steps, llm_judge
from verifiers.clustering import cluster_weighted_vote

import yaml
from pathlib import Path

# FrugalReason Config
config_path = Path(__file__).parent.parent / "core" / "config.yaml"
with open(config_path, "r") as f:
    _cfg = yaml.safe_load(f)
CONFIG = _cfg["frugal_reason"]

def frugal_reason_evaluate(client, task: str, question: str, input_metadata: str = "") -> dict:
    """
    Implements FrugalReason cascade.
    """
    log_data = {
        "phase_reached": 0,
        "a_io": None,
        "a_cot": None,
        "pool_answers": [],
        "cluster_weights": {},
        "consistency_c": 0.0,
        "verifier_used": None,
        "verifier_passes": 0,
        "judge_scores": {},
        "judge_margin": 0.0,
        "final_answer": None,
        "fallback_used": False,
        "model_calls": 0,
        "prompt_tokens_total": 0,
        "completion_tokens_total": 0,
        "latency_seconds_total": 0.0
    }
    
    start_time = time.time()
    parser = get_parser(task)
    
    def _call(prompt, max_t=1024, temp=0.0):
        t0 = time.time()
        resp = client.generate(prompt, max_tokens=max_t, temperature=temp)
        log_data["model_calls"] += 1
        log_data["prompt_tokens_total"] += resp.get("prompt_tokens", 0)
        log_data["completion_tokens_total"] += resp.get("completion_tokens", 0)
        log_data["latency_seconds_total"] += (time.time() - t0)
        return resp.get("text", "")

    try:
        # ---------------------------------------------------------
        # PHASE 1 - DUAL PROBE (2 calls)
        # ---------------------------------------------------------
        prompt_io = get_prompt("greedy_io", task, question)
        prompt_cot = get_prompt("greedy_cot", task, question)
        
        resp_io = _call(prompt_io, temp=0.0)
        resp_cot = _call(prompt_cot, temp=0.0)
        
        parse_io = parser(resp_io)
        parse_cot = parser(resp_cot)
        
        a_io = parse_io["final_answer"]
        a_cot = parse_cot["final_answer"]
        
        log_data["a_io"] = a_io
        log_data["a_cot"] = a_cot
        log_data["pool_answers"] = [a_cot]
        
        if not CONFIG.get("disable_p1", False) and parse_io["parse_success"] and parse_cot["parse_success"] and str(a_io).strip() == str(a_cot).strip() and a_io is not None:
            log_data["phase_reached"] = 1
            log_data["final_answer"] = a_cot
            return log_data
            
        # ---------------------------------------------------------
        # PHASE 2 - CONSISTENCY AMPLIFICATION (+3 calls)
        # ---------------------------------------------------------
        rationales = [resp_cot]
        answers = [a_cot]
        
        for _ in range(CONFIG["k_probe"]):
            r = _call(prompt_cot, temp=0.7)
            a = parser(r)["final_answer"]
            rationales.append(r)
            answers.append(a)
            log_data["pool_answers"].append(a)
            
        m, c, clusters = cluster_weighted_vote(rationales, answers, CONFIG["cluster_sim"])
        log_data["consistency_c"] = c
        # Store weights and track majority votes
        max_votes = 0
        for cid, members in clusters.items():
            ans = members[0]["answer"]
            votes = len(members)
            log_data["cluster_weights"][str(ans)] = log_data["cluster_weights"].get(str(ans), 0) + votes
            if log_data["cluster_weights"][str(ans)] > max_votes:
                max_votes = log_data["cluster_weights"][str(ans)]
            
        if not CONFIG.get("disable_p2", False) and max_votes >= CONFIG["min_votes"] and c >= CONFIG["tau_consistency"] and m is not None:
            log_data["phase_reached"] = 2
            log_data["final_answer"] = m
            return log_data
            
        # ---------------------------------------------------------
        # PHASE 3 - HETEROGENEOUS VERIFICATION
        # ---------------------------------------------------------
        if CONFIG.get("disable_p3", False):
            # Fallback to majority vote
            log_data["phase_reached"] = 3
            log_data["verifier_used"] = "majority_vote_fallback"
            unique_answers = list(set([a for a in answers if a is not None]))
            if unique_answers:
                unique_answers.sort(key=lambda x: log_data["cluster_weights"].get(str(x), 0), reverse=True)
                log_data["final_answer"] = unique_answers[0]
            return log_data
            
        log_data["phase_reached"] = 3
        if task == "game24":
            log_data["verifier_used"] = "exact_rational"
            passed = []
            for r, a in zip(rationales, answers):
                if verify_game24(a, input_metadata):
                    passed.append(a)
            
            if not passed:
                for _ in range(CONFIG["k_verify_extra"]):
                    r = _call(prompt_cot, temp=0.7)
                    a = parser(r)["final_answer"]
                    if verify_game24(a, input_metadata):
                        passed.append(a)
                        
            log_data["verifier_passes"] = len(passed)
            if passed:
                # highest cluster weight passing
                best_pass = max(passed, key=lambda x: log_data["cluster_weights"].get(str(x), 0))
                log_data["final_answer"] = best_pass
                return log_data
                
        elif task == "gsm8k":
            log_data["verifier_used"] = "step_recompute"
            passed = []
            for r, a in zip(rationales, answers):
                v = verify_gsm8k_steps(r, a)
                if v["all_steps_pass"] and v["final_matches"]:
                    passed.append(a)
            log_data["verifier_passes"] = len(passed)
            if passed:
                best_pass = max(passed, key=lambda x: log_data["cluster_weights"].get(str(x), 0))
                log_data["final_answer"] = best_pass
                return log_data
                
        # LLM Judge Fallback
        log_data["verifier_used"] = "llm_judge"
        unique_answers = list(set([a for a in answers if a is not None]))
        unique_answers.sort(key=lambda x: log_data["cluster_weights"].get(str(x), 0), reverse=True)
        top_k_answers = unique_answers[:CONFIG["judge_top_k"]]
        
        best_score = -1.0
        best_ans = None
        scores = []
        for a in top_k_answers:
            # find first rationale for this answer
            r = next((rat for rat, ans in zip(rationales, answers) if ans == a), "")
            s = llm_judge(client, question, r)
            log_data["judge_scores"][str(a)] = s
            scores.append(s)
            if s > best_score:
                best_score = s
                best_ans = a
                
        if len(scores) == 2:
            log_data["judge_margin"] = abs(scores[0] - scores[1])
            
        # ---------------------------------------------------------
        # PHASE 4 - ESCALATION (Simulated via Judge resolution here, 
        # or bounded ToT. We just return the judge winner for now to 
        # save massive overhead, as per simplified FrugalReason).
        # ---------------------------------------------------------
        if best_ans is not None:
            log_data["final_answer"] = best_ans
        else:
            log_data["fallback_used"] = True
            log_data["final_answer"] = a_cot
            
        return log_data

    except Exception as e:
        print(f"Strategy Error: {e}")
        log_data["fallback_used"] = True
        log_data["final_answer"] = log_data.get("a_cot", None)
        return log_data
