import time
import math
import json
from typing import Dict, Any, List
from core.prompt_manager import get_prompt
from core.parsers import get_parser
from verifiers.verifiers import verify_game24, verify_gsm8k_steps, llm_judge
from verifiers.clustering import cluster_rationales

def frugal_reason_v3_evaluate(client, task: str, question: str, input_metadata: str = "", enable_early_exit: bool = True, alpha: float = 0.6) -> dict:
    """
    Implements FrugalReason v3 (PAV)
    """
    log_data = {
        "early_exit": False,
        "N": 5,
        "clusters": [],
        "candidates": [], 
        "selected_answer": None,
        "alpha_used": alpha,
        "tokens": 0,
        "latency": 0.0,
        "calls": 0,
        "config_hash": "v3_primary",
        "route_used": "none",
        "judge_parse_fails": 0,
        "hidden_judge_scores": {} # for post-hoc judge-all-raw ablation
    }
    
    start_time = time.time()
    parser = get_parser(task)
    
    def _call(prompt, max_t=1024, temp=0.0):
        t0 = time.time()
        resp = client.generate(prompt, max_tokens=max_t, temperature=temp)
        log_data["calls"] += 1
        log_data["tokens"] += resp.get("prompt_tokens", 0) + resp.get("completion_tokens", 0)
        log_data["latency"] += (time.time() - t0)
        return resp.get("text", "")

    try:
        # STEP 0: Optional early exit
        if enable_early_exit:
            prompt_io = get_prompt("greedy_io", task, question)
            prompt_cot = get_prompt("greedy_cot", task, question)
            
            resp_io = _call(prompt_io, temp=0.0)
            resp_cot = _call(prompt_cot, temp=0.0)
            
            parse_io = parser(resp_io)
            parse_cot = parser(resp_cot)
            
            a_io = parse_io["final_answer"]
            a_cot = parse_cot["final_answer"]
            
            if parse_io["parse_success"] and parse_cot["parse_success"] and str(a_io).strip() == str(a_cot).strip() and a_io is not None:
                log_data["early_exit"] = True
                log_data["selected_answer"] = a_io
                log_data["parse_success"] = True
                return log_data

        # STEP 1 & 2: Sample N=5 CoT paths, T=0.7 and Parse
        prompt_cot = get_prompt("greedy_cot", task, question)
        N = 5
        rationales = []
        answers = []
        for _ in range(N):
            r = _call(prompt_cot, temp=0.7)
            a = parser(r)["final_answer"]
            rationales.append(r)
            answers.append(a)
            
        # STEP 3: Semantic Clustering
        cluster_ids = cluster_rationales(rationales, threshold=0.5)
        
        # Group by cluster ID
        cluster_map = {}
        for idx, (cid, r, a) in enumerate(zip(cluster_ids, rationales, answers)):
            if cid not in cluster_map:
                cluster_map[cid] = []
            cluster_map[cid].append({"idx": idx, "rationale": r, "answer": a})
            
        # Determine representative and majority answer for each cluster
        clusters_info = []
        for cid, members in cluster_map.items():
            ans_counts = {}
            for m in members:
                ans_str = str(m["answer"])
                ans_counts[ans_str] = ans_counts.get(ans_str, 0) + 1
            
            majority_answer_str = max(ans_counts.items(), key=lambda x: x[1])[0]
            # Find the actual original answer object (since we converted to str for counting)
            majority_answer = next((m["answer"] for m in members if str(m["answer"]) == majority_answer_str), None)
            
            # Representative = longest rationale
            representative = max(members, key=lambda x: len(x["rationale"]))
            
            clusters_info.append({
                "cluster_id": cid,
                "size": len(members),
                "majority_answer": majority_answer,
                "representative_idx": representative["idx"],
                "representative_rationale": representative["rationale"]
            })
            log_data["clusters"].append({
                "size": len(members),
                "majority_answer": majority_answer,
                "representative_idx": representative["idx"]
            })
            
        # STEP 4: Prior
        # prior(a) = (sum of sizes of clusters whose majority answer is a) / N
        distinct_answers = []
        for c in clusters_info:
            if c["majority_answer"] not in distinct_answers and c["majority_answer"] is not None:
                distinct_answers.append(c["majority_answer"])
                
        priors = {}
        for a in distinct_answers:
            size_sum = sum([c["size"] for c in clusters_info if str(c["majority_answer"]) == str(a)])
            priors[str(a)] = size_sum / float(N)
            
        # Pick the largest cluster for each distinct answer to act as its representative for Verifier
        answer_reps = {}
        for a in distinct_answers:
            clusters_for_a = [c for c in clusters_info if str(c["majority_answer"]) == str(a)]
            largest_cluster = max(clusters_for_a, key=lambda x: x["size"])
            answer_reps[str(a)] = largest_cluster

        # STEP 5: Verifier Routing
        V_scores = {}
        route = "none"
        
        if task == "game24":
            route = "exec"
            for a in distinct_answers:
                rep = answer_reps[str(a)]
                passes = verify_game24(a, input_metadata)
                V_scores[str(a)] = 1.0 if passes else 0.0
                
        elif task == "gsm8k":
            route = "exec"
            any_passed = False
            for a in distinct_answers:
                rep = answer_reps[str(a)]
                v_res = verify_gsm8k_steps(rep["representative_rationale"], a)
                if v_res["all_steps_pass"] and v_res["final_matches"]:
                    V_scores[str(a)] = 1.0
                    any_passed = True
                else:
                    V_scores[str(a)] = 0.0
            
            if not any_passed:
                route = "fallback_judge"
                
        if task not in ["game24", "gsm8k"] or route == "fallback_judge":
            if task not in ["game24", "gsm8k"]:
                route = "judge"
                
            # LLM Judge on top-2 cluster representatives
            sorted_answers = sorted(distinct_answers, key=lambda x: priors[str(x)], reverse=True)
            top_2 = sorted_answers[:2]
            
            for a in top_2:
                rep = answer_reps[str(a)]
                score = llm_judge(client, question, rep["representative_rationale"])
                # Note: llm_judge handles parsing and returns 0.0 to 1.0. If it fails, we want 0.5.
                # Since the existing llm_judge returns 0.0 on fallback, let's wrap it to detect parse failure if needed,
                # but llm_judge already returns 0.0 for fallback. Let's strictly implement 0.5 on fail:
                # We'll just call it and assume the raw judge returns a float. If the implementation returns 0.0 for unknown, we'll patch it below.
                # Let's re-parse here for safety just in case.
                prompt = get_prompt("best_of_n", task="", question=question, candidate=rep["representative_rationale"])
                resp = client.generate(prompt, max_tokens=256, temperature=0.0)
                log_data["calls"] += 1
                log_data["tokens"] += resp.get("prompt_tokens", 0) + resp.get("completion_tokens", 0)
                
                text = resp.get("text", "").lower()
                import re
                score_match = re.search(r'confidence:\s*(\d+)', text)
                if score_match:
                    V_scores[str(a)] = float(score_match.group(1)) / 100.0
                else:
                    raw_nums = re.findall(r'\b(100|[1-9]?[0-9])\b', text)
                    if raw_nums:
                        V_scores[str(a)] = float(raw_nums[-1]) / 100.0
                    else:
                        if "yes" in text or "correct" in text:
                            V_scores[str(a)] = 1.0
                        else:
                            V_scores[str(a)] = 0.5
                            log_data["judge_parse_fails"] += 1

            # For unjudged distinct answers, default V = 0.0
            for a in distinct_answers:
                if str(a) not in V_scores:
                    V_scores[str(a)] = 0.0
                    
            # HIDDEN JUDGE SCORES FOR POST-HOC ANALYSIS (does not count towards model calls/latency)
            for a in distinct_answers:
                if str(a) not in top_2:
                    rep = answer_reps[str(a)]
                    prompt = get_prompt("best_of_n", task="", question=question, candidate=rep["representative_rationale"])
                    hidden_resp = client.generate(prompt, max_tokens=256, temperature=0.0)
                    text = hidden_resp.get("text", "").lower()
                    score_match = re.search(r'confidence:\s*(\d+)', text)
                    if score_match:
                        log_data["hidden_judge_scores"][str(a)] = float(score_match.group(1)) / 100.0
                    else:
                        raw_nums = re.findall(r'\b(100|[1-9]?[0-9])\b', text)
                        if raw_nums:
                            log_data["hidden_judge_scores"][str(a)] = float(raw_nums[-1]) / 100.0
                        elif "yes" in text or "correct" in text:
                            log_data["hidden_judge_scores"][str(a)] = 1.0
                        else:
                            log_data["hidden_judge_scores"][str(a)] = 0.5
                else:
                    log_data["hidden_judge_scores"][str(a)] = V_scores[str(a)]

        log_data["route_used"] = route
        
        # STEP 6: Bayesian-Calibrated Selection
        best_a = None
        best_S = -float('inf')
        
        for a in distinct_answers:
            prior_a = priors[str(a)]
            V_a = V_scores.get(str(a), 0.0)
            S_a = alpha * V_a + (1.0 - alpha) * math.log(prior_a + 1e-6)
            
            log_data["candidates"].append({
                "answer": a,
                "prior": prior_a,
                "V": V_a,
                "S": S_a
            })
            
            # Tie-break: higher prior, then first seen (which is inherent in > vs >=)
            if S_a > best_S:
                best_S = S_a
                best_a = a
            elif abs(S_a - best_S) < 1e-9:
                if prior_a > priors[str(best_a)]:
                    best_a = a
                    best_S = S_a
                    
        if best_a is None and distinct_answers:
            best_a = distinct_answers[0]
            
        log_data["selected_answer"] = best_a
        log_data["parse_success"] = (best_a is not None and str(best_a).strip() != "")
        return log_data

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Strategy Error (v3): {e}")
        log_data["selected_answer"] = log_data.get("a_cot", None)
        log_data["parse_success"] = False
        return log_data
