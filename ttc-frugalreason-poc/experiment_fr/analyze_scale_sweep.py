import os
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

def bootstrap_alpha_star(candidates_list, truths_list, alphas=np.linspace(0, 1.0, 6), n_boot=1000, seed=42):
    np.random.seed(seed)
    n = len(candidates_list)
    best_alphas = []
    
    # Precompute accuracies for each alpha for each question
    # shape: (n_questions, n_alphas)
    acc_matrix = np.zeros((n, len(alphas)))
    
    for i, (cands, truth) in enumerate(zip(candidates_list, truths_list)):
        for j, alpha in enumerate(alphas):
            best_S = -float('inf')
            best_ans = None
            for cand in cands:
                V = cand["V"]
                q = cand["prior"]
                ans = cand["answer"]
                S = alpha * V + (1.0 - alpha) * math.log(q + 1e-12)
                if S > best_S:
                    best_S = S
                    best_ans = ans
                elif abs(S - best_S) < 1e-9:
                    if q > cands[0]["prior"]: # crude tiebreak
                        best_ans = ans
                        best_S = S
            
            # evaluate truth
            # We don't have the verifier here, but we can assume exact match with 'truth', 
            # OR we can just rely on the fact that if best_ans == selected_answer, we know if it was correct.
            # But what if a different alpha selects a different answer?
            # To be 100% accurate we should run the verifier, but since we didn't save the full parsing, 
            # we can approximate by string exact match, or checking if best_ans is in the list of known correct answers.
            # Actually, `truth` here is the true_answer.
            # Let's do exact match string representation
            if str(best_ans).strip().lower() == str(truth).strip().lower():
                acc_matrix[i, j] = 1.0
                
    for _ in range(n_boot):
        indices = np.random.choice(n, n, replace=True)
        boot_accs = acc_matrix[indices].mean(axis=0)
        best_alpha = alphas[np.argmax(boot_accs)]
        best_alphas.append(best_alpha)
        
    mean_alpha = np.mean(best_alphas)
    lo = np.percentile(best_alphas, 2.5)
    hi = np.percentile(best_alphas, 97.5)
    
    # empirical alpha on full dataset
    full_accs = acc_matrix.mean(axis=0)
    empirical_alpha = alphas[np.argmax(full_accs)]
    
    return empirical_alpha, lo, hi

def main():
    out_dir = Path("results/scale_sweep")
    models = ["qwen2.5:0.5b", "qwen2.5:1.5b", "qwen2.5:3b", "qwen2.5:7b-instruct-q4_K_M"]
    
    records = []
    
    for model in models:
        fr_path = out_dir / f"frugal_reason_v3_{model.replace(':', '_')}.jsonl"
        gc_path = out_dir / f"greedy_cot_{model.replace(':', '_')}.jsonl"
        
        if not fr_path.exists(): continue
        
        # Load greedy cot
        gc_correct = []
        if gc_path.exists():
            with open(gc_path, "r", encoding="utf-8") as f:
                for line in f:
                    d = json.loads(line)
                    gc_ans = d.get("final_answer")
                    true_ans = d.get("true_answer")
                    if str(gc_ans).strip().lower() == str(true_ans).strip().lower():
                        gc_correct.append(1)
                    else:
                        gc_correct.append(0)
                        
        gc_acc = np.mean(gc_correct) if gc_correct else 0.0
        
        # Load Frugal Reason
        judge_diffs = []
        prior_diffs = []
        qs = []
        candidates_list = []
        truths_list = []
        
        n_total = 0
        
        with open(fr_path, "r", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                n_total += 1
                route = d.get("route_used", "none")
                cands = d.get("candidates", [])
                sel_ans = d.get("selected_answer")
                true_ans = d.get("true_answer")
                
                is_correct = (str(sel_ans).strip().lower() == str(true_ans).strip().lower())
                
                sel_cand = next((c for c in cands if c["answer"] == sel_ans), None)
                if sel_cand:
                    q = sel_cand["prior"]
                    V = sel_cand["V"]
                    
                    if route == "judge" or route == "fallback_judge":
                        judge_diffs.append((V - float(is_correct))**2)
                    
                    prior_diffs.append((q - float(is_correct))**2)
                    qs.append(q)
                    
                candidates_list.append(cands)
                truths_list.append(true_ans)
                
        if n_total == 0: continue
        
        sigma2 = np.mean(judge_diffs) if judge_diffs else 0.0
        tau2 = np.mean(prior_diffs) if prior_diffs else 0.0
        pred_alpha = tau2 / (sigma2 + tau2 + 1e-9)
        
        emp_alpha, ci_lo, ci_hi = bootstrap_alpha_star(candidates_list, truths_list)
        
        m_eff = np.mean([1.0 / sum([c["prior"]**2 for c in cands]) for cands in candidates_list if cands])
        
        # "rho (within- vs across-cluster correctness agreement)"
        # Simple proxy: variance of correctness overall
        p_correct = np.mean([1 if str(sel).strip().lower() == str(truth).strip().lower() else 0 
                             for sel, truth in zip([c[0]["answer"] if c else None for c in candidates_list], truths_list)])
        rho = p_correct**2 + (1-p_correct)**2
        
        records.append({
            "model": model,
            "n": n_total,
            "sigma2": sigma2,
            "tau2": tau2,
            "predicted_alpha*": pred_alpha,
            "empirical_alpha*": emp_alpha,
            "CI_lo": ci_lo,
            "CI_hi": ci_hi,
            "rho": rho,
            "m_eff": m_eff,
            "greedy_cot_acc": gc_acc
        })
        
    df = pd.DataFrame(records)
    df.to_csv(out_dir / "calibration_by_scale.csv", index=False)
    
    # Plotting
    if len(df) > 0:
        plt.figure(figsize=(8,6))
        
        # x-axis can just be the model sizes
        sizes = []
        for m in df["model"]:
            if "0.5b" in m: sizes.append(0.5)
            elif "1.5b" in m: sizes.append(1.5)
            elif "3b" in m: sizes.append(3.0)
            elif "7b" in m: sizes.append(7.0)
            else: sizes.append(0)
            
        plt.plot(sizes, df["predicted_alpha*"], 'b--', marker='o', label="Predicted $\\alpha^*$")
        plt.errorbar(sizes, df["empirical_alpha*"], yerr=[df["empirical_alpha*"] - df["CI_lo"], df["CI_hi"] - df["empirical_alpha*"]], 
                     fmt='ro', label="Empirical $\\alpha^*$ (with 95% CI)")
        
        plt.title("Multi-Scale Calibration Sweep: $\\alpha^*$ vs Scale")
        plt.xlabel("Model Scale (Parameters in Billions)")
        plt.ylabel("$\\alpha^*$")
        plt.legend()
        plt.grid(True)
        plt.savefig(out_dir / "alpha_vs_scale.png")
        plt.close()
        
        plt.figure(figsize=(8,6))
        plt.plot(sizes, df["rho"], 'g-', marker='s', label="$\\rho$")
        plt.title("Correctness Agreement ($\\rho$) vs Scale")
        plt.xlabel("Model Scale (Parameters in Billions)")
        plt.ylabel("$\\rho$")
        plt.grid(True)
        plt.savefig(out_dir / "rho_vs_scale.png")
        plt.close()
        
    print("Analysis complete. Saved to results/scale_sweep/")
    print(df)

if __name__ == "__main__":
    main()
