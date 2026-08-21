import os
import json
import numpy as np
import pandas as pd
import argparse
from pathlib import Path

def compute_theoretical_alpha(logs):
    """
    Computes alpha* = tau^2 / (sigma^2 + tau^2) based on the FR logs.
    tau^2 = variance of log(P_prior) across the dataset.
    sigma^2 = variance of the LLM Judge V_raw scores (or their uncalibrated variance).
    """
    priors = []
    vs = []
    
    for row in logs:
        cands = row.get("candidates", [])
        for c in cands:
            priors.append(c.get("log_prior", 0.0))
            vs.append(c.get("V_raw", 0.0))
            
    if not priors:
        return 0.5
        
    tau_sq = np.var(priors)
    sigma_sq = np.var(vs)
    
    if (sigma_sq + tau_sq) == 0:
        return 0.5
        
    alpha_star = tau_sq / (sigma_sq + tau_sq)
    return alpha_star, sigma_sq, tau_sq

def sweep_empirical_alpha(logs, alphas=np.linspace(0.0, 1.0, 11)):
    """
    Computes empirical accuracy over a grid of alpha values by reranking.
    """
    accs = {}
    for a in alphas:
        correct = 0
        total = 0
        for row in logs:
            cands = row.get("candidates", [])
            if not cands:
                continue
            
            best_cand = None
            best_score = -float('inf')
            
            for c in cands:
                # S(A) = alpha * V_raw + (1 - alpha) * log_prior
                score = a * c.get("V_raw", 0.0) + (1.0 - a) * c.get("log_prior", 0.0)
                if score > best_score:
                    best_score = score
                    best_cand = c
            
            if best_cand and best_cand.get("answer") == row.get("gold"):
                correct += 1
            total += 1
            
        accs[round(a, 2)] = correct / max(total, 1)
        
    return accs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, default="qwen2.5:1.5b,qwen2.5:3b,llama3.2:3b,qwen2.5:7b,qwen2.5:70b")
    args = parser.parse_args()
    
    models = args.models.split(",")
    base_dir = Path(__file__).resolve().parent.parent / "results"
    
    results = []
    
    for m in models:
        m_safe = m.replace(":", "_")
        all_logs = []
        for path in base_dir.rglob(f"{m_safe}_*_frugal_reason_v3.jsonl"):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        all_logs.append(json.loads(line))
        
        if not all_logs:
            print(f"[{m}] No logs found.")
            continue
            
        a_star, sig2, tau2 = compute_theoretical_alpha(all_logs)
        emp_accs = sweep_empirical_alpha(all_logs)
        
        best_emp_alpha = max(emp_accs.items(), key=lambda x: x[1])[0]
        
        print(f"\n--- {m} ---")
        print(f"σ²_V = {sig2:.4f}, τ² = {tau2:.4f} -> α*_theory = {a_star:.4f}")
        print(f"α*_empirical = {best_emp_alpha:.2f} (acc = {emp_accs[best_emp_alpha]:.4f})")
        
        results.append({
            "model": m,
            "sigma_sq": sig2,
            "tau_sq": tau2,
            "alpha_theory": a_star,
            "alpha_emp": best_emp_alpha,
            "acc_grid": emp_accs
        })
        
    # Save to CSV
    if results:
        df = pd.DataFrame(results)
        out_path = Path(__file__).resolve().parent.parent / "results" / "alpha_grid_summary.csv"
        df.to_csv(out_path, index=False)
        print(f"\nSaved summary to {out_path}")

if __name__ == "__main__":
    main()
