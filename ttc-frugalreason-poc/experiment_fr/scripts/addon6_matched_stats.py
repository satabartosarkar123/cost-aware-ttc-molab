import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import binomtest
from statsmodels.stats.contingency_tables import mcnemar

def get_outcomes(model, strategy, base_dir, datasets=["math", "gsm8k", "aqua", "strategyqa"]):
    """Returns a dict of qid -> bool (correctness) for a specific run."""
    outcomes = {}
    m_safe = model.replace(":", "_")
    for ds in datasets:
        # Check both mega_sweeps and block_a_logs/block_b_logs
        paths = list(base_dir.rglob(f"{m_safe}_{ds}_{strategy}.jsonl"))
        for p in paths:
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        row = json.loads(line)
                        outcomes[f"{ds}_{row['qid']}"] = row["correct"]
    return outcomes

def wilson_ci(k, n, z=1.96):
    if n == 0: return 0, 0
    p = k / n
    denom = 1 + z**2/n
    center = (p + z**2 / (2*n)) / denom
    spread = z * np.sqrt(p*(1-p)/n + z**2/(4*n**2)) / denom
    return center - spread, center + spread

def main():
    base_dir = Path(__file__).resolve().parent.parent / "results"
    
    # We want to compare 3B FR (small model, frugal reason) against 70B zero-shot (large model)
    baseline_70b = get_outcomes("qwen2.5:70b", "greedy_cot", base_dir)
    fr_3b = get_outcomes("qwen2.5:3b", "frugal_reason_v3", base_dir)
    
    common_qids = set(baseline_70b.keys()).intersection(fr_3b.keys())
    
    if not common_qids:
        print("No intersecting QIDs found between Qwen2.5:70b (CoT) and Qwen2.5:3b (FR-v3). Run sweeps first.")
        return
        
    print(f"Analyzing {len(common_qids)} matched questions.")
    
    # Build contingency table
    # [ [ Both Correct,   70B Correct / 3B Wrong ],
    #   [ 3B Correct / 70B Wrong,  Both Wrong ] ]
    b_corr = 0
    b_70b_only = 0
    b_3b_only = 0
    b_wrong = 0
    
    for qid in common_qids:
        c_70b = baseline_70b[qid]
        c_3b = fr_3b[qid]
        if c_70b and c_3b: b_corr += 1
        elif c_70b and not c_3b: b_70b_only += 1
        elif not c_70b and c_3b: b_3b_only += 1
        else: b_wrong += 1
        
    table = [[b_corr, b_70b_only], [b_3b_only, b_wrong]]
    
    # McNemar's Test
    res = mcnemar(table, exact=False, correction=True)
    
    print("\nContingency Table:")
    print(f"Both Correct: {b_corr}")
    print(f"70B CoT Only Correct: {b_70b_only}")
    print(f"3B FR-v3 Only Correct: {b_3b_only}")
    print(f"Both Wrong: {b_wrong}")
    
    print(f"\nMcNemar's Test: statistic={res.statistic:.4f}, p-value={res.pvalue:.4e}")
    
    acc_70b = (b_corr + b_70b_only) / len(common_qids)
    acc_3b = (b_corr + b_3b_only) / len(common_qids)
    
    ci_70b = wilson_ci(b_corr + b_70b_only, len(common_qids))
    ci_3b = wilson_ci(b_corr + b_3b_only, len(common_qids))
    
    print(f"\n70B CoT Accuracy: {acc_70b:.4f} (95% CI: {ci_70b[0]:.4f} - {ci_70b[1]:.4f})")
    print(f"3B FR-v3 Accuracy: {acc_3b:.4f} (95% CI: {ci_3b[0]:.4f} - {ci_3b[1]:.4f})")
    
if __name__ == "__main__":
    main()
