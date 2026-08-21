import json, os, glob
import pandas as pd
from pathlib import Path

log_dir = Path(r"c:\Users\USER\Cost-Aware-Test-Time\Cost-Aware-Test-time-molab\block_b_llama32\block_b_logs")

datasets = ["gsm8k", "aqua", "math", "strategyqa"]
strategies = ["greedy_io", "greedy_cot", "zero_shot_tot_k3", "self_consistency_k5", "best_of_n_k5_self_eval", "frugal_reason_v3"]

rows = []

for ds in datasets:
    for strat in strategies:
        file_path = log_dir / f"llama32_{ds}_{strat}.jsonl"
        if not file_path.exists():
            print(f"Missing: {file_path.name}")
            continue
            
        correct = 0
        total = 0
        
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                rec = json.loads(line)
                total += 1
                if rec.get("correct", False):
                    correct += 1
                    
        acc = correct / total if total > 0 else 0
        rows.append({
            "Dataset": ds,
            "Strategy": strat,
            "Accuracy": acc,
            "Correct": correct,
            "Total": total
        })

df = pd.DataFrame(rows)
if not df.empty:
    pivot = df.pivot_table(values="Accuracy", index="Dataset", columns="Strategy", aggfunc="first")
    
    # Reorder columns to match canonical order
    cols = [c for c in strategies if c in pivot.columns]
    pivot = pivot[cols]
    
    print("\n============================================================")
    print("   LLAMA 3.2 3B (DAY 3 SWEEP) RESULTS")
    print("============================================================\n")
    print(pivot.to_string(float_format=lambda x: f"{x:.1%}"))
    
else:
    print("No data found!")
