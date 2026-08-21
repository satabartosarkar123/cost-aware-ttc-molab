import os
import pandas as pd
from pathlib import Path
import json

def main():
    base_dir = Path(__file__).resolve().parent.parent / "results"
    
    # We will gather all models and compile the main results table.
    # Metrics: Accuracy across GSM8K, AQUA, MATH, StrategyQA for IO, CoT, SC, BoN, ToT, FR-v3
    
    models = ["qwen2.5:1.5b", "qwen2.5:3b", "llama3.2:3b", "qwen2.5:7b", "qwen2.5:70b"]
    datasets = ["gsm8k", "aqua", "math", "strategyqa"]
    strategies = ["greedy_io", "greedy_cot", "self_consistency_k5", "best_of_n_k5_self_eval", "zero_shot_tot_k3", "frugal_reason_v3"]
    
    data = []
    
    for m in models:
        m_safe = m.replace(":", "_")
        for ds in datasets:
            row = {"Model": m, "Dataset": ds}
            for strat in strategies:
                paths = list(base_dir.rglob(f"{m_safe}_{ds}_{strat}.jsonl"))
                correct = 0
                total = 0
                for p in paths:
                    with open(p, "r", encoding="utf-8") as f:
                        for line in f:
                            if not line.strip(): continue
                            log = json.loads(line)
                            if log["correct"]: correct += 1
                            total += 1
                
                acc = (correct / total) if total > 0 else None
                row[strat] = acc
                row[f"{strat}_n"] = total
            
            data.append(row)
            
    if data:
        df = pd.DataFrame(data)
        out_csv = base_dir / "final_results_table.csv"
        df.to_csv(out_csv, index=False)
        print(f"Aggregated final results into {out_csv}")
        
        # Optionally dump a LaTeX table
        out_tex = base_dir / "final_results_table.tex"
        with open(out_tex, "w") as f:
            f.write(df.to_latex(index=False))
        print(f"Generated LaTeX table at {out_tex}")
        
if __name__ == "__main__":
    main()
