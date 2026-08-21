import os
import json
import numpy as np
import pandas as pd
from pathlib import Path

def expected_calibration_error(y_true, y_prob, n_bins=10):
    bins = np.linspace(0., 1., n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1
    
    ece = 0.0
    for i in range(n_bins):
        bin_mask = (binids == i)
        if not np.any(bin_mask):
            continue
        
        bin_acc = np.mean(y_true[bin_mask])
        bin_conf = np.mean(y_prob[bin_mask])
        bin_weight = np.sum(bin_mask) / len(y_prob)
        
        ece += bin_weight * np.abs(bin_acc - bin_conf)
        
    return ece

def main():
    base_dir = Path(__file__).resolve().parent.parent / "results"
    
    # We load FR logs and look at the V_raw scores vs whether the candidate was actually correct.
    models = ["qwen2.5:1.5b", "qwen2.5:3b", "llama3.2:3b", "qwen2.5:7b", "qwen2.5:70b"]
    
    results = []
    
    for m in models:
        m_safe = m.replace(":", "_")
        paths = list(base_dir.rglob(f"{m_safe}_*_frugal_reason_v3.jsonl"))
        
        y_true = []
        y_prob_raw = []
        
        for p in paths:
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip(): continue
                    row = json.loads(line)
                    gold = row.get("gold")
                    for c in row.get("candidates", []):
                        y_true.append(1 if c.get("answer") == gold else 0)
                        y_prob_raw.append(c.get("V_raw", 0.0))
                        
        if not y_true:
            print(f"No data for {m}")
            continue
            
        y_true = np.array(y_true)
        y_prob_raw = np.array(y_prob_raw)
        
        # Raw ECE
        ece_raw = expected_calibration_error(y_true, y_prob_raw)
        
        # Temp scaled ECE (Assuming T=2.0 as an example derived parameter)
        T = 2.0
        # Since V_raw is often 0 or 1 from the binary ORM judge, we might need a softmax over logits. 
        # But if V_raw is already a probability, temperature scaling on a probability p:
        # q = p^(1/T) / (p^(1/T) + (1-p)^(1/T))
        y_prob_scaled = (y_prob_raw**(1/T)) / (y_prob_raw**(1/T) + (1 - y_prob_raw)**(1/T) + 1e-9)
        # Handle nan/inf
        y_prob_scaled = np.nan_to_num(y_prob_scaled, nan=0.5, posinf=1.0, neginf=0.0)
        
        ece_scaled = expected_calibration_error(y_true, y_prob_scaled)
        
        print(f"[{m}] ECE (Raw): {ece_raw:.4f} | ECE (T={T}): {ece_scaled:.4f}")
        results.append({
            "model": m,
            "ece_raw": ece_raw,
            "ece_scaled": ece_scaled
        })

    if results:
        df = pd.DataFrame(results)
        out = base_dir / "ece_calibration_results.csv"
        df.to_csv(out, index=False)
        print(f"Saved ECE results to {out}")

if __name__ == "__main__":
    main()
