import json
import pandas as pd
from pathlib import Path

def generate_report():
    in_dir = Path("results/strict_eval")
    if not in_dir.exists():
        print(f"Directory {in_dir} not found.")
        return

    data = []
    
    for file_path in in_dir.glob("*.jsonl"):
        strategy = file_path.stem
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                try:
                    d = json.loads(line)
                    
                    # Token extraction
                    tokens = d.get("tokens", 0)
                    if tokens == 0:
                        tokens = d.get("prompt_tokens_total", 0) + d.get("completion_tokens_total", 0)
                        
                    # Calls extraction
                    calls = d.get("calls", d.get("model_calls", 0))
                    
                    # Latency extraction
                    latency = d.get("latency", d.get("latency_seconds_total", d.get("wall_seconds", 0.0)))
                    
                    # Parse success extraction
                    ans = d.get("selected_answer", d.get("final_answer", None))
                    parse_success = d.get("parse_success", ans is not None and str(ans).strip() != "")
                    
                    data.append({
                        "Strategy": strategy,
                        "Dataset": d.get("task", ""),
                        "QuestionID": d.get("question_id", ""),
                        "Correct": 1 if d.get("correct", False) else 0,
                        "Tokens": tokens,
                        "Latency": latency,
                        "Calls": calls,
                        "ParseSuccess": 1 if parse_success else 0
                    })
                except Exception as e:
                    pass
                    
    if not data:
        print("No data found.")
        return
        
    df = pd.DataFrame(data)
    
    # 1. Main Table: Accuracy, Avg Tokens, Avg Latency, Avg Calls
    print("================================================================")
    print("FINAL STRICT EVALUATION REPORT (qwen2.5:3b)")
    print("================================================================")
    
    metrics = df.groupby(["Dataset", "Strategy"]).agg({
        "Correct": ["mean", "count"],
        "Tokens": "mean",
        "Latency": "mean",
        "Calls": "mean",
        "ParseSuccess": "mean"
    }).reset_index()
    
    metrics.columns = ["Dataset", "Strategy", "Accuracy", "Count", "AvgTokens", "AvgLatency", "AvgCalls", "ParseRate"]
    metrics["Accuracy"] = metrics["Accuracy"] * 100
    metrics["ParseRate"] = metrics["ParseRate"] * 100
    
    print("\n--- METRICS BY DATASET AND STRATEGY ---")
    for dataset in sorted(metrics["Dataset"].unique()):
        print(f"\nDataset: {dataset.upper()}")
        d_df = metrics[metrics["Dataset"] == dataset].copy()
        
        # Sort strategies in display order
        strat_order = {"greedy_cot": 1, "self_consistency_k5": 2, "best_of_n_k5_self_eval": 3, "frugal_reason_v3": 4}
        d_df["order"] = d_df["Strategy"].map(strat_order)
        d_df = d_df.sort_values("order").drop("order", axis=1)
        
        for _, row in d_df.iterrows():
            print(f"  [{row['Strategy']:<25}] Acc: {row['Accuracy']:5.1f}% | Tokens: {row['AvgTokens']:6.0f} | Latency: {row['AvgLatency']:5.1f}s | Calls: {row['AvgCalls']:4.1f} | Parse: {row['ParseRate']:5.1f}% (n={row['Count']})")
            
    print("\n================================================================")
    print("GATE CHECK (Frugal vs Greedy)")
    print("================================================================")
    
    for dataset in sorted(metrics["Dataset"].unique()):
        d_df = metrics[metrics["Dataset"] == dataset]
        
        greedy_row = d_df[d_df["Strategy"] == "greedy_cot"]
        frugal_row = d_df[d_df["Strategy"] == "frugal_reason_v3"]
        
        g_acc = greedy_row["Accuracy"].values[0] if not greedy_row.empty else 0
        f_acc = frugal_row["Accuracy"].values[0] if not frugal_row.empty else 0
        diff = f_acc - g_acc
        f_parse = frugal_row["ParseRate"].values[0] if not frugal_row.empty else 0
        
        print(f"{dataset.upper():<10} -> Greedy Acc: {g_acc:5.1f}% | Frugal Acc: {f_acc:5.1f}% | Diff: {diff:+5.1f}% | Frugal Parse Rate: {f_parse:5.1f}%")

if __name__ == "__main__":
    generate_report()
