import json
import pandas as pd
from pathlib import Path

def generate_pilot_table():
    log_dir = Path("results/pilot_logs")
    
    records = []
    for strat in ["greedy_cot", "self_consistency_k5", "frugal_reason_v3"]:
        file_path = log_dir / f"{strat}_pilot.jsonl"
        if not file_path.exists():
            continue
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                    if "metadata" in d: continue
                    records.append(d)
                except:
                    pass
                    
    df = pd.DataFrame(records)
    if df.empty:
        print("No pilot logs found.")
        return
        
    # Calculate metrics grouped by dataset and strategy
    df["strat"] = df["strategy"]
    
    # Check what columns we have. We should have 'correct', 'parse_success', 'task', 'strategy'
    # Actually 'strategy' is missing for the rows! The rows have task, question_id, correct, etc.
    # We must infer strategy from the file name. Let's fix the parsing loop.
    pass

def generate_pilot_table():
    log_dir = Path("results/pilot_logs")
    
    records = []
    strategies = ["greedy_cot", "self_consistency_k5", "frugal_reason_v3"]
    for strat in strategies:
        file_path = log_dir / f"{strat}_pilot.jsonl"
        if not file_path.exists(): continue
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                    if "metadata" in d: continue
                    d["strategy"] = strat
                    records.append(d)
                except: pass
                
    df = pd.DataFrame(records)
    if df.empty:
        print("No data.")
        return
        
    datasets = df["task"].unique()
    
    table_lines = [
        "| Dataset | Greedy Acc | FR_v3 Acc | Diff (FR - Greedy) | SC_k5 Acc | Parse Rate (Min) | Pass Gate? |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
    ]
    
    results = []
    
    for ds in datasets:
        ds_df = df[df["task"] == ds]
        
        accs = {}
        parse_rates = {}
        for strat in strategies:
            s_df = ds_df[ds_df["strategy"] == strat]
            if s_df.empty:
                accs[strat] = 0.0
                parse_rates[strat] = 0.0
                continue
            accs[strat] = s_df["correct"].mean()
            parse_rates[strat] = s_df["parse_success"].mean()
            
        g_acc = accs.get("greedy_cot", 0)
        fr_acc = accs.get("frugal_reason_v3", 0)
        sc_acc = accs.get("self_consistency_k5", 0)
        diff = fr_acc - g_acc
        min_parse = min(parse_rates.values()) if parse_rates else 0.0
        
        # Criteria
        pass_i = 0.25 <= g_acc <= 0.65
        pass_ii = diff >= 0.10
        pass_iii = min_parse >= 0.90
        
        passed = pass_i and pass_ii and pass_iii
        
        results.append({
            "ds": ds, "g_acc": g_acc, "fr_acc": fr_acc, "sc_acc": sc_acc,
            "diff": diff, "min_parse": min_parse, "passed": passed
        })
        
    # Rank by diff (descending)
    results.sort(key=lambda x: x["diff"], reverse=True)
    
    for r in results:
        pass_str = "YES" if r["passed"] else "NO"
        table_lines.append(f"| {r['ds']} | {r['g_acc']:.2f} | {r['fr_acc']:.2f} | {r['diff']:+.2f} | {r['sc_acc']:.2f} | {r['min_parse']:.2f} | {pass_str} |")
        
    out_md = Path("reports/pilot_selection.md")
    out_md.parent.mkdir(exist_ok=True)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# Pilot Selection Table\n\n")
        f.write("\n".join(table_lines) + "\n")
        
    print("Generated reports/pilot_selection.md")
    print("\n".join(table_lines))

if __name__ == "__main__":
    generate_pilot_table()
