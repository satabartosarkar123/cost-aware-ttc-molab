import yaml
import json
import pandas as pd
from pathlib import Path
from run_frugalreason import evaluate_frugal_reason
from analysis.evaluate_locked import load_raw_jsonl
import shutil

BASE_DIR = Path(__file__).parent
CONFIG_PATH = BASE_DIR / "core" / "config.yaml"
RAW_LOG_PATH = BASE_DIR / "results" / "raw_logs" / "frugal_reason_raw_seed0.jsonl"
ABLATION_DIR = BASE_DIR / "results" / "ablations"
ABLATION_DIR.mkdir(parents=True, exist_ok=True)

def update_config(updates: dict):
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)
    cfg["frugal_reason"].update(updates)
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(cfg, f)

def reset_config():
    update_config({
        "tau_consistency": 0.60,
        "k_probe": 3,
        "k_verify_extra": 3,
        "cluster_sim": 0.85,
        "min_votes": 3,
        "judge_top_k": 2
    })

def run_and_collect(name: str):
    print(f"\n--- Running Ablation: {name} ---")
    evaluate_frugal_reason(seed=0)
    df = load_raw_jsonl(RAW_LOG_PATH)
    shutil.copy(RAW_LOG_PATH, ABLATION_DIR / f"{name}_raw.jsonl")
    
    # summarize
    df['correct'] = df['correct'].fillna(False).astype(bool)
    if 'total_tokens_total' not in df.columns:
        df['total_tokens_total'] = df.get('prompt_tokens_total', 0) + df.get('completion_tokens_total', 0)
    summaries = []
    for task in df['task'].unique():
        tdf = df[df['task'] == task]
        summaries.append({
            "ablation": name,
            "task": task,
            "accuracy": tdf['correct'].fillna(False).astype(bool).mean(),
            "tokens": tdf['total_tokens_total'].mean(),
            "latency": tdf['latency_seconds_total'].mean()
        })
    return pd.DataFrame(summaries)

def run_all_ablations():
    all_summaries = []
    
    # [4.1] Judge-all vs Cluster-then-judge (judge_top_k = 5)
    reset_config()
    all_summaries.append(run_and_collect("FR_ClusterThenJudge_Top2"))
    
    update_config({"judge_top_k": 5})
    all_summaries.append(run_and_collect("FR_JudgeAll_Top5"))
    
    # [4.2] Phase Ablations
    reset_config()
    update_config({"disable_p1": True})
    all_summaries.append(run_and_collect("FR_noP1"))
    
    reset_config()
    update_config({"disable_p2": True})
    all_summaries.append(run_and_collect("FR_noP2"))
    
    reset_config()
    update_config({"disable_p3": True})
    all_summaries.append(run_and_collect("FR_noP3"))
    
    # [4.3] Tau Sweep
    reset_config()
    for tau in [0.5, 0.6, 0.7, 0.8]:
        update_config({"tau_consistency": tau})
        all_summaries.append(run_and_collect(f"FR_Tau_{tau}"))
        
    reset_config()
    
    final_df = pd.concat(all_summaries, ignore_index=True)
    final_df.to_csv(ABLATION_DIR / "ablation_summary.csv", index=False)
    print("Ablation summary saved to results/ablations/ablation_summary.csv")

if __name__ == "__main__":
    run_all_ablations()
