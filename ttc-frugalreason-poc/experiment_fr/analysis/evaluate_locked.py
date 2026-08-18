import json
import math
import pandas as pd
from pathlib import Path
import numpy as np
from scipy.stats import binom, norm

BASE_DIR = Path(__file__).parent.parent
RAW_LOG_PATH = BASE_DIR / "results" / "raw_logs" / "frugal_reason_v3_raw_seed0.jsonl"
BASELINE_PATH = BASE_DIR.parent.parent / "rq2_part1" / "results" / "cost_profile.csv"
SUMMARY_DIR = BASE_DIR / "results" / "summary"
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

def wilson_score_interval(successes, n, confidence=0.95):
    if n == 0: return 0.0, 0.0
    z = norm.ppf(1 - (1 - confidence) / 2)
    p = successes / n
    denominator = 1 + z**2 / n
    centre_adjusted_probability = p + z**2 / (2 * n)
    adjusted_standard_deviation = math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)
    lower_bound = (centre_adjusted_probability - z * adjusted_standard_deviation) / denominator
    upper_bound = (centre_adjusted_probability + z * adjusted_standard_deviation) / denominator
    return max(0, lower_bound), min(1, upper_bound)

def mcnemar_exact(b, c):
    return binom.test(int(b), int(b + c), 0.5).pvalue if (b + c) > 0 else 1.0

def paired_bootstrap(diffs, n_resamples=10000, seed=42):
    np.random.seed(seed)
    diffs = np.array(diffs)
    if len(diffs) == 0: return 0.0, 0.0
    resamples = np.random.choice(diffs, size=(n_resamples, len(diffs)), replace=True)
    means = np.mean(resamples, axis=1)
    return np.percentile(means, 2.5), np.percentile(means, 97.5)

def load_raw_jsonl(path: Path) -> pd.DataFrame:
    records = []
    if not path.exists(): return pd.DataFrame()
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            records.append(json.loads(line))
    return pd.DataFrame(records)

def evaluate_locked():
    print("Evaluating FrugalReason purely from raw JSONL logs...")
    df_fr = load_raw_jsonl(RAW_LOG_PATH)
    if df_fr.empty:
        print("No FR data found.")
        return
        
    df_fr['correct'] = df_fr['correct'].fillna(False).astype(bool)
    if 'total_tokens_total' not in df_fr.columns:
        df_fr['total_tokens_total'] = df_fr.get('prompt_tokens_total', 0) + df_fr.get('completion_tokens_total', 0)
    
    # 1. Basic Stats with Wilson CI
    summary_data = []
    for task in df_fr['task'].unique():
        task_df = df_fr[df_fr['task'] == task]
        correct = task_df['correct'].sum()
        total = len(task_df)
        acc = correct / total if total > 0 else 0
        ci_low, ci_high = wilson_score_interval(correct, total)
        
        print(f"[{task}] Accuracy: {correct}/{total} ({acc*100:.1f}%) [95% CI: {ci_low*100:.1f}% - {ci_high*100:.1f}%]")
        
        # 2. Route/Phase-conditional table
        if 'phase_reached' in task_df.columns:
            for p in [1, 2, 3]:
                pdf = task_df[task_df['phase_reached'] == p]
                if len(pdf) > 0:
                    p_corr = pdf['correct'].sum()
                    print(f"  Phase {p}: {len(pdf)} items, Acc: {p_corr}/{len(pdf)}, Avg Tok: {pdf['total_tokens_total'].mean():.1f}")
        elif 'route_used' in task_df.columns:
            for r in task_df['route_used'].unique():
                pdf = task_df[task_df['route_used'] == r]
                if len(pdf) > 0:
                    p_corr = pdf['correct'].sum()
                    print(f"  Route {r}: {len(pdf)} items, Acc: {p_corr}/{len(pdf)}, Avg Tok: {pdf['total_tokens_total'].mean():.1f}")
                
        summary_data.append({
            "task": task,
            "strategy": "frugal_reason",
            "accuracy": acc,
            "acc_ci_low": ci_low,
            "acc_ci_high": ci_high,
            "avg_tokens": task_df['total_tokens_total'].mean(),
            "avg_latency": task_df['latency_seconds_total'].mean(),
            "avg_calls": task_df['model_calls'].mean()
        })
        
    # 3. Paired Tests against Baselines
    if BASELINE_PATH.exists():
        df_base = pd.read_csv(BASELINE_PATH)
        print("\n--- Paired Statistical Tests vs Baselines ---")
        for task in df_fr['task'].unique():
            fr_t = df_fr[df_fr['task'] == task].set_index('question_id')
            b_t = df_base[df_base['task'] == task]
            
            for strat in b_t['strategy'].unique():
                bs_s = b_t[b_t['strategy'] == strat].set_index('local_question_id')
                
                # Join on index
                joined = fr_t.join(bs_s, how='inner', lsuffix='_fr', rsuffix='_base')
                if joined.empty: continue
                
                b = ((joined['correct_fr'] == True) & (joined['correct_base'] == False)).sum()
                c = ((joined['correct_fr'] == False) & (joined['correct_base'] == True)).sum()
                
                p_val = mcnemar_exact(b, c)
                
                tok_diffs = joined['total_tokens_total_fr'] - joined['total_tokens_total_base']
                lat_diffs = joined['latency_seconds_total_fr'] - joined['latency_seconds_total_base']
                
                tok_ci_low, tok_ci_high = paired_bootstrap(tok_diffs.dropna())
                lat_ci_low, lat_ci_high = paired_bootstrap(lat_diffs.dropna())
                
                print(f"[{task}] FR vs {strat}: McNemar p={p_val:.4f} | Tok Diff 95% CI: [{tok_ci_low:.1f}, {tok_ci_high:.1f}] | Lat Diff CI: [{lat_ci_low:.1f}, {lat_ci_high:.1f}]")
                
    pd.DataFrame(summary_data).to_csv(SUMMARY_DIR / "locked_evaluation_summary.csv", index=False)

if __name__ == "__main__":
    evaluate_locked()
