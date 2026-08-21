import json
import os

notebook_path = "molab_run.ipynb"
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

markdown_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# Day 1 Master Order — Statistics on Saved Logs\n",
        "**CRITICAL:** Do NOT start Ollama, do NOT run any strategy, do NOT generate any model output.\n",
        "Every number must be computed EXCLUSIVELY from the already-saved Block A logs in `results/block_a_logs/*.jsonl`."
    ]
}

code_source = '''import os
import json
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats import binomtest

DATASETS = ["gsm8k", "aqua", "math", "strategyqa"]
STRATEGIES = [
    "greedy_io", "greedy_cot", "zero_shot_tot_k3",
    "self_consistency_k5", "best_of_n_k5_self_eval", "frugal_reason_v3"
]
EXPECTED_COUNTS = {"gsm8k": 300, "aqua": 254, "math": 238, "strategyqa": 300}
LOG_DIR = "results/block_a_logs"

print("SECTION 1 — LOAD & VALIDATE COMPLETENESS")
data = []
gaps = []

for d in DATASETS:
    for s in STRATEGIES:
        filepath = os.path.join(LOG_DIR, f"{d}_{s}.jsonl")
        if not os.path.exists(filepath):
            gaps.append(f"Missing file: {d} - {s}")
            continue
        
        count = 0
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                record = json.loads(line)
                
                qid = record.get("qid", f"{d}_{count}")
                correct = int(record.get("correct", False))
                
                tokens = record.get("tokens", 0)
                calls = record.get("calls", 1)
                parse = int(record.get("parse_success", True))
                
                data.append({
                    "dataset": d,
                    "strategy": s,
                    "qid": qid,
                    "correct": correct,
                    "tokens": tokens,
                    "calls": calls,
                    "parse": parse
                })
                count += 1
        
        if count != EXPECTED_COUNTS[d]:
            gaps.append(f"Mismatch {d}-{s}: expected {EXPECTED_COUNTS[d]}, got {count}")

df = pd.DataFrame(data)

print(f"Loaded {len(df)} records.")
if gaps:
    print("GAPS FOUND:")
    for g in gaps:
        print(" -", g)
else:
    print("No gaps found. All counts match expected!")

print("\\nSECTION 2 — WILSON 95% CONFIDENCE INTERVALS")
def wilson_ci(k, n, z=1.96):
    if n == 0: return 0.0, 0.0
    p = k / n
    denominator = 1 + z**2/n
    centre_adjusted_probability = p + z**2 / (2*n)
    adjusted_standard_deviation = np.sqrt((p*(1 - p) + z**2 / (4*n)) / n)
    
    lower_bound = (centre_adjusted_probability - z*adjusted_standard_deviation) / denominator
    upper_bound = (centre_adjusted_probability + z*adjusted_standard_deviation) / denominator
    return lower_bound, upper_bound

stats_rows = []
for d in DATASETS:
    for s in STRATEGIES:
        subset = df[(df["dataset"] == d) & (df["strategy"] == s)]
        n = len(subset)
        if n == 0: continue
        k = subset["correct"].sum()
        acc = k / n
        lo, hi = wilson_ci(k, n)
        
        avg_tokens = subset["tokens"].mean()
        avg_calls = subset["calls"].mean()
        parse_rate = subset["parse"].mean()
        
        stats_rows.append({
            "dataset": d,
            "strategy": s,
            "correct": k,
            "total": n,
            "acc": acc,
            "wilson_lo": lo,
            "wilson_hi": hi,
            "avg_tokens": avg_tokens,
            "avg_calls": avg_calls,
            "parse_rate": parse_rate
        })
        print(f"{d} | {s}: {acc*100:.1f}% [{lo*100:.1f}, {hi*100:.1f}]")

stats_df = pd.DataFrame(stats_rows)
os.makedirs("results", exist_ok=True)
stats_df.to_csv("results/block_a_final_stats.csv", index=False)

print("\\nSECTION 3 — McNEMAR EXACT TESTS (accuracy, paired by qid)")
mcnemar_rows = []
for d in DATASETS:
    fr_subset = df[(df["dataset"] == d) & (df["strategy"] == "frugal_reason_v3")].set_index("qid")
    if len(fr_subset) == 0: continue
    
    for s in STRATEGIES:
        if s == "frugal_reason_v3": continue
        bs_subset = df[(df["dataset"] == d) & (df["strategy"] == s)].set_index("qid")
        
        aligned = fr_subset.join(bs_subset, lsuffix="_fr", rsuffix="_bs", how="inner")
        
        b = ((aligned["correct_fr"] == 1) & (aligned["correct_bs"] == 0)).sum()
        c = ((aligned["correct_fr"] == 0) & (aligned["correct_bs"] == 1)).sum()
        
        if b + c == 0:
            p_val = 1.0
        else:
            p_val = binomtest(b, n=b+c, p=0.5, alternative='two-sided').pvalue
            
        stars = ""
        if p_val < 0.001: stars = "***"
        elif p_val < 0.01: stars = "**"
        elif p_val < 0.05: stars = "*"
        
        acc_diff = aligned["correct_fr"].mean() - aligned["correct_bs"].mean()
        mcnemar_rows.append({
            "dataset": d,
            "baseline": s,
            "b_fr_only": b,
            "c_bs_only": c,
            "p_value": p_val,
            "significance": stars,
            "acc_diff": acc_diff
        })

mcnemar_df = pd.DataFrame(mcnemar_rows)
mcnemar_df.to_csv("results/mcnemar_table.csv", index=False)
print(mcnemar_df[["dataset", "baseline", "b_fr_only", "c_bs_only", "p_value", "significance", "acc_diff"]])

print("\\nSECTION 4 — PAIRED BOOTSTRAP (10,000 resamples, seed=0)")
np.random.seed(0)
bootstrap_rows = []
for d in DATASETS:
    fr_subset = df[(df["dataset"] == d) & (df["strategy"] == "frugal_reason_v3")].set_index("qid")
    if len(fr_subset) == 0: continue
    
    for s in STRATEGIES:
        if s == "frugal_reason_v3": continue
        bs_subset = df[(df["dataset"] == d) & (df["strategy"] == s)].set_index("qid")
        aligned = fr_subset.join(bs_subset, lsuffix="_fr", rsuffix="_bs", how="inner")
        
        n = len(aligned)
        if n == 0: continue
        diffs = aligned["correct_fr"].values - aligned["correct_bs"].values
        
        resamples = np.random.choice(diffs, (10000, n), replace=True)
        means = resamples.mean(axis=1)
        lo = np.percentile(means, 2.5)
        hi = np.percentile(means, 97.5)
        
        bootstrap_rows.append({
            "dataset": d,
            "baseline": s,
            "metric": "accuracy",
            "mean_diff": diffs.mean(),
            "ci_lo": lo,
            "ci_hi": hi
        })
        
        if s in ["self_consistency_k5", "best_of_n_k5_self_eval"]:
            tok_diffs = aligned["tokens_fr"].values - aligned["tokens_bs"].values
            resamples_tok = np.random.choice(tok_diffs, (10000, n), replace=True)
            means_tok = resamples_tok.mean(axis=1)
            bootstrap_rows.append({
                "dataset": d,
                "baseline": s,
                "metric": "tokens",
                "mean_diff": tok_diffs.mean(),
                "ci_lo": np.percentile(means_tok, 2.5),
                "ci_hi": np.percentile(means_tok, 97.5)
            })
            
            call_diffs = aligned["calls_fr"].values - aligned["calls_bs"].values
            resamples_call = np.random.choice(call_diffs, (10000, n), replace=True)
            means_call = resamples_call.mean(axis=1)
            bootstrap_rows.append({
                "dataset": d,
                "baseline": s,
                "metric": "calls",
                "mean_diff": call_diffs.mean(),
                "ci_lo": np.percentile(means_call, 2.5),
                "ci_hi": np.percentile(means_call, 97.5)
            })

boot_df = pd.DataFrame(bootstrap_rows)
boot_df.to_csv("results/bootstrap_table.csv", index=False)
print("Bootstrap finished. Wrote results/bootstrap_table.csv")

print("\\nSECTION 5 — SANITY SPOT-CHECK (anti-fabrication)")
expected = {
    "gsm8k": 0.820,
    "aqua": 0.709,
    "math": 0.735,
    "strategyqa": 0.653
}
for d, exp_val in expected.items():
    val = stats_df[(stats_df["dataset"] == d) & (stats_df["strategy"] == "frugal_reason_v3")]["acc"].values
    if len(val) > 0:
        actual = val[0]
        if abs(actual - exp_val) > 0.005:
            print(f"STOP! Deviation found in {d}: Expected ~{exp_val}, Got {actual:.3f}")
        else:
            print(f"Spot-check passed for {d}: {actual:.3f} matches ~{exp_val}")

print("\\nSECTION 6 — OUTPUTS / MARKDOWN SUMMARY")
for d in DATASETS:
    print(f"### {d.upper()}")
    fr_row = stats_df[(stats_df["dataset"] == d) & (stats_df["strategy"] == "frugal_reason_v3")]
    if len(fr_row) == 0: continue
    fr_acc = fr_row["acc"].values[0]
    fr_lo, fr_hi = fr_row["wilson_lo"].values[0], fr_row["wilson_hi"].values[0]
    print(f"FRUGAL_REASON_V3: {fr_acc*100:.1f}% [{fr_lo*100:.1f}, {fr_hi*100:.1f}]")
    
    for s in STRATEGIES:
        if s == "frugal_reason_v3": continue
        s_row = stats_df[(stats_df["dataset"] == d) & (stats_df["strategy"] == s)]
        if len(s_row) == 0: continue
        s_acc = s_row["acc"].values[0]
        s_lo, s_hi = s_row["wilson_lo"].values[0], s_row["wilson_hi"].values[0]
        
        mc_row = mcnemar_df[(mcnemar_df["dataset"] == d) & (mcnemar_df["baseline"] == s)]
        stars = mc_row["significance"].values[0] if len(mc_row)>0 else ""
        
        bt_row = boot_df[(boot_df["dataset"] == d) & (boot_df["baseline"] == s) & (boot_df["metric"] == "accuracy")]
        if len(bt_row) > 0:
            diff = bt_row["mean_diff"].values[0] * 100
            diff_lo = bt_row["ci_lo"].values[0] * 100
            diff_hi = bt_row["ci_hi"].values[0] * 100
            diff_str = f"diff {diff:+.1f} CI[{diff_lo:+.1f}, {diff_hi:+.1f}]"
        else:
            diff_str = ""
            
        print(f" - {s}: {s_acc*100:.1f}% [{s_lo*100:.1f}, {s_hi*100:.1f}] {stars} | {diff_str}")
    print()
'''

code_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [line + "\n" for line in code_source.split("\n")]
}

# Fix last newline
if code_cell["source"]:
    code_cell["source"][-1] = code_cell["source"][-1].rstrip("\n")

nb["cells"].append(markdown_cell)
nb["cells"].append(code_cell)

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
print("Updated Notebook successfully.")
