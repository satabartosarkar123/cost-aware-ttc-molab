def execute_cell_63():
    # AO-70B-Matched-Stats — FR-3B vs 70B baselines, matched 100q sets
    import os, sys, json, math, re, warnings
    import numpy as np
    import pandas as pd
    from pathlib import Path
    from scipy.stats import binom_test

    warnings.filterwarnings("ignore", category=SyntaxWarning)

    _nb = Path(os.environ.get("NOTEBOOK_DIR", "."))
    cal_dir = _nb / "results" / "calibration"
    cal_dir.mkdir(parents=True, exist_ok=True)

    DATASETS = ["gsm8k", "aqua", "math", "strategyqa"]

    def load_jsonl(path):
        records = []
        if not path.exists(): return records
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip(): records.append(json.loads(line))
        return records

    def wilson_ci(k, n, z=1.96):
        if n == 0: return 0, 0, 0
        p = k / n
        denom = 1 + z**2 / n
        centre = (p + z**2 / (2 * n)) / denom
        margin = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
        return p, max(0, centre - margin), min(1, centre + margin)

    print("=" * 60)
    print("  Add-On 6 — Matched FR-3B vs 70B Stats")
    print("=" * 60)

    # FR-3B logs (from Block A)
    # 70B logs: greedy_io, greedy_cot, self_consistency_k5, frugal_reason_v3
    comparisons = ["greedy_io", "greedy_cot", "self_consistency_k5", "frugal_reason_v3"]

    rows = []
    for ds in DATASETS:
        # Load FR-3B
        fr3b_path = _nb / "results" / "block_a_logs" / f"{ds}_frugal_reason_v3.jsonl"
        if not fr3b_path.exists():
            fr3b_path = _nb / "ttc-frugalreason-poc" / "experiment_fr" / "results" / "block_a_logs" / f"{ds}_frugal_reason_v3.jsonl"
        fr3b_recs = load_jsonl(fr3b_path)
        fr3b_by_qid = {r.get("qid", r.get("question_id", "")): r.get("correct", False) for r in fr3b_recs}

        for strat in comparisons:
            # Load 70B
            path_70b = _nb / "results" / "block_b_logs" / f"llama70b_{ds}_{strat}.jsonl"
            recs_70b = load_jsonl(path_70b)
            if not recs_70b:
                print(f"  {ds}/{strat}: no 70B logs found, skipping")
                continue

            r70b_by_qid = {r.get("qid", r.get("question_id", "")): r.get("correct", False) for r in recs_70b}

            # Matched QIDs
            matched = set(fr3b_by_qid.keys()) & set(r70b_by_qid.keys())
            if not matched:
                print(f"  {ds}/{strat}: no matched QIDs, skipping")
                continue

            n = len(matched)
            fr3b_correct = sum(1 for q in matched if fr3b_by_qid[q])
            r70b_correct = sum(1 for q in matched if r70b_by_qid[q])

            acc_fr3b, lo_fr, hi_fr = wilson_ci(fr3b_correct, n)
            acc_70b, lo_70, hi_70 = wilson_ci(r70b_correct, n)
            delta = acc_fr3b - acc_70b

            # McNemar exact: count discordant pairs
            b = sum(1 for q in matched if fr3b_by_qid[q] and not r70b_by_qid[q])  # FR correct, 70B wrong
            c = sum(1 for q in matched if not fr3b_by_qid[q] and r70b_by_qid[q])  # FR wrong, 70B correct
            if b + c > 0:
                try:
                    p_val = binom_test(b, b + c, 0.5)
                except:
                    from scipy.stats import binomtest
                    p_val = binomtest(b, b + c, 0.5).pvalue
            else:
                p_val = 1.0

            stars = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""

            rows.append({
                "dataset": ds, "comparison": f"FR-3B vs 70B-{strat}",
                "n_matched": n,
                "fr3b_acc": acc_fr3b, "fr3b_ci": f"[{lo_fr:.1%},{hi_fr:.1%}]",
                "70b_acc": acc_70b, "70b_ci": f"[{lo_70:.1%},{hi_70:.1%}]",
                "delta": delta, "p_value": p_val, "sig": stars
            })
            print(f"  {ds:12s} | FR-3B vs 70B-{strat:25s} | Δ={delta:+.1%} | p={p_val:.4f}{stars}")

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(str(cal_dir / "matched_70b_stats.csv"), index=False)
        print(f"\nSaved results/calibration/matched_70b_stats.csv ({len(df)} rows)")
        print("\nFull table:")
        print(df.to_string(index=False))
    else:
        print("No matched comparisons could be made (70B logs may not exist yet)")

    print("\nAdd-On 6 DONE.")

execute_cell_63()