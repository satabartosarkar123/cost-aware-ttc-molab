def execute_cell_49():
    # D11-Figures — Publication-quality figures (300 dpi PNG + PDF)
    import os, sys, json, math, hashlib, warnings
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from pathlib import Path
    from scipy.stats import norm

    warnings.filterwarnings("ignore", category=SyntaxWarning)

    _nb = Path(os.environ.get("NOTEBOOK_DIR", "."))
    fig_dir = _nb / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ── Helper: save both PNG 300dpi + PDF ──────────────────────────
    def save_fig(fig, name):
        fig.savefig(str(fig_dir / f"{name}.png"), dpi=300, bbox_inches="tight")
        fig.savefig(str(fig_dir / f"{name}.pdf"), bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {name}.png + .pdf")

    # ── Load data ───────────────────────────────────────────────────
    DATASETS = ["gsm8k", "aqua", "math", "strategyqa"]
    STRATEGIES = ["greedy_io", "greedy_cot", "zero_shot_tot_k3",
                  "self_consistency_k5", "best_of_n_k5_self_eval", "frugal_reason_v3"]
    STRAT_LABELS = {"greedy_io": "IO", "greedy_cot": "CoT",
                    "zero_shot_tot_k3": "ToT-3", "self_consistency_k5": "SC-5",
                    "best_of_n_k5_self_eval": "BoN-5", "frugal_reason_v3": "FR-v3"}

    def load_jsonl(path):
        records = []
        if not path.exists():
            return records
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        return records

    def wilson_ci(k, n, z=1.96):
        if n == 0:
            return 0, 0, 0
        p = k / n
        denom = 1 + z**2 / n
        centre = (p + z**2 / (2 * n)) / denom
        margin = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
        return p, max(0, centre - margin), min(1, centre + margin)

    # Load Block A logs (qwen2.5:3b)
    block_a_data = {}
    for ds in DATASETS:
        block_a_data[ds] = {}
        for strat in STRATEGIES:
            # Try multiple possible log locations
            candidates = [
                _nb / "results" / "block_a_logs" / f"{ds}_{strat}.jsonl",
                _nb / "ttc-frugalreason-poc" / "experiment_fr" / "results" / "block_a_logs" / f"{ds}_{strat}.jsonl",
                _nb / "results" / "raw_logs" / f"{ds}_{strat}.jsonl",
            ]
            recs = []
            for cp in candidates:
                recs = load_jsonl(cp)
                if recs:
                    break
            block_a_data[ds][strat] = recs

    print("=" * 60)
    print("  D11 — GENERATING PUBLICATION FIGURES")
    print("=" * 60)

    # ══════════════════════════════════════════════════════════════
    # F1: Main Results Table (4 ds × 6 methods, CI + stars)
    # ══════════════════════════════════════════════════════════════
    print("\nF1: Main Results Table")

    # Load McNemar stars if available
    mcnemar_path = _nb / "results" / "mcnemar_table.csv"
    stars_map = {}
    if mcnemar_path.exists():
        mc_df = pd.read_csv(mcnemar_path)
        for _, row in mc_df.iterrows():
            key = (row.get("dataset", ""), row.get("baseline", ""))
            p = row.get("p_value", 1.0)
            s = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            stars_map[key] = s

    table_rows = []
    for ds in DATASETS:
        row_data = {"Dataset": ds.upper()}
        for strat in STRATEGIES:
            recs = block_a_data[ds][strat]
            n = len(recs)
            k = sum(1 for r in recs if r.get("correct", False))
            acc, lo, hi = wilson_ci(k, n)
            label = STRAT_LABELS[strat]
            star = stars_map.get((ds, strat), "")
            row_data[label] = f"{acc:.1%}\n[{lo:.1%},{hi:.1%}]{star}"
        table_rows.append(row_data)

    if table_rows:
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.axis("off")
        # Build a simple heatmap-style table
        cell_text = []
        col_labels = ["Dataset"] + [STRAT_LABELS[s] for s in STRATEGIES]
        for row in table_rows:
            cell_text.append([row.get(c, "") for c in col_labels])
        table = ax.table(cellText=cell_text, colLabels=col_labels, loc="center",
                         cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.8)
        # Color the FR column
        for i in range(len(table_rows) + 1):
            table[i, len(col_labels) - 1].set_facecolor("#e6f3ff")
        ax.set_title("F1: Main Results — Block A (qwen2.5:3b)", fontsize=14, fontweight="bold", pad=20)
        save_fig(fig, "F1_main_results_table")

    # ══════════════════════════════════════════════════════════════
    # F2: Pareto (acc vs tokens; acc vs calls)
    # ══════════════════════════════════════════════════════════════
    print("\nF2: Pareto Curves")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ds in DATASETS:
        accs = []
        tokens_list = []
        calls_list = []
        labels = []
        for strat in STRATEGIES:
            recs = block_a_data[ds][strat]
            if not recs:
                continue
            n = len(recs)
            k = sum(1 for r in recs if r.get("correct", False))
            acc = k / n if n > 0 else 0
            avg_tok = np.mean([r.get("tokens", r.get("prompt_tokens_total", 0) + r.get("completion_tokens_total", 0)) for r in recs])
            avg_calls = np.mean([r.get("calls", r.get("model_calls", 1)) for r in recs])
            accs.append(acc)
            tokens_list.append(avg_tok)
            calls_list.append(avg_calls)
            labels.append(STRAT_LABELS[strat])

        if accs:
            axes[0].scatter(tokens_list, accs, label=ds, s=50, alpha=0.7)
            axes[1].scatter(calls_list, accs, label=ds, s=50, alpha=0.7)
            # Label FR point
            if len(accs) >= 6:
                axes[0].annotate("FR", (tokens_list[-1], accs[-1]), fontsize=8, fontweight="bold")
                axes[1].annotate("FR", (calls_list[-1], accs[-1]), fontsize=8, fontweight="bold")

    axes[0].set_xlabel("Avg Tokens per Question")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Accuracy vs Token Cost")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Avg Model Calls per Question")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy vs Call Cost")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("F2: Pareto Front — Cost vs Accuracy", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_fig(fig, "F2_pareto_cost_accuracy")

    # ══════════════════════════════════════════════════════════════
    # F3: α* curve (acc(α) per model + markers)
    # ══════════════════════════════════════════════════════════════
    print("\nF3: Alpha Curve")
    alpha_csv = _nb / "results" / "calibration" / "alpha_grid.csv"
    if alpha_csv.exists():
        alpha_df = pd.read_csv(alpha_csv)
        fig, ax = plt.subplots(figsize=(10, 6))
        for model_name in alpha_df["model"].unique() if "model" in alpha_df.columns else []:
            sub = alpha_df[alpha_df["model"] == model_name]
            if "alpha" in sub.columns and "accuracy" in sub.columns:
                ax.plot(sub["alpha"], sub["accuracy"], marker="o", label=model_name, markersize=4)
                best_idx = sub["accuracy"].idxmax()
                ax.axvline(sub.loc[best_idx, "alpha"], linestyle="--", alpha=0.4)
                ax.annotate(f'α*={sub.loc[best_idx, "alpha"]:.2f}',
                            (sub.loc[best_idx, "alpha"], sub.loc[best_idx, "accuracy"]),
                            fontsize=9, fontweight="bold")
        ax.set_xlabel("α (prior weight vs judge weight)")
        ax.set_ylabel("Accuracy")
        ax.set_title("F3: α* Curve — Accuracy vs α per Model")
        ax.legend()
        ax.grid(True, alpha=0.3)
        save_fig(fig, "F3_alpha_curve")
    else:
        print("  SKIP: alpha_grid.csv not found (run D4 first)")

    # ══════════════════════════════════════════════════════════════
    # F4: ECE bars before/after
    # ══════════════════════════════════════════════════════════════
    print("\nF4: ECE Bars")
    ts_path = _nb / "results" / "calibration" / "temp_scaling.json"
    if ts_path.exists():
        with open(ts_path, "r") as f:
            ts_data = json.load(f)
        models = list(ts_data.keys()) if isinstance(ts_data, dict) else []
        if models:
            ece_before = [ts_data[m].get("ece_before", 0) for m in models]
            ece_after = [ts_data[m].get("ece_after", 0) for m in models]
            x = np.arange(len(models))
            width = 0.35
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar(x - width / 2, ece_before, width, label="Before Calibration", color="#ff6b6b")
            ax.bar(x + width / 2, ece_after, width, label="After Calibration", color="#51cf66")
            ax.set_xticks(x)
            ax.set_xticklabels(models, fontsize=9)
            ax.set_ylabel("ECE")
            ax.set_title("F4: Expected Calibration Error — Before vs After Temp Scaling")
            ax.legend()
            ax.grid(True, alpha=0.3, axis="y")
            save_fig(fig, "F4_ece_bars")
        else:
            print("  SKIP: temp_scaling.json has no model entries")
    else:
        print("  SKIP: temp_scaling.json not found (run D6 first)")

    # ══════════════════════════════════════════════════════════════
    # F5: BBH Table
    # ══════════════════════════════════════════════════════════════
    print("\nF5: BBH Table")
    bbh_dir = _nb / "results" / "bbh_logs"
    bbh_rows = []
    if bbh_dir.exists():
        for strat in STRATEGIES:
            fp = bbh_dir / f"bbh_logical_deduction_{strat}.jsonl"
            if not fp.exists():
                fp = bbh_dir / f"bbh_{strat}.jsonl"
            if not fp.exists():
                continue
            recs = load_jsonl(fp)
            n = len(recs)
            k = sum(1 for r in recs if r.get("correct", False))
            acc, lo, hi = wilson_ci(k, n)
            bbh_rows.append({"Strategy": STRAT_LABELS[strat], "Accuracy": f"{acc:.1%}",
                              "CI": f"[{lo:.1%},{hi:.1%}]", "N": n})

    if bbh_rows:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.axis("off")
        col_labels = ["Strategy", "Accuracy", "95% CI", "N"]
        cell_text = [[r["Strategy"], r["Accuracy"], r["CI"], str(r["N"])] for r in bbh_rows]
        table = ax.table(cellText=cell_text, colLabels=col_labels, loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.6)
        ax.set_title("F5: BBH Logical Deduction (qwen2.5:3b)", fontsize=13, fontweight="bold", pad=20)
        save_fig(fig, "F5_bbh_table")
    else:
        print("  SKIP: No BBH logs found (run D9 first)")

    # ══════════════════════════════════════════════════════════════
    # F6: Phase-exit histogram + cost-savings bar
    # ══════════════════════════════════════════════════════════════
    print("\nF6: Phase-Exit Histogram + Cost Savings")

    # Collect early_exit data from FR logs
    exit_counts = {"early_exit": 0, "full_pipeline": 0}
    exit_tokens = {"early_exit": [], "full_pipeline": []}
    for ds in DATASETS:
        recs = block_a_data[ds].get("frugal_reason_v3", [])
        for r in recs:
            early = r.get("early_exit", False)
            tok = r.get("tokens", r.get("prompt_tokens_total", 0) + r.get("completion_tokens_total", 0))
            if early:
                exit_counts["early_exit"] += 1
                exit_tokens["early_exit"].append(tok)
            else:
                exit_counts["full_pipeline"] += 1
                exit_tokens["full_pipeline"].append(tok)

    if exit_counts["early_exit"] + exit_counts["full_pipeline"] > 0:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Left: histogram of exit vs full
        labels = ["Early Exit", "Full Pipeline"]
        counts = [exit_counts["early_exit"], exit_counts["full_pipeline"]]
        colors = ["#51cf66", "#ff6b6b"]
        axes[0].bar(labels, counts, color=colors)
        axes[0].set_ylabel("Number of Questions")
        axes[0].set_title("Phase Exit Distribution")
        for i, v in enumerate(counts):
            axes[0].text(i, v + 1, str(v), ha="center", fontweight="bold")

        # Right: cost savings
        avg_exit = np.mean(exit_tokens["early_exit"]) if exit_tokens["early_exit"] else 0
        avg_full = np.mean(exit_tokens["full_pipeline"]) if exit_tokens["full_pipeline"] else 0
        axes[1].bar(["Early Exit", "Full Pipeline"], [avg_exit, avg_full], color=colors)
        axes[1].set_ylabel("Avg Tokens per Question")
        axes[1].set_title("Token Cost: Early Exit vs Full")
        if avg_full > 0:
            saving = (1 - avg_exit / avg_full) * 100
            axes[1].annotate(f"{saving:.0f}% savings", xy=(0, avg_exit),
                             fontsize=12, fontweight="bold", ha="center",
                             xytext=(0, avg_exit + avg_full * 0.1))

        fig.suptitle("F6: FrugalReason Phase-Exit Analysis", fontsize=14, fontweight="bold")
        plt.tight_layout()
        save_fig(fig, "F6_phase_exit_cost_savings")
    else:
        print("  SKIP: No FR logs with early_exit data found")

    # ── Summary ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    existing = list(fig_dir.glob("*"))
    png_count = sum(1 for f in existing if f.suffix == ".png")
    pdf_count = sum(1 for f in existing if f.suffix == ".pdf")
    print(f"  D11 DONE: {png_count} PNGs + {pdf_count} PDFs in results/figures/")
    print("=" * 60)

execute_cell_49()