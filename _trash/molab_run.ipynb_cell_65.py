def execute_cell_65():
    # AO-Figures-EXT — 8 publication figures, 70B included, OVERWRITES results/figures/
    import os, sys, json, math, hashlib, warnings
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from pathlib import Path

    warnings.filterwarnings("ignore", category=SyntaxWarning)

    _nb = Path(os.environ.get("NOTEBOOK_DIR", "."))
    fig_dir = _nb / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    DATASETS = ["gsm8k", "aqua", "math", "strategyqa"]
    STRATEGIES = ["greedy_io", "greedy_cot", "zero_shot_tot_k3",
                  "self_consistency_k5", "best_of_n_k5_self_eval", "frugal_reason_v3"]
    STRAT_LABELS = {"greedy_io": "IO", "greedy_cot": "CoT",
                    "zero_shot_tot_k3": "ToT-3", "self_consistency_k5": "SC-5",
                    "best_of_n_k5_self_eval": "BoN-5", "frugal_reason_v3": "FR-v3"}
    MODELS_INFO = {
        "qwen2.5:1.5b": ("block_b_logs", "qwen15b_", 1.5),
        "qwen2.5:3b":   ("block_a_logs", "", 3.0),
        "qwen2.5:7b":   ("block_b_logs", "qwen7b_", 7.0),
        "llama3.2:3b":  ("block_b_logs", "llama32_", 3.0),
        "llama3.3:70b": ("block_b_logs", "llama70b_", 70.0),
    }

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

    def save_fig(fig, name):
        fig.savefig(str(fig_dir / f"{name}.png"), dpi=300, bbox_inches="tight")
        fig.savefig(str(fig_dir / f"{name}.pdf"), bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {name}.png + .pdf")

    print("=" * 60)
    print("  Add-On 7 — EXT Figures (8 total, 70B included)")
    print("=" * 60)

    # ── Load ALL data ────────────────────────────────────────────────
    all_data = {}
    for model_name, (subdir, prefix, _) in MODELS_INFO.items():
        all_data[model_name] = {}
        for ds in DATASETS:
            all_data[model_name][ds] = {}
            for strat in STRATEGIES:
                log_path = _nb / "results" / subdir / f"{prefix}{ds}_{strat}.jsonl"
                if not log_path.exists():
                    log_path = _nb / "ttc-frugalreason-poc" / "experiment_fr" / "results" / subdir / f"{prefix}{ds}_{strat}.jsonl"
                all_data[model_name][ds][strat] = load_jsonl(log_path)

    # ── Fig3: ECE bars ───────────────────────────────────────────────
    print("\nFig3: ECE Bars (fit T for 7B & 70B)")
    ts_path = _nb / "results" / "calibration" / "temp_scaling.json"
    ts_data = {}
    if ts_path.exists():
        with open(ts_path) as f:
            ts_data = json.load(f)

    # Fit T for any model that has FR logs but no temp_scaling entry
    from scipy.optimize import minimize_scalar

    for model_name in MODELS_INFO:
        if model_name in ts_data:
            continue
        # Collect (V_raw, correct) pairs from FR logs
        pairs = []
        for ds in DATASETS:
            recs = all_data[model_name][ds].get("frugal_reason_v3", [])
            for r in recs:
                cands = r.get("candidates", [])
                if not cands:
                    try: cands = json.loads(r.get("raw_response", "{}")).get("candidates", [])
                    except: pass
                for c in cands:
                    V = c.get("V_raw", c.get("V", 0))
                    import re
                    norm_a = re.sub(r"[,$\\\s]", "", str(c.get("answer", ""))).lower().strip()
                    norm_g = re.sub(r"[,$\\\s]", "", str(r.get("gold", r.get("gold_answer", "")))).lower().strip()
                    correct = 1.0 if norm_a == norm_g else 0.0
                    pairs.append((V, correct))

        if len(pairs) < 20:
            continue

        # Hash split FIT/EVAL
        V_arr = np.array([p[0] for p in pairs])
        C_arr = np.array([p[1] for p in pairs])
        fit_mask = np.array([i % 2 == 0 for i in range(len(pairs))])

        V_fit, C_fit = V_arr[fit_mask], C_arr[fit_mask]
        V_eval, C_eval = V_arr[~fit_mask], C_arr[~fit_mask]

        def nll(T):
            z = np.clip(V_fit / T, -20, 20)
            p = 1.0 / (1.0 + np.exp(-z))
            p = np.clip(p, 1e-7, 1 - 1e-7)
            return -np.mean(C_fit * np.log(p) + (1 - C_fit) * np.log(1 - p))

        res = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
        T_star = res.x

        def ece(V, C, T=1.0, bins=10):
            z = np.clip(V / T, -20, 20)
            probs = 1.0 / (1.0 + np.exp(-z))
            total = 0
            for b in range(bins):
                lo, hi = b / bins, (b + 1) / bins
                mask = (probs >= lo) & (probs < hi)
                if mask.sum() == 0: continue
                total += mask.sum() * abs(probs[mask].mean() - C[mask].mean())
            return total / len(V)

        ece_before = ece(V_eval, C_eval, T=1.0)
        ece_after = ece(V_eval, C_eval, T=T_star)

        ts_data[model_name] = {"T_star": float(T_star), "ece_before": float(ece_before),
                                "ece_after": float(ece_after), "n_pairs": len(pairs)}
        print(f"  {model_name}: T*={T_star:.3f}, ECE {ece_before:.4f} → {ece_after:.4f}")

    with open(ts_path, "w") as f:
        json.dump(ts_data, f, indent=2)

    if ts_data:
        models_ts = list(ts_data.keys())
        ece_before = [ts_data[m].get("ece_before", 0) for m in models_ts]
        ece_after = [ts_data[m].get("ece_after", 0) for m in models_ts]
        x = np.arange(len(models_ts)); width = 0.35
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(x - width/2, ece_before, width, label="Before", color="#ff6b6b")
        ax.bar(x + width/2, ece_after, width, label="After", color="#51cf66")
        ax.set_xticks(x); ax.set_xticklabels(models_ts, fontsize=8, rotation=30)
        ax.set_ylabel("ECE"); ax.set_title("Fig3: ECE Before/After Temp Scaling (All Models)")
        ax.legend(); ax.grid(True, alpha=0.3, axis="y")
        save_fig(fig, "F3_ece_bars_ext")

    # ── Fig4: Main results table (3B + 70B row + BBH) ───────────────
    print("\nFig4: Main Results Table (3B + 70B + BBH)")
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.axis("off")
    col_labels = ["Model", "Dataset"] + [STRAT_LABELS[s] for s in STRATEGIES]
    cell_text = []
    for model_name in ["qwen2.5:3b", "llama3.3:70b"]:
        for ds in DATASETS:
            row = [model_name, ds.upper()]
            for strat in STRATEGIES:
                recs = all_data.get(model_name, {}).get(ds, {}).get(strat, [])
                n = len(recs); k = sum(1 for r in recs if r.get("correct", False))
                acc, lo, hi = wilson_ci(k, n)
                row.append(f"{acc:.0%}" if n > 0 else "—")
            cell_text.append(row)

    if cell_text:
        table = ax.table(cellText=cell_text, colLabels=col_labels, loc="center", cellLoc="center")
        table.auto_set_font_size(False); table.set_fontsize(8); table.scale(1.1, 1.6)
        ax.set_title("Fig4: Main Results — 3B + 70B", fontsize=14, fontweight="bold", pad=20)
        save_fig(fig, "F4_main_results_ext")

    # ── Fig5: Pareto (all models, acc vs tokens/calls) ──────────────
    print("\nFig5: Pareto (all models)")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    colors = {"qwen2.5:1.5b": "blue", "qwen2.5:3b": "green", "qwen2.5:7b": "orange",
              "llama3.2:3b": "purple", "llama3.3:70b": "red"}
    for model_name in MODELS_INFO:
        for ds in DATASETS:
            for strat in STRATEGIES:
                recs = all_data.get(model_name, {}).get(ds, {}).get(strat, [])
                if not recs: continue
                n = len(recs); k = sum(1 for r in recs if r.get("correct", False))
                acc = k / n if n > 0 else 0
                avg_tok = np.mean([r.get("tokens", r.get("total_tokens", 0)) for r in recs])
                avg_calls = np.mean([r.get("calls", r.get("model_calls", 1)) for r in recs])
                marker = "D" if "frugal" in strat else "o"
                size = 80 if "frugal" in strat else 30
                axes[0].scatter(avg_tok, acc, c=colors.get(model_name, "gray"), s=size, marker=marker, alpha=0.6)
                axes[1].scatter(avg_calls, acc, c=colors.get(model_name, "gray"), s=size, marker=marker, alpha=0.6)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=8, label=m)
                        for m, c in colors.items()]
    legend_elements.append(Line2D([0], [0], marker="D", color="w", markerfacecolor="black", markersize=8, label="FrugalReason"))
    axes[0].legend(handles=legend_elements, fontsize=7)
    axes[0].set_xlabel("Avg Tokens"); axes[0].set_ylabel("Accuracy"); axes[0].set_title("Acc vs Tokens")
    axes[0].grid(True, alpha=0.3)
    axes[1].legend(handles=legend_elements, fontsize=7)
    axes[1].set_xlabel("Avg Calls"); axes[1].set_ylabel("Accuracy"); axes[1].set_title("Acc vs Calls")
    axes[1].grid(True, alpha=0.3)
    fig.suptitle("Fig5: Pareto Front — All Models", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_fig(fig, "F5_pareto_ext")

    # ── Fig6: Ablation bar chart ────────────────────────────────────
    print("\nFig6: Ablation Bar")
    abl_path = _nb / "results" / "ablations" / "ablation_table.csv"
    if abl_path.exists():
        abl_df = pd.read_csv(abl_path)
        if "dataset" in abl_df.columns:
            fig, ax = plt.subplots(figsize=(10, 5))
            abl_df.plot(kind="bar", x="dataset", ax=ax)
            ax.set_title("Fig6: Ablation Results"); ax.set_ylabel("Accuracy"); ax.grid(True, alpha=0.3, axis="y")
            save_fig(fig, "F6_ablation_bar")
    else:
        print("  SKIP: ablation_table.csv not found")

    # ── Fig7: BBH table ─────────────────────────────────────────────
    print("\nFig7: BBH Table")
    bbh_dir = _nb / "results" / "bbh_logs"
    bbh_rows = []
    if bbh_dir.exists():
        for strat in STRATEGIES:
            for pat in [f"bbh_logical_deduction_{strat}.jsonl", f"bbh_{strat}.jsonl"]:
                fp = bbh_dir / pat
                recs = load_jsonl(fp)
                if recs:
                    n = len(recs); k = sum(1 for r in recs if r.get("correct", False))
                    acc, lo, hi = wilson_ci(k, n)
                    bbh_rows.append([STRAT_LABELS[strat], f"{acc:.1%}", f"[{lo:.1%},{hi:.1%}]", str(n)])
                    break
    if bbh_rows:
        fig, ax = plt.subplots(figsize=(8, 4)); ax.axis("off")
        table = ax.table(cellText=bbh_rows, colLabels=["Strategy", "Acc", "95% CI", "N"],
                         loc="center", cellLoc="center")
        table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1.2, 1.6)
        ax.set_title("Fig7: BBH Logical Deduction", fontsize=13, fontweight="bold", pad=20)
        save_fig(fig, "F7_bbh_table")

    # ── Fig8: Phase-exit histogram ──────────────────────────────────
    print("\nFig8: Phase-Exit Histogram")
    exit_counts = {"early_exit": 0, "full_pipeline": 0}
    exit_tokens = {"early_exit": [], "full_pipeline": []}
    for model_name in MODELS_INFO:
        for ds in DATASETS:
            recs = all_data.get(model_name, {}).get(ds, {}).get("frugal_reason_v3", [])
            for r in recs:
                early = r.get("early_exit", False)
                tok = r.get("tokens", r.get("total_tokens", 0))
                if early: exit_counts["early_exit"] += 1; exit_tokens["early_exit"].append(tok)
                else: exit_counts["full_pipeline"] += 1; exit_tokens["full_pipeline"].append(tok)

    if exit_counts["early_exit"] + exit_counts["full_pipeline"] > 0:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        labels = ["Early Exit", "Full Pipeline"]
        counts = [exit_counts["early_exit"], exit_counts["full_pipeline"]]
        colors = ["#51cf66", "#ff6b6b"]
        axes[0].bar(labels, counts, color=colors)
        axes[0].set_ylabel("Count"); axes[0].set_title("Phase Exit Distribution (All Models)")
        for i, v in enumerate(counts): axes[0].text(i, v + 1, str(v), ha="center", fontweight="bold")

        avg_e = np.mean(exit_tokens["early_exit"]) if exit_tokens["early_exit"] else 0
        avg_f = np.mean(exit_tokens["full_pipeline"]) if exit_tokens["full_pipeline"] else 0
        axes[1].bar(labels, [avg_e, avg_f], color=colors)
        axes[1].set_ylabel("Avg Tokens"); axes[1].set_title("Token Cost: Exit vs Full")
        if avg_f > 0:
            saving = (1 - avg_e / avg_f) * 100
            axes[1].annotate(f"{saving:.0f}% savings", xy=(0, avg_e), fontsize=11, fontweight="bold", ha="center")
        fig.suptitle("Fig8: Phase-Exit Analysis (All Models)", fontsize=14, fontweight="bold")
        plt.tight_layout()
        save_fig(fig, "F8_phase_exit_ext")

    # Summary
    existing = list(fig_dir.glob("*"))
    print(f"\nAdd-On 7 DONE: {sum(1 for f in existing if f.suffix=='.png')} PNGs + {sum(1 for f in existing if f.suffix=='.pdf')} PDFs")

execute_cell_65()