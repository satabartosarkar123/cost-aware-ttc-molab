def execute_cell_61():
    # AO-AlphaGrid-EXT — 5-model α* scaling law (post-hoc, no inference)
    import os, sys, json, math, hashlib, warnings
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path
    from scipy.stats import pearsonr

    warnings.filterwarnings("ignore", category=SyntaxWarning)

    _nb = Path(os.environ.get("NOTEBOOK_DIR", "."))
    cal_dir = _nb / "results" / "calibration"
    cal_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = _nb / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    DATASETS = ["gsm8k", "aqua", "math", "strategyqa"]
    ALPHAS = [round(a * 0.05, 2) for a in range(21)]  # 0.0 to 1.0 step 0.05

    # Model info: (name, param_billions, log_dir, prefix)
    MODELS = [
        ("qwen2.5:1.5b", 1.5, "block_b_logs", "qwen15b_"),
        ("qwen2.5:3b",   3.0, "block_a_logs", ""),
        ("qwen2.5:7b",   7.0, "block_b_logs", "qwen7b_"),
        ("llama3.2:3b",  3.0, "block_b_logs", "llama32_"),
        ("llama3.3:70b", 70.0, "block_b_logs", "llama70b_"),
    ]

    def load_jsonl(path):
        records = []
        if not path.exists(): return records
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip(): records.append(json.loads(line))
        return records

    print("=" * 60)
    print("  Add-On 5 — EXT α* Scaling Law (5 models)")
    print("=" * 60)

    rows = []
    for model_name, params, subdir, prefix in MODELS:
        # Collect all FR candidates across datasets
        all_candidates = []
        for ds in DATASETS:
            log_path = _nb / "results" / subdir / f"{prefix}{ds}_frugal_reason_v3.jsonl"
            if not log_path.exists():
                log_path = _nb / "ttc-frugalreason-poc" / "experiment_fr" / "results" / subdir / f"{prefix}{ds}_frugal_reason_v3.jsonl"
            recs = load_jsonl(log_path)
            for rec in recs:
                cands = rec.get("candidates", [])
                if not cands:
                    try: cands = json.loads(rec.get("raw_response", "{}")).get("candidates", [])
                    except: pass
                if cands:
                    all_candidates.append({"candidates": cands, "gold": rec.get("gold", rec.get("gold_answer", "")),
                                            "dataset": ds})

        if not all_candidates:
            print(f"  {model_name}: NO FR candidates found, skipping.")
            continue

        # α sweep
        best_alpha = 0.6; best_acc = 0.0
        alpha_accs = []
        for alpha in ALPHAS:
            correct = 0; total = len(all_candidates)
            for item in all_candidates:
                cands = item["candidates"]
                if not cands: continue
                best_a = None; best_S = -float("inf")
                for c in cands:
                    V = c.get("V_raw", c.get("V", 0))
                    prior = c.get("prior", 0)
                    S = alpha * V + (1.0 - alpha) * math.log(prior + 1e-6)
                    if S > best_S: best_S = S; best_a = c.get("answer")
                import re
                norm_a = re.sub(r"[,$\\\s]", "", str(best_a or "")).lower().strip()
                norm_g = re.sub(r"[,$\\\s]", "", str(item["gold"] or "")).lower().strip()
                if norm_a == norm_g: correct += 1
            acc = correct / max(total, 1)
            alpha_accs.append(acc)
            if acc > best_acc: best_acc = acc; best_alpha = alpha

        # Theory: σ²_V and τ² from Brier estimators
        V_vals = []; P_vals = []; correct_flags = []
        for item in all_candidates:
            for c in item["candidates"]:
                V = c.get("V_raw", c.get("V", 0))
                prior = c.get("prior", 0)
                V_vals.append(V)
                P_vals.append(prior)
                import re
                norm_a = re.sub(r"[,$\\\s]", "", str(c.get("answer", ""))).lower().strip()
                norm_g = re.sub(r"[,$\\\s]", "", str(item["gold"] or "")).lower().strip()
                correct_flags.append(1.0 if norm_a == norm_g else 0.0)

        if V_vals:
            V_arr = np.array(V_vals); P_arr = np.array(P_vals); C_arr = np.array(correct_flags)
            sigma2_V = np.mean((V_arr - C_arr) ** 2)
            tau2 = np.mean((P_arr - C_arr) ** 2)
            alpha_theory = tau2 / (sigma2_V + tau2) if (sigma2_V + tau2) > 0 else 0.5
        else:
            sigma2_V = tau2 = 0; alpha_theory = 0.5

        acc_at_0 = alpha_accs[0] if alpha_accs else 0
        print(f"  {model_name:18s} | α*_emp={best_alpha:.2f} acc={best_acc:.1%} | α*_theory={alpha_theory:.2f} | acc@α=0={acc_at_0:.1%} | N={len(all_candidates)}")

        rows.append({
            "model": model_name, "params_B": params,
            "alpha_star_emp": best_alpha, "acc_at_alpha_star": best_acc,
            "alpha_star_theory": alpha_theory,
            "sigma2_V": sigma2_V, "tau2": tau2,
            "acc_at_alpha_0": acc_at_0, "n_questions": len(all_candidates),
        })

        # Also save per-alpha curve for this model
        for i, alpha in enumerate(ALPHAS):
            rows.append({
                "model": model_name, "params_B": params,
                "alpha": alpha, "accuracy": alpha_accs[i],
                "alpha_star_emp": best_alpha, "alpha_star_theory": alpha_theory,
            })

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(str(cal_dir / "alpha_grid.csv"), index=False)
        print(f"\nSaved results/calibration/alpha_grid.csv ({len(df)} rows)")

        # ── Fig1: α* vs log(model size) ─────────────────────────────
        summary = df.dropna(subset=["acc_at_alpha_star"]).drop_duplicates("model")
        if len(summary) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            log_sizes = np.log10(summary["params_B"])
            ax.scatter(log_sizes, summary["alpha_star_emp"], s=100, c="blue", label="α*_emp", zorder=5)
            ax.scatter(log_sizes, summary["alpha_star_theory"], s=100, c="red", marker="^", label="α*_theory", zorder=5)
            for _, r in summary.iterrows():
                ax.annotate(r["model"], (np.log10(r["params_B"]), r["alpha_star_emp"]),
                            fontsize=8, textcoords="offset points", xytext=(5, 5))
            ax.set_xlabel("log₁₀(Model Size in Billions)")
            ax.set_ylabel("α*")
            ax.set_title("Fig1: α* Scaling Law — α* vs Model Size")
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.savefig(str(fig_dir / "fig1_scaling_law.png"), dpi=300, bbox_inches="tight")
            fig.savefig(str(fig_dir / "fig1_scaling_law.pdf"), bbox_inches="tight")
            plt.close(fig)
            print("Saved fig1_scaling_law.png + .pdf")

        # ── Fig2: scatter α*_theory vs α*_emp ────────────────────────
        if len(summary) >= 3:
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.scatter(summary["alpha_star_theory"], summary["alpha_star_emp"], s=120, c="green", zorder=5)
            for _, r in summary.iterrows():
                ax.annotate(r["model"], (r["alpha_star_theory"], r["alpha_star_emp"]),
                            fontsize=8, textcoords="offset points", xytext=(5, 5))
            lims = [0, 1]
            ax.plot(lims, lims, "--", c="gray", alpha=0.5, label="y=x (perfect)")
            r_val, p_val = pearsonr(summary["alpha_star_theory"], summary["alpha_star_emp"])
            ax.set_xlabel("α*_theory (τ²/(σ²_V+τ²))")
            ax.set_ylabel("α*_emp (argmax accuracy)")
            ax.set_title(f"Fig2: Theory vs Empirical α* (Pearson r={r_val:.3f}, p={p_val:.4f})")
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.savefig(str(fig_dir / "fig2_theory_vs_emp.png"), dpi=300, bbox_inches="tight")
            fig.savefig(str(fig_dir / "fig2_theory_vs_emp.pdf"), bbox_inches="tight")
            plt.close(fig)
            print(f"Saved fig2_theory_vs_emp.png + .pdf (Pearson r={r_val:.3f})")
        else:
            print("Not enough models for Fig2 (need ≥3)")
    else:
        print("No data to plot!")

    print("\nAdd-On 5 DONE.")

execute_cell_61()