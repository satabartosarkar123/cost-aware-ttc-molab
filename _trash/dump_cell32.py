# Day4-AlphaGrid — Post-hoc α sweep, NO new inference

_nb = Path(os.environ.get("NOTEBOOK_DIR", "."))

# Define models and their log directories
MODELS = {
    "qwen2.5:3b": ("block_a_logs", ""),           # Block A logs (no prefix)
    "qwen2.5:1.5b": ("block_b_logs", "qwen15b_"),
    "llama3.2:3b": ("block_b_logs", "llama32_"),
}
DATASETS = ["gsm8k", "aqua", "math", "strategyqa"]
ALPHAS = np.arange(0.0, 1.05, 0.05)

cal_dir = _nb / "results" / "calibration"
cal_dir.mkdir(parents=True, exist_ok=True)

results_rows = []
alpha_curves = []

for model_name, (log_subdir, prefix) in MODELS.items():
    print(f"\n{'='*60}")
    print(f"  Model: {model_name}")
    print(f"{'='*60}")

    all_questions = []  # list of dicts with candidates + gold

    for ds in DATASETS:
        fr_path = _nb / log_subdir / f"{prefix}{ds}_frugal_reason_v3.jsonl"
        if not fr_path.exists():
            fr_path = _nb / "results" / log_subdir / f"{prefix}{ds}_frugal_reason_v3.jsonl"
        if not fr_path.exists():
            # Try experiment_fr path for Block A
            fr_path = _nb / "ttc-frugalreason-poc" / "experiment_fr" / "results" / log_subdir / f"{prefix}{ds}_frugal_reason_v3.jsonl"
        if not fr_path.exists():
            fr_path = _nb / f"block_b_{model_name.split(':')[0].replace('.', '')}" / "block_b_logs" / f"{prefix}{ds}_frugal_reason_v3.jsonl"
        if not fr_path.exists():
            # specific overrides since qwen2.5:1.5b -> block_b_qwen15b
            if "qwen2.5:1.5b" in model_name:
                fr_path = _nb / "block_b_qwen15b" / "block_b_logs" / f"{prefix}{ds}_frugal_reason_v3.jsonl"
            elif "llama3.2:3b" in model_name:
                fr_path = _nb / "block_b_llama32" / "block_b_logs" / f"{prefix}{ds}_frugal_reason_v3.jsonl"
        
        if not fr_path.exists():
            print(f"  SKIP {ds}: FR log not found")
            continue

        with open(fr_path, encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                rec = json.loads(line)
                cands = rec.get("candidates", [])
                if not cands:
                    # Try parsing from raw_response if it was JSON-encoded
                    try:
                        raw = json.loads(rec.get("raw_response", "{}"))
                        cands = raw.get("candidates", [])
                    except: pass
                if cands:
                    all_questions.append({
                        "dataset": ds, "qid": rec.get("qid"),
                        "gold": rec.get("gold", rec.get("gold_answer")),
                        "selected": rec.get("selected_answer"),
                        "correct": rec.get("correct", False),
                        "candidates": cands
                    })

    if not all_questions:
        print(f"  No candidate data found for {model_name}")
        continue

    # Get base CoT accuracy from greedy_cot logs
    base_cot_correct = 0; base_cot_total = 0
    for ds in DATASETS:
        log_dir = _nb / "results" / log_subdir
        cot_path = log_dir / f"{prefix}{ds}_greedy_cot.jsonl"
        if not cot_path.exists():
            cot_path = _nb / "ttc-frugalreason-poc" / "experiment_fr" / "results" / log_subdir / f"{prefix}{ds}_greedy_cot.jsonl"
        if cot_path.exists():
            with open(cot_path, encoding="utf-8") as f:
                for line in f:
                    if not line.strip(): continue
                    rec = json.loads(line)
                    base_cot_total += 1
                    if rec.get("correct", False): base_cot_correct += 1

    base_cot_acc = base_cot_correct / max(base_cot_total, 1)

    # Sweep α
    best_alpha = 0.6; best_acc = 0.0
    for alpha in ALPHAS:
        correct = 0
        for q in all_questions:
            cands = q["candidates"]
            best_a = None; best_S = -float("inf")
            for c in cands:
                V = c.get("V_raw", c.get("V", 0.0))
                prior = c.get("prior", 0.0)
                S = alpha * V + (1.0 - alpha) * math.log(prior + 1e-6)
                if S > best_S: best_S = S; best_a = c.get("answer")
            gold = q["gold"]
            # Normalize for comparison
            import re
            p = re.sub(r"[,\$\s]", "", str(best_a or "")).lower().strip()
            g = re.sub(r"[,\$\s]", "", str(gold or "")).lower().strip()
            if p == g: correct += 1
        acc = correct / len(all_questions)
        alpha_curves.append({"model": model_name, "alpha": round(float(alpha), 2), "accuracy": acc})
        if acc > best_acc: best_acc = acc; best_alpha = round(float(alpha), 2)

    # α=0 (SC-like)
    acc_alpha0 = next((r["accuracy"] for r in alpha_curves
                       if r["model"] == model_name and r["alpha"] == 0.0), 0.0)

    results_rows.append({
        "model": model_name, "base_cot_acc": base_cot_acc,
        "alpha_star_emp": best_alpha, "acc_at_alpha_star": best_acc,
        "acc_at_alpha_0": acc_alpha0, "n_questions": len(all_questions)
    })
    print(f"  α*_emp = {best_alpha} | acc@α* = {best_acc:.3f} | acc@α=0 = {acc_alpha0:.3f} | base_cot = {base_cot_acc:.3f}")

# Save CSV
results_df = pd.DataFrame(results_rows)
results_df.to_csv(str(cal_dir / "alpha_grid.csv"), index=False)
print("\nSaved results/calibration/alpha_grid.csv")
print(results_df.to_string(index=False))

# Plot α curves
curves_df = pd.DataFrame(alpha_curves)
fig, ax = plt.subplots(figsize=(10, 6))
for model_name in curves_df["model"].unique():
    subset = curves_df[curves_df["model"] == model_name]
    ax.plot(subset["alpha"], subset["accuracy"], marker="o", label=model_name, markersize=3)
ax.set_xlabel("α"); ax.set_ylabel("Accuracy"); ax.set_title("α Sweep: Accuracy vs α")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(str(cal_dir / "alpha_curve.png"), dpi=150)
plt.show()
print("Saved results/calibration/alpha_curve.png")

# ── HF CONTINUOUS SYNC ──
try:
    import os
    from pathlib import Path
    from huggingface_hub import HfApi
    _nb = Path(os.environ.get("NOTEBOOK_DIR", "."))
    _res = _nb / "results"
    if _res.exists():
        _api = HfApi(token="REDACTED")
        _api.upload_folder(
            folder_path=str(_res),
            path_in_repo="results_sync",
            repo_id="Satabarto/Molab_Checkpoints_Cost_AWARE",
            repo_type="dataset",
        )
        print("\n" + "="*50)
        print("  HF DATA SYNCED SUCCESSFULLY")
        print("="*50)
    else:
        print("No results dir yet - skipping HF sync.")
except Exception as e:
    print(f"HF sync warning (non-fatal): {e}")
