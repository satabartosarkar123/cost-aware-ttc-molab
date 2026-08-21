def execute_cell_67():
    # AO-Final-Report — Final aggregation, VRAM cleanup, push
    import os, sys, json, time, math, re, warnings, subprocess
    import numpy as np
    import pandas as pd
    from pathlib import Path

    warnings.filterwarnings("ignore", category=SyntaxWarning)

    _nb = Path(os.environ.get("NOTEBOOK_DIR", "."))

    # ── 1. FINAL KICK-OUT: unload all add-on models ─────────────────
    def _ollama_unload(model_name):
        import subprocess, time, requests as _req
        r = subprocess.run(f"ollama stop {model_name}", shell=True, capture_output=True, text=True)
        if r.returncode == 0:
            print(f"  Stopped {model_name}")
        else:
            try:
                _req.post("http://localhost:11434/api/generate",
                           json={"model": model_name, "prompt": "", "keep_alive": 0}, timeout=30)
            except: pass

    print("=" * 70)
    print("  FINAL VRAM CLEANUP")
    print("=" * 70)
    for m in ["qwen2.5:7b", "llama3.3:70b", "qwen2.5:72b"]:
        _ollama_unload(m)
        time.sleep(2)

    ps = subprocess.run("ollama ps", shell=True, capture_output=True, text=True)
    print(f"\nollama ps:\n{ps.stdout}")
    subprocess.run("nvidia-smi --query-gpu=memory.free --format=csv,noheader", shell=True)

    # ── 2. UPDATED MASTER TABLE (5 models × 6 strats × 4 ds) ────────
    print("\n" + "=" * 70)
    print("  UPDATED MASTER TABLE")
    print("=" * 70)

    DATASETS = ["gsm8k", "aqua", "math", "strategyqa"]
    STRATEGIES = ["greedy_io", "greedy_cot", "zero_shot_tot_k3",
                  "self_consistency_k5", "best_of_n_k5_self_eval", "frugal_reason_v3"]
    STRAT_LABELS = {"greedy_io": "IO", "greedy_cot": "CoT",
                    "zero_shot_tot_k3": "ToT", "self_consistency_k5": "SC",
                    "best_of_n_k5_self_eval": "BoN", "frugal_reason_v3": "FR"}
    MODELS_INFO = {
        "qwen2.5:1.5b": ("block_b_logs", "qwen15b_"),
        "qwen2.5:3b":   ("block_a_logs", ""),
        "qwen2.5:7b":   ("block_b_logs", "qwen7b_"),
        "llama3.2:3b":  ("block_b_logs", "llama32_"),
        "llama3.3:70b": ("block_b_logs", "llama70b_"),
    }

    def load_jsonl(path):
        records = []
        if not path.exists(): return records
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip(): records.append(json.loads(line))
        return records

    total_runs = 0
    master_rows = []
    for model_name, (subdir, prefix) in MODELS_INFO.items():
        for ds in DATASETS:
            for strat in STRATEGIES:
                log_path = _nb / "results" / subdir / f"{prefix}{ds}_{strat}.jsonl"
                if not log_path.exists():
                    log_path = _nb / "ttc-frugalreason-poc" / "experiment_fr" / "results" / subdir / f"{prefix}{ds}_{strat}.jsonl"
                recs = load_jsonl(log_path)
                if not recs: continue
                n = len(recs); k = sum(1 for r in recs if r.get("correct", False))
                total_runs += n
                master_rows.append({"model": model_name, "dataset": ds, "strategy": strat,
                                    "correct": k, "total": n, "accuracy": k/n if n>0 else 0})

    if master_rows:
        master_df = pd.DataFrame(master_rows)
        pivot = master_df.pivot_table(values="accuracy", index=["model", "dataset"],
                                      columns="strategy", aggfunc="first")
        cols = [c for c in STRATEGIES if c in pivot.columns]
        pivot = pivot[cols]
        pivot.columns = [STRAT_LABELS.get(c, c) for c in cols]
        print(pivot.to_string(float_format=lambda x: f"{x:.1%}"))
        master_df.to_csv(str(_nb / "results" / "addon_master_table.csv"), index=False)

    # ── 3. α TABLE ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  α TABLE")
    print("=" * 70)
    alpha_path = _nb / "results" / "calibration" / "alpha_grid.csv"
    if alpha_path.exists():
        adf = pd.read_csv(alpha_path)
        summary = adf.dropna(subset=["acc_at_alpha_star"]).drop_duplicates("model")
        if not summary.empty:
            print(summary[["model", "params_B", "alpha_star_emp", "acc_at_alpha_star",
                            "alpha_star_theory", "acc_at_alpha_0"]].to_string(index=False))

    # ── 4. MATCHED 70B TABLE ────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  MATCHED FR-3B vs 70B TABLE")
    print("=" * 70)
    m70_path = _nb / "results" / "calibration" / "matched_70b_stats.csv"
    if m70_path.exists():
        print(pd.read_csv(m70_path).to_string(index=False))

    # ── 5. ECE TABLE ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  ECE TABLE")
    print("=" * 70)
    ts_path = _nb / "results" / "calibration" / "temp_scaling.json"
    if ts_path.exists():
        ts = json.load(open(ts_path))
        for m, v in ts.items():
            print(f"  {m:18s} | T*={v.get('T_star',0):.3f} | ECE: {v.get('ece_before',0):.4f} → {v.get('ece_after',0):.4f}")

    # ── 6. 72B CROSS TABLE ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  72B CROSS-VALIDATION TABLE")
    print("=" * 70)
    for strat in ["greedy_io", "greedy_cot", "frugal_reason_v4"]:
        fp = _nb / "results" / "block_b_logs" / f"qwen72b_math_{strat}.jsonl"
        recs = load_jsonl(fp)
        if recs:
            n = len(recs); k = sum(1 for r in recs if r.get("correct", False))
            print(f"  qwen2.5:72b MATH {strat}: {k}/{n} ({k/n:.1%})")

    # ── 7. FINAL PUSH TO HF ─────────────────────────────────────────
    try:
        from huggingface_hub import HfApi
        import zipfile
        _api = HfApi(token="REDACTED")
        final_zip = str(_nb / "results" / "addon_final_all_results.zip")
        with zipfile.ZipFile(final_zip, "w", zipfile.ZIP_DEFLATED) as zf:
            results_root = _nb / "results"
            for fp in results_root.rglob("*"):
                if fp.is_file() and "__pycache__" not in str(fp):
                    zf.write(str(fp), str(fp.relative_to(results_root)))
        _api.upload_file(path_or_fileobj=final_zip,
                         path_in_repo="results_sync/addon_final_all_results.zip",
                         repo_id="Satabarto/Molab_Checkpoints_Cost_AWARE",
                         repo_type="dataset")
        print("\nPushed addon_final_all_results.zip to HF.")
    except Exception as e:
        print(f"HF push failed: {e}")

    # ── FINAL LINE ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"  ADD-ON COMPLETE — {total_runs} total runs logged; VRAM freed.")
    print(f"  STOP. Writing phase (D13+) consumes ONLY the overwritten artifacts.")
    print("=" * 70)

execute_cell_67()