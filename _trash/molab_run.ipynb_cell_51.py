def execute_cell_51():
    # D12-Buffer-Report — Final sanity, spot-check, aggregate, cleanup
    import os, sys, json, time, math, random, re, hashlib, warnings
    import numpy as np
    import pandas as pd
    from pathlib import Path

    warnings.filterwarnings("ignore", category=SyntaxWarning)

    _nb = Path(os.environ.get("NOTEBOOK_DIR", "."))
    SEED = 0
    random.seed(SEED)

    print("=" * 70)
    print("  D12 — BUFFER / SANITY / FINAL REPORT")
    print("=" * 70)

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

    # ── 1. SPOT-CHECK 20 random rows ────────────────────────────
    print("\n(1) Spot-check: recompute `correct` from selected_answer vs gold")
    print("-" * 70)
    all_records = []
    for ds in DATASETS:
        for strat in STRATEGIES:
            candidates = [
                _nb / "results" / "block_a_logs" / f"{ds}_{strat}.jsonl",
                _nb / "ttc-frugalreason-poc" / "experiment_fr" / "results" / "block_a_logs" / f"{ds}_{strat}.jsonl",
                _nb / "results" / "raw_logs" / f"{ds}_{strat}.jsonl",
            ]
            for cp in candidates:
                recs = load_jsonl(cp)
                if recs:
                    for r in recs:
                        r["_ds"] = ds
                        r["_strat"] = strat
                    all_records.extend(recs)
                    break

    if len(all_records) >= 20:
        sample = random.sample(all_records, 20)
    else:
        sample = all_records

    spot_pass = 0
    spot_fail = 0
    for rec in sample:
        ans = str(rec.get("selected_answer", "")).strip()
        gold = str(rec.get("gold", rec.get("gold_answer", ""))).strip()
        logged_correct = rec.get("correct", False)

        # Normalize: strip $, commas, whitespace, lowercase
        norm_ans = re.sub(r"[,$\\\s]", "", ans).lower().strip().rstrip(".")
        norm_gold = re.sub(r"[,$\\\s]", "", gold).lower().strip().rstrip(".")

        recomputed = (norm_ans == norm_gold)

        if recomputed == logged_correct:
            spot_pass += 1
        else:
            spot_fail += 1
            print(f"  MISMATCH: {rec['_ds']}/{rec['_strat']} qid={rec.get('question_id','?')} "
                  f"ans={ans!r} gold={gold!r} logged={logged_correct} recomputed={recomputed}")

    print(f"  Spot-check: {spot_pass}/20 PASS, {spot_fail}/20 FAIL")
    if spot_fail > 0:
        print("  WARNING: Some mismatches detected — review verifier logic.")
    else:
        print("  All 20 spot-checks PASSED.")

    # ── 2. COMPLETENESS MATRIX ──────────────────────────────────
    print("\n(2) Completeness Matrix")
    print("-" * 70)
    MODELS_INFO = {
        "qwen2.5:3b": ("block_a_logs", ""),
        "qwen2.5:1.5b": ("block_b_logs", "qwen15b_"),
        "llama3.2:3b": ("block_b_logs", "llama32_"),
    }

    total_runs = 0
    for model_name, (subdir, prefix) in MODELS_INFO.items():
        print(f"\n  {model_name}:")
        for ds in DATASETS:
            counts = []
            for strat in STRATEGIES:
                log_path = _nb / "results" / subdir / f"{prefix}{ds}_{strat}.jsonl"
                if not log_path.exists():
                    log_path = _nb / "ttc-frugalreason-poc" / "experiment_fr" / "results" / subdir / f"{prefix}{ds}_{strat}.jsonl"
                recs = load_jsonl(log_path)
                counts.append(len(recs))
                total_runs += len(recs)
            print(f"    {ds:12s}: {' | '.join(f'{c:3d}' for c in counts)}")

    # ── 3. BLOCK B MASTER TABLE ─────────────────────────────────
    print("\n(3) Block B Master Table (3 models × 6 strategies × 4 datasets)")
    print("-" * 70)
    master_rows = []
    for model_name, (subdir, prefix) in MODELS_INFO.items():
        for ds in DATASETS:
            for strat in STRATEGIES:
                log_path = _nb / "results" / subdir / f"{prefix}{ds}_{strat}.jsonl"
                if not log_path.exists():
                    log_path = _nb / "ttc-frugalreason-poc" / "experiment_fr" / "results" / subdir / f"{prefix}{ds}_{strat}.jsonl"
                recs = load_jsonl(log_path)
                if not recs:
                    continue
                n = len(recs)
                k = sum(1 for r in recs if r.get("correct", False))
                acc = k / n if n > 0 else 0
                master_rows.append({"model": model_name, "dataset": ds, "strategy": strat,
                                    "correct": k, "total": n, "accuracy": acc})

    if master_rows:
        master_df = pd.DataFrame(master_rows)
        pivot = master_df.pivot_table(values="accuracy", index=["model", "dataset"],
                                      columns="strategy", aggfunc="first")
        cols = [c for c in STRATEGIES if c in pivot.columns]
        pivot = pivot[cols]
        pivot.columns = [STRAT_LABELS.get(c, c) for c in cols]
        print(pivot.to_string(float_format=lambda x: f"{x:.1%}"))
        master_df.to_csv(str(_nb / "results" / "block_b_master_table.csv"), index=False)

    # ── 4. α TABLE ──────────────────────────────────────────────
    print("\n(4) Alpha Table")
    print("-" * 70)
    alpha_path = _nb / "results" / "calibration" / "alpha_grid.csv"
    if alpha_path.exists():
        print(pd.read_csv(alpha_path).to_string(index=False))
    else:
        print("  Not yet available (run D4)")

    # ── 5. ECE TABLE ────────────────────────────────────────────
    print("\n(5) ECE Table")
    print("-" * 70)
    ts_path = _nb / "results" / "calibration" / "temp_scaling.json"
    if ts_path.exists():
        print(json.dumps(json.load(open(ts_path)), indent=2))
    else:
        print("  Not yet available (run D6)")

    # ── 6. BBH TABLE ────────────────────────────────────────────
    print("\n(6) BBH Table")
    print("-" * 70)
    bbh_dir = _nb / "results" / "bbh_logs"
    if bbh_dir.exists() and any(bbh_dir.glob("*.jsonl")):
        for fp in sorted(bbh_dir.glob("*.jsonl")):
            recs = load_jsonl(fp)
            n = len(recs)
            k = sum(1 for r in recs if r.get("correct", False))
            acc = k / n if n > 0 else 0
            print(f"  {fp.stem}: {acc:.1%} ({k}/{n})")
    else:
        print("  Not yet available (run D9)")

    # ── 7. ABLATION TABLE ──────────────────────────────────────
    print("\n(7) Ablation Table")
    print("-" * 70)
    abl_path = _nb / "results" / "ablations" / "ablation_table.csv"
    if abl_path.exists():
        print(pd.read_csv(abl_path).to_string(index=False))
    else:
        print("  Not yet available (run D10)")

    # ── 8. FINAL PUSH TO HF ────────────────────────────────────
    print("\n(8) Final push to Hugging Face")
    print("-" * 70)
    try:
        from huggingface_hub import HfApi
        import zipfile
        _api = HfApi(token="REDACTED")
        final_zip = str(_nb / "results" / "d12_final_all_results.zip")
        with zipfile.ZipFile(final_zip, "w", zipfile.ZIP_DEFLATED) as zf:
            results_root = _nb / "results"
            for fp in results_root.rglob("*"):
                if fp.is_file() and "__pycache__" not in str(fp):
                    zf.write(str(fp), str(fp.relative_to(results_root)))
        _api.upload_file(path_or_fileobj=final_zip,
                         path_in_repo="results_sync/d12_final_all_results.zip",
                         repo_id="Satabarto/Molab_Checkpoints_Cost_AWARE",
                         repo_type="dataset")
        print("  Pushed d12_final_all_results.zip to HF.")
    except Exception as e:
        print(f"  HF push failed: {e}")

    # ── 9. APPENDIX SKELETON ───────────────────────────────────
    appendix_path = _nb / "results" / "appendix_skeleton.md"
    appendix_content = """# Appendix

    ## A. Dataset Cards
    | Dataset     | Source                              | N     | Task Type    |
    |-------------|-------------------------------------|-------|--------------|
    | GSM8K       | openai/gsm8k (test)                 | 300   | Math (grade) |
    | AQuA        | aqua_rat (test)                     | 254   | Math (MCQ)   |
    | MATH        | hendrycks/competition_math (test)   | 238   | Math (comp.) |
    | StrategyQA  | wics/strategy-qa                    | 300   | Boolean QA   |
    | BBH-LD      | lukaemon/bbh logical_deduction_7obj | 250   | Logic (MCQ)  |

    ## B. Prompt Templates
    See `ttc-frugalreason-poc/experiment_fr/core/prompt_manager.py` for full templates.

    ## C. Hyperparameters
    | Parameter | Value   | Source     |
    |-----------|---------|------------|
    | seed      | 0       | fixed      |
    | SC k      | 5       | standard   |
    | BoN k     | 5       | standard   |
    | ToT breadth | 3     | standard   |
    | α (FR)    | 0.6     | D4 α-grid  |
    | β (Dirichlet) | TBD | D7 sweep  |
    | T (temp scale) | TBD | D6 fit   |

    ## D. Extra Tables
    (Placeholder for supplementary material)
    """
    appendix_path.write_text(appendix_content, encoding="utf-8")
    print(f"\n  Appendix skeleton saved: results/appendix_skeleton.md")

    # ── FINAL LINE ──────────────────────────────────────────────
    end_time = time.time()
    print("\n" + "=" * 70)
    print(f"  D1–D12 COMPLETE — {total_runs} runs logged.")
    print(f"  STOP. Do not start writing-phase cells.")
    print("=" * 70)

execute_cell_51()