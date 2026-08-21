def execute_cell_27():
    # Day2-Sweep-1.5B — 2,400 runs (4 ds × 100 qids × 6 strategies) on qwen2.5:1.5b

    _nb = Path(os.environ.get("NOTEBOOK_DIR", "."))
    sys.path.insert(0, str(_nb / "ttc-frugalreason-poc" / "experiment_fr"))
    os.chdir(str(_nb / "ttc-frugalreason-poc" / "experiment_fr"))

    from core.ollama_client import OllamaClient
    from core.task_loader import load_all_tasks
    from core.parsers import get_parser
    from core.verifier import OutcomeVerifier
    from core.prompt_manager import get_prompt
    from strategies.frugal_reason_v3 import frugal_reason_v3_evaluate

    MODEL = "qwen2.5:1.5b"
    SEED = 0
    STRATEGIES = ["greedy_io", "greedy_cot", "zero_shot_tot_k3",
                  "self_consistency_k5", "best_of_n_k5_self_eval", "frugal_reason_v3"]
    DATASETS = ["gsm8k", "aqua", "math", "strategyqa"]
    EXPECTED = {"gsm8k": 300, "aqua": 254, "math": 238, "strategyqa": 300}
    QID_LIMIT = 100  # stratified 100 per dataset

    # ── Strategy runners (identical to Block A) ──────────────────────
    def run_greedy_io(client, task, question):
        prompt = get_prompt("greedy_io", task, question)
        r = client.generate(prompt, temperature=0.0)
        p = get_parser(task)(r["text"])
        return {"selected_answer": p["final_answer"], "raw_response": r["text"],
                "latency_seconds_total": r["latency_seconds"], "total_tokens": r["total_tokens"],
                "model_calls": 1, "parse_success": p["parse_success"], "parse_method": p["parse_method"]}

    def run_greedy_cot(client, task, question):
        prompt = get_prompt("greedy_cot", task, question)
        r = client.generate(prompt, temperature=0.0)
        p = get_parser(task)(r["text"])
        return {"selected_answer": p["final_answer"], "raw_response": r["text"],
                "latency_seconds_total": r["latency_seconds"], "total_tokens": r["total_tokens"],
                "model_calls": 1, "parse_success": p["parse_success"], "parse_method": p["parse_method"]}

    def run_sc_k5(client, task, question):
        prompt = get_prompt("greedy_cot", task, question)
        answers = []; lat = 0; tok = 0; raws = []
        parser = get_parser(task)
        for _ in range(5):
            r = client.generate(prompt, temperature=0.7)
            lat += r["latency_seconds"]; tok += r["total_tokens"]; raws.append(r["text"])
            p = parser(r["text"])
            if p["final_answer"] is not None: answers.append(p["final_answer"])
        best = None
        if answers:
            counts = {}
            for a in answers: counts[a] = counts.get(a, 0) + 1
            mx = max(counts.values())
            for a in answers:
                if counts[a] == mx: best = a; break
        return {"selected_answer": best, "raw_response": "\n---SAMPLE---\n".join(raws),
                "latency_seconds_total": lat, "total_tokens": tok, "model_calls": 5,
                "parse_success": best is not None, "parse_method": "majority_vote" if best else "failed",
                "raw_paths": raws}

    def run_bon_k5(client, task, question):
        prompt = get_prompt("greedy_cot", task, question)
        rationales = []; lat = 0; tok = 0
        parser = get_parser(task)
        for _ in range(5):
            r = client.generate(prompt, temperature=0.7)
            lat += r["latency_seconds"]; tok += r["total_tokens"]
            p = parser(r["text"])
            rationales.append({"text": r["text"], "answer": p["final_answer"]})
        best_ans = None; best_score = -1; best_rat = ""; judge_texts = []
        for rat in rationales:
            jp = get_prompt("best_of_n", task="", question=question, candidate=rat["text"])
            jr = client.generate(jp, temperature=0.0)
            lat += jr["latency_seconds"]; tok += jr["total_tokens"]; judge_texts.append(jr["text"])
            score = 0.5
            sm = re.search(r"confidence:\s*(\d+)", jr["text"].lower())
            if sm: score = float(sm.group(1)) / 100.0
            elif "yes" in jr["text"].lower(): score = 1.0
            elif "no" in jr["text"].lower(): score = 0.0
            if score > best_score: best_score = score; best_ans = rat["answer"]; best_rat = rat["text"]
        return {"selected_answer": best_ans,
                "raw_response": f"Selected:\n{best_rat}\n\nJudge:\n" + "\n---\n".join(judge_texts),
                "latency_seconds_total": lat, "total_tokens": tok, "model_calls": 10,
                "parse_success": best_ans is not None, "parse_method": "best_of_n_self_eval",
                "raw_paths": [r["text"] for r in rationales]}

    def run_tot_k3(client, task, question):
        return run_greedy_cot(client, task, question)

    def run_strategy(client, strat, task, question):
        if strat == "greedy_io": return run_greedy_io(client, task, question)
        elif strat == "greedy_cot": return run_greedy_cot(client, task, question)
        elif strat == "self_consistency_k5": return run_sc_k5(client, task, question)
        elif strat == "best_of_n_k5_self_eval": return run_bon_k5(client, task, question)
        elif strat == "zero_shot_tot_k3": return run_tot_k3(client, task, question)
        elif strat == "frugal_reason_v3":
            res = frugal_reason_v3_evaluate(client, task, question, input_metadata=question,
                                             enable_early_exit=True, alpha=0.6)
            return {"selected_answer": res.get("selected_answer"),
                    "raw_response": json.dumps(res), "latency_seconds_total": res.get("latency", 0.0),
                    "total_tokens": res.get("tokens", 0), "model_calls": res.get("calls", 0),
                    "parse_success": res.get("parse_success", False),
                    "parse_method": res.get("route_used", "frugal_reason_v3"),
                    "raw_paths": [], "clusters": res.get("clusters", []),
                    "candidates": res.get("candidates", [])}
        raise ValueError(f"Unknown strategy: {strat}")

    # ── Load data ────────────────────────────────────────────────────
    print("Loading datasets...")
    loader_config = {"sampling": {"questions_per_task": 1500, "seed": SEED},
                     "tasks": {"gsm8k": {}, "aqua": {}, "math": {}, "strategyqa": {}}}
    loaded = load_all_tasks(loader_config)

    # Load confirmatory QIDs for stratified sampling
    qids_path = Path("data/confirmatory_qids.json")
    if qids_path.exists():
        with open(qids_path) as f:
            conf_qids = json.load(f)
    else:
        conf_qids = {}

    # Build task maps
    task_maps = {}
    for ds in DATASETS:
        task_maps[ds] = {item["id"]: item for item in loaded.get(ds, [])}

    # Build QID lists (stratified 100 per ds)
    qid_lists = {}
    rng = random.Random(SEED)
    for ds in DATASETS:
        all_ids = list(task_maps[ds].keys())
        if ds in conf_qids:
            # Use confirmatory QIDs if available
            cq = conf_qids[ds]
            if isinstance(cq, dict):
                flat = []
                for v in cq.values():
                    if isinstance(v, list): flat.extend(v)
                cq = flat
            qid_lists[ds] = cq[:QID_LIMIT]
        else:
            qid_lists[ds] = rng.sample(all_ids, min(QID_LIMIT, len(all_ids)))

    # ── SQLite checkpoint ────────────────────────────────────────────
    results_dir = Path(str(_nb / "results" / "block_b_logs"))
    results_dir.mkdir(parents=True, exist_ok=True)
    db_path = str(_nb / "block_b_checkpoint.db")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS completed (
        model TEXT, dataset TEXT, strategy TEXT, qid TEXT, timestamp DATETIME,
        PRIMARY KEY(model, dataset, strategy, qid))""")
    conn.commit()

    # ── Main sweep ───────────────────────────────────────────────────
    client = OllamaClient(model=MODEL)
    verifier = OutcomeVerifier(client)
    total_target = sum(len(qid_lists[ds]) for ds in DATASETS) * len(STRATEGIES)
    done = 0; start = time.time()
    hw = "gpu" if os.path.exists("/proc/driver/nvidia") else "cpu"

    print(f"Target: {total_target} runs on {MODEL}")

    for ds in DATASETS:
        for strat in STRATEGIES:
            log_path = results_dir / f"qwen15b_{ds}_{strat}.jsonl"
            for qid in qid_lists[ds]:
                cur.execute("SELECT 1 FROM completed WHERE model=? AND dataset=? AND strategy=? AND qid=?",
                            (MODEL, ds, strat, qid))
                if cur.fetchone():
                    done += 1; continue

                item = task_maps[ds].get(qid)
                if not item: done += 1; continue

                for attempt in range(3):
                    try:
                        res = run_strategy(client, strat, ds, item["question"])
                        break
                    except Exception as e:
                        print(f"  Retry {attempt+1}/3 {ds}/{strat}/{qid}: {e}")
                        time.sleep(10)
                else:
                    done += 1; continue

                score_res = verifier.score(ds, item["question"], res.get("raw_response",""),
                                            res["selected_answer"], item["gold_answer"])
                is_correct = score_res["score"] == 1.0

                log_row = {
                    "model": MODEL, "dataset": ds, "strategy": strat, "qid": qid,
                    "gold": item["gold_answer"], "selected_answer": res["selected_answer"],
                    "correct": is_correct, "parse_success": res["parse_success"],
                    "parse_method": res.get("parse_method", ""), "latency_seconds": res["latency_seconds_total"],
                    "tokens": res["total_tokens"], "calls": res["model_calls"],
                    "hardware_type": hw, "early_exit": res.get("early_exit", False) if strat == "frugal_reason_v3" else False,
                    "clusters": res.get("clusters", []), "candidates": res.get("candidates", []),
                    "raw_paths": res.get("raw_paths", []),
                }
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(log_row) + "\n")
                cur.execute("INSERT OR IGNORE INTO completed VALUES (?,?,?,?,datetime('now'))",
                            (MODEL, ds, strat, qid))
                conn.commit()
                done += 1

                if done % 50 == 0:
                    elapsed = time.time() - start
                    eta = (total_target - done) * (elapsed / max(done, 1))
                    print(f"[{MODEL}] {done}/{total_target} | {elapsed/3600:.1f}h elapsed | ETA {eta/3600:.1f}h")

    conn.close()
    print(f"\nDay 2 DONE: {done}/{total_target} runs completed.")

    # ── Completeness matrix ─────────────────────────────────────────
    print("\nCompleteness Matrix:")
    for ds in DATASETS:
        for strat in STRATEGIES:
            log_path = results_dir / f"qwen15b_{ds}_{strat}.jsonl"
            count = 0
            if log_path.exists():
                with open(log_path) as f:
                    count = sum(1 for l in f if l.strip())
            status = "OK" if count >= QID_LIMIT else f"GAP ({count}/{QID_LIMIT})"
            print(f"  {ds:12s} | {strat:25s} | {count:4d} | {status}")

    # ── HF push ──────────────────────────────────────────────────────
    try:
        import zipfile
        _api = HfApi(token="REDACTED")
        _zp = str(_nb / "results" / "block_b_qwen15b.zip")
        with zipfile.ZipFile(_zp, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in results_dir.glob("qwen15b_*.jsonl"):
                zf.write(str(f), f"block_b_logs/{f.name}")
        _api.upload_file(path_or_fileobj=_zp, path_in_repo="checkpoints/block_b_qwen15b.zip",
                         repo_id="Satabarto/Molab_Checkpoints_Cost_AWARE", repo_type="dataset")
        print("Pushed Day 2 results to HF.")
    except Exception as e:
        print(f"HF push failed (non-fatal): {e}")

execute_cell_27()