def execute_cell_59():
    # AO-CrossModel-72B — MATH-238 × 3 strategies on qwen2.5:72b (714 runs)
    import os, sys, json, time, math, random, re, sqlite3, subprocess, warnings
    import requests
    from pathlib import Path

    warnings.filterwarnings("ignore", category=SyntaxWarning)

    _nb = Path(os.environ.get("NOTEBOOK_DIR", "."))
    sys.path.insert(0, str(_nb / "ttc-frugalreason-poc" / "experiment_fr"))
    os.chdir(str(_nb / "ttc-frugalreason-poc" / "experiment_fr"))

    from core.ollama_client import OllamaClient
    from core.task_loader import load_all_tasks
    from core.parsers import get_parser
    from core.verifier import OutcomeVerifier
    from core.prompt_manager import get_prompt
    from strategies.frugal_reason_v3 import frugal_reason_v3_evaluate

    # ── PRE-ASSERT: 70B must NOT be loaded ──────────────────────────
    ps = subprocess.run("ollama ps", shell=True, capture_output=True, text=True)
    if "llama3.3:70b" in ps.stdout or "70b" in ps.stdout.lower():
        print("WARNING: 70B still loaded! Attempting to unload...")
        subprocess.run("ollama stop llama3.3:70b", shell=True)
        time.sleep(5)
        # Fallback: keep_alive=0
        try:
            requests.post("http://localhost:11434/api/generate",
                           json={"model": "llama3.3:70b", "prompt": "", "keep_alive": 0}, timeout=30)
        except: pass
        time.sleep(5)
        ps2 = subprocess.run("ollama ps", shell=True, capture_output=True, text=True)
        assert "70b" not in ps2.stdout.lower(), f"FATAL: 70B still loaded after stop!\n{ps2.stdout}"
        print("70B successfully unloaded.")
    else:
        print("PRE-ASSERT PASS: No 70B model loaded.")

    subprocess.run("nvidia-smi --query-gpu=memory.free --format=csv,noheader", shell=True)

    MODEL = "qwen2.5:72b"
    PREFIX = "qwen72b_"
    SEED = 0
    # Only 3 strategies for cross-validation
    STRATEGIES = ["greedy_io", "greedy_cot", "frugal_reason_v4"]
    DATASETS = ["math"]

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

    def run_strategy(client, strat, task, question):
        if strat == "greedy_io": return run_greedy_io(client, task, question)
        elif strat == "greedy_cot": return run_greedy_cot(client, task, question)
        elif strat == "frugal_reason_v4":
            res = frugal_reason_v3_evaluate(client, task, question, input_metadata=question,
                                             enable_early_exit=True, alpha=0.6)
            return {"selected_answer": res.get("selected_answer"),
                    "raw_response": json.dumps(res), "latency_seconds_total": res.get("latency", 0.0),
                    "total_tokens": res.get("tokens", 0), "model_calls": res.get("calls", 0),
                    "parse_success": res.get("parse_success", False),
                    "parse_method": res.get("route_used", "frugal_reason_v3"),
                    "raw_paths": [], "clusters": res.get("clusters", []),
                    "candidates": res.get("candidates", []),
                    "early_exit": res.get("early_exit", False)}
        raise ValueError(f"Unknown strategy: {strat}")

    # ── Load MATH-238 ────────────────────────────────────────────────
    loader_config = {"sampling": {"questions_per_task": 1500, "seed": SEED},
                     "tasks": {"math": {}}}
    loaded = load_all_tasks(loader_config)
    math_items = loaded.get("math", [])
    task_map = {item["id"]: item for item in math_items}
    qid_list = list(task_map.keys())[:238]

    # ── SQLite checkpoint ────────────────────────────────────────────
    results_dir = _nb / "results" / "block_b_logs"
    results_dir.mkdir(parents=True, exist_ok=True)
    db_path = str(_nb / "block_b_checkpoint.db")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS completed (
        model TEXT, dataset TEXT, strategy TEXT, qid TEXT, timestamp DATETIME,
        PRIMARY KEY(model, dataset, strategy, qid))""")
    conn.commit()

    client = OllamaClient(model=MODEL)
    verifier = OutcomeVerifier(client)
    total_target = len(qid_list) * len(STRATEGIES)
    done = 0; start = time.time()
    hw = "gpu" if os.path.exists("/proc/driver/nvidia") else "cpu"

    print(f"\nTarget: {total_target} runs on {MODEL} (MATH-238 × 3 strategies)")

    for strat in STRATEGIES:
        log_path = results_dir / f"{PREFIX}math_{strat}.jsonl"
        for qid in qid_list:
            cur.execute("SELECT 1 FROM completed WHERE model=? AND dataset=? AND strategy=? AND qid=?",
                        (MODEL, "math", strat, qid))
            if cur.fetchone():
                done += 1; continue

            item = task_map.get(qid)
            if not item: done += 1; continue

            for attempt in range(3):
                try:
                    res = run_strategy(client, strat, "math", item["question"])
                    break
                except Exception as e:
                    print(f"  Retry {attempt+1}/3 math/{strat}/{qid}: {e}")
                    time.sleep(10)
            else:
                done += 1; continue

            score_res = verifier.score("math", item["question"], res.get("raw_response",""),
                                        res["selected_answer"], item["gold_answer"])
            is_correct = score_res["score"] == 1.0

            log_row = {
                "model": MODEL, "dataset": "math", "strategy": strat, "qid": qid,
                "gold": item["gold_answer"], "selected_answer": res["selected_answer"],
                "correct": is_correct, "parse_success": res["parse_success"],
                "parse_method": res.get("parse_method", ""), "latency_seconds": res["latency_seconds_total"],
                "tokens": res["total_tokens"], "calls": res["model_calls"],
                "hardware_type": hw, "early_exit": res.get("early_exit", False),
                "clusters": res.get("clusters", []), "candidates": res.get("candidates", []),
                "raw_paths": res.get("raw_paths", []),
            }
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_row) + "\n")
            cur.execute("INSERT OR IGNORE INTO completed VALUES (?,?,?,?,datetime('now'))",
                        (MODEL, "math", strat, qid))
            conn.commit()
            done += 1

            if done % 20 == 0:
                elapsed = time.time() - start
                eta = (total_target - done) * (elapsed / max(done, 1))
                print(f"[{MODEL}] {done}/{total_target} | {elapsed/3600:.1f}h elapsed | ETA {eta/3600:.1f}h")

    conn.close()
    print(f"\n72B Cross-Model DONE: {done}/{total_target} runs.")

    # ── HF push ──────────────────────────────────────────────────────
    try:
        from huggingface_hub import HfApi
        import zipfile
        _api = HfApi(token="REDACTED")
        _zp = str(_nb / "results" / "block_b_qwen72b.zip")
        with zipfile.ZipFile(_zp, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in results_dir.glob(f"{PREFIX}*.jsonl"):
                zf.write(str(f), f"block_b_logs/{f.name}")
        _api.upload_file(path_or_fileobj=_zp, path_in_repo="checkpoints/block_b_qwen72b.zip",
                         repo_id="Satabarto/Molab_Checkpoints_Cost_AWARE", repo_type="dataset")
        print("Pushed 72B results to HF.")
    except Exception as e:
        print(f"HF push failed (non-fatal): {e}")

    # ── VRAM UNLOAD ──────────────────────────────────────────────────
    def _ollama_unload(model_name):
        import subprocess, time, requests as _req
        print(f"\n--- VRAM HYGIENE: unloading {model_name} ---")
        r = subprocess.run(f"ollama stop {model_name}", shell=True, capture_output=True, text=True)
        if r.returncode == 0:
            print(f"  ollama stop {model_name}: OK")
        else:
            try:
                _req.post("http://localhost:11434/api/generate",
                           json={"model": model_name, "prompt": "", "keep_alive": 0}, timeout=30)
                print(f"  keep_alive=0 sent to {model_name}")
            except Exception:
                print(f"  Restarting Ollama server to free VRAM...")
                subprocess.run("pkill -f 'ollama serve' 2>/dev/null || true", shell=True)
                time.sleep(3)
                subprocess.Popen("nohup ollama serve > /dev/null 2>&1 &", shell=True)
                time.sleep(5)
        time.sleep(3)
        ps = subprocess.run("ollama ps", shell=True, capture_output=True, text=True)
        if model_name.split(":")[0] in ps.stdout:
            print(f"  WARNING: {model_name} still loaded! Output:\n{ps.stdout}")
        else:
            print(f"  CONFIRMED: {model_name} unloaded from VRAM")
        subprocess.run("nvidia-smi --query-gpu=memory.free --format=csv,noheader", shell=True)

    _ollama_unload(MODEL)

execute_cell_59()