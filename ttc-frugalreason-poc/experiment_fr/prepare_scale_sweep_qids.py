import json
from pathlib import Path
from core.task_loader import load_all_tasks

def prepare():
    qids = {"game24": [], "gsm8k": [], "strategyqa": []}
    
    log_path = Path("results/raw_logs/frugal_reason_v3_raw_seed0.jsonl")
    if log_path.exists():
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                    t = d.get("task")
                    qid = d.get("question_id")
                    if t in qids and qid not in qids[t] and len(qids[t]) < 30:
                        qids[t].append(qid)
                except: pass
                
    # New datasets
    config = {
        "sampling": {"questions_per_task": 30, "seed": 0},
        "tasks": {"gsm_hard": {}, "math": {}, "aqua": {}, "svamp": {}}
    }
    tasks = load_all_tasks(config)
    
    for t in ["gsm_hard", "math", "aqua", "svamp"]:
        if t in tasks:
            qids[t] = [item["id"] for item in tasks[t]]
            
    out_path = Path("data/scale_sweep_qids.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(qids, f, indent=2)
        
    print(f"Saved QIDs to {out_path}")
    for k, v in qids.items():
        print(f" - {k}: {len(v)}")

if __name__ == "__main__":
    prepare()
