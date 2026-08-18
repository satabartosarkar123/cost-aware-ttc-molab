import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from core.ollama_client import OllamaClient
from core.task_loader import load_all_tasks
from core.verifier import OutcomeVerifier
from strategies.frugal_reason_v3 import frugal_reason_v3_evaluate

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    
    # We load 36 questions per task, same as baseline.
    tasks = load_all_tasks({"sampling": {"questions_per_task": 36, "seed": args.seed}})
    verifier = OutcomeVerifier()
    client = OllamaClient()
    
    out_dir = Path("results/ablations")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "frugal_reason_v3_raw.jsonl"
    
    # clear if exists
    if out_file.exists():
        os.remove(out_file)
        
    for task_name, items in tasks.items():
        print(f"Running {task_name}...")
        for i, item in enumerate(tqdm(items)):
            q_id = f"{task_name}_{i}"
            q_text = item["question"]
            gold = item["gold_answer"]
            
            res = frugal_reason_v3_evaluate(client, task_name, q_text, enable_early_exit=True, alpha=0.6)
            
            # evaluate correctness
            ans = res.get("selected_answer", "")
            eval_res = verifier.score(task_name, q_text, str(ans), str(ans), gold)
            
            res.update({
                "task": task_name,
                "question_id": q_id,
                "gold_answer": gold,
                "strict_answer": eval_res["score"] == 1.0,
                "lenient_answer": eval_res["score"] == 1.0,
                "correct": eval_res["score"] == 1.0
            })
            
            with open(out_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(res) + "\n")
                
if __name__ == "__main__":
    main()
