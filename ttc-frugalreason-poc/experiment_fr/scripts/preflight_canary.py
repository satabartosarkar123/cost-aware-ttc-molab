import os
import sys
import json
import random
from pathlib import Path

base_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(base_dir))

from core.ollama_client import OllamaClient
from core.task_loader import load_all_tasks
from strategies.frugal_reason_v3 import frugal_reason_v3_evaluate

def main():
    client = OllamaClient(model="qwen2.5:3b")
    loader_config = {"sampling": {"questions_per_task": 20, "seed": 42}, "tasks": {"math": {}, "aqua": {}}}
    loaded = load_all_tasks(loader_config)
    
    questions = []
    if "math" in loaded: questions.extend([("math", item["question"]) for item in loaded["math"][:10]])
    if "aqua" in loaded: questions.extend([("aqua", item["question"]) for item in loaded["aqua"][:10]])
    
    print("=== STARTING V-SIGNAL CANARY ===")
    
    for ds, q in questions:
        print(f"\nTask: {ds}")
        res = frugal_reason_v3_evaluate(client, ds, q, input_metadata=q, enable_early_exit=True, alpha=0.6)
        
        candidates = res.get("candidates", [])
        if not candidates:
            print("  No candidates generated.")
            continue
            
        print(f"  Generated {len(candidates)} candidates.")
        for idx, c in enumerate(candidates):
            v_raw = c.get("V_raw")
            judge_text = c.get("raw_judge_text", "MISSING_JUDGE_TEXT").replace("\n", " ")
            print(f"  Cand {idx+1}: V_raw={v_raw} | Judge: '{judge_text[:100]}...'")

if __name__ == "__main__":
    main()
