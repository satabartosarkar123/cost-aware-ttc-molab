import os
import sys

# Ensure the experiment_fr folder is in the python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.ollama_client import OllamaClient
from core.task_loader import load_all_tasks
from strategies.frugal_reason_v3 import frugal_reason_v3_evaluate

def main():
    print("================================================================")
    print("V-SIGNAL SANITY GATE (QWEN2.5:3B)")
    print("================================================================")
    
    try:
        client = OllamaClient(model="qwen2.5:3b", host="http://localhost:11434")
    except Exception as e:
        print(f"Failed to connect to Ollama: {e}")
        return

    print("Loading MATH dataset...")
    tasks = load_all_tasks()
    math_data = tasks.get("math", [])
    
    if not math_data:
        print("MATH dataset not found!")
        return
        
    num_tested = 0
    num_ones = 0
    num_zeros = 0
    num_others = 0
    total_judges = 0
    
    for i, item in enumerate(math_data):
        if num_tested >= 20:
            break
            
        qid = item.get("id", f"math_{i}")
        question = item["question"]
        
        print(f"\n[{num_tested+1}/20] Running QID: {qid} ...")
        
        # Run the fixed FR strategy
        try:
            res = frugal_reason_v3_evaluate(client, task="math", question=question, enable_early_exit=False)
        except Exception as e:
            print(f"Error evaluating {qid}: {e}")
            continue
            
        judge_texts = res.get("judge_texts", [])
        if not judge_texts:
            print("  (No judge calls made for this question)")
            num_tested += 1
            continue
            
        print(f"  {len(judge_texts)} judge calls made:")
        for j, jt in enumerate(judge_texts):
            raw = jt.get("raw_text", "").replace('\n', ' ')
            score = jt.get("parsed_score", 0.0)
            
            # Print truncated raw text
            display_text = (raw[:150] + '...') if len(raw) > 150 else raw
            print(f"    Call {j+1} -> Parsed V: {score} | Raw: {display_text}")
            
            total_judges += 1
            if score == 1.0:
                num_ones += 1
            elif score == 0.0:
                num_zeros += 1
            else:
                num_others += 1
                
        num_tested += 1
        
    print("\n================================================================")
    print("SANITY GATE SUMMARY")
    print("================================================================")
    if total_judges > 0:
        print(f"Total Judge Calls Evaluated: {total_judges}")
        print(f"  V = 1.0 : {num_ones} ({(num_ones/total_judges)*100:.1f}%)")
        print(f"  V = 0.0 : {num_zeros} ({(num_zeros/total_judges)*100:.1f}%)")
        print(f"  Other   : {num_others} ({(num_others/total_judges)*100:.1f}%)")
        
        if num_others > 0:
            print("\n❌ FAILED: Found non-binary V scores! The parser bug is still active.")
        elif num_ones == 0:
            print("\n❌ FAILED: All scores were 0.0! The prompt fix might not be working.")
        else:
            print("\n✅ PASSED! Clean binary V-signal achieved. You may proceed with the full re-runs.")
    else:
        print("No judge calls were evaluated.")

if __name__ == "__main__":
    main()
