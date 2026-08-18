import sys
from core.ollama_client import OllamaClient
from core.task_loader import load_all_tasks
from core.verifier import OutcomeVerifier
from strategies.frugal_reason_v3 import frugal_reason_v3_evaluate

def main():
    client = OllamaClient()
    verifier = OutcomeVerifier(ollama_client=client)
    
    print("Loading datasets...")
    # Load with identical sampling to day0 smoke
    all_tasks = load_all_tasks({}) 
    
    results = {}
    
    for task_name in ["gsm8k", "strategyqa", "aqua", "math"]:
        dataset = all_tasks[task_name][:10]
        correct_count = 0
        parse_count = 0
        
        print(f"Running {task_name} x frugal_reason_v3...")
        for i, item in enumerate(dataset):
            res = frugal_reason_v3_evaluate(client, task_name, item["question"], input_metadata="", enable_early_exit=True, alpha=0.6)
            ans = res.get("selected_answer", "")
            
            # CONFIRMATION: Using the exact same strict OutcomeVerifier as all baselines
            eval_res = verifier.score(task_name, item["question"], str(ans), str(ans), item["gold_answer"])
            
            if res.get("parse_success"):
                parse_count += 1
            if eval_res["score"] == 1.0:
                correct_count += 1
                
        results[task_name] = {
            "parse_rate": parse_count / len(dataset),
            "accuracy": correct_count / len(dataset)
        }
        print(f"  {task_name}: parse_rate={results[task_name]['parse_rate']:.2f}, acc={results[task_name]['accuracy']:.2f}")

    print("\n============================================================")
    print("CORRECTED FRUGAL REASON V3 SMOKE TEST REPORT")
    print("============================================================")
    for k, v in results.items():
        print(f"  {k}_frugal_reason_v3: parse_rate={v['parse_rate']:.2f}, acc={v['accuracy']:.2f}")

if __name__ == "__main__":
    main()
