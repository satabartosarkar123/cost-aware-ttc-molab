import json
import os

base_dir = "c:/Users/USER/Cost-Aware-Test-Time/Cost-Aware-Test-time/ttc-frugalreason-poc/experiment_fr/results/block_a_part2_logs"

def load_results(strategy):
    results = {}
    path = os.path.join(base_dir, f"strategyqa_{strategy}.jsonl")
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as file:
            for line in file:
                if line.strip():
                    try:
                        data = json.loads(line)
                        results[data['qid']] = {
                            'correct': data['correct'],
                            'tokens': data['tokens'],
                            'calls': data.get('calls', data.get('model_calls', 1))
                        }
                    except Exception as e:
                        pass
    return results

frugal = load_results("frugal_reason_v3")
greedy_cot = load_results("greedy_cot")
sc5 = load_results("self_consistency_k5")
bon = load_results("best_of_n_k5_self_eval")

all_qids = set(greedy_cot.keys())
frugal_qids = set(frugal.keys())
missing_qids = all_qids - frugal_qids

def get_acc(results_dict, qids):
    correct = sum(1 for q in qids if results_dict.get(q, {}).get('correct', False))
    return correct / len(qids) * 100 if qids else 0

def get_avg(results_dict, qids, key):
    total = sum(results_dict.get(q, {}).get(key, 0) for q in qids if q in results_dict)
    return total / len(qids) if qids else 0

print(f"Frugal has answered {len(frugal_qids)} questions.")
print(f"Frugal has {len(missing_qids)} questions left.")

print(f"\n--- ACCURACY ON REMAINING {len(missing_qids)} QUESTIONS ---")
print(f"Greedy CoT accuracy on the REMAINING QIDs: {get_acc(greedy_cot, missing_qids):.2f}% (Overall: {get_acc(greedy_cot, all_qids):.2f}%)")
print(f"SC@5 accuracy on the REMAINING QIDs: {get_acc(sc5, missing_qids):.2f}% (Overall: {get_acc(sc5, all_qids):.2f}%)")
print(f"BoN@5 accuracy on the REMAINING QIDs: {get_acc(bon, missing_qids):.2f}% (Overall: {get_acc(bon, all_qids):.2f}%)")

print(f"\n--- ACCURACY ON COMPLETED {len(frugal_qids)} QUESTIONS (FOR COMPARISON) ---")
print(f"Greedy CoT accuracy on the COMPLETED QIDs: {get_acc(greedy_cot, frugal_qids):.2f}%")
print(f"SC@5 accuracy on the COMPLETED QIDs: {get_acc(sc5, frugal_qids):.2f}%")
print(f"BoN@5 accuracy on the COMPLETED QIDs: {get_acc(bon, frugal_qids):.2f}%")

# Predict final based on remaining
if len(missing_qids) > 0 and len(frugal_qids) > 0:
    # Frugal's delta to SC@5 on the COMPLETED questions
    delta = get_acc(frugal, frugal_qids) - get_acc(sc5, frugal_qids)
    
    # Assume it maintains this exact delta on the REMAINING questions
    predicted_remaining_acc = get_acc(sc5, missing_qids) + delta
    
    # Calculate weighted final average
    completed_correct = get_acc(frugal, frugal_qids) / 100 * len(frugal_qids)
    predicted_remaining_correct = predicted_remaining_acc / 100 * len(missing_qids)
    
    predicted_final = (completed_correct + predicted_remaining_correct) / len(all_qids) * 100
    print(f"\nBased on SC@5 performance, predicted accuracy for Frugal on the remaining {len(missing_qids)} questions is: {predicted_remaining_acc:.2f}%")
    print(f"Weighted Predicted Final Frugal Accuracy at 300: {predicted_final:.2f}%")
