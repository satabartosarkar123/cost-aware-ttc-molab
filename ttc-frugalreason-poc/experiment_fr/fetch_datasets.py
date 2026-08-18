import os
import json
from datasets import load_dataset
from pathlib import Path

BASE_DIR = Path(r'c:\Users\USER\Cost-Aware-Test-Time\Cost-Aware-Test-time\ttc-frugalreason-poc\experiment_fr')
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

manifest_lines = ["# Data Manifest\n\n| Dataset | Expected Counts (approx) | Actual Count | Source | MATCH |\n| :--- | :--- | :--- | :--- | :--- |"]

def write_manifest(name, expected, actual, source):
    match = "YES" if expected == actual or expected == "N/A" else "NO"
    if isinstance(expected, int) and isinstance(actual, int) and expected != actual and abs(expected - actual) < 50:
        match = "YES (approx)"
    manifest_lines.append(f"| {name} | {expected} | {actual} | {source} | {match} |")

# 1. GSM-HARD
print("Fetching GSM-HARD...")
try:
    ds = load_dataset('reasoning-machines/gsm-hard', split='train')
    actual_count = 0
    with open(DATA_DIR / "gsm_hard.jsonl", "w", encoding="utf-8") as f:
        for item in ds:
            # GSM-HARD has 'input' and 'target'
            f.write(json.dumps({"question": item["input"], "gold_answer": str(item["target"])}) + "\n")
            actual_count += 1
    write_manifest("GSM-HARD", 1319, actual_count, "reasoning-machines/gsm-hard (HF)")
except Exception as e:
    print("Failed GSM-HARD", e)

# 2. SVAMP
print("Fetching SVAMP...")
try:
    ds = load_dataset('ChilleD/SVAMP', split='test')
    actual_count = 0
    with open(DATA_DIR / "svamp.jsonl", "w", encoding="utf-8") as f:
        for item in ds:
            # SVAMP has 'Question' and 'Answer'
            q = item.get("Question") or item.get("question", "")
            a = item.get("Answer") or item.get("answer", "")
            f.write(json.dumps({"question": q, "gold_answer": str(a)}) + "\n")
            actual_count += 1
    write_manifest("SVAMP", 300, actual_count, "ChilleD/SVAMP (HF)")
except Exception as e:
    print("Failed SVAMP", e)

# 3. AQuA
print("Fetching AQuA...")
try:
    ds = load_dataset('deepmind/aqua_rat', split='test')
    actual_count = 0
    with open(DATA_DIR / "aqua.jsonl", "w", encoding="utf-8") as f:
        for item in ds:
            # AQuA has 'question', 'options', 'correct'
            q = item["question"] + "\nOptions: " + ", ".join(item["options"])
            a = item["correct"]
            f.write(json.dumps({"question": q, "gold_answer": str(a)}) + "\n")
            actual_count += 1
    write_manifest("AQuA", 254, actual_count, "deepmind/aqua_rat (HF)")
except Exception as e:
    print("Failed AQuA", e)

# 4. MATH L1-3
print("Fetching MATH L1-3 from local prm800k repo...")
try:
    math_path = BASE_DIR / "temp_prm800k" / "prm800k" / "math_splits" / "test.jsonl"
    actual_count = 0
    with open(DATA_DIR / "math_l123.jsonl", "w", encoding="utf-8") as f_out:
        if math_path.exists():
            with open(math_path, "r", encoding="utf-8") as f_in:
                for line in f_in:
                    d = json.loads(line)
                    # Filter level 1, 2, 3
                    level_val = d.get("level", "")
                    if level_val in ["Level 1", "Level 2", "Level 3"] or level_val in [1, 2, 3]:
                        f_out.write(json.dumps({
                            "question": d["problem"],
                            "gold_answer": d["solution"], # Will be parsed later
                            "level": level_val,
                            "subject": d.get("type", d.get("subject", ""))
                        }) + "\n")
                        actual_count += 1
        else:
            print("prm800k math splits not found")
    write_manifest("MATH L1-3", "N/A", actual_count, "openai/prm800k (local clone)")
except Exception as e:
    print("Failed MATH", e)

# Write manifest
with open(BASE_DIR / "reports" / "data_manifest.md", "w", encoding="utf-8") as f:
    f.write("\n".join(manifest_lines) + "\n")
print("Data manifest written.")
