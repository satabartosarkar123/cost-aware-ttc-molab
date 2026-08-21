import os
import re

with open('ttc-frugalreason-poc/experiment_fr/run_strict_eval.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix Bug 4: HF token
new_content = re.sub(
    r'api = HfApi\(token=".*?"\)',
    'token = os.environ.get("HF_TOKEN")\n    if not token:\n        print(" [HF SYNC] HF_TOKEN env var not set. Skipping sync.")\n        return\n    api = HfApi(token=token)',
    content
)

# Fix Bug 5: db_completed unused
new_content = new_content.replace(
    'cursor.execute("SELECT 1 FROM completed WHERE dataset=? AND strategy=? AND qid=?", (dataset_name, strategy, q_id))\n                if cursor.fetchone():\n                    continue',
    'if q_id in db_completed:\n                    continue\n                db_completed.add(q_id)'
)

with open('ttc-frugalreason-poc/experiment_fr/run_strict_eval.py', 'w', encoding='utf-8') as f:
    f.write(new_content)
