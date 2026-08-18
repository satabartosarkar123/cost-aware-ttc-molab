import json
from pathlib import Path
from core.verifier import OutcomeVerifier

BASE_DIR = Path(r'c:\Users\USER\Cost-Aware-Test-Time\Cost-Aware-Test-time\ttc-frugalreason-poc\experiment_fr')
RAW_LOG = BASE_DIR / 'results' / 'raw_logs' / 'frugal_reason_v3_raw_seed0.jsonl'
AUDIT_FILE = BASE_DIR / 'reports' / 'game24_raw_audit.md'

verifier = OutcomeVerifier()

with open(RAW_LOG, 'r', encoding='utf-8') as f:
    lines = [json.loads(line) for line in f if '"task": "game24"' in line]

out = ['# Game24 Raw Audit\n\n']
count = 0
for d in lines:
    if count >= 20: break
    if d['task'] == 'game24':
        out.append(f'## Question ID: {d.get("question_id")}\n')
        out.append(f'**Gold Answer:** {d.get("gold_answer")}\n')
        ans = d.get('selected_answer', '')
        
        eval_res = verifier.score("game24", "", ans, ans, d.get("gold_answer"))
        
        out.append(f'**Selected Answer / Raw Text:**\n```\n{ans}\n```\n')
        out.append(f'**Verifier Trace:** {eval_res}\n')
        out.append(f'**Correct:** {d.get("correct")}\n')
        out.append('---\n')
        count += 1

with open(AUDIT_FILE, 'w', encoding='utf-8') as f:
    f.writelines(out)
print(f'Audit generated at {AUDIT_FILE} with {count} items.')
