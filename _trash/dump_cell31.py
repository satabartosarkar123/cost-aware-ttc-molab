import json
nb = json.load(open('molab_run_fixed.ipynb', encoding='utf-8'))
for i, c in enumerate(nb['cells']):
    if c['cell_type'] == 'code':
        src = ''.join(c['source']) if isinstance(c['source'], list) else c['source']
        if 'cot_path = log_dir / f"{prefix}{ds}_greedy_cot.jsonl"' in src:
            with open('cell31.py', 'w', encoding='utf-8') as f:
                f.write(src)
