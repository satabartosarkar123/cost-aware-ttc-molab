import json
nb = json.load(open('molab_run.ipynb', encoding='utf-8'))
for i, c in enumerate(nb['cells']):
    if c['cell_type'] == 'code':
        src = ''.join(c['source']) if isinstance(c['source'], list) else c['source']
        if 'fr_path = _nb / log_subdir / f"{prefix}{ds}_frugal_reason_v3.jsonl"' in src:
            print(f'Found in cell {i}:')
            for line in src.split('\n'):
                if 'frugal_reason_v3.jsonl' in line:
                    print(repr(line))
