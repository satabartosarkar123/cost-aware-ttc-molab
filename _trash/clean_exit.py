import json
from pathlib import Path

nb_path = Path('molab_run.ipynb')
nb = json.load(nb_path.open(encoding='utf-8'))

for i, c in enumerate(nb['cells']):
    if c['cell_type'] == 'code':
        source = ''.join(c['source'])
        if 'sys.exit(0)' in source:
            source = source.replace('import sys; sys.exit(0)', 'pass  # skip re-run')
            c['source'] = [source]
            print(f"Cleaned cell {i}")

nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
print("Done")
