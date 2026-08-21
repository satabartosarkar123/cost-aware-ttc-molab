"""Dump every cell of molab_run_fixed.ipynb with line numbers."""
import json, sys
sys.stdout.reconfigure(encoding='utf-8')

with open('molab_run_fixed.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

total = len(nb['cells'])
print(f'Total cells: {total}')

for i, c in enumerate(nb['cells']):
    src = ''.join(c['source']) if isinstance(c['source'], list) else c['source']
    lines = src.split('\n')
    print(f'\n{"="*60}')
    print(f'CELL {i}  |  type={c["cell_type"]}  |  {len(lines)} lines')
    print('='*60)
    for ln, line in enumerate(lines, 1):
        print(f'{ln:4d}: {line}')
