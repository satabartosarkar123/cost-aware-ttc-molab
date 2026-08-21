import json
nb = json.load(open('molab_run.ipynb', encoding='utf-8'))
for i, c in enumerate(nb['cells']):
    if c['cell_type'] == 'code':
        src = ''.join(c['source']) if isinstance(c['source'], list) else c['source']
        print(f'Cell {i}: {repr(src[:80])}...')
