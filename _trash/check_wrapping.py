import json

nb = json.load(open('molab_run.ipynb', 'r', encoding='utf-8'))
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code':
        continue
    src = ''.join(cell['source'])
    has_sub = f'def execute_cell_{i}()' in src
    first_line = cell['source'][0].strip() if cell['source'] else ''
    # Sanitize for cp1252
    first_line = first_line.encode('ascii', 'replace').decode('ascii')[:80]
    print(f'Cell {i:2d} | wrapped={str(has_sub):5} | {first_line}')
