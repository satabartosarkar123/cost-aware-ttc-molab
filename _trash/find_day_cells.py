import json
nb=json.load(open('molab_run.ipynb','r',encoding='utf-8'))
for i, c in enumerate(nb['cells']):
    src = "".join(c.get('source', []))
    if 'Day' in src:
        first_line = c['source'][0].strip()[:60] if c['source'] else ""
        print(f"Cell {i} ({c['cell_type']}): {first_line}")
