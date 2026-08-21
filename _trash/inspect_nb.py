import json

with open('molab_run.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells'][:35]):
    ct = cell['cell_type']
    if cell['source']:
        first_line = cell['source'][0].strip()
        # safe print
        first_line = first_line.encode('ascii', 'ignore').decode('ascii')
    else:
        first_line = "EMPTY"
    
    print(f"Cell {i} ({ct}): {first_line[:50]}")
