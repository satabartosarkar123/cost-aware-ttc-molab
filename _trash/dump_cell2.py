import json

nb = json.load(open('molab_run.ipynb', 'r', encoding='utf-8'))
cell2_src = "".join(nb['cells'][2]['source'])

with open('cell2_dump.py', 'w', encoding='utf-8') as f:
    f.write(cell2_src)

print(f"Dumped Cell 2: {len(cell2_src)} chars, {len(cell2_src.splitlines())} lines")
