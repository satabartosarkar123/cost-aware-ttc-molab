import json
import ast

with open('molab_run.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        src = "".join(cell['source'])
        # Fix literal newline in string
        if 'print("\n    Restoring' in src:
            src = src.replace('print("\n    Restoring', 'print("\\nRestoring')
            lines = src.split('\n')
            cell['source'] = [l + '\n' for l in lines]
            cell['source'][-1] = cell['source'][-1].rstrip('\n')

with open('molab_run.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    
# Run syntax check
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        src = "".join(cell['source'])
        try:
            ast.parse(src)
        except SyntaxError as e:
            print(f"SyntaxError in Cell {i}:\n{e}")
            import sys; sys.exit(1)
print("Syntax OK!")
