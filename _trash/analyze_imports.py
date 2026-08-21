import json
import re

def analyze_notebook():
    with open('molab_run.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    imports = set()
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            lines = cell['source']
            for line in lines:
                if line.startswith('import ') or line.startswith('from '):
                    imports.add(line.strip())
                    
    print(f"Found {len(imports)} unique import statements.")
    for imp in sorted(list(imports)):
        print("  " + imp)

if __name__ == "__main__":
    analyze_notebook()
