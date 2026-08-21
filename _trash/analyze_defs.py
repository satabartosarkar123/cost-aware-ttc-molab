import json

def analyze_exports():
    with open('molab_run.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            lines = cell['source']
            defs = [l for l in lines if l.startswith('def ') or l.startswith('class ')]
            if defs:
                print(f"Cell {i} defines: {', '.join(d.strip() for d in defs)}")

if __name__ == "__main__":
    analyze_exports()
