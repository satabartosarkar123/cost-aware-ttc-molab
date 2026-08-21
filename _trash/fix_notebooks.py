import json
import os

files_to_patch = ['molab_run.ipynb', 'broken_test/molab_run.ipynb']

for fname in files_to_patch:
    if not os.path.exists(fname):
        continue
    
    with open(fname, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    for cell in nb.get('cells', []):
        if cell['cell_type'] == 'code':
            source = cell['source']
            for i in range(len(source)):
                if '\\n' in source[i]:
                    source[i] = source[i].replace('\\n', '\n')

    with open(fname, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    
    print(f"Fixed {fname}")
