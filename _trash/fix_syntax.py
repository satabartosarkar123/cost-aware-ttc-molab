import json
nb = json.load(open('molab_run.ipynb', encoding='utf-8'))

for c in nb['cells']:
    if c['cell_type'] == 'code':
        if isinstance(c['source'], list):
            for i, line in enumerate(c['source']):
                if '\\\n' in line:
                    c['source'][i] = line.replace('\\\n', '\n')
                if '\"\"\"\\n' in line:
                    c['source'][i] = line.replace('\"\"\"\\n', '\"\"\"\n')
                if '\'\'\'\\n' in line:
                    c['source'][i] = line.replace('\'\'\'\\n', '\'\'\'\n')
                if ')\\n' in line:
                    c['source'][i] = line.replace(')\\n', ')\n')
        else:
            c['source'] = c['source'].replace('\"\"\"\\n', '\"\"\"\n')
            c['source'] = c['source'].replace('\'\'\'\\n', '\'\'\'\n')
            c['source'] = c['source'].replace(')\\n', ')\n')

with open('molab_run.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
