import json

nb = json.load(open('molab_run.ipynb', encoding='utf-8'))
for i in [11, 13]:
    c = nb['cells'][i]
    source = ''.join(c['source'])
    for j, line in enumerate(source.splitlines()):
        if 'sys.exit' in line or 'SystemExit' in line:
            print(f"Cell {i}, line {j}: {line.strip()}")
