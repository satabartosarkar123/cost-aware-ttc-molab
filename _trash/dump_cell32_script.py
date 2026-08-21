import json
nb = json.load(open('molab_run.ipynb', 'r', encoding='utf-8'))
open('dump_cell32.py', 'w', encoding='utf-8').write("".join(nb['cells'][32]['source']))
