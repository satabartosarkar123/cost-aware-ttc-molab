import json
import sys

nb = json.load(open('molab_run.ipynb', encoding='utf-8'))
code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
try:
    exec(''.join(code_cells[0]['source'])) # New merged Cell 1
except Exception as e:
    import traceback
    traceback.print_exc()
