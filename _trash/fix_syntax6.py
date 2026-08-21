import json
import re
nb = json.load(open('molab_run.ipynb', encoding='utf-8'))
for c in nb['cells']:
    if c['cell_type'] == 'code':
        src = ''.join(c['source']) if isinstance(c['source'], list) else c['source']
        lines = src.split('\n')
        i = 0
        while i < len(lines) - 1:
            if lines[i].strip() == 'print(\"':
                lines[i] = lines[i] + '\\n' + lines[i+1]
                lines.pop(i+1)
            else:
                i += 1
        c['source'] = '\n'.join(lines)
with open('molab_run.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
