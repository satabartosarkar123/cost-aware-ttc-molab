import json
nb = json.load(open('molab_run.ipynb', encoding='utf-8'))

src24 = ''.join(nb['cells'][24]['source']) if isinstance(nb['cells'][24]['source'], list) else nb['cells'][24]['source']
lines = src24.split('\n')
lines[64] = '    print("\\nSECTION 2  WILSON 95% CONFIDENCE INTERVALS")'
lines.pop(65)
nb['cells'][24]['source'] = '\n'.join(lines)

with open('molab_run.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
