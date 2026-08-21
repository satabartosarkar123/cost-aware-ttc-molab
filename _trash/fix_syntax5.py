import json
nb = json.load(open('molab_run.ipynb', encoding='utf-8'))

src24 = ''.join(nb['cells'][24]['source']) if isinstance(nb['cells'][24]['source'], list) else nb['cells'][24]['source']
lines = src24.split('\n')
lines[108] = '    print("\\nSECTION 3  McNEMAR EXACT TESTS (accuracy, paired by qid)")'
lines.pop(109)
nb['cells'][24]['source'] = '\n'.join(lines)

with open('molab_run.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
