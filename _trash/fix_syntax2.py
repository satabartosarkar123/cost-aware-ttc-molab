import json
nb = json.load(open('molab_run.ipynb', encoding='utf-8'))

# cell 24 line 65
src24 = ''.join(nb['cells'][24]['source']) if isinstance(nb['cells'][24]['source'], list) else nb['cells'][24]['source']
src24 = src24.replace('print(\"\\nSECTION 2', 'print(\"\\\\nSECTION 2')
nb['cells'][24]['source'] = src24

# cell 26 line 15
src26 = ''.join(nb['cells'][26]['source']) if isinstance(nb['cells'][26]['source'], list) else nb['cells'][26]['source']
lines26 = src26.split('\n')
print('cell 26 line 14:', repr(lines26[13]))
print('cell 26 line 15:', repr(lines26[14]))
print('cell 26 line 16:', repr(lines26[15]))

with open('molab_run.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
