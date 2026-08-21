import json
nb = json.load(open('molab_run.ipynb', encoding='utf-8'))

# cell 26 line 15
src26 = ''.join(nb['cells'][26]['source']) if isinstance(nb['cells'][26]['source'], list) else nb['cells'][26]['source']
src26 = src26.replace('), \\n        f\"qwen2.5:1.5b', '), \\\n        f\"qwen2.5:1.5b')
src26 = src26.replace('), \n        f\"qwen2.5:1.5b', '), \\\n        f\"qwen2.5:1.5b')
nb['cells'][26]['source'] = src26

with open('molab_run.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
