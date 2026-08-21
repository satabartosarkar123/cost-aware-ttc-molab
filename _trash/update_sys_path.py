import json
nb = json.load(open('molab_run.ipynb', encoding='utf-8'))
c1_idx = [i for i, c in enumerate(nb['cells']) if c['cell_type'] == 'code'][0]

src = ''.join(nb['cells'][c1_idx]['source']) if isinstance(nb['cells'][c1_idx]['source'], list) else nb['cells'][c1_idx]['source']

replacement = '''    sys.path.append(str(base_dir.resolve()))
    sys.path.insert(0, str((base_dir / "ttc-frugalreason-poc" / "experiment_fr").resolve()))
    sys.path.insert(0, str((base_dir / "ttc-task-poc" / "experiment").resolve()))'''

src = src.replace('    sys.path.append(str(base_dir.resolve()))', replacement)

nb['cells'][c1_idx]['source'] = src
with open('molab_run.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
