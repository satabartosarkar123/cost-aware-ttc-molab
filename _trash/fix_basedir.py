import json

nb = json.load(open('molab_run.ipynb', encoding='utf-8'))
for c in nb['cells']:
    if c['cell_type'] == 'code':
        source = ''.join(c['source'])
        if 'base_dir = Path' in source and 'Cost-Aware' in source:
            old_logic = '''    base_dir = Path(".")
    if (base_dir / "Cost-Aware-Test-Time-upload").exists():
        base_dir = base_dir / "Cost-Aware-Test-Time-upload"
    elif (base_dir / "Cost-Aware-Test-time-molab").exists():
        base_dir = base_dir / "Cost-Aware-Test-time-molab"'''
            
            new_logic = '''    base_dir = Path(".")
    if (base_dir / "gdrive_oauth.py").exists() and (base_dir / "rq2_part1/run_rq2_part1.py").exists():
        pass # Files are directly in current dir
    elif (base_dir / "Cost-Aware-Test-time-molab" / "gdrive_oauth.py").exists():
        base_dir = base_dir / "Cost-Aware-Test-time-molab"
    elif (base_dir / "Cost-Aware-Test-Time-upload" / "gdrive_oauth.py").exists():
        base_dir = base_dir / "Cost-Aware-Test-Time-upload"'''
            
            source = source.replace(old_logic, new_logic)
            c['source'] = [source]
            
with open('molab_run.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("Updated base_dir extraction logic!")
