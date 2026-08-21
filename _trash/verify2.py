import json

nb = json.load(open('molab_run.ipynb', encoding='utf-8'))
for i, c in enumerate(nb['cells']):
    if c['cell_type'] == 'code':
        source = ''.join(c['source'])
        has_git = 'ghp_aG2QUWAK8q3' in source
        has_exit = 'SystemExit' in source
        has_sysexit = 'sys.exit' in source
        if has_git or has_exit or has_sysexit:
            print(f"Cell {i}: git={has_git} SystemExit={has_exit} sys.exit={has_sysexit}")

# If nothing prints, we're clean
print("\nIf no cells listed above, all fixes are clean!")
print(f"Total cells: {len(nb['cells'])}")
