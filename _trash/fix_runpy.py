import json
with open('c:/Users/USER/Cost-Aware-Test-Time/Cost-Aware-Test-time-molab/molab_run_fixed.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb.get('cells', []):
    if cell['cell_type'] == 'code':
        new_source = []
        for line in cell['source']:
            if 'runpy.run_path' in line:
                line = line.replace('runpy.run_path(str(script), run_name="__main__")', 'subprocess.run([sys.executable, str(script)])')
                line = line.replace('runpy.run_path(str(smoke), run_name="__main__")', 'subprocess.run([sys.executable, str(smoke)])')
            new_source.append(line)
        cell['source'] = new_source

with open('c:/Users/USER/Cost-Aware-Test-Time/Cost-Aware-Test-time-molab/molab_run_fixed.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
print("Fixed notebook!")
