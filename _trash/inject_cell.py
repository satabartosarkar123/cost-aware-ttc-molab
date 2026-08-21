import json
import os

fname = 'c:/Users/USER/Cost-Aware-Test-Time/Cost-Aware-Test-time-molab/molab_run_fixed.ipynb'
with open(fname, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# check if we already appended it
already_there = False
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'run_sanity_gate.py' in ''.join(cell.get('source', [])):
        already_there = True
        break

if not already_there:
    sanity_cell = {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': [
            "# 1. Download the fixed frugalreason code\n",
            "!wget -O fr_patch.zip \"https://huggingface.co/datasets/Satabarto/Molab_Checkpoints_Cost_AWARE/resolve/main/fr_patch.zip?download=true\"\n",
            "!unzip -o fr_patch.zip\n",
            "!rm fr_patch.zip\n",
            "\n",
            "# 2. Run the Sanity Gate\n",
            "!python ttc-frugalreason-poc/experiment_fr/run_sanity_gate.py\n"
        ]
    }
    nb['cells'].insert(2, sanity_cell) # insert near the top
    with open(fname, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print('Sanity cell appended to molab_run_fixed.ipynb')
else:
    print('Sanity cell already exists in molab_run_fixed.ipynb')
