import json
import re
import os
from pathlib import Path

try:
    with open('molab_run_fixed.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # 1. Remove duplicate data integrity checks
    cells = []
    seen_integrity_check = False
    for c in nb['cells']:
        if c['cell_type'] == 'code':
            src = ''.join(c['source']) if isinstance(c['source'], list) else c['source']
            if 'VERIFY DATA INTEGRITY BEFORE PROCEEDING' in src:
                if seen_integrity_check:
                    continue # Skip duplicate
                seen_integrity_check = True
        cells.append(c)
    nb['cells'] = cells

    # 2. Extract pip install logic from Ollama cell
    ollama_cell_idx = -1
    for i, c in enumerate(nb['cells']):
        if c['cell_type'] == 'code':
            src = ''.join(c['source']) if isinstance(c['source'], list) else c['source']
            if 'OLLAMA_MODEL' in src and 'pip install' in src:
                ollama_cell_idx = i
                break
    
    if ollama_cell_idx != -1:
        c = nb['cells'][ollama_cell_idx]
        src = ''.join(c['source']) if isinstance(c['source'], list) else c['source']
        
        # Remove pip install lines from Ollama cell
        new_src = []
        for line in src.split('\n'):
            if 'req = Path(nb_dir) / "requirements_molab.txt"' in line or \
               'if req.exists(): _run(f"pip install' in line or \
               'else: _run("pip install' in line or \
               'print("[5/6] Python dependencies ready")' in line:
                continue
            new_src.append(line)
        c['source'] = '\n'.join(new_src)

    # 3. Add pip install logic to the top of Cell 1, right before third party imports
    c1_idx = -1
    for i, c in enumerate(nb['cells']):
        if c['cell_type'] == 'code':
            src = ''.join(c['source']) if isinstance(c['source'], list) else c['source']
            if 'THIRD PARTY & LOCAL IMPORTS' in src:
                c1_idx = i
                break
    
    if c1_idx != -1:
        c = nb['cells'][c1_idx]
        src = ''.join(c['source']) if isinstance(c['source'], list) else c['source']
        
        pip_install_block = '''
# ── PIP INSTALL DEPENDENCIES ──
print("Installing dependencies...")
import subprocess
nb_dir = os.environ.get("NOTEBOOK_DIR", str(Path(".").resolve()))
req = Path(nb_dir) / "requirements_molab.txt"
if req.exists():
    subprocess.run(f"pip install -q --root-user-action=ignore -r {req}", shell=True)
else:
    subprocess.run("pip install -q --root-user-action=ignore requests datasets pandas numpy matplotlib seaborn tqdm pynvml psutil pyyaml tabulate reportlab scipy fpdf2", shell=True)
print("Dependencies installed.")

'''
        src = src.replace('# ── THIRD PARTY & LOCAL IMPORTS ──', pip_install_block + '# ── THIRD PARTY & LOCAL IMPORTS ──')
        
        # Fix the hf_hub_download issue by just using subprocess and not importing if not needed, or explicit import
        # It's better to just do explicit import
        src = src.replace('from huggingface_hub import hf_hub_download', 'import huggingface_hub\n    hf_hub_download = huggingface_hub.hf_hub_download')
        c['source'] = src

    with open('molab_run_fixed.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    
    print("Successfully refactored notebook!")
except Exception as e:
    import traceback
    traceback.print_exc()
