import json
import re
import os
import subprocess
from pathlib import Path

# 1. Fix requirements_molab.txt
req_path = Path('requirements_molab.txt')
reqs = req_path.read_text(encoding='utf-8')
reqs = re.sub(r'numpy==.*', 'numpy>=1.26.4', reqs)
reqs = re.sub(r'scipy==.*', 'scipy>=1.13.1', reqs)
reqs = re.sub(r'pandas==.*', 'pandas>=2.2.2', reqs)
req_path.write_text(reqs, encoding='utf-8')
print("Fixed requirements_molab.txt")

# 2. Fix the notebooks
for nb_name in ['molab_run.ipynb', 'molab_run_fixed.ipynb']:
    with open(nb_name, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    for c in nb['cells']:
        if c['cell_type'] == 'code':
            src = ''.join(c['source']) if isinstance(c['source'], list) else c['source']
            
            # Fix hf_hub_download issue by using huggingface-cli
            if 'hf_hub_download' in src or 'huggingface_hub' in src:
                new_src = []
                in_hf_block = False
                for line in src.split('\n'):
                    if 'import huggingface_hub' in line or 'from huggingface_hub' in line:
                        continue
                    if 'hf_hub_download =' in line:
                        continue
                    if 'hf_path = hf_hub_download(' in line:
                        cli_cmd = "        subprocess.run(f'huggingface-cli download --repo-type dataset {HF_REPO} {ZIP_NAME} --local-dir . --token {HF_TOKEN}', shell=True, check=True)\n        hf_path = ZIP_NAME"
                        new_src.append(cli_cmd)
                        continue
                    new_src.append(line)
                src = '\n'.join(new_src)
            
            c['source'] = src
    with open(nb_name, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print(f"Fixed {nb_name}")

print("Done fixing files!")
