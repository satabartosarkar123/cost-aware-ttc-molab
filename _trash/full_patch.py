import json
import os

fname = 'molab_run.ipynb'

if not os.path.exists(fname):
    print("Not found")
    exit(0)

with open(fname, 'r', encoding='utf-8') as f:
    nb = json.load(f)

imports = [
    "import os\n",
    "import sys\n",
    "import json\n",
    "import subprocess\n",
    "import zipfile\n",
    "import shutil\n",
    "import time\n",
    "import re\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "from pathlib import Path\n",
    "from collections import defaultdict, Counter\n",
    "from tqdm import tqdm\n",
    "from huggingface_hub import HfApi, snapshot_download, hf_hub_download\n",
    "\n"
]

cells = nb.get('cells', [])
new_cells = []

# Insert block_b sync exactly before Day 4
sync_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Pre-Day 4: Fetch Block B data to ensure it exists\n",
        "import os, zipfile\n",
        "from pathlib import Path\n",
        "from huggingface_hub import hf_hub_download\n",
        "\n",
        "HF_REPO = 'Satabarto/Molab_Checkpoints_Cost_AWARE'\n",
        "HF_TOKEN = 'REDACTED'\n",
        "base_dir = Path(os.environ.get('NOTEBOOK_DIR', '.'))\n",
        "\n",
        "print('Fetching Block B data before Day 4...')\n",
        "for zname in ['block_b_qwen15b.zip', 'block_b_llama32.zip']:\n",
        "    try:\n",
        "        print(f'  Downloading {zname}...')\n",
        "        zpath = hf_hub_download(repo_id=HF_REPO, filename=f'checkpoints/{zname}', repo_type='dataset', token=HF_TOKEN)\n",
        "        with zipfile.ZipFile(zpath, 'r') as zf:\n",
        "            zf.extractall(str(base_dir))\n",
        "        print(f'  Restored {zname} successfully!')\n",
        "    except Exception as e:\n",
        "        print(f'  Warning: could not restore {zname}: {e}')\n",
        "print('Block B data fetch complete.')\n"
    ]
}

first_code_cell_idx = -1
for i, cell in enumerate(cells):
    if cell['cell_type'] == 'code':
        first_code_cell_idx = i
        break

if first_code_cell_idx != -1:
    source = cells[first_code_cell_idx]['source']
    if isinstance(source, str):
        source = "".join(imports) + source
    else:
        source = imports + source
    cells[first_code_cell_idx]['source'] = source

for i, cell in enumerate(cells):
    # If it is the Day 4 cell, we insert the sync_cell before it, unless it's already there
    if cell['cell_type'] == 'code' and any('Day4-AlphaGrid' in line for line in (cell['source'] if isinstance(cell['source'], list) else [cell['source']])):
        if i > 0 and 'Pre-Day 4: Fetch Block B data' in (cells[i-1]['source'][0] if isinstance(cells[i-1]['source'], list) else cells[i-1]['source']):
            pass
        else:
            new_cells.append(sync_cell)
            
        # We also need to patch the Day 4 cell for the block_b path logic without breaking string literals
        # Let's find where 'log_dir = _nb / "results" / log_subdir' is
        source = cell['source']
        if isinstance(source, list):
            new_source = []
            for j, line in enumerate(source):
                new_source.append(line)
                if 'log_dir = _nb / "results" / log_subdir' in line and j+1 < len(source) and '# Find FR log' in source[j+1]:
                    # We will replace the next few lines
                    pass
            # Instead of complex logic, let's just do a string replacement on the whole cell source string
            text = "".join(source)
            old_code = '''            log_dir = _nb / "results" / log_subdir
            # Find FR log
            fr_path = log_dir / f"{prefix}{ds}_frugal_reason_v3.jsonl"
            if not fr_path.exists():
                # Try experiment_fr path for Block A
                fr_path = _nb / "ttc-frugalreason-poc" / "experiment_fr" / "results" / log_subdir / f"{prefix}{ds}_frugal_reason_v3.jsonl"
            if not fr_path.exists():
                print(f"  SKIP {ds}: FR log not found")
                continue'''
            new_code = '''            fr_path = _nb / log_subdir / f"{prefix}{ds}_frugal_reason_v3.jsonl"
            if not fr_path.exists():
                fr_path = _nb / "results" / log_subdir / f"{prefix}{ds}_frugal_reason_v3.jsonl"
            if not fr_path.exists():
                # Try experiment_fr path for Block A
                fr_path = _nb / "ttc-frugalreason-poc" / "experiment_fr" / "results" / log_subdir / f"{prefix}{ds}_frugal_reason_v3.jsonl"
            if not fr_path.exists():
                if "qwen2.5:1.5b" in model_name:
                    fr_path = _nb / "block_b_qwen15b" / "block_b_logs" / f"{prefix}{ds}_frugal_reason_v3.jsonl"
                elif "llama3.2:3b" in model_name:
                    fr_path = _nb / "block_b_llama32" / "block_b_logs" / f"{prefix}{ds}_frugal_reason_v3.jsonl"
            if not fr_path.exists():
                print(f"  SKIP {ds}: FR log not found")
                continue'''
            text = text.replace(old_code, new_code)
            # convert back to list of lines with \n
            lines = text.split('\n')
            cell['source'] = [line + '\n' for line in lines[:-1]] + [lines[-1]] if lines[-1] else [line + '\n' for line in lines[:-1]]
            
    new_cells.append(cell)

nb['cells'] = new_cells

with open(fname, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print(f"Fully patched {fname}")
