import json
import os

files_to_patch = ['molab_run.ipynb', 'broken_test/molab_run.ipynb']

for fname in files_to_patch:
    if not os.path.exists(fname):
        continue
    
    with open(fname, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    cells = nb.get('cells', [])
    new_cells = []
    
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
            "HF_TOKEN = 'REDACTED'  # Make sure this is correct or loaded from env\n",
            "base_dir = Path(os.environ.get('NOTEBOOK_DIR', '.'))\n",
            "\n",
            "print('Fetching Block B data before Day 4...')\n",
            "for zname in ['block_b_qwen15b.zip', 'block_b_llama32.zip']:\n",
            "    try:\n",
            "        print(f'  Downloading {zname}...')\n",
            "        zpath = hf_hub_download(repo_id=HF_REPO, filename=f'checkpoints/{zname}', repo_type='dataset', token=HF_TOKEN)\n",
            "        with zipfile.ZipFile(zpath, 'r') as zf:\n",
            "            zf.extractall(str(base_dir))\n",
            "            # Note: extractall to base_dir if the zip contains 'block_b_qwen15b' folder directly\n",
            "        print(f'  Restored {zname} successfully!')\n",
            "    except Exception as e:\n",
            "        print(f'  Warning: could not restore {zname}: {e}')\n",
            "print('Block B data fetch complete.')\n"
        ]
    }
    
    for i, cell in enumerate(cells):
        if cell['cell_type'] == 'code' and any('Day4-AlphaGrid' in line for line in cell.get('source', [])):
            # Found Day 4 cell
            # Only add it once
            # First check if the cell before is already our sync cell
            if i > 0 and 'Pre-Day 4: Fetch Block B data' in cells[i-1].get('source', [''])[0]:
                pass
            else:
                new_cells.append(sync_cell)
        new_cells.append(cell)
        
    nb['cells'] = new_cells

    with open(fname, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    
    print(f"Patched {fname}")
