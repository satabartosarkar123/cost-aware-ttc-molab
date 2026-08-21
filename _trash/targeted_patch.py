import json

with open('molab_run.ipynb', 'r', encoding='utf-8') as f:
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

# Ensure cell 1 (first code cell) has imports
cells = nb.get('cells', [])
for cell in cells:
    if cell['cell_type'] == 'code':
        source = cell['source']
        if isinstance(source, list):
            if not any('import pandas as pd' in line for line in source):
                cell['source'] = imports + source
        else:
            if 'import pandas as pd' not in source:
                cell['source'] = "".join(imports) + source
        break

# Find execute_cell_32 and add imports to it + fix NOTEBOOK_DIR
for cell in cells:
    if cell['cell_type'] == 'code':
        source = cell['source']
        is_list = isinstance(source, list)
        text = "".join(source) if is_list else source
        
        if 'def execute_cell_32():' in text:
            # We want to inject imports right after def execute_cell_32():
            if 'import pandas as pd' not in text:
                text = text.replace('def execute_cell_32():\n', 'def execute_cell_32():\n' + "".join(["    " + imp for imp in imports]))
            
            # Fix NOTEBOOK_DIR fallback to /kaggle/working if it is '.'
            old_nb = '_nb = Path(os.environ.get("NOTEBOOK_DIR", "."))'
            new_nb = '_nb_str = os.environ.get("NOTEBOOK_DIR", ".")\n    if _nb_str == ".": _nb_str = "/kaggle/working"\n    _nb = Path(_nb_str)\n    os.environ["NOTEBOOK_DIR"] = _nb_str'
            if old_nb in text:
                text = text.replace(old_nb, new_nb)
                
            if is_list:
                lines = text.split('\n')
                cell['source'] = [line + '\n' for line in lines[:-1]] + [lines[-1]] if lines[-1] else [line + '\n' for line in lines[:-1]]
            else:
                cell['source'] = text

# Find the sync block B cell and also make sure it has the /kaggle/working fix
for cell in cells:
    if cell['cell_type'] == 'code':
        source = cell['source']
        is_list = isinstance(source, list)
        text = "".join(source) if is_list else source
        if 'Fetching Block B data before Day 4' in text:
            old_base = 'base_dir = Path(os.environ.get(\'NOTEBOOK_DIR\', \'.\'))'
            new_base = '_nb_str = os.environ.get("NOTEBOOK_DIR", ".")\nif _nb_str == ".": _nb_str = "/kaggle/working"\nbase_dir = Path(_nb_str)\nos.environ["NOTEBOOK_DIR"] = _nb_str'
            if old_base in text:
                text = text.replace(old_base, new_base)
            if is_list:
                lines = text.split('\n')
                cell['source'] = [line + '\n' for line in lines[:-1]] + [lines[-1]] if lines[-1] else [line + '\n' for line in lines[:-1]]
            else:
                cell['source'] = text


with open('molab_run.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Targeted patch successful.")
