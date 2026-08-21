import json, re

NB_PATH = "molab_run.ipynb"
with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

# 1. Create Cell 0 with all standard imports
GLOBAL_IMPORTS = """# ── GLOBAL IMPORTS ──
import os
import sys
import json
import time
import subprocess
import zipfile
import shutil
import sqlite3
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, binomtest
from huggingface_hub import hf_hub_download, HfApi
"""

new_cell_0 = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [line + "\n" for line in GLOBAL_IMPORTS.split("\n")]
}
new_cell_0["source"][-1] = new_cell_0["source"][-1].rstrip("\n")
nb["cells"].insert(0, new_cell_0)

# Now, we need to find the Ollama GPU Setup cell and pull out the qwen fetching.
# Because we inserted a cell, all indices are shifted by +1.
# The original Ollama Setup was cell index 5, so now it should be index 6.

setup_cell_idx = None
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code" and "OLLAMA_MODEL   = \"qwen2.5:3b\"" in "".join(cell.get("source", [])):
        setup_cell_idx = i
        break

if setup_cell_idx is not None:
    src = "".join(nb["cells"][setup_cell_idx]["source"])
    
    # We will remove the exact pull logic block from the setup cell
    pull_regex = r'tags_out = subprocess\.run\("curl -sf http://localhost:11434/api/tags".*?    print\(f"  \{OLLAMA_MODEL\} ready"\)\n'
    src = re.sub(pull_regex, '', src, flags=re.DOTALL)
    
    nb["cells"][setup_cell_idx]["source"] = [l + "\n" for l in src.split("\n")]
    nb["cells"][setup_cell_idx]["source"][-1] = nb["cells"][setup_cell_idx]["source"][-1].rstrip("\n")
    
    # 3. Create a new cell for Qwen 3B fetching
    qwen_pull_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# ── PULL DEFAULT MODEL ──\n",
            "import subprocess\n",
            "print(\"Pulling qwen2.5:3b...\")\n",
            "subprocess.run(['ollama', 'pull', 'qwen2.5:3b'])\n",
            "print(\"Model ready.\")"
        ]
    }
    
    # Insert it right after the HF download cell.
    # Original HF download was Cell 2 (index 2). With Cell 0 inserted, it is index 3.
    # So we insert the new pull cell at index 4 (between HF download and Ollama Setup).
    hf_cell_idx = None
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code" and "ZIP_NAME = \"Cost-Aware-Test-Time-upload.zip\"" in "".join(cell.get("source", [])):
            hf_cell_idx = i
            break
            
    if hf_cell_idx is not None:
        nb["cells"].insert(hf_cell_idx + 1, qwen_pull_cell)

# 4. Remove all standard imports from the rest of the cells to keep them "chill"
# We only strip exact full line imports so we don't break inline logic.
remove_imports = ["import os", "import sys", "import json", "import time", 
                  "import subprocess", "import zipfile", "import shutil", "import sqlite3", 
                  "import numpy as np", "import pandas as pd", "from pathlib import Path"]

for i in range(1, len(nb["cells"])):
    if nb["cells"][i]["cell_type"] == "code":
        new_source = []
        for line in nb["cells"][i]["source"]:
            clean_line = line.strip()
            # If the line is an import statement that we moved to Cell 0, skip it
            if any(clean_line == imp for imp in remove_imports) or clean_line.startswith("import subprocess, os, sys"):
                continue
            if clean_line.startswith("import subprocess, zipfile"):
                continue
            if clean_line.startswith("import subprocess, os, sys, time"):
                continue
            new_source.append(line)
        nb["cells"][i]["source"] = new_source

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Applied exact structural fixes as requested!")
