import json

NB_PATH = "molab_run.ipynb"
with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Locate the ALL IMPORTS cell
imports_cell_idx = None
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] == "code" and "CELL 0: ALL IMPORTS" in "".join(c.get("source", [])):
        imports_cell_idx = i
        break

if imports_cell_idx is not None:
    imports_cell = nb["cells"].pop(imports_cell_idx)
    src = "".join(imports_cell["source"])
    
    # Split into stdlib and thirdparty
    lines = src.split("\n")
    stdlib = []
    thirdparty = []
    
    is_thirdparty = False
    for line in lines:
        if "# Third-party" in line:
            is_thirdparty = True
            continue
        if line.startswith("import ") or line.startswith("from "):
            if is_thirdparty:
                thirdparty.append(line)
            else:
                stdlib.append(line)

    cell_0_stdlib = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# ── CELL 0: STANDARD Python IMPORTS ──\n",
            "import os, sys, json, time, subprocess, zipfile, shutil, sqlite3, importlib.util\n",
            "from pathlib import Path\n",
            "print('Standard Python libraries loaded.')"
        ]
    }
    
    cell_3_thirdparty = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# ── CELL 3: THIRD-PARTY IMPORTS ──\n",
            "# Run this AFTER Cell 1 and Cell 2 have installed the pip packages.\n"
        ] + [l + "\n" for l in thirdparty] + ["print('Third-party libraries loaded.')"]
    }

    # Now find where to insert them.
    # We want cell_0_stdlib to be the absolute FIRST code cell.
    # We want cell_3_thirdparty to be AFTER the Ollama Setup cell.
    
    first_code_idx = None
    ollama_idx = None
    
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] == "code" and first_code_idx is None:
            first_code_idx = i
        if c["cell_type"] == "code" and "OLLAMA_MODEL" in "".join(c.get("source", [])):
            ollama_idx = i
            
    if ollama_idx is not None:
        nb["cells"].insert(ollama_idx + 1, cell_3_thirdparty)
    if first_code_idx is not None:
        nb["cells"].insert(first_code_idx, cell_0_stdlib)

# Also, ensure that the HF download cell has its local `huggingface_hub` import intact.
# Wait, my final_clean.py didn't strip indented imports, but let's be double sure.
for c in nb["cells"]:
    if c["cell_type"] == "code" and "HF_REPO =" in "".join(c.get("source", [])):
        src = "".join(c["source"])
        if "from huggingface_hub import hf_hub_download" not in src:
            src = src.replace("subprocess.run(\"pip install -q huggingface_hub\", shell=True)", 
                              "subprocess.run(\"pip install -q huggingface_hub\", shell=True)\n    from huggingface_hub import hf_hub_download")
        c["source"] = [l + "\n" for l in src.split("\n")]
        c["source"][-1] = c["source"][-1].rstrip("\n")

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Imports correctly split and ordered to prevent NameError!")
