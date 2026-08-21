import json
import re

NB_PATH = "molab_run.ipynb"
with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]

# We will classify cells and rebuild the notebook from scratch.
imports_cell = None
hf_download_cell = None
ollama_setup_cell = None
smoke_tests = []
full_runs = []
day_runs = []

current_section = None

for c in cells:
    src = "".join(c.get("source", []))
    
    # Identify the 3 mandatory setup cells
    if c["cell_type"] == "code" and "CELL 0: ALL IMPORTS" in src:
        imports_cell = c
        continue
    if c["cell_type"] == "code" and "HF_REPO =" in src and "ZIP_NAME =" in src:
        hf_download_cell = c
        continue
    if c["cell_type"] == "code" and "OLLAMA_MODEL   = \"qwen2.5:3b\"" in src:
        ollama_setup_cell = c
        continue
        
    # Skip old markdown headers that we are replacing
    if c["cell_type"] == "markdown":
        if "Cost-Aware Test-Time Compute  Molab Notebook" in src: continue
        if "CELL 1" in src and "Download & Extract" in src: continue
        if "CELL 2" in src and "Install Ollama" in src: continue
        if "SMOKE TESTS" in src: current_section = "smoke"; continue
        if "FULL RUNS" in src: current_section = "full"; continue
        if "MONITORING" in src: current_section = "monitoring"; continue # We can drop monitoring if it's junk, or keep it. Let's drop it if it's unused.
        if "Day 1 Master Order" in src or "Block B" in src or "Day " in src: current_section = "day"; 

    # Assign remaining cells to their respective sections
    if current_section == "smoke":
        smoke_tests.append(c)
    elif current_section == "full":
        full_runs.append(c)
    elif current_section == "day":
        day_runs.append(c)
    else:
        # Before any section was defined (might be random markdown)
        pass

# Now rebuild the notebook structure
new_cells = []

# --- 1. MANDATORY SETUP ---
new_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# 1. MANDATORY GLOBAL SETUP\n",
        "> **Run these 3 cells in order on EVERY new instance.** They download the codebase from HuggingFace, install all dependencies, start the Ollama GPU server, and load global variables."
    ]
})
if hf_download_cell:
    new_cells.append(hf_download_cell)
if ollama_setup_cell:
    new_cells.append(ollama_setup_cell)
if imports_cell:
    # We run imports LAST in the setup phase because HF download and Ollama setup pip install the dependencies!
    new_cells.append(imports_cell)

# --- 2. PRE-DAY 1 (SMOKE TESTS & FULL RUNS) ---
new_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "---\n",
        "# 2. PRE-DAY 1 EXPERIMENTS\n",
        "> These are your historical smoke tests and Block A runs. You can skip these if you are doing Day 1+."
    ]
})
new_cells.extend(smoke_tests)
new_cells.extend(full_runs)

# --- 3. DAY 1 ONWARD ---
new_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "---\n",
        "# 3. DAY 1 ONWARD (BLOCK B)\n",
        "> Run these strictly in order, starting with Day 1 Statistics."
    ]
})
new_cells.extend(day_runs)

# Remove any lingering "HF sync warning" prints from the very first cell if it existed
for c in new_cells:
    if c["cell_type"] == "code":
        src = "".join(c["source"])
        # We ensure HF sync is only on experiments, not on the setup cells!
        if c in [imports_cell, hf_download_cell, ollama_setup_cell]:
            if "HF CONTINUOUS SYNC" in src:
                src = re.sub(r'# ── HF CONTINUOUS SYNC ──.*', '', src, flags=re.DOTALL)
                c["source"] = [l + "\n" for l in src.strip().split("\n")]

nb["cells"] = new_cells

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Notebook reorganized perfectly into {len(new_cells)} cells.")
