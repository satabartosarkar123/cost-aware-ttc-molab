"""
ONE-SHOT CLEAN: 
1. Scan every code cell, collect all import lines
2. Strip them from their original cells
3. Insert a new Cell 0 with all imports
4. Remove ALL git push cells and google drive sync code
5. Add HF sync block at end of every code cell from Day 1 onward
6. Add clear "HF DATA SYNCED" print
"""
import json, re

NB_PATH = "molab_run.ipynb"
with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

# ============================================================
# STEP 1: Collect ALL unique import lines from ALL code cells
# ============================================================
import_pattern = re.compile(r'^(import\s+.+|from\s+\S+\s+import\s+.+)$')
all_imports = set()

for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell.get("source", []))
    for line in src.split("\n"):
        stripped = line.strip()
        if import_pattern.match(stripped):
            # Keep local project imports (from core..., from strategies...) where they are
            if stripped.startswith("from core") or stripped.startswith("from strategies"):
                continue
            # Skip inline imports inside try/except or conditional blocks
            # We'll handle those by leaving indented imports alone
            if line.startswith("    ") or line.startswith("\t"):
                continue
            all_imports.add(stripped)

# Expand compound imports like "import subprocess, zipfile, os, sys"
expanded = set()
for imp in all_imports:
    if imp.startswith("import ") and "," in imp:
        # e.g. "import subprocess, zipfile, os, sys"
        parts = imp.replace("import ", "").split(",")
        for p in parts:
            p = p.strip()
            if p:
                expanded.add(f"import {p}")
    else:
        expanded.add(imp)

# Sort them nicely: stdlib first, then third-party
stdlib = sorted([i for i in expanded if not any(i.startswith(f"from {x}") or i.startswith(f"import {x}") 
    for x in ["huggingface", "gdown", "scipy", "numpy", "pandas", "matplotlib", "seaborn", 
              "tqdm", "pynvml", "psutil", "yaml", "tabulate", "reportlab", "fpdf"])])
thirdparty = sorted([i for i in expanded if i not in stdlib])

cell0_lines = ["# ── CELL 0: ALL IMPORTS ──\n"]
cell0_lines.append("# Standard library\n")
for imp in stdlib:
    cell0_lines.append(imp + "\n")
cell0_lines.append("\n# Third-party\n")
for imp in thirdparty:
    cell0_lines.append(imp + "\n")
cell0_lines.append("\nprint('All imports loaded.')\n")

cell0 = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": cell0_lines
}

# ============================================================
# STEP 2: Strip top-level import lines from ALL other code cells
# ============================================================
for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    new_lines = []
    src_lines = "".join(cell.get("source", [])).split("\n")
    for line in src_lines:
        stripped = line.strip()
        # Remove top-level (non-indented) import lines that aren't project-local
        if import_pattern.match(stripped) and not line.startswith("    ") and not line.startswith("\t"):
            if stripped.startswith("from core") or stripped.startswith("from strategies"):
                new_lines.append(line)  # keep project imports
            else:
                continue  # strip it
        else:
            new_lines.append(line)
    
    # Clean up leading blank lines
    while new_lines and not new_lines[0].strip():
        new_lines.pop(0)
    
    cell["source"] = [l + "\n" for l in new_lines]
    if cell["source"]:
        cell["source"][-1] = cell["source"][-1].rstrip("\n")

# ============================================================
# STEP 3: Remove ALL git push cells and google drive sync code
# ============================================================
cells_to_remove = []
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        src = "".join(cell.get("source", []))
        # Remove git push cells
        if "GIT PUSH" in src and "git add" in src and len(src) < 2000:
            cells_to_remove.append(i)
            continue
    if cell["cell_type"] == "markdown":
        src = "".join(cell.get("source", []))
        if "PUSH RESULTS TO GITHUB" in src:
            cells_to_remove.append(i)
            continue

# Remove git push cell that follows the PUSH RESULTS TO GITHUB markdown
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        src = "".join(cell.get("source", []))
        if "git add" in src and "git commit" in src and "git push" in src:
            if i not in cells_to_remove:
                cells_to_remove.append(i)

# Remove in reverse order so indices don't shift
for idx in sorted(cells_to_remove, reverse=True):
    nb["cells"].pop(idx)

# Now strip any remaining google drive / rclone / git references from code cells
for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell.get("source", []))
    
    # Remove rclone/drive sync blocks from the Ollama setup cell
    src = re.sub(r"if not shutil\.which\(\"rclone\"\).*?print\(f\"  Sync error: \{_e\}\"\)", 
                 'print("[6/6] Storage: HuggingFace Hub ONLY")', src, flags=re.DOTALL)
    
    # Remove DRIVE_SYNC variable references
    src = src.replace('DRIVE_SYNC = False', '')
    src = src.replace("DRIVE_SYNC = True", '')
    src = re.sub(r"print\(f\"  Drive sync:.*?\"\)", '', src)
    src = src.replace("if DRIVE_SYNC else 'DISABLED'", "'HuggingFace Hub'")
    
    # Remove gdrive_oauth references  
    src = re.sub(r"try:\s*spec = importlib.*?gdrive_oauth.*?except.*?print.*?\n", "", src, flags=re.DOTALL)
    
    cell["source"] = [l + "\n" for l in src.split("\n")]
    if cell["source"]:
        cell["source"][-1] = cell["source"][-1].rstrip("\n")

# ============================================================
# STEP 4: Add HF sync block at end of every code cell from Day 1 onward
# ============================================================
HF_SYNC = """
# ── HF CONTINUOUS SYNC ──
try:
    import os
    from pathlib import Path
    from huggingface_hub import HfApi
    _nb = Path(os.environ.get("NOTEBOOK_DIR", "."))
    _res = _nb / "results"
    if _res.exists():
        _api = HfApi(token="REDACTED")
        _api.upload_folder(
            folder_path=str(_res),
            path_in_repo="results_sync",
            repo_id="Satabarto/Molab_Checkpoints_Cost_AWARE",
            repo_type="dataset",
        )
        print("\\n" + "="*50)
        print("  HF DATA SYNCED SUCCESSFULLY")
        print("="*50)
    else:
        print("No results dir yet - skipping HF sync.")
except Exception as e:
    print(f"HF sync warning (non-fatal): {e}")
"""

# Find Day 1 cell index
day1_idx = None
for i, cell in enumerate(nb["cells"]):
    src = "".join(cell.get("source", []))
    if "Day 1 Master Order" in src or "SECTION 1" in src and "LOAD" in src:
        day1_idx = i
        break

if day1_idx is None:
    # Fallback: find the cell with Wilson CI or block_a_logs
    for i, cell in enumerate(nb["cells"]):
        src = "".join(cell.get("source", []))
        if "wilson_ci" in src or "block_a_logs" in src:
            day1_idx = i
            break

if day1_idx is not None:
    for i in range(day1_idx, len(nb["cells"])):
        cell = nb["cells"][i]
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell.get("source", []))
        # Don't add to tiny cells or cells that already have it
        if "HF CONTINUOUS SYNC" in src:
            continue
        if len(src.strip()) < 20:
            continue
        
        src = src.rstrip() + "\n" + HF_SYNC
        cell["source"] = [l + "\n" for l in src.split("\n")]
        cell["source"][-1] = cell["source"][-1].rstrip("\n")

# ============================================================
# STEP 5: Insert Cell 0 at the beginning
# ============================================================
nb["cells"].insert(0, cell0)

# ============================================================
# SAVE
# ============================================================
with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"DONE. Notebook now has {len(nb['cells'])} cells.")
print(f"Cell 0 = ALL IMPORTS ({len(expanded)} import statements)")
print(f"Day 1 found at index {day1_idx}")
print(f"Removed {len(cells_to_remove)} git push cells")
print("HF sync added to all cells from Day 1 onward")
print("All Google Drive / rclone / git push code stripped")
