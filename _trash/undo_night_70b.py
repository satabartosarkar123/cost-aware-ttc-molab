import json, os

NB_PATH = "molab_run.ipynb"
with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Find where the Night Run starts and remove it
new_cells = []
for cell in nb["cells"]:
    src = "".join(cell.get("source", []))
    if "Night Run — 70B-A Part 1" in src or "Night Run - 70B-A Part 1" in src:
        print("Found and removed 70B Night Run cell.")
        continue # Skip this and subsequent (wait, actually, just skip the matching cells)
    new_cells.append(cell)

nb["cells"] = new_cells

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Done. Notebook now has {len(nb['cells'])} cells.")
