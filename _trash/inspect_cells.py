import json, sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

with open("molab_run.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for i, c in enumerate(nb["cells"]):
    src = "".join(c.get("source", []))
    first_line = src.split("\n")[0][:90]
    # Strip non-ascii
    first_line = first_line.encode('ascii', 'replace').decode('ascii')
    print(f"Cell {i:2d} ({c['cell_type']:8s}): {first_line}")

print(f"\nTotal cells: {len(nb['cells'])}")
