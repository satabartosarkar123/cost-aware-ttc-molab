import json

NB_PATH = "molab_run.ipynb"
with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Find the download cell (the one with GDRIVE_FILE_ID)
download_idx = None
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        src = "".join(cell.get("source", []))
        if "GDRIVE_FILE_ID" in src or "ZIP_NAME" in src:
            download_idx = i
            print(f"Found download cell at index {i}")
            print("---CURRENT SOURCE---")
            print(src[:500])
            print("---END---")
            break

if download_idx is None:
    print("ERROR: Could not find download cell!")
