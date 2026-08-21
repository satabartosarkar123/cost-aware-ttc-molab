import json
with open("molab_run.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)
with open("cell59.py", "w", encoding="utf-8") as f:
    f.write("".join(nb["cells"][59]["source"]))
