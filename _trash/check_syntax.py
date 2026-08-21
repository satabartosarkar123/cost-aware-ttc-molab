import json
import ast

with open("molab_run.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        src = "".join(cell["source"])
        try:
            ast.parse(src)
        except SyntaxError as e:
            print(f"SyntaxError in Cell {i}:\n{e}\n---\n{src[:200]}...")
