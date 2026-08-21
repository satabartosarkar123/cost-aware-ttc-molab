import json, re

NB_PATH = "molab_run.ipynb"
with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

# We want to remove the specific zip creation blocks from the notebook cells
# since the new HF CONTINUOUS SYNC block perfectly handles everything via upload_folder.
for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        src = "".join(cell["source"])
        
        # Remove zip push blocks from Day 2, Day 3, Day 9, and Summary
        # We look for the try block that imports zipfile and uses HfApi
        src = re.sub(r'# ── HF push ──────────────────────────────────────────────────────\n.*?except Exception as e:.*?$', '', src, flags=re.DOTALL|re.MULTILINE)
        src = re.sub(r'# HF push\ntry:\n.*?except Exception as e:.*?$', '', src, flags=re.DOTALL|re.MULTILINE)
        src = re.sub(r'# Final push to HF\ntry:\n.*?except Exception as e:.*?$', '', src, flags=re.DOTALL|re.MULTILINE)
        
        lines = src.split("\n")
        # clean up trailing newlines
        while lines and not lines[-1].strip() and not lines[-1].startswith("# ── HF CONTINUOUS SYNC"):
            lines.pop()
            
        cell["source"] = [line + "\n" for line in lines]
        if cell["source"]:
            cell["source"][-1] = cell["source"][-1].rstrip("\n")

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Removed redundant zip uploads from notebook!")
