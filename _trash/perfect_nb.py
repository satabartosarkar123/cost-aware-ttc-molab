import json
import re
from pathlib import Path

nb_path = Path(r"c:\Users\USER\Cost-Aware-Test-Time\Cost-Aware-Test-time-molab\molab_run.ipynb")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

def wrap_in_function(source, func_name):
    # Wrap the entire source code in a private function to avoid Marimo variable redefine errors
    lines = source.split('\n')
    indented = ["    " + line for line in lines]
    return f"def {func_name}():\n" + "\n".join(indented) + f"\n\n{func_name}()"

for i, cell in enumerate(nb.get("cells", [])):
    if cell.get("cell_type") != "code":
        continue
        
    source = "".join(cell["source"])
    
    # 1. Drive Extraction Fix in Cell 1 (and patch parsers.py)
    if 'z.extractall(".")' in source and 'GDRIVE_FILE_ID' in source:
        new_source = source.replace('z.extractall(".")\n    os.environ["NOTEBOOK_DIR"] = str(Path(".").resolve())', 
'''z.extractall(".")
    
    # Locate actual extracted folder
    _base_dir = Path(".").resolve()
    _notebook_dir = _base_dir
    for _d in _base_dir.iterdir():
        if _d.is_dir() and (_d / "rq2_part1").exists():
            _notebook_dir = _d
            break
            
    os.environ["NOTEBOOK_DIR"] = str(_notebook_dir)
    os.chdir(_notebook_dir)
    
    # Patch parsers.py to fix the AQUA parser case bug
    _parsers_path = _notebook_dir / "ttc-frugalreason-poc/experiment_fr/core/parsers.py"
    if _parsers_path.exists():
        _content = _parsers_path.read_text(encoding="utf-8")
        _content = _content.replace("match.group(1).lower()", "match.group(1).upper()")
        _content = _content.replace("matches[-1].lower()", "matches[-1].upper()")
        _parsers_path.write_text(_content, encoding="utf-8")
''')
        source = new_source

    # 2. Fix the "except Exception:" blocks to use private traceback
    if "except Exception:" in source:
        source = re.sub(r'except Exception:.*?traceback\.print_exc\(\)', 'except Exception:\\n        import traceback as _tb; _tb.print_exc()', source, flags=re.DOTALL)

    # 3. Wrap Git Push cells in a function
    if "GIT PUSH" in source and "GITHUB_TOKEN" in source:
        if "def _git_push" not in source:
            source = wrap_in_function(source, f"_git_push_{i}")

    # 4. Wrap "Run" cells in a function
    if "runpy.run_path" in source and "def _run_" not in source:
        source = wrap_in_function(source, f"_run_cell_{i}")

    # Pack back into lines
    lines = [line + "\n" for line in source.split("\n")]
    if lines:
        lines[-1] = lines[-1][:-1]
    cell["source"] = lines

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Notebook successfully transformed to be 100% Marimo-safe.")
