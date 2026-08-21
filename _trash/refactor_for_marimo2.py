import json
import re
import ast

def refactor_notebook():
    # First, restore to the original version I had right before I refactored it
    # Luckily I can just fetch it from the zip file I created!
    # import zipfile
    # with zipfile.ZipFile("../Cost-Aware-Test-Time-upload.zip", "r") as z:
        # z.extract("molab_run.ipynb", ".")

    with open('molab_run.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)

    all_imports = set()
    
    # Pass 1: Extract all imports and replace with 'pass' to preserve indentation blocks
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] != 'code':
            continue
            
        new_source = []
        source_lines = cell['source'] if isinstance(cell['source'], list) else cell['source'].splitlines(keepends=True)
        for line in source_lines:
            stripped = line.strip()
            # If it's an import statement, add to set, but replace with 'pass'
            if re.match(r'^(import\s+|from\s+[\w\.]+\s+import\s+)', stripped):
                # Multiple imports on one line? Split by semicolon if needed
                for stmt in stripped.split(';'):
                    stmt = stmt.strip()
                    if stmt.startswith('import ') or stmt.startswith('from '):
                        all_imports.add(stmt)
                
                # Replace the line with 'pass' maintaining its original indentation
                indent = line[:len(line) - len(line.lstrip())]
                new_source.append(indent + "pass\n")
            else:
                new_source.append(line)
        cell['source'] = new_source

    # Pass 2: Deduplicate imports and organize them
    sorted_imports = sorted(list(all_imports))
    import_block = ["# ── ALL NOTEBOOK IMPORTS (DEDUPLICATED FOR MARIMO) ──\n"]
    for imp in sorted_imports:
        import_block.append(imp + "\n")
        
    # Put all imports into the first code cell
    first_code_idx = next(i for i, c in enumerate(nb['cells']) if c['cell_type'] == 'code')
    nb['cells'][first_code_idx]['source'] = import_block

    # Pass 3: Wrap all other code cells in a subroutine
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] != 'code':
            continue
        if i == first_code_idx:
            continue
            
        # Skip empty cells
        src = "".join(cell['source']).strip()
        if not src:
            continue
            
        # Indent everything
        indented_source = []
        if isinstance(cell['source'], list):
            lines = []
            for s in cell['source']:
                lines.extend(s.splitlines(keepends=True))
        else:
            lines = cell['source'].splitlines(keepends=True)
            
        for line in lines:
            indented_source.append("    " + line if line.strip() else line)
            
        wrapper_start = [f"def execute_cell_{i}():\n"]
        wrapper_end = [f"\nexecute_cell_{i}()\n"]
        
        cell['source'] = wrapper_start + indented_source + wrapper_end

    with open('molab_run.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        
    print("Notebook refactored for Marimo!")
    
    # Run syntax check
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            src = "".join(cell['source'])
            try:
                ast.parse(src)
            except SyntaxError as e:
                print(f"SyntaxError in Cell {i}:\n{e}")
                import sys; sys.exit(1)

if __name__ == "__main__":
    refactor_notebook()
