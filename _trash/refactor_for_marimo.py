import json
import re

def refactor_notebook():
    with open('molab_run.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)

    all_imports = set()
    
    # Pass 1: Extract all imports
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] != 'code':
            continue
            
        new_source = []
        for line in cell['source']:
            stripped = line.strip()
            # If it's an import statement, add to set, but don't keep in cell
            # Be careful not to match things like "import_feature = True"
            if re.match(r'^(import\s+|from\s+[\w\.]+\s+import\s+)', stripped):
                # Multiple imports on one line? Split by semicolon if needed
                for stmt in stripped.split(';'):
                    stmt = stmt.strip()
                    if stmt.startswith('import ') or stmt.startswith('from '):
                        all_imports.add(stmt)
            else:
                new_source.append(line)
        cell['source'] = new_source

    # Pass 2: Deduplicate imports and organize them
    sorted_imports = sorted(list(all_imports))
    import_block = ["# ── ALL NOTEBOOK IMPORTS (DEDUPLICATED FOR MARIMO) ──\n"]
    for imp in sorted_imports:
        import_block.append(imp + "\n")
        
    # Put all imports into Cell 1 (the first code cell)
    # Ensure Cell 1 is the import cell
    assert nb['cells'][1]['cell_type'] == 'code'
    nb['cells'][1]['source'] = import_block

    # Pass 3: Wrap all other code cells in a subroutine
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] != 'code':
            continue
        if i == 1:
            # Skip the import cell
            continue
            
        # Skip empty cells
        if not "".join(cell['source']).strip():
            continue
            
        # Indent everything
        indented_source = []
        for line in cell['source']:
            indented_source.append("    " + line if line.strip() else line)
            
        # Wrap in subroutine
        wrapper_start = [f"def execute_cell_{i}():\n"]
        # Make os and sys available if they were shadowed? 
        # Actually imports are global, so the subroutine can access them.
        
        wrapper_end = [f"\nexecute_cell_{i}()\n"]
        
        cell['source'] = wrapper_start + indented_source + wrapper_end

    with open('molab_run.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        
    print("Notebook refactored for Marimo!")

if __name__ == "__main__":
    refactor_notebook()
