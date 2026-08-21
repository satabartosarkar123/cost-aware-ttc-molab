"""
Scan all wrapped cells for UnboundLocalError risks.
For each execute_cell_X() function, find names that are:
  1. Used BEFORE being assigned/imported within the function, AND
  2. Assigned/imported LATER in the function (making them local)

Then fix by hoisting all imports to the top of each function.
"""
import json
import ast
import sys

def get_cell1_imports(nb):
    """Extract all names imported in Cell 1 (the global imports cell)."""
    src = "".join(nb['cells'][1]['source'])
    tree = ast.parse(src)
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname or alias.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(alias.asname or alias.name)
    return names

def analyze_cell(cell_idx, src):
    """Find variables that would cause UnboundLocalError when wrapped."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return [], []
    
    # Find all names that are assigned/imported (making them local)
    assigned = {}  # name -> first assignment line
    used_before = {}  # name -> first usage line
    
    for node in ast.walk(tree):
        # Track imports (they count as assignments)
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.asname or alias.name.split('.')[0]
                if name not in assigned:
                    assigned[name] = node.lineno
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                name = alias.asname or alias.name
                if name not in assigned:
                    assigned[name] = node.lineno
        # Track regular assignments
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    if target.id not in assigned:
                        assigned[target.id] = node.lineno
        # Track Name usage (loads)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id not in used_before:
                used_before[node.id] = node.lineno

    # Find conflicts: used before assigned within the same function
    problems = []
    import_lines = []
    for name, assign_line in assigned.items():
        if name in used_before and used_before[name] < assign_line:
            problems.append((name, used_before[name], assign_line))
    
    return problems, assigned

def scan_notebook():
    with open('molab_run.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    cell1_imports = get_cell1_imports(nb)
    print(f"Cell 1 global imports: {sorted(cell1_imports)}\n")
    
    all_problems = {}
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] != 'code':
            continue
        src = "".join(cell['source'])
        if f'def execute_cell_{i}()' not in src:
            continue
        
        # Extract just the function body (skip def line and call line)
        lines = src.split('\n')
        body_lines = []
        in_func = False
        for line in lines:
            if line.startswith(f'def execute_cell_{i}():'):
                in_func = True
                continue
            if line.startswith(f'execute_cell_{i}()'):
                break
            if in_func:
                # Remove one level of indentation
                body_lines.append(line[4:] if line.startswith('    ') else line)
        
        body = '\n'.join(body_lines)
        problems, assigned = analyze_cell(i, body)
        
        if problems:
            # Filter to only problems where the name is a Cell 1 global import
            real_problems = [(n, u, a) for n, u, a in problems if n in cell1_imports]
            if real_problems:
                all_problems[i] = real_problems
                print(f"Cell {i}: PROBLEMS FOUND")
                for name, use_line, assign_line in real_problems:
                    print(f"  {name}: used on line {use_line}, but imported/assigned on line {assign_line}")
    
    if not all_problems:
        print("No UnboundLocalError risks found!")
    else:
        print(f"\nTotal: {len(all_problems)} cells with problems")
    
    return all_problems

if __name__ == "__main__":
    scan_notebook()
