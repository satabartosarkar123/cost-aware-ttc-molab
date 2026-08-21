"""Scan for ALL UnboundLocalError risks, not just Cell 1 globals."""
import json
import ast

def scan_all():
    with open('molab_run.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # Get ALL global imports from unwrapped cells (1, 2, 3, 4)
    global_names = set()
    for idx in [1, 2, 3, 4]:
        cell = nb['cells'][idx]
        if cell['cell_type'] != 'code':
            continue
        src = "".join(cell['source'])
        if f'def execute_cell_' in src:
            continue
        try:
            tree = ast.parse(src)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        global_names.add(alias.asname or alias.name.split('.')[0])
                elif isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        global_names.add(alias.asname or alias.name)
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            global_names.add(target.id)
        except SyntaxError:
            pass
    
    print(f"All global names from unwrapped cells: {len(global_names)}")
    
    found_any = False
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] != 'code':
            continue
        src = "".join(cell['source'])
        if f'def execute_cell_{i}()' not in src:
            continue
        
        # Extract function body
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
                body_lines.append(line[4:] if line.startswith('    ') else line)
        
        body = '\n'.join(body_lines)
        try:
            tree = ast.parse(body)
        except SyntaxError:
            continue
        
        # Find all assigned names
        assigned = {}
        for node in ast.walk(tree):
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
        
        # Find first usage of each name
        used = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                if node.id not in used:
                    used[node.id] = node.lineno
        
        # Check for conflicts
        for name, assign_line in assigned.items():
            if name in used and used[name] < assign_line and name in global_names:
                print(f"  Cell {i}: '{name}' used on line {used[name]}, imported on line {assign_line}")
                found_any = True
    
    if not found_any:
        print("ALL CLEAR! No UnboundLocalError risks in any wrapped cell.")

if __name__ == "__main__":
    scan_all()
