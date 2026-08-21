import json

def fix_notebook():
    with open('molab_run.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)

    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = cell['source']
            if isinstance(source, list):
                new_source = []
                for line in source:
                    # A line in Jupyter should just be a single line of text, ending with \n or not.
                    # If it has \n in the middle, it was corrupted!
                    # Example: 'print("
                    # All files present")\n'
                    # Actually, if the string itself literally contains a newline character,
                    # JSON parses it as a string with a newline.
                    # We can fix this by replacing all \n with \\n, except the one at the very end.
                    
                    if line.endswith('\n'):
                        fixed_line = line[:-1].replace('\n', '\\n') + '\n'
                    else:
                        fixed_line = line.replace('\n', '\\n')
                    new_source.append(fixed_line)
                cell['source'] = new_source
            else:
                # If it's a single string, we need to be careful.
                # But it's usually a list of strings.
                lines = source.split('\n')
                # Actually if it's a single string, it's just the whole cell text.
                pass

    with open('molab_run.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

fix_notebook()
print("Fixed newlines!")
