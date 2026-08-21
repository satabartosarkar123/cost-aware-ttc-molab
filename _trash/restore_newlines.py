import json

def fix_jupyter_lines():
    with open('molab_run.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)

    def escape_newlines_in_strings(text):
        in_string = False
        string_char = ''
        escape_next = False
        result = []
        for c in text:
            if escape_next:
                result.append(c)
                escape_next = False
                continue

            if c == '\\':
                escape_next = True
                result.append(c)
                continue

            if c in ("'", '"'):
                if not in_string:
                    in_string = True
                    string_char = c
                elif string_char == c:
                    in_string = False
                result.append(c)
            elif c == '\n':
                if in_string:
                    result.append('\\n')
                else:
                    result.append(c)
            else:
                result.append(c)
        return "".join(result)

    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = cell['source']
            if isinstance(source, list):
                new_source = []
                for line in source:
                    # Parse line to escape \n inside quotes
                    fixed = escape_newlines_in_strings(line)
                    new_source.append(fixed)
                cell['source'] = new_source
            else:
                cell['source'] = escape_newlines_in_strings(source)

    with open('molab_run.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

fix_jupyter_lines()
print("Fixed newlines safely!")
