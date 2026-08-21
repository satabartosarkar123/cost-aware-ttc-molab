import json
import os

files_to_patch = ['molab_run.ipynb', 'broken_test/molab_run.ipynb']

for fname in files_to_patch:
    if not os.path.exists(fname):
        continue
    
    with open(fname, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    for cell in nb.get('cells', []):
        if cell['cell_type'] == 'code':
            source = cell['source']
            for i in range(len(source)):
                if 'log_dir = _nb / "results" / log_subdir' in source[i] and '# Find FR log' in source[i+1]:
                    # Replace lines
                    source[i] = '            fr_path = _nb / log_subdir / f"{prefix}{ds}_frugal_reason_v3.jsonl"\\n'
                    source[i+1] = '            if not fr_path.exists():\\n'
                    source[i+2] = '                fr_path = _nb / "results" / log_subdir / f"{prefix}{ds}_frugal_reason_v3.jsonl"\\n'
                    
                    # Need to add new lines
                    to_insert = [
                        '            if not fr_path.exists():\\n',
                        '                # Try experiment_fr path for Block A\\n',
                        '                fr_path = _nb / "ttc-frugalreason-poc" / "experiment_fr" / "results" / log_subdir / f"{prefix}{ds}_frugal_reason_v3.jsonl"\\n',
                        '            if not fr_path.exists():\\n',
                        '                if "qwen2.5:1.5b" in model_name:\\n',
                        '                    fr_path = _nb / "block_b_qwen15b" / "block_b_logs" / f"{prefix}{ds}_frugal_reason_v3.jsonl"\\n',
                        '                elif "llama3.2:3b" in model_name:\\n',
                        '                    fr_path = _nb / "block_b_llama32" / "block_b_logs" / f"{prefix}{ds}_frugal_reason_v3.jsonl"\\n',
                        '            if not fr_path.exists():\\n',
                        '                print(f"  SKIP {ds}: FR log not found")\\n',
                        '                continue\\n'
                    ]
                    
                    # Remove the old if not exists lines
                    del source[i+3:i+9]
                    
                    # Insert the new ones
                    for j, line in enumerate(to_insert):
                        source.insert(i+3+j, line)
                    break

    with open(fname, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    
    print(f"Patched {fname}")
