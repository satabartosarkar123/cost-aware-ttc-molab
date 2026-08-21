import json
from pathlib import Path

nb_path = Path('molab_run.ipynb')
nb = json.load(nb_path.open(encoding='utf-8'))

changed = 0
for i, c in enumerate(nb['cells']):
    if c['cell_type'] != 'code':
        continue
    source = ''.join(c['source'])
    modified = False
    
    # FIX 1: Remove raise SystemExit(0) — it kills Marimo's cell chain
    if 'raise SystemExit(0)' in source:
        # Replace the whole "if done, skip" block with a simple print+pass
        # The pattern is: if _flag.exists(): print("ALREADY DONE: ..."); raise SystemExit(0)
        import re
        source = re.sub(
            r'if _flag\.exists\(\): print\("ALREADY DONE: [^"]*"\); raise SystemExit\(0\)',
            lambda m: m.group(0).replace('raise SystemExit(0)', 'import sys; sys.exit(0)').replace('raise SystemExit(0)', 'pass'),
            source
        )
        # Actually, just replace ALL raise SystemExit(0) with a no-op skip
        source = source.replace(
            'raise SystemExit(0)',
            'pass  # skip re-run'
        )
        modified = True
    
    # FIX 2: Remove the entire git push block — token is dead, it just errors
    if 'ghp_aG2QUWAK8q3' in source:
        # Find where git block starts and cut it
        git_marker = '_t = "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"'
        idx = source.find(git_marker)
        if idx > 0:
            # Keep everything before the git block, add a simple print
            source = source[:idx].rstrip() + '\nprint("Run complete.")\n'
            modified = True
    
    if modified:
        c['source'] = [source]
        changed += 1
        print(f"Fixed cell {i}")

nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
print(f"\nTotal cells fixed: {changed}")
