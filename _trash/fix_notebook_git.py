import json
from pathlib import Path

# Fix gdrive_oauth.py
gd_path = Path('gdrive_oauth.py')
gd_content = gd_path.read_text(encoding='utf-8')
old_gd = '''def push_results_full(workspace: Path):
    """
    Zip all result/checkpoint dirs and upload to Drive.
    Called after each run cell completes.
    """
    if _svc is None:
        log.warning("Drive not initialised -- skipping push")
        return'''
new_gd = '''def push_results_full(workspace: Path):
    """
    Zip all result/checkpoint dirs and upload to Drive.
    Called after each run cell completes.
    """
    global _svc
    if _svc is None:
        if not init(workspace):
            log.warning("Drive not initialised -- skipping push")
            return'''
if old_gd in gd_content:
    gd_content = gd_content.replace(old_gd, new_gd)
    gd_path.write_text(gd_content, encoding='utf-8')
    print('Patched gdrive_oauth.py')
else:
    print('Could not find push_results_full string in gdrive_oauth.py')

# Fix molab_run.ipynb
nb_path = Path('molab_run.ipynb')
nb = json.load(nb_path.open(encoding='utf-8'))
changed = False
for c in nb['cells']:
    if c['cell_type'] == 'code':
        source_lines = c['source']
        source = ''.join(source_lines)
        if 'subprocess.run("git add ."' in source:
            # We insert the safe.directory command before git add .
            if 'safe.directory' not in source:
                new_source = source.replace('subprocess.run("git add .",', 'subprocess.run("git config --global --add safe.directory \'*\'", shell=True, cwd=str(_g))\nsubprocess.run("git add .",')
                if new_source != source:
                    c['source'] = [new_source]
                    changed = True

if changed:
    nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
    print('Patched molab_run.ipynb')
else:
    print('No changes needed in molab_run.ipynb')
