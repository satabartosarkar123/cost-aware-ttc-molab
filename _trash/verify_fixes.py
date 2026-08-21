import json

nb = json.load(open('molab_run.ipynb', encoding='utf-8'))

git_cells = []
for i, c in enumerate(nb['cells']):
    if c['cell_type'] == 'code':
        source = ''.join(c['source'])
        if 'ghp_aG2QUWAK8q3' in source:
            git_cells.append(i)
            # Check key things
            has_safe_dir = "safe.directory" in source
            has_remote_remove = "git remote remove origin" in source
            has_no_spaces_ts = '%Y-%m-%d_%H-%M_UTC' in source
            has_old_spaces_ts = '%Y-%m-%d %H:%M UTC' in source
            has_systemExit = 'raise SystemExit(0)' in source
            print(f"Cell {i}:")
            print(f"  safe.directory fix: {'YES' if has_safe_dir else 'NO'}")
            print(f"  remote remove fix: {'YES' if has_remote_remove else 'NO'}")  
            print(f"  timestamp no-spaces: {'YES' if has_no_spaces_ts else 'NO'}")
            print(f"  OLD timestamp w/spaces: {'YES (BAD!)' if has_old_spaces_ts else 'NO (good)'}")
            print(f"  raise SystemExit: {'YES' if has_systemExit else 'NO'}")
            print()

print(f"Total cells with git: {len(git_cells)}")

# Also check the zip
import zipfile
zpath = '../Cost-Aware-Test-Time-upload.zip'
with zipfile.ZipFile(zpath) as z:
    with z.open('molab_run.ipynb') as f:
        znb = json.load(f)
    zgit = 0
    zfixed = 0
    for c in znb['cells']:
        if c['cell_type'] == 'code':
            src = ''.join(c['source'])
            if 'ghp_aG2QUWAK8q3' in src:
                zgit += 1
                if 'safe.directory' in src and 'git remote remove' in src:
                    zfixed += 1
    print(f"\nIN ZIP: {zgit} git cells, {zfixed} fixed")
    
    with z.open('gdrive_oauth.py') as f:
        gd = f.read().decode('utf-8')
    has_sa = 'service_account' in gd and 'InstalledAppFlow' not in gd
    has_auto_init = 'if not init(workspace)' in gd
    print(f"gdrive_oauth.py: SA-only={'YES' if has_sa else 'NO'}, auto-init={'YES' if has_auto_init else 'NO'}")
