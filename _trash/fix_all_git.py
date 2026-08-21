import json
from pathlib import Path

nb_path = Path('molab_run.ipynb')
nb = json.load(nb_path.open(encoding='utf-8'))

old_git_start = '_t = "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"'

new_git_block = '''_t = "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"
_repo = "https://github.com/satabartosarkar123/cost-aware-ttc-molab.git"
_g = Path(os.environ.get("NOTEBOOK_DIR", "."))
_a = _repo.replace("https://", "https://" + _t + "@")

subprocess.run("git config --global --add safe.directory '*'", shell=True, cwd=str(_g))

if not (_g / ".git").exists():
    subprocess.run("git init -b main", shell=True, cwd=str(_g))
    subprocess.run("git config user.name Molab", shell=True, cwd=str(_g))
    subprocess.run("git config user.email molab@run.local", shell=True, cwd=str(_g))

subprocess.run("git remote remove origin", shell=True, cwd=str(_g), stderr=subprocess.DEVNULL)
subprocess.run(f"git remote add origin {_a}", shell=True, cwd=str(_g))

subprocess.run("git add .", shell=True, cwd=str(_g))
_ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M_UTC")
subprocess.run(f'git commit -m "ckpt-{_ts}"', shell=True, cwd=str(_g))

r2 = subprocess.run("git push origin main", shell=True, cwd=str(_g), capture_output=True, text=True)
if r2.returncode != 0:
    subprocess.run("git push -u origin main", shell=True, cwd=str(_g))
print(f"Git push done: {_ts}")'''

changed_count = 0
for c in nb['cells']:
    if c['cell_type'] == 'code':
        source = ''.join(c['source'])
        if old_git_start in source:
            # We replace everything from old_git_start to the end of the source with the new_git_block
            start_idx = source.find(old_git_start)
            new_source = source[:start_idx] + new_git_block
            if new_source != source:
                c['source'] = [new_source]
                changed_count += 1

if changed_count > 0:
    nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
    print(f"Patched {changed_count} cells in molab_run.ipynb")
else:
    print("No cells needed patching.")
