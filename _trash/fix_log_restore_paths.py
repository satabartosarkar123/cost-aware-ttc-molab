"""
Fix the Log Restore extraction path to use base_dir instead of '.'.
Also fix the results_sync extraction to use base_dir.
"""
import json
import re

def fix_log_restore():
    with open('molab_run.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)

    cell2 = nb['cells'][2]
    src = "".join(cell2['source'])

    # Fix 1: os.makedirs("results") -> os.makedirs(str(base_dir / "results"))
    src = src.replace(
        '                os.makedirs("results", exist_ok=True)',
        '                os.makedirs(str(base_dir / "results"), exist_ok=True)'
    )

    # Fix 2: snapshot_download local_dir="." -> local_dir=str(base_dir)
    src = src.replace(
        '                    local_dir=".",',
        '                    local_dir=str(base_dir),'
    )

    # Fix 3: os.path.exists("results_sync") -> os.path.exists(str(base_dir / "results_sync"))
    src = src.replace(
        '                if os.path.exists("results_sync"):',
        '                if os.path.exists(str(base_dir / "results_sync")):'
    )

    # Fix 4: os.listdir("results_sync") -> os.listdir(str(base_dir / "results_sync"))
    src = src.replace(
        '                    for item in os.listdir("results_sync"):',
        '                    for item in os.listdir(str(base_dir / "results_sync")):'
    )

    # Fix 5: os.path.join("results_sync", item) -> os.path.join(str(base_dir / "results_sync"), item)
    src = src.replace(
        '                        src = os.path.join("results_sync", item)',
        '                        src = os.path.join(str(base_dir / "results_sync"), item)'
    )

    # Fix 6: os.path.join("results", item) -> os.path.join(str(base_dir / "results"), item)
    src = src.replace(
        '                        dst = os.path.join("results", item)',
        '                        dst = os.path.join(str(base_dir / "results"), item)'
    )

    # Fix 7: z.extractall("results") -> z.extractall(str(base_dir / "results"))
    src = src.replace(
        '                    with zipfile.ZipFile(zpath, "r") as z:\n                        z.extractall("results")',
        '                    with zipfile.ZipFile(zpath, "r") as z:\n                        z.extractall(str(base_dir / "results"))'
    )

    cell2['source'] = [l + '\n' for l in src.split('\n')]
    cell2['source'][-1] = cell2['source'][-1].rstrip('\n')

    with open('molab_run.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print("Fixed Log Restore to use base_dir paths!")

if __name__ == "__main__":
    fix_log_restore()
