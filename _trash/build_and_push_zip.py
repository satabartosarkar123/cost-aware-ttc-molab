import os
import zipfile
from pathlib import Path
from huggingface_hub import HfApi

PROJECT_ROOT = Path('..').resolve()
ZIP_PATH = PROJECT_ROOT / 'Cost-Aware-Test-Time-upload.zip'

INCLUDE = [
    "Cost-Aware-Test-time-molab/rq2_part1",
    "Cost-Aware-Test-time-molab/ttc-frugalreason-poc",
    "Cost-Aware-Test-time-molab/ttc-task-poc",
    "Cost-Aware-Test-time-molab/molab_run.ipynb",
    "Cost-Aware-Test-time-molab/molab_run_fixed.ipynb",
    "Cost-Aware-Test-time-molab/auto_backup.py",
    "Cost-Aware-Test-time-molab/drive_checkpoint.py",
    "Cost-Aware-Test-time-molab/requirements_molab.txt",
    "Cost-Aware-Test-time-molab/MOLAB_README.md",
    "Cost-Aware-Test-time-molab/EXPERIMENT_CATALOG.md",
]

SKIP_DIRS = {".git", "__pycache__", ".venv", "node_modules", "temp_prm800k"}
SKIP_EXTS = {".pyc", ".exe", ".db", ".log", ".db-shm", ".db-wal"}

def build_zip():
    count = 0
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
        for item in INCLUDE:
            src = PROJECT_ROOT / item
            if not src.exists():
                print(f"  skip (not found): {item}")
                continue
            if src.is_file():
                # Store it such that when extracted, it just goes into the current directory
                # wait, if the zip needs to be extracted flat, or with the Cost-Aware-Test-time-molab folder?
                # The user's notebook extracts it directly into /marimo.
                arcname = str(src.relative_to(PROJECT_ROOT / 'Cost-Aware-Test-time-molab'))
                zf.write(src, arcname)
                count += 1
            else:
                for fpath in src.rglob("*"):
                    if not fpath.is_file():
                        continue
                    if any(p.name in SKIP_DIRS for p in fpath.parents):
                        continue
                    if fpath.suffix.lower() in SKIP_EXTS:
                        continue
                    arcname = str(fpath.relative_to(PROJECT_ROOT / 'Cost-Aware-Test-time-molab'))
                    zf.write(fpath, arcname)
                    count += 1
    return count

print("Building fresh zip...")
build_zip()
print(f"Zip built! Size: {ZIP_PATH.stat().st_size / 1024 / 1024:.2f} MB")

_api = HfApi(token="REDACTED")
print("Uploading to HF...")
_api.upload_file(
    path_or_fileobj=str(ZIP_PATH),
    path_in_repo="Cost-Aware-Test-Time-upload.zip",
    repo_id="Satabarto/Molab_Checkpoints_Cost_AWARE",
    repo_type="dataset",
)
print("HF upload successful!")
