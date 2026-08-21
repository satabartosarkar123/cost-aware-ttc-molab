import os
import sys
import json
import zipfile
from pathlib import Path
from datetime import datetime

ROOT_DIR = Path(__file__).resolve().parent
ZIP_OUT = ROOT_DIR.parent / "Cost-Aware-Test-Time-upload.zip"

print(f"Root: {ROOT_DIR}")
print(f"Target Zip: {ZIP_OUT}")

# 2. Build the ZIP
INCLUDE_DIRS = [
    "results",
    "rq2_part1",
    "ttc-frugalreason-poc",
    "ttc-task-poc",
]
INCLUDE_FILES = [
    "molab_run.ipynb",
    "molab_run.py",
    "molab_runner.py",
    "molab_setup.py",
    "gdrive_oauth.py",
    "auto_backup.py",
    "drive_checkpoint.py",
    "requirements_molab.txt",
    "MOLAB_README.md",
    "EXPERIMENT_CATALOG.md",
    "MOLAB_READY.md",
    "README.md",
    ".gitignore",
]

SKIP_DIRS = {".git", "__pycache__", ".venv", "node_modules", "temp_prm800k"}
SKIP_EXTS = {".pyc", ".exe"}

tmp_zip = ROOT_DIR.parent / "temp_build.zip"
if tmp_zip.exists():
    tmp_zip.unlink()

file_count = 0
with zipfile.ZipFile(str(tmp_zip), "w", zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
    for f in INCLUDE_FILES:
        p = ROOT_DIR / f
        if p.exists() and p.is_file():
            zf.write(p, f)
            file_count += 1
            
    for d in INCLUDE_DIRS:
        dp = ROOT_DIR / d
        if dp.exists() and dp.is_dir():
            for fp in dp.rglob("*"):
                if not fp.is_file(): continue
                if any(part in SKIP_DIRS for part in fp.parts): continue
                if fp.suffix.lower() in SKIP_EXTS: continue
                rel = fp.relative_to(ROOT_DIR)
                zf.write(fp, str(rel).replace("\\", "/"))
                file_count += 1

if ZIP_OUT.exists():
    ZIP_OUT.unlink()
tmp_zip.rename(ZIP_OUT)

size_mb = ZIP_OUT.stat().st_size / (1024 * 1024)
print(f"Created {ZIP_OUT.name}: {file_count} files, {size_mb:.2f} MB")

# 3. Upload to Hugging Face Hub dataset repo
print("\nUploading to Hugging Face Hub (Satabarto/Molab_Checkpoints_Cost_AWARE)...")
from huggingface_hub import HfApi
api = HfApi(token="REDACTED")

upload_res = api.upload_file(
    path_or_fileobj=str(ZIP_OUT),
    path_in_repo="Cost-Aware-Test-Time-upload.zip",
    repo_id="Satabarto/Molab_Checkpoints_Cost_AWARE",
    repo_type="dataset",
)
print(f"Successfully uploaded to Hugging Face Hub: {upload_res}")
print("ALL DONE!")
