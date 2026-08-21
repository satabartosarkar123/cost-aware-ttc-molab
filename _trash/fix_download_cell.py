import json

NB_PATH = "molab_run.ipynb"
with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

# New Cell 3 source: download from HF first, fallback to GDrive, with proper path discovery
new_download_source = '''import subprocess, zipfile, os, sys, importlib.util, shutil
from pathlib import Path

HF_REPO = "Satabarto/Molab_Checkpoints_Cost_AWARE"
HF_TOKEN = "REDACTED"
ZIP_NAME = "Cost-Aware-Test-Time-upload.zip"

if Path(ZIP_NAME).exists():
    os.remove(ZIP_NAME)

downloaded = False

# 1. Primary: Hugging Face Hub (Fast & Reliable - 13.5MB)
try:
    print("Downloading project zip from HuggingFace...")
    subprocess.run("pip install -q huggingface_hub", shell=True)
    from huggingface_hub import hf_hub_download
    hf_path = hf_hub_download(repo_id=HF_REPO, filename=ZIP_NAME, repo_type="dataset", token=HF_TOKEN)
    shutil.copy(hf_path, ZIP_NAME)
    downloaded = True
    print("Downloaded from HuggingFace Hub successfully!")
except Exception as e:
    print(f"HF download failed: {e}")

# 2. Fallback: Google Drive (old zip, different structure)
if not downloaded or not Path(ZIP_NAME).exists():
    GDRIVE_FILE_ID = "1F7lBEOBDC9FNHyK-gjOLoEQ24SpzC0tr"
    subprocess.run("pip install -q gdown", shell=True)
    import gdown
    print("Downloading from Google Drive via gdown (fallback)...")
    gdown.download(id=GDRIVE_FILE_ID, output=ZIP_NAME, quiet=False)

if Path(ZIP_NAME).exists():
    print(f"Extracting {ZIP_NAME}...")
    with zipfile.ZipFile(ZIP_NAME, "r") as z:
        z.extractall(".")

    # Smart path discovery: find where the files actually landed
    base_dir = Path(".")
    search_files = ["auto_backup.py", "requirements_molab.txt"]
    
    # Check current dir first
    if all((base_dir / f).exists() for f in search_files):
        pass  # Files are right here
    else:
        # Search one level of subdirs
        for d in sorted(base_dir.iterdir()):
            if d.is_dir() and all((d / f).exists() for f in search_files):
                base_dir = d
                break
    
    os.environ["NOTEBOOK_DIR"] = str(base_dir.resolve())
    print(f"NOTEBOOK_DIR = {os.environ['NOTEBOOK_DIR']}")
    
    checks = [
        "rq2_part1/run_rq2_part1.py",
        "ttc-frugalreason-poc/experiment_fr/run_frugal_smoke.py",
        "ttc-frugalreason-poc/experiment_fr/run_day0.py",
        "ttc-frugalreason-poc/experiment_fr/run_real_experiment.py",
        "ttc-task-poc/experiment/run_poc.py",
        "auto_backup.py", "drive_checkpoint.py", "requirements_molab.txt",
    ]
    all_ok = True
    for f in checks:
        exists = (base_dir / f).exists()
        print(f"  {'OK     ' if exists else 'MISSING'}: {f}")
        if not exists: all_ok = False
    
    if all_ok:
        print("\\nAll files present and verified!")
    else:
        print("\\nSome files missing — check zip contents.")
        # Debug: show what IS in the directory
        print("\\nFiles in base_dir:")
        for p in sorted(base_dir.iterdir()):
            print(f"  {p.name}{'/' if p.is_dir() else ''}")
else:
    print(f"ERROR: {ZIP_NAME} not found after download attempt.")
'''

# Find and replace the download cell
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        src = "".join(cell.get("source", []))
        if "GDRIVE_FILE_ID" in src and "ZIP_NAME" in src:
            lines = new_download_source.split("\n")
            cell["source"] = [line + "\n" for line in lines]
            cell["source"][-1] = cell["source"][-1].rstrip("\n")
            print(f"Replaced download cell at index {i}")
            break

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Download cell fixed!")
