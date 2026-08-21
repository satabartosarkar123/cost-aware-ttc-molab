import os, subprocess, zipfile, shutil, sys, importlib.util
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
        print("\nAll files present and verified!")
    else:
        print("\nSome files missing — check zip contents.")
        # Debug: show what IS in the directory
        print("\nFiles in base_dir:")
        for p in sorted(base_dir.iterdir()):
            print(f"  {p.name}{'/' if p.is_dir() else ''}")

    # --- LOG RESTORE (always runs) ---
    print("\nRestoring previous logs from Hugging Face...")
    try:
        from huggingface_hub import snapshot_download, hf_hub_download
        import zipfile as _zf
        
        # 1. Restore results_sync (contains block_a_logs and early data)
        try:
            print("  Syncing results_sync directory...")
            os.makedirs(str(base_dir / "results"), exist_ok=True)
            snapshot_download(
                repo_id=HF_REPO, 
                repo_type="dataset", 
                token=HF_TOKEN,
                allow_patterns=["results_sync/*"],
                local_dir=str(base_dir),
                local_dir_use_symlinks=False
            )
            _rs = base_dir / "results_sync"
            if _rs.exists():
                for item in os.listdir(str(_rs)):
                    _src = _rs / item
                    _dst = base_dir / "results" / item
                    if not _dst.exists():
                        shutil.move(str(_src), str(_dst))
        except Exception as e:
            print(f"  Warning: could not sync results_sync: {e}")

        # 2. Restore block_b zips (Day 2/Day 3 logs)
        zips_to_restore = ["block_b_qwen15b.zip", "block_b_llama32.zip"]
        for zname in zips_to_restore:
            try:
                print(f"  Downloading checkpoint {zname}...")
                zpath = hf_hub_download(repo_id=HF_REPO, filename=f"checkpoints/{zname}", repo_type="dataset", token=HF_TOKEN)
                with _zf.ZipFile(zpath, "r") as zf:
                    zf.extractall(str(base_dir / "results"))
                print(f"  Restored {zname}!")
            except Exception as e:
                print(f"  Warning: could not restore {zname}: {e}")
                
        print("Log restore complete! All past data is loaded.")
    except Exception as e:
        print(f"Log restore failed: {e}")
else:
    print(f"ERROR: {ZIP_NAME} not found after download attempt.")
