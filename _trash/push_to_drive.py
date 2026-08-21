"""
push_to_drive.py — Run this on your LOCAL PC after any code change.

What it does:
  1. Builds a fresh zip of the project
  2. Uploads it to Google Drive as 'Cost-Aware-Test-Time-LATEST.zip'
  3. Deletes the previous version so there's only ever one file
  4. Prints the new shareable link

Run with:
    python push_to_drive.py

Requirements:
    pip install pydrive2
    rclone must be configured with a 'gdrive' remote (see MOLAB_README.md)
"""

import os
import sys
import zipfile
import subprocess
import json
import tempfile
from pathlib import Path
from datetime import datetime

# ── Config ────────────────────────────────────────────────────────────────────
GDRIVE_FOLDER   = "gdrive:Molab_Uploads"          # rclone remote folder
FILENAME        = "Cost-Aware-Test-Time-LATEST.zip"
PROJECT_ROOT    = Path(__file__).resolve().parent
NOTEBOOK_ID_KEY = "GDRIVE_FILE_ID"               # key to update in molab_run.ipynb

INCLUDE = [
    "rq2_part1",
    "ttc-frugalreason-poc",
    "ttc-task-poc",
    "molab_run.ipynb",
    "auto_backup.py",
    "drive_checkpoint.py",
    "requirements_molab.txt",
    "MOLAB_README.md",
    "EXPERIMENT_CATALOG.md",
]
SKIP_DIRS = {".git", "__pycache__", ".venv", "node_modules", "temp_prm800k"}
SKIP_EXTS = {".pyc", ".exe", ".db", ".log", ".db-shm", ".db-wal"}
# ─────────────────────────────────────────────────────────────────────────────


def _rclone(*args):
    result = subprocess.run(["rclone"] + list(args), capture_output=True, text=True)
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def check_rclone():
    rc, _, err = _rclone("lsd", f"{GDRIVE_FOLDER.split(':')[0]}:", "--max-depth", "1")
    if rc != 0:
        print(f"ERROR: rclone cannot reach Google Drive: {err}")
        print("Make sure rclone is installed and 'gdrive' remote is configured.")
        print("Run: rclone config")
        sys.exit(1)
    print("  rclone → Google Drive: OK")


def build_zip(zip_path: Path) -> int:
    count = 0
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
        for item in INCLUDE:
            src = PROJECT_ROOT / item
            if not src.exists():
                print(f"  skip (not found): {item}")
                continue
            if src.is_file():
                zf.write(src, item)
                count += 1
            else:
                for fpath in src.rglob("*"):
                    if not fpath.is_file():
                        continue
                    if any(p.name in SKIP_DIRS for p in fpath.parents):
                        continue
                    if fpath.suffix.lower() in SKIP_EXTS:
                        continue
                    arcname = str(fpath.relative_to(PROJECT_ROOT))
                    zf.write(fpath, arcname)
                    count += 1
    return count


def delete_old_versions():
    """Delete all existing LATEST zips from the Drive folder."""
    rc, out, _ = _rclone("lsf", GDRIVE_FOLDER, "--format", "n")
    if rc != 0:
        return  # folder may not exist yet, that's fine
    for line in out.splitlines():
        if "Cost-Aware-Test-Time-LATEST" in line:
            name = line.strip()
            print(f"  Deleting old: {name}")
            _rclone("delete", f"{GDRIVE_FOLDER}/{name}")


def upload(zip_path: Path):
    print(f"  Uploading {zip_path.name} ({zip_path.stat().st_size // 1024} KB)...")
    rc, _, err = _rclone("copy", str(zip_path), GDRIVE_FOLDER)
    if rc != 0:
        print(f"  ERROR uploading: {err}")
        sys.exit(1)
    print("  Upload complete.")


def get_file_id() -> str:
    """Get the Drive file ID of the uploaded zip using rclone lsjson."""
    rc, out, _ = _rclone("lsjson", GDRIVE_FOLDER)
    if rc != 0 or not out:
        return ""
    try:
        files = json.loads(out)
        for f in files:
            if f.get("Name") == FILENAME:
                return f.get("ID", "")
    except Exception:
        pass
    return ""


def make_shareable(file_id: str):
    """Set the file to 'anyone with link can view' using rclone."""
    if not file_id:
        return
    _rclone(
        "backend", "set-perm",
        f"{GDRIVE_FOLDER}/{FILENAME}",
        "--perm", "reader",
        "--type", "anyone",
    )


def update_notebook_id(new_id: str):
    """Update GDRIVE_FILE_ID in molab_run.ipynb."""
    nb_path = PROJECT_ROOT / "molab_run.ipynb"
    if not nb_path.exists():
        print("  molab_run.ipynb not found — skipping notebook update")
        return
    content = nb_path.read_text(encoding="utf-8")
    import re
    updated = re.sub(
        r'(GDRIVE_FILE_ID\s*=\s*")[^"]*(")',
        f'\\g<1>{new_id}\\g<2>',
        content,
    )
    if updated != content:
        nb_path.write_text(updated, encoding="utf-8")
        print(f"  molab_run.ipynb updated: GDRIVE_FILE_ID = {new_id}")
    else:
        print("  molab_run.ipynb: GDRIVE_FILE_ID pattern not found — update manually")


def main():
    print("=" * 60)
    print("  push_to_drive.py — Build + Upload LATEST zip")
    print("=" * 60)

    # 1. Check rclone
    print("\n[1/5] Checking rclone...")
    check_rclone()

    # 2. Build zip
    print("\n[2/5] Building zip...")
    with tempfile.TemporaryDirectory() as tmp:
        zip_path = Path(tmp) / FILENAME
        count = build_zip(zip_path)
        size_mb = zip_path.stat().st_size / (1024 ** 2)
        print(f"  {count} files → {size_mb:.1f} MB")

        # 3. Delete old versions
        print("\n[3/5] Removing old versions from Drive...")
        delete_old_versions()

        # 4. Upload
        print("\n[4/5] Uploading...")
        upload(zip_path)

    # 5. Get file ID + share
    print("\n[5/5] Getting shareable link...")
    file_id = get_file_id()
    if file_id:
        make_shareable(file_id)
        link = f"https://drive.google.com/file/d/{file_id}/view?usp=sharing"
        print(f"\n  File ID : {file_id}")
        print(f"  Link    : {link}")
        update_notebook_id(file_id)
    else:
        print("  Could not retrieve file ID via rclone lsjson.")
        print(f"  Go to Drive → {GDRIVE_FOLDER.split(':')[1]} → find {FILENAME} → get shareable link manually")
        print("  Then paste the ID into molab_run.ipynb GDRIVE_FILE_ID")

    print("\n" + "=" * 60)
    print("  DONE. Upload complete.")
    print("  Upload molab_run.ipynb to Molab — it already has the new file ID.")
    print("=" * 60)


if __name__ == "__main__":
    main()
