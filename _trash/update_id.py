import json, ast, os
def W(f,s): open(f,'w',encoding='utf-8').write(s)
def R(f): return open(f,encoding='utf-8').read()
def md(t): return {'cell_type':'markdown','metadata':{},'source':[t]}
def code(s): return {'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':[s]}

SRC_C1 = '''import subprocess, zipfile, os, sys, importlib.util
from pathlib import Path

GDRIVE_FILE_ID = "1MgNSRcN3-WjDo-aRwVgPqCGNdfWJwOzY"
ZIP_NAME = "Cost-Aware-Test-Time-upload.zip"

if Path(ZIP_NAME).exists():
    os.remove(ZIP_NAME)
subprocess.run("pip install -q gdown", shell=True)
import gdown
print("Downloading project zip...")
gdown.download(id=GDRIVE_FILE_ID, output=ZIP_NAME, quiet=False)

if Path(ZIP_NAME).exists():
    print("Extracting...")
    with zipfile.ZipFile(ZIP_NAME, "r") as z:
        z.extractall(".")
    base_dir = Path(".")
    if (base_dir / "Cost-Aware-Test-Time-upload").exists():
        base_dir = base_dir / "Cost-Aware-Test-Time-upload"
    elif (base_dir / "Cost-Aware-Test-time-molab").exists():
        base_dir = base_dir / "Cost-Aware-Test-time-molab"
    os.environ["NOTEBOOK_DIR"] = str(base_dir.resolve())
    print(f"NOTEBOOK_DIR = {os.environ['NOTEBOOK_DIR']}")
    checks = [
        "rq2_part1/run_rq2_part1.py",
        "ttc-frugalreason-poc/experiment_fr/run_frugal_smoke.py",
        "ttc-frugalreason-poc/experiment_fr/run_day0.py",
        "ttc-frugalreason-poc/experiment_fr/run_real_experiment.py",
        "ttc-task-poc/experiment/run_poc.py",
        "gdrive_oauth.py", "auto_backup.py", "requirements_molab.txt",
    ]
    for f in checks:
        ok = (base_dir / f).exists()
        print(f"  {'OK    ' if ok else 'MISS  '}: {f}")
    # Drive auth - runs immediately after extraction
    print("\\n" + "="*60 + "\\n  Connecting Google Drive...\\n" + "="*60)
    subprocess.run(
        "pip install -q google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client",
        shell=True)
    try:
        _sp = importlib.util.spec_from_file_location("gdrive_oauth", str(base_dir / "gdrive_oauth.py"))
        _gd = importlib.util.module_from_spec(_sp)
        _sp.loader.exec_module(_gd)
        _ok = _gd.init(base_dir)
        print("  Drive: ACTIVE - checkpoints restored." if _ok else "  Drive: DISABLED.")
    except Exception as _e:
        print(f"  Drive error: {_e}")
else:
    print("ERROR: zip not found.")
'''
W('src_c1.py', SRC_C1)
print('Updated Cell 1 with new file ID')