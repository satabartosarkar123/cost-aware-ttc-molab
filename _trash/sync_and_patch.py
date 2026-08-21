"""
Compare all .py files in ttc-frugalreason-poc and ttc-task-poc between
the SOURCE (Cost-Aware-Test-time) and MOLAB (Cost-Aware-Test-time-molab) dirs.
Report any differences. Then sync source -> molab for any mismatches.
"""
import os, filecmp, shutil

SRC  = r"c:\Users\USER\Cost-Aware-Test-Time\Cost-Aware-Test-time"
DST  = r"c:\Users\USER\Cost-Aware-Test-Time\Cost-Aware-Test-time-molab"

SUBDIRS = [
    "ttc-frugalreason-poc/experiment_fr/strategies",
    "ttc-frugalreason-poc/experiment_fr/parsers",
    "ttc-frugalreason-poc/experiment_fr/core",
    "ttc-frugalreason-poc/experiment_fr",          # top-level runners
    "ttc-task-poc/experiment/strategies",
    "ttc-task-poc/experiment/parsers",
    "ttc-task-poc/experiment/core",
    "ttc-task-poc/experiment",                      # top-level runners
    "rq2_part1",
]

SKIP = {".venv", "__pycache__", "node_modules", ".git", "temp_prm800k"}

diffs = []
missing_in_molab = []
missing_in_src = []
identical = 0

for subdir in SUBDIRS:
    src_dir = os.path.join(SRC, subdir)
    dst_dir = os.path.join(DST, subdir)
    
    if not os.path.isdir(src_dir):
        print(f"SKIP (no src): {subdir}")
        continue
    
    for root, dirs, files in os.walk(src_dir):
        # Skip unwanted dirs
        dirs[:] = [d for d in dirs if d not in SKIP]
        
        for fname in files:
            if not fname.endswith(".py"):
                continue
            
            src_path = os.path.join(root, fname)
            rel = os.path.relpath(src_path, SRC)
            dst_path = os.path.join(DST, rel)
            
            if not os.path.exists(dst_path):
                missing_in_molab.append(rel)
                continue
            
            if filecmp.cmp(src_path, dst_path, shallow=False):
                identical += 1
            else:
                diffs.append(rel)

print(f"\n{'='*60}")
print(f"COMPARISON RESULTS")
print(f"{'='*60}")
print(f"Identical files: {identical}")
print(f"DIFFERENT files: {len(diffs)}")
print(f"Missing in molab: {len(missing_in_molab)}")

if diffs:
    print(f"\n--- FILES THAT DIFFER ---")
    for f in sorted(diffs):
        src_size = os.path.getsize(os.path.join(SRC, f))
        dst_size = os.path.getsize(os.path.join(DST, f))
        print(f"  {f}  (src={src_size}B, molab={dst_size}B)")

if missing_in_molab:
    print(f"\n--- MISSING IN MOLAB ---")
    for f in sorted(missing_in_molab):
        print(f"  {f}")

# Now SYNC: copy all differing and missing files from SRC -> DST
synced = 0
for f in diffs + missing_in_molab:
    src_path = os.path.join(SRC, f)
    dst_path = os.path.join(DST, f)
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    shutil.copy2(src_path, dst_path)
    synced += 1
    print(f"  SYNCED: {f}")

if synced:
    print(f"\n>>> Synced {synced} files from source -> molab")
else:
    print(f"\n>>> All files already in sync!")

# Now apply the warnings patch to the synced verifier files
print(f"\n{'='*60}")
print("Applying warnings.catch_warnings patch to all eval() calls...")
print(f"{'='*60}")

import re

EVAL_FILES = [
    os.path.join(DST, "ttc-frugalreason-poc/experiment_fr/core/verifier.py"),
    os.path.join(DST, "ttc-task-poc/experiment/core/verifier.py"),
    os.path.join(DST, "ttc-task-poc/experiment/parsers/game24_parser.py"),
    os.path.join(DST, "ttc-frugalreason-poc/experiment_fr/parsers/game24_parser.py"),
]

for fpath in EVAL_FILES:
    if not os.path.exists(fpath):
        continue
    with open(fpath, "r", encoding="utf-8") as f:
        code = f.read()
    
    if "warnings.catch_warnings" in code:
        print(f"  ALREADY PATCHED: {os.path.relpath(fpath, DST)}")
        continue
    
    # Pattern: bare eval() calls that need wrapping
    # We add import warnings at the top if not present
    if "import warnings" not in code:
        code = "import warnings\n" + code
    
    # Wrap eval(..., {"__builtins__": {}}, {}) calls
    lines = code.split("\n")
    new_lines = []
    i = 0
    patched = False
    while i < len(lines):
        line = lines[i]
        stripped = line.lstrip()
        indent = line[:len(line) - len(stripped)]
        
        if 'eval(' in stripped and '{"__builtins__": {}}' in stripped and 'warnings' not in stripped:
            # Find the try: block this belongs to and wrap the eval
            new_lines.append(f"{indent}with warnings.catch_warnings():")
            new_lines.append(f"{indent}    warnings.simplefilter('ignore', SyntaxWarning)")
            # Re-indent the eval line
            new_lines.append(f"{indent}    {stripped}")
            patched = True
        else:
            new_lines.append(line)
        i += 1
    
    if patched:
        with open(fpath, "w", encoding="utf-8") as f:
            f.write("\n".join(new_lines))
        print(f"  PATCHED: {os.path.relpath(fpath, DST)}")
    else:
        print(f"  NO EVAL FOUND: {os.path.relpath(fpath, DST)}")

print("\nDONE!")
