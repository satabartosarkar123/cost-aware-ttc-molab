"""
github_sync.py — Git sync helper for Molab.

Called automatically by molab_run.ipynb:
  - At start: clones the repo (or pulls if already cloned)
  - After each run: commits + pushes results/checkpoints back

You never need to manually download results — just run:
    git pull
on your local PC and everything appears.

SETUP (one-time):
  1. Create a GitHub Personal Access Token (PAT):
     github.com → Settings → Developer settings → Personal access tokens → Tokens (classic)
     → Generate new token → check 'repo' scope → copy it
  2. Set GITHUB_TOKEN below (or set it as env var before running)
  3. Set GITHUB_REPO to your new repo URL
"""

import os
import subprocess
import sys
from pathlib import Path

# ── CONFIGURE THESE ───────────────────────────────────────────────────────────
GITHUB_REPO  = "https://github.com/satabartosarkar123/cost-aware-ttc-molab.git"
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")   # set as env var or paste here
GIT_NAME     = "Molab Runner"
GIT_EMAIL    = "molab@runner.local"
CLONE_DIR    = "/marimo"                             # where Molab puts files
# ─────────────────────────────────────────────────────────────────────────────

# Dirs whose contents get committed after a run
RESULT_DIRS = [
    "rq2_part1/results",
    "rq2_part1/checkpoints",
    "rq2_part1/reports",
    "rq2_part1/plots",
    "ttc-frugalreason-poc/experiment_fr/results",
    "ttc-frugalreason-poc/experiment_fr/reports",
    "ttc-task-poc/experiment/results_50",
    "ttc-task-poc/experiment/reports_50",
]


def _run(cmd, cwd=None, capture=False):
    r = subprocess.run(cmd, shell=True, cwd=cwd,
                       capture_output=capture, text=True)
    if not capture and r.returncode != 0:
        print(f"  [warn] exit {r.returncode}: {cmd[:80]}")
    return r


def _auth_url():
    """Inject PAT into repo URL for push auth."""
    if not GITHUB_TOKEN:
        return GITHUB_REPO
    if "https://" in GITHUB_REPO:
        return GITHUB_REPO.replace("https://", f"https://{GITHUB_TOKEN}@")
    return GITHUB_REPO


def setup(project_dir: str = None):
    """
    Clone or pull the repo. Call once at the start of each Molab session.
    Returns the path to the project root.
    """
    if GITHUB_TOKEN == "" or "PASTE" in GITHUB_TOKEN:
        print("  github_sync: no token set — skipping git sync")
        print("  Set GITHUB_TOKEN env var or paste it into github_sync.py")
        return project_dir or CLONE_DIR

    root = Path(project_dir or CLONE_DIR)

    if (root / ".git").exists():
        print("  Repo already present — pulling latest...")
        _run("git pull --ff-only", cwd=str(root))
    else:
        print(f"  Cloning {GITHUB_REPO} → {root} ...")
        root.mkdir(parents=True, exist_ok=True)
        _run(f"git clone {_auth_url()} .", cwd=str(root))

    # Configure git identity (needed to commit on remote machines)
    _run(f'git config user.name "{GIT_NAME}"', cwd=str(root))
    _run(f'git config user.email "{GIT_EMAIL}"', cwd=str(root))

    # Store token so push works without prompting
    if GITHUB_TOKEN:
        _run(f"git remote set-url origin {_auth_url()}", cwd=str(root))

    print(f"  Repo ready at {root}")
    return str(root)


def push_results(project_dir: str, message: str = None):
    """
    Stage result dirs, commit, and push to GitHub.
    Call this after any run cell finishes.
    """
    if GITHUB_TOKEN == "" or "PASTE" in GITHUB_TOKEN:
        print("  github_sync: no token — skipping push")
        return

    root = Path(project_dir)
    if not (root / ".git").exists():
        print("  Not a git repo — run setup() first")
        return

    # Stage result dirs that exist
    staged = 0
    for rel in RESULT_DIRS:
        d = root / rel
        if d.exists():
            _run(f"git add {rel}", cwd=str(root))
            staged += 1

    if staged == 0:
        print("  Nothing to push (no result dirs found yet)")
        return

    # Check if there's anything new to commit
    r = _run("git diff --cached --quiet", cwd=str(root), capture=True)
    if r.returncode == 0:
        print("  No changes since last push")
        return

    from datetime import datetime, timezone
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    msg = message or f"Molab results — {ts}"

    _run(f'git commit -m "{msg}"', cwd=str(root))
    _run("git push origin main", cwd=str(root))
    print(f"  Pushed: {msg}")


if __name__ == "__main__":
    # Called directly: python github_sync.py setup | push
    cmd = sys.argv[1] if len(sys.argv) > 1 else "setup"
    proj = sys.argv[2] if len(sys.argv) > 2 else CLONE_DIR
    if cmd == "setup":
        setup(proj)
    elif cmd == "push":
        push_results(proj)
    else:
        print(f"Usage: python github_sync.py [setup|push] [project_dir]")
