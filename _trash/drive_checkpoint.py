"""
drive_checkpoint.py — Live Google Drive checkpoint sync for Molab.

Used by run_rq2_part1.py (and other pipeline scripts) to:
  • pull the latest checkpoint from Drive on startup (so a fresh session
    resumes exactly where the last one died)
  • push the checkpoint file to Drive immediately after every write
    (so you never lose more than one completed question on a crash)

Also imported by auto_backup.py for the hourly full-results backup.

REQUIREMENTS
------------
rclone must be installed and configured with a remote named 'gdrive'.
molab_setup.py installs rclone automatically.
You must paste your rclone.conf content into auto_backup.py (RCLONE_CONF).
drive_checkpoint.py reads the config from the same place so you only
edit it once.

GDRIVE layout created automatically:
  gdrive:Molab_Backups/Cost-Aware-Test-Time/
    checkpoints/
      completed.jsonl          ← live-synced after every question
    results/
      question_strategy_summary.jsonl
      raw_model_calls.jsonl
      cost_profile.csv
      token_percentiles.csv
      progress_log.jsonl
    (hourly zips go here too, written by auto_backup.py)
"""

from __future__ import annotations

import os
import subprocess
import logging
import shutil
from pathlib import Path

log = logging.getLogger(__name__)

# ── Remote base — must match GDRIVE_FOLDER in auto_backup.py ──────────────────
GDRIVE_REMOTE      = "gdrive:Molab_Backups/Cost-Aware-Test-Time"
REMOTE_CKPT_DIR    = f"{GDRIVE_REMOTE}/checkpoints"
REMOTE_RESULTS_DIR = f"{GDRIVE_REMOTE}/results"

# ── Local workspace — auto-detected, override with env var WORKSPACE ──────────
# Molab may use /workspace, /root, /notebooks, or /content (Colab).
# We also accept an explicit env var so molab_run.py can set it.
def _detect_workspace() -> Path:
    # 1. Explicit override
    env = os.environ.get("WORKSPACE")
    if env:
        return Path(env)
    # 2. Try known platform roots in order
    candidates = [
        Path("/workspace/Cost-Aware-Test-Time"),
        Path("/root/Cost-Aware-Test-Time"),
        Path("/notebooks/Cost-Aware-Test-Time"),
        Path("/content/Cost-Aware-Test-Time"),
    ]
    for c in candidates:
        if c.exists():
            return c
    # 3. Fall back to directory containing this script's parent
    return Path(__file__).resolve().parent

WORKSPACE = _detect_workspace()

# Files that get live-synced on every checkpoint write
CHECKPOINT_FILES = [
    "rq2_part1/checkpoints/completed.jsonl",
]

# Files synced after every checkpoint write (lightweight rolling results)
ROLLING_RESULT_FILES = [
    "rq2_part1/results/question_strategy_summary.jsonl",
    "rq2_part1/results/raw_model_calls.jsonl",
    "rq2_part1/results/progress_log.jsonl",
]


# ── Internal helpers ───────────────────────────────────────────────────────────

def _rclone_available() -> bool:
    return shutil.which("rclone") is not None


def _rclone(args: list[str], timeout: int = 60) -> bool:
    """Run rclone with args. Returns True on success."""
    if not _rclone_available():
        log.debug("rclone not found — skipping Drive sync")
        return False
    try:
        result = subprocess.run(
            ["rclone"] + args,
            capture_output=True, text=True, timeout=timeout,
        )
        if result.returncode != 0:
            log.warning(f"rclone {' '.join(args[:2])} failed: {result.stderr.strip()[:200]}")
            return False
        return True
    except subprocess.TimeoutExpired:
        log.warning(f"rclone {' '.join(args[:2])} timed out after {timeout}s")
        return False
    except Exception as e:
        log.warning(f"rclone error: {e}")
        return False


def _write_rclone_conf() -> bool:
    """
    Write rclone.conf from auto_backup.RCLONE_CONF if not already on disk.
    Called once at startup.
    """
    conf_path = Path("/root/.config/rclone/rclone.conf")
    if conf_path.exists() and conf_path.stat().st_size > 50:
        return True  # already configured

    try:
        # Import the constant from auto_backup — only file the user edits
        import importlib.util, sys
        ab_path = Path(__file__).resolve().parent / "auto_backup.py"
        spec = importlib.util.spec_from_file_location("auto_backup", ab_path)
        ab = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ab)
        conf_content = ab.RCLONE_CONF.strip()
        # Write SA JSON so rclone service_account_file path resolves
        sa_json = getattr(ab, "SERVICE_ACCOUNT_JSON", None)
        if sa_json:
            Path("/tmp/molab_sa.json").write_text(sa_json, encoding="utf-8")
            log.info("SA JSON written to /tmp/molab_sa.json")
    except Exception as e:
        log.warning(f"Could not load RCLONE_CONF from auto_backup.py: {e}")
        return False

    if "YOUR_CLIENT_ID_HERE" in conf_content or "YOUR_ACCESS_TOKEN" in conf_content:
        log.warning(
            "RCLONE_CONF in auto_backup.py still has placeholder values. "
            "Drive sync is disabled until you paste your real rclone.conf."
        )
        return False

    conf_path.parent.mkdir(parents=True, exist_ok=True)
    conf_path.write_text(conf_content, encoding="utf-8")
    log.info("rclone.conf written from auto_backup.RCLONE_CONF")
    return True


# ── Public API ─────────────────────────────────────────────────────────────────

_initialised = False


def init() -> bool:
    # Uses gdrive_oauth for fully automatic Drive sync (OAuth2 user creds).
    # First run: interactive URL auth. All runs after: fully automatic.
    global _initialised
    try:
        import importlib.util as _ilu
        _sp = _ilu.spec_from_file_location(
            'gdrive_oauth',
            str(Path(__file__).resolve().parent / 'gdrive_oauth.py'))
        _gd = _ilu.module_from_spec(_sp)
        _sp.loader.exec_module(_gd)
        ok = _gd.init(WORKSPACE)
        if ok:
            _initialised = True
        return ok
    except Exception as e:
        log.warning(f'[DriveCP] gdrive_oauth init failed: {e}')
        return False

def _pull_checkpoints():
    """
    Download checkpoint + rolling result files from Drive to local workspace.
    Only overwrites if the remote file is newer (rclone --update flag).
    """
    for rel in CHECKPOINT_FILES + ROLLING_RESULT_FILES:
        local_path = WORKSPACE / rel
        local_path.parent.mkdir(parents=True, exist_ok=True)
        remote_path = f"{GDRIVE_REMOTE}/{rel}"
        # --update: skip if local is newer or same age
        _rclone(["copy", remote_path, str(local_path.parent),
                 "--update", "--no-traverse"], timeout=30)

    log.info("[DriveCP] Checkpoint pull complete.")


def push_checkpoint(checkpoint_file=None):
    """
    Push checkpoint + rolling results to Drive immediately.
    Called by the pipeline after every completed (task, question, strategy).

    checkpoint_file: path to the completed.jsonl being written.
                     If None, uses default CHECKPOINT_FILES list.
    """
    if not _initialised:
        return  # Drive sync not configured — silent no-op

    files_to_push = list(CHECKPOINT_FILES) + list(ROLLING_RESULT_FILES)
    if checkpoint_file is not None:
        cp = Path(checkpoint_file)
        try:
            rel = str(cp.relative_to(WORKSPACE))
            if rel not in files_to_push:
                files_to_push.insert(0, rel)
        except ValueError:
            files_to_push.insert(0, str(cp))

    for rel in files_to_push:
        local_path = WORKSPACE / rel if not Path(rel).is_absolute() else Path(rel)
        if not local_path.exists():
            continue
        remote_dir = f"{GDRIVE_REMOTE}/{Path(rel).parent}"
        _rclone(["copy", str(local_path), remote_dir, "--no-traverse"], timeout=45)

    log.debug("[DriveCP] Checkpoint pushed to Drive.")



def push_results_full():
    # Push ALL results dirs via gdrive_oauth (OAuth2, writes to your Drive).
    try:
        import importlib.util as _ilu
        _sp = _ilu.spec_from_file_location(
            'gdrive_oauth',
            str(Path(__file__).resolve().parent / 'gdrive_oauth.py'))
        _gd = _ilu.module_from_spec(_sp)
        _sp.loader.exec_module(_gd)
        _gd.push_results_full(WORKSPACE)
        log.info('[DriveCP] Full results push complete.')
    except Exception as e:
        log.warning(f'[DriveCP] push_results_full failed: {e}')
