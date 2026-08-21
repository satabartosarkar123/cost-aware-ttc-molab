"""
gdrive_oauth.py - Checkpoint sync using Hugging Face Hub.

Zero user interaction. Uses HF token to push results to a dataset repo.
"""
from __future__ import annotations
import json, os, zipfile, logging, warnings
from pathlib import Path
from datetime import datetime, timezone

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
HF_REPO = "Satabarto/Molab_Checkpoints_Cost_AWARE"
HF_TOKEN = "REDACTED"

RESULT_DIRS = [
    "rq2_part1/results",
    "rq2_part1/checkpoints",
    "rq2_part1/plots",
    "rq2_part1/reports",
    "ttc-frugalreason-poc/experiment_fr/results",
    "ttc-frugalreason-poc/experiment_fr/reports",
    "ttc-task-poc/experiment/results_50",
    "ttc-task-poc/experiment/reports_50",
]

# ── Install deps if needed ─────────────────────────────────────────────────────
def _ensure_deps():
    try:
        from huggingface_hub import HfApi
    except ImportError:
        import subprocess
        subprocess.run("pip install -q huggingface_hub", shell=True)


# ── Public API ─────────────────────────────────────────────────────────────────
_api = None

def init(workspace: Path) -> bool:
    """
    Call at session start.
    Authenticates with HuggingFace and pulls any existing checkpoints.
    Returns True if sync is active.
    """
    global _api
    _ensure_deps()

    try:
        from huggingface_hub import HfApi
        _api = HfApi(token=HF_TOKEN)
        # Verify access
        _api.repo_info(repo_id=HF_REPO, repo_type="dataset")
        print(f"  HF sync: ACTIVE (repo: {HF_REPO})")
        return True
    except Exception as e:
        print(f"  HF auth failed: {e}")
        _api = None
        return False


def push_results_full(workspace: Path):
    """
    Zip all result/checkpoint dirs and upload to HuggingFace dataset repo.
    Called after each run cell completes.
    """
    global _api
    if _api is None:
        if not init(workspace):
            log.warning("HF not initialised -- skipping push")
            return

    from huggingface_hub import HfApi

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    pushed = 0

    for rel_dir in RESULT_DIRS:
        local_dir = workspace / rel_dir
        if not local_dir.exists():
            continue

        zip_name = rel_dir.replace("/", "_").replace("-", "_") + f"_{ts}.zip"
        import tempfile
        tmp_zip = Path(tempfile.gettempdir()) / zip_name

        try:
            with zipfile.ZipFile(str(tmp_zip), "w", zipfile.ZIP_DEFLATED) as zf:
                count = 0
                for fp in local_dir.rglob("*"):
                    if fp.is_file():
                        zf.write(str(fp), str(fp.relative_to(workspace)))
                        count += 1

            if count > 0:
                _api.upload_file(
                    path_or_fileobj=str(tmp_zip),
                    path_in_repo=f"checkpoints/{zip_name}",
                    repo_id=HF_REPO,
                    repo_type="dataset",
                    token=HF_TOKEN,
                )
                pushed += 1
                log.info(f"Pushed {rel_dir} ({count} files)")
        except Exception as e:
            log.warning(f"Push failed for {rel_dir}: {e}")
        finally:
            if tmp_zip.exists():
                tmp_zip.unlink()

    print(f"  HF push done: {pushed} archive(s) uploaded to {HF_REPO}")