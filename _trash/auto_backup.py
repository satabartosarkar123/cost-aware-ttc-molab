"""
auto_backup.py — Google Drive backup daemon for Molab.

Run this ON THE MOLAB INSTANCE (deploy_molab.py launches it automatically).
It zips checkpoints + results every 60 minutes and pushes them to Google Drive
via rclone.

SETUP (one-time, on your LOCAL PC before deploying):
  1. Install rclone locally:  https://rclone.org/downloads/
  2. Run:  rclone config
     - Name the remote exactly:  gdrive
     - Storage type: Google Drive
     - Follow browser auth prompts
  3. Open your rclone.conf and copy the [gdrive] block.
  4. Paste it into RCLONE_CONF below, replacing the placeholder.
  5. Set GDRIVE_FOLDER to match your desired Drive path.

Your rclone.conf is usually at:
  Windows : %APPDATA%\\rclone\\rclone.conf
  Linux/Mac: ~/.config/rclone/rclone.conf
"""

import os
import time
import subprocess
import logging
import zipfile
from pathlib import Path
from datetime import datetime, timezone

# ── Drive checkpoint module (shared with the pipeline) ───────────────────────
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    import drive_checkpoint as _dcp
    _DRIVE_MODULE = True
except ImportError:
    _dcp = None
    _DRIVE_MODULE = False

# ── User configuration ────────────────────────────────────────────────────────

WORKSPACE       = "/workspace/Cost-Aware-Test-Time"
GDRIVE_FOLDER   = "gdrive:molab-checkpoints"  # folder shared with SA
BACKUP_INTERVAL = 3600          # seconds between backups (1 hour)
ZIP_PATH        = "/workspace/backup_cat.zip"

# Service account — pre-configured.
SERVICE_ACCOUNT_JSON = '{\n  "type": "service_account",\n  "project_id": "molabber",\n  "private_key_id": "c6a8ef4866b4b2970555522fcf151f5412b6b735",\n  "private_key": "-----BEGIN PRIVATE KEY-----\\nMIIEvAIBADANBgkqhkiG9w0BAQEFAASCBKYwggSiAgEAAoIBAQDPsuFcOmsxaDMY\\n9UfRS6Wwe3elcOOE5P/BRoNklPcu7JwKF/5yssgJwhzFUumm6jBxUvJpuvMPKEFP\\nzH+DVaUO7Dd56vdGM4Ovfwa1LLse9XuHvRKOr4RavvSOmPmAbOYs7EwLO1cGt2xM\\nuAMVxjVLUXIAM54c1lglUfa6VodWt6FkNGjui5GYenKAMgw2Ywjhc2GOdvTb6ruc\\nSxWcJ4z/BLZItVVFUqdRjKB8cVlcsrfdsId4VIUqAav0UzNszrxOLt6BuuqLQVQ9\\nhh5LGF//paGoCjg/PCFX/yIAenAkfoKMeaZTbDbK92Rh6jmi9un6Wlog2khQKHBN\\nEeBjVDFpAgMBAAECggEAAKTiqlti0guz6NzN3GCHLYJtk1LMVH3OPV3Ux3qjLoej\\nHBsrisw/d2zQT21GS8AdTROZEshDiH7/9i7jmN/VWcUQF2DuH3FO/KrWAnSwJ7rs\\nJ0/5J7nJaeuPm9rL7BPupEQnua5kMbr4nnBrgtRGbKJV/Yii6UC3LXYu+XSfTfQq\\nmI058R5JHL32LYIYXWKK45uU7KZwPrQEanzYDbTdOyr6/avpau0C6hQwENla3J5u\\ncW7vuRpV0t4FEb1jgAm5Y1DUIWyG8lP9hW9h/XdM0P1D5Pqzo2ydGhOmiRCwCCkK\\nMUZMduqMJb1Ti8QOF0WIbZEh7fXVrUv8H9v9+9KI2QKBgQDuHI8PDkHsvNSxBdzc\\nD3xvva6p9YzmvZkHpLQiYLnQnk1NaIAmkg9DjuOUr7hzIT2fR8jiFCO+Z7rlRtK4\\n/70w5i3O+2jDFNUW9CQmaqefbJrCCFaE3hmnzGVqOf4vVXjjk07qH6az0nEmw4Yt\\nsbZB702ykFXRpmouBOUCedaIRQKBgQDfTWmGyfUgObuww0H94CRaY3VyETbbbblO\\nDnCFdEzFHOQ4e3m4X4C5rgqnUYD1NoLqfDvtupyxmAYcV4WM6LvwzHXQA8q9Catz\\n9v9ZWmK1YIMlr/+CLc1KXO2NY8jtPAQgfNocdi5mr/Xc2LXve6gSsUl/2mCPGjFc\\n43etUFaQ1QKBgEl6hqseuzlTDE+Uf5NpM/1Hi57nJ5QM7ixtpyj0sGKwdypsFR/R\\n8uPmNFSt5T2iBGIixNr/XAhl+kbGlECCqt9sKLa23p1U0G1E6eLxBskrupYl/I0D\\n/ObLLICbZNU2ixevXariGY9kYYaUz9NKA/RU5KU15UXNFPcei404C/wBAoGAQM5q\\nrd+28F9RBX/lixSd+E2dLDmqvgweF3VBWrnh/eLgqTPMo6Gz7i+AkAarcn8bh4n+\\nqoPaLgB85YTREZAJ21y7ZF91W1+PDtzERt5gf1s/NJTbhqBcUBSgLMSk75TXbcZO\\nVqxF0y+GH04Vnyc4JBSnzB9Inr9vTBIIDZifRvECgYBTmmH/2vsc4awWxLaM0pof\\nvF332vO9UTjIeQo5ZrVyROtcjjxiEO5bgQHlvOd0LKfthN0i/LPsTIDlxbZrTWgD\\npquXAtsrmwPq3IHRNoypyR5r53MQEDI5QrDeyDVeo7AQyrPmkFaPW/UxMmE7gOGd\\nTRiuOdHf3z410DxhC65q7Q==\\n-----END PRIVATE KEY-----\\n",\n  "client_email": "molab-448@molabber.iam.gserviceaccount.com",\n  "client_id": "116797867432972363847",\n  "auth_uri": "https://accounts.google.com/o/oauth2/auth",\n  "token_uri": "https://oauth2.googleapis.com/token",\n  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",\n  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/molab-448%40molabber.iam.gserviceaccount.com",\n  "universe_domain": "googleapis.com"\n}'

RCLONE_CONF = """
[gdrive]
type = drive
scope = drive
service_account_file = /tmp/molab_sa.json
"""

# ── Directories to back up (relative to WORKSPACE) ────────────────────────────
BACKUP_DIRS = [
    "rq2_part1/checkpoints",
    "rq2_part1/results",
    "rq2_part1/reports",
    "rq2_part1/plots",
    "ttc-frugalreason-poc/experiment_fr/results",
    "ttc-frugalreason-poc/experiment_fr/reports",
    "ttc-frugalreason-poc/experiment_fr/plots",
    "ttc-task-poc/experiment/results",
    "ttc-task-poc/experiment/results_50",
    "ttc-task-poc/experiment/reports",
]

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [BACKUP] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("/workspace/auto_backup.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


# ── rclone setup ──────────────────────────────────────────────────────────────

def setup_rclone() -> bool:
    """Install rclone and write the config file."""
    # Install rclone if missing
    if subprocess.run("which rclone", shell=True, capture_output=True).returncode != 0:
        log.info("Installing rclone...")
        rc = os.system("curl -fsSL https://rclone.org/install.sh | sudo bash > /dev/null 2>&1")
        if rc != 0:
            log.error("rclone install failed — backups will be skipped.")
            return False

    # Write config
    conf_dir = Path("/root/.config/rclone")
    conf_dir.mkdir(parents=True, exist_ok=True)
    conf_file = conf_dir / "rclone.conf"

    conf_content = RCLONE_CONF.strip()
    if "YOUR_CLIENT_ID_HERE" in conf_content or "YOUR_ACCESS_TOKEN" in conf_content:
        log.error(
            "RCLONE_CONF still contains placeholder values! "
            "Edit auto_backup.py and paste your real rclone.conf content."
        )
        return False

    conf_file.write_text(conf_content, encoding="utf-8")
    log.info(f"rclone.conf written to {conf_file}")

    # Quick connectivity test
    result = subprocess.run(
        f"rclone lsd {GDRIVE_FOLDER.split(':')[0]}: --max-depth 1",
        shell=True, capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        log.warning(
            f"rclone connectivity test failed: {result.stderr.strip()}\n"
            "Backups will still be attempted each hour."
        )
    else:
        log.info("rclone connectivity test passed — Google Drive is reachable.")

    return True


# ── Zip helper ────────────────────────────────────────────────────────────────

def create_zip() -> bool:
    """Zip all BACKUP_DIRS that exist into ZIP_PATH."""
    workspace = Path(WORKSPACE)
    included = 0

    try:
        with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
            for rel_dir in BACKUP_DIRS:
                src = workspace / rel_dir
                if not src.exists():
                    log.debug(f"  skip (not found): {src}")
                    continue
                for fpath in src.rglob("*"):
                    if fpath.is_file():
                        arcname = str(fpath.relative_to(workspace))
                        zf.write(fpath, arcname)
                        included += 1

        size_mb = Path(ZIP_PATH).stat().st_size / (1024 ** 2)
        log.info(f"Zip created: {included} files, {size_mb:.1f} MB → {ZIP_PATH}")
        return included > 0

    except Exception as e:
        log.error(f"Zip creation failed: {e}")
        return False


# ── rclone push ───────────────────────────────────────────────────────────────

def push_to_drive() -> bool:
    """Upload ZIP_PATH to Google Drive."""
    # Put the zip inside a timestamped subfolder so every backup is kept
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    dest = f"{GDRIVE_FOLDER}/{ts}"

    log.info(f"Uploading to {dest} ...")
    result = subprocess.run(
        f"rclone copy {ZIP_PATH} {dest} --progress",
        shell=True, capture_output=True, text=True, timeout=600,
    )
    if result.returncode == 0:
        log.info("Upload complete.")
        return True
    else:
        log.error(f"Upload failed (exit {result.returncode}):\n{result.stderr.strip()}")
        return False


# ── Main loop ─────────────────────────────────────────────────────────────────

def run_backup_cycle():
    """Run one full backup cycle using drive_checkpoint.push_results_full()."""
    log.info("─── Backup cycle starting ───")

    if _DRIVE_MODULE:
        # Preferred path: use the shared drive_checkpoint module
        try:
            _dcp.push_results_full()
            log.info("─── Backup cycle done (drive_checkpoint) ───")
            return
        except Exception as e:
            log.warning(f"drive_checkpoint.push_results_full failed: {e} — falling back to zip method")

    # ── Fallback: zip + rclone copy (original behaviour) ─────────────────────
    log.info("─── Backup cycle (zip fallback) ───")
    if not create_zip():
        log.warning("Nothing to zip or zip failed — skipping upload.")
        return
    push_to_drive()
    try:
        os.remove(ZIP_PATH)
    except OSError:
        pass
    log.info("─── Backup cycle done (zip fallback) ───")


def main():
    log.info("Auto-backup daemon starting.")
    log.info(f"  Workspace : {WORKSPACE}")
    log.info(f"  Destination: {GDRIVE_FOLDER}")
    log.info(f"  Interval  : {BACKUP_INTERVAL // 60} minutes")

    # ── Init via drive_checkpoint (handles rclone install + conf write) ───────
    if _DRIVE_MODULE:
        rclone_ok = _dcp._write_rclone_conf()
    else:
        rclone_ok = setup_rclone()

    if not rclone_ok:
        log.warning(
            "rclone setup had issues. Will still run the loop and retry "
            "each cycle in case the issue is transient."
        )

    # Run an immediate first backup, then go into the hourly loop
    run_backup_cycle()

    while True:
        log.info(f"Next backup in {BACKUP_INTERVAL // 60} minutes.")
        time.sleep(BACKUP_INTERVAL)
        # Re-check rclone each cycle in case config was fixed mid-run
        if "YOUR_CLIENT_ID_HERE" not in RCLONE_CONF:
            setup_rclone()
        run_backup_cycle()


if __name__ == "__main__":
    main()
