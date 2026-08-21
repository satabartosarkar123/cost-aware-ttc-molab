"""
deploy_molab.py — Run this on your LOCAL PC to push the project to Molab
and launch the pipeline + backup daemon remotely.

Prerequisites (local):
  pip install paramiko

Workflow:
  1. Run molab_setup.py in a Molab cell and wait for the Pinggy URL.
  2. Fill in HOST and PORT below from the tcp://...pinggy-free.link:PORT line.
  3. Run:  python deploy_molab.py
  4. Monitor:  python deploy_molab.py --logs          (tail pipeline log)
               python deploy_molab.py --status        (quick status check)
               python deploy_molab.py --download      (pull results back)
               python deploy_molab.py --tunnel        (get current tunnel URL)

NOTE on tunnel expiry:
  The free Pinggy tunnel auto-refreshes every 55 min on the Molab side.
  After a refresh the HOST:PORT changes. Run:
      python deploy_molab.py --tunnel
  to fetch the current URL without needing to know it in advance.
  Then update HOST/PORT here and you're reconnected.
"""

import argparse
import os
import subprocess
import sys
import zipfile
import tempfile
from pathlib import Path

# ── CONFIGURATION — update HOST/PORT after each new Molab session ────────────
HOST = "hiuob-166-19-116-193.run.pinggy-free.link"
PORT = 42347
# ─────────────────────────────────────────────────────────────────────────────

USER       = "root"
PASSWORD   = "molab2026"
REMOTE_WS  = "/workspace/Cost-Aware-Test-Time"

# ── Tunnel URL cache file (written by molab_setup.py on refresh) ─────────────
TUNNEL_URL_FILE = "/workspace/tunnel_url.txt"


def _load_current_tunnel(ssh):
    """
    Read /workspace/tunnel_url.txt from the remote machine.
    Returns (host, port) if found, else None.
    Called automatically by --tunnel and --status.
    """
    try:
        _, stdout, _ = ssh.exec_command(f"cat {TUNNEL_URL_FILE} 2>/dev/null")
        content = stdout.read().decode().strip()
        if ":" in content:
            h, p = content.rsplit(":", 1)
            return h.strip(), int(p.strip())
    except Exception:
        pass
    return None

# Which local files/dirs to ship (relative to this script's directory)
# Excludes large binary/cache artifacts automatically
INCLUDE = [
    "rq2_part1",
    "ttc-frugalreason-poc",
    "ttc-task-poc",
    "auto_backup.py",
    "requirements_molab.txt",
    "MOLAB_README.md",
]
EXCLUDE_SUFFIXES = {".pyc", ".exe", ".db", ".log"}
EXCLUDE_DIRS     = {".git", "__pycache__", ".venv", "node_modules",
                    "temp_prm800k", ".git"}

# The main entry-point to run on Molab
PIPELINE_CMD = (
    f"cd {REMOTE_WS}/rq2_part1 && "
    f"nohup python run_rq2_part1.py > {REMOTE_WS}/pipeline_run.log 2>&1 &"
)
BACKUP_CMD = (
    f"cd {REMOTE_WS} && "
    f"nohup python auto_backup.py > {REMOTE_WS}/auto_backup.log 2>&1 &"
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ssh_client():
    """Return an authenticated paramiko SSHClient."""
    try:
        import paramiko
    except ImportError:
        print("ERROR: paramiko not installed.  Run:  pip install paramiko")
        sys.exit(1)

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(HOST, PORT, USER, PASSWORD, timeout=15)
    except Exception as e:
        print(f"ERROR: Could not connect to {HOST}:{PORT} — {e}")
        print("Make sure molab_setup.py is running and you copied the right HOST/PORT.")
        sys.exit(1)
    return client


def _exec(ssh, cmd: str, show_output: bool = True) -> tuple[int, str, str]:
    """Execute a remote command and return (exit_code, stdout, stderr)."""
    _, stdout, stderr = ssh.exec_command(cmd)
    exit_code = stdout.channel.recv_exit_status()
    out = stdout.read().decode(errors="replace")
    err = stderr.read().decode(errors="replace")
    if show_output:
        if out.strip():
            print(out.strip())
        if err.strip():
            print(f"[stderr] {err.strip()}")
    return exit_code, out, err


def _build_zip(local_root: Path, zip_path: Path):
    """Zip INCLUDE paths into zip_path, honouring EXCLUDE rules."""
    added = 0
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
        for item in INCLUDE:
            src = local_root / item
            if not src.exists():
                print(f"  [skip] not found locally: {src}")
                continue
            if src.is_file():
                if src.suffix not in EXCLUDE_SUFFIXES:
                    zf.write(src, item)
                    added += 1
            else:
                for fpath in src.rglob("*"):
                    if fpath.is_file():
                        # Skip by suffix
                        if fpath.suffix in EXCLUDE_SUFFIXES:
                            continue
                        # Skip by directory name
                        if any(p.name in EXCLUDE_DIRS for p in fpath.parents):
                            continue
                        arcname = str(fpath.relative_to(local_root))
                        zf.write(fpath, arcname)
                        added += 1
    size_mb = zip_path.stat().st_size / (1024 ** 2)
    print(f"  Zipped {added} files → {size_mb:.1f} MB")


# ── Actions ───────────────────────────────────────────────────────────────────

def deploy():
    """Full deploy: zip → upload → unzip → start pipeline + backup."""
    local_root = Path(__file__).resolve().parent

    print("=" * 60)
    print("DEPLOY → Molab")
    print(f"  Target : {USER}@{HOST}:{PORT}")
    print(f"  Remote : {REMOTE_WS}")
    print("=" * 60)

    # ── 1. Build zip ─────────────────────────────────────────────
    print("\n[1/5] Building zip archive...")
    with tempfile.TemporaryDirectory() as tmp:
        zip_path = Path(tmp) / "cat_deploy.zip"
        _build_zip(local_root, zip_path)

        # ── 2. Connect ───────────────────────────────────────────
        print("\n[2/5] Connecting to Molab...")
        ssh = _ssh_client()
        print(f"  ✅ Connected to {HOST}:{PORT}")

        # ── 3. Upload ────────────────────────────────────────────
        print(f"\n[3/5] Uploading archive ({zip_path.stat().st_size / 1024:.0f} KB)...")
        _exec(ssh, f"mkdir -p {REMOTE_WS}", show_output=False)
        sftp = ssh.open_sftp()
        remote_zip = f"{REMOTE_WS}/cat_deploy.zip"
        sftp.put(str(zip_path), remote_zip)
        sftp.close()
        print("  ✅ Upload complete")

    # ── 4. Unzip ─────────────────────────────────────────────────
    print(f"\n[4/5] Extracting on Molab...")
    _exec(ssh, f"cd {REMOTE_WS} && unzip -o -q {remote_zip} && rm {remote_zip}")
    print("  ✅ Extracted")

    # ── 5. Launch processes ───────────────────────────────────────
    print("\n[5/5] Starting pipeline and backup daemon...")

    # Kill any stale runs first
    _exec(ssh, "pkill -f 'run_rq2_part1.py' 2>/dev/null; pkill -f 'auto_backup.py' 2>/dev/null",
          show_output=False)

    # Install dependencies (in case this is a fresh session after re-deploy)
    req_remote = f"{REMOTE_WS}/requirements_molab.txt"
    _exec(ssh,
          f"[ -f {req_remote} ] && pip install -q -r {req_remote} || true",
          show_output=False)

    # Launch pipeline
    _exec(ssh, PIPELINE_CMD, show_output=False)
    # Launch backup daemon
    _exec(ssh, BACKUP_CMD, show_output=False)

    # Confirm processes started
    _, procs, _ = _exec(ssh, "pgrep -fa 'run_rq2_part1|auto_backup'", show_output=False)
    print("  Running processes:")
    for line in procs.strip().splitlines():
        print(f"    {line}")

    ssh.close()

    print("\n" + "=" * 60)
    print("✅ DEPLOYMENT SUCCESSFUL")
    print(f"   Pipeline log : ssh root@{HOST} -p {PORT}")
    print(f'                  "tail -f {REMOTE_WS}/pipeline_run.log"')
    print(f"   Or run       : python deploy_molab.py --logs")
    print("=" * 60)


def show_logs():
    """Tail the remote pipeline log live (Ctrl-C to stop)."""
    print(f"Tailing {REMOTE_WS}/pipeline_run.log  (Ctrl-C to stop)\n")
    try:
        subprocess.run([
            "ssh", "-p", str(PORT),
            "-o", "StrictHostKeyChecking=no",
            f"{USER}@{HOST}",
            f"tail -f {REMOTE_WS}/pipeline_run.log",
        ])
    except KeyboardInterrupt:
        print("\nDetached from log stream.")


def show_status():
    """Print a quick status snapshot from the remote machine."""
    ssh = _ssh_client()
    print("=" * 60)
    print("REMOTE STATUS")
    print("=" * 60)

    print("\n── GPU ─────────────────────────────────────────────────")
    _exec(ssh, "nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total "
               "--format=csv,noheader,nounits 2>/dev/null || echo 'nvidia-smi not available'")

    print("\n── Ollama ──────────────────────────────────────────────")
    _exec(ssh, "curl -sf http://localhost:11434/api/tags | python3 -c \""
               "import sys,json; d=json.load(sys.stdin); "
               "[print(' ',m['name']) for m in d.get('models',[])]\"")

    print("\n── Processes ───────────────────────────────────────────")
    _exec(ssh, "pgrep -fa 'run_rq2_part1|auto_backup|ollama serve'")

    print("\n── Last 20 lines of pipeline log ───────────────────────")
    _exec(ssh, f"tail -20 {REMOTE_WS}/pipeline_run.log 2>/dev/null || echo '(no log yet)'")

    print("\n── Disk ────────────────────────────────────────────────")
    _exec(ssh, f"df -h {REMOTE_WS}")

    print("\n── Current tunnel URL ──────────────────────────────────")
    tunnel = _load_current_tunnel(ssh)
    if tunnel:
        h, p = tunnel
        print(f"  ssh root@{h} -p {p}  (password: {PASSWORD})")
        if h != HOST or p != PORT:
            print(f"\n  ⚠️  Tunnel has refreshed since your last deploy!")
            print(f"  Update deploy_molab.py:  HOST = \"{h}\"  PORT = {p}")
            _update_self(h, p)
    else:
        print("  Could not read tunnel_url.txt")

    ssh.close()


def show_tunnel():
    """
    Fetch the current Pinggy tunnel URL from Molab and update
    HOST/PORT in this file automatically so you're always connected.
    Uses the HOST/PORT currently in the script to connect once,
    then reads the latest tunnel_url.txt.
    """
    print("Connecting to fetch current tunnel URL...")
    ssh = _ssh_client()
    tunnel = _load_current_tunnel(ssh)
    ssh.close()

    if not tunnel:
        print("Could not read /workspace/tunnel_url.txt from remote.")
        print("Make sure molab_setup.py is still running.")
        return

    h, p = tunnel
    print(f"\nCurrent tunnel: ssh root@{h} -p {p}")
    print(f"Password: {PASSWORD}")

    if h != HOST or p != PORT:
        print("\nTunnel has changed — updating HOST/PORT in deploy_molab.py...")
        _update_self(h, p)
        print("✅ deploy_molab.py updated. You can now run any --flags normally.")
    else:
        print("HOST/PORT are already up to date.")


def _update_self(new_host: str, new_port: int):
    """Rewrite HOST and PORT in this script file."""
    self_path = Path(__file__).resolve()
    content = self_path.read_text(encoding="utf-8")
    import re
    content = re.sub(r'^HOST\s*=\s*".*?"', f'HOST = "{new_host}"', content, flags=re.MULTILINE)
    content = re.sub(r'^PORT\s*=\s*\d+',   f'PORT = {new_port}',  content, flags=re.MULTILINE)
    self_path.write_text(content, encoding="utf-8")
    print(f"  Updated: HOST = \"{new_host}\"  PORT = {new_port}")


def download_results():
    """Pull results/checkpoints/reports back to local machine."""
    import paramiko
    from pathlib import Path as P

    local_root = P(__file__).resolve().parent
    ssh = _ssh_client()

    DOWNLOAD_DIRS = [
        "rq2_part1/results",
        "rq2_part1/checkpoints",
        "rq2_part1/reports",
        "rq2_part1/plots",
        "ttc-frugalreason-poc/experiment_fr/results",
        "ttc-frugalreason-poc/experiment_fr/reports",
        "ttc-task-poc/experiment/results",
    ]

    print("Downloading results from Molab...\n")
    sftp = ssh.open_sftp()

    def _download_dir(remote_dir: str, local_dir: P):
        """Recursively download a remote directory."""
        local_dir.mkdir(parents=True, exist_ok=True)
        try:
            entries = sftp.listdir_attr(remote_dir)
        except FileNotFoundError:
            print(f"  [skip] {remote_dir} not found on remote")
            return 0
        count = 0
        for entry in entries:
            rpath = f"{remote_dir}/{entry.filename}"
            lpath = local_dir / entry.filename
            import stat
            if stat.S_ISDIR(entry.st_mode):
                count += _download_dir(rpath, lpath)
            else:
                sftp.get(rpath, str(lpath))
                print(f"  ← {rpath}")
                count += 1
        return count

    total = 0
    for rel_dir in DOWNLOAD_DIRS:
        total += _download_dir(
            f"{REMOTE_WS}/{rel_dir}",
            local_root / rel_dir,
        )

    sftp.close()
    ssh.close()
    print(f"\n✅ Downloaded {total} files to {local_root}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _check_config():
    if HOST == "REPLACE_WITH_PINGGY_HOST":
        print("ERROR: You must set HOST and PORT in deploy_molab.py first.")
        print("  Open the file and fill in the HOST / PORT from your Pinggy tunnel log.")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Deploy and manage Cost-Aware-Test-Time on Molab"
    )
    parser.add_argument("--logs",     action="store_true", help="Tail pipeline log live")
    parser.add_argument("--status",   action="store_true", help="Show remote status snapshot")
    parser.add_argument("--download", action="store_true", help="Pull results to local machine")
    parser.add_argument("--tunnel",   action="store_true", help="Fetch + auto-update current tunnel URL")
    args = parser.parse_args()

    _check_config()

    if args.logs:
        show_logs()
    elif args.status:
        show_status()
    elif args.download:
        download_results()
    elif args.tunnel:
        show_tunnel()
    else:
        deploy()
