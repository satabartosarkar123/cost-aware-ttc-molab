"""
molab_setup.py — Minimal, bulletproof Molab setup.

Paste this into one Molab cell and run it. That's it.

What it does:
  1. Starts SSH server (so you can connect from your PC)
  2. Starts Pinggy tunnel (gives you the ssh command)
  3. Installs Ollama + pulls model
  4. Creates /workspace/Cost-Aware-Test-Time
  5. Keeps everything alive

After running, you get:
    ssh root@<HOST> -p <PORT>
    Password: molab2026

Then from your PC:
    1. Fill HOST/PORT in deploy_molab.py
    2. Run: python deploy_molab.py
    3. Pipeline starts running on the GPU
"""

import os
import subprocess
import time

PASSWORD = "molab2026"
MODEL = "qwen2.5:3b"

print("="*60)
print("  MOLAB SETUP")
print("="*60)

# ── Ensure /workspace exists ──────────────────────────────────────
os.makedirs("/workspace", exist_ok=True)
os.chdir("/workspace")

# ── Step 1: SSH server ────────────────────────────────────────────
print("\n[1/5] Setting up SSH...")
subprocess.run("apt-get update -qq && apt-get install -y -qq openssh-server zstd curl > /dev/null 2>&1", shell=True)
subprocess.run("mkdir -p /var/run/sshd", shell=True)
subprocess.run(f"echo 'root:{PASSWORD}' | chpasswd", shell=True)
subprocess.run("sed -i 's/^#*PermitRootLogin.*/PermitRootLogin yes/' /etc/ssh/sshd_config", shell=True)
subprocess.run("sed -i 's/^#*PasswordAuthentication.*/PasswordAuthentication yes/' /etc/ssh/sshd_config", shell=True)
subprocess.run("/usr/sbin/sshd", shell=True)
print("  ✅ SSH running")

# ── Step 2: Start Pinggy tunnel ──────────────────────────────────
print("\n[2/5] Starting Pinggy tunnel...")
subprocess.run("pkill -f 'tcp@a.pinggy.io' 2>/dev/null", shell=True)
time.sleep(2)
subprocess.Popen(
    "ssh -p 443 -R0:localhost:22 -o StrictHostKeyChecking=no tcp@a.pinggy.io > /workspace/pinggy.log 2>&1",
    shell=True
)
print("  Waiting for tunnel URL...")
time.sleep(10)

# Parse the tunnel URL
tunnel_host = None
tunnel_port = None
try:
    with open("/workspace/pinggy.log", "r") as f:
        for line in f:
            if "tcp://" in line and "pinggy" in line:
                url = line.split("tcp://")[-1].strip()
                if ":" in url:
                    h, p = url.rsplit(":", 1)
                    tunnel_host = h
                    tunnel_port = "".join(c for c in p if c.isdigit())
                    break
except:
    pass

if tunnel_host and tunnel_port:
    print(f"  ✅ Tunnel: ssh root@{tunnel_host} -p {tunnel_port}")
    with open("/workspace/tunnel_url.txt", "w") as f:
        f.write(f"{tunnel_host}:{tunnel_port}\n")
else:
    print("  ⚠️  Could not parse tunnel URL. Check /workspace/pinggy.log manually.")
    subprocess.run("cat /workspace/pinggy.log", shell=True)

# ── Step 3: Install Ollama ───────────────────────────────────────
print("\n[3/5] Installing Ollama...")
if subprocess.run("which ollama", shell=True, capture_output=True).returncode != 0:
    subprocess.run("curl -fsSL https://ollama.com/install.sh | sh", shell=True)
    time.sleep(3)
else:
    print("  Already installed")

# Find ollama binary
ollama_bin = "/usr/local/bin/ollama"
if not os.path.isfile(ollama_bin):
    ollama_bin = "/usr/bin/ollama"
if not os.path.isfile(ollama_bin):
    result = subprocess.run("which ollama", shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        ollama_bin = result.stdout.strip()

print(f"  Ollama binary: {ollama_bin}")

# Start Ollama server
subprocess.run("pkill -f 'ollama serve'", shell=True, stderr=subprocess.DEVNULL)
time.sleep(2)
subprocess.Popen(
    f"{ollama_bin} serve >> /workspace/ollama.log 2>&1",
    shell=True
)
print("  Starting Ollama server...")
time.sleep(10)

# Verify it's running
if subprocess.run("curl -sf http://localhost:11434/api/tags > /dev/null", shell=True).returncode == 0:
    print("  ✅ Ollama is running")
else:
    print("  ⚠️  Ollama may not be ready yet")

# ── Step 4: Pull model ────────────────────────────────────────────
print(f"\n[4/5] Pulling {MODEL}...")
result = subprocess.run(f"{ollama_bin} pull {MODEL}", shell=True)
if result.returncode == 0:
    print(f"  ✅ {MODEL} ready")
else:
    print(f"  ⚠️  Model pull failed (may need to retry manually)")

# ── Step 5: Create workspace ──────────────────────────────────────
print("\n[5/5] Setting up workspace...")
os.makedirs("/workspace/Cost-Aware-Test-Time", exist_ok=True)
print("  ✅ /workspace/Cost-Aware-Test-Time created")

# ── Install Python deps ───────────────────────────────────────────
print("\nInstalling Python dependencies...")
pkgs = (
    "requests datasets pandas numpy matplotlib seaborn tqdm "
    "pynvml psutil pyyaml tabulate paramiko reportlab scipy fpdf2"
)
subprocess.run(f"pip install -q --root-user-action=ignore {pkgs}", shell=True)
print("  ✅ Dependencies installed")

# ── Final summary ─────────────────────────────────────────────────
print("\n" + "="*60)
print("  ✅  SETUP COMPLETE")
print("="*60)
if tunnel_host and tunnel_port:
    print(f"\n  🔗 SSH: ssh root@{tunnel_host} -p {tunnel_port}")
    print(f"  🔑 Password: {PASSWORD}")
    print(f"\n  On your LOCAL PC:")
    print(f"    1. Edit deploy_molab.py:")
    print(f"         HOST = \"{tunnel_host}\"")
    print(f"         PORT = {tunnel_port}")
    print(f"    2. Run: python deploy_molab.py")
    print(f"    3. Monitor: python deploy_molab.py --logs")
else:
    print("\n  ⚠️  Tunnel setup incomplete. Get URL manually:")
    print("     cat /workspace/pinggy.log")

print("\n  📁 Workspace: /workspace/Cost-Aware-Test-Time")
print(f"  🤖 Model: {MODEL}")
print("\n  ⚠️  DO NOT STOP THIS CELL")
print("="*60)

# ── Keep-alive loop ───────────────────────────────────────────────
print("\nEntering keep-alive loop (prints every 5 min)...\n")
counter = 0
while True:
    time.sleep(300)
    counter += 1
    h = counter * 5 / 60
    
    # Check Ollama
    ollama_ok = subprocess.run("curl -sf http://localhost:11434/api/tags > /dev/null", shell=True).returncode == 0
    
    if not ollama_ok:
        print(f"[{h:.1f}h] ⚠️  Ollama died, restarting...")
        subprocess.Popen(f"{ollama_bin} serve >> /workspace/ollama.log 2>&1", shell=True)
        time.sleep(5)
    else:
        # Check if pipeline is running
        pipeline_running = subprocess.run("pgrep -f run_rq2_part1.py > /dev/null", shell=True).returncode == 0
        status = "RUNNING" if pipeline_running else "NOT STARTED"
        print(f"[{h:.1f}h] ✅ Ollama OK | Pipeline {status}")
