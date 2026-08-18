# Cost-Aware-Test-Time — Molab Deployment Guide

New architecture. No SSH. No tunnels. No local PC required to run the experiment.

---

## How it works

```
Kaggle notebook (free GPU)
  └── Runs Ollama + qwen2.5:3b
  └── Exposes it via ngrok public URL
           ↓  HTTP
Molab notebook (free GPU)
  └── You upload the zip directly in the browser
  └── molab_run.py sets the ngrok URL and runs the pipeline
  └── Checkpoints pushed to Google Drive after every question
           ↓
Google Drive
  └── gdrive:Molab_Backups/Cost-Aware-Test-Time/checkpoints/
  └── Persists across all sessions — never lose progress
```

---

## Files

| File | Purpose |
|------|---------|
| `kaggle_ollama_server.ipynb` | Run on Kaggle — serves the LLM over HTTP |
| `molab_run.py` | Run on Molab — installs deps, connects to Kaggle, runs pipeline |
| `auto_backup.py` | Hourly full backup to Google Drive (optional but recommended) |
| `drive_checkpoint.py` | Live checkpoint push after every question (automatic) |
| `requirements_molab.txt` | All Python dependencies |
| `Cost-Aware-Test-Time-upload.zip` | Upload this to Molab directly |

---

## One-time setup

### A. Get a free ngrok account
1. Sign up at https://dashboard.ngrok.com (free)
2. Copy your **authtoken** from the dashboard
3. In Kaggle → Add-ons → Secrets → add a secret:
   - Name: `NGROK_TOKEN`
   - Value: your authtoken

### B. Configure Google Drive backup (optional)
1. Install rclone locally: https://rclone.org/downloads/
2. Run `rclone config` → create remote named `gdrive` → Google Drive → follow browser auth
3. Open your rclone.conf:
   - Windows: `%APPDATA%\rclone\rclone.conf`
   - Linux/Mac: `~/.config/rclone/rclone.conf`
4. Copy the `[gdrive]` block
5. Open `auto_backup.py` → paste into `RCLONE_CONF = """..."""`

If you skip this, the pipeline still runs and checkpoints locally — you just lose progress when the Molab session ends.

---

## Every session — step by step

### Step 1 — Start the Kaggle Ollama server

1. Open `kaggle_ollama_server.ipynb` in Kaggle
2. Set GPU runtime (T4 x2 recommended)
3. Run all cells in order
4. Wait for Cell 5 output:
   ```
   Public URL: https://a1b2-34-56-78-90.ngrok-free.app
   ```
5. **Copy that URL** — you need it in Step 2

Keep this notebook running the entire time.

### Step 2 — Upload project to Molab

1. Open a new Molab workspace
2. Upload `Cost-Aware-Test-Time-upload.zip` using the file browser
3. In a Molab cell, extract it:
   ```python
   import zipfile, os
   with zipfile.ZipFile("Cost-Aware-Test-Time-upload.zip", "r") as z:
       z.extractall(".")
   print("Extracted.")
   ```

### Step 3 — Configure and run

1. Open `molab_run.py`
2. Set your ngrok URL:
   ```python
   OLLAMA_BASE_URL = "https://a1b2-34-56-78-90.ngrok-free.app"
   ```
3. Run the file in a cell:
   ```python
   exec(open("molab_run.py").read())
   ```
   Or from terminal:
   ```bash
   python molab_run.py
   ```

The pipeline starts immediately. You see live output in the cell.

---

## Resuming after a session dies

The pipeline checkpoints to Google Drive after **every single completed question**.

When your Molab session ends:
1. Start a new Molab session
2. Upload and extract the zip again (same zip, it includes the code not the results)
3. Run `molab_run.py` with the same ngrok URL (or new one if Kaggle restarted)
4. Pipeline calls `drive_checkpoint.init()` on startup → pulls `completed.jsonl` from Drive → skips all already-done work

**Worst case loss: one question** (the one that was mid-execution when the session died).

### Drive layout
```
gdrive:Molab_Backups/Cost-Aware-Test-Time/
  checkpoints/
    completed.jsonl          ← pushed after every question
  results/
    question_strategy_summary.jsonl
    raw_model_calls.jsonl
    progress_log.jsonl
  rq2_part1/results/         ← full results (hourly via auto_backup.py)
  rq2_part1/reports/
  rq2_part1/plots/
```

---

## Monitoring from Molab interface

Everything runs in-process in your Molab cell — output streams live.

To check progress mid-run in another cell:
```python
# How many questions done?
import json
done = sum(1 for _ in open("rq2_part1/checkpoints/completed.jsonl"))
print(f"Completed: {done} / 540 strategy-runs")

# Last 10 log lines
import subprocess
subprocess.run("tail -10 rq2_part1/results/progress_log.jsonl", shell=True)
```

---

## Troubleshooting

**Ollama not reachable:**
- Make sure kaggle_ollama_server.ipynb Cell 6 (keep-alive) is still running
- The ngrok URL changes if you restart the Kaggle notebook — get the new one from Cell 5

**Pipeline crashes on dataset download:**
- HuggingFace downloads need internet — Molab has internet by default
- If StrategyQA fails, the pipeline falls back to ARC-Challenge automatically

**Drive sync not working:**
- Check that `RCLONE_CONF` in `auto_backup.py` has real tokens, not placeholders
- Run in a Molab cell: `!rclone lsd gdrive: --max-depth 1`

**Out of disk space on Molab:**
- Results are small (< 50 MB total) — disk is not the issue
- If you see disk errors, check `/tmp` usage: `!df -h`

---

## Key numbers

| Thing | Value |
|-------|-------|
| Total strategy-runs | 540 (3 tasks × 36 questions × 5 strategies) |
| Model calls per question | up to 20 (across all 5 strategies) |
| Checkpoint granularity | After every strategy-run (1/540) |
| Model | qwen2.5:3b (~2 GB VRAM) |
| Kaggle session limit | ~9h GPU / 12h CPU |
| Molab session limit | ~6-12h depending on tier |
| ngrok free tier | 1 tunnel, no time limit |
