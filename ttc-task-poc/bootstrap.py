#!/usr/bin/env python3
"""
bootstrap.py — Automated environment setup for the TTC-Task POC.

Run this ONCE before run_poc.py. It will:
  1. Check Python, pip, git
  2. Clone required repositories (tree-of-thought-llm, prm800k)
  3. Create virtual environment and install dependencies
  4. Check Ollama and pull models
  5. Audit cloned repos
  6. Generate SETUP_REPORT.md and REPO_AUDIT.md
"""

import os
import sys
import subprocess
import shutil
import json
import time
from datetime import datetime, timezone
from pathlib import Path

# Fix Windows SSLKEYLOGFILE permission issue and console encoding
os.environ.pop("SSLKEYLOGFILE", None)
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ── Paths ────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent                         # ttc-task-poc/
EXPERIMENT_DIR = SCRIPT_DIR / "experiment"
REPORTS_DIR = EXPERIMENT_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

REPOS = {
    "tree-of-thought-llm": {
        "url": "https://github.com/princeton-nlp/tree-of-thought-llm.git",
        "path": SCRIPT_DIR / "tree-of-thought-llm",
    },
    "prm800k": {
        "url": "https://github.com/openai/prm800k.git",
        "path": SCRIPT_DIR / "prm800k",
    },
}

VENV_DIR = EXPERIMENT_DIR / ".venv"
REQUIREMENTS = EXPERIMENT_DIR / "requirements.txt"

issues = []  # (severity, component, message)


def log(msg):
    print(f"[SETUP] {msg}")


def add_issue(severity, component, message, cause="", fix=""):
    ts = datetime.now(timezone.utc).isoformat()
    issues.append({
        "timestamp": ts,
        "severity": severity,
        "component": component,
        "message": message,
        "cause": cause,
        "fix": fix,
    })
    tag = "!" if severity == "WARNING" else ("X" if severity == "CRITICAL" else "i")
    print(f"  {tag} [{severity}] {component}: {message}")


# ======================================================================
# 1. Environment checks
# ======================================================================

def check_env():
    log("Checking environment...")
    # Python
    py_version = sys.version
    log(f"  Python: {py_version}")

    # pip
    try:
        subprocess.run([sys.executable, "-m", "pip", "--version"],
                       capture_output=True, check=True)
        log("  pip: OK")
    except Exception:
        add_issue("WARNING", "pip", "pip not found", "Missing pip", "Install pip")

    # git
    try:
        r = subprocess.run(["git", "--version"], capture_output=True, check=True, text=True)
        log(f"  git: {r.stdout.strip()}")
    except Exception:
        add_issue("CRITICAL", "git", "git not found", "Missing git", "Install git")

    # Internet
    try:
        import urllib.request
        req = urllib.request.Request("https://github.com", headers={"User-Agent": "Mozilla/5.0"})
        urllib.request.urlopen(req, timeout=10)
        log("  Internet: OK")
    except Exception:
        add_issue("WARNING", "internet", "No internet connectivity",
                  "Network issue", "Check connection; repos won't clone")


# ======================================================================
# 2. Clone repos
# ======================================================================

def clone_repos():
    log("Cloning repositories...")
    clone_status = {}

    for name, info in REPOS.items():
        url = info["url"]
        path = info["path"]

        if path.exists():
            # Check if it's a valid git repo
            if (path / ".git").exists():
                log(f"  {name}: already exists and valid [OK]")
                clone_status[name] = "exists"
                continue
            else:
                # Rename broken
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                broken_name = f"{name}.broken_{ts}"
                broken_path = path.parent / broken_name
                log(f"  {name}: exists but invalid, renaming to {broken_name}")
                shutil.move(str(path), str(broken_path))
                add_issue("WARNING", name, f"Renamed invalid repo to {broken_name}",
                          "Missing .git directory", "Cloning fresh")

        # Clone
        try:
            log(f"  Cloning {name} from {url}...")
            subprocess.run(
                ["git", "clone", "--depth", "1", url, str(path)],
                check=True, capture_output=True, text=True, timeout=120,
            )
            log(f"  {name}: cloned [OK]")
            clone_status[name] = "cloned"
        except subprocess.TimeoutExpired:
            add_issue("WARNING", name, "Clone timed out (120s)",
                      "Slow network", "Try manual clone")
            clone_status[name] = "timeout"
        except Exception as exc:
            add_issue("WARNING", name, f"Clone failed: {exc}",
                      "Network or git issue", "Clone manually or continue in degraded mode")
            clone_status[name] = "failed"

    return clone_status


# ======================================================================
# 3. Audit repos
# ======================================================================

def audit_repos(clone_status):
    log("Auditing repositories...")
    audit_lines = [
        "# Repository Audit Report\n\n",
        f"**Generated**: {datetime.now(timezone.utc).isoformat()}\n\n",
    ]

    # ── tree-of-thought-llm ──────────────────────────────────────────
    tot_path = REPOS["tree-of-thought-llm"]["path"]
    audit_lines.append("## tree-of-thought-llm (Princeton NLP)\n\n")
    audit_lines.append(f"- Clone status: `{clone_status.get('tree-of-thought-llm', 'unknown')}`\n")
    audit_lines.append(f"- Path: `{tot_path}`\n\n")

    if tot_path.exists():
        # Look for Game of 24 prompts
        game24_files = list(tot_path.rglob("*game24*")) + list(tot_path.rglob("*24*"))
        prompt_files = list(tot_path.rglob("*prompt*"))
        src_files = list(tot_path.rglob("*.py"))

        audit_lines.append("### Key Files Found\n\n")
        for f in game24_files[:10]:
            audit_lines.append(f"- Game24 related: `{f.relative_to(tot_path)}`\n")
        for f in prompt_files[:10]:
            audit_lines.append(f"- Prompt file: `{f.relative_to(tot_path)}`\n")
        audit_lines.append(f"- Total Python files: {len(src_files)}\n")

        audit_lines.append("\n### What Will Be Used\n\n")
        audit_lines.append("- Game of 24 problem format and prompt structure (reference only)\n")
        audit_lines.append("- Zero-shot ToT-BFS approach from Appendix B.1\n")
        audit_lines.append("- **NOT used**: OpenAI API calls, any paid service\n\n")
    else:
        audit_lines.append("- **Not available** — using embedded prompts as fallback\n\n")

    # ── prm800k ──────────────────────────────────────────────────────
    prm_path = REPOS["prm800k"]["path"]
    audit_lines.append("## prm800k (OpenAI)\n\n")
    audit_lines.append(f"- Clone status: `{clone_status.get('prm800k', 'unknown')}`\n")
    audit_lines.append(f"- Path: `{prm_path}`\n\n")

    if prm_path.exists():
        math_files = list(prm_path.rglob("*math*")) + list(prm_path.rglob("*test*"))
        json_files = list(prm_path.rglob("*.json")) + list(prm_path.rglob("*.jsonl"))

        audit_lines.append("### Key Files Found\n\n")
        for f in (math_files + json_files)[:15]:
            audit_lines.append(f"- `{f.relative_to(prm_path)}`\n")

        audit_lines.append("\n### What Will Be Used\n\n")
        audit_lines.append("- ORM evaluation structure (reference for verifier design)\n")
        audit_lines.append("- **NOT used**: PRM training, MATH test subset (we use GSM8K instead)\n")
        audit_lines.append("- **NOT trained**: No reward model training in this POC\n\n")
    else:
        audit_lines.append("- **Not available** — ORM proxy implemented independently\n\n")

    with open(REPORTS_DIR / "REPO_AUDIT.md", "w", encoding="utf-8") as f:
        f.writelines(audit_lines)
    log("  Wrote REPO_AUDIT.md")


# ======================================================================
# 4. Virtual environment & dependencies
# ======================================================================

def setup_venv():
    log("Setting up virtual environment...")

    venv_python = VENV_DIR / ("Scripts" if os.name == "nt" else "bin") / "python"
    if os.name == "nt":
        venv_python = venv_python.with_suffix(".exe")

    if not VENV_DIR.exists():
        try:
            subprocess.run(
                [sys.executable, "-m", "venv", str(VENV_DIR)],
                check=True, capture_output=True, text=True,
            )
            log(f"  Created venv at {VENV_DIR}")
        except Exception as exc:
            add_issue("WARNING", "venv", f"Could not create venv: {exc}",
                      "Missing venv module", "Install with --user instead")
            return _install_user()
    else:
        log(f"  venv already exists at {VENV_DIR}")

    # Install deps
    if REQUIREMENTS.exists():
        try:
            subprocess.run(
                [str(venv_python), "-m", "pip", "install", "-r", str(REQUIREMENTS)],
                check=True, capture_output=True, text=True, timeout=300,
            )
            log("  Dependencies installed in venv [OK]")
        except Exception as exc:
            add_issue("WARNING", "pip_install", f"Failed: {exc}",
                      "Dependency issue", "Try manual pip install")
            return _install_user()
    return True


def _install_user():
    """Fallback: install to user site-packages."""
    log("  Falling back to --user install...")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--user",
             "-r", str(REQUIREMENTS)],
            check=True, capture_output=True, text=True, timeout=300,
        )
        log("  Dependencies installed with --user [OK]")
        return True
    except Exception as exc:
        add_issue("WARNING", "pip_install_user", f"Failed: {exc}",
                  "pip issue", "Install dependencies manually")
        return False


# ======================================================================
# 5. Ollama check & model pull
# ======================================================================

def check_ollama():
    log("Checking Ollama...")

    try:
        import urllib.request
        resp = urllib.request.urlopen("http://localhost:11434/api/tags", timeout=5)
        data = json.loads(resp.read())
        models = [m["name"] for m in data.get("models", [])]
        log(f"  Ollama reachable. Models installed: {models}")
    except Exception:
        log("  Ollama not reachable. Attempting to start...")
        try:
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            time.sleep(5)
            resp = urllib.request.urlopen("http://localhost:11434/api/tags", timeout=5)
            data = json.loads(resp.read())
            models = [m["name"] for m in data.get("models", [])]
            log(f"  Ollama started. Models: {models}")
        except Exception as exc:
            add_issue("CRITICAL", "ollama", f"Cannot reach Ollama: {exc}",
                      "Ollama not installed or not running",
                      "Install from https://ollama.com and run `ollama serve`")
            return "BLOCKED"

    preferred = "qwen2.5:3b"
    fallback = "llama3.2:3b"

    models_lower = [m.lower() for m in models]

    def has_model(name):
        for m in models_lower:
            if name.lower() in m:
                return True
        return False

    if has_model(preferred):
        log(f"  Preferred model {preferred} found [OK]")
        return "READY"
    elif has_model(fallback):
        log(f"  Fallback model {fallback} found [OK]")
        add_issue("INFO", "model", f"Using fallback model {fallback}")
        return "READY"
    else:
        log(f"  Pulling {preferred}...")
        try:
            subprocess.run(["ollama", "pull", preferred], check=True, timeout=600)
            log(f"  {preferred} pulled [OK]")
            return "READY"
        except Exception:
            log(f"  Pull {preferred} failed. Trying {fallback}...")
            try:
                subprocess.run(["ollama", "pull", fallback], check=True, timeout=600)
                log(f"  {fallback} pulled [OK]")
                return "READY"
            except Exception as exc:
                add_issue("CRITICAL", "model_pull", f"Cannot pull any model: {exc}",
                          "Network or Ollama issue", "Manually run `ollama pull qwen2.5:3b`")
                return "BLOCKED"


# ======================================================================
# 6. Write setup report
# ======================================================================

def write_setup_report(ollama_status):
    if any(i["severity"] == "CRITICAL" for i in issues):
        verdict = "BLOCKED"
    elif any(i["severity"] == "WARNING" for i in issues):
        verdict = "READY WITH WARNINGS"
    else:
        verdict = "READY TO RUN"

    # Override if Ollama is blocked
    if ollama_status == "BLOCKED":
        verdict = "BLOCKED"

    lines = [
        "# Setup Report\n\n",
        f"**Generated**: {datetime.now(timezone.utc).isoformat()}\n\n",
        f"## Verdict: **{verdict}**\n\n",
        "## Environment\n\n",
        f"- Python: `{sys.version}`\n",
        f"- Platform: `{sys.platform}`\n",
        f"- Ollama: `{ollama_status}`\n\n",
    ]

    if issues:
        lines.append("## Issues\n\n")
        lines.append("| Timestamp | Severity | Component | Message | Fix |\n")
        lines.append("|-----------|----------|-----------|---------|-----|\n")
        for i in issues:
            lines.append(
                f"| {i['timestamp']} | {i['severity']} | {i['component']} "
                f"| {i['message']} | {i['fix']} |\n"
            )
    else:
        lines.append("## Issues\n\nNo issues detected.\n")

    lines.append(f"\n## Next Step\n\n")
    if verdict == "READY TO RUN":
        lines.append("```bash\ncd ttc-task-poc/experiment\npython run_poc.py\n```\n")
    elif verdict == "READY WITH WARNINGS":
        lines.append("Review warnings above, then:\n")
        lines.append("```bash\ncd ttc-task-poc/experiment\npython run_poc.py\n```\n")
    else:
        lines.append("Fix CRITICAL issues above before running.\n")

    with open(REPORTS_DIR / "SETUP_REPORT.md", "w", encoding="utf-8") as f:
        f.writelines(lines)

    # Also write ISSUES.md
    if issues:
        with open(REPORTS_DIR / "ISSUES.md", "w", encoding="utf-8") as f:
            f.write("# Issues Log\n\n")
            for i in issues:
                f.write(
                    f"- **[{i['severity']}]** `{i['component']}`: {i['message']}\n"
                    f"  - Cause: {i['cause']}\n"
                    f"  - Fix: {i['fix']}\n"
                    f"  - Time: {i['timestamp']}\n\n"
                )

    log(f"\n{'='*60}")
    log(f"  SETUP VERDICT: {verdict}")
    log(f"{'='*60}\n")
    return verdict


# ======================================================================
# MAIN
# ======================================================================

def main():
    print("=" * 60)
    print("  TTC-TASK POC — AUTOMATED BOOTSTRAP")
    print("=" * 60)

    # Create remaining directories
    for d in [
        EXPERIMENT_DIR / "results" / "raw_logs",
        EXPERIMENT_DIR / "results" / "parsed",
        EXPERIMENT_DIR / "results" / "summary",
        EXPERIMENT_DIR / "plots",
        EXPERIMENT_DIR / "logs",
        EXPERIMENT_DIR / "self_consistency_prompts",
    ]:
        d.mkdir(parents=True, exist_ok=True)

    # Also create top-level self_consistency_prompts
    (SCRIPT_DIR / "self_consistency_prompts").mkdir(parents=True, exist_ok=True)

    check_env()
    clone_status = clone_repos()
    audit_repos(clone_status)
    setup_venv()
    ollama_status = check_ollama()
    verdict = write_setup_report(ollama_status)

    return 0 if verdict != "BLOCKED" else 1


if __name__ == "__main__":
    sys.exit(main())
