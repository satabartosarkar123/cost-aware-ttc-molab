"""
sync_from_molab.py — Pull Molab results to your local PC.

Run this on your LOCAL PC after a Molab run finishes:
    python sync_from_molab.py

It does: git pull
That is literally it — all results, checkpoints, reports, and plots
that Molab pushed appear in your local folder instantly.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def pull():
    print("Pulling latest results from GitHub...")
    r = subprocess.run("git pull --ff-only", shell=True, cwd=str(ROOT))
    if r.returncode != 0:
        print("\nPull failed. Possible reasons:")
        print("  - You have uncommitted local changes (run: git status)")
        print("  - No internet connection")
        print("  - Molab hasn't pushed anything yet")
        sys.exit(1)
    print("\nDone. Latest Molab results are now in your local folder.")
    _show_summary()


def _show_summary():
    """Show what result files exist."""
    interesting = [
        "rq2_part1/results/cost_profile.csv",
        "rq2_part1/results/question_strategy_summary.jsonl",
        "rq2_part1/reports/FINAL_REPORT.md",
        "rq2_part1/checkpoints/completed.jsonl",
        "rq2_part1/plots",
    ]
    print("\nResult files:")
    for rel in interesting:
        p = ROOT / rel
        if p.exists():
            if p.is_dir():
                count = len(list(p.rglob("*")))
                print(f"  OK  {rel}/ ({count} files)")
            else:
                size = p.stat().st_size // 1024
                print(f"  OK  {rel} ({size} KB)")
        else:
            print(f"  --  {rel} (not yet)")

    ckpt = ROOT / "rq2_part1/checkpoints/completed.jsonl"
    if ckpt.exists():
        done = len(ckpt.read_text().strip().splitlines())
        print(f"\n  rq2_part1 progress: {done}/540 strategy-runs ({done/540*100:.1f}%)")


if __name__ == "__main__":
    pull()
