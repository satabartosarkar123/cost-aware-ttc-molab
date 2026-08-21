import os
import sys
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm

os.environ.pop("SSLKEYLOGFILE", None)
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import yaml
from core.task_loader import load_all_tasks
from core.hardware_monitor import HardwareMonitor

from strategies.greedy_io import GreedyIO
from strategies.greedy_cot import GreedyCoT
from strategies.self_consistency import SelfConsistency
from strategies.best_of_n import BestOfN
from strategies.tree_of_thought import TreeOfThought

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# Paths for 50-question run
RESULTS_DIR = Path("results_50")
RAW_LOGS_DIR = RESULTS_DIR / "raw_logs"
PARSED_DIR = RESULTS_DIR / "parsed"
SUMMARY_DIR = RESULTS_DIR / "summary"
PLOTS_DIR = Path("plots_50")
REPORTS_DIR = Path("reports_50")

for d in [RAW_LOGS_DIR, PARSED_DIR, SUMMARY_DIR, PLOTS_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

def append_jsonl(path, record):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")

def run_experiment(config: dict, model_name: str, tasks: dict):
    from core.ollama_client import OllamaClient
    from core.verifier import OutcomeVerifier

    ollama_cfg = config.get("ollama", {})
    client = OllamaClient(
        base_url=ollama_cfg.get("base_url", "http://localhost:11434"),
        model=model_name,
        timeout=ollama_cfg.get("timeout_seconds", 60),
        max_retries=ollama_cfg.get("max_retries", 3),
        retry_delay=ollama_cfg.get("retry_delay_seconds", 2),
    )
    verifier = OutcomeVerifier(ollama_client=client)
    strat_cfgs = config.get("strategies", {})

    strategies = {
        "greedy_io": GreedyIO(client, strat_cfgs.get("greedy_io", {}), HardwareMonitor),
        "greedy_cot": GreedyCoT(client, strat_cfgs.get("greedy_cot", {}), HardwareMonitor),
        "self_consistency": SelfConsistency(client, strat_cfgs.get("self_consistency", {}), HardwareMonitor),
        "best_of_n": BestOfN(client, strat_cfgs.get("best_of_n", {}), HardwareMonitor, verifier),
        "tree_of_thought": TreeOfThought(client, strat_cfgs.get("tree_of_thought", {}), HardwareMonitor),
    }

    log_file = RAW_LOGS_DIR / f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.jsonl"
    all_records = []
    
    for strat_name, strategy in strategies.items():
        logger.info(f"Running strategy: {strat_name}")
        for task_name, examples in tasks.items():
            logger.info(f"  Task: {task_name} ({len(examples)} examples)")
            for ex in tqdm(examples, desc=f"{strat_name}/{task_name}"):
                question = ex["question"]
                gold = ex["gold_answer"]
                
                try:
                    result = strategy.run(task_name, question, gold, ex["id"], ex)
                except Exception as e:
                    logger.error(f"Strategy {strat_name} crashed on {ex['id']}: {e}")
                    # Crash fallback dictionary
                    result = {
                        "task": task_name,
                        "example_id": ex["id"],
                        "strategy": strat_name,
                        "correct": False,
                        "parse_success": False,
                        "error": str(e),
                        "output_text": "",
                        "prompt": ""
                    }

                # Enrich record
                result["timestamp"] = datetime.now(timezone.utc).isoformat()
                result["model_name"] = model_name

                # Flatten hardware energy
                hw_list = result.get("hardware_metrics", [])
                if hw_list:
                    energies = [h.get("energy_joules") for h in hw_list if h.get("energy_joules") is not None]
                    cpus = [h.get("avg_cpu_percent") for h in hw_list if h.get("avg_cpu_percent") is not None]
                    result["energy_joules"] = sum(energies) if energies else None
                    result["avg_cpu_percent"] = sum(cpus) / len(cpus) if cpus else None
                    result["hardware_type"] = hw_list[0].get("hardware_type", "unknown") if hw_list else "unknown"
                else:
                    result["energy_joules"] = None
                    result["avg_cpu_percent"] = None
                    result["hardware_type"] = "unknown"

                # Log continuously
                append_jsonl(log_file, result)
                all_records.append(result)

                status_icon = "[OK]" if result.get("correct") else "[X]"
                logger.info(
                    "    %s answer=%s gold=%s (%.1fs, %d calls)",
                    status_icon,
                    result.get("final_answer"),
                    gold,
                    result.get("latency_seconds", 0),
                    result.get("model_calls", 0),
                )

    return all_records


# ======================================================================
# Aggregation
# ======================================================================

def aggregate(records: List[Dict], config: dict):
    """Produce per-strategy CSVs and a summary CSV."""
    try:
        import pandas as pd
    except ImportError:
        logger.error("pandas not installed -- skipping aggregation.")
        return None

    df = pd.DataFrame(records)

    # ── Per-strategy parsed CSVs ─────────────────────────────────────
    cols_to_save = [
        "timestamp", "task", "example_id", "strategy", "model_name",
        "final_answer", "gold_answer", "correct", "parse_method",
        "parse_success", "latency_seconds", "total_prompt_tokens",
        "total_completion_tokens", "model_calls", "energy_joules",
        "avg_cpu_percent", "hardware_type", "error",
    ]
    available_cols = [c for c in cols_to_save if c in df.columns]

    for strat_name, sdf in df.groupby("strategy"):
        out_path = PARSED_DIR / f"{strat_name}_parsed.csv"
        sdf[available_cols].to_csv(out_path, index=False)
        logger.info("Saved %s", out_path)

    # ── Summary CSV ──────────────────────────────────────────────────
    summary_rows = []
    for (task, strat), g in df.groupby(["task", "strategy"]):
        n = len(g)
        row = {
            "task": task,
            "strategy": strat,
            "num_examples": n,
            "accuracy": g["correct"].mean() if "correct" in g else 0,
            "parse_rate": g["parse_success"].mean() if "parse_success" in g else 0,
            "avg_latency_seconds": g["latency_seconds"].mean() if "latency_seconds" in g else 0,
            "avg_total_tokens": (
                (g["total_prompt_tokens"].fillna(0) + g["total_completion_tokens"].fillna(0)).mean()
                if "total_prompt_tokens" in g else 0
            ),
            "avg_model_calls": g["model_calls"].mean() if "model_calls" in g else 0,
        }
        if "energy_joules" in g.columns:
            ej = g["energy_joules"].dropna()
            row["avg_energy_joules"] = ej.mean() if len(ej) > 0 else None
        else:
            row["avg_energy_joules"] = None

        # Oracle accuracy for SC and BoN
        if "oracle_correct" in g.columns:
            oc = g["oracle_correct"].dropna()
            row["oracle_accuracy"] = oc.mean() if len(oc) > 0 else None
        else:
            row["oracle_accuracy"] = None

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = SUMMARY_DIR / "summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info("Saved summary: %s", summary_path)

    return summary_df


# ======================================================================
# Plotting
# ======================================================================

def generate_plots(summary_df):
    """Generate publication-quality plots from the summary DataFrame."""
    if summary_df is None or summary_df.empty:
        logger.warning("No summary data -- skipping plots.")
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        logger.error("matplotlib/seaborn not installed -- skipping plots.")
        return

    sns.set_theme(style="whitegrid", font_scale=1.1)

    def _bar_plot(data, y_col, title, ylabel, filename):
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=data, x="task", y=y_col, hue="strategy", ax=ax)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Task")
        ax.legend(title="Strategy", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        fig.savefig(PLOTS_DIR / filename, dpi=150)
        plt.close(fig)
        logger.info("Saved plot: %s", filename)

    _bar_plot(summary_df, "accuracy", "Accuracy by Task x Strategy",
              "Accuracy", "accuracy_by_task_strategy.png")

    _bar_plot(summary_df, "parse_rate", "Parse Rate by Task x Strategy",
              "Parse Rate", "parse_rate_by_task_strategy.png")

    _bar_plot(summary_df, "avg_latency_seconds", "Latency by Task x Strategy",
              "Avg Latency (s)", "latency_by_task_strategy.png")

    _bar_plot(summary_df, "avg_total_tokens", "Token Usage by Task x Strategy",
              "Avg Total Tokens", "tokens_by_task_strategy.png")

    # Energy plot (only if data exists)
    if "avg_energy_joules" in summary_df.columns and summary_df["avg_energy_joules"].notna().any():
        _bar_plot(summary_df, "avg_energy_joules", "Energy by Task x Strategy",
                  "Avg Energy (Joules)", "energy_by_task_strategy.png")

    # Oracle vs actual accuracy
    oracle_data = summary_df[summary_df["oracle_accuracy"].notna()].copy()
    if not oracle_data.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        width = 0.35
        tasks = oracle_data["task"].unique()
        strategies = oracle_data["strategy"].unique()

        x_labels = []
        actual_vals = []
        oracle_vals = []
        for t in tasks:
            for s in strategies:
                row = oracle_data[(oracle_data["task"] == t) & (oracle_data["strategy"] == s)]
                if not row.empty:
                    x_labels.append(f"{t}\\n{s}")
                    actual_vals.append(row["accuracy"].values[0])
                    oracle_vals.append(row["oracle_accuracy"].values[0])

        if x_labels:
            import numpy as np
            x = np.arange(len(x_labels))
            ax.bar(x - width / 2, actual_vals, width, label="Actual Accuracy", color="#4C72B0")
            ax.bar(x + width / 2, oracle_vals, width, label="Oracle Accuracy", color="#55A868")
            ax.set_xticks(x)
            ax.set_xticklabels(x_labels, fontsize=9)
            ax.set_ylabel("Accuracy")
            ax.set_title("Oracle vs Actual Accuracy (SC & BoN)", fontsize=14, fontweight="bold")
            ax.legend()
            plt.tight_layout()
            fig.savefig(PLOTS_DIR / "oracle_vs_actual_accuracy.png", dpi=150)
            plt.close(fig)
            logger.info("Saved plot: oracle_vs_actual_accuracy.png")


# ======================================================================
# Parse-rate safety check
# ======================================================================

def check_parse_rates(records: List[Dict], threshold: float = 0.85):
    """Generate LOW_PARSE_RATE.md if any task x strategy combo is below threshold."""
    try:
        import pandas as pd
    except ImportError:
        return

    df = pd.DataFrame(records)
    if df.empty:
        return

    failures = []
    low_combos = []

    for (task, strat), g in df.groupby(["task", "strategy"]):
        rate = g["parse_success"].mean() if "parse_success" in g.columns else 0
        if rate < threshold:
            low_combos.append((task, strat, rate, len(g)))
            fails = g[g["parse_success"] == False].head(20) if "parse_success" in g.columns else g.head(0)
            for _, row in fails.iterrows():
                failures.append({
                    "task": task,
                    "strategy": strat,
                    "example_id": row.get("example_id"),
                    "raw_output": str(row.get("output_text", ""))[:500],
                    "parse_method": row.get("parse_method"),
                })

    if low_combos:
        lines = ["# Low Parse Rate Report\\n"]
        lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}\\n")
        lines.append(f"Threshold: {threshold}\\n\\n")
        lines.append("| Task | Strategy | Parse Rate | N |\\n")
        lines.append("|------|----------|-----------|---|\\n")
        for task, strat, rate, n in low_combos:
            lines.append(f"| {task} | {strat} | {rate:.2%} | {n} |\\n")

        lines.append("\\n## Example Failures\\n\\n")
        for f in failures[:20]:
            lines.append(f"### {f['task']} / {f['strategy']} / {f['example_id']}\\n")
            lines.append(f"Parse method: `{f['parse_method']}`\\n")
            lines.append(f"```\\n{f['raw_output']}\\n```\\n\\n")

        with open(REPORTS_DIR / "LOW_PARSE_RATE.md", "w", encoding="utf-8") as fp:
            fp.writelines(lines)
        logger.warning("Low parse rate detected! See reports/LOW_PARSE_RATE.md")

    if failures:
        fail_path = RAW_LOGS_DIR / "parse_failures.jsonl"
        with open(fail_path, "w", encoding="utf-8") as fp:
            for f in failures:
                fp.write(json.dumps(f, default=str) + "\\n")


# ======================================================================
# Report generation
# ======================================================================

def generate_report(summary_df, records, config, model_name, hw_info):
    """Generate the final POC_REPORT.md."""
    if summary_df is None:
        return

    lines = [
        "# TTC-Task POC - Experiment Report\\n\\n",
        f"**Generated**: {datetime.now(timezone.utc).isoformat()}\\n\\n",
        "## 1. Research Question\\n\\n",
        "> Across different reasoning task types, do different test-time compute (TTC) "
        "strategies produce different accuracy, cost, and energy behaviours when "
        "constrained to local, small-scale models?\\n\\n",
        f"**Model**: `{model_name}`\\n\\n",
        "## 2. Hardware\\n\\n",
        f"- GPU Available: {hw_info.get('gpu_available')}\\n",
        f"- GPU Name: {hw_info.get('gpu_name', 'N/A')}\\n",
        f"- GPU Memory: {hw_info.get('gpu_memory_mb', 'N/A')} MB\\n",
        f"- CPU Cores: {hw_info.get('cpu_count', 'N/A')}\\n",
        f"- RAM: {hw_info.get('ram_total_mb', 'N/A')} MB\\n\\n",
        "## 3. Methodology\\n\\n",
        "| Strategy | Paper | Details |\\n",
        "|----------|-------|---------|\\n",
        "| Greedy IO | Baseline | Standard prompt, T=0, 1 call |\\n",
        "| Greedy CoT (Zero-Shot) | ToT Appendix B.1 (Yao et al., 2023) | Zero-shot CoT prompt, T=0, 1 call |\\n",
        "| Self-Consistency | Wang et al., 2023 | Zero-shot CoT (ToT App B.1), T=0.7, k=5 (scale constrained), majority vote |\\n",
        "| Best-of-N (LLM Verifier) | proxy for Cobbe et al., 2021 | LLM-as-a-Judge Outcome Verifier proxy, k=5 |\\n",
        "| Tree-of-Thought | Yao et al., NeurIPS 2023 | Zero-shot BFS, k=3, algorithmic search (task-dependent) |\\n\\n",
        "## 4. Results\\n\\n",
        "### Accuracy\\n\\n",
    ]

    try:
        lines.append(summary_df.to_markdown(index=False))
    except ImportError:
        cols = list(summary_df.columns)
        lines.append("| " + " | ".join(cols) + " |\\n")
        lines.append("| " + " | ".join(["---"] * len(cols)) + " |\\n")
        for _, row in summary_df.iterrows():
            vals = [str(row[c]) if not isinstance(row[c], float) else f"{row[c]:.4f}" for c in cols]
            lines.append("| " + " | ".join(vals) + " |\\n")
    lines.append("\\n\\n")

    lines.append("## 5. Plots\\n\\n")
    plot_files = list(PLOTS_DIR.glob("*.png"))
    for pf in sorted(plot_files):
        lines.append(f"### {pf.stem.replace('_', ' ').title()}\\n")
        lines.append(f"![{pf.name}](../plots/{pf.name})\\n\\n")

    with open(REPORTS_DIR / "POC_REPORT.md", "w", encoding="utf-8") as fp:
        fp.writelines([L + "\\n" if not L.endswith("\\n") else L for L in lines])
    logger.info("Saved POC_REPORT.md")


def load_config():
    config_path = Path(__file__).resolve().parent / "core" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    logger.info("Initializing Test-Time Compute (TTC) POC...")
    
    config = load_config()
    model_name = config.get("ollama", {}).get("preferred_model", "qwen2.5:3b")
    
    hw_info = HardwareMonitor.detect_hardware()
    
    tasks = load_all_tasks(config)
    
    # 1. Run inference
    records = run_experiment(config, model_name, tasks)
    
    # 2. Check parses safely
    check_parse_rates(records, threshold=config.get("parse_rate_threshold", 0.85))
    
    # 3. Aggregate
    summary_df = aggregate(records, config)
    
    # 4. Generate plots
    try:
        generate_plots(summary_df)
    except Exception as e:
        logger.error(f"Plotting failed: {e}")
        
    # 5. Generate final report
    try:
        generate_report(summary_df, records, config, model_name, hw_info)
    except Exception as e:
        logger.error(f"Reporting failed: {e}")
        
    logger.info("POC pipeline complete!")

if __name__ == "__main__":
    main()
