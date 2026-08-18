import json
import math
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom, norm

import sys
sys.path.append(str(Path(__file__).parent.parent))
from core.verifier import OutcomeVerifier

BASE_DIR = Path(__file__).parent.parent
RAW_LOG_PATH = BASE_DIR / "results" / "raw_logs" / "frugal_reason_v3_raw_seed0.jsonl"
BASELINE_PATH = BASE_DIR.parent.parent / "rq2_part1" / "results" / "cost_profile.csv"
SUMMARY_DIR = BASE_DIR / "results" / "summary"
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

def wilson_score_interval(successes, n, confidence=0.95):
    if n == 0: return 0.0, 0.0
    z = norm.ppf(1 - (1 - confidence) / 2)
    p = successes / n
    denominator = 1 + z**2 / n
    centre_adjusted_probability = p + z**2 / (2 * n)
    adjusted_standard_deviation = math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)
    lower_bound = (centre_adjusted_probability - z * adjusted_standard_deviation) / denominator
    upper_bound = (centre_adjusted_probability + z * adjusted_standard_deviation) / denominator
    return max(0, lower_bound), min(1, upper_bound)

def mcnemar_exact(b, c):
    return binom.test(int(b), int(b + c), 0.5).pvalue if (b + c) > 0 else 1.0

def paired_bootstrap(diffs, n_resamples=10000, seed=42):
    np.random.seed(seed)
    diffs = np.array(diffs)
    if len(diffs) == 0: return 0.0, 0.0
    resamples = np.random.choice(diffs, size=(n_resamples, len(diffs)), replace=True)
    means = np.mean(resamples, axis=1)
    return np.percentile(means, 2.5), np.percentile(means, 97.5)

def analyze_pav():
    if not RAW_LOG_PATH.exists():
        print(f"Error: {RAW_LOG_PATH} not found.")
        return
        
    records = []
    with open(RAW_LOG_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            records.append(json.loads(line))
            
    df = pd.DataFrame(records)
    if df.empty: return
    
    verifier = OutcomeVerifier()
    
    # Pre-calculate correctness for candidates if needed
    # But wait, df already has `correct` for the primary alpha=0.6. We need to re-evaluate for sweep
    
    def score_alpha(alpha: float, use_hidden: bool = False):
        correct_count = {task: 0 for task in df['task'].unique()}
        totals = {task: 0 for task in df['task'].unique()}
        
        for _, row in df.iterrows():
            if row['early_exit']:
                ans = row['selected_answer']
            else:
                best_S = -float('inf')
                best_a = None
                
                for cand in row['candidates']:
                    a = cand['answer']
                    prior_a = cand['prior']
                    if use_hidden and 'hidden_judge_scores' in row:
                        V_a = row['hidden_judge_scores'].get(str(a), cand['V'])
                    else:
                        V_a = cand['V']
                        
                    S_a = alpha * V_a + (1.0 - alpha) * math.log(prior_a + 1e-6)
                    
                    if S_a > best_S:
                        best_S = S_a
                        best_a = a
                    elif abs(S_a - best_S) < 1e-9:
                        # Find the prior of best_a
                        best_a_prior = next((c['prior'] for c in row['candidates'] if c['answer'] == best_a), 0)
                        if prior_a > best_a_prior:
                            best_a = a
                            best_S = S_a
                
                if best_a is None and row['candidates']:
                    best_a = row['candidates'][0]['answer']
                ans = best_a
                
            task = row['task']
            eval_res = verifier.score(task, "", str(ans), str(ans), row['gold_answer'])
            is_correct = eval_res["score"] == 1.0
            
            if is_correct:
                correct_count[task] += 1
            totals[task] += 1
            
        return {task: correct_count[task]/totals[task] for task in totals}

    # 1. Post-Hoc Alpha Sweep
    alphas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    alpha_results = {a: score_alpha(a) for a in alphas}
    
    tasks = df['task'].unique()
    for task in tasks:
        accs = [alpha_results[a][task] for a in alphas]
        plt.plot(alphas, accs, marker='o', label=task)
        
    plt.xlabel('Alpha')
    plt.ylabel('Accuracy')
    plt.title('Alpha vs Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(SUMMARY_DIR / "alpha_sweep.png")
    plt.close()
    
    # 2. Variants Table
    # (a) majority-vote-only (=alpha 0)
    acc_maj = score_alpha(0.0)
    # (b) judge-all-raw (no clustering, alpha 1). Here we use hidden judge scores (evaluate all)
    acc_judge_all = score_alpha(1.0, use_hidden=True)
    # (c) cluster-then-judge (alpha 1)
    acc_clust_judge = score_alpha(1.0)
    # (d) PAV primary (alpha 0.6)
    acc_primary = score_alpha(0.6)
    
    print("--- Variants Table ---")
    for task in tasks:
        print(f"[{task}] Majority-Vote: {acc_maj[task]:.2f} | Judge-All-Raw: {acc_judge_all[task]:.2f} | Cluster-then-Judge: {acc_clust_judge[task]:.2f} | PAV Primary: {acc_primary[task]:.2f}")
        
    # 3. Judge calibration AUROC
    print("\n--- Judge Calibration ---")
    for task in tasks:
        y_true = []
        y_score = []
        task_df = df[(df['task'] == task) & (df['early_exit'] == False)]
        for _, row in task_df.iterrows():
            if row['route_used'] == 'judge':
                for cand in row['candidates']:
                    ans = cand['answer']
                    # Was this answer actually correct?
                    eval_res = verifier.score(task, "", str(ans), str(ans), row['gold_answer'])
                    is_correct = eval_res["score"] == 1.0
                    y_true.append(1 if is_correct else 0)
                    y_score.append(cand['V'])
        
        if len(set(y_true)) > 1:
            # Manual AUROC calculation
            pos_scores = [s for t, s in zip(y_true, y_score) if t == 1]
            neg_scores = [s for t, s in zip(y_true, y_score) if t == 0]
            
            if len(pos_scores) > 0 and len(neg_scores) > 0:
                concordant = 0.0
                for p in pos_scores:
                    for n in neg_scores:
                        if p > n:
                            concordant += 1.0
                        elif p == n:
                            concordant += 0.5
                auc = concordant / (len(pos_scores) * len(neg_scores))
                print(f"[{task}] Judge AUROC: {auc:.3f}")
            else:
                print(f"[{task}] Judge AUROC: N/A (only one class)")
        else:
            print(f"[{task}] Judge AUROC: N/A (only one class)")

    # 4. Exit-rate histogram
    exit_counts = df.groupby(['task', 'early_exit']).size().unstack(fill_value=0)
    if True in exit_counts.columns and False in exit_counts.columns:
        exit_counts.plot(kind='bar', stacked=True)
        plt.title('Early Exit Rates by Task')
        plt.ylabel('Count')
        plt.savefig(SUMMARY_DIR / "early_exit_histogram.png")
        plt.close()
        
    # Append to Comparison CSV
    comp_csv = SUMMARY_DIR / "comparison_vs_baselines.csv"
    if comp_csv.exists():
        comp_df = pd.read_csv(comp_csv)
        # Remove any existing v3 rows
        comp_df = comp_df[comp_df['strategy'] != 'frugal_reason_v3']
        
        new_rows = []
        for task in tasks:
            task_df = df[df['task'] == task]
            acc = acc_primary[task]
            new_rows.append({
                'task': task,
                'strategy': 'frugal_reason_v3',
                'accuracy': acc,
                'avg_latency_seconds': task_df['latency'].mean(),
                'avg_total_tokens': task_df['tokens'].mean(),
                'avg_model_calls': task_df['calls'].mean()
            })
        
        comp_df = pd.concat([comp_df, pd.DataFrame(new_rows)], ignore_index=True)
        comp_df.to_csv(comp_csv, index=False)
        
    # Generate FrugalReason_v3_Detailed_Report.pdf
    try:
        from fpdf import FPDF
        class PDFReport(FPDF):
            def header(self):
                self.set_font("Helvetica", "B", 15)
                self.cell(0, 10, "FrugalReason v3 (PAV) Detailed Report", align="C", new_x="LMARGIN", new_y="NEXT")
                self.ln(5)
            def footer(self):
                self.set_y(-15)
                self.set_font("Helvetica", "I", 8)
                self.cell(0, 10, f"Page {self.page_no()}", align="C")
                
        pdf = PDFReport()
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 10, "1. Analysis Plots", new_x="LMARGIN", new_y="NEXT")
        if (SUMMARY_DIR / "alpha_sweep.png").exists():
            pdf.image(str(SUMMARY_DIR / "alpha_sweep.png"), w=190)
        pdf.add_page()
        if (SUMMARY_DIR / "early_exit_histogram.png").exists():
            pdf.image(str(SUMMARY_DIR / "early_exit_histogram.png"), w=190)
            
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 10, "2. Question-Level Metrics", new_x="LMARGIN", new_y="NEXT")
        
        col_widths = [15, 25, 20, 20, 25, 30]
        headers = ["QID", "Task", "Corr", "Clusters", "Route", "Tokens"]
        pdf.set_font("Helvetica", "B", 8)
        for i, h in enumerate(headers):
            if i == len(headers) - 1:
                pdf.cell(col_widths[i], 8, h, border=1, new_x="LMARGIN", new_y="NEXT")
            else:
                pdf.cell(col_widths[i], 8, h, border=1)
                
        pdf.set_font("Helvetica", "", 8)
        for _, row in df.iterrows():
            if pdf.get_y() > 260:
                pdf.add_page()
                pdf.set_font("Helvetica", "B", 8)
                for i, h in enumerate(headers):
                    if i == len(headers) - 1:
                        pdf.cell(col_widths[i], 8, h, border=1, new_x="LMARGIN", new_y="NEXT")
                    else:
                        pdf.cell(col_widths[i], 8, h, border=1)
                pdf.set_font("Helvetica", "", 8)
            
            # evaluate for Correct column
            ans = row['selected_answer']
            eval_res = verifier.score(row['task'], "", str(ans), str(ans), row['gold_answer'])
            corr = "Y" if (eval_res["score"] == 1.0) else "N"
            
            pdf.cell(col_widths[0], 6, str(row.get('question_id', '')).split('_')[-1], border=1)
            pdf.cell(col_widths[1], 6, str(row['task']), border=1)
            pdf.cell(col_widths[2], 6, corr, border=1)
            pdf.cell(col_widths[3], 6, str(len(row['clusters'])), border=1)
            pdf.cell(col_widths[4], 6, str(row['route_used']), border=1)
            pdf.cell(col_widths[5], 6, str(row['tokens']), border=1, new_x="LMARGIN", new_y="NEXT")
            
        pdf.output(str(BASE_DIR / "reports" / "FrugalReason_v3_Detailed_Report.pdf"))
    except ImportError:
        print("fpdf not installed, skipping PDF generation")
        
    print("\nAnalysis complete! Plots saved to results/summary.")

if __name__ == "__main__":
    analyze_pav()
