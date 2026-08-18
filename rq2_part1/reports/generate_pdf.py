import os
from pathlib import Path
import pandas as pd
from fpdf import FPDF
import matplotlib.pyplot as plt

class PDFReport(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 15)
        self.cell(0, 10, f"RQ2 Part 1: Detailed Report - {self.strategy_name.upper()}", align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

def generate_pdfs():
    base_dir = Path(__file__).parent.parent
    csv_file = base_dir / "results" / "cost_profile.csv"

    if not csv_file.exists():
        print(f"Cannot find {csv_file}")
        return

    df = pd.read_csv(csv_file)
    
    # Calculate cumulative percentages per task/strategy combo
    df_sorted = df.sort_values(by=["task", "strategy", "local_question_id"])
    group_totals = df_sorted.groupby(["task", "strategy"])[["total_tokens_total", "latency_seconds_total"]].transform('sum')
    df_sorted["cum_tokens"] = df_sorted.groupby(["task", "strategy"])["total_tokens_total"].cumsum()
    df_sorted["cum_latency"] = df_sorted.groupby(["task", "strategy"])["latency_seconds_total"].cumsum()
    df_sorted["cum_token_pct"] = (df_sorted["cum_tokens"] / group_totals["total_tokens_total"]) * 100
    df_sorted["cum_latency_pct"] = (df_sorted["cum_latency"] / group_totals["latency_seconds_total"]) * 100
    
    strategies = df_sorted['strategy'].unique()
    
    for strat in strategies:
        output_pdf = base_dir / "reports" / f"RQ2_Part1_{strat}_Report.pdf"
        
        pdf = PDFReport()
        pdf.strategy_name = strat
        pdf.add_page()
        
        # 1. Strategy Summary (Across all tasks)
        strat_df = df_sorted[df_sorted['strategy'] == strat]
        total_q = len(strat_df)
        total_tokens = strat_df['total_tokens_total'].sum()
        total_latency = strat_df['latency_seconds_total'].sum()
        overall_acc = strat_df['correct'].mean()
        
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 10, "1. Strategy Overview", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        
        pdf.cell(0, 8, f"Total Questions Evaluated: {total_q}", new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 8, f"Overall Accuracy: {overall_acc:.1%}", new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 8, f"Total Tokens Processed: {total_tokens:,}", new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 8, f"Total Execution Latency: {total_latency/3600:.2f} hours", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(5)
        
        # 2. Breakdown By Task
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 10, "2. Breakdown by Task", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        
        agg = strat_df.groupby("task").agg({
            "correct": "mean",
            "total_tokens_total": "mean",
            "latency_seconds_total": "mean",
            "number_of_model_calls": "mean"
        }).reset_index()
        
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_fill_color(200, 220, 255)
        pdf.cell(40, 8, "Task", border=1, fill=True)
        pdf.cell(25, 8, "Accuracy", border=1, fill=True)
        pdf.cell(30, 8, "Avg Tokens", border=1, fill=True)
        pdf.cell(30, 8, "Avg Latency", border=1, fill=True)
        pdf.cell(25, 8, "Avg Calls", border=1, new_x="LMARGIN", new_y="NEXT", fill=True)
        
        pdf.set_font("Helvetica", "", 9)
        for _, row in agg.iterrows():
            pdf.cell(40, 8, str(row['task']).upper(), border=1)
            pdf.cell(25, 8, f"{row['correct']:.1%}", border=1)
            pdf.cell(30, 8, f"{row['total_tokens_total']:.0f}", border=1)
            pdf.cell(30, 8, f"{row['latency_seconds_total']:.1f}s", border=1)
            pdf.cell(25, 8, f"{row['number_of_model_calls']:.1f}", border=1, new_x="LMARGIN", new_y="NEXT")
        
        pdf.ln(10)
        
        # 3. Generate and Embed Plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for task in strat_df['task'].unique():
            task_data = strat_df[strat_df['task'] == task].sort_values(by="local_question_id")
            x_vals = range(1, len(task_data) + 1)
            axes[0].plot(x_vals, task_data['cum_token_pct'], label=f"{task}", marker='o', markersize=3)
            axes[1].plot(x_vals, task_data['cum_latency_pct'], label=f"{task}", marker='o', markersize=3)
            
        axes[0].set_title("Cumulative Tokens %")
        axes[0].set_xlabel("Question Sequence")
        axes[0].set_ylabel("Cumulative %")
        axes[0].legend()
        axes[0].grid(True)
        
        axes[1].set_title("Cumulative Latency %")
        axes[1].set_xlabel("Question Sequence")
        axes[1].set_ylabel("Cumulative %")
        axes[1].legend()
        axes[1].grid(True)
        
        plot_path = base_dir / "reports" / f"cum_plot_{strat}.png"
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

        pdf.add_page()
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 10, "3. Cumulative Cost Curves", new_x="LMARGIN", new_y="NEXT")
        pdf.image(str(plot_path), w=190)
        pdf.ln(5)

        # 4. Detailed Question-Level Metrics
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 10, "4. Detailed Question-Level Metrics", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 9)
        pdf.cell(0, 8, "Below is the token and latency breakdown for every question for this strategy.", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(5)
        
        col_widths = [15, 30, 15, 25, 25, 30, 30]
        headers = ["QID", "Task", "Corr", "Tokens", "Lat.", "Cum Tok%", "Cum Lat%"]
        
        pdf.set_font("Helvetica", "B", 8)
        for i, h in enumerate(headers):
            if i == len(headers) - 1:
                pdf.cell(col_widths[i], 8, h, border=1, new_x="LMARGIN", new_y="NEXT", fill=True)
            else:
                pdf.cell(col_widths[i], 8, h, border=1, fill=True)
        
        pdf.set_font("Helvetica", "", 8)
        
        strat_df = strat_df.sort_values(by=["task", "local_question_id"])
        
        for _, row in strat_df.iterrows():
            if pdf.get_y() > 260:
                pdf.add_page()
                pdf.set_font("Helvetica", "B", 8)
                for i, h in enumerate(headers):
                    if i == len(headers) - 1:
                        pdf.cell(col_widths[i], 8, h, border=1, new_x="LMARGIN", new_y="NEXT", fill=True)
                    else:
                        pdf.cell(col_widths[i], 8, h, border=1, fill=True)
                pdf.set_font("Helvetica", "", 8)

            qid = str(row['local_question_id'])
            task = str(row['task'])
            corr = "Y" if row['correct'] else "N"
            toks = str(int(row['total_tokens_total']))
            lat = f"{row['latency_seconds_total']:.1f}s"
            cum_tok = f"{row['cum_token_pct']:.1f}%"
            cum_lat = f"{row['cum_latency_pct']:.1f}%"

            pdf.cell(col_widths[0], 6, qid, border=1)
            pdf.cell(col_widths[1], 6, task, border=1)
            pdf.cell(col_widths[2], 6, corr, border=1)
            pdf.cell(col_widths[3], 6, toks, border=1)
            pdf.cell(col_widths[4], 6, lat, border=1)
            pdf.cell(col_widths[5], 6, cum_tok, border=1)
            pdf.cell(col_widths[6], 6, cum_lat, border=1, new_x="LMARGIN", new_y="NEXT")

        pdf.output(str(output_pdf))
        print(f"Successfully generated PDF: {output_pdf}")
        
        # Cleanup temporary plot image
        try:
            os.remove(plot_path)
        except Exception as e:
            pass

if __name__ == "__main__":
    generate_pdfs()
