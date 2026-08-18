import os
import json
import pandas as pd
from pathlib import Path
from fpdf import FPDF

class PDFReport(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 14)
        self.cell(0, 10, "FrugalReason Master Evaluation Report (qwen2.5:3b)", align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(3)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

def build_pdf():
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = reports_dir / "Strict_Evaluation_Master_Report.pdf"
    
    in_dir = Path("results/strict_eval")
    data = []
    
    if in_dir.exists():
        for file_path in in_dir.glob("*.jsonl"):
            strategy = file_path.stem
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip(): continue
                    try:
                        d = json.loads(line)
                        tokens = d.get("tokens", 0)
                        if tokens == 0:
                            tokens = d.get("prompt_tokens_total", 0) + d.get("completion_tokens_total", 0)
                        calls = d.get("calls", d.get("model_calls", 0))
                        latency = d.get("latency", d.get("latency_seconds_total", d.get("wall_seconds", 0.0)))
                        ans = d.get("selected_answer", d.get("final_answer", None))
                        parse_success = d.get("parse_success", ans is not None and str(ans).strip() != "")
                        
                        data.append({
                            "Strategy": strategy,
                            "Dataset": d.get("task", ""),
                            "Correct": 1 if d.get("correct", False) else 0,
                            "Tokens": tokens,
                            "Latency": latency,
                            "Calls": calls,
                            "ParseSuccess": 1 if parse_success else 0
                        })
                    except: pass
                    
    df = pd.DataFrame(data)
    
    metrics = df.groupby(["Dataset", "Strategy"]).agg({
        "Correct": ["mean", "count"],
        "Tokens": "mean",
        "Latency": "mean",
        "Calls": "mean",
        "ParseSuccess": "mean"
    }).reset_index()
    
    metrics.columns = ["Dataset", "Strategy", "Accuracy", "Count", "AvgTokens", "AvgLatency", "AvgCalls", "ParseRate"]
    metrics["Accuracy"] = metrics["Accuracy"] * 100
    metrics["ParseRate"] = metrics["ParseRate"] * 100
    
    pdf = PDFReport()
    pdf.add_page()
    
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 10, "1. Executive Summary & Core Results", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 6, "This report details the final 16-cell strict evaluation run comparing FrugalReason v3 against 3 standard baselines across 4 benchmark datasets using the qwen2.5:3b model.")
    pdf.ln(4)
    
    # Render Tables Dataset by Dataset
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "2. Performance Metrics by Dataset", new_x="LMARGIN", new_y="NEXT")
    
    strat_order = {"greedy_cot": 1, "self_consistency_k5": 2, "best_of_n_k5_self_eval": 3, "frugal_reason_v3": 4}
    
    for dataset in sorted(metrics["Dataset"].unique()):
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 6, f"Dataset: {dataset.upper()}", new_x="LMARGIN", new_y="NEXT")
        
        pdf.set_font("Helvetica", "B", 8)
        # Header
        pdf.cell(45, 6, "Strategy", border=1)
        pdf.cell(25, 6, "Accuracy", border=1, align="C")
        pdf.cell(30, 6, "Avg Tokens", border=1, align="C")
        pdf.cell(30, 6, "Avg Latency", border=1, align="C")
        pdf.cell(25, 6, "Avg Calls", border=1, align="C")
        pdf.cell(30, 6, "Parse Rate", border=1, align="C", new_x="LMARGIN", new_y="NEXT")
        
        pdf.set_font("Helvetica", "", 8)
        d_df = metrics[metrics["Dataset"] == dataset].copy()
        d_df["order"] = d_df["Strategy"].map(strat_order)
        d_df = d_df.sort_values("order").drop("order", axis=1)
        
        for _, row in d_df.iterrows():
            pdf.cell(45, 6, str(row['Strategy']), border=1)
            pdf.cell(25, 6, f"{row['Accuracy']:.1f}%", border=1, align="C")
            pdf.cell(30, 6, f"{row['AvgTokens']:.0f}", border=1, align="C")
            pdf.cell(30, 6, f"{row['AvgLatency']:.1f}s", border=1, align="C")
            pdf.cell(25, 6, f"{row['AvgCalls']:.1f}", border=1, align="C")
            pdf.cell(30, 6, f"{row['ParseRate']:.1f}%", border=1, align="C", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(3)
        
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "3. GATE Check (FrugalReason v3 vs. Greedy CoT)", new_x="LMARGIN", new_y="NEXT")
    
    pdf.set_font("Helvetica", "B", 8)
    pdf.cell(35, 6, "Dataset", border=1)
    pdf.cell(35, 6, "Greedy Acc", border=1, align="C")
    pdf.cell(35, 6, "Frugal Acc", border=1, align="C")
    pdf.cell(40, 6, "Accuracy Diff", border=1, align="C")
    pdf.cell(40, 6, "Frugal Parse Rate", border=1, align="C", new_x="LMARGIN", new_y="NEXT")
    
    pdf.set_font("Helvetica", "", 8)
    for dataset in sorted(metrics["Dataset"].unique()):
        d_df = metrics[metrics["Dataset"] == dataset]
        greedy_row = d_df[d_df["Strategy"] == "greedy_cot"]
        frugal_row = d_df[d_df["Strategy"] == "frugal_reason_v3"]
        
        g_acc = greedy_row["Accuracy"].values[0] if not greedy_row.empty else 0
        f_acc = frugal_row["Accuracy"].values[0] if not frugal_row.empty else 0
        diff = f_acc - g_acc
        f_parse = frugal_row["ParseRate"].values[0] if not frugal_row.empty else 0
        
        pdf.cell(35, 6, dataset.upper(), border=1)
        pdf.cell(35, 6, f"{g_acc:.1f}%", border=1, align="C")
        pdf.cell(35, 6, f"{f_acc:.1f}%", border=1, align="C")
        pdf.cell(40, 6, f"{diff:+.1f}%", border=1, align="C")
        pdf.cell(40, 6, f"{f_parse:.1f}%", border=1, align="C", new_x="LMARGIN", new_y="NEXT")
        
    pdf.output(str(pdf_path))
    print(f"PDF generated successfully at {pdf_path}")

if __name__ == "__main__":
    build_pdf()
