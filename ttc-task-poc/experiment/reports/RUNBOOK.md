# RUNBOOK — TTC-Task POC

## Prerequisites

1. **Python 3.9+** (with pip)
2. **Git**
3. **Ollama** — https://ollama.com/download

## Setup Steps

```bash
# 1. Install Ollama (if not already installed)
# Download from https://ollama.com/download and install

# 2. Start Ollama
ollama serve

# 3. Pull the model (in a new terminal)
ollama pull qwen2.5:3b
# Fallback:
# ollama pull llama3.2:3b

# 4. Navigate to the experiment directory
cd ttc-task-poc/experiment

# 5. Create virtual environment (optional but recommended)
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Unix:
# source .venv/bin/activate

# 6. Install dependencies
pip install -r requirements.txt

# 7. Run the experiment
python run_poc.py
```

## Expected Runtime

- **CPU only (no GPU)**: ~30-60 minutes for all 5 strategies × 3 tasks × 10 questions
- **With GPU (RTX 3050 6GB)**: ~15-30 minutes

## Output

- Raw logs: `results/raw_logs/`
- Parsed CSVs: `results/parsed/`
- Summary: `results/summary/summary.csv`
- Plots: `plots/`
- Reports: `reports/POC_REPORT.md`
