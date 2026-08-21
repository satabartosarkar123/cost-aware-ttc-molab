import json
import re

notebook_path = "molab_run.ipynb"
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# The markdown for Day 1
markdown_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "---\n",
        "# Day 1 Master Order — Statistics on Saved Logs\n",
        "**CRITICAL:** Do NOT start Ollama, do NOT run any strategy, do NOT generate any model output.\n",
        "Every number must be computed EXCLUSIVELY from the already-saved Block A logs in `results/block_a_logs/*.jsonl`."
    ]
}

# We pull the actual code source directly from append_cells.py
with open("append_cells.py", "r", encoding="utf-8") as f:
    append_src = f.read()

# Extract code_source string
start_idx = append_src.find("code_source = '''") + len("code_source = '''")
end_idx = append_src.find("'''\n\ncode_cell = {", start_idx)
code_source = append_src[start_idx:end_idx]

# Don't forget the HF Push patch that we know build_days2_10.py applied to Day 1!
hf_push_block = '''

# ── PUSH RESULTS TO HUGGING FACE ──────────────────────────────
print("\\nPushing Day 1 results to Hugging Face Hub...")
try:
    import zipfile as _zf
    import os
    from huggingface_hub import HfApi
    _api = HfApi(token="REDACTED")
    _out_dir = "results"
    _csvs = ["block_a_final_stats.csv", "mcnemar_table.csv", "bootstrap_table.csv"]
    for _csv in _csvs:
        _p = os.path.join(_out_dir, _csv)
        if os.path.exists(_p):
            _api.upload_file(
                path_or_fileobj=_p,
                path_in_repo=f"day1_stats/{_csv}",
                repo_id="Satabarto/Molab_Checkpoints_Cost_AWARE",
                repo_type="dataset",
            )
            print(f"  Uploaded {_csv}")
    print("Day 1 HF push complete.")
except Exception as _e:
    print(f"HF push failed (non-fatal): {_e}")
'''
code_source += hf_push_block

code_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [line + "\n" for line in code_source.split("\n")]
}
if code_cell["source"]:
    code_cell["source"][-1] = code_cell["source"][-1].rstrip("\n")

# Find where to insert: Right before Day 2
insert_idx = None
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] == "markdown" and "Day 2" in "".join(c.get("source", [])) and "Block B" in "".join(c.get("source", [])):
        insert_idx = i
        break

if insert_idx is not None:
    # Let's also remove any previous duplicate Day 1 if it magically existed
    # (We know it doesn't, but just to be safe)
    nb["cells"].insert(insert_idx, markdown_cell)
    nb["cells"].insert(insert_idx + 1, code_cell)
    print("Day 1 restored exactly where it belongs!")
else:
    print("Could not find Day 2 to insert before!")

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

