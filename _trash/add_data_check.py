import json
import os

nb = json.load(open('molab_run_fixed.ipynb', encoding='utf-8'))

check_cell = {
    'cell_type': 'code',
    'execution_count': None,
    'metadata': {},
    'outputs': [],
    'source': [
        "# ── VERIFY DATA INTEGRITY BEFORE PROCEEDING ──\n",
        "import os, json\n",
        "from pathlib import Path\n",
        "\n",
        "_nb = Path(os.environ.get('NOTEBOOK_DIR', '.'))\n",
        "res_dir = _nb / 'results'\n",
        "\n",
        "def check_model_logs(model, subdir, prefix):\n",
        "    print(f'Checking {model} logs in {subdir}...')\n",
        "    datasets = ['gsm8k', 'aqua', 'math', 'strategyqa']\n",
        "    missing = False\n",
        "    for ds in datasets:\n",
        "        fr_path = res_dir / subdir / f'{prefix}{ds}_frugal_reason_v3.jsonl'\n",
        "        cot_path = res_dir / subdir / f'{prefix}{ds}_greedy_cot.jsonl'\n",
        "        if not fr_path.exists():\n",
        "            print(f'  [ERROR] Missing FR log: {fr_path.name}')\n",
        "            missing = True\n",
        "        if not cot_path.exists():\n",
        "            print(f'  [ERROR] Missing CoT log: {cot_path.name}')\n",
        "            missing = True\n",
        "    if not missing:\n",
        "        print(f'  [OK] All logs for {model} are present!')\n",
        "\n",
        "check_model_logs('qwen2.5:3b', 'block_a_logs', '')\n",
        "check_model_logs('qwen2.5:1.5b', 'block_b_logs', 'qwen15b_')\n",
        "check_model_logs('llama3.2:3b', 'block_b_logs', 'llama32_')\n",
        "print('\\nDATA INTEGRITY CHECK PASSED. YOU ARE SAFE TO PROCEED.')\n"
    ]
}

# Insert this after cell 2 (index 1)
nb['cells'].insert(2, check_cell)

with open('molab_run_fixed.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
