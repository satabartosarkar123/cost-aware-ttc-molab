import json

NB_PATH = "molab_run.ipynb"
with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] != "code": continue
    src = "".join(cell.get("source", []))
    
    if "CELL 0: STANDARD Python IMPORTS" in src:
        nb["cells"][i]["source"] = [
            "# ── CELL 0: STANDARD Python IMPORTS ──\n",
            "import os, sys, json, time, subprocess, zipfile, shutil, sqlite3, importlib.util, importlib\n",
            "import math, random, re, io, csv, gc, traceback, glob\n",
            "from pathlib import Path\n",
            "from collections import Counter, defaultdict\n",
            "from datetime import datetime, timezone\n",
            "print('All standard Python libraries loaded.')"
        ]
        
    if "HF_REPO = " in src and "ZIP_NAME = " in src:
        # Secretly restore local imports for the download cell to guarantee it runs no matter what
        new_src = (
            "import os, subprocess, zipfile, shutil, sys, importlib.util\n"
            "from pathlib import Path\n"
        ) + src
        nb["cells"][i]["source"] = [line + "\n" for line in new_src.split("\n")]
        nb["cells"][i]["source"][-1] = nb["cells"][i]["source"][-1].rstrip("\n")

    if "CELL 3: THIRD-PARTY IMPORTS" in src:
        nb["cells"][i]["source"] = [
            "# ── CELL 3: THIRD-PARTY IMPORTS ──\n",
            "# Run this AFTER Cell 2 has installed the pip packages.\n",
            "import numpy as np\n",
            "import pandas as pd\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "import requests\n",
            "from scipy.stats import norm, binomtest\n",
            "from scipy.optimize import minimize_scalar\n",
            "try:\n",
            "    import huggingface_hub\n",
            "    from huggingface_hub import HfApi, hf_hub_download\n",
            "except Exception:\n",
            "    pass\n",
            "print('All third-party libraries loaded.')"
        ]

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Enriched imports heavily.")
