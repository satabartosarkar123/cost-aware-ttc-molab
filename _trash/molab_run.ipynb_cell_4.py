# ── CELL 3: THIRD-PARTY IMPORTS ──
# Run this AFTER Cell 1 (pip install) and Cell 2 (Ollama) have completed.
import warnings
warnings.filterwarnings('ignore', category=SyntaxWarning)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # headless
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import yaml
import sympy
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
from scipy.stats import norm, binomtest
from scipy.optimize import minimize_scalar
from tqdm import tqdm
from tabulate import tabulate
import psutil
try:
    import pynvml
except Exception:
    pass
try:
    import huggingface_hub
    from huggingface_hub import HfApi, hf_hub_download
except Exception:
    pass
try:
    import datasets
except Exception:
    pass
print('All third-party libraries loaded successfully.')
