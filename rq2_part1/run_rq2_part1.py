#!/usr/bin/env python3
"""
RQ2 Part 1 — Cost-Profiling Experiment for Test-Time Compute Strategies.
========================================================================
Measures tokens, latency, model calls, parse behavior, energy (if available),
and accuracy-related signals for each strategy and each selected question.

Part 1 is NOT budget-constrained. No token budgets are enforced.
Budget simulation will be Part 2.
"""

import os
import sys
import json
import csv
import re
import time
import uuid
import traceback
import subprocess
import platform
from datetime import datetime, timezone
from pathlib import Path
from collections import Counter
from typing import Dict, Any, Optional, List, Tuple

# ── Force UTF-8 output ──────────────────────────────────────────────
os.environ.pop("SSLKEYLOGFILE", None)
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ── CUDA Check ──────────────────────────────────────────────────────
try:
    import torch
    if not torch.cuda.is_available():
        print("[ERROR] CUDA is not detected by PyTorch! Exiting to ensure GPU is used.")
        sys.exit(1)
    gpu_name = torch.cuda.get_device_name(0)
    print(f"[CUDA CHECK] SUCCESS: {gpu_name} detected! GPU acceleration is active.\n" + "-"*60)
except ImportError:
    print("[WARNING] PyTorch not found. Skipping strict CUDA check, assuming Ollama handles GPU.\n" + "-"*60)

# ====================================================================
# CONSTANTS
# ====================================================================
RUN_ID = f"rq2p1_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
MAX_RUNTIME_HOURS = 15
OLLAMA_BASE_URL = "http://localhost:11434"
PREFERRED_MODEL = "qwen2.5:3b"
FALLBACK_MODEL = "llama3.2:3b"
MODEL_CALL_TIMEOUT = 180
MAX_RETRIES = 2
TOTAL_QUESTIONS_PER_TASK = 36
TOTAL_TASKS = 3
TOTAL_QUESTIONS = TOTAL_QUESTIONS_PER_TASK * TOTAL_TASKS
STRATEGIES = ["greedy_io", "greedy_cot", "self_consistency_k5", "best_of_n_k5_self_eval", "zero_shot_tot_k3"]

# ── Output directories ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
LOGS_DIR = BASE_DIR / "logs"
PLOTS_DIR = BASE_DIR / "plots"
REPORTS_DIR = BASE_DIR / "reports"
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"

for d in [RESULTS_DIR, LOGS_DIR, PLOTS_DIR, REPORTS_DIR, CHECKPOINTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Output files ────────────────────────────────────────────────────
RAW_CALLS_FILE = RESULTS_DIR / "raw_model_calls.jsonl"
QS_SUMMARY_FILE = RESULTS_DIR / "question_strategy_summary.jsonl"
PROGRESS_FILE = RESULTS_DIR / "progress_log.jsonl"
COST_PROFILE_CSV = RESULTS_DIR / "cost_profile.csv"
TOKEN_PERCENTILES_CSV = RESULTS_DIR / "token_percentiles.csv"
CHECKPOINT_FILE = CHECKPOINTS_DIR / "completed.jsonl"

# ── Global issue tracker ────────────────────────────────────────────
ISSUES: List[Dict[str, str]] = []
START_TIME = time.time()

# ── Global cumulative counters ──────────────────────────────────────
CUMULATIVE = {
    "tokens_run": 0,
    "latency_run": 0.0,
    "model_calls_run": 0,
    "energy_run": 0.0,
    "questions_completed_run": 0,
    "strategy_runs_completed": 0,
    # Per-task cumulatives
    "tokens_task": {},
    "latency_task": {},
    "model_calls_task": {},
    "energy_task": {},
    "questions_completed_task": {},
}

# ====================================================================
# PROMPT TEMPLATES
# ====================================================================
ANSWER_FORMAT_GSM8K = '"the answer is n" where n is a number'
ANSWER_FORMAT_STRATEGYQA = 'either "the answer is yes" or "the answer is no"'
ANSWER_FORMAT_GAME24 = '"the answer is <expression> = 24" using only +, -, *, / and parentheses'

STANDARD_PROMPT = "Answer the following question with {answer_format}: {input}"

COT_PROMPT = """Answer the following question: {input}

Make a strategy then write. Your output should be of the following format:

Strategy: Your strategy about how to answer the question.

Answer: Your answer to the question. It should end with {answer_format}."""

VOTE_PROMPT = """Given an instruction and several choices, decide which choice is most promising.

Analyze each choice in detail, then conclude in the last line:

"The best choice is {{s}}"

where {{s}} is the integer id of the choice.

Instruction: {instruction}

{choices_text}"""

SELF_EVAL_PROMPT = """Review the following solution to this problem:

Problem: {problem}

Solution:
{solution}

Is the final answer correct and well-supported?

Output only a score from 0.0 to 1.0 in this exact format:

Score: X.X"""

TOT_STRATEGY_PROMPT = """Given the following problem, generate a strategy or plan for solving it.

Problem: {input}

Provide a clear, step-by-step strategy for solving this problem. End with {answer_format}."""

TOT_SOLUTION_PROMPT = """Given the following problem and a strategy, generate a complete solution.

Problem: {input}

Strategy: {strategy}

Now solve the problem step by step following the strategy above. End with {answer_format}."""

# ====================================================================
# UTILITY FUNCTIONS
# ====================================================================

def log_issue(severity: str, component: str, message: str, cause: str = "", fix: str = ""):
    """Log an issue to the global issue tracker."""
    ISSUES.append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "severity": severity,
        "component": component,
        "error_message": message,
        "likely_cause": cause,
        "suggested_fix": fix,
    })
    print(f"[ISSUE][{severity}] {component}: {message}")


def append_jsonl(path: Path, record: Dict):
    """Append a single JSON record to a JSONL file."""
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[ERROR] Failed to write to {path}: {e}")


def load_checkpoints() -> set:
    """Load completed checkpoint keys as a set of (task, local_question_id, strategy) tuples."""
    completed = set()
    if CHECKPOINT_FILE.exists():
        try:
            with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    key = (rec["task"], rec["local_question_id"], rec["strategy"])
                    completed.add(key)
        except Exception as e:
            print(f"[WARN] Error loading checkpoints: {e}")
    return completed


def check_runtime():
    """Return True if we've exceeded MAX_RUNTIME_HOURS."""
    elapsed = (time.time() - START_TIME) / 3600
    return elapsed >= MAX_RUNTIME_HOURS


def get_answer_format(task: str) -> str:
    if task == "gsm8k":
        return ANSWER_FORMAT_GSM8K
    elif task == "strategyqa":
        return ANSWER_FORMAT_STRATEGYQA
    elif task == "game24":
        return ANSWER_FORMAT_GAME24
    return ANSWER_FORMAT_GSM8K


# ====================================================================
# HARDWARE MONITOR
# ====================================================================

_PYNVML_AVAILABLE = False
_PSUTIL_AVAILABLE = False
_GPU_HANDLE = None
_HW_INFO = {}

def init_hardware():
    global _PYNVML_AVAILABLE, _PSUTIL_AVAILABLE, _GPU_HANDLE, _HW_INFO
    try:
        import pynvml
        pynvml.nvmlInit()
        _PYNVML_AVAILABLE = True
        _GPU_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(_GPU_HANDLE)
        if isinstance(name, bytes):
            name = name.decode()
        mem = pynvml.nvmlDeviceGetMemoryInfo(_GPU_HANDLE)
        _HW_INFO["gpu_available"] = True
        _HW_INFO["gpu_name"] = name
        _HW_INFO["gpu_memory_mb"] = round(mem.total / (1024**2))
    except Exception:
        _HW_INFO["gpu_available"] = False
        _HW_INFO["gpu_name"] = "N/A"
        _HW_INFO["gpu_memory_mb"] = "N/A"

    try:
        import psutil
        _PSUTIL_AVAILABLE = True
        _HW_INFO["cpu_count"] = psutil.cpu_count(logical=True)
        _HW_INFO["ram_total_mb"] = round(psutil.virtual_memory().total / (1024**2))
    except Exception:
        _HW_INFO["cpu_count"] = "N/A"
        _HW_INFO["ram_total_mb"] = "N/A"


def get_energy_start() -> Optional[int]:
    if _PYNVML_AVAILABLE and _GPU_HANDLE:
        try:
            import pynvml
            return pynvml.nvmlDeviceGetTotalEnergyConsumption(_GPU_HANDLE)
        except Exception:
            pass
    return None


def get_energy_joules(start_energy: Optional[int]) -> Optional[float]:
    if start_energy is not None and _PYNVML_AVAILABLE and _GPU_HANDLE:
        try:
            import pynvml
            end_energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(_GPU_HANDLE)
            return (end_energy - start_energy) / 1000.0
        except Exception:
            pass
    return None


def get_hardware_type() -> str:
    if _PYNVML_AVAILABLE:
        return "gpu"
    if _PSUTIL_AVAILABLE:
        return "cpu_fallback"
    return "unknown"


# ====================================================================
# OLLAMA CLIENT
# ====================================================================

def check_ollama() -> bool:
    """Check if Ollama is reachable."""
    try:
        import requests
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10)
        return r.status_code == 200
    except Exception:
        return False


def start_ollama_serve():
    """Try to start ollama serve in background."""
    try:
        if platform.system() == "Windows":
            subprocess.Popen(["ollama", "serve"], creationflags=subprocess.CREATE_NO_WINDOW)
        else:
            subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(5)
    except Exception as e:
        log_issue("WARNING", "ollama", f"Could not start ollama serve: {e}")


def list_ollama_models() -> List[str]:
    """List locally available models."""
    try:
        import requests
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return []


def pull_model(model_name: str) -> bool:
    """Attempt to pull a model via ollama pull."""
    try:
        print(f"[INFO] Pulling model {model_name}...")
        result = subprocess.run(
            ["ollama", "pull", model_name],
            timeout=600, capture_output=True, text=True
        )
        return result.returncode == 0
    except Exception as e:
        log_issue("WARNING", "ollama", f"Failed to pull {model_name}: {e}")
        return False


def resolve_model() -> Optional[str]:
    """Find or pull a usable model. Returns model name or None."""
    models = list_ollama_models()
    model_names_lower = [m.lower() for m in models]

    # Check preferred
    for m in models:
        if PREFERRED_MODEL in m.lower() or m.lower().startswith(PREFERRED_MODEL.split(":")[0]):
            return m
    # Try exact match
    if PREFERRED_MODEL in model_names_lower:
        return PREFERRED_MODEL
    
    # Try pull preferred
    if pull_model(PREFERRED_MODEL):
        return PREFERRED_MODEL

    # Check fallback
    for m in models:
        if FALLBACK_MODEL in m.lower() or m.lower().startswith(FALLBACK_MODEL.split(":")[0]):
            return m
    if FALLBACK_MODEL in model_names_lower:
        return FALLBACK_MODEL

    # Try pull fallback
    if pull_model(FALLBACK_MODEL):
        return FALLBACK_MODEL

    # Use any available model
    if models:
        log_issue("WARNING", "model", f"Using first available model: {models[0]}")
        return models[0]

    return None


def ollama_generate(
    prompt: str,
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 512,
    top_k: Optional[int] = None,
) -> Dict[str, Any]:
    """Call Ollama /api/generate with retries."""
    import requests

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        }
    }
    if top_k is not None:
        payload["options"]["top_k"] = top_k

    last_error = None
    retry_count = 0

    for attempt in range(1, MAX_RETRIES + 2):  # 1 try + MAX_RETRIES retries
        energy_start = get_energy_start()
        t0 = time.perf_counter()
        try:
            resp = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json=payload,
                timeout=MODEL_CALL_TIMEOUT,
            )
            latency = time.perf_counter() - t0
            energy = get_energy_joules(energy_start)
            resp.raise_for_status()
            data = resp.json()

            return {
                "text": data.get("response", ""),
                "latency_seconds": round(latency, 4),
                "prompt_tokens": data.get("prompt_eval_count", 0) or 0,
                "completion_tokens": data.get("eval_count", 0) or 0,
                "total_tokens": (data.get("prompt_eval_count", 0) or 0) + (data.get("eval_count", 0) or 0),
                "model": model,
                "error": None,
                "retry_count": retry_count,
                "energy_joules": energy,
                "hardware_type": get_hardware_type(),
            }
        except requests.exceptions.Timeout:
            last_error = f"Timeout after {MODEL_CALL_TIMEOUT}s (attempt {attempt})"
        except requests.exceptions.ConnectionError as exc:
            last_error = f"Connection error: {exc} (attempt {attempt})"
        except Exception as exc:
            last_error = f"Unexpected error: {exc} (attempt {attempt})"

        retry_count += 1
        if attempt <= MAX_RETRIES:
            time.sleep(2)

    latency = time.perf_counter() - t0 if 't0' in dir() else 0.0
    return {
        "text": "",
        "latency_seconds": round(latency, 4),
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "model": model,
        "error": last_error,
        "retry_count": retry_count,
        "energy_joules": None,
        "hardware_type": get_hardware_type(),
    }


def log_model_call(
    task: str, local_qid: int, total_local: int,
    original_qnum: int, original_idx0: int,
    strategy: str, stage: str, sample_index: int,
    model_name: str, temperature: float, max_tokens: int,
    top_k: Optional[int], prompt: str, result: Dict[str, Any],
):
    """Log a single model call to raw_model_calls.jsonl."""
    record = {
        "run_id": RUN_ID,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "task": task,
        "local_question_id": local_qid,
        "total_local_questions": total_local,
        "original_question_number": original_qnum,
        "original_index_0based": original_idx0,
        "strategy": strategy,
        "stage": stage,
        "sample_index": sample_index,
        "model_name": model_name,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_k": top_k,
        "prompt": prompt,
        "output_text": result.get("text", ""),
        "prompt_tokens": result.get("prompt_tokens", 0),
        "completion_tokens": result.get("completion_tokens", 0),
        "total_tokens": result.get("total_tokens", 0),
        "latency_seconds": result.get("latency_seconds", 0),
        "retry_count": result.get("retry_count", 0),
        "status": "success" if result.get("error") is None else "error",
        "error_message": result.get("error"),
        "hardware_type": result.get("hardware_type", "unknown"),
        "energy_joules_if_available": result.get("energy_joules"),
    }
    append_jsonl(RAW_CALLS_FILE, record)
    return record


# ====================================================================
# PARSERS
# ====================================================================

def parse_gsm8k(text: str) -> Dict[str, Any]:
    """Multi-tier GSM8K numeric answer parser."""
    result = {
        "raw_text": text,
        "strict_answer": None,
        "lenient_answer": None,
        "final_answer": None,
        "parse_method": "failed",
        "parse_success": False,
    }
    if not text or not text.strip():
        return result

    # Strict: "the answer is <number>"
    m = re.search(r"(?i)the\s+answer\s+is\s*[:=]?\s*(?:[\$€£])?\s*(-?\d[\d,]*(?:\.\d+)?)", text)
    if m:
        val = m.group(1).replace(",", "")
        result["strict_answer"] = val
        result["final_answer"] = val
        result["parse_method"] = "strict"
        result["parse_success"] = True
        return result

    # Lenient patterns
    lenient_patterns = [
        r"(?i)answer\s*[:=]\s*(?:[\$€£])?\s*(-?\d[\d,]*(?:\.\d+)?)",
        r"(?i)####\s*(-?\d[\d,]*(?:\.\d+)?)",
        r"(?i)[\$€£]\s*(-?\d[\d,]*(?:\.\d+)?)",
        r"(?i)(-?\d[\d,]*(?:\.\d+)?)\s*(?:dollars|cents|units|items|people|hours|minutes|days|pounds|kg|miles|meters|gallons|liters)",
        r"(?i)(?:therefore|thus|hence|so|total|result)\s*[,:=]?\s*(?:[\$€£])?\s*(-?\d[\d,]*(?:\.\d+)?)",
    ]
    for pat in lenient_patterns:
        m = re.search(pat, text)
        if m:
            val = m.group(1).replace(",", "")
            result["lenient_answer"] = val
            result["final_answer"] = val
            result["parse_method"] = "lenient"
            result["parse_success"] = True
            return result

    # Fallback: last numeric value
    nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
    if nums:
        val = nums[-1].replace(",", "")
        result["final_answer"] = val
        result["parse_method"] = "fallback"
        result["parse_success"] = True
        return result

    return result


def parse_strategyqa(text: str) -> Dict[str, Any]:
    """Multi-tier StrategyQA yes/no parser."""
    result = {
        "raw_text": text,
        "strict_answer": None,
        "lenient_answer": None,
        "final_answer": None,
        "parse_method": "failed",
        "parse_success": False,
    }
    if not text or not text.strip():
        return result

    # Strict: "the answer is yes/no"
    m = re.search(r"(?i)the\s+answer\s+is\s*[:=]?\s*(yes|no)\b", text)
    if m:
        val = m.group(1).lower()
        result["strict_answer"] = val
        result["final_answer"] = val
        result["parse_method"] = "strict"
        result["parse_success"] = True
        return result

    # Lenient
    lenient_patterns = [
        r"(?i)answer\s*[:=]\s*(yes|no)\b",
        r"(?i)final\s+answer\s*[:=]?\s*(yes|no)\b",
        r"(?i)(?:therefore|thus|hence|so|conclusion)\s*[,:=]?\s*(yes|no)\b",
    ]
    for pat in lenient_patterns:
        m = re.search(pat, text)
        if m:
            val = m.group(1).lower()
            result["lenient_answer"] = val
            result["final_answer"] = val
            result["parse_method"] = "lenient"
            result["parse_success"] = True
            return result

    # Fallback: last occurrence of yes/no
    yn = re.findall(r"\b(yes|no)\b", text.lower())
    if yn:
        val = yn[-1]
        result["final_answer"] = val
        result["parse_method"] = "fallback"
        result["parse_success"] = True
        return result

    return result


def _safe_eval_expr(expr: str) -> Optional[float]:
    """Safely evaluate a simple arithmetic expression."""
    if not re.fullmatch(r"[\d+\-*/().\s]+", expr):
        return None
    try:
        result = eval(expr, {"__builtins__": {}}, {})
        return float(result)
    except Exception:
        return None


def parse_game24(text: str, input_nums: Optional[str] = None) -> Dict[str, Any]:
    """Multi-tier Game of 24 parser."""
    result = {
        "raw_text": text,
        "strict_answer": None,
        "lenient_answer": None,
        "final_answer": None,
        "parse_method": "failed",
        "parse_success": False,
    }
    if not text or not text.strip():
        return result

    expected_nums = sorted([int(x) for x in input_nums.split()]) if input_nums else None

    candidates = []
    # Pattern: "X = 24"
    for m in re.finditer(r"([\d+\-*/() ]+)\s*=\s*24", text):
        candidates.append(m.group(1).strip())
    # Lines that look like arithmetic
    for line in text.split("\n"):
        line = line.strip()
        line = re.sub(r"^[\d.)\-*•]+\s*", "", line)
        if re.fullmatch(r"[\d+\-*/().\s]+", line) and len(line) > 2:
            candidates.append(line)
    # Expressions in parentheses
    for m in re.finditer(r"\([\d+\-*/() ]+\)", text):
        candidates.append(m.group(0))

    for expr in candidates:
        val = _safe_eval_expr(expr)
        if val is not None and abs(val - 24) < 1e-6:
            # Check if all input numbers are used exactly once
            nums_in_expr = sorted([int(x) for x in re.findall(r"\b(\d+)\b", expr)])
            if expected_nums is None or nums_in_expr == expected_nums:
                result["strict_answer"] = expr
                result["final_answer"] = expr
                result["parse_method"] = "strict"
                result["parse_success"] = True
                return result

    # Lenient: expression evaluates to 24 but maybe wrong numbers
    for expr in candidates:
        val = _safe_eval_expr(expr)
        if val is not None and abs(val - 24) < 1e-6:
            result["lenient_answer"] = expr
            result["final_answer"] = expr
            result["parse_method"] = "lenient"
            result["parse_success"] = True
            return result

    # Fallback: any expression
    for expr in candidates:
        val = _safe_eval_expr(expr)
        if val is not None:
            result["final_answer"] = expr
            result["parse_method"] = "fallback"
            result["parse_success"] = True
            return result

    return result


def parse_vote(text: str, k: int) -> int:
    """Parse vote response. Returns 1-based choice index."""
    # Strict: "The best choice is N"
    m = re.search(r"(?i)the\s+best\s+choice\s+is\s+(\d+)", text)
    if m:
        val = int(m.group(1))
        if 1 <= val <= k:
            return val
    # Lenient: "best choice N", "choice N"
    m = re.search(r"(?i)(?:best\s+)?choice\s+(\d+)", text)
    if m:
        val = int(m.group(1))
        if 1 <= val <= k:
            return val
    # Lenient: just find a number
    nums = re.findall(r"\b(\d+)\b", text)
    for n in reversed(nums):
        val = int(n)
        if 1 <= val <= k:
            return val
    log_issue("WARNING", "vote_parser", f"Could not parse vote from: {text[:200]}")
    return 1


def parse_self_eval_score(text: str) -> float:
    """Parse self-evaluation score. Returns float between 0.0 and 1.0."""
    m = re.search(r"(?i)score\s*[:=]?\s*([\d.]+)", text)
    if m:
        try:
            val = float(m.group(1))
            return max(0.0, min(1.0, val))
        except ValueError:
            pass
    log_issue("WARNING", "self_eval_parser", f"Could not parse score from: {text[:200]}")
    return 0.0


def parse_answer(task: str, text: str, input_nums: Optional[str] = None) -> Dict[str, Any]:
    """Dispatch to the correct parser based on task."""
    try:
        if task == "gsm8k":
            return parse_gsm8k(text)
        elif task == "strategyqa":
            return parse_strategyqa(text)
        elif task == "game24":
            return parse_game24(text, input_nums)
        else:
            return parse_gsm8k(text)  # default
    except Exception as e:
        log_issue("WARNING", "parser", f"Parser crashed: {e}")
        return {
            "raw_text": text,
            "strict_answer": None,
            "lenient_answer": None,
            "final_answer": None,
            "parse_method": "failed",
            "parse_success": False,
        }


def check_correct(task: str, parsed_answer: Optional[str], gold_answer: str) -> bool:
    """Check if parsed answer matches gold."""
    if parsed_answer is None:
        return False
    try:
        if task == "gsm8k":
            p = re.sub(r"[,\$€£\s]", "", str(parsed_answer)).strip()
            g = re.sub(r"[,\$€£\s]", "", str(gold_answer)).strip()
            g = re.sub(r"\.0+$", "", g)
            p = re.sub(r"\.0+$", "", p)
            try:
                return float(p) == float(g)
            except ValueError:
                return p == g
        elif task == "strategyqa":
            return str(parsed_answer).strip().lower() == str(gold_answer).strip().lower()
        elif task == "game24":
            val = _safe_eval_expr(str(parsed_answer))
            return val is not None and abs(val - 24) < 1e-6
        else:
            return str(parsed_answer).strip().lower() == str(gold_answer).strip().lower()
    except Exception:
        return False


# ====================================================================
# STRATEGIES
# ====================================================================

def run_greedy_io(
    task: str, question: str, model: str,
    local_qid: int, total_local: int,
    original_qnum: int, original_idx0: int,
    input_nums: Optional[str] = None,
) -> Dict[str, Any]:
    """Strategy 1: greedy_io — Direct answer, T=0, 1 call."""
    answer_format = get_answer_format(task)
    prompt = STANDARD_PROMPT.format(answer_format=answer_format, input=question)

    result = ollama_generate(prompt, model, temperature=0.0, max_tokens=512)
    log_model_call(task, local_qid, total_local, original_qnum, original_idx0,
                   "greedy_io", "generation", 0, model, 0.0, 512, None, prompt, result)

    parsed = parse_answer(task, result["text"], input_nums)
    return {
        "final_answer": parsed["final_answer"],
        "parse_result": parsed,
        "model_calls": 1,
        "prompt_tokens_total": result["prompt_tokens"],
        "completion_tokens_total": result["completion_tokens"],
        "total_tokens_total": result["total_tokens"],
        "latency_seconds_total": result["latency_seconds"],
        "energy_joules_total": result.get("energy_joules"),
        "samples_completed": 1,
        "samples_attempted": 1,
        "stages_completed": 1,
        "error": result.get("error"),
    }


def run_greedy_cot(
    task: str, question: str, model: str,
    local_qid: int, total_local: int,
    original_qnum: int, original_idx0: int,
    input_nums: Optional[str] = None,
) -> Dict[str, Any]:
    """Strategy 2: greedy_cot — Chain of Thought, T=0, 1 call."""
    answer_format = get_answer_format(task)
    prompt = COT_PROMPT.format(answer_format=answer_format, input=question)

    result = ollama_generate(prompt, model, temperature=0.0, max_tokens=512)
    log_model_call(task, local_qid, total_local, original_qnum, original_idx0,
                   "greedy_cot", "generation", 0, model, 0.0, 512, None, prompt, result)

    parsed = parse_answer(task, result["text"], input_nums)
    return {
        "final_answer": parsed["final_answer"],
        "parse_result": parsed,
        "model_calls": 1,
        "prompt_tokens_total": result["prompt_tokens"],
        "completion_tokens_total": result["completion_tokens"],
        "total_tokens_total": result["total_tokens"],
        "latency_seconds_total": result["latency_seconds"],
        "energy_joules_total": result.get("energy_joules"),
        "samples_completed": 1,
        "samples_attempted": 1,
        "stages_completed": 1,
        "error": result.get("error"),
    }


def run_self_consistency_k5(
    task: str, question: str, model: str,
    local_qid: int, total_local: int,
    original_qnum: int, original_idx0: int,
    input_nums: Optional[str] = None,
) -> Dict[str, Any]:
    """Strategy 3: self_consistency_k5 — 5 CoT samples, majority vote."""
    answer_format = get_answer_format(task)
    prompt = COT_PROMPT.format(answer_format=answer_format, input=question)

    total_prompt_tok = 0
    total_comp_tok = 0
    total_tok = 0
    total_lat = 0.0
    total_energy = 0.0
    samples_completed = 0
    answers = []
    answer_order = []

    for i in range(5):
        result = ollama_generate(prompt, model, temperature=0.7, max_tokens=512, top_k=40)
        log_model_call(task, local_qid, total_local, original_qnum, original_idx0,
                       "self_consistency_k5", "generation", i, model, 0.7, 512, 40, prompt, result)

        total_prompt_tok += result["prompt_tokens"]
        total_comp_tok += result["completion_tokens"]
        total_tok += result["total_tokens"]
        total_lat += result["latency_seconds"]
        if result.get("energy_joules") is not None:
            total_energy += result["energy_joules"]

        if result.get("error") is None:
            samples_completed += 1
            parsed = parse_answer(task, result["text"], input_nums)
            ans = parsed["final_answer"]
            if ans is not None:
                answers.append(ans)
                if ans not in answer_order:
                    answer_order.append(ans)

    # Majority vote with first-appearance tie-breaking
    if answers:
        counter = Counter(answers)
        max_count = max(counter.values())
        # Among those with max_count, pick the one that appeared first
        final_answer = None
        for ans in answer_order:
            if counter[ans] == max_count:
                final_answer = ans
                break
        if final_answer is None:
            final_answer = answers[0]
    else:
        final_answer = None

    parsed_final = parse_answer(task, final_answer or "", input_nums) if final_answer else {
        "raw_text": "", "strict_answer": None, "lenient_answer": None,
        "final_answer": None, "parse_method": "failed", "parse_success": False,
    }
    if final_answer:
        parsed_final["final_answer"] = final_answer
        parsed_final["parse_success"] = True

    return {
        "final_answer": final_answer,
        "parse_result": parsed_final,
        "model_calls": 5,
        "prompt_tokens_total": total_prompt_tok,
        "completion_tokens_total": total_comp_tok,
        "total_tokens_total": total_tok,
        "latency_seconds_total": round(total_lat, 4),
        "energy_joules_total": total_energy if total_energy > 0 else None,
        "samples_completed": samples_completed,
        "samples_attempted": 5,
        "stages_completed": 1,
        "error": None,
    }


def run_best_of_n_k5_self_eval(
    task: str, question: str, model: str,
    local_qid: int, total_local: int,
    original_qnum: int, original_idx0: int,
    input_nums: Optional[str] = None,
) -> Dict[str, Any]:
    """Strategy 4: best_of_n_k5_self_eval — 5 candidates + self-evaluation scoring."""
    answer_format = get_answer_format(task)
    prompt = COT_PROMPT.format(answer_format=answer_format, input=question)

    total_prompt_tok = 0
    total_comp_tok = 0
    total_tok = 0
    total_lat = 0.0
    total_energy = 0.0
    model_calls = 0
    candidates = []
    scores = []
    samples_completed = 0

    # Generate 5 candidates
    for i in range(5):
        result = ollama_generate(prompt, model, temperature=0.7, max_tokens=512)
        log_model_call(task, local_qid, total_local, original_qnum, original_idx0,
                       "best_of_n_k5_self_eval", "generation", i, model, 0.7, 512, None, prompt, result)
        model_calls += 1
        total_prompt_tok += result["prompt_tokens"]
        total_comp_tok += result["completion_tokens"]
        total_tok += result["total_tokens"]
        total_lat += result["latency_seconds"]
        if result.get("energy_joules") is not None:
            total_energy += result["energy_joules"]

        if result.get("error") is None:
            samples_completed += 1
            candidates.append(result["text"])
        else:
            candidates.append("")

    # Self-evaluate each candidate
    for i, candidate in enumerate(candidates):
        if not candidate.strip():
            scores.append(0.0)
            continue

        eval_prompt = SELF_EVAL_PROMPT.format(problem=question, solution=candidate)
        eval_result = ollama_generate(eval_prompt, model, temperature=0.0, max_tokens=128)
        log_model_call(task, local_qid, total_local, original_qnum, original_idx0,
                       "best_of_n_k5_self_eval", "self_eval", i, model, 0.0, 128, None, eval_prompt, eval_result)
        model_calls += 1
        total_prompt_tok += eval_result["prompt_tokens"]
        total_comp_tok += eval_result["completion_tokens"]
        total_tok += eval_result["total_tokens"]
        total_lat += eval_result["latency_seconds"]
        if eval_result.get("energy_joules") is not None:
            total_energy += eval_result["energy_joules"]

        score = parse_self_eval_score(eval_result["text"])
        scores.append(score)

    # Select best candidate
    if scores:
        best_idx = max(range(len(scores)), key=lambda x: scores[x])
        best_text = candidates[best_idx] if best_idx < len(candidates) else ""
    else:
        best_idx = 0
        best_text = candidates[0] if candidates else ""

    parsed = parse_answer(task, best_text, input_nums)

    return {
        "final_answer": parsed["final_answer"],
        "parse_result": parsed,
        "model_calls": model_calls,
        "prompt_tokens_total": total_prompt_tok,
        "completion_tokens_total": total_comp_tok,
        "total_tokens_total": total_tok,
        "latency_seconds_total": round(total_lat, 4),
        "energy_joules_total": total_energy if total_energy > 0 else None,
        "samples_completed": samples_completed,
        "samples_attempted": 5,
        "stages_completed": 2,
        "error": None,
    }


def run_zero_shot_tot_k3(
    task: str, question: str, model: str,
    local_qid: int, total_local: int,
    original_qnum: int, original_idx0: int,
    input_nums: Optional[str] = None,
) -> Dict[str, Any]:
    """Strategy 5: zero_shot_tot_k3 — Simplified ToT-BFS with k=3."""
    answer_format = get_answer_format(task)

    total_prompt_tok = 0
    total_comp_tok = 0
    total_tok = 0
    total_lat = 0.0
    total_energy = 0.0
    model_calls = 0
    stages_completed = 0

    # Step 1: Generate k=3 candidate strategies
    strategy_prompt = TOT_STRATEGY_PROMPT.format(input=question, answer_format=answer_format)
    strategy_texts = []
    for i in range(3):
        result = ollama_generate(strategy_prompt, model, temperature=0.7, max_tokens=512)
        log_model_call(task, local_qid, total_local, original_qnum, original_idx0,
                       "zero_shot_tot_k3", "strategy_generation", i, model, 0.7, 512, None, strategy_prompt, result)
        model_calls += 1
        total_prompt_tok += result["prompt_tokens"]
        total_comp_tok += result["completion_tokens"]
        total_tok += result["total_tokens"]
        total_lat += result["latency_seconds"]
        if result.get("energy_joules") is not None:
            total_energy += result["energy_joules"]
        strategy_texts.append(result["text"])
    stages_completed += 1

    # Step 2: Vote for best strategy
    choices_text = "\n".join([f"Choice {i+1}:\n{s}\n" for i, s in enumerate(strategy_texts)])
    vote_prompt_text = VOTE_PROMPT.format(instruction=question, choices_text=choices_text)
    vote_result = ollama_generate(vote_prompt_text, model, temperature=0.0, max_tokens=256)
    log_model_call(task, local_qid, total_local, original_qnum, original_idx0,
                   "zero_shot_tot_k3", "vote", 0, model, 0.0, 256, None, vote_prompt_text, vote_result)
    model_calls += 1
    total_prompt_tok += vote_result["prompt_tokens"]
    total_comp_tok += vote_result["completion_tokens"]
    total_tok += vote_result["total_tokens"]
    total_lat += vote_result["latency_seconds"]
    if vote_result.get("energy_joules") is not None:
        total_energy += vote_result["energy_joules"]

    best_strategy_idx = parse_vote(vote_result["text"], 3) - 1  # 0-based
    best_strategy = strategy_texts[best_strategy_idx] if best_strategy_idx < len(strategy_texts) else strategy_texts[0]
    stages_completed += 1

    # Step 3: Generate k=3 refined solutions conditioned on best strategy
    solution_prompt = TOT_SOLUTION_PROMPT.format(input=question, strategy=best_strategy, answer_format=answer_format)
    solution_texts = []
    for i in range(3):
        result = ollama_generate(solution_prompt, model, temperature=0.7, max_tokens=512)
        log_model_call(task, local_qid, total_local, original_qnum, original_idx0,
                       "zero_shot_tot_k3", "solution_generation", i, model, 0.7, 512, None, solution_prompt, result)
        model_calls += 1
        total_prompt_tok += result["prompt_tokens"]
        total_comp_tok += result["completion_tokens"]
        total_tok += result["total_tokens"]
        total_lat += result["latency_seconds"]
        if result.get("energy_joules") is not None:
            total_energy += result["energy_joules"]
        solution_texts.append(result["text"])
    stages_completed += 1

    # Step 4: Vote for best solution
    choices_text2 = "\n".join([f"Choice {i+1}:\n{s}\n" for i, s in enumerate(solution_texts)])
    vote_prompt_text2 = VOTE_PROMPT.format(instruction=question, choices_text=choices_text2)
    vote_result2 = ollama_generate(vote_prompt_text2, model, temperature=0.0, max_tokens=256)
    log_model_call(task, local_qid, total_local, original_qnum, original_idx0,
                   "zero_shot_tot_k3", "vote", 1, model, 0.0, 256, None, vote_prompt_text2, vote_result2)
    model_calls += 1
    total_prompt_tok += vote_result2["prompt_tokens"]
    total_comp_tok += vote_result2["completion_tokens"]
    total_tok += vote_result2["total_tokens"]
    total_lat += vote_result2["latency_seconds"]
    if vote_result2.get("energy_joules") is not None:
        total_energy += vote_result2["energy_joules"]

    best_solution_idx = parse_vote(vote_result2["text"], 3) - 1
    best_solution = solution_texts[best_solution_idx] if best_solution_idx < len(solution_texts) else solution_texts[0]
    stages_completed += 1

    # Step 5: Parse final answer
    parsed = parse_answer(task, best_solution, input_nums)
    stages_completed += 1

    return {
        "final_answer": parsed["final_answer"],
        "parse_result": parsed,
        "model_calls": model_calls,
        "prompt_tokens_total": total_prompt_tok,
        "completion_tokens_total": total_comp_tok,
        "total_tokens_total": total_tok,
        "latency_seconds_total": round(total_lat, 4),
        "energy_joules_total": total_energy if total_energy > 0 else None,
        "samples_completed": 3,
        "samples_attempted": 3,
        "stages_completed": stages_completed,
        "error": None,
    }


STRATEGY_DISPATCH = {
    "greedy_io": run_greedy_io,
    "greedy_cot": run_greedy_cot,
    "self_consistency_k5": run_self_consistency_k5,
    "best_of_n_k5_self_eval": run_best_of_n_k5_self_eval,
    "zero_shot_tot_k3": run_zero_shot_tot_k3,
}


# ====================================================================
# DATASET LOADING
# ====================================================================

def load_gsm8k_selected() -> List[Dict[str, Any]]:
    """Load GSM8K test split, select questions 642-677 (0-based indices 641-676)."""
    try:
        from datasets import load_dataset
        ds = load_dataset("openai/gsm8k", "main", split="test", trust_remote_code=True)
        selected = ds.select(range(641, 677))
        results = []
        for i, row in enumerate(selected):
            answer_text = row.get("answer", "")
            gold = answer_text.split("####")[-1].strip() if "####" in answer_text else answer_text.strip()
            results.append({
                "local_question_id": i + 1,
                "original_question_number": 642 + i,
                "original_index_0based": 641 + i,
                "question": row["question"],
                "gold_answer": gold,
                "task": "gsm8k",
                "input_nums": None,
            })
        print(f"[INFO] Loaded {len(results)} GSM8K questions (original Q642-Q677)")
        return results
    except Exception as e:
        log_issue("CRITICAL", "gsm8k_loader", f"Failed to load GSM8K: {e}",
                  "HuggingFace datasets issue", "Check internet and datasets library")
        return []


def load_strategyqa_selected() -> List[Dict[str, Any]]:
    """Load StrategyQA dataset, select middle 36 questions."""
    from datasets import load_dataset

    slugs_splits = [
        ("wics/strategy_qa", "test"),
        ("tasksource/strategy_qa", "test"),
        ("ChilleD/StrategyQA", "test"),
        ("wics/strategy_qa", "train"),
        ("tasksource/strategy_qa", "train"),
        ("ChilleD/StrategyQA", "train"),
    ]

    for slug, split in slugs_splits:
        try:
            print(f"[INFO] Trying StrategyQA: {slug} ({split})...")
            ds = load_dataset(slug, split=split, trust_remote_code=True)
            total = len(ds)
            start = (total - 36) // 2
            selected = ds.select(range(start, start + 36))

            results = []
            for i, row in enumerate(selected):
                q = row.get("question", row.get("input", ""))
                a = row.get("answer", row.get("label", ""))
                if isinstance(a, bool):
                    a = "yes" if a else "no"
                a = str(a).strip().lower()
                # Map true/false to yes/no
                if a == "true":
                    a = "yes"
                elif a == "false":
                    a = "no"
                results.append({
                    "local_question_id": i + 1,
                    "original_question_number": start + i + 1,
                    "original_index_0based": start + i,
                    "question": q,
                    "gold_answer": a,
                    "task": "strategyqa",
                    "input_nums": None,
                })
            print(f"[INFO] Loaded {len(results)} StrategyQA questions from {slug} ({split}), "
                  f"total={total}, original Q{start+1}-Q{start+36}")
            return results
        except Exception as e:
            print(f"[WARN] StrategyQA slug {slug} ({split}) failed: {e}")
            continue

    log_issue("CRITICAL", "strategyqa_loader", "All StrategyQA slugs failed",
              "Dataset not available", "Check HuggingFace access")
    return []


def load_game24_selected() -> List[Dict[str, Any]]:
    """Load Game of 24 hard subset (games 901-1000), select indices 32-67."""
    # Try to download from the official Tree of Thoughts repo
    game24_data = None

    # Method 1: Try to read from local file if it exists
    local_paths = [
        BASE_DIR / "data" / "24.csv",
        BASE_DIR.parent / "ttc-task-poc" / "tree-of-thought-llm" / "data" / "24" / "24.csv",
        Path("24.csv"),
    ]
    for p in local_paths:
        if p.exists():
            try:
                import csv as csvmod
                with open(p, "r", encoding="utf-8") as f:
                    reader = csvmod.DictReader(f)
                    game24_data = list(reader)
                print(f"[INFO] Loaded Game of 24 data from {p} ({len(game24_data)} entries)")
                break
            except Exception as e:
                print(f"[WARN] Failed to read {p}: {e}")

    # Method 2: Try to download from GitHub
    if game24_data is None:
        try:
            import requests
            url = "https://raw.githubusercontent.com/princeton-nlp/tree-of-thought-llm/master/src/tot/data/24/24.csv"
            print(f"[INFO] Downloading Game of 24 data from GitHub...")
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            # Save locally
            data_dir = BASE_DIR / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            data_file = data_dir / "24.csv"
            with open(data_file, "w", encoding="utf-8") as f:
                f.write(r.text)
            import csv as csvmod
            import io
            reader = csvmod.DictReader(io.StringIO(r.text))
            game24_data = list(reader)
            print(f"[INFO] Downloaded Game of 24 data ({len(game24_data)} entries)")
        except Exception as e:
            print(f"[WARN] Failed to download Game of 24 data: {e}")

    if game24_data is None or len(game24_data) == 0:
        log_issue("CRITICAL", "game24_loader",
                  "Could not load Game of 24 data from any source",
                  "Data file not found and download failed",
                  "Manually place 24.csv in the data/ folder")
        return []

    # The full file has 1362 games. Hard subset is games 901-1000 (indices 900-999).
    total_games = len(game24_data)
    if total_games >= 1000:
        hard_subset = game24_data[900:1000]  # indices 900-999 = games 901-1000
    elif total_games >= 100:
        # Use last 100
        hard_subset = game24_data[-100:]
    else:
        hard_subset = game24_data

    print(f"[INFO] Game of 24 hard subset: {len(hard_subset)} games")

    # Select indices 32-67 from the hard subset
    if len(hard_subset) >= 68:
        selected = hard_subset[32:68]
    else:
        selected = hard_subset[:min(36, len(hard_subset))]

    results = []
    for i, row in enumerate(selected):
        # CSV columns might be "Puzzles" or "Rank" etc.
        nums = row.get("Puzzles", row.get("puzzles", row.get("nums", "")))
        if not nums:
            # Try first non-Rank column
            for k, v in row.items():
                if k.lower() != "rank" and v:
                    nums = str(v).strip()
                    break

        game_id = 933 + i  # Games 933-968
        results.append({
            "local_question_id": i + 1,
            "original_question_number": game_id,
            "original_index_0based": game_id - 1,
            "question": f"Use the numbers {nums} and basic arithmetic (+, -, *, /) to make 24.",
            "gold_answer": "24",
            "task": "game24",
            "input_nums": nums,
        })

    print(f"[INFO] Loaded {len(results)} Game of 24 questions (Game IDs 933-968)")
    return results


# ====================================================================
# MAIN EXECUTION
# ====================================================================

def run_strategy_for_question(
    strategy_name: str,
    task: str,
    question_data: Dict[str, Any],
    model: str,
    total_local: int,
) -> Dict[str, Any]:
    """Run a single strategy for a single question. Returns summary dict."""
    local_qid = question_data["local_question_id"]
    original_qnum = question_data["original_question_number"]
    original_idx0 = question_data["original_index_0based"]
    question = question_data["question"]
    gold = question_data["gold_answer"]
    input_nums = question_data.get("input_nums")

    strategy_fn = STRATEGY_DISPATCH.get(strategy_name)
    if strategy_fn is None:
        return {"status": "error", "error": f"Unknown strategy: {strategy_name}"}

    try:
        result = strategy_fn(
            task=task, question=question, model=model,
            local_qid=local_qid, total_local=total_local,
            original_qnum=original_qnum, original_idx0=original_idx0,
            input_nums=input_nums,
        )
    except Exception as e:
        log_issue("ERROR", f"strategy_{strategy_name}",
                  f"Crashed on {task} local_q={local_qid}: {e}")
        traceback.print_exc()
        return {
            "status": "error",
            "final_answer": None,
            "parse_result": {"raw_text": "", "strict_answer": None, "lenient_answer": None,
                             "final_answer": None, "parse_method": "failed", "parse_success": False},
            "model_calls": 0,
            "prompt_tokens_total": 0,
            "completion_tokens_total": 0,
            "total_tokens_total": 0,
            "latency_seconds_total": 0,
            "energy_joules_total": None,
            "samples_completed": 0,
            "samples_attempted": 0,
            "stages_completed": 0,
            "error": str(e),
        }

    # Determine correctness
    final_answer = result.get("final_answer")
    correct = check_correct(task, final_answer, gold)
    parse_result = result.get("parse_result", {})

    summary = {
        "run_id": RUN_ID,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "task": task,
        "local_question_id": local_qid,
        "total_local_questions": total_local,
        "original_question_number": original_qnum,
        "original_index_0based": original_idx0,
        "strategy": strategy_name,
        "status": "success" if result.get("error") is None else "error",
        "final_answer": final_answer,
        "gold_answer": gold,
        "correct": correct,
        "strict_answer": parse_result.get("strict_answer"),
        "lenient_answer": parse_result.get("lenient_answer"),
        "parse_method": parse_result.get("parse_method", "failed"),
        "parse_success": parse_result.get("parse_success", False),
        "number_of_model_calls": result.get("model_calls", 0),
        "prompt_tokens_total": result.get("prompt_tokens_total", 0),
        "completion_tokens_total": result.get("completion_tokens_total", 0),
        "total_tokens_total": result.get("total_tokens_total", 0),
        "latency_seconds_total": result.get("latency_seconds_total", 0),
        "energy_joules_total_if_available": result.get("energy_joules_total"),
        "samples_completed": result.get("samples_completed", 0),
        "samples_attempted": result.get("samples_attempted", 0),
        "stages_completed": result.get("stages_completed", 0),
        "budget_exhausted": False,  # Part 1: never enforced
        "error_message": result.get("error"),
    }

    # Log to question_strategy_summary
    append_jsonl(QS_SUMMARY_FILE, summary)

    # Save checkpoint
    append_jsonl(CHECKPOINT_FILE, {
        "task": task,
        "local_question_id": local_qid,
        "strategy": strategy_name,
        "status": summary["status"],
        "timestamp": summary["timestamp"],
    })

    return summary


def generate_reports(
    all_summaries: List[Dict],
    model_used: str,
    tasks_loaded: Dict[str, bool],
    strategyqa_info: Dict[str, Any],
):
    """Generate all markdown reports."""
    try:
        # ── SETUP_REPORT.md ──────────────────────────────────────
        setup_lines = [
            "# RQ2 Part 1 — Setup Report\n\n",
            f"**Generated**: {datetime.now(timezone.utc).isoformat()}\n\n",
            "## Environment\n\n",
            f"- Python version: {sys.version}\n",
            f"- Platform: {platform.platform()}\n",
            f"- Ollama status: {'Available' if check_ollama() else 'Unavailable'}\n",
            f"- Model used: `{model_used}`\n\n",
            "## Hardware\n\n",
            f"- GPU Available: {_HW_INFO.get('gpu_available', False)}\n",
            f"- GPU Name: {_HW_INFO.get('gpu_name', 'N/A')}\n",
            f"- GPU Memory: {_HW_INFO.get('gpu_memory_mb', 'N/A')} MB\n",
            f"- CPU Cores: {_HW_INFO.get('cpu_count', 'N/A')}\n",
            f"- RAM: {_HW_INFO.get('ram_total_mb', 'N/A')} MB\n",
            f"- Energy Monitoring: {'GPU (pynvml)' if _PYNVML_AVAILABLE else 'CPU fallback' if _PSUTIL_AVAILABLE else 'Unavailable'}\n\n",
            "## Dataset Load Status\n\n",
            f"- GSM8K: {'Loaded' if tasks_loaded.get('gsm8k') else 'FAILED'}\n",
            f"- StrategyQA: {'Loaded' if tasks_loaded.get('strategyqa') else 'FAILED'}\n",
            f"- Game of 24: {'Loaded' if tasks_loaded.get('game24') else 'FAILED'}\n\n",
            "## Selected Slices\n\n",
            "- GSM8K: Original questions 642–677 (0-based indices 641–676), 36 questions\n",
            f"- StrategyQA: Original questions {strategyqa_info.get('start_qnum', '?')}–{strategyqa_info.get('end_qnum', '?')} "
            f"(total dataset size: {strategyqa_info.get('total', '?')}), 36 questions\n",
            "- Game of 24: Game IDs 933–968 (hard subset indices 32–67), 36 questions\n\n",
            "## Local ID Mapping\n\n",
            "- local_question_id starts at 1 for each task.\n",
            "- GSM8K: local 1 = original 642, local 36 = original 677\n",
            f"- StrategyQA: local 1 = original {strategyqa_info.get('start_qnum', '?')}, "
            f"local 36 = original {strategyqa_info.get('end_qnum', '?')}\n",
            "- Game of 24: local 1 = game 933, local 36 = game 968\n\n",
            "## Readiness Verdict\n\n",
            f"**{'READY' if any(tasks_loaded.values()) else 'NOT READY'}**\n",
        ]
        with open(REPORTS_DIR / "SETUP_REPORT.md", "w", encoding="utf-8") as f:
            f.writelines(setup_lines)
        print("[INFO] Wrote SETUP_REPORT.md")

        # ── ASSUMPTIONS.md ───────────────────────────────────────
        assumptions_lines = [
            "# RQ2 Part 1 — Assumptions\n\n",
            f"**Generated**: {datetime.now(timezone.utc).isoformat()}\n\n",
            "## Question Selection Rules\n\n",
            "- GSM8K: `gsm8k_test.select(range(641, 677))` → original questions 642–677\n",
            f"- StrategyQA: `start = ({strategyqa_info.get('total', '?')} - 36) // 2 = {strategyqa_info.get('start_idx', '?')}` → "
            f"original questions {strategyqa_info.get('start_qnum', '?')}–{strategyqa_info.get('end_qnum', '?')}\n",
            "- Game of 24: hard subset (games 901–1000), indices 32–67 → game IDs 933–968\n\n",
            "## Local Question ID Rule\n\n",
            "- All logs, reports, and plots use 1-based local_question_id.\n",
            "- local_question_id resets to 1 for each task.\n\n",
            f"## Model Used\n\n- `{model_used}`\n\n",
            "## Fallbacks\n\n",
            f"- Preferred: `{PREFERRED_MODEL}`\n",
            f"- Fallback: `{FALLBACK_MODEL}`\n",
            f"- Actual: `{model_used}`\n\n",
            "## Budget Enforcement\n\n",
            "- **Part 1**: No budget constraints enforced. All strategies run to completion.\n",
        ]
        with open(REPORTS_DIR / "ASSUMPTIONS.md", "w", encoding="utf-8") as f:
            f.writelines(assumptions_lines)
        print("[INFO] Wrote ASSUMPTIONS.md")

        # ── ISSUES.md ────────────────────────────────────────────
        issues_lines = [
            "# RQ2 Part 1 — Issues Log\n\n",
            f"**Generated**: {datetime.now(timezone.utc).isoformat()}\n\n",
        ]
        if ISSUES:
            issues_lines.append("| Timestamp | Severity | Component | Error | Likely Cause | Suggested Fix |\n")
            issues_lines.append("|-----------|----------|-----------|-------|-------------|---------------|\n")
            for iss in ISSUES:
                issues_lines.append(
                    f"| {iss['timestamp']} | {iss['severity']} | {iss['component']} | "
                    f"{iss['error_message'][:80]} | {iss['likely_cause'][:60]} | {iss['suggested_fix'][:60]} |\n"
                )
        else:
            issues_lines.append("No issues detected.\n")

        with open(REPORTS_DIR / "ISSUES.md", "w", encoding="utf-8") as f:
            f.writelines(issues_lines)
        print("[INFO] Wrote ISSUES.md")

    except Exception as e:
        print(f"[ERROR] Report generation failed: {e}")
        traceback.print_exc()


def generate_final_report(all_summaries: List[Dict], model_used: str, partial: bool = False):
    """Generate FINAL_REPORT.md with full analysis."""
    try:
        import numpy as np

        lines = [
            "# RQ2 Part 1 — Final Report\n\n",
            f"**Generated**: {datetime.now(timezone.utc).isoformat()}\n",
            f"**Run ID**: {RUN_ID}\n",
            f"**Status**: {'PARTIAL_RUN' if partial else 'COMPLETE'}\n\n",
        ]

        # 1. Objective
        lines.append("## 1. Objective\n\n")
        lines.append("Measure tokens, latency, model calls, parse behavior, energy (if available), ")
        lines.append("and accuracy-related signals for each test-time compute strategy across three tasks.\n\n")

        # 2. Question slices
        lines.append("## 2. Selected Question Slices\n\n")
        lines.append("| Task | Count | Original Range | Local IDs |\n")
        lines.append("|------|-------|---------------|----------|\n")
        lines.append("| GSM8K | 36 | Q642–Q677 | 1–36 |\n")
        lines.append("| StrategyQA | 36 | See ASSUMPTIONS.md | 1–36 |\n")
        lines.append("| Game of 24 | 36 | Games 933–968 | 1–36 |\n\n")

        # 3. Local ID mapping
        lines.append("## 3. Local Question ID Mapping\n\n")
        lines.append("- 1-based local IDs reset per task.\n")
        lines.append("- GSM8K local 1 = original 642\n")
        lines.append("- Game of 24 local 1 = game 933\n\n")

        # 4. Strategies
        lines.append("## 4. Strategies Used\n\n")
        lines.append("1. `greedy_io` — Direct IO, T=0.0, 1 call\n")
        lines.append("2. `greedy_cot` — Chain-of-Thought, T=0.0, 1 call\n")
        lines.append("3. `self_consistency_k5` — 5 CoT samples, majority vote, T=0.7\n")
        lines.append("4. `best_of_n_k5_self_eval` — 5 candidates + ORM-style self-eval, T=0.7\n")
        lines.append("5. `zero_shot_tot_k3` — Simplified ToT-BFS, k=3\n\n")

        if not all_summaries:
            lines.append("## 5. Results\n\nNo results collected.\n\n")
            with open(REPORTS_DIR / "FINAL_REPORT.md", "w", encoding="utf-8") as f:
                f.writelines(lines)
            return

        # Build quick aggregations
        tasks_completed = set()
        questions_completed = set()
        for s in all_summaries:
            tasks_completed.add(s["task"])
            questions_completed.add((s["task"], s["local_question_id"]))

        total_strategy_runs = len(all_summaries)
        total_tokens = sum(s.get("total_tokens_total", 0) for s in all_summaries)
        total_latency = sum(s.get("latency_seconds_total", 0) for s in all_summaries)
        total_energy_vals = [s.get("energy_joules_total_if_available") for s in all_summaries
                            if s.get("energy_joules_total_if_available") is not None]
        total_energy = sum(total_energy_vals) if total_energy_vals else None

        parse_successes = sum(1 for s in all_summaries if s.get("parse_success"))
        parse_rate = parse_successes / len(all_summaries) if all_summaries else 0

        correct_count = sum(1 for s in all_summaries if s.get("correct"))
        accuracy = correct_count / len(all_summaries) if all_summaries else 0

        # 5. Totals
        lines.append("## 5. Summary\n\n")
        lines.append(f"- Tasks completed: {len(tasks_completed)}\n")
        lines.append(f"- Questions completed: {len(questions_completed)}\n")
        lines.append(f"- Strategy runs completed: {total_strategy_runs}\n\n")

        # 7. Token usage
        lines.append("## 6. Token Usage Summary\n\n")
        lines.append(f"- Total tokens: {total_tokens:,}\n")
        if all_summaries:
            tok_values = [s.get("total_tokens_total", 0) for s in all_summaries]
            lines.append(f"- Mean per strategy-run: {np.mean(tok_values):,.1f}\n")
            lines.append(f"- Median per strategy-run: {np.median(tok_values):,.1f}\n")
            lines.append(f"- Min: {min(tok_values):,} | Max: {max(tok_values):,}\n\n")

        # 8. Latency
        lines.append("## 7. Latency Summary\n\n")
        lines.append(f"- Total latency: {total_latency:,.1f}s ({total_latency/3600:.2f}h)\n")
        if all_summaries:
            lat_values = [s.get("latency_seconds_total", 0) for s in all_summaries]
            lines.append(f"- Mean per strategy-run: {np.mean(lat_values):.2f}s\n")
            lines.append(f"- Median: {np.median(lat_values):.2f}s\n\n")

        # 9. Parse rate
        lines.append("## 8. Parse Rate Summary\n\n")
        lines.append(f"- Overall parse rate: {parse_rate:.2%}\n\n")

        # Per task-strategy parse rate table
        lines.append("| Task | Strategy | Parse Rate | Accuracy |\n")
        lines.append("|------|----------|-----------|----------|\n")
        # Group by task+strategy
        from collections import defaultdict
        groups = defaultdict(list)
        for s in all_summaries:
            groups[(s["task"], s["strategy"])].append(s)
        for (task, strat), items in sorted(groups.items()):
            pr = sum(1 for x in items if x.get("parse_success")) / len(items) if items else 0
            acc = sum(1 for x in items if x.get("correct")) / len(items) if items else 0
            lines.append(f"| {task} | {strat} | {pr:.2%} | {acc:.2%} |\n")
        lines.append("\n")

        # 10. Accuracy
        lines.append("## 9. Accuracy Summary\n\n")
        lines.append(f"- Overall accuracy: {accuracy:.2%}\n\n")

        # 11. Token percentile table
        lines.append("## 10. Token Percentile Table\n\n")
        lines.append("See `results/token_percentiles.csv` for full details.\n\n")

        # 12. Extrapolation
        lines.append("## 11. Full-Dataset Extrapolation\n\n")
        if all_summaries:
            # Group by question to get per-question total across all strategies
            q_totals = defaultdict(lambda: {"tokens": 0, "latency": 0.0})
            for s in all_summaries:
                key = (s["task"], s["local_question_id"])
                q_totals[key]["tokens"] += s.get("total_tokens_total", 0)
                q_totals[key]["latency"] += s.get("latency_seconds_total", 0)

            gsm8k_questions = [(k, v) for k, v in q_totals.items() if k[0] == "gsm8k"]
            if gsm8k_questions:
                avg_tokens_per_q = np.mean([v["tokens"] for _, v in gsm8k_questions])
                avg_latency_per_q = np.mean([v["latency"] for _, v in gsm8k_questions])
                est_tokens = avg_tokens_per_q * 1319
                est_hours = avg_latency_per_q * 1319 / 3600
                lines.append(f"### GSM8K Full Test Set (1319 questions)\n\n")
                lines.append(f"- Avg tokens per question (all strategies): {avg_tokens_per_q:,.0f}\n")
                lines.append(f"- Estimated total tokens: {est_tokens:,.0f}\n")
                lines.append(f"- Avg latency per question: {avg_latency_per_q:.1f}s\n")
                lines.append(f"- Estimated total latency: {est_hours:.1f} hours\n\n")
            else:
                lines.append("GSM8K data not available for extrapolation.\n\n")

            # Per-task extrapolation
            for task_name in ["strategyqa", "game24"]:
                task_qs = [(k, v) for k, v in q_totals.items() if k[0] == task_name]
                if task_qs:
                    avg_tok = np.mean([v["tokens"] for _, v in task_qs])
                    avg_lat = np.mean([v["latency"] for _, v in task_qs])
                    lines.append(f"### {task_name} (sampled {len(task_qs)} questions)\n\n")
                    lines.append(f"- Avg tokens per question: {avg_tok:,.0f}\n")
                    lines.append(f"- Avg latency per question: {avg_lat:.1f}s\n\n")

        # 13. Issues summary
        lines.append("## 12. Issues Summary\n\n")
        if ISSUES:
            lines.append(f"Total issues logged: {len(ISSUES)}\n\n")
            sev_counts = Counter(i["severity"] for i in ISSUES)
            for sev, cnt in sev_counts.items():
                lines.append(f"- {sev}: {cnt}\n")
            lines.append("\nSee `reports/ISSUES.md` for full details.\n\n")
        else:
            lines.append("No issues detected.\n\n")

        # 14. Clear statement
        lines.append("---\n\n")
        lines.append("**This is Part 1 cost profiling only. No budget constraints were enforced.**\n")

        with open(REPORTS_DIR / "FINAL_REPORT.md", "w", encoding="utf-8") as f:
            f.writelines(lines)
        print("[INFO] Wrote FINAL_REPORT.md")

    except Exception as e:
        print(f"[ERROR] Final report generation failed: {e}")
        traceback.print_exc()


def generate_cost_profile_csv(all_summaries: List[Dict]):
    """Generate cost_profile.csv with percentage columns."""
    try:
        from collections import defaultdict

        # Group summaries by (task, local_qid) to compute per-question totals
        q_groups = defaultdict(list)
        for s in all_summaries:
            q_groups[(s["task"], s["local_question_id"])].append(s)

        # Track cumulative tokens per task and overall
        cum_task = defaultdict(int)
        cum_run = 0

        rows = []
        for (task, lqid), strategies in sorted(q_groups.items()):
            # Total tokens for this question across all strategies
            q_total_tokens = sum(s.get("total_tokens_total", 0) for s in strategies)

            for s in strategies:
                tok = s.get("total_tokens_total", 0)
                cum_task[task] += tok
                cum_run += tok

                tok_pct_q = (tok / q_total_tokens * 100) if q_total_tokens > 0 else 0
                tok_pct_task = (tok / cum_task[task] * 100) if cum_task[task] > 0 else 0
                tok_pct_run = (tok / cum_run * 100) if cum_run > 0 else 0

                rows.append({
                    "task": s["task"],
                    "local_question_id": s["local_question_id"],
                    "original_question_number": s["original_question_number"],
                    "strategy": s["strategy"],
                    "status": s["status"],
                    "correct": s["correct"],
                    "parse_success": s["parse_success"],
                    "number_of_model_calls": s["number_of_model_calls"],
                    "prompt_tokens_total": s["prompt_tokens_total"],
                    "completion_tokens_total": s["completion_tokens_total"],
                    "total_tokens_total": s["total_tokens_total"],
                    "latency_seconds_total": s["latency_seconds_total"],
                    "energy_joules_total_if_available": s.get("energy_joules_total_if_available"),
                    "tokens_percent_within_question": round(tok_pct_q, 2),
                    "tokens_percent_within_task_so_far": round(tok_pct_task, 2),
                    "tokens_percent_within_run_so_far": round(tok_pct_run, 2),
                })

        if rows:
            with open(COST_PROFILE_CSV, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
            print(f"[INFO] Wrote cost_profile.csv ({len(rows)} rows)")
    except Exception as e:
        print(f"[ERROR] cost_profile.csv generation failed: {e}")
        traceback.print_exc()


def generate_token_percentiles_csv(all_summaries: List[Dict]):
    """Generate token_percentiles.csv."""
    try:
        import numpy as np
        from collections import defaultdict

        groups = defaultdict(list)
        for s in all_summaries:
            groups[(s["task"], s["strategy"])].append(s)

        rows = []
        for (task, strategy), items in sorted(groups.items()):
            tok_vals = [s.get("total_tokens_total", 0) for s in items]
            lat_vals = [s.get("latency_seconds_total", 0) for s in items]
            mc_vals = [s.get("number_of_model_calls", 0) for s in items]
            parse_vals = [1 if s.get("parse_success") else 0 for s in items]
            correct_vals = [1 if s.get("correct") else 0 for s in items]
            has_gold = any(s.get("gold_answer") for s in items)

            rows.append({
                "task": task,
                "strategy": strategy,
                "count": len(items),
                "mean_tokens": round(np.mean(tok_vals), 1),
                "min_tokens": min(tok_vals),
                "p10_tokens": round(np.percentile(tok_vals, 10), 1),
                "p25_tokens": round(np.percentile(tok_vals, 25), 1),
                "median_tokens": round(np.median(tok_vals), 1),
                "p75_tokens": round(np.percentile(tok_vals, 75), 1),
                "p90_tokens": round(np.percentile(tok_vals, 90), 1),
                "p95_tokens": round(np.percentile(tok_vals, 95), 1),
                "max_tokens": max(tok_vals),
                "mean_latency_seconds": round(np.mean(lat_vals), 2),
                "mean_model_calls": round(np.mean(mc_vals), 1),
                "parse_rate": round(np.mean(parse_vals), 4),
                "accuracy_if_gold_available": round(np.mean(correct_vals), 4) if has_gold else "N/A",
            })

        if rows:
            with open(TOKEN_PERCENTILES_CSV, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
            print(f"[INFO] Wrote token_percentiles.csv ({len(rows)} rows)")
    except Exception as e:
        print(f"[ERROR] token_percentiles.csv generation failed: {e}")
        traceback.print_exc()


def generate_plots(all_summaries: List[Dict]):
    """Generate all required plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        from collections import defaultdict

        if not all_summaries:
            print("[WARN] No summaries for plots, skipping.")
            return

        # Build a DataFrame-like structure
        groups = defaultdict(list)
        for s in all_summaries:
            groups[(s["task"], s["strategy"])].append(s)

        tasks = sorted(set(s["task"] for s in all_summaries))
        strategies = STRATEGIES

        # ── tokens_by_task_strategy.png ──────────────────────────
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(tasks))
        width = 0.15
        for i, strat in enumerate(strategies):
            means = []
            for t in tasks:
                vals = [s.get("total_tokens_total", 0) for s in groups.get((t, strat), [])]
                means.append(np.mean(vals) if vals else 0)
            ax.bar(x + i * width, means, width, label=strat)
        ax.set_xlabel("Task")
        ax.set_ylabel("Mean Total Tokens")
        ax.set_title("Token Usage by Task × Strategy")
        ax.set_xticks(x + width * 2)
        ax.set_xticklabels(tasks)
        ax.legend(fontsize=8, loc="upper left")
        plt.tight_layout()
        fig.savefig(PLOTS_DIR / "tokens_by_task_strategy.png", dpi=150)
        plt.close(fig)
        print("[INFO] Saved tokens_by_task_strategy.png")

        # ── latency_by_task_strategy.png ─────────────────────────
        fig, ax = plt.subplots(figsize=(12, 6))
        for i, strat in enumerate(strategies):
            means = []
            for t in tasks:
                vals = [s.get("latency_seconds_total", 0) for s in groups.get((t, strat), [])]
                means.append(np.mean(vals) if vals else 0)
            ax.bar(x + i * width, means, width, label=strat)
        ax.set_xlabel("Task")
        ax.set_ylabel("Mean Latency (s)")
        ax.set_title("Latency by Task × Strategy")
        ax.set_xticks(x + width * 2)
        ax.set_xticklabels(tasks)
        ax.legend(fontsize=8, loc="upper left")
        plt.tight_layout()
        fig.savefig(PLOTS_DIR / "latency_by_task_strategy.png", dpi=150)
        plt.close(fig)
        print("[INFO] Saved latency_by_task_strategy.png")

        # ── parse_rate_by_task_strategy.png ───────────────────────
        fig, ax = plt.subplots(figsize=(12, 6))
        for i, strat in enumerate(strategies):
            rates = []
            for t in tasks:
                vals = [1 if s.get("parse_success") else 0 for s in groups.get((t, strat), [])]
                rates.append(np.mean(vals) if vals else 0)
            ax.bar(x + i * width, rates, width, label=strat)
        ax.set_xlabel("Task")
        ax.set_ylabel("Parse Rate")
        ax.set_title("Parse Rate by Task × Strategy")
        ax.set_xticks(x + width * 2)
        ax.set_xticklabels(tasks)
        ax.set_ylim(0, 1.1)
        ax.legend(fontsize=8, loc="upper left")
        plt.tight_layout()
        fig.savefig(PLOTS_DIR / "parse_rate_by_task_strategy.png", dpi=150)
        plt.close(fig)
        print("[INFO] Saved parse_rate_by_task_strategy.png")

        # ── accuracy_by_task_strategy.png ─────────────────────────
        fig, ax = plt.subplots(figsize=(12, 6))
        for i, strat in enumerate(strategies):
            accs = []
            for t in tasks:
                vals = [1 if s.get("correct") else 0 for s in groups.get((t, strat), [])]
                accs.append(np.mean(vals) if vals else 0)
            ax.bar(x + i * width, accs, width, label=strat)
        ax.set_xlabel("Task")
        ax.set_ylabel("Accuracy")
        ax.set_title("Accuracy by Task × Strategy")
        ax.set_xticks(x + width * 2)
        ax.set_xticklabels(tasks)
        ax.set_ylim(0, 1.1)
        ax.legend(fontsize=8, loc="upper left")
        plt.tight_layout()
        fig.savefig(PLOTS_DIR / "accuracy_by_task_strategy.png", dpi=150)
        plt.close(fig)
        print("[INFO] Saved accuracy_by_task_strategy.png")

        # ── cumulative_tokens_progress.png ────────────────────────
        fig, ax = plt.subplots(figsize=(12, 6))
        for t in tasks:
            q_data = defaultdict(int)
            for s in all_summaries:
                if s["task"] == t:
                    q_data[s["local_question_id"]] += s.get("total_tokens_total", 0)
            if q_data:
                sorted_qs = sorted(q_data.keys())
                cum = []
                running = 0
                for q in sorted_qs:
                    running += q_data[q]
                    cum.append(running)
                ax.plot(sorted_qs, cum, marker="o", markersize=3, label=t)
        ax.set_xlabel("Local Question ID")
        ax.set_ylabel("Cumulative Tokens")
        ax.set_title("Cumulative Token Progress by Task")
        ax.legend()
        plt.tight_layout()
        fig.savefig(PLOTS_DIR / "cumulative_tokens_progress.png", dpi=150)
        plt.close(fig)
        print("[INFO] Saved cumulative_tokens_progress.png")

        # ── token_percentiles_by_task_strategy.png ────────────────
        fig, axes = plt.subplots(1, len(tasks), figsize=(6 * len(tasks), 6), squeeze=False)
        for tidx, t in enumerate(tasks):
            ax = axes[0][tidx]
            data_for_bp = []
            labels_for_bp = []
            for strat in strategies:
                vals = [s.get("total_tokens_total", 0) for s in groups.get((t, strat), [])]
                if vals:
                    data_for_bp.append(vals)
                    labels_for_bp.append(strat.replace("_", "\n"))
            if data_for_bp:
                ax.boxplot(data_for_bp, labels=labels_for_bp)
            ax.set_title(f"{t}")
            ax.set_ylabel("Total Tokens")
            ax.tick_params(axis="x", rotation=45, labelsize=7)
        plt.suptitle("Token Distribution by Task × Strategy", fontsize=14)
        plt.tight_layout()
        fig.savefig(PLOTS_DIR / "token_percentiles_by_task_strategy.png", dpi=150)
        plt.close(fig)
        print("[INFO] Saved token_percentiles_by_task_strategy.png")

    except Exception as e:
        print(f"[ERROR] Plot generation failed: {e}")
        traceback.print_exc()


# ====================================================================
# MAIN
# ====================================================================

def main():
    global START_TIME
    START_TIME = time.time()

    print("=" * 70)
    print("RQ2 PART 1 — Cost-Profiling Experiment")
    print(f"Run ID: {RUN_ID}")
    print(f"Start time: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 70)

    # ── Initialize hardware monitoring ───────────────────────────
    init_hardware()

    # ── Check Ollama ─────────────────────────────────────────────
    if not check_ollama():
        print("[WARN] Ollama not reachable. Attempting to start...")
        start_ollama_serve()
        if not check_ollama():
            print("[BLOCKED] Ollama is not available. Cannot proceed.")
            log_issue("CRITICAL", "ollama", "Ollama is not reachable after start attempt",
                      "Ollama not installed or not running", "Start Ollama manually")
            # Write BLOCKED report
            with open(REPORTS_DIR / "FINAL_REPORT.md", "w", encoding="utf-8") as f:
                f.write("# RQ2 Part 1 — BLOCKED\n\n")
                f.write(f"**Generated**: {datetime.now(timezone.utc).isoformat()}\n\n")
                f.write("Ollama is not reachable. Experiment cannot proceed.\n")
            generate_reports([], "N/A", {}, {})
            return

    # ── Resolve model ────────────────────────────────────────────
    model_used = resolve_model()
    if model_used is None:
        print("[BLOCKED] No model available. Cannot proceed.")
        log_issue("CRITICAL", "model", "No model available",
                  "No models installed in Ollama", f"Run: ollama pull {PREFERRED_MODEL}")
        with open(REPORTS_DIR / "FINAL_REPORT.md", "w", encoding="utf-8") as f:
            f.write("# RQ2 Part 1 — BLOCKED\n\n")
            f.write("No model available in Ollama.\n")
        generate_reports([], "N/A", {}, {})
        return

    print(f"[INFO] Using model: {model_used}")

    # ── Load datasets ────────────────────────────────────────────
    tasks_data = {}
    tasks_loaded = {}
    strategyqa_info = {"total": "?", "start_idx": "?", "start_qnum": "?", "end_qnum": "?"}

    # GSM8K
    try:
        gsm8k_data = load_gsm8k_selected()
        if gsm8k_data:
            tasks_data["gsm8k"] = gsm8k_data
            tasks_loaded["gsm8k"] = True
        else:
            tasks_loaded["gsm8k"] = False
    except Exception as e:
        log_issue("CRITICAL", "gsm8k_loader", f"Exception: {e}")
        tasks_loaded["gsm8k"] = False

    # StrategyQA
    try:
        sqa_data = load_strategyqa_selected()
        if sqa_data:
            tasks_data["strategyqa"] = sqa_data
            tasks_loaded["strategyqa"] = True
            # Capture info for reports
            strategyqa_info["start_qnum"] = sqa_data[0]["original_question_number"]
            strategyqa_info["end_qnum"] = sqa_data[-1]["original_question_number"]
            strategyqa_info["start_idx"] = sqa_data[0]["original_index_0based"]
        else:
            tasks_loaded["strategyqa"] = False
    except Exception as e:
        log_issue("CRITICAL", "strategyqa_loader", f"Exception: {e}")
        tasks_loaded["strategyqa"] = False

    # Game of 24
    try:
        g24_data = load_game24_selected()
        if g24_data:
            tasks_data["game24"] = g24_data
            tasks_loaded["game24"] = True
        else:
            tasks_loaded["game24"] = False
    except Exception as e:
        log_issue("CRITICAL", "game24_loader", f"Exception: {e}")
        tasks_loaded["game24"] = False

    if not tasks_data:
        print("[BLOCKED] No datasets loaded. Cannot proceed.")
        generate_reports([], model_used, tasks_loaded, strategyqa_info)
        return

    # ── Write setup report ───────────────────────────────────────
    generate_reports([], model_used, tasks_loaded, strategyqa_info)

    # ── Load checkpoints ─────────────────────────────────────────
    completed = load_checkpoints()
    if completed:
        print(f"[INFO] Resuming from checkpoint: {len(completed)} strategy-runs already done.")

    # ── Run experiment ───────────────────────────────────────────
    all_summaries = []
    # Also reload existing summaries from previous runs for complete analysis
    if QS_SUMMARY_FILE.exists():
        try:
            with open(QS_SUMMARY_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        all_summaries.append(json.loads(line))
        except Exception:
            pass

    # Track per-task cumulatives
    task_cum_tokens = {t: 0 for t in tasks_data}
    task_cum_latency = {t: 0.0 for t in tasks_data}
    task_cum_model_calls = {t: 0 for t in tasks_data}
    task_cum_energy = {t: 0.0 for t in tasks_data}
    task_cum_questions = {t: 0 for t in tasks_data}
    run_cum_tokens = 0
    run_cum_latency = 0.0
    run_cum_model_calls = 0
    run_cum_energy = 0.0
    run_cum_questions = 0

    # Build ordered task list
    task_order = [t for t in ["gsm8k", "strategyqa", "game24"] if t in tasks_data]
    total_questions_in_run = sum(len(tasks_data[t]) for t in task_order)

    partial = False

    for task_name in task_order:
        questions = tasks_data[task_name]
        total_local = len(questions)
        print(f"\n{'='*60}")
        print(f"TASK: {task_name} ({total_local} questions)")
        print(f"{'='*60}")

        for q_data in questions:
            if check_runtime():
                print(f"\n[TIMEOUT] Max runtime of {MAX_RUNTIME_HOURS}h exceeded. Stopping cleanly.")
                partial = True
                break

            local_qid = q_data["local_question_id"]
            original_qnum = q_data["original_question_number"]

            q_tokens = 0
            q_latency = 0.0
            q_model_calls = 0
            q_energy = 0.0
            strategies_done = 0

            for strategy_name in STRATEGIES:
                # Check if already completed
                checkpoint_key = (task_name, local_qid, strategy_name)
                if checkpoint_key in completed:
                    strategies_done += 1
                    continue

                if check_runtime():
                    partial = True
                    break

                try:
                    summary = run_strategy_for_question(
                        strategy_name, task_name, q_data, model_used, total_local
                    )
                    all_summaries.append(summary)
                    completed.add(checkpoint_key)

                    q_tokens += summary.get("total_tokens_total", 0)
                    q_latency += summary.get("latency_seconds_total", 0)
                    q_model_calls += summary.get("number_of_model_calls", 0)
                    if summary.get("energy_joules_total_if_available") is not None:
                        q_energy += summary["energy_joules_total_if_available"]

                    strategies_done += 1
                    CUMULATIVE["strategy_runs_completed"] += 1

                except Exception as e:
                    log_issue("ERROR", f"strategy_{strategy_name}",
                              f"Unhandled error on {task_name} q{local_qid}: {e}")
                    traceback.print_exc()
                    strategies_done += 1

            # Update cumulatives
            task_cum_tokens[task_name] += q_tokens
            task_cum_latency[task_name] += q_latency
            task_cum_model_calls[task_name] += q_model_calls
            task_cum_energy[task_name] += q_energy
            task_cum_questions[task_name] += 1

            run_cum_tokens += q_tokens
            run_cum_latency += q_latency
            run_cum_model_calls += q_model_calls
            run_cum_energy += q_energy
            run_cum_questions += 1

            # Compute progress percentages
            pct_task = task_cum_questions[task_name] / total_local * 100
            pct_run = run_cum_questions / total_questions_in_run * 100

            # Token % of task total so far
            tok_pct_task = (q_tokens / task_cum_tokens[task_name] * 100) if task_cum_tokens[task_name] > 0 else 0

            # Estimated remaining hours
            elapsed_hours = (time.time() - START_TIME) / 3600
            if run_cum_questions > 0:
                remaining_qs = total_questions_in_run - run_cum_questions
                avg_time_per_q = (time.time() - START_TIME) / run_cum_questions
                est_remaining_hours = remaining_qs * avg_time_per_q / 3600
            else:
                est_remaining_hours = 0

            # Log progress
            progress_record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "task": task_name,
                "local_question_id": local_qid,
                "total_local_questions": total_local,
                "percent_questions_completed_for_task": round(pct_task, 1),
                "percent_questions_completed_for_run": round(pct_run, 1),
                "tokens_used_this_question": q_tokens,
                "tokens_used_this_question_percent_of_task_total_so_far": round(tok_pct_task, 2),
                "cumulative_tokens_task": task_cum_tokens[task_name],
                "cumulative_tokens_run": run_cum_tokens,
                "latency_this_question_seconds": round(q_latency, 2),
                "cumulative_latency_task_seconds": round(task_cum_latency[task_name], 2),
                "cumulative_latency_run_seconds": round(run_cum_latency, 2),
                "model_calls_this_question": q_model_calls,
                "cumulative_model_calls_task": task_cum_model_calls[task_name],
                "cumulative_model_calls_run": run_cum_model_calls,
                "energy_joules_this_question_if_available": q_energy if q_energy > 0 else None,
                "cumulative_energy_joules_task_if_available": task_cum_energy[task_name] if task_cum_energy[task_name] > 0 else None,
                "cumulative_energy_joules_run_if_available": run_cum_energy if run_cum_energy > 0 else None,
                "estimated_remaining_hours": round(est_remaining_hours, 1),
                "status": "complete",
            }
            append_jsonl(PROGRESS_FILE, progress_record)

            # Console progress
            print(f"[PROGRESS] task={task_name} | local_q={local_qid}/{total_local} | "
                  f"original_q={original_qnum} | strategies_done={strategies_done} | "
                  f"tokens={q_tokens} | latency={q_latency:.1f}s | "
                  f"run_progress={pct_run:.1f}% | est_remaining={est_remaining_hours:.1f}h")

        if partial:
            break

    # ── Post-processing ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print("POST-PROCESSING")
    print(f"{'='*60}")

    # Update RUN_PROGRESS report
    try:
        with open(REPORTS_DIR / "RUN_PROGRESS.md", "w", encoding="utf-8") as f:
            f.write(f"# RQ2 Part 1 — Run Progress\n\n")
            f.write(f"**Generated**: {datetime.now(timezone.utc).isoformat()}\n\n")
            f.write(f"- Status: {'PARTIAL' if partial else 'COMPLETE'}\n")
            f.write(f"- Questions completed: {run_cum_questions}/{total_questions_in_run}\n")
            f.write(f"- Total tokens: {run_cum_tokens:,}\n")
            f.write(f"- Total latency: {run_cum_latency:.1f}s ({run_cum_latency/3600:.2f}h)\n")
            f.write(f"- Total model calls: {run_cum_model_calls}\n")
    except Exception as e:
        print(f"[ERROR] Failed to write RUN_PROGRESS.md: {e}")

    # Generate CSV outputs
    generate_cost_profile_csv(all_summaries)
    generate_token_percentiles_csv(all_summaries)

    # Generate plots
    try:
        generate_plots(all_summaries)
    except Exception as e:
        print(f"[ERROR] Plot generation failed: {e}")
        traceback.print_exc()

    # Generate final report
    generate_final_report(all_summaries, model_used, partial)

    # Update ISSUES report (re-generate with latest issues)
    generate_reports(all_summaries, model_used, tasks_loaded, strategyqa_info)

    # ── Final console summary ────────────────────────────────────
    total_tokens_final = sum(s.get("total_tokens_total", 0) for s in all_summaries)
    total_latency_final = sum(s.get("latency_seconds_total", 0) for s in all_summaries)
    total_energy_vals = [s.get("energy_joules_total_if_available") for s in all_summaries
                         if s.get("energy_joules_total_if_available") is not None]
    total_energy_final = sum(total_energy_vals) if total_energy_vals else None
    parse_successes = sum(1 for s in all_summaries if s.get("parse_success"))
    parse_rate = parse_successes / len(all_summaries) if all_summaries else 0
    correct_count = sum(1 for s in all_summaries if s.get("correct"))
    accuracy = correct_count / len(all_summaries) if all_summaries else 0

    tasks_done = set(s["task"] for s in all_summaries)
    qs_done = set((s["task"], s["local_question_id"]) for s in all_summaries)

    print(f"\n{'='*70}")
    print("RQ2 PART 1 COMPLETE" if not partial else "RQ2 PART 1 PARTIAL")
    print(f"{'='*70}")
    print(f"Tasks completed: {len(tasks_done)}")
    print(f"Questions completed: {len(qs_done)}")
    print(f"Strategy runs completed: {len(all_summaries)}")
    print(f"Total tokens: {total_tokens_final:,}")
    print(f"Total latency: {total_latency_final:.1f}s ({total_latency_final/3600:.2f}h)")
    print(f"Total energy: {total_energy_final:.2f}J" if total_energy_final else "Total energy: N/A")
    print(f"Parse rate: {parse_rate:.2%}")
    print(f"Accuracy summary: {accuracy:.2%} ({correct_count}/{len(all_summaries)})")
    print(f"Results: rq2_part1/results/cost_profile.csv")
    print(f"Report: rq2_part1/reports/FINAL_REPORT.md")
    print(f"Issues: rq2_part1/reports/ISSUES.md")
    print(f"{'='*70}")

    elapsed_total = (time.time() - START_TIME) / 3600
    print(f"\nTotal wall-clock time: {elapsed_total:.2f} hours")


if __name__ == "__main__":
    main()
