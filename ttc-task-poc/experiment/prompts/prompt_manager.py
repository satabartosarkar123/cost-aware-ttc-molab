"""
prompt_manager.py — Load and format prompt templates.

Templates live in prompts/raw_text/<name>.txt and use {placeholder} syntax.
"""

import os
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

_TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "raw_text")

# Cache loaded templates
_cache: Dict[str, str] = {}


def _load(name: str) -> str:
    """Load a raw text template by name (without .txt extension)."""
    if name in _cache:
        return _cache[name]
    path = os.path.join(_TEMPLATE_DIR, f"{name}.txt")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Prompt template not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        tpl = f.read()
    _cache[name] = tpl
    return tpl


def get_io_prompt(task: str, question: str) -> str:
    """Return a standard IO prompt for the given task."""
    if task == "gsm8k":
        tpl = _load("standard_io_math")
        return tpl.replace("{input}", question)
    elif task == "game24":
        tpl = _load("standard_io_math")
        return tpl.replace("{input}", question)
    else:
        # Check if it looks like a multiple-choice question
        if "(" in question and ")" in question:
            tpl = _load("standard_io_mc")
        else:
            tpl = _load("standard_io_yesno")
        return tpl.replace("{input}", question)


def get_cot_prompt(task: str, question: str) -> str:
    """Return a Chain-of-Thought prompt for the given task."""
    tpl = _load("cot")
    if task == "gsm8k":
        answer_fmt = "a single numeric value"
    elif task == "game24":
        answer_fmt = "an equation that equals 24"
    else:
        answer_fmt = '"yes" or "no"'
    return tpl.replace("{input}", question).replace("{answer_format}", answer_fmt)


def get_vote_prompt(instruction: str, choices: list) -> str:
    """Return a ToT voting prompt."""
    tpl = _load("vote")
    choices_text = "\n".join(
        f"Choice {i+1}:\n{c}" for i, c in enumerate(choices)
    )
    return tpl.replace("{instruction}", instruction).replace("{choices}", choices_text)


def get_tot_propose_prompt(nums: str, k: int = 3) -> str:
    """Return a ToT propose prompt for Game of 24."""
    tpl = _load("tot_propose")
    return tpl.replace("{nums}", nums).replace("{k}", str(k))


def get_tot_value_prompt(nums: str, steps: str = "none") -> str:
    """Return a ToT value evaluation prompt."""
    tpl = _load("tot_value")
    return tpl.replace("{nums}", nums).replace("{steps}", steps)


def get_orm_judge_prompt(question: str, reasoning: str, gold: str) -> str:
    """Return an ORM judge prompt."""
    tpl = _load("orm_judge")
    return (
        tpl.replace("{question}", question)
        .replace("{reasoning}", reasoning)
        .replace("{gold}", gold)
    )
