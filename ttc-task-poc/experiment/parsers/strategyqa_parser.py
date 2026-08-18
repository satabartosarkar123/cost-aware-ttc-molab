"""
strategyqa_parser.py — Extract yes/no (or A/B/C/D for ARC fallback) answers.

Returns
-------
dict  {"raw_text", "strict_answer", "lenient_answer", "final_answer",
       "parse_method", "parse_success"}
"""

import re
from typing import Dict, Any, Optional


def parse(text: str) -> Dict[str, Any]:
    """Parse a yes/no or multiple-choice answer."""
    result: Dict[str, Any] = {
        "raw_text": text,
        "strict_answer": None,
        "lenient_answer": None,
        "final_answer": None,
        "parse_method": "failed",
        "parse_success": False,
    }
    if not text or not text.strip():
        return result

    lower = text.lower()

    # ── Strict: "the answer is yes/no" ───────────────────────────────
    strict_pat = r"(?i)the\s+answer\s+is\s*[:=]?\s*(yes|no)\b"
    m = re.search(strict_pat, text)
    if m:
        val = m.group(1).lower()
        result["strict_answer"] = val
        result["final_answer"] = val
        result["parse_method"] = "strict"
        result["parse_success"] = True
        return result

    # ── Strict for ARC-style: "the answer is (A)" ───────────────────
    arc_strict = r"(?i)the\s+answer\s+is\s*[:=]?\s*\(?([A-Da-d])\)?"
    m = re.search(arc_strict, text)
    if m:
        val = m.group(1).upper()
        result["strict_answer"] = val
        result["final_answer"] = val
        result["parse_method"] = "strict"
        result["parse_success"] = True
        return result

    # ── Lenient: "Answer: yes/no" ────────────────────────────────────
    lenient_patterns = [
        r"(?i)answer\s*[:=]\s*(yes|no)\b",
        r"(?i)(?:therefore|thus|hence|so|conclusion)\s*[,:=]?\s*(yes|no)\b",
        r"(?i)answer\s*[:=]\s*\(?([A-Da-d])\)?",
    ]
    for pat in lenient_patterns:
        m = re.search(pat, text)
        if m:
            val = m.group(1).strip()
            if val.lower() in ("yes", "no"):
                val = val.lower()
            else:
                val = val.upper()
            result["lenient_answer"] = val
            result["final_answer"] = val
            result["parse_method"] = "lenient"
            result["parse_success"] = True
            return result

    # ── Fallback: last occurrence of yes/no ──────────────────────────
    yn_matches = re.findall(r"\b(yes|no)\b", lower)
    if yn_matches:
        val = yn_matches[-1]
        result["final_answer"] = val
        result["parse_method"] = "fallback"
        result["parse_success"] = True
        return result

    # ── Fallback for ARC: last standalone letter A-D ─────────────────
    letter_matches = re.findall(r"\b([A-Da-d])\b", text)
    if letter_matches:
        val = letter_matches[-1].upper()
        result["final_answer"] = val
        result["parse_method"] = "fallback"
        result["parse_success"] = True
        return result

    return result


def check_correct(parsed_answer: Optional[str], gold_answer: str) -> bool:
    """Return True if parsed matches gold (case-insensitive)."""
    if parsed_answer is None:
        return False
    return str(parsed_answer).strip().lower() == str(gold_answer).strip().lower()
