"""
gsm8k_parser.py — Multi-tier answer extraction for math word problems.

Returns
-------
dict  {"raw_text", "strict_answer", "lenient_answer", "final_answer",
       "parse_method", "parse_success"}
"""

import re
from typing import Dict, Any, Optional


def parse(text: str) -> Dict[str, Any]:
    """Parse a GSM8K-style numeric answer from model output."""
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

    # ── Strict: "the answer is <number>" ─────────────────────────────
    strict_pat = r"(?i)the\s+answer\s+is\s*[:=]?\s*(?:[\$€£])?\s*(-?\d[\d,]*(?:\.\d+)?)"
    m = re.search(strict_pat, text)
    if m:
        val = m.group(1).replace(",", "")
        result["strict_answer"] = val
        result["final_answer"] = val
        result["parse_method"] = "strict"
        result["parse_success"] = True
        return result

    # ── Lenient: "Answer: <number>", "$18", "18 dollars", etc. ───────
    lenient_patterns = [
        r"(?i)answer\s*[:=]\s*(?:[\$€£])?\s*(-?\d[\d,]*(?:\.\d+)?)",
        r"(?i)####\s*(-?\d[\d,]*(?:\.\d+)?)",
        r"(?i)(?:[\$€£])\s*(-?\d[\d,]*(?:\.\d+)?)",
        r"(?i)(-?\d[\d,]*(?:\.\d+)?)\s*(?:dollars|cents|units|items|people|hours|minutes|days|pounds|kg|miles|meters|gallons|liters)",
        r"(?i)(?:therefore|thus|hence|so|total|result)\s*[,:=]?\s*(?:[\$€£])?\s*(-?\d[\d,]*(?:\.\d+)?)",
    ]
    for pat in lenient_patterns:
        m = re.search(pat, text)
        if m:
            val = m.group(1).replace(",", "")
            result["lenient_answer"] = val
            if result["final_answer"] is None:
                result["final_answer"] = val
                result["parse_method"] = "lenient"
                result["parse_success"] = True
            return result

    # ── Fallback: last numeric value in the text ─────────────────────
    nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
    if nums:
        val = nums[-1].replace(",", "")
        result["final_answer"] = val
        result["parse_method"] = "fallback"
        result["parse_success"] = True
        return result

    return result


def normalize_gold(gold: str) -> str:
    """Normalize a GSM8K gold answer string for comparison."""
    gold = gold.strip()
    gold = re.sub(r"[,\$€£\s]", "", gold)
    # Remove trailing ".0" or ".00"
    gold = re.sub(r"\.0+$", "", gold)
    return gold


def check_correct(parsed_answer: Optional[str], gold_answer: str) -> bool:
    """Return True if the parsed answer matches the gold."""
    if parsed_answer is None:
        return False
    p = re.sub(r"[,\$€£\s]", "", str(parsed_answer)).strip()
    g = normalize_gold(gold_answer)
    # Remove trailing zeros after decimal for comparison
    try:
        return float(p) == float(g)
    except ValueError:
        return p == g
