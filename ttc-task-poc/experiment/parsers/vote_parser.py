"""
vote_parser.py — Extract the chosen option from a ToT voting response.

Expects: "The best choice is N" (N = 1, 2, 3, …).
Fallback: first integer found, or default to 1.
"""

import re
from typing import Dict, Any


def parse(text: str, default: int = 1) -> Dict[str, Any]:
    """Parse a vote response and return the chosen index (1-based)."""
    result: Dict[str, Any] = {
        "raw_text": text,
        "strict_answer": None,
        "lenient_answer": None,
        "final_answer": str(default),
        "parse_method": "failed",
        "parse_success": False,
    }
    if not text or not text.strip():
        return result

    # ── Strict: "The best choice is N" ───────────────────────────────
    m = re.search(r"(?i)the\s+best\s+choice\s+is\s+(\d+)", text)
    if m:
        val = m.group(1)
        result["strict_answer"] = val
        result["final_answer"] = val
        result["parse_method"] = "strict"
        result["parse_success"] = True
        return result

    # ── Lenient: "choice N", "option N", "I choose N" ───────────────
    lenient_patterns = [
        r"(?i)(?:choice|option|pick|select|choose)\s*[:=]?\s*(\d+)",
        r"(?i)(?:I\s+(?:choose|pick|select))\s+(\d+)",
        r"(?i)(?:answer|best)\s*[:=]?\s*(\d+)",
    ]
    for pat in lenient_patterns:
        m = re.search(pat, text)
        if m:
            val = m.group(1)
            result["lenient_answer"] = val
            result["final_answer"] = val
            result["parse_method"] = "lenient"
            result["parse_success"] = True
            return result

    # ── Fallback: first integer ──────────────────────────────────────
    nums = re.findall(r"\b(\d+)\b", text)
    if nums:
        val = nums[0]
        result["final_answer"] = val
        result["parse_method"] = "fallback"
        result["parse_success"] = True
        return result

    # Default to 1
    result["parse_method"] = "fallback"
    return result
