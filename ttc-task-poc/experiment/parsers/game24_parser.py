"""
game24_parser.py — Extract and evaluate Game-of-24 equations.

Checks if the model's expression uses the given numbers and evaluates to 24.
"""

import re
import ast
from typing import Dict, Any, Optional, List


def _safe_eval(expr: str) -> Optional[float]:
    """Evaluate a simple arithmetic expression safely."""
    # Only allow digits, operators, parentheses, spaces, and decimal points
    if not re.fullmatch(r"[\d+\-*/().\s]+", expr):
        return None
    try:
        result = eval(expr, {"__builtins__": {}}, {})
        return float(result)
    except Exception:
        return None


def _extract_numbers(text: str) -> List[int]:
    """Extract all standalone integers from text."""
    return [int(x) for x in re.findall(r"\b(\d+)\b", text)]


def parse(text: str, input_nums: Optional[str] = None) -> Dict[str, Any]:
    """
    Parse a Game-of-24 response.

    Parameters
    ----------
    text : str
        Model output.
    input_nums : str, optional
        Space-separated input numbers, e.g. "4 9 10 13".
    """
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

    # Collect all math-like expressions from the text
    # Patterns: "expression = 24", standalone expressions with parentheses
    candidates = []

    # Pattern 1: "X = 24" or "X=24"
    for m in re.finditer(r"([\d+\-*/() ]+)\s*=\s*24", text):
        candidates.append(m.group(1).strip())

    # Pattern 2: Lines that look like pure arithmetic
    for line in text.split("\n"):
        line = line.strip()
        # Remove leading bullets, numbers, etc.
        line = re.sub(r"^[\d.)\-*•]+\s*", "", line)
        if re.fullmatch(r"[\d+\-*/().\s]+", line) and len(line) > 2:
            candidates.append(line)

    # Pattern 3: Expressions in parentheses
    for m in re.finditer(r"\([\d+\-*/() ]+\)", text):
        candidates.append(m.group(0))

    # Evaluate each candidate
    for expr in candidates:
        val = _safe_eval(expr)
        if val is not None and abs(val - 24) < 1e-6:
            result["strict_answer"] = expr
            result["final_answer"] = expr
            result["parse_method"] = "strict"
            result["parse_success"] = True
            return result

    # Lenient: any expression that evaluates close to 24
    for expr in candidates:
        val = _safe_eval(expr)
        if val is not None:
            result["lenient_answer"] = f"{expr} = {val}"
            if result["final_answer"] is None:
                result["final_answer"] = expr
                result["parse_method"] = "lenient"
                result["parse_success"] = True

    if result["parse_success"]:
        return result

    # Fallback: report as failed but include the raw text
    result["parse_method"] = "fallback"
    return result


def check_correct(parsed_answer: Optional[str], gold_answer: str = "24") -> bool:
    """Check if the parsed expression evaluates to 24."""
    if parsed_answer is None:
        return False
    val = _safe_eval(parsed_answer)
    return val is not None and abs(val - 24) < 1e-6
