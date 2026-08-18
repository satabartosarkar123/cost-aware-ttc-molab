"""
verifier.py — ORM-proxy outcome verifier for Best-of-N strategy.

• GSM8K  : deterministic exact-match (parsed answer vs gold).
• StrategyQA / commonsense : LLM-as-a-judge ("1" correct, "0" wrong).
• Game of 24 : evaluates the equation to check if it equals 24.
"""

import re
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class OutcomeVerifier:
    """Score a candidate answer against the ground truth."""

    ORM_JUDGE_TEMPLATE = (
        "You are a strict answer verifier.\n\n"
        "Question: {question}\n"
        "Proposed answer and reasoning:\n{reasoning}\n\n"
        "Ground-truth answer: {gold}\n\n"
        "Does the proposed reasoning correctly arrive at the ground-truth answer? "
        'Respond with only "1" if correct or "0" if incorrect.'
    )

    def __init__(self, ollama_client=None):
        self.client = ollama_client

    # ── public API ───────────────────────────────────────────────────
    def score(
        self,
        task: str,
        question: str,
        generated_text: str,
        parsed_answer: Optional[str],
        gold_answer: str,
    ) -> Dict[str, Any]:
        """
        Return {"score": float, "method": str, "detail": str}.
        """
        if task in ["gsm8k", "gsm_hard", "svamp", "aqua"]:
            return self._score_exact_match(parsed_answer, gold_answer)
        elif task == "math":
            return self._score_math(parsed_answer, gold_answer)
        elif task == "game24":
            return self._score_game24(generated_text, question)
        else:
            # commonsense / strategyqa → LLM-as-judge or exact match
            em = self._score_exact_match(parsed_answer, gold_answer)
            if em["score"] == 1.0:
                return em
            # Try LLM-as-judge
            if self.client is not None:
                return self._score_llm_judge(question, generated_text, gold_answer)
            return em

    # ── exact match ──────────────────────────────────────────────────
    @staticmethod
    def _score_exact_match(parsed: Optional[str], gold: str) -> Dict[str, Any]:
        if parsed is None:
            return {"score": 0.0, "method": "exact_match", "detail": "parse_failed"}
        # Normalise
        p = re.sub(r"[,\$\s]", "", str(parsed)).lower().strip()
        g = re.sub(r"[,\$\s]", "", str(gold)).lower().strip()
        match = p == g
        return {
            "score": 1.0 if match else 0.0,
            "method": "exact_match",
            "detail": f"parsed='{p}' gold='{g}'",
        }

    # ── math equation eval ──────────────────────────────────────────
    @staticmethod
    def _score_math(parsed: Optional[str], gold: str) -> Dict[str, Any]:
        if parsed is None:
            return {"score": 0.0, "method": "math_match", "detail": "parse_failed"}

        def extract_boxed(s: str) -> str:
            boxed_idx = s.rfind(r"\boxed{")
            if boxed_idx != -1:
                start_idx = boxed_idx + 7
                brace_count = 1
                end_idx = start_idx
                while end_idx < len(s) and brace_count > 0:
                    if s[end_idx] == '{': brace_count += 1
                    elif s[end_idx] == '}': brace_count -= 1
                    end_idx += 1
                if brace_count == 0:
                    return s[start_idx:end_idx-1]
            return s
            
        def clean_math(s: str) -> str:
            s = str(s).lower().strip()
            # remove \boxed{ ... }
            s = s.replace(r"\boxed{", "")
            s = s.replace(r"\text{", "")
            s = s.replace("}", "")
            s = s.replace("{", "")
            s = re.sub(r"[,\$\s]", "", s)
            # convert \frac a b to a/b
            s = re.sub(r'\\frac([a-z0-9]+)([a-z0-9]+)', r'(\1/\2)', s)
            s = re.sub(r'\\frac\(([^()]+)\)\(([^()]+)\)', r'(\1/\2)', s)
            return s

        p = clean_math(parsed)
        g = clean_math(extract_boxed(gold))
        
        # 1. Exact string match
        if p == g:
            return {"score": 1.0, "method": "math_match", "detail": "exact"}
            
        # 2. Sympy equivalence if importable
        try:
            import sympy
            from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
            transformations = (standard_transformations + (implicit_multiplication_application,))
            expr_p = parse_expr(p, transformations=transformations)
            expr_g = parse_expr(g, transformations=transformations)
            if sympy.simplify(expr_p - expr_g) == 0:
                return {"score": 1.0, "method": "math_match", "detail": "sympy"}
        except Exception:
            pass
            
        # 3. Float equivalence for decimals/fractions
        try:
            p_val = float(eval(p, {"__builtins__": {}}, {}))
            g_val = float(eval(g, {"__builtins__": {}}, {}))
            if abs(p_val - g_val) < 1e-6:
                return {"score": 1.0, "method": "math_match", "detail": "float_eval"}
        except Exception:
            pass
            
        return {"score": 0.0, "method": "math_match", "detail": f"parsed='{p}' gold='{g}'"}

    # ── game 24 equation eval ────────────────────────────────────────
    @staticmethod
    def _score_game24(text: str, question: str) -> Dict[str, Any]:
        """Try to extract and evaluate a math expression; check if it equals 24 and uses exact 4 numbers."""
        import collections
        
        # Extract the 4 numbers from the question (skip 24)
        q_nums = re.findall(r'\b\d+\b', question)
        q_nums = [n for n in q_nums if n != '24']
        
        # Strip/normalize LaTeX
        text = text.replace(r'\times', '*').replace(r'\div', '/')
        text = re.sub(r'\\frac\{([^{}]+)\}\{([^{}]+)\}', r'(\1/\2)', text)
        text = text.replace(r'\[', '').replace(r'\]', '')
        text = text.replace(r'\(', '').replace(r'\)', '')
        text = text.replace('$', '')
        
        # A robust regex for math expressions containing digits and operators
        expressions = re.findall(r"[\d(][\d+\-*/().\s]+[\d)]", text)
        
        for expr in expressions:
            expr = expr.replace("=", "").strip()
            if not expr:
                continue
                
            # Count numbers in the expression
            expr_nums = re.findall(r'\b\d+\b', expr)
            if sorted(expr_nums) != sorted(q_nums):
                continue
                
            try:
                # Very restricted eval: only digits and arithmetic
                if re.fullmatch(r"[\d+\-*/().\s]+", expr):
                    val = eval(expr, {"__builtins__": {}}, {})
                    if abs(val - 24) < 1e-6:
                        return {"score": 1.0, "method": "eval", "detail": f"expr='{expr}' -> {val}"}
            except Exception:
                pass
        return {"score": 0.0, "method": "eval", "detail": "no_valid_expression_or_wrong_numbers"}

    # ── LLM-as-judge ─────────────────────────────────────────────────
    def _score_llm_judge(
        self, question: str, reasoning: str, gold: str
    ) -> Dict[str, Any]:
        prompt = self.ORM_JUDGE_TEMPLATE.format(
            question=question, reasoning=reasoning, gold=gold
        )
        try:
            result = self.client.generate(prompt, temperature=0.0, max_tokens=8)
            txt = result.get("text", "").strip()
            if "1" in txt:
                return {"score": 1.0, "method": "llm_judge", "detail": txt}
            return {"score": 0.0, "method": "llm_judge", "detail": txt}
        except Exception as exc:
            logger.warning("LLM judge failed: %s", exc)
            return {"score": 0.0, "method": "llm_judge_error", "detail": str(exc)}
