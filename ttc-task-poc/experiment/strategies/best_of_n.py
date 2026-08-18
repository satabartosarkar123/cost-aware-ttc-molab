"""
best_of_n.py — Outcome-Supervised Best-of-N (Cobbe et al., 2021; Lightman et al., ICLR 2024).

Official "ORM Best-of-N" approach:
  • Uses CoT prompt at temperature=0.7.
  • Samples N=5 independent reasoning chains.
  • Scores each via outcome verifier (exact-match for math, LLM-judge for commonsense).
  • Selects the generation with the highest score (argmax); ties → first.
  • Also logs "oracle" best-of-N (did ANY of the 5 match ground truth?).
"""

import logging
from typing import Dict, Any, Optional, List

from strategies.base_strategy import BaseStrategy
from prompts.prompt_manager import get_cot_prompt
from parsers import gsm8k_parser, strategyqa_parser, game24_parser

logger = logging.getLogger(__name__)


class BestOfN(BaseStrategy):
    name = "best_of_n"

    def run(self, task, question, gold_answer, example_id, extra=None):
        prompt = get_cot_prompt(task, question)
        temperature = self.config.get("temperature", 0.7)
        max_tokens = self.config.get("max_tokens", 1024)
        n_samples = self.config.get("n_samples", 5)

        candidates = []  # list of (text, parsed, score_result, llm_result)
        total_latency = 0.0
        total_prompt_tokens = 0
        total_completion_tokens = 0
        hw_metrics = []
        any_correct = False

        for i in range(n_samples):
            result = self._call_llm(
                prompt, temperature=temperature,
                max_tokens=max_tokens, stage=f"sample_{i+1}",
            )
            total_latency += result.get("latency_seconds", 0)
            total_prompt_tokens += result.get("prompt_tokens", 0)
            total_completion_tokens += result.get("completion_tokens", 0)
            hw_metrics.append(result.get("hardware_metrics", {}))

            text = result.get("text", "")
            parsed = self._parse(task, text, extra)
            parsed_answer = parsed.get("final_answer")

            # ── ORM scoring ──────────────────────────────────────────
            if self.verifier:
                score_result = self.verifier.score(
                    task=task,
                    question=question,
                    generated_text=text,
                    parsed_answer=parsed_answer,
                    gold_answer=gold_answer,
                )
                # The verifier scoring call may itself call the LLM for
                # commonsense tasks — we count that latency too
                total_latency += score_result.get("latency_added", 0)
            else:
                # Fallback: exact-match only
                correct_em = self._check_correct(task, parsed_answer, gold_answer)
                score_result = {
                    "score": 1.0 if correct_em else 0.0,
                    "method": "exact_match_fallback",
                    "detail": "",
                }

            if self._check_correct(task, parsed_answer, gold_answer):
                any_correct = True

            candidates.append({
                "index": i,
                "text": text,
                "parsed": parsed,
                "score": score_result["score"],
                "score_detail": score_result,
                "llm_result": result,
            })

        # ── Select best (argmax score, first on tie) ─────────────────
        best = max(candidates, key=lambda c: c["score"])
        best_parsed = best["parsed"]
        final_answer = best_parsed.get("final_answer")
        correct = self._check_correct(task, final_answer, gold_answer)

        return {
            "strategy": self.name,
            "task": task,
            "example_id": example_id,
            "prompt": prompt,
            "output_text": best["text"],
            "final_answer": final_answer,
            "strict_answer": best_parsed.get("strict_answer"),
            "lenient_answer": best_parsed.get("lenient_answer"),
            "parse_method": best_parsed.get("parse_method", "best_of_n"),
            "parse_success": best_parsed.get("parse_success", False),
            "gold_answer": gold_answer,
            "correct": correct,
            "oracle_correct": any_correct,
            "selected_index": best["index"],
            "candidate_scores": [c["score"] for c in candidates],
            "latency_seconds": round(total_latency, 4),
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "model_calls": n_samples + (1 if task != "gsm8k" else 0),
            "hardware_metrics": hw_metrics,
            "sub_results": [c["llm_result"] for c in candidates],
            "error": None,
        }

    @staticmethod
    def _parse(task, text, extra=None):
        if task == "gsm8k":
            return gsm8k_parser.parse(text)
        elif task == "game24":
            nums = extra.get("nums") if extra else None
            return game24_parser.parse(text, input_nums=nums)
        else:
            return strategyqa_parser.parse(text)

    @staticmethod
    def _check_correct(task, parsed_answer, gold_answer):
        if task == "gsm8k":
            return gsm8k_parser.check_correct(parsed_answer, gold_answer)
        elif task == "game24":
            return game24_parser.check_correct(parsed_answer, gold_answer)
        else:
            return strategyqa_parser.check_correct(parsed_answer, gold_answer)
