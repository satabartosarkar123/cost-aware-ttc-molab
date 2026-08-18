"""
self_consistency.py — Self-Consistency with Majority Voting (Wang et al., ICLR 2023).

Official "Sample-and-Marginalize" approach:
  • Uses CoT prompt at temperature=0.7.
  • Samples N=5 independent reasoning chains.
  • Parses each answer, performs unweighted majority vote.
  • Tie-breaker: first appearance order.
"""

import logging
from collections import Counter
from typing import Dict, Any, Optional, List

from strategies.base_strategy import BaseStrategy
from prompts.prompt_manager import get_cot_prompt
from parsers import gsm8k_parser, strategyqa_parser, game24_parser

logger = logging.getLogger(__name__)


class SelfConsistency(BaseStrategy):
    name = "self_consistency"

    def run(self, task, question, gold_answer, example_id, extra=None):
        prompt = get_cot_prompt(task, question)
        temperature = self.config.get("temperature", 0.7)
        max_tokens = self.config.get("max_tokens", 1024)
        n_samples = self.config.get("n_samples", 5)

        all_results = []
        all_answers: List[Optional[str]] = []
        total_latency = 0.0
        total_prompt_tokens = 0
        total_completion_tokens = 0
        hw_metrics = []
        any_correct = False  # oracle tracking

        for i in range(n_samples):
            result = self._call_llm(
                prompt, temperature=temperature,
                max_tokens=max_tokens, stage=f"sample_{i+1}",
            )
            all_results.append(result)
            total_latency += result.get("latency_seconds", 0)
            total_prompt_tokens += result.get("prompt_tokens", 0)
            total_completion_tokens += result.get("completion_tokens", 0)
            hw_metrics.append(result.get("hardware_metrics", {}))

            text = result.get("text", "")
            parsed = self._parse(task, text, extra)
            answer = parsed.get("final_answer")
            all_answers.append(answer)

            # Track oracle correctness
            if self._check_correct(task, answer, gold_answer):
                any_correct = True

        # ── Majority vote ────────────────────────────────────────────
        valid_answers = [a for a in all_answers if a is not None]

        if valid_answers:
            # Counter preserves insertion order for ties (Python 3.7+)
            counts = Counter(valid_answers)
            majority_answer = counts.most_common(1)[0][0]
        else:
            majority_answer = None

        # Parse info for the final voted answer
        parse_success = majority_answer is not None
        correct = self._check_correct(task, majority_answer, gold_answer)

        # Build vote distribution
        vote_dist = dict(Counter(str(a) for a in all_answers))

        return {
            "strategy": self.name,
            "task": task,
            "example_id": example_id,
            "prompt": prompt,
            "output_text": "\n---\n".join(r.get("text", "") for r in all_results),
            "final_answer": majority_answer,
            "strict_answer": None,
            "lenient_answer": None,
            "parse_method": "majority_vote",
            "parse_success": parse_success,
            "gold_answer": gold_answer,
            "correct": correct,
            "oracle_correct": any_correct,
            "vote_distribution": vote_dist,
            "all_answers": all_answers,
            "latency_seconds": round(total_latency, 4),
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "model_calls": n_samples,
            "hardware_metrics": hw_metrics,
            "sub_results": all_results,
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
