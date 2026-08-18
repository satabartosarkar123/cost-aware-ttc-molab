"""
greedy_io.py — Standard Input-Output baseline (Greedy, Temp=0).
"""

import logging
from typing import Dict, Any, Optional

from strategies.base_strategy import BaseStrategy
from prompts.prompt_manager import get_io_prompt
from parsers import gsm8k_parser, strategyqa_parser, game24_parser

logger = logging.getLogger(__name__)


class GreedyIO(BaseStrategy):
    name = "greedy_io"

    def run(self, task, question, gold_answer, example_id, extra=None):
        prompt = get_io_prompt(task, question)
        temperature = self.config.get("temperature", 0.0)
        max_tokens = self.config.get("max_tokens", 1024)

        result = self._call_llm(prompt, temperature=temperature,
                                max_tokens=max_tokens, stage="generation")

        text = result.get("text", "")
        parsed = self._parse(task, text, extra)

        correct = self._check_correct(task, parsed.get("final_answer"), gold_answer)

        return {
            "strategy": self.name,
            "task": task,
            "example_id": example_id,
            "prompt": prompt,
            "output_text": text,
            "final_answer": parsed.get("final_answer"),
            "strict_answer": parsed.get("strict_answer"),
            "lenient_answer": parsed.get("lenient_answer"),
            "parse_method": parsed.get("parse_method"),
            "parse_success": parsed.get("parse_success"),
            "gold_answer": gold_answer,
            "correct": correct,
            "latency_seconds": result.get("latency_seconds", 0),
            "total_prompt_tokens": result.get("prompt_tokens", 0),
            "total_completion_tokens": result.get("completion_tokens", 0),
            "model_calls": 1,
            "hardware_metrics": [result.get("hardware_metrics", {})],
            "sub_results": [result],
            "error": result.get("error"),
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
