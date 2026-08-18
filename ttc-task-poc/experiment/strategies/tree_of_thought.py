"""
tree_of_thought.py — Zero-Shot Tree-of-Thought BFS (Yao et al., NeurIPS 2023).

Official Zero-Shot ToT-BFS approach (Appendix B.1):
  Step 1: Generate k=3 high-level strategies using CoT prompt (Temp=0.7).
  Step 2: Vote to select the best strategy (Temp=0.0).
  Step 3: Conditioned on winning strategy, generate k=3 solution drafts (Temp=0.7).
  Step 4: Vote to select the best draft (Temp=0.0).
  Step 5: Parse final answer from the winning draft.
"""

import logging
from typing import Dict, Any, Optional

from strategies.base_strategy import BaseStrategy
from prompts.prompt_manager import get_cot_prompt, get_vote_prompt
from parsers import gsm8k_parser, strategyqa_parser, game24_parser, vote_parser

logger = logging.getLogger(__name__)


class TreeOfThought(BaseStrategy):
    name = "tree_of_thought"

    def run(self, task, question, gold_answer, example_id, extra=None):
        cfg = self.config
        propose_temp = cfg.get("propose_temperature", 0.7)
        vote_temp = cfg.get("vote_temperature", 0.0)
        max_gen = cfg.get("max_tokens_generation", 1024)
        max_vote = cfg.get("max_tokens_vote", 256)
        k = cfg.get("breadth_k", 3)

        total_latency = 0.0
        total_pt = 0
        total_ct = 0
        model_calls = 0
        hw_metrics = []
        sub_results = []

        # ──────────────────────────────────────────────────────────────
        # STEP 1: Generate k high-level strategies
        # ──────────────────────────────────────────────────────────────
        strategies_texts = []
        strategy_prompt_base = get_cot_prompt(task, question)

        for i in range(k):
            strategy_prompt = (
                f"{strategy_prompt_base}\n\n"
                f"Provide a unique high-level strategy (strategy #{i+1} of {k}). "
                f"Be creative and consider a different approach from other strategies."
            )
            r = self._call_llm(
                strategy_prompt, temperature=propose_temp,
                max_tokens=max_gen, stage=f"propose_strategy_{i+1}",
            )
            sub_results.append(r)
            total_latency += r.get("latency_seconds", 0)
            total_pt += r.get("prompt_tokens", 0)
            total_ct += r.get("completion_tokens", 0)
            hw_metrics.append(r.get("hardware_metrics", {}))
            model_calls += 1
            strategies_texts.append(r.get("text", f"[Strategy {i+1} generation failed]"))

        # ──────────────────────────────────────────────────────────────
        # STEP 2: Vote on the best strategy
        # ──────────────────────────────────────────────────────────────
        vote_instruction = f"Choose the best strategy for answering: {question}"
        vote_prompt = get_vote_prompt(vote_instruction, strategies_texts)
        r_vote = self._call_llm(
            vote_prompt, temperature=vote_temp,
            max_tokens=max_vote, stage="vote_strategy",
        )
        sub_results.append(r_vote)
        total_latency += r_vote.get("latency_seconds", 0)
        total_pt += r_vote.get("prompt_tokens", 0)
        total_ct += r_vote.get("completion_tokens", 0)
        hw_metrics.append(r_vote.get("hardware_metrics", {}))
        model_calls += 1

        vote_parsed = vote_parser.parse(r_vote.get("text", ""))
        try:
            best_strategy_idx = int(vote_parsed["final_answer"]) - 1
            if best_strategy_idx < 0 or best_strategy_idx >= k:
                best_strategy_idx = 0
        except (ValueError, TypeError):
            best_strategy_idx = 0

        winning_strategy = strategies_texts[best_strategy_idx]

        # ──────────────────────────────────────────────────────────────
        # STEP 3: Generate k solution drafts conditioned on winning strategy
        # ──────────────────────────────────────────────────────────────
        drafts = []
        for i in range(k):
            draft_prompt = (
                f"Question: {question}\n\n"
                f"Use the following strategy:\n{winning_strategy}\n\n"
                f"Now write a complete solution (draft #{i+1} of {k}). "
                f"End with: The answer is <your answer>."
            )
            r = self._call_llm(
                draft_prompt, temperature=propose_temp,
                max_tokens=max_gen, stage=f"draft_{i+1}",
            )
            sub_results.append(r)
            total_latency += r.get("latency_seconds", 0)
            total_pt += r.get("prompt_tokens", 0)
            total_ct += r.get("completion_tokens", 0)
            hw_metrics.append(r.get("hardware_metrics", {}))
            model_calls += 1
            drafts.append(r.get("text", f"[Draft {i+1} generation failed]"))

        # ──────────────────────────────────────────────────────────────
        # STEP 4: Vote on the best draft
        # ──────────────────────────────────────────────────────────────
        vote_instruction2 = f"Choose the best and most correct solution for: {question}"
        vote_prompt2 = get_vote_prompt(vote_instruction2, drafts)
        r_vote2 = self._call_llm(
            vote_prompt2, temperature=vote_temp,
            max_tokens=max_vote, stage="vote_draft",
        )
        sub_results.append(r_vote2)
        total_latency += r_vote2.get("latency_seconds", 0)
        total_pt += r_vote2.get("prompt_tokens", 0)
        total_ct += r_vote2.get("completion_tokens", 0)
        hw_metrics.append(r_vote2.get("hardware_metrics", {}))
        model_calls += 1

        vote_parsed2 = vote_parser.parse(r_vote2.get("text", ""))
        try:
            best_draft_idx = int(vote_parsed2["final_answer"]) - 1
            if best_draft_idx < 0 or best_draft_idx >= k:
                best_draft_idx = 0
        except (ValueError, TypeError):
            best_draft_idx = 0

        winning_draft = drafts[best_draft_idx]

        # ──────────────────────────────────────────────────────────────
        # STEP 5: Parse final answer from winning draft
        # ──────────────────────────────────────────────────────────────
        parsed = self._parse(task, winning_draft, extra)
        final_answer = parsed.get("final_answer")
        correct = self._check_correct(task, final_answer, gold_answer)

        return {
            "strategy": self.name,
            "task": task,
            "example_id": example_id,
            "prompt": f"[ToT multi-stage — {model_calls} calls]",
            "output_text": winning_draft,
            "final_answer": final_answer,
            "strict_answer": parsed.get("strict_answer"),
            "lenient_answer": parsed.get("lenient_answer"),
            "parse_method": parsed.get("parse_method"),
            "parse_success": parsed.get("parse_success"),
            "gold_answer": gold_answer,
            "correct": correct,
            "winning_strategy_index": best_strategy_idx,
            "winning_draft_index": best_draft_idx,
            "latency_seconds": round(total_latency, 4),
            "total_prompt_tokens": total_pt,
            "total_completion_tokens": total_ct,
            "model_calls": model_calls,
            "hardware_metrics": hw_metrics,
            "sub_results": sub_results,
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
