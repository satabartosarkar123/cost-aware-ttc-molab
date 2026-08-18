"""
task_loader.py — Load & subsample GSM8K, StrategyQA, and Game of 24.

Each loader returns a list[dict] with at minimum:
    {"id": str, "question": str, "gold_answer": str, "task": str}
"""

import random
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# ── Hardcoded Game-of-24 problems (from Yao et al.) ─────────────────
GAME24_PROBLEMS = [
    {"nums": "4 9 10 13", "target": 24},
    {"nums": "1 2 3 4", "target": 24},
    {"nums": "1 3 4 6", "target": 24},
    {"nums": "2 3 5 12", "target": 24},
    {"nums": "1 4 6 8", "target": 24},
    {"nums": "1 5 5 5", "target": 24},
    {"nums": "3 3 8 8", "target": 24},
    {"nums": "2 7 8 9", "target": 24},
    {"nums": "1 2 7 7", "target": 24},
    {"nums": "3 4 5 6", "target": 24},
]


def _subsample(items: list, n: int, seed: int = 0) -> list:
    """Deterministic subsample of at most n items."""
    if len(items) <= n:
        return items
    rng = random.Random(seed)
    return rng.sample(items, n)


# ── GSM8K ────────────────────────────────────────────────────────────
def load_gsm8k(n: int = 10, seed: int = 0) -> List[Dict[str, Any]]:
    """Load GSM8K test split and subsample."""
    try:
        from datasets import load_dataset
        ds = load_dataset("openai/gsm8k", "main", split="test", trust_remote_code=True)
        items = list(ds)
    except Exception as exc:
        logger.error("Failed to load GSM8K: %s", exc)
        return []

    items = _subsample(items, n, seed)
    results = []
    for i, row in enumerate(items):
        # GSM8K gold answer is after "####" in the "answer" field
        answer_text = row.get("answer", "")
        gold = answer_text.split("####")[-1].strip() if "####" in answer_text else answer_text.strip()
        results.append({
            "id": f"gsm8k_{i}",
            "question": row["question"],
            "gold_answer": gold,
            "task": "gsm8k",
            "full_answer": answer_text,
        })
    logger.info("Loaded %d GSM8K examples.", len(results))
    return results


# ── StrategyQA (with fallback chain) ────────────────────────────────
def load_strategyqa(
    n: int = 10,
    seed: int = 0,
    datasets_to_try: Optional[List[str]] = None,
    fallback_dataset: str = "allenai/ai2_arc",
    fallback_subset: str = "ARC-Challenge",
) -> List[Dict[str, Any]]:
    """Try multiple HF slugs for StrategyQA; fallback to ARC-Challenge."""
    if datasets_to_try is None:
        datasets_to_try = [
            "wics/strategy_qa",
            "tasksource/strategy_qa",
            "ChilleD/StrategyQA",
        ]

    from datasets import load_dataset  # may raise later

    # Try StrategyQA slugs
    for slug in datasets_to_try:
        try:
            logger.info("Trying StrategyQA dataset: %s", slug)
            ds = load_dataset(slug, split="test", trust_remote_code=True)
            items = _subsample(list(ds), n, seed)
            results = []
            for i, row in enumerate(items):
                q = row.get("question", row.get("input", ""))
                a = row.get("answer", row.get("label", ""))
                if isinstance(a, bool):
                    a = "yes" if a else "no"
                a = str(a).strip().lower()
                results.append({
                    "id": f"strategyqa_{i}",
                    "question": q,
                    "gold_answer": a,
                    "task": "strategyqa",
                })
            if results:
                logger.info("Loaded %d StrategyQA examples from %s.", len(results), slug)
                return results
        except Exception as exc:
            logger.warning("StrategyQA slug %s failed: %s", slug, exc)

    # Try train split variants
    for slug in datasets_to_try:
        try:
            logger.info("Trying StrategyQA (train split): %s", slug)
            ds = load_dataset(slug, split="train", trust_remote_code=True)
            items = _subsample(list(ds), n, seed)
            results = []
            for i, row in enumerate(items):
                q = row.get("question", row.get("input", ""))
                a = row.get("answer", row.get("label", ""))
                if isinstance(a, bool):
                    a = "yes" if a else "no"
                a = str(a).strip().lower()
                results.append({
                    "id": f"strategyqa_{i}",
                    "question": q,
                    "gold_answer": a,
                    "task": "strategyqa",
                })
            if results:
                logger.info("Loaded %d StrategyQA examples (train split) from %s.", len(results), slug)
                return results
        except Exception as exc:
            logger.warning("StrategyQA slug %s (train) failed: %s", slug, exc)

    # Fallback to ARC-Challenge
    try:
        logger.warning("All StrategyQA slugs failed. Falling back to ARC-Challenge.")
        ds = load_dataset(fallback_dataset, fallback_subset, split="test", trust_remote_code=True)
        items = _subsample(list(ds), n, seed)
        results = []
        for i, row in enumerate(items):
            q = row.get("question", "")
            choices = row.get("choices", {})
            if isinstance(choices, dict):
                labels = choices.get("label", [])
                texts = choices.get("text", [])
                choice_str = " ".join(
                    f"({l}) {t}" for l, t in zip(labels, texts)
                )
                q = f"{q}\n{choice_str}"
            gold = row.get("answerKey", "")
            results.append({
                "id": f"arc_{i}",
                "question": q,
                "gold_answer": gold.strip(),
                "task": "strategyqa",  # treated as commonsense slot
            })
        logger.info("Loaded %d ARC-Challenge examples as StrategyQA fallback.", len(results))
        return results
    except Exception as exc:
        logger.error("ARC-Challenge fallback also failed: %s", exc)
        return []


# ── Game of 24 ───────────────────────────────────────────────────────
def load_game24(n: int = 10, seed: int = 0) -> List[Dict[str, Any]]:
    """Return hardcoded Game-of-24 problems."""
    items = _subsample(GAME24_PROBLEMS, n, seed)
    results = []
    for i, row in enumerate(items):
        results.append({
            "id": f"game24_{i}",
            "question": f"Use the numbers {row['nums']} and basic arithmetic (+, -, *, /) to make 24.",
            "gold_answer": "24",
            "task": "game24",
            "nums": row["nums"],
        })
    logger.info("Loaded %d Game-of-24 problems.", len(results))
    return results


# ── Convenience loader ───────────────────────────────────────────────
def load_all_tasks(config: dict) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load all task datasets according to config.

    Returns {"gsm8k": [...], "strategyqa": [...], "game24": [...]}.
    Missing tasks map to empty lists.
    """
    sampling = config.get("sampling", {})
    n = sampling.get("questions_per_task", 10)
    seed = sampling.get("seed", 0)
    task_cfg = config.get("tasks", {})

    tasks: Dict[str, List[Dict[str, Any]]] = {}

    # GSM8K
    tasks["gsm8k"] = load_gsm8k(n=n, seed=seed)

    # StrategyQA
    sqa = task_cfg.get("strategyqa", {})
    tasks["strategyqa"] = load_strategyqa(
        n=n,
        seed=seed,
        datasets_to_try=sqa.get("datasets_to_try"),
        fallback_dataset=sqa.get("fallback_dataset", "allenai/ai2_arc"),
        fallback_subset=sqa.get("fallback_subset", "ARC-Challenge"),
    )

    # Game of 24
    tasks["game24"] = load_game24(n=n, seed=seed)

    return tasks

