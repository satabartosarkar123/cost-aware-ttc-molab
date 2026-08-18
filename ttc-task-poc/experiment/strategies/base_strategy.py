"""
base_strategy.py — Abstract base class for all TTC strategies.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class BaseStrategy(ABC):
    """
    Every strategy must implement `run()` and return a standardised result dict.
    """

    name: str = "base"

    def __init__(self, client, config: dict, monitor_cls=None, verifier=None):
        """
        Parameters
        ----------
        client : OllamaClient
        config : dict (strategy-specific section of config.yaml)
        monitor_cls : class HardwareMonitor (passed so strategies can wrap calls)
        verifier : OutcomeVerifier (only needed by BoN)
        """
        self.client = client
        self.config = config
        self.monitor_cls = monitor_cls
        self.verifier = verifier

    @abstractmethod
    def run(
        self,
        task: str,
        question: str,
        gold_answer: str,
        example_id: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute the strategy on a single example.

        Returns
        -------
        dict with at minimum:
            strategy        – str
            task            – str
            example_id      – str
            final_answer    – str | None
            correct         – bool
            parse_method    – str
            parse_success   – bool
            latency_seconds – float  (total wall time for all API calls)
            total_prompt_tokens     – int
            total_completion_tokens – int
            model_calls     – int
            hardware_metrics – list[dict]
            sub_results     – list[dict]  (raw per-call data)
            error           – str | None
        """
        ...

    # ── helpers ──────────────────────────────────────────────────────
    def _call_llm(self, prompt: str, *, temperature: float = 0.0,
                  max_tokens: int = 1024, stage: str = "generation") -> Dict[str, Any]:
        """Wrap a single LLM call with hardware monitoring."""
        hw_metrics = {}
        if self.monitor_cls is not None:
            mon = self.monitor_cls()
            mon.__enter__()
            result = self.client.generate(
                prompt, temperature=temperature, max_tokens=max_tokens,
            )
            mon.__exit__(None, None, None)
            hw_metrics = mon.metrics()
        else:
            result = self.client.generate(
                prompt, temperature=temperature, max_tokens=max_tokens,
            )
        result["stage"] = stage
        result["hardware_metrics"] = hw_metrics
        return result
