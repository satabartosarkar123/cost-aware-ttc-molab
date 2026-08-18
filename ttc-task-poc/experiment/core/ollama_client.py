"""
ollama_client.py — Robust Ollama API client with retries, timeouts, and metadata.

Uses the /api/generate endpoint (streaming disabled for simplicity).
"""

import time
import logging
import os
import requests
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


class OllamaClient:
    """Thin wrapper around the Ollama HTTP API."""

    def __init__(
        self,
        base_url: str = None,
        model: str = "qwen2.5:3b",
        timeout: int = 60,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ):
        # Priority: constructor arg → OLLAMA_BASE_URL env var → localhost
        if base_url is None:
            base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        self.base_url = base_url.rstrip("/")
        self.model = os.environ.get("OLLAMA_MODEL", model)
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    # ------------------------------------------------------------------
    # Health / model listing
    # ------------------------------------------------------------------
    def is_reachable(self) -> bool:
        """Return True if Ollama responds to /api/tags."""
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    def list_models(self) -> List[str]:
        """Return list of locally-installed model names."""
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=10)
            r.raise_for_status()
            return [m["name"] for m in r.json().get("models", [])]
        except Exception as exc:
            logger.warning("Could not list Ollama models: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def generate(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        stop: Optional[List[str]] = None,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Call /api/generate and return a structured result dict.

        Returns
        -------
        dict with keys:
            text            – generated text (str)
            latency_seconds – wall-clock seconds (float)
            prompt_tokens   – int (from Ollama response, may be 0)
            completion_tokens – int
            model           – model name used
            raw             – full JSON response from Ollama
            error           – None on success, str on failure
        """
        use_model = model or self.model
        use_timeout = timeout or self.timeout

        payload: Dict[str, Any] = {
            "model": use_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if stop:
            payload["options"]["stop"] = stop

        last_error: Optional[str] = None

        for attempt in range(1, self.max_retries + 1):
            t0 = time.perf_counter()
            try:
                resp = requests.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=use_timeout,
                )
                latency = time.perf_counter() - t0
                resp.raise_for_status()
                data = resp.json()

                return {
                    "text": data.get("response", ""),
                    "latency_seconds": round(latency, 4),
                    "prompt_tokens": data.get("prompt_eval_count", 0),
                    "completion_tokens": data.get("eval_count", 0),
                    "model": use_model,
                    "raw": data,
                    "error": None,
                }

            except requests.exceptions.Timeout:
                last_error = f"Timeout after {use_timeout}s (attempt {attempt})"
                logger.warning(last_error)
            except requests.exceptions.ConnectionError as exc:
                last_error = f"Connection error: {exc} (attempt {attempt})"
                logger.warning(last_error)
            except Exception as exc:
                last_error = f"Unexpected error: {exc} (attempt {attempt})"
                logger.warning(last_error)

            if attempt < self.max_retries:
                time.sleep(self.retry_delay)

        # All retries exhausted
        return {
            "text": "",
            "latency_seconds": 0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "model": use_model,
            "raw": {},
            "error": last_error,
        }
