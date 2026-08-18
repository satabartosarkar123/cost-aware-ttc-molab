"""
hardware_monitor.py — Context-manager for measuring energy / CPU around Ollama calls.

Usage
-----
    with HardwareMonitor() as mon:
        result = client.generate(...)
    metrics = mon.metrics()   # dict with energy_joules, cpu_percent, etc.
"""

import time
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# ── Optional GPU support via pynvml ──────────────────────────────────
_PYNVML_AVAILABLE = False
try:
    import pynvml
    pynvml.nvmlInit()
    _PYNVML_AVAILABLE = True
    logger.info("pynvml initialized -- GPU energy tracking enabled.")
except Exception:
    pass

# ── CPU fallback via psutil ──────────────────────────────────────────
_PSUTIL_AVAILABLE = False
try:
    import psutil
    _PSUTIL_AVAILABLE = True
except Exception:
    pass


class HardwareMonitor:
    """
    Lightweight context-manager that records:
      • GPU energy (Joules)  — if pynvml is available
      • CPU percent          — if psutil is available
      • Wall-clock time
    """

    def __init__(self, gpu_index: int = 0):
        self.gpu_index = gpu_index
        self._handle: Optional[Any] = None
        self._start_energy: Optional[int] = None
        self._start_time: float = 0.0
        self._cpu_start: Optional[float] = None

        # Result storage
        self._energy_joules: Optional[float] = None
        self._avg_cpu_percent: Optional[float] = None
        self._wall_seconds: float = 0.0
        self._hardware_type: str = "unknown"

    # ── context manager ──────────────────────────────────────────────
    def __enter__(self):
        self._start_time = time.perf_counter()

        # GPU energy
        if _PYNVML_AVAILABLE:
            try:
                self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
                self._start_energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(self._handle)
                self._hardware_type = "gpu"
            except Exception as exc:
                logger.debug("GPU energy read failed at start: %s", exc)
                self._handle = None

        # CPU baseline
        if _PSUTIL_AVAILABLE:
            psutil.cpu_percent(interval=None)  # prime the counter
            if self._hardware_type == "unknown":
                self._hardware_type = "cpu"

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._wall_seconds = time.perf_counter() - self._start_time

        # GPU energy delta
        if self._handle is not None and self._start_energy is not None:
            try:
                end_energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(self._handle)
                # nvml returns millijoules
                self._energy_joules = (end_energy - self._start_energy) / 1000.0
            except Exception as exc:
                logger.debug("GPU energy read failed at end: %s", exc)

        # CPU percent (non-blocking snapshot)
        if _PSUTIL_AVAILABLE:
            self._avg_cpu_percent = psutil.cpu_percent(interval=None)

        return False  # do not suppress exceptions

    # ── public API ───────────────────────────────────────────────────
    def metrics(self) -> Dict[str, Any]:
        return {
            "hardware_type": self._hardware_type,
            "energy_joules": self._energy_joules,
            "avg_cpu_percent": self._avg_cpu_percent,
            "wall_seconds": round(self._wall_seconds, 4),
        }

    @staticmethod
    def detect_hardware() -> Dict[str, Any]:
        """Return a one-shot hardware summary for reports."""
        info: Dict[str, Any] = {
            "gpu_available": _PYNVML_AVAILABLE,
            "gpu_name": None,
            "gpu_memory_mb": None,
            "cpu_count": None,
            "ram_total_mb": None,
        }
        if _PYNVML_AVAILABLE:
            try:
                h = pynvml.nvmlDeviceGetHandleByIndex(0)
                info["gpu_name"] = pynvml.nvmlDeviceGetName(h)
                if isinstance(info["gpu_name"], bytes):
                    info["gpu_name"] = info["gpu_name"].decode()
                mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                info["gpu_memory_mb"] = round(mem.total / (1024 ** 2))
            except Exception:
                pass
        if _PSUTIL_AVAILABLE:
            info["cpu_count"] = psutil.cpu_count(logical=True)
            info["ram_total_mb"] = round(psutil.virtual_memory().total / (1024 ** 2))
        return info
