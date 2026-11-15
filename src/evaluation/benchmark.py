"""Latency benchmark helper (honest timing: warm-up, many runs, mean/median/p95, no cherry-picking)."""
from __future__ import annotations

import platform
import time
from typing import Callable, Dict, List

from src.utils.timing import latency_stats


def time_callable(fn: Callable[[], object], warmup: int = 5, runs: int = 30) -> Dict[str, float]:
    for _ in range(warmup):
        fn()
    times: List[float] = []
    for _ in range(runs):
        t = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t) * 1000.0)
    return latency_stats(times)


def machine_info() -> Dict[str, str]:
    info = {"platform": platform.platform(), "processor": platform.processor(), "python": platform.python_version()}
    try:
        import onnxruntime as ort
        info["onnxruntime"] = ort.__version__
    except ImportError:
        pass
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = str(torch.cuda.is_available())
    except ImportError:
        pass
    return info


def speedup(baseline_ms: float, other_ms: float) -> float:
    return baseline_ms / other_ms if other_ms > 0 else float("inf")
