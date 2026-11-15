import time
from typing import Dict, List

import numpy as np


class Timer:
    """Context manager that stores elapsed milliseconds in a dict."""

    def __init__(self, store: Dict[str, float], key: str):
        self.store, self.key = store, key

    def __enter__(self):
        self._t = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.store[self.key] = (time.perf_counter() - self._t) * 1000.0
        return False


def latency_stats(times_ms: List[float]) -> Dict[str, float]:
    """Summary statistics for a list of latencies in milliseconds. Raises on empty input."""
    if len(times_ms) == 0:
        raise ValueError("No timing samples")
    a = np.asarray(times_ms, dtype=np.float64)
    mean = float(a.mean())
    return {
        "n": int(a.size),
        "mean_ms": mean,
        "median_ms": float(np.median(a)),
        "p95_ms": float(np.percentile(a, 95)),
        "min_ms": float(a.min()),
        "max_ms": float(a.max()),
        "std_ms": float(a.std()),
        "fps_mean": float(1000.0 / mean) if mean > 0 else float("inf"),
    }
