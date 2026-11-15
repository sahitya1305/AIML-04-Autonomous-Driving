import json

import pytest

from src.evaluation.benchmark import speedup, time_callable
from src.utils.config import resolve_path
from src.utils.timing import latency_stats


def test_latency_stats():
    s = latency_stats([10, 20, 30])
    assert s["mean_ms"] == 20 and s["median_ms"] == 20 and abs(s["fps_mean"] - 50) < 1e-9
    with pytest.raises(ValueError):
        latency_stats([])


def test_time_callable_counts_runs():
    calls = []
    s = time_callable(lambda: calls.append(1), warmup=2, runs=5)
    assert len(calls) == 7 and s["n"] == 5


def test_speedup():
    assert speedup(100, 50) == 2.0


@pytest.mark.parametrize("nb", ["01_tensorrt_benchmark_colab", "02_finetune_yolo_colab"])
def test_notebooks_are_valid_and_have_no_outputs_or_keys(nb):
    d = json.loads(resolve_path(f"notebooks/{nb}.ipynb").read_text(encoding="utf-8"))
    assert d["nbformat"] == 4
    for c in d["cells"]:
        if c["cell_type"] == "code":
            assert c["outputs"] == [], "notebook must ship without fabricated outputs"
            src = "".join(c["source"])
            assert "wandb_v1" not in src and "api_key=" not in src.lower()
