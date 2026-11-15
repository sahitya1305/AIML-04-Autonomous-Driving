import numpy as np
import pytest

from src.depth_estimation.midas import box_proximity, colorize_depth, normalize_depth


def test_normalize_range_and_orientation():
    raw = np.linspace(5, 50, 10000, dtype=np.float32).reshape(100, 100)
    n = normalize_depth(raw)
    assert n.dtype == np.float32 and 0.0 <= n.min() and n.max() <= 1.0
    assert n[-1, -1] > n[0, 0]


def test_normalize_constant_map_is_safe():
    assert normalize_depth(np.ones((10, 10))).max() == 0.0


def test_colorize_shape():
    out = colorize_depth(np.random.rand(20, 30).astype(np.float32))
    assert out.shape == (20, 30, 3) and out.dtype == np.uint8


def test_box_proximity_uses_lower_part_of_box():
    d = np.zeros((100, 100), np.float32)
    d[60:100, 20:60] = 0.9
    assert box_proximity(d, (20, 20, 60, 100)) > 0.8
    assert box_proximity(d, (0, 0, 1, 1)) is None


@pytest.mark.needs_torch
@pytest.mark.needs_network
def test_midas_real_model_runs():
    """Runs only if torch+timm are installed AND the MiDaS weights can be downloaded/cached."""
    pytest.importorskip("torch"); pytest.importorskip("timm")
    from src.depth_estimation.midas import MidasDepth
    try:
        m = MidasDepth("MiDaS_small")
    except RuntimeError as e:
        pytest.skip(f"MiDaS could not be loaded (offline?): {e}")
    img = (np.random.rand(240, 320, 3) * 255).astype(np.uint8)
    d = m.estimate(img)
    assert d.shape == (240, 320) and 0 <= d.min() and d.max() <= 1
