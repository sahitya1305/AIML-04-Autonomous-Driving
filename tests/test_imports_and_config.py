import importlib

import pytest

from src.utils.config import apply_set_overrides, load_config, resolve_path

CORE = ["numpy", "cv2", "yaml", "scipy", "matplotlib", "onnxruntime"]
MODULES = ["src.perception.pipeline", "src.perception.obstacle", "src.perception.render", "src.tracking.byte_tracker",
           "src.lane_detection.classical", "src.depth_estimation.midas", "src.detection.onnx_detector", "src.detection.factory",
           "src.data.analysis", "src.data.convert", "src.simulator.synthetic_sim", "src.utils.wandb_utils", "src.app.session"]


@pytest.mark.parametrize("name", CORE)
def test_core_dependency_importable(name):
    importlib.import_module(name)


@pytest.mark.parametrize("name", MODULES)
def test_project_module_importable(name):
    importlib.import_module(name)


def test_default_config_loads(cfg):
    for key in ("detector", "tracker", "obstacle", "depth", "lanes", "wandb"):
        assert key in cfg
    assert cfg["wandb"]["enabled"] is False


def test_other_configs_layer_on_default():
    c = load_config("configs/ros2.yaml")
    assert c["detector"]["backend"] == "onnx" and "tracker" in c


def test_set_override():
    c = apply_set_overrides(load_config(), ["detector.conf=0.3", "run.max_frames=5"])
    assert c["detector"]["conf"] == 0.3 and c["run"]["max_frames"] == 5


def test_missing_config_raises():
    with pytest.raises(FileNotFoundError):
        load_config("configs/nope.yaml")


def test_paths_resolve_against_root():
    assert resolve_path("configs/default.yaml").exists()


def test_no_hardcoded_wandb_key_in_repo():
    import re
    root = resolve_path(".")
    pat = re.compile(r"[0-9a-f]{40}")
    for p in list(root.glob("src/**/*.py")) + list(root.glob("scripts/*.py")) + list(root.glob("configs/*.yaml")):
        assert not pat.search(p.read_text(encoding="utf-8", errors="ignore")), f"possible secret in {p}"
