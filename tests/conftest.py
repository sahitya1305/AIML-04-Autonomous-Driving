import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest

from src.utils.config import load_config


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture()
def empty_scene():
    from src.data.synthetic import SyntheticScene, build_scenario
    return build_scenario(SyntheticScene(seed=0), "empty")
