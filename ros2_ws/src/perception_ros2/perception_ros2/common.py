"""Shared helper: make the project's `src` package importable from inside a ROS 2 node."""
import os
import sys
from pathlib import Path


def add_project_to_path(explicit: str = "") -> Path:
    root = explicit or os.environ.get("AIML04_ROOT", "")
    if not root:
        raise RuntimeError("Set the project root first:  export AIML04_ROOT=/mnt/c/Users/<you>/Projects/AIML-04-Autonomous-Driving  "
                           "(scripts/ros2_run_demo.sh does this for you)")
    p = Path(root).expanduser()
    if not (p / "src" / "perception").exists():
        raise RuntimeError(f"AIML04_ROOT does not look like the project root: {p}")
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
    return p
