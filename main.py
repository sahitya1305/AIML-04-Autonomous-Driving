"""Single entry point.

  python main.py gui                       # open the demo window
  python main.py run --source <file> ...   # same options as scripts/run_pipeline.py
  python main.py check                     # environment / dependency check
  python main.py closed-loop --scenario stopped
"""
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        return
    cmd, rest = sys.argv[1], sys.argv[2:]
    if cmd == "gui":
        from src.app.demo_gui import launch
        launch(rest[0] if rest else None)
        return
    script = {"run": "run_pipeline.py", "check": "env_check.py", "closed-loop": "run_closed_loop_demo.py"}.get(cmd)
    if script is None:
        raise SystemExit(f"Unknown command '{cmd}'.\n{__doc__}")
    sys.argv = [script] + rest
    runpy.run_path(str(ROOT / "scripts" / script), run_name="__main__")


if __name__ == "__main__":
    main()
