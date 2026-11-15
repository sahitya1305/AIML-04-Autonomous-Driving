"""Download one real sample image (Ultralytics' bus.jpg: a street scene with a bus and pedestrians).

It is a small file from the Ultralytics assets repo (AGPL-3.0 project). It is NOT bundled in the ZIP for licence reasons.
For driving VIDEO use your own dashcam clip, or a KITTI raw sequence / BDD100K video (see docs/DATASETS.md).
"""
import _bootstrap  # noqa: F401
import urllib.request

from src.utils.config import resolve_path

URL = "https://ultralytics.com/images/bus.jpg"


def main():
    dest = resolve_path("datasets/samples/bus.jpg")
    if dest.exists():
        print("Already present:", dest)
        return
    try:
        urllib.request.urlretrieve(URL, dest)
    except Exception as e:
        raise SystemExit(f"Download failed ({e}). Download {URL} manually and save it as {dest}")
    print("Saved", dest)


if __name__ == "__main__":
    main()
