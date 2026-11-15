import json

from src.data.analysis import compute_stats
from src.data.convert import convert_records, to_yolo_lines
from src.data.labels import BDD_TO_UNIFIED, ImageRecord, parse_bdd100k, parse_kitti


def test_bdd_parse_and_stats(tmp_path):
    items = [{"name": "a.jpg", "attributes": {"weather": "clear"}, "labels": [
        {"category": "car", "box2d": {"x1": 10, "y1": 10, "x2": 110, "y2": 60}},
        {"category": "car", "box2d": {"x1": 200, "y1": 10, "x2": 300, "y2": 60}},
        {"category": "lane", "poly2d": []},
        {"category": "pedestrian", "box2d": {"x1": 5, "y1": 5, "x2": 25, "y2": 65}}]}]
    p = tmp_path / "l.json"; p.write_text(json.dumps(items))
    recs = parse_bdd100k(p)
    s = compute_stats(recs, "t")
    assert s["n_objects"] == 3 and s["classes"]["car"]["count"] == 2 and s["attributes"]["weather"]["clear"] == 1


def test_yolo_conversion_normalised_and_clipped():
    rec = ImageRecord("a.jpg", 1280, 720, objects=[("car", (-10, 100, 640, 460)), ("traffic sign", (0, 0, 10, 10)), ("rider", (0, 0, 1, 1))])
    lines = to_yolo_lines(rec, BDD_TO_UNIFIED)
    assert len(lines) == 1
    cid, cx, cy, w, h = lines[0].split()
    assert cid == "2" and abs(float(cx) - 0.25) < 1e-3 and abs(float(w) - 0.5) < 1e-3 and 0 <= float(cy) <= 1


def test_kitti_parse_and_convert(tmp_path):
    lab = tmp_path / "label_2"; lab.mkdir()
    (lab / "000001.txt").write_text("Car 0 0 0 100 100 200 180 1 1 1 0 0 0 0\nDontCare -1 -1 -10 0 0 10 10 -1 -1 -1 0 0 0 0\nCyclist 0 0 0 300 120 340 200 1 1 1 0 0 0 0\n")
    recs = parse_kitti(lab)
    assert len(recs) == 1 and len(recs[0].objects) == 3
    out = tmp_path / "yolo"
    st = convert_records(recs, None, out, "kitti", require_image=False)
    assert st["val"] + st["train"] == 1 and (out / "data.yaml").exists()
    txt = next((out / "labels").rglob("*.txt")).read_text().splitlines()
    assert [l.split()[0] for l in txt] == ["2", "1"]       # car, cyclist->rider; DontCare dropped
