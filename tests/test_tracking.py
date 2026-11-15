import numpy as np

from src.perception.types import Detection
from src.tracking.byte_tracker import ByteTracker, iou_matrix


def det(x, y, w=50, h=80, conf=0.9, cls=2, label="car"):
    return Detection((x, y, x + w, y + h), conf, cls, label)


def test_iou_basic():
    a = np.array([[0, 0, 10, 10.0]])
    assert iou_matrix(a, a)[0, 0] == 1.0
    assert iou_matrix(a, np.array([[20, 20, 30, 30.0]]))[0, 0] == 0.0
    assert abs(iou_matrix(a, np.array([[5, 0, 15, 10.0]]))[0, 0] - 1 / 3) < 1e-6
    assert iou_matrix(np.zeros((0, 4)), a).shape == (0, 1)


def test_ids_stable_for_moving_objects():
    tr = ByteTracker()
    ids_seen = []
    for f in range(30):
        out = tr.update([det(100 + 4 * f, 100), det(400 - 3 * f, 150, label="truck", cls=7)])
        ids_seen.append({t.label: t.track_id for t in out})
    last = ids_seen[-1]
    assert len(last) == 2 and last["car"] != last["truck"]
    for d in ids_seen[3:]:
        assert d == last


def test_id_survives_short_occlusion():
    tr = ByteTracker(track_buffer=30)
    for f in range(10):
        out = tr.update([det(100 + 3 * f, 100)])
    tid = out[0].track_id
    for f in range(10, 16):
        tr.update([])                       # missed for 6 frames
    out = tr.update([det(100 + 3 * 16, 100)])
    assert out and out[0].track_id == tid


def test_low_confidence_box_keeps_track():
    tr = ByteTracker()
    for f in range(8):
        out = tr.update([det(100 + 2 * f, 100, conf=0.9)])
    tid = out[0].track_id
    out = tr.update([det(100 + 2 * 8, 100, conf=0.3)])    # below high threshold -> second association
    assert out and out[0].track_id == tid


def test_new_object_gets_new_id():
    tr = ByteTracker()
    for f in range(5):
        tr.update([det(100, 100)])
    for f in range(5):
        out = tr.update([det(100, 100), det(300, 100)])
    assert len({t.track_id for t in out}) == 2


def test_track_dropped_after_buffer():
    tr = ByteTracker(track_buffer=5)
    for f in range(5):
        tr.update([det(100, 100)])
    for f in range(10):
        tr.update([])
    assert tr.tracks == []


def test_low_conf_alone_does_not_spawn_track():
    tr = ByteTracker()
    for _ in range(5):
        assert tr.update([det(100, 100, conf=0.3)]) == []
