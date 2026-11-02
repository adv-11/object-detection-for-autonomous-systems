"""KITTI ignore semantics and AP|R40 on hand-checkable synthetic cases (CPU only)."""

import pytest

from det_eval.data import KITTI_CAT_ID
from det_eval.metrics import kitti_eval, kitti_level_eval

CAR, PED = KITTI_CAT_ID["Car"], KITTI_CAT_ID["Pedestrian"]


def gt(*anns):
    out = []
    for i, (cat, box, kw) in enumerate(anns, 1):
        x1, y1, x2, y2 = box
        a = dict(id=i, image_id=1, category_id=cat, bbox=[x1, y1, x2 - x1, y2 - y1], area=(x2 - x1) * (y2 - y1),
                 iscrowd=0, kitti_ignore=0, ignore_reason=None, kitti_difficulty="easy")
        a.update(kw)
        out.append(a)
    cats = [dict(id=v, name=k) for k, v in KITTI_CAT_ID.items()]
    return dict(images=[dict(id=1, width=1242, height=375)], annotations=out, categories=cats)


def det(cat, box, score):
    x1, y1, x2, y2 = box
    return dict(image_id=1, category_id=cat, bbox=[x1, y1, x2 - x1, y2 - y1], score=score)


CAR_BOX, VAN_BOX = (100, 100, 200, 160), (400, 100, 500, 160)
NEIGHBOUR = dict(kitti_ignore=1, ignore_reason="neighbour:Van", kitti_difficulty=None)
DONTCARE = dict(iscrowd=1, kitti_ignore=1, ignore_reason="dontcare", kitti_difficulty=None)


def test_perfect_detector_is_100():
    g = gt((CAR, CAR_BOX, {}))
    assert kitti_level_eval(g, [det(CAR, CAR_BOX, 0.9)], "moderate")["Car"] == pytest.approx(100)


def test_one_detection_on_neighbour_is_ignored():
    g = gt((CAR, CAR_BOX, {}), (CAR, VAN_BOX, NEIGHBOUR))
    d = [det(CAR, VAN_BOX, 0.99), det(CAR, CAR_BOX, 0.5)]
    assert kitti_level_eval(g, d, "moderate")["Car"] == pytest.approx(100)


def test_duplicate_on_neighbour_is_a_false_positive():
    # devkit: a neighbour object absorbs one detection; the second is an FP ranked above the TP
    g = gt((CAR, CAR_BOX, {}), (CAR, VAN_BOX, NEIGHBOUR))
    d = [det(CAR, VAN_BOX, 0.99), det(CAR, (402, 101, 501, 161), 0.98), det(CAR, CAR_BOX, 0.5)]
    assert kitti_level_eval(g, d, "moderate")["Car"] == pytest.approx(50)


def test_partial_box_inside_hard_object_is_a_false_positive():
    # IoU 0.25 with an ignored 'hard' car: the devkit matches by IoU, so this is an FP
    g = gt((CAR, CAR_BOX, {}), (CAR, (400, 100, 520, 180), dict(kitti_difficulty="hard")))
    d = [det(CAR, (430, 120, 490, 160), 0.99), det(CAR, CAR_BOX, 0.5)]
    assert kitti_level_eval(g, d, "moderate")["Car"] == pytest.approx(50)


def test_missed_hard_object_does_not_hurt_moderate():
    g = gt((CAR, CAR_BOX, {}), (CAR, (400, 100, 520, 180), dict(kitti_difficulty="hard")))
    assert kitti_level_eval(g, [det(CAR, CAR_BOX, 0.9)], "moderate")["Car"] == pytest.approx(100)
    assert kitti_level_eval(g, [det(CAR, CAR_BOX, 0.9)], "hard")["Car"] == pytest.approx(50)


def test_dontcare_absorbs_any_number_of_detections():
    g = gt((CAR, CAR_BOX, {}), (CAR, (600, 50, 900, 200), DONTCARE))
    d = [det(CAR, (610, 60, 700, 120), 0.99), det(CAR, (720, 60, 800, 120), 0.98), det(CAR, CAR_BOX, 0.5)]
    assert kitti_level_eval(g, d, "moderate")["Car"] == pytest.approx(100)


def test_car_iou_threshold_is_0_7_and_pedestrian_0_5():
    shifted = (100, 100, 200, 148)  # IoU 48/60 = 0.8
    loose = (100, 100, 200, 136)  # IoU 0.6
    g = gt((CAR, CAR_BOX, {}), (PED, (300, 100, 330, 160), {}))
    r = kitti_level_eval(g, [det(CAR, loose, 0.9), det(PED, (300, 100, 330, 136), 0.9)], "moderate")
    assert r["Car"] == pytest.approx(0)  # 0.6 < 0.7
    assert r["Pedestrian"] == pytest.approx(100)  # 0.6 >= 0.5
    assert kitti_level_eval(g, [det(CAR, shifted, 0.9)], "moderate")["Car"] == pytest.approx(100)


def test_short_detections_are_dropped_at_moderate():
    tiny = (100, 100, 120, 120)  # 20 px tall, below the 25 px moderate minimum
    g = gt((CAR, CAR_BOX, {}))
    r = kitti_level_eval(g, [det(CAR, tiny, 0.99), det(CAR, CAR_BOX, 0.5)], "moderate")
    assert r["Car"] == pytest.approx(100)


def test_coco_style_protocol_uses_the_same_ignore_rules():
    g = gt((CAR, CAR_BOX, {}), (CAR, VAN_BOX, NEIGHBOUR))
    d = [det(CAR, VAN_BOX, 0.99), det(CAR, (402, 101, 501, 161), 0.98), det(CAR, CAR_BOX, 0.5)]
    ap50 = kitti_eval(g, d)["coco_style"]["per_class_AP50"]["Car"]
    assert ap50 == pytest.approx(50, abs=1)
