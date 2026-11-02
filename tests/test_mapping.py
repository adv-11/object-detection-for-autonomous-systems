"""COCO -> KITTI taxonomy mapping on synthetic rider / bystander layouts (CPU only)."""

import numpy as np

from det_eval.mapping import cap, coco_to_kitti, prompts_to_kitti


def run(dets, strategy="fused"):
    boxes = np.array([d[1] for d in dets], dtype=np.float32).reshape(-1, 4)
    scores = np.array([d[2] for d in dets], dtype=np.float32)
    names = [d[0] for d in dets]
    return coco_to_kitti(boxes, scores, names, strategy)


RIDER = ("person", (100, 40, 140, 140), 0.9)
BIKE = ("bicycle", (90, 90, 160, 150), 0.8)


def test_rider_on_bike_becomes_cyclist_with_union_box():
    b, s, c = run([RIDER, BIKE])
    cyc = c == "Cyclist"
    assert cyc.sum() == 1
    np.testing.assert_allclose(b[cyc][0], [90, 40, 160, 150])
    np.testing.assert_allclose(s[cyc][0], 0.9 * 0.8, rtol=1e-6)
    np.testing.assert_allclose(s[c == "Pedestrian"][0], 0.9 * 0.2, rtol=1e-6)  # soft re-label


def test_parked_bike_is_dropped():
    b, s, c = run([BIKE])
    assert len(c) == 0


def test_distant_pedestrian_behind_parked_bike_is_not_a_rider():
    small_far = ("person", (110, 60, 125, 100), 0.9)  # feet above the bike's lower half
    _, _, c = run([small_far, BIKE])
    assert list(c) == ["Pedestrian"]
    _, _, c_loose = run([small_far, BIKE], "fused_loose")
    assert "Cyclist" in c_loose  # the loose rule accepts it (that is why it was tightened)


def test_two_riders_two_bikes_pair_one_to_one():
    r2 = ("person", (300, 40, 340, 140), 0.85)
    b2 = ("bicycle", (290, 90, 360, 150), 0.7)
    b, s, c = run([RIDER, BIKE, r2, b2])
    assert (c == "Cyclist").sum() == 2
    assert (c == "Pedestrian").sum() == 2


def test_duplicate_person_box_on_rider_is_down_weighted():
    upper = ("person", (102, 40, 138, 100), 0.6)  # partial box inside the rider
    _, s, c = run([RIDER, upper, BIKE])
    np.testing.assert_allclose(s[c == "Pedestrian"], [0.9 * 0.2, 0.6 * 0.2], rtol=1e-6)


def test_direct_mapping_keeps_bicycle_box():
    b, _, c = run([RIDER, BIKE], "direct")
    np.testing.assert_allclose(b[c == "Cyclist"][0], BIKE[1])


def test_lengths_consistent_and_empty_input():
    b, s, c = run([])
    assert b.shape == (0, 4) and len(s) == len(c) == 0
    b, s, c = run([RIDER, BIKE, ("car", (0, 0, 10, 10), 0.5), ("truck", (0, 0, 9, 9), 0.5)])
    assert len(b) == len(s) == len(c)


def test_prompt_mapping_mask_is_elementwise_and_case_insensitive():
    boxes = np.zeros((4, 4), np.float32)
    scores = np.array([0.9, 0.8, 0.7, 0.6], np.float32)
    b, s, c = prompts_to_kitti(boxes, scores, ["Car", "van", "Cyclist", "person riding a bicycle"])
    assert list(c) == ["Car", "Cyclist", "Cyclist"]


def test_cap_keeps_top_k_by_score():
    b = np.zeros((5, 4), np.float32)
    s = np.array([0.1, 0.9, 0.5, 0.7, 0.3], np.float32)
    c = np.array(list("abcde"), dtype=object)
    _, s2, c2 = cap(b, s, c, 2)
    assert list(c2) == ["b", "d"]
