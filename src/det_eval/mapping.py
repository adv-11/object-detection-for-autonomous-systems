"""Label / taxonomy mapping between detector vocabularies and evaluation datasets.

COCO -> KITTI (closed-set detectors, no retraining)
---------------------------------------------------
KITTI class   COCO source                     Note
Car           car                              KITTI 'Van' GT is an ignore region for Car
Pedestrian    person (not riding)              KITTI 'Person_sitting' GT is an ignore region
Cyclist       bicycle + rider person (fused)   KITTI box spans rider *and* bicycle
(dropped)     every other COCO class           e.g. truck/bus/train: KITTI 'Truck'/'Tram' are not evaluated

The Cyclist definition is the real taxonomy mismatch: COCO annotates the rider as
'person' and the bicycle separately, while a KITTI Cyclist box encloses both. Two
strategies are implemented and both are reported:

* ``direct``: bicycle -> Cyclist, person -> Pedestrian. Box geometry is wrong
  (bicycle-only box) and riders become Pedestrian false positives.
* ``fused`` (default): a person and a bicycle form a rider pair when they overlap
  horizontally by >= 50% of the narrower box, intersect vertically, and the person's
  top edge is above the bicycle's. Pairs are matched greedily by ``s_person * s_bike``.
  The Cyclist box is the union of the two boxes with score ``s_person * s_bike``; the
  person stays a Pedestrian candidate with score ``s_person * (1 - s_bike)`` (a soft
  re-label, so no score threshold is needed). Unpaired bicycles (parked bikes) are
  dropped. The geometric rule was fixed a priori and not tuned on KITTI val.
"""

from __future__ import annotations

import numpy as np

COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush",
]

# Prompt vocabularies for open-vocabulary detectors.
VOCABULARIES = {
    "coco": COCO_NAMES,
    # KITTI-native class names (Person_sitting and Van are still ignore regions at eval time).
    "kitti": ["car", "van", "truck", "pedestrian", "cyclist", "person sitting", "tram"],
    # Descriptive prompt for the class whose name is least common in web image-text data.
    "kittidesc": ["car", "van", "truck", "pedestrian", "person riding a bicycle", "person sitting", "tram"],
}

# Open-vocab prompt -> KITTI evaluation class (prompts not listed are dropped).
PROMPT_TO_KITTI = {
    "car": "Car",
    "pedestrian": "Pedestrian",
    "cyclist": "Cyclist",
    "person riding a bicycle": "Cyclist",
}


def coco_name_to_id(categories: list[dict]) -> dict[str, int]:
    return {c["name"]: c["id"] for c in categories}


def _rider_pairs(pb, ps, bb, bs):
    """Greedy person-bicycle pairing; returns list of (person_idx, bike_idx)."""
    if len(pb) == 0 or len(bb) == 0:
        return []
    px1, py1, px2, py2 = (pb[:, i : i + 1] for i in range(4))
    bx1, by1, bx2, by2 = (bb[None, :, i] for i in range(4))
    ox = np.clip(np.minimum(px2, bx2) - np.maximum(px1, bx1), 0, None)
    oy = np.minimum(py2, by2) - np.maximum(py1, by1)
    narrow = np.minimum(px2 - px1, bx2 - bx1)
    ok = (ox >= 0.5 * narrow) & (oy > 0) & (py1 < by1)
    pi, bi = np.nonzero(ok)
    if len(pi) == 0:
        return []
    order = np.argsort(-(ps[pi] * bs[bi]), kind="stable")
    used_p, used_b, pairs = set(), set(), []
    for k in order:
        p, b = pi[k], bi[k]
        if p not in used_p and b not in used_b:
            used_p.add(p)
            used_b.add(b)
            pairs.append((p, b))
    return pairs


def coco_to_kitti(boxes, scores, names, strategy="fused"):
    """Map one image's COCO-vocabulary detections to KITTI classes.

    ``names`` are the class-name strings of each detection. Returns (boxes, scores, cls)
    where cls are KITTI class names.
    """
    names = np.asarray(names)
    out_b, out_s, out_c = [], [], []
    car = names == "car"
    out_b.append(boxes[car]); out_s.append(scores[car]); out_c += ["Car"] * int(car.sum())

    person, bike = names == "person", names == "bicycle"
    pb, ps, bb, bs = boxes[person], scores[person], boxes[bike], scores[bike]
    if strategy == "direct":
        out_b += [pb, bb]; out_s += [ps, bs]
        out_c += ["Pedestrian"] * len(ps) + ["Cyclist"] * len(bs)
    elif strategy == "fused":
        ped_s = ps.copy()
        for p, b in _rider_pairs(pb, ps, bb, bs):
            u = np.array([min(pb[p, 0], bb[b, 0]), min(pb[p, 1], bb[b, 1]),
                          max(pb[p, 2], bb[b, 2]), max(pb[p, 3], bb[b, 3])], dtype=np.float32)
            out_b.append(u[None]); out_s.append(np.array([ps[p] * bs[b]], dtype=np.float32))
            out_c.append("Cyclist")
            ped_s[p] = ps[p] * (1 - bs[b])
        out_b.append(pb); out_s.append(ped_s); out_c += ["Pedestrian"] * len(ps)
    else:
        raise ValueError(strategy)

    if not out_c:
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.array([], dtype=object)
    return np.concatenate(out_b), np.concatenate(out_s), np.array(out_c, dtype=object)


def prompts_to_kitti(boxes, scores, names):
    names = np.asarray(names)
    cls = np.array([PROMPT_TO_KITTI.get(n) for n in names], dtype=object)
    keep = cls != None  # noqa: E711 (elementwise)
    return boxes[keep], scores[keep], cls[keep]
