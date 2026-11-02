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
  horizontally by >= 50% of the narrower box, the person's top edge is above the
  bicycle's, the person's bottom edge lies in the lower half of the bicycle's vertical
  span (or <= 15% below it), and the person is 0.8-3x the bicycle's height. Pairs are
  matched greedily by ``s_person * s_bike * overlap fraction``. The Cyclist box is the
  union of the two boxes with score ``s_person * s_bike``; the person stays a Pedestrian
  candidate with score ``s_person * (1 - s_bike)`` (a soft re-label, so no score
  threshold is needed), and other person boxes lying >= 50% inside the rider are
  down-weighted the same way. Unpaired bicycles (parked bikes) are dropped.
* ``fused_loose``: the first version (overlap + top-edge test only, no duplicate
  suppression), kept as a sensitivity variant.

The geometric rules were fixed from synthetic cases during code review, before any
full KITTI val evaluation, and are not tuned on KITTI val; all three are reported.
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


def _rider_pairs(pb, ps, bb, bs, strict=True):
    """Greedy person-bicycle pairing; returns list of (person_idx, bike_idx).

    loose:  horizontal overlap >= 50% of the narrower box, vertical intersection, person top
            above bike top; ranked by s_person * s_bike.
    strict: additionally the person's bottom edge lies in the lower half of the bike's
            vertical span (up to 15% below it) and person height is 0.8-3x bike height;
            ranked by s_person * s_bike * horizontal-overlap fraction, so a geometrically
            better-aligned rider wins over a higher-scoring bystander.
    """
    if len(pb) == 0 or len(bb) == 0:
        return []
    px1, py1, px2, py2 = (pb[:, i : i + 1] for i in range(4))
    bx1, by1, bx2, by2 = (bb[None, :, i] for i in range(4))
    ox = np.clip(np.minimum(px2, bx2) - np.maximum(px1, bx1), 0, None)
    oy = np.minimum(py2, by2) - np.maximum(py1, by1)
    narrow = np.maximum(np.minimum(px2 - px1, bx2 - bx1), 1e-6)
    ok = (ox >= 0.5 * narrow) & (oy > 0) & (py1 < by1)
    rank = ps[:, None] * bs[None, :]
    if strict:
        bh = np.maximum(by2 - by1, 1e-6)
        ratio = (py2 - py1) / bh
        ok &= (py2 >= by1 + 0.5 * bh) & (py2 <= by2 + 0.15 * bh) & (ratio >= 0.8) & (ratio <= 3.0)
        rank = rank * (ox / narrow)
    pi, bi = np.nonzero(ok)
    if len(pi) == 0:
        return []
    order = np.argsort(-rank[pi, bi], kind="stable")
    used_p, used_b, pairs = set(), set(), []
    for k in order:
        p, b = pi[k], bi[k]
        if p not in used_p and b not in used_b:
            used_p.add(p)
            used_b.add(b)
            pairs.append((p, b))
    return pairs


def _ioa(box, others):
    """Intersection of each row of ``others`` with ``box``, over the row's own area."""
    x1 = np.maximum(others[:, 0], box[0]); y1 = np.maximum(others[:, 1], box[1])
    x2 = np.minimum(others[:, 2], box[2]); y2 = np.minimum(others[:, 3], box[3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    return inter / np.maximum((others[:, 2] - others[:, 0]) * (others[:, 3] - others[:, 1]), 1e-6)


def coco_to_kitti(boxes, scores, names, strategy="fused"):
    """Map one image's COCO-vocabulary detections to KITTI classes.

    ``names`` are the class-name strings of each detection. ``strategy`` is ``fused``
    (strict rider rule, default), ``fused_loose`` (sensitivity variant) or ``direct``.
    Returns (boxes, scores, cls) where cls are KITTI class names.
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
    elif strategy in ("fused", "fused_loose"):
        strict = strategy == "fused"
        ped_s = ps.copy()
        pairs = _rider_pairs(pb, ps, bb, bs, strict=strict)
        paired = {p for p, _ in pairs}
        for p, b in pairs:
            u = np.array([min(pb[p, 0], bb[b, 0]), min(pb[p, 1], bb[b, 1]),
                          max(pb[p, 2], bb[b, 2]), max(pb[p, 3], bb[b, 3])], dtype=np.float32)
            out_b.append(u[None]); out_s.append(np.array([ps[p] * bs[b]], dtype=np.float32))
            out_c.append("Cyclist")
            ped_s[p] = ps[p] * (1 - bs[b])
            if strict:  # duplicate / partial person boxes on the same rider
                dup = np.nonzero(_ioa(pb[p], pb) >= 0.5)[0]
                for q in dup:
                    if q not in paired:
                        ped_s[q] = min(ped_s[q], ps[q] * (1 - bs[b]))
        out_b.append(pb); out_s.append(ped_s); out_c += ["Pedestrian"] * len(ps)
    else:
        raise ValueError(strategy)

    if not out_c:
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.array([], dtype=object)
    return np.concatenate(out_b), np.concatenate(out_s), np.array(out_c, dtype=object)


def cap(boxes, scores, cls, k=100):
    """Keep the ``k`` highest-scoring mapped detections (the per-image evaluation budget)."""
    order = np.argsort(-scores, kind="stable")[:k]
    return boxes[order], scores[order], cls[order]


def prompts_to_kitti(boxes, scores, names):
    names = np.asarray(names)
    cls = np.array([PROMPT_TO_KITTI.get(str(n).lower()) for n in names], dtype=object)
    keep = np.array([c is not None for c in cls], dtype=bool)
    return boxes[keep], scores[keep], cls[keep]
