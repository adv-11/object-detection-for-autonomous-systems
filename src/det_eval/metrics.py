"""COCO-style and KITTI-style evaluation on top of pycocotools.

Protocols
---------
coco        Standard COCOeval (IoU 0.50:0.95, 101-point AP, area ranges, maxDets 100).
kitti-coco  COCOeval on KITTI val, 3 classes, every labelled object counts (no
            difficulty filter); DontCare is a crowd region, neighbour classes are
            IoU-matched one-to-one ignores (see data.py).
kitti-<d>   KITTI devkit style for d in {easy, moderate, hard}: objects failing the
            difficulty criteria become IoU-matched ignores, detections shorter than the
            level's minimum height are discarded, AP uses class-specific IoU
            (Car 0.7, Pedestrian/Cyclist 0.5) and 40 recall points (AP|R40), the
            metric the official benchmark has reported since 2019.
            Dropping short detections gives the same AP as the devkit, which keeps
            them as ignored detections: a GT they would have matched still counts
            as a positive there and is then missed, so precision/recall agree.
"""

from __future__ import annotations

import contextlib
import copy
import io

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from det_eval.data import DIFFICULTY, KITTI_CAT_ID

STAT_NAMES = ["AP", "AP50", "AP75", "APs", "APm", "APl", "AR1", "AR10", "AR100", "ARs", "ARm", "ARl"]
KITTI_IOU = {"Car": 0.7, "Pedestrian": 0.5, "Cyclist": 0.5}
LEVEL_ORDER = ["easy", "moderate", "hard"]


class KittiEval(COCOeval):
    """COCOeval that also ignores ``kitti_ignore`` ground truth.

    pycocotools' ``_prepare`` overwrites ``gt['ignore']`` with ``iscrowd``; we re-apply the
    KITTI flag afterwards. Non-crowd ignored GT is matched by IoU, absorbs at most one
    detection, and that detection is ignored -- the devkit's neighbour / difficulty rule.
    """

    def _prepare(self):
        super()._prepare()
        for gts in self._gts.values():
            for g in gts:
                g["ignore"] = int(bool(g["iscrowd"]) or bool(g.get("kitti_ignore", 0)))


def coco_from_dict(d: dict) -> COCO:
    c = COCO()
    c.dataset = copy.deepcopy(d)
    with contextlib.redirect_stdout(io.StringIO()):
        c.createIndex()
    return c


def _load_res(gt: COCO, dets: list[dict]) -> COCO:
    if not dets:  # pycocotools cannot load an empty result list
        dets = [dict(image_id=gt.getImgIds()[0], category_id=gt.getCatIds()[0], bbox=[0, 0, 1, 1], score=1e-9)]
    with contextlib.redirect_stdout(io.StringIO()):
        return gt.loadRes(dets)


def _per_class_ap(E: COCOeval, cat_names: dict[int, str], iou: float | None = None, area: int = 0) -> dict:
    p = E.eval["precision"]  # [T, R, K, A, M]
    t = slice(None) if iou is None else int(np.argmin(np.abs(E.params.iouThrs - iou)))
    out = {}
    for k, cid in enumerate(E.params.catIds):
        pr = p[t, :, k, area, -1]
        pr = pr[pr > -1]
        out[cat_names[cid]] = float(pr.mean()) * 100 if pr.size else float("nan")
    return out


def coco_eval(gt_dict: dict, dets: list[dict], evaluator=COCOeval) -> dict:
    gt = coco_from_dict(gt_dict)
    E = evaluator(gt, _load_res(gt, dets), "bbox")
    with contextlib.redirect_stdout(io.StringIO()):
        E.evaluate()
        E.accumulate()
        E.summarize()
    names = {c["id"]: c["name"] for c in gt_dict["categories"]}
    return dict(
        summary={k: round(float(v) * 100, 2) for k, v in zip(STAT_NAMES, E.stats)},
        per_class_AP=_per_class_ap(E, names),
        per_class_AP50=_per_class_ap(E, names, iou=0.5),
        per_class_APs=_per_class_ap(E, names, area=1),
        per_class_APm=_per_class_ap(E, names, area=2),
        per_class_APl=_per_class_ap(E, names, area=3),
    )


def kitti_level_gt(gt_dict: dict, level: str) -> dict:
    """Ground truth where objects that do not satisfy ``level`` become ignore regions."""
    allowed = set(LEVEL_ORDER[: LEVEL_ORDER.index(level) + 1])
    g = copy.deepcopy(gt_dict)
    for a in g["annotations"]:
        if not a["kitti_ignore"] and a["kitti_difficulty"] not in allowed:
            a["kitti_ignore"] = 1
            a["ignore_reason"] = f"difficulty>{level}"
    return g


def kitti_level_eval(gt_dict: dict, dets: list[dict], level: str) -> dict:
    min_h = DIFFICULTY[level]["min_height"]
    gt = coco_from_dict(kitti_level_gt(gt_dict, level))
    dets = [d for d in dets if d["bbox"][3] >= min_h]
    E = KittiEval(gt, _load_res(gt, dets), "bbox")
    E.params.iouThrs = np.array([0.5, 0.7])
    E.params.recThrs = np.linspace(1 / 40, 1.0, 40)  # AP|R40
    E.params.areaRng = [[0, 1e10]]
    E.params.areaRngLbl = ["all"]
    E.params.maxDets = [100]
    with contextlib.redirect_stdout(io.StringIO()):
        E.evaluate()
        E.accumulate()
    names = {v: k for k, v in KITTI_CAT_ID.items()}
    out = {}
    for name, cid in KITTI_CAT_ID.items():
        out[name] = _per_class_ap(E, names, iou=KITTI_IOU[name])[name]
    out["mAP"] = float(np.nanmean(list(out.values())))
    return {k: round(v, 2) for k, v in out.items()}


def kitti_eval(gt_dict: dict, dets: list[dict]) -> dict:
    res = dict(coco_style=coco_eval(gt_dict, dets, KittiEval))
    res["kitti_R40"] = {lvl: kitti_level_eval(gt_dict, dets, lvl) for lvl in LEVEL_ORDER}
    return res
