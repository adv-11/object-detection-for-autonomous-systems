"""Evaluate raw detections (results/raw/*.json.gz) and write metrics JSON.

    uv run python scripts/evaluate.py                 # every raw run not yet evaluated
    uv run python scripts/evaluate.py --force coco__yolo11m

COCO runs: standard COCOeval on val2017.
KITTI runs: COCO-vocabulary detectors are mapped with the ``fused`` (default),
``fused_loose`` and ``direct`` Cyclist strategies (see det_eval/mapping.py); runs with a
KITTI vocabulary (open-vocab prompts, fine-tuned models) are mapped by class name. Each
mapping is evaluated with the coco-style and KITTI AP|R40 (easy / moderate / hard) protocols.
In every case the per-image budget of 100 detections is applied after mapping.
"""

from __future__ import annotations

import argparse
import gzip
import json

import numpy as np

from det_eval.data import COCO_ANN, DATA, KITTI_CAT_ID, ROOT
from det_eval.detectors import MAX_DET
from det_eval.mapping import cap, coco_name_to_id, coco_to_kitti, prompts_to_kitti
from det_eval.metrics import coco_eval, kitti_eval

RAW = ROOT / "results" / "raw"
OUT = ROOT / "results" / "metrics"


def group_by_image(rows):
    rows = np.asarray(rows, dtype=np.float64)
    if rows.size == 0:
        return {}
    order = np.argsort(rows[:, 0], kind="stable")
    rows = rows[order]
    ids, starts = np.unique(rows[:, 0], return_index=True)
    return {int(i): rows[s:e] for i, s, e in zip(ids, starts, list(starts[1:]) + [len(rows)])}


def restrict(gt, image_ids):
    """Limit ground truth to the images that were actually processed (debug --limit runs)."""
    ids = set(image_ids)
    if len(ids) == len(gt["images"]):
        return gt
    gt["images"] = [i for i in gt["images"] if i["id"] in ids]
    gt["annotations"] = [a for a in gt["annotations"] if a["image_id"] in ids]
    return gt


def to_coco_dets(img_id, boxes, scores, cat_ids):
    return [
        dict(image_id=img_id, category_id=int(c), bbox=[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])], score=float(s))
        for b, s, c in zip(boxes, scores, cat_ids)
    ]


def eval_coco(raw):
    gt = restrict(json.loads(COCO_ANN.read_text()), raw["meta"]["image_ids"])
    name2id = coco_name_to_id(gt["categories"])
    vocab = raw["meta"]["vocabulary"]
    dets = []
    for img_id, r in group_by_image(raw["detections"]).items():
        names = np.array([vocab[int(l)] for l in r[:, 6]], dtype=object)
        keep = np.array([n in name2id for n in names], dtype=bool)  # drops torchvision 'N/A' slots
        b, s, c = cap(r[keep, 1:5], r[keep, 5], names[keep], MAX_DET)
        dets += to_coco_dets(img_id, b, s, [name2id[n] for n in c])
    return {"coco": coco_eval(gt, dets)}


def eval_kitti(raw):
    gt = restrict(json.loads((DATA / "kitti" / "coco_gt_val.json").read_text()), raw["meta"]["image_ids"])
    vocab = raw["meta"]["vocabulary"]
    by_img = group_by_image(raw["detections"])
    strategies = ["fused", "fused_loose", "direct"] if "person" in vocab else ["prompt"]
    out = {}
    for strat in strategies:
        dets = []
        for img_id, r in by_img.items():
            names = [vocab[int(l)] for l in r[:, 6]]
            boxes, scores = r[:, 1:5].astype(np.float32), r[:, 5].astype(np.float32)
            if strat == "prompt":
                b, s, c = prompts_to_kitti(boxes, scores, names)
            else:
                b, s, c = coco_to_kitti(boxes, scores, names, strat)
            b, s, c = cap(b, s, c, MAX_DET)
            dets += to_coco_dets(img_id, b, s, [KITTI_CAT_ID[x] for x in c])
        out[strat] = kitti_eval(gt, dets)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*", help="run names (default: all raw runs)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    paths = sorted(RAW.glob("*.json.gz"))
    if args.runs:
        paths = [p for p in paths if p.name.removesuffix(".json.gz") in args.runs]
    for p in paths:
        run = p.name.removesuffix(".json.gz")
        dst = OUT / f"{run}.json"
        if dst.exists() and not args.force and dst.stat().st_mtime > p.stat().st_mtime:
            continue
        with gzip.open(p, "rt") as f:
            raw = json.load(f)
        metrics = eval_coco(raw) if raw["meta"]["dataset"] == "coco" else eval_kitti(raw)
        dst.write_text(json.dumps(dict(meta=raw["meta"], metrics=metrics), indent=1))
        if "coco" in metrics:
            s = metrics["coco"]["summary"]
            print(f"{run:45s} AP {s['AP']:5.1f}  AP50 {s['AP50']:5.1f}  APs {s['APs']:5.1f}")
        else:
            for strat, m in metrics.items():
                s, k = m["coco_style"]["summary"], m["kitti_R40"]["moderate"]
                print(f"{run:45s} [{strat:11s}] AP {s['AP']:5.1f}  AP50 {s['AP50']:5.1f}  "
                      f"KITTI-mod Car {k['Car']:5.1f} Ped {k['Pedestrian']:5.1f} Cyc {k['Cyclist']:5.1f}")


if __name__ == "__main__":
    main()
