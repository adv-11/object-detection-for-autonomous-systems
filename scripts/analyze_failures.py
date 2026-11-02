"""Failure analysis on KITTI val for selected runs.

    uv run python scripts/analyze_failures.py kitti__yolo11m_1280 kitti__yolo11m_1280_ft ...

For each run (mapped to KITTI classes with the run's default mapping):

1. Recall vs. object height / occlusion / truncation, per class, at the class IoU
   (Car 0.7, Ped/Cyc 0.5) and also at IoU 0.5 for Car, using (a) every detection
   (max recall) and (b) detections at the operating threshold of 0.3.
2. False-positive taxonomy at the operating threshold (TIDE-style, simplified), using
   the *full* KITTI label set (incl. Van/Truck/Tram/Misc/DontCare):
     ignored    DontCare (IoA >= class IoU) or neighbour class (IoU >= class IoU); not penalised
     duplicate  matches a GT already claimed by a higher-scoring detection
     localization  same-class GT with 0.1 <= IoU < class threshold
     confusion  IoU >= 0.5 with a GT of another KITTI class (e.g. Car on Truck)
     background everything else
3. IoU distribution of Car detections matched greedily at IoU >= 0.5 (how many true
   detections the 0.7 Car threshold rejects).

Writes results/failures/<run>.json and qualitative examples to results/failures/img/.
"""

from __future__ import annotations

import argparse
import gzip
import json

import numpy as np

from det_eval.data import KITTI_CAT_ID, ROOT, parse_kitti_label, read_split
from det_eval.detectors import MAX_DET
from det_eval.mapping import cap, coco_to_kitti, prompts_to_kitti
from det_eval.metrics import KITTI_IOU

RAW = ROOT / "results" / "raw"
OUT = ROOT / "results" / "failures"
OP_THRESH = 0.3
HEIGHT_BINS = [0, 25, 40, 60, 100, 1e9]
HEIGHT_LABELS = ["<25", "25-40", "40-60", "60-100", ">100"]
NEIGHBOUR = {"Car": {"Van"}, "Pedestrian": {"Person_sitting"}, "Cyclist": set()}


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])  # noqa: E731
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def ioa_det(d, g):
    """Intersection over detection area (how pycocotools matches ignore regions)."""
    if len(d) == 0 or len(g) == 0:
        return np.zeros((len(d), len(g)))
    x1 = np.maximum(d[:, None, 0], g[None, :, 0]); y1 = np.maximum(d[:, None, 1], g[None, :, 1])
    x2 = np.minimum(d[:, None, 2], g[None, :, 2]); y2 = np.minimum(d[:, None, 3], g[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    return inter / ((d[:, 2] - d[:, 0]) * (d[:, 3] - d[:, 1]) + 1e-9)[:, None]


def load_mapped(run):
    with gzip.open(RAW / f"{run}.json.gz", "rt") as f:
        raw = json.load(f)
    vocab = raw["meta"]["vocabulary"]
    rows = np.asarray(raw["detections"], dtype=np.float64)
    by_img = {}
    for img_id in raw["meta"]["image_ids"]:
        by_img[img_id] = (np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.array([], dtype=object))
    if rows.size:
        for img_id in np.unique(rows[:, 0]).astype(int):
            r = rows[rows[:, 0] == img_id]
            names = [vocab[int(l)] for l in r[:, 6]]
            b, s = r[:, 1:5].astype(np.float32), r[:, 5].astype(np.float32)
            mapped = coco_to_kitti(b, s, names, "fused") if "person" in vocab else prompts_to_kitti(b, s, names)
            by_img[img_id] = cap(*mapped, MAX_DET)
    return raw["meta"], by_img


def analyze(run):
    meta, dets = load_mapped(run)
    recall_rows, fp_counts, car_ious = [], {c: {} for c in KITTI_CAT_ID}, []
    examples = []
    for stem in read_split("val"):
        img_id = int(stem)
        objs = parse_kitti_label(stem)
        gt_box = np.array([o.box for o in objs], dtype=np.float32).reshape(-1, 4)
        gt_cls = np.array([o.cls for o in objs], dtype=object)
        db, ds, dc = dets[img_id]
        img_fp, img_missed = 0, 0
        for cls in KITTI_CAT_ID:
            thr = KITTI_IOU[cls]
            gi = np.nonzero(gt_cls == cls)[0]
            sel = np.nonzero(dc == cls)[0]
            sel = sel[np.argsort(-ds[sel], kind="stable")]
            iou = iou_matrix(db[sel], gt_box)
            # --- recall per GT object (best IoU over all / operating-point detections)
            for g in gi:
                o = objs[g]
                col = iou[:, g] if len(sel) else np.zeros(0)
                op = ds[sel] >= OP_THRESH
                best_all = float(col.max()) if col.size else 0.0
                best_op = float(col[op].max()) if op.any() else 0.0
                recall_rows.append(dict(cls=cls, height=o.height, occlusion=o.occlusion, truncation=o.truncation,
                                        best_iou_all=best_all, best_iou_op=best_op))
            # --- greedy matching at operating threshold, FP taxonomy
            op_idx = [k for k in range(len(sel)) if ds[sel[k]] >= OP_THRESH]
            claimed = set()
            dontcare = np.nonzero(gt_cls == "DontCare")[0]
            neigh = np.nonzero(np.isin(gt_cls, list(NEIGHBOUR[cls])))[0]
            other = np.nonzero(np.isin(gt_cls, [c for c in ("Car", "Van", "Truck", "Pedestrian", "Person_sitting", "Cyclist", "Tram", "Misc") if c != cls]))[0]
            for k in op_idx:
                row = iou[k]
                cand = [g for g in gi if row[g] >= thr]
                free = [g for g in cand if g not in claimed]
                if free:
                    g = max(free, key=lambda g: row[g])
                    claimed.add(g)
                    continue
                # same order and overlap rules as the metric: DontCare by IoA, neighbours by IoU
                if (len(dontcare) and ioa_det(db[sel[k]][None], gt_box[dontcare]).max() >= thr) or                         (len(neigh) and row[neigh].max() >= thr):
                    kind = "ignored"
                elif cand:
                    kind = "duplicate"
                elif len(gi) and row[gi].max() >= 0.1:
                    kind = "localization"
                elif len(other) and row[other].max() >= 0.5:
                    kind = f"confusion:{gt_cls[other[np.argmax(row[other])]]}"
                else:
                    kind = "background"
                fp_counts[cls][kind] = fp_counts[cls].get(kind, 0) + 1
                img_fp += kind != "ignored"
            img_missed += sum(1 for g in gi if g not in claimed)
            if cls == "Car":  # separate greedy pass at IoU 0.5 for the localisation-quality histogram
                claimed50 = set()
                for k in op_idx:
                    free50 = [g for g in gi if iou[k, g] >= 0.5 and g not in claimed50]
                    if free50:
                        g = max(free50, key=lambda g: iou[k, g])
                        claimed50.add(g)
                        car_ious.append(float(iou[k, g]))
        examples.append((img_id, img_fp, img_missed))

    rec = recall_rows
    def recall(rows, key, thr_fn):
        return float(np.mean([r[key] >= thr_fn(r) for r in rows])) * 100 if rows else float("nan")

    by_height = {}
    for cls in KITTI_CAT_ID:
        by_height[cls] = {}
        for lo, hi, lab in zip(HEIGHT_BINS[:-1], HEIGHT_BINS[1:], HEIGHT_LABELS):
            rows = [r for r in rec if r["cls"] == cls and lo <= r["height"] < hi]
            by_height[cls][lab] = dict(
                n=len(rows),
                recall_max=recall(rows, "best_iou_all", lambda r: KITTI_IOU[r["cls"]]),
                recall_op=recall(rows, "best_iou_op", lambda r: KITTI_IOU[r["cls"]]),
                recall_max_iou50=recall(rows, "best_iou_all", lambda r: 0.5),
            )
    by_occ = {cls: {str(o): dict(n=len(rows := [r for r in rec if r["cls"] == cls and r["occlusion"] == o]),
                                 recall_max=recall(rows, "best_iou_all", lambda r: KITTI_IOU[r["cls"]]))
                    for o in range(4)} for cls in KITTI_CAT_ID}
    by_trunc = {cls: {lab: dict(n=len(rows := [r for r in rec if r["cls"] == cls and lo <= r["truncation"] < hi]),
                                recall_max=recall(rows, "best_iou_all", lambda r: KITTI_IOU[r["cls"]]))
                      for lo, hi, lab in ((0, 0.01, "0"), (0.01, 0.3, "0-0.3"), (0.3, 1.01, ">0.3"))}
                for cls in KITTI_CAT_ID}
    res = dict(run=run, model=meta["detector"]["name"], op_thresh=OP_THRESH,
               recall_by_height=by_height, recall_by_occlusion=by_occ, recall_by_truncation=by_trunc,
               fp_taxonomy=fp_counts,
               car_matched_iou=dict(n=len(car_ious), mean=float(np.mean(car_ious)) if car_ious else None,
                                    frac_below_0_7=float(np.mean(np.array(car_ious) < 0.7)) if car_ious else None,
                                    hist=np.histogram(car_ious, bins=np.linspace(0.5, 1, 11))[0].tolist()),
               worst_images=sorted(examples, key=lambda e: -(e[1] + e[2]))[:20])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for run in args.runs:
        res = analyze(run)
        (OUT / f"{run}.json").write_text(json.dumps(res, indent=1, default=int))
        print(run)
        for cls, bins in res["recall_by_height"].items():
            print(f"  {cls:10s}", "  ".join(f"{k}:{v['recall_max']:.0f}%(n={v['n']})" for k, v in bins.items()))
        for cls, fp in res["fp_taxonomy"].items():
            print(f"  FP {cls:10s}", dict(sorted(fp.items(), key=lambda kv: -kv[1])))
        print("  car matched IoU:", {k: v for k, v in res["car_matched_iou"].items() if k != "hist"})


if __name__ == "__main__":
    main()
