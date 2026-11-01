"""Qualitative KITTI examples: ground truth vs. detections of several runs (score >= 0.3).

    uv run python scripts/visualize.py --images 000123 000456 --runs kitti__yolo11m_1280 kitti__yolo11m_1280_ft

Ground truth: solid boxes (Car blue, Pedestrian orange, Cyclist aqua); ignore regions
(DontCare / Van / Person_sitting) grey. Detections are drawn per run in its own panel.
"""

from __future__ import annotations

import argparse
import gzip
import json

import numpy as np
from PIL import Image

from det_eval import plotstyle
from det_eval.data import ROOT, kitti_image_path, parse_kitti_label
from det_eval.mapping import coco_to_kitti, prompts_to_kitti

COLORS = {"Car": plotstyle.SERIES[0], "Pedestrian": plotstyle.SERIES[1], "Cyclist": plotstyle.SERIES[2]}
IGNORE = {"DontCare", "Van", "Person_sitting"}


def run_dets(run, img_id):
    with gzip.open(ROOT / "results" / "raw" / f"{run}.json.gz", "rt") as f:
        raw = json.load(f)
    vocab = raw["meta"]["vocabulary"]
    r = np.array([d for d in raw["detections"] if d[0] == img_id], dtype=np.float64).reshape(-1, 7)
    names = [vocab[int(l)] for l in r[:, 6]]
    b, s = r[:, 1:5].astype(np.float32), r[:, 5].astype(np.float32)
    out = coco_to_kitti(b, s, names, "fused") if "person" in vocab else prompts_to_kitti(b, s, names)
    return raw["meta"]["detector"]["name"], out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--out", default="figures/fig_qualitative.png")
    ap.add_argument("--thresh", type=float, default=0.3)
    args = ap.parse_args()
    plt = plotstyle.apply()
    nr, nc = len(args.images), 1 + len(args.runs)
    fig, axes = plt.subplots(nr, nc, figsize=(3.3 * nc, 1.15 * nr + 0.3), squeeze=False)
    for i, stem in enumerate(args.images):
        img = Image.open(kitti_image_path(stem))
        panels = [("Ground truth", None)] + [(None, r) for r in args.runs]
        for j, (title, run) in enumerate(panels):
            ax = axes[i, j]
            ax.imshow(img)
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for sp in ax.spines.values():
                sp.set_visible(False)
            if run is None:
                for o in parse_kitti_label(stem):
                    if o.cls in COLORS or o.cls in IGNORE:
                        c = COLORS.get(o.cls, "#9a9a95")
                        x1, y1, x2, y2 = o.box
                        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color=c, linewidth=1.1,
                                                   linestyle="-" if o.cls in COLORS else ":"))
            else:
                name, (b, s, c) = run_dets(run, int(stem))
                title = args.labels[j - 1] if args.labels else name
                for bb, ss, cc in zip(b, s, c):
                    if ss >= args.thresh:
                        x1, y1, x2, y2 = bb
                        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color=COLORS[cc], linewidth=1.1))
            if i == 0:
                ax.set_title(title, fontsize=8, loc="left")
        axes[i, 0].set_ylabel(stem, fontsize=7)
    fig.subplots_adjust(wspace=0.02, hspace=0.08)
    fig.savefig(ROOT / args.out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
