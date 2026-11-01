"""Object-scale statistics: COCO val2017 vs KITTI val for the shared classes.

    uv run python scripts/dataset_stats.py

Writes results/dataset_stats.json and figures/fig_scale_distribution.png.
"""

import json

import numpy as np

from det_eval import plotstyle
from det_eval.data import COCO_ANN, DATA, KITTI_CAT_ID, ROOT

FIG = ROOT / "figures"
PAIRS = [("person", "Pedestrian"), ("car", "Car"), ("bicycle", "Cyclist")]


def size_class(area):
    return np.select([area < 32**2, area < 96**2], ["small", "medium"], "large")


def main():
    coco = json.loads(COCO_ANN.read_text())
    kitti = json.loads((DATA / "kitti" / "coco_gt_val.json").read_text())
    cid = {c["name"]: c["id"] for c in coco["categories"]}
    stats, heights = {}, {}
    for cname, kname in PAIRS:
        ca = [a for a in coco["annotations"] if a["category_id"] == cid[cname] and not a["iscrowd"]]
        ka = [a for a in kitti["annotations"] if a["category_id"] == KITTI_CAT_ID[kname] and not a["iscrowd"]]
        for ds, anns in (("COCO", ca), ("KITTI", ka)):
            area = np.array([a["bbox"][2] * a["bbox"][3] for a in anns])
            h = np.array([a["bbox"][3] for a in anns])
            sc = size_class(area)
            stats[f"{ds}/{cname if ds == 'COCO' else kname}"] = dict(
                n=len(anns), median_height_px=float(np.median(h)),
                small=float((sc == "small").mean() * 100), medium=float((sc == "medium").mean() * 100),
                large=float((sc == "large").mean() * 100))
            heights[(ds, kname)] = h
    for ds, d in (("COCO", coco), ("KITTI", kitti)):
        imgs = d["images"]
        stats[f"{ds}/image_size"] = dict(median_w=float(np.median([i["width"] for i in imgs])),
                                         median_h=float(np.median([i["height"] for i in imgs])))
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "dataset_stats.json").write_text(json.dumps(stats, indent=1))
    for k, v in stats.items():
        print(k, {kk: round(vv, 1) for kk, vv in v.items()})

    plt = plotstyle.apply()
    FIG.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(7.8, 2.5), sharey=True)
    bins = np.logspace(np.log10(4), np.log10(640), 30)
    for ax, (cname, kname) in zip(axes, PAIRS):
        for ds, color in (("COCO", plotstyle.SERIES[0]), ("KITTI", plotstyle.SERIES[1])):
            h = heights[(ds, kname)]
            w = np.ones_like(h) / len(h) * 100
            ax.hist(h, bins=bins, weights=w, histtype="step", color=color, linewidth=2, label=ds)
        ax.set_xscale("log")
        ax.axvline(25, color=plotstyle.TEXT2, linewidth=1)
        ax.set_title(f"{kname}  (COCO: {cname})", loc="left", fontsize=8.5)

    axes[0].set_ylabel("% of objects")
    fig.supxlabel("ground-truth box height at native image resolution (px, log scale)", fontsize=8, color=plotstyle.TEXT2, y=-0.04)
    axes[0].text(27, 12.2, "25 px: KITTI\nmoderate min.", fontsize=6.5, color=plotstyle.TEXT2, va="top")
    handles, _ = axes[0].get_legend_handles_labels()
    fig.legend(handles, ["COCO val2017", "KITTI val"], loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.08))
    fig.savefig(FIG / "fig_scale_distribution.png")
    print("wrote", FIG / "fig_scale_distribution.png")


if __name__ == "__main__":
    main()
