"""Report figures from results/ (metrics, speed, failures, fine-tuning logs).

    uv run python scripts/make_figures.py
"""

from __future__ import annotations

import json

import matplotlib as mpl
import numpy as np
import pandas as pd

from det_eval import plotstyle
from det_eval.data import ROOT

MET, FIG = ROOT / "results" / "metrics", ROOT / "figures"
S = plotstyle.SERIES
plt = plotstyle.apply()


def metrics(run):
    p = MET / f"{run}.json"
    return json.loads(p.read_text()) if p.exists() else None


def short(name):
    return name.replace(" R50-FPN v2", "").replace(" R50-FPN", "").replace(" v2", "")


# ----------------------------------------------------------------------------- 1. speed vs accuracy
def fig_speed_accuracy():
    """COCO AP vs. median batch-1 latency from the dedicated benchmark (FP32 and FP16 panels)."""
    sp = pd.DataFrame(json.loads((ROOT / "results" / "speed" / "speed.json").read_text())["rows"])
    sp = sp[sp.domain == "coco"].dropna(subset=["ms_p50"])
    coco = pd.read_csv(ROOT / "results" / "tables" / "coco.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 2.9), sharey=True)
    ticks = [20, 30, 50, 100, 200, 500]
    for ax, prec in zip(axes, ("fp32", "fp16")):
        s_ = sp[sp.precision == prec].set_index("model")
        for _, r in coco.iterrows():
            if r.model not in s_.index:
                continue
            x = s_.loc[r.model, "ms_p50"]
            g = plotstyle.group_of(r.family)
            ax.scatter(x, r.AP, s=46, color=plotstyle.GROUP_COLOR[g], edgecolor="white", linewidth=1.5, zorder=3, label=g)
            ax.annotate(short(r.model).replace("@640", ""), (x, r.AP), xytext=(5, -3), textcoords="offset points",
                        fontsize=6.6, color=plotstyle.TEXT)
        ax.set_xscale("log")
        ax.set_xticks(ticks, [str(t) for t in ticks])
        ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        ax.set_xlim(18, 800)
        ax.set_xlabel(f"median latency, batch 1, {prec.upper()} (ms, log)")
        ax.set_title(f"{prec.upper()}", loc="left")
    axes[0].set_ylabel("COCO val2017 AP@[.50:.95]")
    h, l = axes[0].get_legend_handles_labels()
    uniq = dict(zip(l, h))
    fig.legend(uniq.values(), uniq.keys(), loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.07), fontsize=7.5)
    fig.savefig(FIG / "fig_speed_accuracy.png")


# ----------------------------------------------------------------------------- 2. per-class domain shift
KITTI_PAIRS = [("person", "Pedestrian"), ("car", "Car"), ("bicycle", "Cyclist")]
SHIFT_RUNS = [  # (coco run, kitti run, label)
    ("coco__faster_rcnn_r50_v2", "kitti__faster_rcnn_r50_v2", "Faster R-CNN"),
    ("coco__retinanet_r50_v2", "kitti__retinanet_r50_v2", "RetinaNet"),
    ("coco__fcos_r50", "kitti__fcos_r50", "FCOS"),
    ("coco__yolo11s", "kitti__yolo11s_640", "YOLO11s"),
    ("coco__yolo11m", "kitti__yolo11m_640", "YOLO11m"),
    ("coco__rtdetr_l", "kitti__rtdetr_l_640", "RT-DETR-L"),
    ("coco__yolo_world_l_vcoco", "kitti__yolo_world_l_1280_vcoco", "YOLO-World-L (KITTI @1280)"),
    ("coco__grounding_dino_t_vcoco", "kitti__grounding_dino_t_vcoco", "Grounding DINO-T"),
]


def fig_domain_shift():
    rows = []
    for cr, kr, lab in SHIFT_RUNS:
        c, k = metrics(cr), metrics(kr)
        if not (c and k):
            continue
        for cname, kname in KITTI_PAIRS:
            rows.append(dict(model=lab, cls=kname, coco=c["metrics"]["coco"]["per_class_AP"][cname],
                             kitti=k["metrics"]["fused"]["coco_style"]["per_class_AP"][kname]))
    if not rows:
        return
    df = pd.DataFrame(rows)
    models = list(dict.fromkeys(df.model))
    fig, axes = plt.subplots(1, 3, figsize=(7.8, 2.9), sharey=True)
    y = np.arange(len(models))[::-1]
    for ax, (cname, kname) in zip(axes, KITTI_PAIRS):
        d = df[df.cls == kname].set_index("model").loc[models]
        ax.hlines(y, d.coco, d.kitti, color=plotstyle.GRID, linewidth=2.5, zorder=1)
        ax.scatter(d.coco, y, s=40, color=S[0], edgecolor="white", linewidth=1.5, zorder=3, label="COCO val2017")
        ax.scatter(d.kitti, y, s=40, color=S[1], edgecolor="white", linewidth=1.5, zorder=3, label="KITTI val (zero-shot)")
        ax.set_title(f"{kname}  (COCO: {cname})", loc="left", fontsize=8.5)
        ax.set_xlim(0, 80)
        ax.set_xlabel("AP@[.50:.95]")
    axes[0].set_yticks(y, models)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.06))
    fig.savefig(FIG / "fig_domain_shift_per_class.png")


# ----------------------------------------------------------------------------- 3. input resolution
def fig_resolution():
    rows = []
    for key, lab in (("yolo11s", "YOLO11s"), ("yolo11m", "YOLO11m"), ("rtdetr_l", "RT-DETR-L")):
        for sz in (640, 1280):
            m = metrics(f"kitti__{key}_{sz}")
            if m:
                f = m["metrics"]["fused"]
                rows.append(dict(model=lab, size=sz, AP=f["coco_style"]["summary"]["AP"],
                                 APs=f["coco_style"]["summary"]["APs"],
                                 mod=f["kitti_R40"]["moderate"]["mAP"], hard=f["kitti_R40"]["hard"]["mAP"]))
    if not rows:
        return
    df = pd.DataFrame(rows)
    both = df.groupby("model")["size"].nunique()
    df = df[df.model.isin(both[both == 2].index)]  # only models run at both input sizes
    if df.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(7.8, 2.5))
    models = list(dict.fromkeys(df.model))
    x = np.arange(len(models))
    for ax, col, title in zip(axes, ("AP", "APs", "mod"),
                              ("AP@[.50:.95] (COCO-style)", "AP small (<32² px)", "KITTI moderate mAP (AP|R40)")):
        for i, (sz, color) in enumerate(((640, S[0]), (1280, S[1]))):
            v = df[df["size"] == sz].set_index("model").loc[models, col]
            bars = ax.bar(x + (i - 0.5) * 0.36, v, width=0.34, color=color, label=f"input {sz} px")
            for b, val in zip(bars, v):
                ax.text(b.get_x() + b.get_width() / 2, val + 0.6, f"{val:.1f}", ha="center", fontsize=6.5,
                        color=plotstyle.TEXT)
        ax.set_xticks(x, models, fontsize=7.5)
        ax.set_title(title, loc="left", fontsize=8.5)
        ax.grid(axis="x", visible=False)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.08))
    fig.savefig(FIG / "fig_resolution_ablation.png")


# ----------------------------------------------------------------------------- 4. recall vs height
def fig_recall_height(runs_labels):
    fdir = ROOT / "results" / "failures"
    data = [(json.loads((fdir / f"{r}.json").read_text()), lab) for r, lab in runs_labels if (fdir / f"{r}.json").exists()]
    if not data:
        return
    fig, axes = plt.subplots(1, 3, figsize=(7.8, 2.5), sharey=True)
    for ax, cls in zip(axes, ("Car", "Pedestrian", "Cyclist")):
        for i, (d, lab) in enumerate(data):
            bins = d["recall_by_height"][cls]
            xs = list(bins)
            ax.plot(xs, [bins[k]["recall_max"] for k in xs], marker="o", markersize=4.5, color=S[i],
                    markeredgecolor="white", label=lab)
        ns = [d["recall_by_height"][cls][k]["n"] for k in xs]
        ax.set_xticks(range(len(xs)), [f"{k}\n(n={n})" for k, n in zip(xs, ns)], fontsize=6.5)
        iou = 0.7 if cls == "Car" else 0.5
        ax.set_title(f"{cls}  (IoU ≥ {iou})", loc="left", fontsize=8.5)
        ax.set_ylim(0, 102)
    axes[0].set_ylabel("max recall (%)")
    fig.supxlabel("ground-truth box height (px)", fontsize=8, color=plotstyle.TEXT2, y=-0.06)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=min(4, len(l)), bbox_to_anchor=(0.5, 1.08))
    fig.savefig(FIG / "fig_recall_vs_height.png")


# ----------------------------------------------------------------------------- 5. FP taxonomy
def fig_fp_taxonomy(runs_labels):
    fdir = ROOT / "results" / "failures"
    kinds = ["localization", "confusion", "duplicate", "background"]
    rows = []
    for r, lab in runs_labels:
        p = fdir / f"{r}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        for cls, fp in d["fp_taxonomy"].items():
            agg = {k: 0 for k in kinds}
            for k, v in fp.items():
                if k.startswith("confusion"):
                    agg["confusion"] += v
                elif k in agg:
                    agg[k] += v
            rows.append(dict(run=lab, cls=cls, **agg))
    if not rows:
        return
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 3, figsize=(7.8, 2.3), sharey=True)
    labs = list(dict.fromkeys(df.run))
    y = np.arange(len(labs))[::-1]
    for ax, cls in zip(axes, ("Car", "Pedestrian", "Cyclist")):
        d = df[df.cls == cls].set_index("run").loc[labs]
        left = np.zeros(len(labs))
        for i, k in enumerate(kinds):
            ax.barh(y, d[k], left=left, height=0.55, color=S[i], edgecolor="white", linewidth=1, label=k)
            left += d[k].values
        ax.set_title(f"{cls}", loc="left", fontsize=8.5)
        ax.set_xlabel("false positives (score ≥ 0.3)", fontsize=7.5)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, labs, fontsize=7.5)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.1))
    fig.savefig(FIG / "fig_fp_taxonomy.png")


# ----------------------------------------------------------------------------- 6. fine-tuning
def fig_finetune(zs_run, ft_runs):
    """Zero-shot vs fine-tuned (several label budgets), KITTI moderate AP|R40 per class."""
    series = [("COCO-pretrained, zero-shot", metrics(zs_run), "fused")]
    series += [(f"fine-tuned, {lab}", metrics(r), "prompt") for r, lab in ft_runs]
    series = [(lab, m, k) for lab, m, k in series if m]
    if len(series) < 2:
        return
    cats = ["Car", "Pedestrian", "Cyclist", "mAP"]
    x = np.arange(len(cats))
    w = 0.8 / len(series)
    fig, ax = plt.subplots(figsize=(5.6, 2.6))
    for i, (lab, m, key) in enumerate(series):
        vals = [m["metrics"][key]["kitti_R40"]["moderate"][c] for c in cats]
        bars = ax.bar(x - 0.4 + w * (i + 0.5), vals, width=w * 0.9, color=S[i], label=lab)
        for b_, v in zip(bars, vals):
            ax.text(b_.get_x() + b_.get_width() / 2, v + 1, f"{v:.0f}", ha="center", fontsize=6.5, color=plotstyle.TEXT)
    ax.set_xticks(x, cats)
    ax.set_ylabel("KITTI val, moderate AP|R40")
    ax.set_ylim(0, 108)
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=7, loc="upper center", ncol=len(series), bbox_to_anchor=(0.5, 1.16))
    fig.savefig(FIG / "fig_finetune.png")


def main():
    FIG.mkdir(exist_ok=True)
    fig_speed_accuracy()
    fig_domain_shift()
    fig_resolution()
    analysed = [("kitti__faster_rcnn_r50_v2", "Faster R-CNN (zero-shot)"),
                ("kitti__grounding_dino_t_vcoco", "Grounding DINO-T (zero-shot)"),
                ("kitti__yolo11s_1280", "YOLO11s@1280 (zero-shot)"),
                ("kitti__yolo11s_1280_ft", "YOLO11s@1280 (fine-tuned)")]
    fig_recall_height(analysed)
    fig_fp_taxonomy(analysed)
    fig_finetune("kitti__yolo11s_1280", [("kitti__yolo11s_1280_ft25", "25% of train"),
                                         ("kitti__yolo11s_1280_ft", "100% of train")])
    print("figures:", sorted(p.name for p in FIG.glob("*.png")))


if __name__ == "__main__":
    main()
