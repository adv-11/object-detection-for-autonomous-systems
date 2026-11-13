"""Extension-study tables and figures: D-FINE COCO vs Objects365 pre-training, YOLO26 vs YOLO11, YOLO-World-S, and
label-budget fine-tuning with 3 seeds.

    uv run python scripts/summarize_extension.py
Writes results/tables/ext_*.md, results/tables/ext_summary.json and figures/ext_*.png.
"""

from __future__ import annotations

import json

import numpy as np

from det_eval import plotstyle
from det_eval.data import ROOT

MET, FIG, TAB = ROOT / "results" / "metrics", ROOT / "figures", ROOT / "results" / "tables"
SPEED = ROOT / "results" / "speed" / "speed_ext.json"
S = plotstyle.SERIES
plt = plotstyle.apply()
CLS = ["Car", "Pedestrian", "Cyclist"]


def load(run):
    return json.loads((MET / f"{run}.json").read_text(encoding="utf-8"))


def kitti(run, strategy=None):
    m = load(run)["metrics"]
    k = strategy or ("prompt" if "prompt" in m else "fused")
    r40 = m[k]["kitti_R40"]
    return {"mod": r40["moderate"]["mAP"], **{c: r40["moderate"][c] for c in CLS},
            "easy": r40["easy"]["mAP"], "hard": r40["hard"]["mAP"], "ap": m[k]["coco_style"]["summary"]["AP"]}


def coco(run):
    s = load(run)["metrics"]["coco"]["summary"]
    return {k: s[k] for k in ("AP", "AP50", "AP75", "APs", "APm", "APl", "AR100")}


def wr(name, text):
    (TAB / f"{name}.md").write_text(text, encoding="utf-8")


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out) + "\n"


f1 = lambda x: f"{x:.1f}"


def main():
    TAB.mkdir(exist_ok=True, parents=True)
    summary = {}

    # ---------------------------------------------------------------- A1. zero-shot: COCO and KITTI @640/1280
    models = [("D-FINE-S (COCO)", "dfine_s_coco"), ("D-FINE-S (O365→COCO)", "dfine_s_o365"),
              ("D-FINE-M (COCO)", "dfine_m_coco"), ("D-FINE-M (O365→COCO)", "dfine_m_o365"),
              ("YOLO11s", "yolo11s"), ("YOLO26s", "yolo26s"), ("YOLO11m", "yolo11m"), ("YOLO26m", "yolo26m")]
    rows, zs = [], {}
    for label, key in models:
        c = coco(f"coco__{key}")
        k6, k12 = kitti(f"kitti__{key}_640"), kitti(f"kitti__{key}_1280")
        zs[key] = {"coco": c, "kitti640": k6, "kitti1280": k12}
        rows.append([label, f1(c["AP"]), f1(c["APs"]), f1(k6["mod"]), f1(k12["mod"]),
                     f1(k12["Car"]), f1(k12["Pedestrian"]), f1(k12["Cyclist"]), f1(k12["hard"])])
    wr("ext_zeroshot", table(
        ["Model", "COCO AP", "COCO APs", "KITTI Mod mAP @640", "KITTI Mod mAP @1280", "Car", "Ped", "Cyc", "Hard mAP @1280"], rows))
    summary["zeroshot"] = zs

    # YOLO-World-S zero-shot, vocab comparison (with L for reference)
    yw = {}
    rows = []
    for label, run_c, run_k, cc in [("YOLO-World-S v2", "kitti__yolo_world_s_1280_vcoco", "kitti__yolo_world_s_1280_vkitti", "coco__yolo_world_s_vcoco"),
                                    ("YOLO-World-L v2", "kitti__yolo_world_l_1280_vcoco", "kitti__yolo_world_l_1280_vkitti", "coco__yolo_world_l_vcoco")]:
        a, b = kitti(run_c, "fused"), kitti(run_k)
        yw[label] = {"coco_AP": coco(cc)["AP"], "coco_vocab": a, "kitti_vocab": b}
        rows.append([label, f1(coco(cc)["AP"]), f1(a["mod"]), f1(b["mod"]), f1(a["mod"] - b["mod"]),
                     f1(a["Cyclist"]), f1(b["Cyclist"])])
    wr("ext_yoloworld_vocab", table(
        ["Model", "COCO AP", "KITTI Mod (COCO names + mapping)", "KITTI Mod (KITTI names)", "Gap", "Cyc (COCO names)", "Cyc (KITTI names)"], rows))
    summary["yoloworld_vocab"] = yw

    # ---------------------------------------------------------------- B. label budgets x seeds
    runs = {
        ("YOLO11s", 0.05): [f"kitti__yolo11s_1280_ft5_s{s}" for s in range(3)],
        ("YOLO11s", 0.25): ["kitti__yolo11s_1280_ft25", "kitti__yolo11s_1280_ft25_s1", "kitti__yolo11s_1280_ft25_s2"],
        ("YOLO11s", 1.0): ["kitti__yolo11s_1280_ft"],
        ("YOLO-World-S", 0.05): [f"kitti__yolo_world_s_1280_ft5_s{s}" for s in range(3)],
        ("YOLO-World-S", 0.25): [f"kitti__yolo_world_s_1280_ft25_s{s}" for s in range(3)],
        ("YOLO-World-S", 1.0): ["kitti__yolo_world_s_1280_ft"],
    }
    zero = {"YOLO11s": kitti("kitti__yolo11s_1280"), "YOLO-World-S": kitti("kitti__yolo_world_s_1280_vcoco", "fused")}
    budget, rows = {}, []
    for model in ("YOLO11s", "YOLO-World-S"):
        z = zero[model]
        rows.append([model, "0 (zero-shot)", "—", f1(z["mod"]), "—", f1(z["Car"]), f1(z["Pedestrian"]), f1(z["Cyclist"])])
        for frac in (0.05, 0.25, 1.0):
            vals = [kitti(r) for r in runs[(model, frac)]]
            agg = {k: (float(np.mean([v[k] for v in vals])), float(np.std([v[k] for v in vals], ddof=1)) if len(vals) > 1 else None)
                   for k in vals[0]}
            budget[f"{model}@{frac}"] = {"n_images": round(3712 * frac), "seeds": len(vals), "runs": runs[(model, frac)],
                                         "per_seed_mod": [v["mod"] for v in vals], **{k: list(v) for k, v in agg.items()}}
            pm = lambda k: f1(agg[k][0]) + (f" ± {agg[k][1]:.1f}" if agg[k][1] is not None else "")
            rng = f"{min(v['mod'] for v in vals):.1f}–{max(v['mod'] for v in vals):.1f}" if len(vals) > 1 else "—"
            rows.append([model, f"{int(frac * 100)}% ({round(3712 * frac)} img)", len(vals), pm("mod"), rng,
                         pm("Car"), pm("Pedestrian"), pm("Cyclist")])
    wr("ext_budget", table(
        ["Model", "KITTI train labels", "Seeds", "Mod mAP (mean ± sd)", "Seed range", "Car", "Ped", "Cyc"], rows))
    summary["budget"] = budget
    summary["budget_zero"] = zero

    # fine-tuning cost (train hours, memory) from results/finetune/*.json
    ft = {}
    for p in sorted((ROOT / "results" / "finetune").glob("*_kitti_1280_e20_*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        ft[d["name"]] = {"train_hours": d.get("train_hours"), "peak_mem_gb": d.get("peak_mem_gb")}
    summary["finetune_cost"] = ft

    # ---------------------------------------------------------------- speed
    sp = json.loads(SPEED.read_text(encoding="utf-8"))["rows"]
    rows = []
    for r in sp:
        if r["domain"] != "kitti" or "ms_mean" not in r:
            continue
        rows.append([r["model"], r["precision"].upper(), r["params_m"], f1(r["ms_mean"]), f1(r["ms_p50"]), f1(r["fps"]), round(r["peak_mem_mb"])])
    (TAB / "ext_speed.md").write_text(table(["Model@input", "Prec.", "Params (M)", "ms mean", "ms p50", "FPS", "Peak MB"], rows), encoding="utf-8")
    summary["speed"] = sp
    (TAB / "ext_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")

    # ---------------------------------------------------------------- figures
    # Fig E1: COCO vs O365 pre-training, paired bars (COCO AP, KITTI Mod @640, @1280)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4), sharey=False)
    for ax, size in zip(axes, ("s", "m")):
        cats = ["COCO AP", "KITTI Mod\n@640", "KITTI Mod\n@1280", "KITTI Ped\n@1280", "KITTI Cyc\n@1280"]
        get = lambda key: [zs[key]["coco"]["AP"], zs[key]["kitti640"]["mod"], zs[key]["kitti1280"]["mod"],
                           zs[key]["kitti1280"]["Pedestrian"], zs[key]["kitti1280"]["Cyclist"]]
        x = np.arange(len(cats))
        for i, (lab, key) in enumerate([("COCO only", f"dfine_{size}_coco"), ("Objects365 → COCO", f"dfine_{size}_o365")]):
            v = get(key)
            b = ax.bar(x + (i - 0.5) * 0.38, v, 0.38, label=lab, color=S[i])
            ax.bar_label(b, fmt="%.0f", fontsize=6, padding=1)
        ax.set_xticks(x, cats, fontsize=7)
        ax.set_title(f"D-FINE-{size.upper()}", fontsize=9)
        ax.set_ylim(0, 100)
    axes[0].set_ylabel("AP / AP|R40 (%)")
    axes[0].legend(fontsize=7, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "ext_pretraining.png")
    plt.close(fig)

    # Fig E2: label-budget curve
    fig, ax = plt.subplots(figsize=(4.2, 2.5))
    for i, model in enumerate(("YOLO11s", "YOLO-World-S")):
        xs = [0, 5, 25, 100]
        mean = [zero[model]["mod"]] + [budget[f"{model}@{f}"]["mod"][0] for f in (0.05, 0.25, 1.0)]
        ax.plot(range(4), mean, marker="o", color=S[i], label=model)
        for j, f in enumerate((0.05, 0.25, 1.0), start=1):
            pts = budget[f"{model}@{f}"]["per_seed_mod"]
            ax.scatter([j + (i - 0.5) * 0.08] * len(pts), pts, s=8, color=S[i], alpha=0.5)
        ax.set_xticks(range(4), [f"{x}%" if x else "0\n(zero-shot)" for x in xs])
    ax.set_xlabel("Fraction of KITTI train labels (3,712 images)")
    ax.set_ylabel("KITTI Moderate mAP")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "ext_label_budget.png")
    plt.close(fig)

    for f in ("ext_zeroshot", "ext_yoloworld_vocab", "ext_budget", "ext_speed"):
        print(f"\n## {f}\n" + (TAB / f"{f}.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
