"""Generate the report's Markdown tables from results/ (so every number traces to a metric file).

    uv run python report/make_tables.py      # writes report/generated/<name>.md
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MET = ROOT / "results" / "metrics"
GEN = Path(__file__).resolve().parent / "generated"

PUBLISHED = {  # verified from primary sources (see report references)
    "coco__faster_rcnn_r50_v2": ("Faster R-CNN R50-FPN v2", "two-stage CNN", 43.7, "46.7"),
    "coco__retinanet_r50_v2": ("RetinaNet R50-FPN v2", "one-stage, anchor-based", 38.2, "41.5"),
    "coco__fcos_r50": ("FCOS R50-FPN", "one-stage, anchor-free", 32.3, "39.2"),
    "coco__yolo11s": ("YOLO11s", "real-time CNN (YOLO)", 9.4, "47.0"),
    "coco__yolo11m": ("YOLO11m", "real-time CNN (YOLO)", 20.1, "51.5"),
    "coco__rtdetr_l": ("RT-DETR-L", "real-time DETR", 32.0, "53.0"),
    "coco__yolo_world_l_vcoco": ("YOLO-World-L v2", "open-vocabulary CNN", None, "45.1*"),
    "coco__grounding_dino_t_vcoco": ("Grounding DINO-T", "open-vocabulary DETR", None, "48.4*"),
}


def _speed():
    p = ROOT / "results" / "speed" / "speed.json"
    out = {}
    if p.exists():
        for r in json.loads(p.read_text())["rows"]:
            if r["domain"] == "coco" and "ms_p50" in r:
                out.setdefault(r["model"].replace("@640", "@640"), {})[r["precision"]] = r
    return out


SPEED = _speed()


def load(run):
    p = MET / f"{run}.json"
    return json.loads(p.read_text()) if p.exists() else None


def md(header, rows, align=None):
    align = align or ["l"] + ["r"] * (len(header) - 1)
    sep = ["---:" if a == "r" else ":---" for a in align]
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join(sep) + " |"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def f(v, d=1):
    return "–" if v is None else f"{v:.{d}f}"


def bold_max(rows, cols):
    for c in cols:
        vals = [float(r[c]) for r in rows if r[c] not in ("–",)]
        if not vals:
            continue
        m = max(vals)
        for r in rows:
            if r[c] != "–" and float(r[c]) == m:
                r[c] = f"**{r[c]}**"
    return rows


def coco_table():
    rows = []
    for run, (name, fam, params, pub) in PUBLISHED.items():
        m = load(run)
        if not m:
            continue
        s = m["metrics"]["coco"]["summary"]
        sp = SPEED.get(m["meta"]["detector"]["name"], {})
        p32, p16 = sp.get("fp32", {}), sp.get("fp16", {})
        params = params or p32.get("params_m")
        rows.append([name, fam, f(params) if params else "–", f(s["AP"]), pub, f(s["AP50"]), f(s["AP75"]),
                     f(s["APs"]), f(s["APm"]), f(s["APl"]), f(s["AR100"]), f(p32.get("ms_p50"), 0), f(p16.get("ms_p50"), 0),
                     f(p32["peak_mem_mb"] / 1024, 2) if p32 else "–"])
    rows.sort(key=lambda r: -float(r[3]))
    rows = bold_max(rows, [3, 5, 6, 7, 8, 9, 10])
    return md(["Model", "Family", "Params (M)", "AP", "Publ. AP", "AP50", "AP75", "APs", "APm", "APl", "AR100", "FP32 ms", "FP16 ms", "Mem (GB)"],
              rows, ["l", "l"] + ["r"] * 12)


KITTI_RUNS = [
    ("kitti__faster_rcnn_r50_v2", "Faster R-CNN R50-FPN v2", "~1333×402"),
    ("kitti__retinanet_r50_v2", "RetinaNet R50-FPN v2", "~1333×402"),
    ("kitti__fcos_r50", "FCOS R50-FPN", "~1333×402"),
    ("kitti__yolo11s_640", "YOLO11s", "640×194"),
    ("kitti__yolo11s_1280", "YOLO11s", "1280×387"),
    ("kitti__yolo11m_640", "YOLO11m", "640×194"),
    ("kitti__yolo11m_1280", "YOLO11m", "1280×387"),
    ("kitti__rtdetr_l_640", "RT-DETR-L", "640×640†"),
    ("kitti__rtdetr_l_1280", "RT-DETR-L", "1280×1280†"),
    ("kitti__yolo_world_l_1280_vcoco", "YOLO-World-L v2", "1280×387"),
    ("kitti__grounding_dino_t_vcoco", "Grounding DINO-T", "~1333×402"),
    ("kitti__yolo11s_640_ft", "YOLO11s FT, 5 ep., 100% train", "640×194"),
    ("kitti__yolo11s_1280_ft25", "YOLO11s FT, 20 ep., 25% train", "1280×387"),
    ("kitti__yolo11s_1280_ft", "YOLO11s FT, 20 ep., 100% train", "1280×387"),
]


def kitti_metrics(m):
    key = "fused" if "fused" in m["metrics"] else "prompt"
    return m["metrics"][key]


def kitti_table():
    rows = []
    for run, name, inp in KITTI_RUNS:
        m = load(run)
        if not m:
            continue
        k = kitti_metrics(m)
        s, cs, r40 = k["coco_style"]["summary"], k["coco_style"]["per_class_AP"], k["kitti_R40"]
        rows.append([name, inp, f(s["AP"]), f(s["AP50"]), f(s["APs"]), f(cs["Car"]), f(cs["Pedestrian"]), f(cs["Cyclist"]),
                     f(r40["moderate"]["Car"]), f(r40["moderate"]["Pedestrian"]), f(r40["moderate"]["Cyclist"]),
                     f(r40["easy"]["mAP"]), f(r40["moderate"]["mAP"]), f(r40["hard"]["mAP"])])
    zs = [r for r in rows if " FT," not in r[0]]
    bold_max(zs, list(range(2, 14)))
    return md(["Model", "Input", "AP", "AP50", "APs", "Car", "Ped", "Cyc", "Car", "Ped", "Cyc", "Easy", "Mod.", "Hard"],
              rows, ["l", "l"] + ["r"] * 12)


def mapping_table():
    rows = []
    for run, name, _ in KITTI_RUNS:
        m = load(run)
        if not m or "fused" not in m["metrics"]:
            continue
        for strat, lab in (("direct", "direct"), ("fused_loose", "fused (loose)"), ("fused", "fused (strict, default)")):
            k = m["metrics"][strat]
            r = k["kitti_R40"]["moderate"]
            cs = k["coco_style"]["per_class_AP"]
            rows.append([name, lab, f(cs["Pedestrian"]), f(cs["Cyclist"]), f(r["Pedestrian"]), f(r["Cyclist"]), f(r["mAP"])])
    return md(["Model", "Cyclist mapping", "Ped AP", "Cyc AP", "Ped mod.", "Cyc mod.", "mAP mod."], rows, ["l", "l"] + ["r"] * 5)


def prompt_table():
    rows = []
    for run, lab in (("kitti__grounding_dino_t_vcoco", "Grounding DINO-T: 80 COCO names + mapping"),
                     ("kitti__grounding_dino_t_vkitti", "Grounding DINO-T: KITTI names"),
                     ("kitti__yolo_world_l_1280_vcoco", "YOLO-World-L: 80 COCO names + mapping"),
                     ("kitti__yolo_world_l_1280_vkitti", "YOLO-World-L: KITTI names (car, pedestrian, cyclist, ...)"),
                     ("kitti__yolo_world_l_1280_vkittidesc", "YOLO-World-L: KITTI names, 'person riding a bicycle'")):
        m = load(run)
        if not m:
            continue
        k = kitti_metrics(m)
        r, cs = k["kitti_R40"]["moderate"], k["coco_style"]["per_class_AP"]
        rows.append([lab, f(cs["Car"]), f(cs["Pedestrian"]), f(cs["Cyclist"]), f(r["Car"]), f(r["Pedestrian"]), f(r["Cyclist"]), f(r["mAP"])])
    return md(["Detector and vocabulary", "Car AP", "Ped AP", "Cyc AP", "Car mod.", "Ped mod.", "Cyc mod.", "mAP mod."],
              rows, ["l"] + ["r"] * 7)


def failure_table():
    fdir = ROOT / "results" / "failures"
    rows = []
    for run, lab in (("kitti__faster_rcnn_r50_v2", "Faster R-CNN"), ("kitti__grounding_dino_t_vcoco", "Grounding DINO-T"),
                     ("kitti__yolo_world_l_1280_vcoco", "YOLO-World-L@1280"),
                     ("kitti__yolo11s_1280", "YOLO11s@1280 zero-shot"), ("kitti__yolo11s_1280_ft", "YOLO11s@1280 fine-tuned")):
        p = fdir / f"{run}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        for cls in ("Car", "Pedestrian"):
            bins = d["recall_by_height"][cls]
            occ = d["recall_by_occlusion"][cls]
            rows.append([lab, cls] + [f(bins[b]["recall_max"], 0) for b in bins] +
                        [f(occ[o]["recall_max"], 0) for o in ("0", "1", "2")])
        ci = d["car_matched_iou"]
        rows[-2].append(f(100 * ci["frac_below_0_7"], 0) if ci["frac_below_0_7"] is not None else "–")
        rows[-1].append("")
    return md(["Model", "Class", "<25", "25–40", "40–60", "60–100", ">100", "occ 0", "occ 1", "occ 2", "Car IoU<0.7 (%)"],
              rows, ["l", "l"] + ["r"] * 9)


def main():
    GEN.mkdir(exist_ok=True)
    for name, fn in (("coco", coco_table), ("kitti", kitti_table), ("mapping", mapping_table),
                     ("prompt", prompt_table), ("failure", failure_table)):
        try:
            (GEN / f"{name}.md").write_text(fn(), encoding="utf-8")
        except Exception as e:  # tables whose runs are missing are simply skipped
            print(f"skip {name}: {e}")
    print("tables:", sorted(p.name for p in GEN.glob("*.md")))


if __name__ == "__main__":
    main()
