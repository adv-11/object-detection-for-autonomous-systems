"""Collect results/metrics/*.json into CSV + Markdown tables (results/tables/).

    uv run python scripts/summarize.py
"""

from __future__ import annotations

import json

import pandas as pd

from det_eval.data import ROOT

MET = ROOT / "results" / "metrics"
TAB = ROOT / "results" / "tables"


def load_all():
    return [json.loads(p.read_text()) for p in sorted(MET.glob("*.json"))]


def coco_table(runs):
    rows = []
    for r in runs:
        if r["meta"]["dataset"] != "coco":
            continue
        m, d, t = r["metrics"]["coco"], r["meta"]["detector"], r["meta"]["timing"]
        rows.append(dict(model=d["name"], family=d["family"], **m["summary"],
                         ms_per_img=t["model_ms_median"], peak_mem_mb=r["meta"]["peak_mem_mb"],
                         AP_person=m["per_class_AP"]["person"], AP_car=m["per_class_AP"]["car"],
                         AP_bicycle=m["per_class_AP"]["bicycle"]))
    df = pd.DataFrame(rows)
    return df.sort_values("AP", ascending=False) if not df.empty else df


def kitti_table(runs):
    rows = []
    for r in runs:
        if r["meta"]["dataset"] != "kitti":
            continue
        d, voc = r["meta"]["detector"], r["meta"]["run"]
        vocab = r["meta"].get("vocab_key", "coco")
        for strat, m in r["metrics"].items():
            s, k = m["coco_style"]["summary"], m["kitti_R40"]
            rows.append(dict(
                run=voc, model=d["name"], family=d["family"], vocab=vocab, mapping=strat,
                AP=s["AP"], AP50=s["AP50"], AP75=s["AP75"], APs=s["APs"], APm=s["APm"], APl=s["APl"], AR100=s["AR100"],
                AP_car=m["coco_style"]["per_class_AP"]["Car"],
                AP_ped=m["coco_style"]["per_class_AP"]["Pedestrian"],
                AP_cyc=m["coco_style"]["per_class_AP"]["Cyclist"],
                **{f"R40_{lvl[:3]}_{c[:3]}": k[lvl][c] for lvl in ("easy", "moderate", "hard")
                   for c in ("Car", "Pedestrian", "Cyclist", "mAP")},
                ms_per_img=r["meta"]["timing"]["model_ms_median"],
            ))
    return pd.DataFrame(rows)


def main():
    TAB.mkdir(parents=True, exist_ok=True)
    runs = load_all()
    for name, df in (("coco", coco_table(runs)), ("kitti", kitti_table(runs))):
        if df.empty:
            continue
        df.to_csv(TAB / f"{name}.csv", index=False)
        (TAB / f"{name}.md").write_text(df.to_markdown(index=False, floatfmt=".1f"))
        print(f"\n## {name}\n")
        print(df.to_markdown(index=False, floatfmt=".1f"))


if __name__ == "__main__":
    main()
