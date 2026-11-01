"""Fine-tune a COCO-pretrained Ultralytics detector on KITTI train (3,712 images).

    uv run python scripts/finetune_yolo.py --model yolo11m.pt --imgsz 1280 --epochs 30 --batch 8

Protocol notes
* Training uses only the KITTI *train* split (Chen et al.); validation during training is
  disabled so KITTI val is never used for checkpoint or hyper-parameter selection. The
  final-epoch weights (last.pt) are evaluated exactly once with scripts/run_inference.py.
* 7 training classes (Car, Van, Truck, Pedestrian, Person_sitting, Cyclist, Tram);
  DontCare and Misc boxes are dropped. Evaluation still scores only Car/Pedestrian/Cyclist.
* All other hyper-parameters are Ultralytics defaults (SGD/AdamW auto, mosaic, flips,
  HSV jitter), so the run is reproducible from this command and the seed.
"""

import argparse
import json
import time

import torch
from ultralytics import RTDETR, YOLO

from det_eval.data import DATA, ROOT
from det_eval.detectors import WEIGHTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11m.pt")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--fraction", type=float, default=1.0, help="fraction of KITTI train to use")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    name = args.name or f"{args.model.removesuffix('.pt')}_kitti_{args.imgsz}_e{args.epochs}_f{args.fraction:g}"
    cls = RTDETR if "rtdetr" in args.model else YOLO
    model = cls(str(WEIGHTS / args.model))
    t0 = time.time()
    model.train(
        data=str(DATA / "kitti_yolo" / "kitti.yaml"),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        fraction=args.fraction,
        seed=args.seed,
        deterministic=True,
        val=False,
        plots=True,
        workers=4,
        device=0,
        project=str(ROOT / "runs" / "finetune"),
        name=name,
        exist_ok=True,
    )
    hours = (time.time() - t0) / 3600
    run_dir = ROOT / "runs" / "finetune" / name
    summary = dict(
        name=name,
        args=vars(args),
        train_hours=round(hours, 2),
        peak_mem_gb=round(torch.cuda.max_memory_reserved() / 2**30, 2),
        weights=str((run_dir / "weights" / "last.pt").relative_to(ROOT)),
    )
    out = ROOT / "results" / "finetune"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(summary, indent=1))
    # Keep the per-epoch training log and hyper-parameters under version control.
    for f in ("results.csv", "args.yaml"):
        if (run_dir / f).exists():
            (out / f"{name}__{f}").write_text((run_dir / f).read_text())
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
