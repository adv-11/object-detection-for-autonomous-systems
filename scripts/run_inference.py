"""Run one detector over a dataset and save raw (unmapped) detections.

    uv run python scripts/run_inference.py --model yolo11m --dataset coco
    uv run python scripts/run_inference.py --model yolo11m --dataset kitti --imgsz 1280
    uv run python scripts/run_inference.py --model grounding_dino_t --dataset kitti --vocab kitti

Output: results/raw/<run>.json.gz with the detector config, vocabulary, per-image
detections [image_id, x1, y1, x2, y2, score, label], wall-clock timing and peak memory.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import platform
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from det_eval import detectors
from det_eval.data import COCO_IMAGES, ROOT, coco_image_records, kitti_image_path, read_split
from det_eval.mapping import COCO_NAMES, VOCABULARIES

RAW = ROOT / "results" / "raw"
PREFETCH = 8


def image_list(dataset: str, limit: int | None):
    if dataset == "coco":
        items = [(r["id"], COCO_IMAGES / r["file_name"]) for r in coco_image_records()]
    else:
        items = [(int(s), kitti_image_path(s)) for s in read_split("val")]
    return items[:limit] if limit else items


def load(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def environment() -> dict:
    import torchvision
    import transformers
    import ultralytics

    return dict(
        gpu=torch.cuda.get_device_name(0),
        cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(),
        torch=torch.__version__,
        torchvision=torchvision.__version__,
        ultralytics=ultralytics.__version__,
        transformers=transformers.__version__,
        python=sys.version.split()[0],
        os=platform.platform(),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(detectors.REGISTRY))
    ap.add_argument("--dataset", required=True, choices=["coco", "kitti"])
    ap.add_argument("--imgsz", type=int, default=None, help="long-side input size (Ultralytics models only)")
    ap.add_argument("--vocab", default="coco", choices=sorted(VOCABULARIES), help="prompt vocabulary for open-vocab models")
    ap.add_argument("--weights", default=None, help="override checkpoint path (fine-tuned models)")
    ap.add_argument("--tag", default=None, help="run name suffix")
    ap.add_argument("--limit", type=int, default=None, help="debug: only the first N images")
    ap.add_argument("--overwrite", action="store_true", help="replace an existing raw file")
    args = ap.parse_args()

    kwargs = {}
    if args.imgsz:
        kwargs["imgsz"] = args.imgsz
    det = detectors.build(args.model, **kwargs)
    if args.weights:
        det = detectors.UltralyticsDetector(det.name + " KITTI-FT", det.family, args.weights, det.imgsz)
        args.tag = args.tag or "ft"  # never collide with the zero-shot run of the same model
    if det.open_vocab:
        det.set_vocabulary(VOCABULARIES[args.vocab])
    elif args.vocab != "coco":
        sys.exit("--vocab only applies to open-vocabulary models")

    run = args.model + (f"_{args.imgsz}" if args.imgsz else "") + (f"_v{args.vocab}" if det.open_vocab else "")
    run = f"{args.dataset}__{run}" + (f"_{args.tag}" if args.tag else "")
    items = image_list(args.dataset, args.limit)
    if (RAW / f"{run}.json.gz").exists() and not args.overwrite:
        sys.exit(f"{run} already exists; pass --overwrite to replace it")

    # Warm up kernels / cudnn autotuning outside the timed loop.
    for _ in range(3):
        det.predict(load(items[0][1]))
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    rows, model_ms = [], []
    t_start = time.perf_counter()
    with ThreadPoolExecutor(4) as pool:
        pending = collections.deque(pool.submit(load, p) for _, p in items[:PREFETCH])  # bounded prefetch
        for i, (img_id, _) in enumerate(tqdm(items, desc=run, mininterval=20)):
            img = pending.popleft().result()
            if i + PREFETCH < len(items):
                pending.append(pool.submit(load, items[i + PREFETCH][1]))
            t0 = time.perf_counter()
            d = det.predict(img)
            torch.cuda.synchronize()
            model_ms.append((time.perf_counter() - t0) * 1e3)
            for b, s, l in zip(d.boxes, d.scores, d.labels):
                rows.append([img_id, *(round(float(v), 2) for v in b), round(float(s), 5), int(l)])
    wall = time.perf_counter() - t_start

    meta = dict(
        run=run,
        dataset=args.dataset,
        split="val2017" if args.dataset == "coco" else "kitti-val-3769",
        num_images=len(items),
        image_ids=[i for i, _ in items],
        detector=det.config(),
        vocab_key=args.vocab if det.open_vocab else ("kitti-ft" if args.weights else "coco"),
        weights=args.weights,
        tag=args.tag,
        vocabulary=det.class_names,
        timing=dict(
            wall_s=round(wall, 1),
            images_per_s=round(len(items) / wall, 2),
            model_ms_mean=round(float(np.mean(model_ms)), 2),
            model_ms_median=round(float(np.median(model_ms)), 2),
            note="per-image predict() incl. pre/post-processing, batch 1, FP32, image decode excluded",
        ),
        peak_mem_mb=round(torch.cuda.max_memory_allocated() / 2**20, 1),
        environment=environment(),
        command=" ".join(["python", *sys.argv]),
    )
    RAW.mkdir(parents=True, exist_ok=True)
    with gzip.open(RAW / f"{run}.json.gz", "wt") as f:
        json.dump(dict(meta=meta, detections=rows), f)
    print(json.dumps(meta["timing"]), f"peak_mem={meta['peak_mem_mb']}MB", f"dets={len(rows)}")


if __name__ == "__main__":
    main()
