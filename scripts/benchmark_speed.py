"""Latency / throughput / memory benchmark (batch 1, FP32 and FP16).

    uv run python scripts/benchmark_speed.py

For each detector and input domain, the same 100 real images are pushed through the
full ``predict()`` path (pre-processing, network, NMS / decoding, post-processing) after
one complete untimed warm-up pass over those images, so cuDNN autotuning for every input
shape happens before timing. CUDA is synchronised around every call. Image decoding is
excluded. Peak memory is ``torch.cuda.max_memory_allocated`` (tensors only; the CUDA
context adds roughly 0.3-0.5 GB on top). Results: results/speed/speed.json
"""

from __future__ import annotations

import gc
import json
import subprocess
import time

import numpy as np
import torch
from PIL import Image

from det_eval import detectors
from det_eval.data import COCO_IMAGES, ROOT, coco_image_records, kitti_image_path, read_split
from det_eval.mapping import VOCABULARIES

CONFIGS = [  # (model key, build kwargs, domains)
    ("faster_rcnn_r50_v2", {}, ["coco", "kitti"]),
    ("retinanet_r50_v2", {}, ["coco", "kitti"]),
    ("fcos_r50", {}, ["coco", "kitti"]),
    ("yolo11s", {"imgsz": 640}, ["coco", "kitti"]),
    ("yolo11s", {"imgsz": 1280}, ["kitti"]),
    ("yolo11m", {"imgsz": 640}, ["coco", "kitti"]),
    ("yolo11m", {"imgsz": 1280}, ["kitti"]),
    ("rtdetr_l", {"imgsz": 640}, ["coco", "kitti"]),
    ("rtdetr_l", {"imgsz": 1280}, ["kitti"]),
    ("yolo_world_l", {"imgsz": 640}, ["coco"]),
    ("yolo_world_l", {"imgsz": 1280}, ["kitti"]),
    ("grounding_dino_t", {}, ["coco", "kitti"]),
]
N_IMAGES = 100


def images(domain):
    if domain == "coco":
        paths = [COCO_IMAGES / r["file_name"] for r in coco_image_records()[:N_IMAGES]]
    else:
        paths = [kitti_image_path(s) for s in read_split("val")[:N_IMAGES]]
    return [np.asarray(Image.open(p).convert("RGB")) for p in paths]


def n_params(det) -> float:
    m = det.model.model if det.framework == "ultralytics" else det.model
    return sum(p.numel() for p in m.parameters()) / 1e6


def gpu_state():
    q = "power.limit,clocks.max.sm,driver_version"
    out = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"], capture_output=True, text=True)
    return out.stdout.strip()


def bench(det, imgs):
    for img in imgs:  # full warm-up pass (all input shapes autotuned)
        det.predict(img)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    lat = []
    for img in imgs:
        t0 = time.perf_counter()
        det.predict(img)
        torch.cuda.synchronize()
        lat.append((time.perf_counter() - t0) * 1e3)
    lat = np.array(lat)
    return dict(
        ms_mean=round(float(lat.mean()), 2),
        ms_p50=round(float(np.median(lat)), 2),
        ms_p90=round(float(np.percentile(lat, 90)), 2),
        fps=round(1000 / float(lat.mean()), 1),
        peak_mem_mb=round(torch.cuda.max_memory_allocated() / 2**20, 1),
    )


def main():
    torch.backends.cudnn.benchmark = True
    data = {d: images(d) for d in ("coco", "kitti")}
    rows = []
    for key, kw, domains in CONFIGS:
        for half in (False, True):
            det = detectors.build(key, half=half, **kw)
            for domain in domains:
                if det.open_vocab:
                    det.set_vocabulary(VOCABULARIES["coco"])
                try:
                    r = dict(model=det.name, key=key, domain=domain, precision="fp16" if half else "fp32",
                             params_m=round(n_params(det), 1), **bench(det, data[domain]))
                except Exception as e:  # e.g. an op without FP16 support: record and keep going
                    r = dict(model=det.name, key=key, domain=domain, precision="fp16" if half else "fp32", error=repr(e)[:200])
                print(r, flush=True)
                rows.append(r)
            del det
            gc.collect()
            torch.cuda.empty_cache()
    out = ROOT / "results" / "speed"
    out.mkdir(parents=True, exist_ok=True)
    meta = dict(gpu=torch.cuda.get_device_name(0), nvidia_smi=gpu_state(), torch=torch.__version__,
                n_images=N_IMAGES, warmup="1 full pass", batch=1, cudnn_benchmark=True)
    (out / "speed.json").write_text(json.dumps(dict(meta=meta, rows=rows), indent=1))


if __name__ == "__main__":
    main()
