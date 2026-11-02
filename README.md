# 2D Object Detection: SOTA Evaluation and Cross-Domain Generalization (COCO → KITTI)

CMPE 249 (Intelligent Autonomous Systems), Track A: experimental model evaluation.
**Report:** [`report/report.pdf`](report/report.pdf)

Eight COCO-trained detector configurations from six families — Faster R-CNN v2, RetinaNet v2, FCOS, YOLO11s/m, RT-DETR-L,
YOLO-World-L v2 and Grounding DINO-T — are evaluated under one shared protocol on **COCO val2017** and, without retraining, on
**KITTI val** (3,769 images). A short fine-tune on KITTI train measures how much of the domain gap adaptation recovers.

## Headline results

| | COCO AP (ours / published) | KITTI Moderate mAP (AP\|R40, zero-shot) |
|---|---|---|
| RT-DETR-L | 52.6 / 53.0 | 60.7 (640², stretched) · 65.3 (1280²) |
| YOLO11m | 51.0 / 51.5 | 57.7 (640) · 65.3 (1280) |
| Grounding DINO-T | 48.3 / 48.4 | **70.5** (COCO names + mapping) · 39.9 (KITTI-name prompts) |
| Faster R-CNN R50-FPN v2 | 46.9 / 46.7 | 58.8 |
| YOLO11s | 46.1 / 47.0 | 52.1 (640) · 64.4 (1280) → **75.5 fine-tuned** (77.0 with 25% of labels) |
| YOLO-World-L v2 | 45.1 / 45.1 | **70.9** (COCO names + mapping) · 46.4 (KITTI-name prompts) |
| RetinaNet R50-FPN v2 | 41.4 / 41.5 | 56.1 |
| FCOS R50-FPN | 39.2 / 39.2 | 53.7 |

* Car transfers from COCO to KITTI almost unchanged (COCO-style AP 49.9 → 50.4 for Faster R-CNN); Pedestrian and Cyclist lose about
  two-thirds of their AP to taxonomy, box-convention and resize effects.
* Keeping KITTI's 1242-px frames at 1280 px instead of 640 adds 5–12 Moderate mAP without retraining and erases the YOLO/RT-DETR gap.
* Mapping COCO `bicycle` to KITTI Cyclist naively gives <1 AP; fusing rider and bike gives up to 68.
* The open-vocabulary models (sixth and third on COCO) are the best zero-shot KITTI detectors, but prompting them with KITTI class names
  instead of COCO names costs 25–31 points.
* Fine-tuning YOLO11s for 20 epochs at 1280 px lifts it from 64.4 to 75.5 Moderate mAP (77.0 with a quarter of the labels), mostly by
  fixing localisation at KITTI's 0.7 Car IoU.
* FP16 speeds up the heavier models 1.1–1.6× on this GPU; batch-1 YOLO latency (~23 ms) is bound by Python pre/post-processing.

## Reproduce

Python 3.12 + [uv](https://docs.astral.sh/uv/); PyTorch built for CUDA 12.8 (required for Blackwell / sm_120 GPUs).
Hardware used: NVIDIA RTX 5060 Laptop GPU (8 GB, 45 W), Windows 11.

```bash
uv sync
bash scripts/download_data.sh                    # COCO val2017 + KITTI object (training split)
uv run python scripts/prepare_kitti.py           # KITTI -> COCO-format GT (devkit ignore rules) + YOLO export
uv run pytest tests                              # 18 regression tests for the metric / mapping code

# inference (raw, unmapped detections -> results/raw/*.json.gz), one run per command
uv run python scripts/run_inference.py --model yolo11m --dataset coco
uv run python scripts/run_inference.py --model yolo11m --dataset kitti --imgsz 1280
uv run python scripts/run_inference.py --model yolo_world_l --dataset kitti --imgsz 1280 --vocab kitti

# fine-tune on KITTI train only, then evaluate once on KITTI val
uv run python scripts/finetune_yolo.py --model yolo11s.pt --imgsz 640 --epochs 5 --batch 16
uv run python scripts/run_inference.py --model yolo11s --dataset kitti --imgsz 640 \
    --weights runs/finetune/yolo11s_kitti_640_e5_f1/weights/last.pt --tag ft

uv run python scripts/evaluate.py                # -> results/metrics/*.json
uv run python scripts/analyze_failures.py kitti__yolo11s_640 kitti__yolo11s_640_ft
uv run python scripts/make_figures.py && bash report/build.sh
```

The exact runs in the report are in [`scripts/run_reduced.sh`](scripts/run_reduced.sh),
[`scripts/run_finetune_short.sh`](scripts/run_finetune_short.sh) and [`scripts/run_overnight.sh`](scripts/run_overnight.sh)
(plus the COCO part of `scripts/run_sweep.sh`);
[`scripts/run_all.sh`](scripts/run_all.sh) is the full, longer queue. Every executed command with timestamps, throughput and
peak memory is in [`logs/command_log.txt`](logs/command_log.txt).

## Layout

| Path | Contents |
|---|---|
| `src/det_eval/detectors.py` | One `predict()` interface over Ultralytics, torchvision and Hugging Face detectors; shared protocol |
| `src/det_eval/mapping.py` | COCO → KITTI taxonomy (direct / fused / fused-loose Cyclist strategies), prompt vocabularies |
| `src/det_eval/metrics.py` | COCOeval plus KITTI devkit-style AP\|R40 (`KittiEval` honours neighbour/difficulty ignores) |
| `src/det_eval/data.py` | KITTI parsing, difficulty levels, COCO-format GT, YOLO export, fine-tuning yaml |
| `results/metrics/` | Per-run metrics with detector config, environment, timing and the exact command |
| `results/failures/` | Recall vs. height/occlusion, false-positive taxonomy, Car IoU distribution |
| `results/tables/`, `report/generated/` | Tables rendered from the metric files |
| `figures/` | All report figures (`scripts/make_figures.py`, `scripts/visualize.py`, `scripts/dataset_stats.py`) |
| `tests/` | Regression tests (ignore semantics, AP\|R40, rider fusion) |

## Scope note

Single 8-GB laptop GPU. Fine-tuning uses YOLO11s with one seed per configuration (YOLO11m at 1280 px does not fit in 8 GB for training).
Raw prediction files (`results/raw/`) and weights are not committed (size); every metric file records how to regenerate them.

## Data splits

| Dataset | Split | Images | Use |
|---|---|---|---|
| COCO 2017 | val2017 | 5,000 | in-domain evaluation |
| KITTI 2D object | train (Chen et al.) | 3,712 | fine-tuning only |
| KITTI 2D object | val (Chen et al.) | 3,769 | evaluation only; lists in `configs/kitti_split/` (from OpenPCDet) |
