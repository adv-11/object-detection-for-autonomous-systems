# 2D Object Detection: SOTA Evaluation and Cross-Domain Generalization (COCO → KITTI)

CMPE 249 (Intelligent Autonomous Systems), Track A: experimental model evaluation.
**Report:** [`report/report.pdf`](report/report.pdf)

Eight COCO-trained detector configurations from six families — Faster R-CNN v2, RetinaNet v2, FCOS, YOLO11s/m, RT-DETR-L,
YOLO-World-L v2 and Grounding DINO-T — are evaluated under one shared protocol on **COCO val2017** and, without retraining, on
**KITTI val** (3,769 images). A short fine-tune on KITTI train measures how much of the domain gap adaptation recovers.

## Headline results

| | COCO AP (ours / published) | KITTI Moderate mAP (AP\|R40, zero-shot) |
|---|---|---|
| RT-DETR-L | 52.6 / 53.0 | 60.7 (640², stretched) |
| YOLO11m | 51.0 / 51.5 | 57.7 (640) · **65.3 (1280)** |
| Grounding DINO-T | 48.3 / 48.4 | see report |
| Faster R-CNN R50-FPN v2 | 46.9 / 46.7 | 58.8 |
| YOLO11s | 46.1 / 47.0 | 52.1 (640) → **65.8 after 5-epoch fine-tune** |
| YOLO-World-L v2 | 45.1 / 45.1 | see report |
| RetinaNet R50-FPN v2 | 41.4 / 41.5 | – |
| FCOS R50-FPN | 39.2 / 39.2 | – |

* Car transfers from COCO to KITTI almost unchanged (COCO-style AP 49.9 → 50.4 for Faster R-CNN); Pedestrian and Cyclist lose about
  two-thirds of their AP to taxonomy, box-convention and resize effects.
* Keeping KITTI's 1242-px frames at 1280 px instead of 640 adds 7.6 Moderate mAP to YOLO11m without retraining.
* Mapping COCO `bicycle` to KITTI Cyclist naively gives 0.4 AP; fusing rider and bike gives up to 51.
* 5 epochs (8 min) of fine-tuning lift YOLO11s from 52.1 to 65.8 Moderate mAP, mostly by fixing localisation at KITTI's 0.7 Car IoU.

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

The exact runs in the report are in [`scripts/run_reduced.sh`](scripts/run_reduced.sh) and
[`scripts/run_finetune_short.sh`](scripts/run_finetune_short.sh) (plus the COCO part of `scripts/run_sweep.sh`);
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

The study was time-boxed to a single laptop GPU. Fine-tuning is one seed of YOLO11s at 640 px for 5 epochs (YOLO11m at 1280 px does not
fit in 8 GB). RetinaNet, FCOS and RT-DETR at 1280 px were evaluated on COCO only, and Grounding DINO on KITTI with one vocabulary.
Raw prediction files (`results/raw/`) and weights are not committed (size); every metric file records how to regenerate them.

## Data splits

| Dataset | Split | Images | Use |
|---|---|---|---|
| COCO 2017 | val2017 | 5,000 | in-domain evaluation |
| KITTI 2D object | train (Chen et al.) | 3,712 | fine-tuning only |
| KITTI 2D object | val (Chen et al.) | 3,769 | evaluation only; lists in `configs/kitti_split/` (from OpenPCDet) |
