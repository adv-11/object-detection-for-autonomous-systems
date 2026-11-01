# 2D Object Detection: SOTA Evaluation and Cross-Domain Generalization

CMPE 249 (Intelligent Autonomous Systems), Track A: experimental model evaluation.

This repository evaluates modern 2D object detectors from several architecture families on
**COCO val2017** and, without retraining, on the **KITTI** autonomous-driving benchmark. It then
fine-tunes a detector on KITTI to measure how much of the domain gap retraining closes.

> Work in progress. The README gains results, commands and the report link as each experiment lands.

## Environment

- Python 3.12, managed with [uv](https://docs.astral.sh/uv/); PyTorch built for CUDA 12.8 (required by Blackwell / sm_120 GPUs).
- Hardware: NVIDIA GeForce RTX 5060 Laptop GPU (8 GB, 45 W TGP), Windows 11.

```bash
uv sync                      # creates .venv from uv.lock
bash scripts/download_data.sh
```

## Data splits

| Dataset | Split | Images | Notes |
|---|---|---|---|
| COCO 2017 | val2017 | 5,000 | 80 classes, official `instances_val2017.json` |
| KITTI 2D object | train (Chen et al. split) | 3,712 | used only for fine-tuning |
| KITTI 2D object | val (Chen et al. split) | 3,769 | evaluation; split lists in `configs/kitti_split/` (from OpenPCDet) |
