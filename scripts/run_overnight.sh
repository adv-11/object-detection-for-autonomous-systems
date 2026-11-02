#!/usr/bin/env bash
# Overnight extension of the time-boxed study (sequential, single GPU).
# Completed runs are skipped (run_inference.py refuses to overwrite).
set -uo pipefail
cd "$(dirname "$0")/.."
log() { echo "[$(date '+%F %T')] $*" | tee -a logs/command_log.txt; }
step() { log "$*"; "$@" 2>&1 | grep -vE "it/s\]|s/it\]|warn|Warning|symlink" | tail -2 | tee -a logs/command_log.txt; }
FT=runs/finetune

# 1. remaining zero-shot KITTI runs
step uv run python scripts/run_inference.py --model retinanet_r50_v2 --dataset kitti
step uv run python scripts/run_inference.py --model fcos_r50 --dataset kitti
step uv run python scripts/run_inference.py --model yolo11s --dataset kitti --imgsz 1280
step uv run python scripts/run_inference.py --model rtdetr_l --dataset kitti --imgsz 1280
step uv run python scripts/run_inference.py --model grounding_dino_t --dataset kitti --vocab coco
step uv run python scripts/evaluate.py

# 2. full fine-tune at native-like resolution + a 25% label-budget run
step uv run python scripts/finetune_yolo.py --model yolo11s.pt --imgsz 1280 --epochs 20 --batch 4 --fraction 1.0
step uv run python scripts/run_inference.py --model yolo11s --dataset kitti --imgsz 1280 \
     --weights $FT/yolo11s_kitti_1280_e20_f1/weights/last.pt --tag ft
step uv run python scripts/finetune_yolo.py --model yolo11s.pt --imgsz 1280 --epochs 20 --batch 4 --fraction 0.25
step uv run python scripts/run_inference.py --model yolo11s --dataset kitti --imgsz 1280 \
     --weights $FT/yolo11s_kitti_1280_e20_f0.25/weights/last.pt --tag ft25
step uv run python scripts/evaluate.py

# 3. dedicated speed / memory benchmark (GPU otherwise idle)
step uv run python scripts/benchmark_speed.py

# 4. analysis
step uv run python scripts/analyze_failures.py kitti__yolo11s_1280 kitti__yolo11s_1280_ft \
     kitti__yolo_world_l_1280_vcoco kitti__grounding_dino_t_vcoco
step uv run python scripts/summarize.py
log "OVERNIGHT DONE"
