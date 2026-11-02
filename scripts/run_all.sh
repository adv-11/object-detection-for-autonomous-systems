#!/usr/bin/env bash
# End-to-end experiment queue (sequential, single GPU). Safe to re-run: completed
# inference runs are skipped (run_inference.py refuses to overwrite) and evaluation
# only re-scores raw files that are newer than their metrics.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
log() { echo "[$(date '+%F %T')] $*" | tee -a logs/command_log.txt; }
step() { log "$*"; "$@" 2>&1 | grep -vE "it/s\]|s/it\]|warn|Warning|symlink" | tail -3 | tee -a logs/command_log.txt; }

# 1. pretrained detectors: COCO val2017 + zero-shot KITTI val (incl. resolution and prompt ablations)
step bash scripts/run_sweep.sh
step uv run python scripts/evaluate.py

# 2. fine-tuning on KITTI train (YOLO11s @1280; full train set and a 25% label-budget run)
FT=runs/finetune
step uv run python scripts/finetune_yolo.py --model yolo11s.pt --imgsz 1280 --epochs 20 --batch 4 --fraction 1.0
step uv run python scripts/run_inference.py --model yolo11s --dataset kitti --imgsz 1280 \
     --weights $FT/yolo11s_kitti_1280_e20_f1/weights/last.pt --tag ft
step uv run python scripts/finetune_yolo.py --model yolo11s.pt --imgsz 1280 --epochs 20 --batch 4 --fraction 0.25
step uv run python scripts/run_inference.py --model yolo11s --dataset kitti --imgsz 1280 \
     --weights $FT/yolo11s_kitti_1280_e20_f0.25/weights/last.pt --tag ft25
step uv run python scripts/evaluate.py

# 3. speed / memory benchmark, failure analysis, tables, figures
step uv run python scripts/benchmark_speed.py
step uv run python scripts/analyze_failures.py kitti__faster_rcnn_r50_v2 kitti__rtdetr_l_1280 \
     kitti__yolo11s_1280 kitti__yolo11s_1280_ft kitti__grounding_dino_t_vkitti
step uv run python scripts/summarize.py
step uv run python scripts/make_figures.py
log "ALL DONE"
