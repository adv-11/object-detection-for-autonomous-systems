#!/usr/bin/env bash
# Time-boxed KITTI zero-shot set actually used in the report (see README "Scope").
set -uo pipefail
cd "$(dirname "$0")/.."
run() {
  echo "[$(date '+%F %T')] uv run python scripts/run_inference.py $*" | tee -a logs/command_log.txt
  uv run python scripts/run_inference.py "$@" 2>&1 | grep -vE "it/s\]|warn|Warning|symlink" | tail -1 | tee -a logs/command_log.txt
}
run --model yolo11s --dataset kitti --imgsz 640
run --model yolo11m --dataset kitti --imgsz 640
run --model yolo11m --dataset kitti --imgsz 1280
run --model rtdetr_l --dataset kitti --imgsz 640
run --model yolo_world_l --dataset kitti --imgsz 1280 --vocab coco
run --model yolo_world_l --dataset kitti --imgsz 1280 --vocab kitti
run --model yolo_world_l --dataset kitti --imgsz 1280 --vocab kittidesc
echo "[$(date '+%F %T')] REDUCED SWEEP DONE" | tee -a logs/command_log.txt
