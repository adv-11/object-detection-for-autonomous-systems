#!/usr/bin/env bash
# Full inference sweep: pretrained detectors on COCO val2017 and (zero-shot) KITTI val.
# Every command is appended to logs/command_log.txt with its start time.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
run() {
  echo "[$(date '+%F %T')] uv run python scripts/run_inference.py $*" | tee -a logs/command_log.txt
  uv run python scripts/run_inference.py "$@" 2>&1 | grep -vE "it/s\]|warn|Warning|symlink" | tail -1 | tee -a logs/command_log.txt
}

# --- COCO val2017 (in-domain)
for m in faster_rcnn_r50_v2 retinanet_r50_v2 fcos_r50 yolo11s yolo11m rtdetr_l yolo_world_l grounding_dino_t; do
  run --model "$m" --dataset coco
done

# --- KITTI val, no retraining (cross-domain)
for m in faster_rcnn_r50_v2 retinanet_r50_v2 fcos_r50; do          # native ~1333x402 input
  run --model "$m" --dataset kitti
done
for m in yolo11s yolo11m rtdetr_l; do                               # input resolution ablation
  for s in 640 1280; do run --model "$m" --dataset kitti --imgsz "$s"; done
done
for v in coco kitti kittidesc; do                                   # open-vocab prompt ablation
  run --model yolo_world_l --dataset kitti --imgsz 1280 --vocab "$v"
  run --model grounding_dino_t --dataset kitti --vocab "$v"
done
uv run python scripts/evaluate.py | tee -a logs/command_log.txt
