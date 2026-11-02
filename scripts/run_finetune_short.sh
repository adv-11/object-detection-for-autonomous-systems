#!/usr/bin/env bash
# Time-boxed fine-tune: YOLO11s, 640 px, 5 epochs, full KITTI train; then one KITTI val evaluation.
set -uo pipefail
cd "$(dirname "$0")/.."
echo "[$(date '+%F %T')] uv run python scripts/finetune_yolo.py --model yolo11s.pt --imgsz 640 --epochs 5 --batch 16" | tee -a logs/command_log.txt
uv run python scripts/finetune_yolo.py --model yolo11s.pt --imgsz 640 --epochs 5 --batch 16 > logs/finetune_stdout.txt 2>&1
tail -1 logs/finetune_stdout.txt | tee -a logs/command_log.txt
echo "[$(date '+%F %T')] uv run python scripts/run_inference.py --model yolo11s --dataset kitti --imgsz 640 --weights runs/finetune/yolo11s_kitti_640_e5_f1/weights/last.pt --tag ft" | tee -a logs/command_log.txt
uv run python scripts/run_inference.py --model yolo11s --dataset kitti --imgsz 640 \
  --weights runs/finetune/yolo11s_kitti_640_e5_f1/weights/last.pt --tag ft 2>&1 | tail -1 | tee -a logs/command_log.txt
echo "[$(date '+%F %T')] FINETUNE DONE" | tee -a logs/command_log.txt
