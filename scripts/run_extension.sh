#!/usr/bin/env bash
# Extension study ("data, not vocabulary?"), sequential on one GPU. Completed runs are skipped.
#  A. zero-shot: D-FINE-S/M trained on COCO vs Objects365->COCO (same architecture), YOLO26s/m, YOLO-World-S
#  B. label budgets: YOLO11s vs YOLO-World-S fine-tuned at 5% / 25% of KITTI train, 1280 px, 3 seeds
set -uo pipefail
cd "$(dirname "$0")/.."
log() { echo "[$(date '+%F %T')] $*" | tee -a logs/command_log.txt; }
step() { log "$*"; "$@" 2>&1 | grep -vE "it/s\]|s/it\]|warn|Warning|symlink" | tail -2 | tee -a logs/command_log.txt; }
inf() { step uv run python scripts/run_inference.py "$@"; }
FT=runs/finetune

# ---- A. zero-shot
for m in dfine_s_coco dfine_s_o365 dfine_m_coco dfine_m_o365 yolo26s yolo26m yolo_world_s; do inf --model $m --dataset coco; done
for m in dfine_s_coco dfine_s_o365 dfine_m_coco dfine_m_o365 yolo26s yolo26m; do
  for s in 640 1280; do inf --model $m --dataset kitti --imgsz $s; done
done
for v in coco kitti; do inf --model yolo_world_s --dataset kitti --imgsz 1280 --vocab $v; done
step uv run python scripts/evaluate.py

# ---- B. label budgets x seeds (YOLO11s 25% seed 0 already exists: kitti__yolo11s_1280_ft25)
ft() {  # model-file model-key fraction seed tag
  step uv run python scripts/finetune_yolo.py --model $1 --imgsz 1280 --epochs 20 --batch 4 --fraction $3 --seed $4
  local name="${1%.pt}_kitti_1280_e20_f$(printf %g $3)"; [ "$4" != "0" ] && name="${name}_s$4"
  inf --model $2 --dataset kitti --imgsz 1280 --weights $FT/$name/weights/last.pt --tag $5
}
for seed in 0 1 2; do
  ft yolo11s.pt yolo11s 0.05 $seed ft5_s$seed
  ft yolov8s-worldv2.pt yolo_world_s 0.05 $seed ft5_s$seed
  [ "$seed" != "0" ] && ft yolo11s.pt yolo11s 0.25 $seed ft25_s$seed
  ft yolov8s-worldv2.pt yolo_world_s 0.25 $seed ft25_s$seed
done
step uv run python scripts/evaluate.py
step uv run python scripts/benchmark_speed.py --ext
ft yolov8s-worldv2.pt yolo_world_s 1.0 0 ft            # full-data reference (last; drop-able)
step uv run python scripts/evaluate.py
log "EXTENSION DONE"
