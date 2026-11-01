"""Build KITTI evaluation ground truth (COCO format) and a YOLO-format training copy.

    uv run python scripts/prepare_kitti.py
"""

import collections
import json

from det_eval.data import DATA, KITTI_CAT_ID, build_kitti_coco_gt, export_kitti_yolo


def main():
    out = DATA / "kitti" / "coco_gt_val.json"
    gt = build_kitti_coco_gt("val")
    out.write_text(json.dumps(gt))
    print(f"wrote {out}  ({len(gt['images'])} images, {len(gt['annotations'])} annotations)")

    id2name = {v: k for k, v in KITTI_CAT_ID.items()}
    stats = collections.Counter()
    for a in gt["annotations"]:
        key = a["ignore_reason"] or f"{id2name[a['category_id']]}/{a['kitti_difficulty']}"
        stats[key] += 1
    for k, v in sorted(stats.items()):
        print(f"  {k:28s} {v}")

    yaml_path = export_kitti_yolo(DATA / "kitti_yolo")
    print(f"wrote {yaml_path}")


if __name__ == "__main__":
    main()
