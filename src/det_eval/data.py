"""Dataset utilities: KITTI label parsing, KITTI -> COCO-format ground truth, YOLO export.

Ignore handling follows the official KITTI devkit (cpp/evaluate_object.cpp), which has two
different kinds of "don't penalise" ground truth, reproduced here with two flags:

* ``DontCare`` regions -> ``iscrowd=1``. pycocotools matches crowd boxes by intersection over
  the *detection* area and lets one region absorb any number of detections, exactly like the
  devkit's ``boxoverlap(det, dc, 0)`` DontCare pass. Written once per evaluated class.
* Neighbour classes (``Van`` for ``Car``, ``Person_sitting`` for ``Pedestrian``) and, at
  evaluation time, objects failing a difficulty level -> ``iscrowd=0, kitti_ignore=1``.
  The devkit matches these by ordinary IoU at the class threshold and each absorbs at most
  one detection; ``metrics.KittiEval`` honours the flag so COCOeval does the same.
* ``Truck``, ``Tram`` and ``Misc`` are not evaluated; a Car detection on a truck is an FP,
  exactly as in the official devkit.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "datasets"
COCO_DIR = DATA / "coco"
COCO_IMAGES = COCO_DIR / "val2017"
COCO_ANN = COCO_DIR / "annotations" / "instances_val2017.json"
KITTI_DIR = DATA / "kitti" / "training"
KITTI_SPLITS = ROOT / "configs" / "kitti_split"

# Evaluated KITTI classes (the three classes of the official 2D benchmark).
KITTI_EVAL_CLASSES = ["Car", "Pedestrian", "Cyclist"]
KITTI_CAT_ID = {name: i + 1 for i, name in enumerate(KITTI_EVAL_CLASSES)}
# Neighbouring classes treated as ignore regions for the evaluated class.
KITTI_NEIGHBOUR = {"Van": "Car", "Person_sitting": "Pedestrian"}
# Classes used for fine-tuning (DontCare and Misc are dropped).
KITTI_TRAIN_CLASSES = ["Car", "Van", "Truck", "Pedestrian", "Person_sitting", "Cyclist", "Tram"]

# Official KITTI difficulty definitions (devkit cpp/evaluate_object.cpp).
DIFFICULTY = {
    "easy": dict(min_height=40, max_occlusion=0, max_truncation=0.15),
    "moderate": dict(min_height=25, max_occlusion=1, max_truncation=0.30),
    "hard": dict(min_height=25, max_occlusion=2, max_truncation=0.50),
}


@dataclass
class KittiObject:
    cls: str
    truncation: float
    occlusion: int
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels

    @property
    def height(self) -> float:
        return self.box[3] - self.box[1]

    def difficulty(self) -> str | None:
        """Easiest difficulty level the object satisfies, or None if it fails all three."""
        for level, d in DIFFICULTY.items():
            if (
                self.height >= d["min_height"]
                and self.occlusion <= d["max_occlusion"]
                and self.truncation <= d["max_truncation"]
            ):
                return level
        return None


def read_split(name: str) -> list[str]:
    return (KITTI_SPLITS / f"{name}.txt").read_text().split()


def kitti_image_path(stem: str) -> Path:
    return KITTI_DIR / "image_2" / f"{stem}.png"


def parse_kitti_label(stem: str) -> list[KittiObject]:
    objs = []
    for line in (KITTI_DIR / "label_2" / f"{stem}.txt").read_text().splitlines():
        p = line.split()
        if not p:
            continue
        objs.append(
            KittiObject(
                cls=p[0],
                truncation=float(p[1]),
                occlusion=int(p[2]),
                box=tuple(float(v) for v in p[4:8]),
            )
        )
    return objs


def coco_image_records() -> list[dict]:
    return json.loads(COCO_ANN.read_text())["images"]


def build_kitti_coco_gt(split: str = "val") -> dict:
    """COCO-format ground truth for a KITTI split.

    Every annotation carries ``kitti_difficulty`` (easy/moderate/hard/None) so that
    difficulty-specific protocols can be derived later without re-reading labels.
    DontCare is ``iscrowd=1``; neighbour classes are ``kitti_ignore=1`` (see module doc).
    """
    images, anns = [], []
    ann_id = 1

    def add(img_id, cat, box, crowd, reason=None, diff=None, occ=None, trunc=None, ignore=False):
        nonlocal ann_id
        x1, y1, x2, y2 = box
        w, h = max(0.0, x2 - x1), max(0.0, y2 - y1)
        anns.append(
            dict(
                id=ann_id,
                image_id=img_id,
                category_id=cat,
                bbox=[x1, y1, w, h],
                area=w * h,
                iscrowd=int(crowd),
                kitti_ignore=int(ignore or crowd),
                ignore_reason=reason,
                kitti_difficulty=diff,
                occlusion=occ,
                truncation=trunc,
            )
        )
        ann_id += 1

    for stem in read_split(split):
        img_id = int(stem)
        w, h = Image.open(kitti_image_path(stem)).size
        images.append(dict(id=img_id, file_name=f"{stem}.png", width=w, height=h))
        for o in parse_kitti_label(stem):
            if o.cls in KITTI_CAT_ID:
                add(img_id, KITTI_CAT_ID[o.cls], o.box, False, None, o.difficulty(), o.occlusion, o.truncation)
            elif o.cls in KITTI_NEIGHBOUR:
                add(img_id, KITTI_CAT_ID[KITTI_NEIGHBOUR[o.cls]], o.box, False, f"neighbour:{o.cls}", ignore=True)
            elif o.cls == "DontCare":
                for cat in KITTI_CAT_ID.values():
                    add(img_id, cat, o.box, True, "dontcare")
            # Truck / Tram / Misc: not evaluated, detections on them are false positives.

    cats = [dict(id=i, name=n) for n, i in KITTI_CAT_ID.items()]
    return dict(images=images, annotations=anns, categories=cats, info=dict(split=split))


def export_kitti_yolo(out_dir: Path, splits=("train", "val")) -> Path:
    """Write a YOLO-format copy of KITTI (labels only; images are hard-linked when possible)."""
    import os
    import shutil

    for split in splits:
        img_dir = out_dir / "images" / split
        lbl_dir = out_dir / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for stem in read_split(split):
            src = kitti_image_path(stem)
            dst = img_dir / src.name
            if not dst.exists():
                try:
                    os.link(src, dst)
                except OSError:
                    shutil.copy2(src, dst)
            w, h = Image.open(src).size
            lines = []
            for o in parse_kitti_label(stem):
                if o.cls not in KITTI_TRAIN_CLASSES:
                    continue
                x1, y1, x2, y2 = o.box
                cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
                bw, bh = (x2 - x1) / w, (y2 - y1) / h
                lines.append(f"{KITTI_TRAIN_CLASSES.index(o.cls)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            (lbl_dir / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))

    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(KITTI_TRAIN_CLASSES))
    yaml_path = out_dir / "kitti.yaml"
    yaml_path.write_text(f"path: {out_dir.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n{names}\n")
    # Fine-tuning config: Ultralytics always validates once after the last epoch, so point
    # 'val' at 300 *training* images (logging only) to keep KITTI val out of training entirely.
    subset = sorted(read_split("train"))[::12][:300]
    (out_dir / "train_subset300.txt").write_text("".join(f"./images/train/{s}.png\n" for s in subset))
    ft_yaml = out_dir / "kitti_finetune.yaml"
    ft_yaml.write_text(f"path: {out_dir.as_posix()}\ntrain: images/train\nval: train_subset300.txt\nnames:\n{names}\n")
    return yaml_path
