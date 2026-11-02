"""Uniform wrappers around detectors from three frameworks.

Every detector exposes ``predict(image_rgb: np.ndarray) -> Detections`` with boxes in
original-image pixel coordinates (x1, y1, x2, y2), confidence scores, and integer labels
indexing ``detector.class_names``. Mapping labels to a dataset taxonomy happens later
(see ``mapping.py``), so the raw outputs can be re-scored under different mappings
without rerunning inference.

Shared protocol (applied identically to every model):
* score threshold 0.001; up to ``MAX_RAW`` = 300 raw detections are kept per image and the
  COCO budget of 100 detections per image is applied *after* mapping to the evaluation
  taxonomy (scripts/evaluate.py), so 80-class models are not handicapped on 3-class KITTI;
* multi-label NMS for every NMS-based model (a box may carry several classes, as in
  torchvision's per-class NMS and Ultralytics' own validator); each model keeps its native
  NMS IoU, set-prediction DETRs use none;
* FP32 weights for accuracy runs (cuDNN's default TF32 convolutions stay enabled); FP16 is
  measured separately in the speed benchmark.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field

import numpy as np
import torch

from det_eval.data import ROOT

WEIGHTS = ROOT / "weights"
SCORE_THRESH = 0.001
MAX_RAW = 300  # raw detections saved per image
MAX_DET = 100  # evaluation budget per image, applied after taxonomy mapping


@dataclass
class Detections:
    boxes: np.ndarray  # (N, 4) xyxy
    scores: np.ndarray  # (N,)
    labels: np.ndarray  # (N,) int, indexes Detector.class_names

    def topk(self, k: int = MAX_RAW) -> "Detections":
        order = np.argsort(-self.scores, kind="stable")[:k]
        return Detections(self.boxes[order], self.scores[order], self.labels[order])


@dataclass
class Detector:
    name: str
    family: str
    checkpoint: str
    framework: str
    input_desc: str
    nms_iou: float | None
    class_names: list[str] = field(default_factory=list)
    open_vocab: bool = False
    half: bool = False

    def set_vocabulary(self, names: list[str]) -> None:
        raise NotImplementedError(f"{self.name} is closed-set")

    def predict(self, img: np.ndarray) -> Detections:
        raise NotImplementedError

    def config(self) -> dict:
        return dict(
            name=self.name,
            family=self.family,
            checkpoint=self.checkpoint,
            framework=self.framework,
            input=self.input_desc,
            nms_iou=self.nms_iou,
            score_thresh=SCORE_THRESH,
            max_raw_det=MAX_RAW,
            max_eval_det=MAX_DET,
            nms_multi_label=self.nms_iou is not None,
            precision="fp16" if self.half else "fp32",
            open_vocab=self.open_vocab,
        )


# --------------------------------------------------------------------------- Ultralytics
class UltralyticsDetector(Detector):
    def __init__(self, name, family, checkpoint, imgsz=640, open_vocab=False, half=False):
        from ultralytics import RTDETR, YOLO, YOLOWorld

        cls = RTDETR if "rtdetr" in checkpoint else (YOLOWorld if open_vocab else YOLO)
        path = checkpoint if "/" in checkpoint or "\\" in checkpoint else str(WEIGHTS / checkpoint)
        self.model = cls(path)
        self.imgsz = imgsz
        nms = None if "rtdetr" in checkpoint else 0.7
        super().__init__(
            name=name,
            family=family,
            checkpoint=checkpoint,
            framework="ultralytics",
            input_desc=(f"stretch to {imgsz}x{imgsz} (no aspect preservation)" if "rtdetr" in checkpoint
                        else f"letterbox, long side {imgsz}, stride-padded (rect)"),
            nms_iou=nms,
            class_names=[self.model.names[i] for i in range(len(self.model.names))],
            open_vocab=open_vocab,
            half=half,
        )

    def set_vocabulary(self, names):
        if not self.open_vocab:
            raise NotImplementedError(f"{self.name} is closed-set")
        self.model.set_classes(list(names))
        self.class_names = list(names)

    def predict(self, img):
        # Ultralytics expects BGR numpy arrays; RT-DETR keeps its own NMS-free predictor.
        extra = {} if self.nms_iou is None else {"predictor": _multilabel_predictor()}
        r = self.model.predict(
            np.ascontiguousarray(img[:, :, ::-1]),
            imgsz=self.imgsz,
            conf=SCORE_THRESH,
            iou=self.nms_iou or 0.7,
            max_det=MAX_RAW,
            quantize=16 if self.half else None,  # None -> FP32 for PyTorch weights
            verbose=False,
            **extra,
        )[0]
        b = r.boxes
        return Detections(
            b.xyxy.cpu().numpy().astype(np.float32),
            b.conf.cpu().numpy().astype(np.float32),
            b.cls.cpu().numpy().astype(np.int64),
        ).topk()


@functools.cache  # one class object, so Ultralytics reuses its predictor between calls
def _multilabel_predictor():
    """DetectionPredictor whose NMS keeps several labels per box (as Ultralytics' validator does)."""
    from ultralytics.models.yolo.detect import DetectionPredictor
    from ultralytics.utils import nms

    class MultiLabelPredictor(DetectionPredictor):
        def postprocess(self, preds, img, orig_imgs, **kwargs):
            preds = nms.non_max_suppression(
                preds, self.args.conf, kwargs.pop("iou", self.args.iou), self.args.classes, self.args.agnostic_nms,
                multi_label=True, max_det=self.args.max_det, nc=0,
                end2end=getattr(self.model, "end2end", False),
            )
            if not isinstance(orig_imgs, list):
                from ultralytics.utils import ops
                orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)[..., ::-1]
            return self.construct_results(preds, img, orig_imgs, **kwargs)

    return MultiLabelPredictor


# --------------------------------------------------------------------------- torchvision
class TorchvisionDetector(Detector):
    def __init__(self, name, family, arch, half=False):
        import torchvision.models.detection as tvd

        builders = {
            "fasterrcnn_resnet50_fpn_v2": (tvd.fasterrcnn_resnet50_fpn_v2, tvd.FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1, 0.5),
            "retinanet_resnet50_fpn_v2": (tvd.retinanet_resnet50_fpn_v2, tvd.RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1, 0.5),
            "fcos_resnet50_fpn": (tvd.fcos_resnet50_fpn, tvd.FCOS_ResNet50_FPN_Weights.COCO_V1, 0.6),
        }
        build, weights, nms = builders[arch]
        if arch.startswith("fasterrcnn"):  # RoI-head kwargs; unknown names are silently swallowed
            kwargs = dict(box_score_thresh=SCORE_THRESH, box_detections_per_img=MAX_RAW)
        else:
            kwargs = dict(score_thresh=SCORE_THRESH, detections_per_img=MAX_RAW)
        torch.hub.set_dir(str(WEIGHTS / "torch_hub"))
        self.model = build(weights=weights, **kwargs).eval().cuda()
        head = getattr(self.model, "roi_heads", self.model)
        assert head.detections_per_img == MAX_RAW and head.score_thresh == SCORE_THRESH
        self.dtype = torch.float16 if half else torch.float32
        super().__init__(
            name=name,
            family=family,
            checkpoint=f"torchvision {weights}",
            framework="torchvision",
            input_desc="resize short side 800 / long side <=1333 (GeneralizedRCNNTransform)",
            nms_iou=nms,
            class_names=list(weights.meta["categories"]),  # 91 COCO ids, incl. N/A slots
            half=half,
        )

    @torch.inference_mode()
    def predict(self, img):
        x = torch.from_numpy(np.ascontiguousarray(img)).cuda().permute(2, 0, 1).float().div_(255)
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.half):
            out = self.model([x])[0]
        return Detections(
            out["boxes"].float().cpu().numpy(),
            out["scores"].float().cpu().numpy(),
            out["labels"].cpu().numpy().astype(np.int64),
        ).topk()


# --------------------------------------------------------------------------- Grounding DINO
class GroundingDinoDetector(Detector):
    """Grounding DINO with the official COCO-style scoring.

    The caption is "name1 . name2 . ... ." and each class score is the mean sigmoid
    probability over the class's tokens (the normalised positive map used by the
    official ``PostProcessCocoGrounding``). The top 300 (query, class) pairs are kept
    before the shared 100-detection cap. Vocabularies longer than the 256-token text
    budget are split into chunks, each costing one forward pass.
    """

    MAX_TOKENS = 256

    def __init__(self, name, family, hf_id, half=False):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(hf_id, cache_dir=WEIGHTS / "hf")
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(hf_id, cache_dir=WEIGHTS / "hf").eval().cuda()
        self.half = half
        super().__init__(
            name=name,
            family=family,
            checkpoint=f"huggingface {hf_id}",
            framework="transformers",
            input_desc="resize short side 800 / long side <=1333",
            nms_iou=None,
            open_vocab=True,
            half=half,
        )
        self.chunks: list[tuple[str, list[int], list[np.ndarray]]] = []

    def set_vocabulary(self, names):
        self.class_names = list(names)
        tok = self.processor.tokenizer
        self.chunks = []
        start = 0
        while start < len(names):
            end = len(names)
            while True:
                caption, spans = self._caption(names[start:end])
                enc = tok(caption, return_offsets_mapping=True)
                if len(enc["input_ids"]) <= self.MAX_TOKENS:
                    break
                end -= 1
            offsets = enc["offset_mapping"]
            token_sets = []
            for cs, ce in spans:
                idx = [t for t, (a, b) in enumerate(offsets) if b > a and a >= cs and b <= ce]
                token_sets.append(np.array(idx))
            self.chunks.append((caption, list(range(start, end)), token_sets))
            start = end

    @staticmethod
    def _caption(names):
        caption, spans = "", []
        for n in names:
            s = len(caption)
            caption += n.lower()
            spans.append((s, len(caption)))
            caption += " . "
        return caption.strip(), spans

    @torch.inference_mode()
    def predict(self, img):
        h, w = img.shape[:2]
        all_boxes, all_scores, all_labels = [], [], []
        for caption, class_ids, token_sets in self.chunks:
            inputs = self.processor(images=img, text=caption, return_tensors="pt").to("cuda")
            with torch.autocast("cuda", dtype=torch.float16, enabled=self.half):
                out = self.model(**inputs)
            prob = out.logits[0].float().sigmoid()  # (Q, T)
            cls_prob = torch.stack([prob[:, ts].mean(-1) for ts in token_sets], dim=1)  # (Q, C)
            k = min(300, cls_prob.numel())
            scores, flat = cls_prob.flatten().topk(k)
            q, c = flat // cls_prob.shape[1], flat % cls_prob.shape[1]
            cxcywh = out.pred_boxes[0].float()[q]
            boxes = torch.stack(
                [
                    (cxcywh[:, 0] - cxcywh[:, 2] / 2) * w,
                    (cxcywh[:, 1] - cxcywh[:, 3] / 2) * h,
                    (cxcywh[:, 0] + cxcywh[:, 2] / 2) * w,
                    (cxcywh[:, 1] + cxcywh[:, 3] / 2) * h,
                ],
                dim=1,
            )
            keep = scores >= SCORE_THRESH
            all_boxes.append(boxes[keep].cpu().numpy())
            all_scores.append(scores[keep].cpu().numpy())
            all_labels.append(np.asarray(class_ids)[c[keep].cpu().numpy()])
        return Detections(
            np.concatenate(all_boxes).astype(np.float32),
            np.concatenate(all_scores).astype(np.float32),
            np.concatenate(all_labels).astype(np.int64),
        ).topk()


# --------------------------------------------------------------------------- registry
REGISTRY = {
    # two-stage CNN
    "faster_rcnn_r50_v2": lambda **kw: TorchvisionDetector("Faster R-CNN R50-FPN v2", "two-stage CNN", "fasterrcnn_resnet50_fpn_v2", **kw),
    # one-stage dense CNNs
    "retinanet_r50_v2": lambda **kw: TorchvisionDetector("RetinaNet R50-FPN v2", "one-stage anchor-based", "retinanet_resnet50_fpn_v2", **kw),
    "fcos_r50": lambda **kw: TorchvisionDetector("FCOS R50-FPN", "one-stage anchor-free", "fcos_resnet50_fpn", **kw),
    # real-time YOLO
    "yolo11s": lambda imgsz=640, **kw: UltralyticsDetector(f"YOLO11s@{imgsz}", "real-time YOLO", "yolo11s.pt", imgsz, **kw),
    "yolo11m": lambda imgsz=640, **kw: UltralyticsDetector(f"YOLO11m@{imgsz}", "real-time YOLO", "yolo11m.pt", imgsz, **kw),
    # real-time DETR
    "rtdetr_l": lambda imgsz=640, **kw: UltralyticsDetector(f"RT-DETR-L@{imgsz}", "real-time DETR", "rtdetr-l.pt", imgsz, **kw),
    # open-vocabulary
    "yolo_world_l": lambda imgsz=640, **kw: UltralyticsDetector(f"YOLO-World-L v2@{imgsz}", "open-vocabulary (CNN)", "yolov8l-worldv2.pt", imgsz, open_vocab=True, **kw),
    "grounding_dino_t": lambda **kw: GroundingDinoDetector("Grounding DINO-T", "open-vocabulary (DETR)", "IDEA-Research/grounding-dino-tiny", **kw),
}


def build(key: str, **kwargs) -> Detector:
    WEIGHTS.mkdir(exist_ok=True)
    return REGISTRY[key](**kwargs)
