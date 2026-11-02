---
title: "From COCO to the Road: A Reproducible Evaluation of Modern 2D Object Detectors Under Domain Shift"
subtitle: "CMPE 249 Intelligent Autonomous Systems · Track A: Experimental Model Evaluation"
author: "Advait Shinde · San José State University · Fall 2026"
date: "Code, configs and command log: github.com/adv-11/object-detection-for-autonomous-systems"
---

<div class="abstract">

ABSTRACT_PLACEHOLDER

</div>

# 1. Introduction

Detectors are trained and ranked on COCO [1], yet an autonomous vehicle sees a different world: wide 1242×375 frames, objects at
distance, heavy occlusion, and a label taxonomy (Car / Pedestrian / Cyclist) that does not match COCO's 80 classes. This report asks
whether published COCO accuracy predicts behaviour on a driving benchmark, KITTI [2], when the detector is **not retrained**, and
what it takes to close the gap. We evaluate **eight model configurations from six detector families** — two-stage CNN, one-stage
anchor-based, one-stage anchor-free, real-time YOLO, real-time DETR and open-vocabulary detectors — under one shared protocol, and
answer four questions:

* **RQ1 (reproduction).** Do our numbers match the published COCO results, and how do the families trade accuracy for speed?
* **RQ2 (cross-dataset generalization).** How much accuracy survives the COCO → KITTI shift, and which classes and object scales degrade most?
* **RQ3 (closed-set vs. open-vocabulary).** Can a text-prompted detector use KITTI's own class names instead of a hand-made mapping?
* **RQ4 (fine-tuning).** How much of the gap does a short fine-tune on KITTI train recover?

Before any result was recorded, the evaluation code was put through an adversarial review (Section 3.4) that found and fixed a
KITTI ignore-rule bug that would have inflated every KITTI number.

# 2. Models and datasets

## 2.1 Detectors

All checkpoints are the official COCO-trained weights (Table 1 lists the reproduced and published accuracy). **Faster R-CNN** [3]
with FPN [4] is the two-stage reference; we use torchvision's improved "v2" recipe [5]. **RetinaNet** [6] (v2 weights trained with
the ATSS recipe [7]) and **FCOS** [8] represent one-stage dense detectors, anchor-based and anchor-free. **YOLO11** [9], [10]
(s and m) is the current Ultralytics real-time CNN, a descendant of YOLO [11]. **RT-DETR-L** [12] is a real-time member of the
DETR family [13], [14]: set prediction with no NMS. **YOLO-World-L v2** [15] and **Grounding DINO-T** [16] are open-vocabulary
detectors that score boxes against text prompts; neither was trained on COCO annotations, so their COCO numbers are
"zero-shot", with the caveat that their grounding data (GoldG) shares images with COCO's sources [15], [16].

## 2.2 Datasets, splits and taxonomy mapping

**COCO val2017** [1] (5,000 images, 80 classes) is the in-domain test. **KITTI 2D object** [2] has 7,481 labelled images; we use the
standard split of Chen et al. [17] (3,712 train / 3,769 val, no shared sequences; lists from OpenPCDet [18]). KITTI val is used only
for evaluation and is never seen during fine-tuning or model selection. Following the official benchmark [19] we score
**Car, Pedestrian and Cyclist**, and reproduce the devkit's ignore rules: `DontCare` regions absorb any detection that falls inside them
(intersection over detection area); `Van` (for Car) and `Person_sitting` (for Pedestrian) are *neighbour classes* that absorb at most
one detection matched by IoU; `Truck`, `Tram` and `Misc` are not evaluated, so a Car detection on a truck is a false positive.

| KITTI class | Source in a COCO-vocabulary detector | Mismatch |
|:---|:---|:---|
| Car | `car` | 1:1. COCO has no "van"; KITTI vans are ignore regions for Car. |
| Pedestrian | `person` that is not riding | COCO `person` also covers riders and sitting people. |
| Cyclist | `person` + `bicycle` fused into one box | COCO boxes the rider and the bike separately; a KITTI Cyclist box encloses both. |
| *(dropped)* | the other 77 COCO classes | `truck`, `bus`, `train` have no evaluated KITTI counterpart. |

<p class="tcap">Table 2. COCO → KITTI taxonomy mapping (src/det_eval/mapping.py).</p>

The Cyclist mapping is the real taxonomy problem. **direct** maps every `bicycle` box to Cyclist — the box covers only the bike and
riders become Pedestrian false positives. **fused** (default) pairs a person with a bicycle when they overlap horizontally by ≥50% of
the narrower box, the person's top is above the bike's, the person's feet lie in the lower half of the bike's vertical extent, and the
person is 0.8–3× the bike's height. The pair becomes one Cyclist with the union box and score s<sub>person</sub>·s<sub>bike</sub>,
the person stays a Pedestrian candidate with score s<sub>person</sub>·(1−s<sub>bike</sub>) (a soft re-label that needs no
threshold), and unpaired bikes (parked) are dropped. **fused-loose** omits the feet/height tests. The rules were fixed from synthetic
cases before any full KITTI evaluation and are reported side by side rather than tuned on val.

# 3. Experimental setup

## 3.1 Hardware and software

NVIDIA GeForce RTX 5060 Laptop GPU (Blackwell, 8 GB, **45 W power limit**), Windows 11, Python 3.12, PyTorch 2.11.0 + CUDA 12.8,
torchvision 0.26.0 [20], Ultralytics 8.4.197, Transformers 5.22.1, pycocotools 2.0 [21]. Every raw-prediction file stores its full
environment, detector configuration and command line.

## 3.2 Inference protocol

The same protocol is applied to every model so that differences come from the models, not the settings:

* score threshold **0.001**; up to 300 raw detections are kept and the COCO budget of **100 detections per image** is applied *after*
  mapping to the evaluation taxonomy (otherwise 80-class models would waste slots on classes KITTI ignores);
* each model's native NMS IoU (YOLO 0.7, Faster R-CNN / RetinaNet 0.5, FCOS 0.6) with **multi-label** NMS; none for the DETRs;
* **batch 1, FP32**, images decoded outside the timed region; latency = median per-image time of the full `predict()` call
  (pre-processing, network, NMS, post-processing) with CUDA synchronisation;
* native input pipelines: torchvision and Grounding DINO resize the short side to 800 px (long side ≤ 1333); YOLO models letterbox
  the long side to 640 (or 1280) with stride padding; RT-DETR stretches to a 640×640 square.

## 3.3 Metrics

**COCO-style** (pycocotools): AP@[.50:.95], AP50, AP75, AP for small (<32² px), medium and large objects, AR@100 and per-class AP.
On KITTI the same metrics are computed for the three classes with every labelled object counted. **KITTI-style**: AP with 40
recall points (AP|R40, the official metric since 2019 [22]) at class-specific IoU (Car 0.7, Pedestrian/Cyclist 0.5) for the
Easy / Moderate / Hard levels (min. height 40/25/25 px, max. occlusion 0/1/2, max. truncation 15/30/50 %); objects outside a level
become ignore regions and detections shorter than its minimum height are discarded.

## 3.4 Verification of the evaluation code

Because every number flows through custom mapping and evaluation code, a 47-agent adversarial review (five reviewers covering metric
semantics, mapping, detector wrappers, data handling and protocol; two independent refuters per finding, one reproducing on CPU and
one reading library source) was run before the final experiments. It upheld 19 of 21 findings. The most important: our first version
wrote Van / Person_sitting / difficulty-filtered objects as COCO "crowd" regions, which match by intersection-over-detection and absorb
*unlimited* detections, whereas the devkit matches them by IoU, one detection each. Duplicate or partial boxes on vans and occluded
cars were silently forgiven, inflating KITTI AP. We now use a `COCOeval` subclass that honours a separate `kitti_ignore` flag. Other
fixes: multi-label NMS for YOLO, applying the detection budget after mapping, and a run-naming collision. Eighteen regression tests in
`tests/` encode the reviewers' reproduction cases (e.g. two detections on one van must give AP 50, not 100).

## 3.5 Commands

```bash
uv sync && bash scripts/download_data.sh && uv run python scripts/prepare_kitti.py
uv run python scripts/run_inference.py --model yolo11m --dataset coco                     # COCO val2017
uv run python scripts/run_inference.py --model yolo11m --dataset kitti --imgsz 1280       # zero-shot KITTI
uv run python scripts/run_inference.py --model yolo_world_l --dataset kitti --imgsz 1280 --vocab kitti
uv run python scripts/finetune_yolo.py --model yolo11s.pt --imgsz 640 --epochs 5 --batch 16
uv run python scripts/run_inference.py --model yolo11s --dataset kitti --imgsz 640 \
       --weights runs/finetune/yolo11s_kitti_640_e5_f1/weights/last.pt --tag ft
uv run python scripts/evaluate.py && uv run python scripts/analyze_failures.py <runs> && uv run python scripts/make_figures.py
uv run pytest tests
```

Every executed command, with its start time, wall-clock time, throughput and peak memory, is in `logs/command_log.txt`; per-run
configurations are in `results/metrics/*.json` (`meta.detector`, `meta.environment`, `meta.command`).

# 4. Results

## 4.1 COCO val2017 (in-domain)

<!--TABLE:coco-->
<p class="tcap">Table 1. COCO val2017, 5,000 images. Publ. AP from the official torchvision / Ultralytics documentation and the papers
(*zero-shot numbers from [15], [16]). ms/img: median batch-1 FP32 latency of the full predict() call; Mem: peak allocated GPU memory.</p>

COCO_RESULTS_PLACEHOLDER

<figure><img src="../figures/fig_speed_accuracy.png" style="width:78%"><figcaption>Figure 1. Speed–accuracy on COCO val2017 (batch 1, FP32, 45 W laptop GPU).</figcaption></figure>

## 4.2 KITTI val without retraining (cross-domain)

<!--TABLE:kitti-->
<p class="tcap">Table 3. KITTI val (3,769 images), three classes, default (fused) mapping. Columns 3–8: COCO-style AP@[.50:.95]; columns
9–11: Moderate AP|R40 per class; columns 12–14: mean AP|R40 per difficulty. † RT-DETR stretches the frame to a square. Bold: best zero-shot value.
The last row is the fine-tuned model (Section 5.4).</p>

KITTI_RESULTS_PLACEHOLDER

# 5. Technical analyses

ANALYSES_PLACEHOLDER

# 6. Failure analysis

FAILURE_PLACEHOLDER

# 7. Limitations

LIMITATIONS_PLACEHOLDER

# 8. Conclusions

CONCLUSIONS_PLACEHOLDER

# References

<ol class="refs">
<li>T.-Y. Lin et al., "Microsoft COCO: Common Objects in Context," in <i>Proc. ECCV</i>, 2014, pp. 740–755. arXiv:1405.0312.</li>
<li>A. Geiger, P. Lenz, and R. Urtasun, "Are We Ready for Autonomous Driving? The KITTI Vision Benchmark Suite," in <i>Proc. IEEE CVPR</i>, 2012, pp. 3354–3361.</li>
<li>S. Ren, K. He, R. Girshick, and J. Sun, "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks," in <i>Advances in NIPS 28</i>, 2015. arXiv:1506.01497.</li>
<li>T.-Y. Lin et al., "Feature Pyramid Networks for Object Detection," in <i>Proc. IEEE CVPR</i>, 2017. arXiv:1612.03144.</li>
<li>Y. Li et al., "Benchmarking Detection Transfer Learning with Vision Transformers," arXiv:2111.11429, 2021.</li>
<li>T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollár, "Focal Loss for Dense Object Detection," in <i>Proc. IEEE ICCV</i>, 2017. arXiv:1708.02002.</li>
<li>S. Zhang et al., "Bridging the Gap Between Anchor-Based and Anchor-Free Detection via Adaptive Training Sample Selection," in <i>Proc. IEEE/CVF CVPR</i>, 2020. arXiv:1912.02424.</li>
<li>Z. Tian, C. Shen, H. Chen, and T. He, "FCOS: Fully Convolutional One-Stage Object Detection," in <i>Proc. IEEE/CVF ICCV</i>, 2019. arXiv:1904.01355.</li>
<li>G. Jocher and J. Qiu, "Ultralytics YOLO11," software v11.0.0, 2024. Docs: https://docs.ultralytics.com/models/yolo11/; code: https://github.com/ultralytics/ultralytics.</li>
<li>R. Khanam and M. Hussain, "YOLOv11: An Overview of the Key Architectural Enhancements," arXiv:2410.17725, 2024.</li>
<li>J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, "You Only Look Once: Unified, Real-Time Object Detection," in <i>Proc. IEEE CVPR</i>, 2016. arXiv:1506.02640.</li>
<li>Y. Zhao et al., "DETRs Beat YOLOs on Real-time Object Detection," in <i>Proc. IEEE/CVF CVPR</i>, 2024, pp. 16965–16974. arXiv:2304.08069.</li>
<li>N. Carion et al., "End-to-End Object Detection with Transformers," in <i>Proc. ECCV</i>, 2020. arXiv:2005.12872.</li>
<li>X. Zhu, W. Su, L. Lu, B. Li, X. Wang, and J. Dai, "Deformable DETR: Deformable Transformers for End-to-End Object Detection," in <i>Proc. ICLR</i>, 2021. arXiv:2010.04159.</li>
<li>T. Cheng, L. Song, Y. Ge, W. Liu, X. Wang, and Y. Shan, "YOLO-World: Real-Time Open-Vocabulary Object Detection," in <i>Proc. IEEE/CVF CVPR</i>, 2024. arXiv:2401.17270.</li>
<li>S. Liu et al., "Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection," in <i>Proc. ECCV</i>, 2024. arXiv:2303.05499.</li>
<li>X. Chen et al., "3D Object Proposals for Accurate Object Class Detection," in <i>Advances in NIPS 28</i>, 2015.</li>
<li>OpenPCDet Development Team, "OpenPCDet: An Open-source Toolbox for 3D Object Detection from Point Clouds," 2020. https://github.com/open-mmlab/OpenPCDet.</li>
<li>A. Geiger, P. Lenz, and R. Urtasun, "KITTI 2D Object Detection Benchmark and devkit," https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d (accessed Nov. 2026).</li>
<li>TorchVision maintainers and contributors, "TorchVision: PyTorch's Computer Vision Library," 2016; detection model docs: https://docs.pytorch.org/vision/stable/models.html.</li>
<li>COCO Consortium, "COCO API (pycocotools)," https://github.com/cocodataset/cocoapi; "Detection evaluation," https://cocodataset.org/#detection-eval.</li>
<li>A. Simonelli, S. Rota Bulò, L. Porzi, M. López-Antequera, and P. Kontschieder, "Disentangling Monocular 3D Object Detection," in <i>Proc. IEEE/CVF ICCV</i>, 2019. arXiv:1905.12365.</li>
<li>D. Bolya, S. Foley, J. Hays, and J. Hoffman, "TIDE: A General Toolbox for Identifying Object Detection Errors," in <i>Proc. ECCV</i>, 2020. arXiv:2008.08115.</li>
<li>Y. Wang et al., "Train in Germany, Test in the USA: Making 3D Object Detectors Generalize," in <i>Proc. IEEE/CVF CVPR</i>, 2020. arXiv:2005.08139.</li>
<li>Hugging Face, "Grounding DINO — Transformers documentation," https://huggingface.co/docs/transformers/model_doc/grounding-dino; checkpoint IDEA-Research/grounding-dino-tiny.</li>
</ol>
