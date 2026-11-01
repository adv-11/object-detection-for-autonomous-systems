#!/usr/bin/env bash
# Download and extract the evaluation datasets into ./datasets
#   COCO 2017 val (5,000 images) + instances annotations
#   KITTI 2D object detection, training split only (7,481 labelled images)
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p datasets/_dl datasets/coco datasets/kitti
cd datasets/_dl
curl -L -C - -o val2017.zip                    http://images.cocodataset.org/zips/val2017.zip
curl -L -C - -o annotations_trainval2017.zip   http://images.cocodataset.org/annotations/annotations_trainval2017.zip
curl -L -C - -o data_object_label_2.zip        https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip
curl -L -C - -o data_object_image_2.zip        https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip
cd ../coco
unzip -q -n ../_dl/val2017.zip
unzip -q -n -j ../_dl/annotations_trainval2017.zip annotations/instances_val2017.json -d annotations
cd ../kitti
unzip -q -n ../_dl/data_object_label_2.zip
unzip -q -n ../_dl/data_object_image_2.zip
rm -rf testing   # no public labels for the KITTI test split
echo "COCO val images: $(ls ../coco/val2017 | wc -l)   KITTI train images: $(ls training/image_2 | wc -l)"
