# CV Final Project
CV Final Project, 2025.

TEMPORARY CONTENT

INSTRUCTIONS TO RUN THE PROJECT
1. cd into build/
2. cmake ..
3. make -> compiles (doesn't compile if there are compiler errors)
4. ./finalProject <parameters> (to define)

Table of Contents
=================

   * [Introduction](#introduction)
   * [Datasets](#what-is-anomaly-detection)
      * [Training](#Training)
      * [Validation](#Validation)
      * [Test](#Test)
   * [Code](#code)
      * [Object Detection and Initial Classification](#Object-Detection-and-Initial-Classification)
      * [Hi-Lo classification and Card Counting](#Hi-Lo-classification-and-Card-Counting)
      * [Visual Overlay](#Visual-Overlay)
      * [Occlusions management](#Occlusions-management)
      * [Metrics](#Metrics)
   * [Results and Discussion](results-and-discussion)

# Introduction
[Read the full proposal](./Cv_final_proposal.pdf).
   
# Datasets
Some datasets of the proposal are used, with the addition of other datasets to have greater variety and robustness.
The program runs using YOLO, which means that the datasets have to follow YOLO's folder structure. See:[YOLO's Dataset Structure for YOLO Classification Tasks](https://docs.ultralytics.com/datasets/classify/).
In this specific case, the dataset structure is defined as:
```
<DATASET_PATH>/
├── Train/
│   └── images
│   │   └── <image1>.jpg
│   │   ...
│   └── labels
│       └── <label1>.txt
│       ...
└── Validation/
│   └── images
│   │   └── <image1>.jpg
│   │   ...
│   └── labels
│       └── <label1>.txt
│       ...
└──Test/
│   └── images
│   │   └── <image1>.jpg
│   │   └──   ...
│   └── labels
│       └── <label1>.txt
│       ...
└── data.yaml
```
Where the labels are .txt files described in YOLO format. See: [Ultralytics YOLO format](https://docs.ultralytics.com/datasets/detect/). Example:
```
obj_class xcenter ycenter width height
```
The data.yaml file is used by the YOLO model to find each part of the dataset and assign each class to its corresponding name. 

No data augmentation has been used for the datasets, as to reduce memory overhead.
## Training
The datasets used are:
   * **The Complete Playing Card Dataset** by **Jay Pradip Shah** on Kaggle. See: https://www.kaggle.com/datasets/jaypradipshah/the-complete-playing-card-dataset.
   * **Playing Cards Object Detection Dataset** by **Andy8744** on Kaggle. See: https://www.kaggle.com/datasets/andy8744/playing-cards-object-detection-dataset.
The two have been merged into one single dataset available here: https://www.kaggle.com/datasets/marcoannunziata/merged-data/settings. The labels have been adjusted to the data.yaml file used for **The Complete Playing Card Dataset**.

## Validation
The **Playing Cards Object Detection Dataset** provides the images used for validation.

## Test
The **Playing Cards Object Detection Dataset** provides some images useful for testing, but the goal is to apply the model on videos from Youtube.
In order to do this, [Label studio](https://github.com/HumanSignal/labelImg) or [Roboflow](https://roboflow.com/) can be used on Youtube clips of poker games.

# Code
## 1.Object Detection and Initial Classification
The YOLO model is trained on the dataset using Kaggle's 2 freely available T4 GPUs. YOLO11s is employed to deliver good performance with enough speed. The training processed has been optimized like this:

```python
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

model = YOLO("/kaggle/working/yolo11s.pt")
results = model.train(data="/kaggle/working/data.yaml", epochs=100, batch=16, augment=False, save_dir="/kaggle/working/output", patience=40, cache=True, workers=8, device=[0, 1])
```

## 2. Hi-Lo classification/Card Counting

## 3. Visual Overlay

## 4. Occlusions management

## 5. Metrics
