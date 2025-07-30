# CV Final Project
CV Final Project, 2025.

Table of Contents
=================

   * [Introduction](#introduction)
   * [Instructions](#Instructions)
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
   
# Instructions
## Requirements
* CMake version:
* OpenCV version:
* ONNXRuntime version: 1.21.0. The binaries are already bundled inside the project in the [external](./external) folder and **CMake is already configured to find the binaries either in this folder or in the system's directories**. If there are problems on using this library with CMake on Linux, you can manually install it:

**On LINUX**
Option 1 - Automatic (system-wide) installation using the bash script:
* Run [onnxruntime_Linux_install.sh](./external/onnxruntime_Linux_install.sh)
* See section: [Runnning the project](#Running-the-project)

Option 2 - Manual (global) installation:
* Extract onnxruntime-linux-x64-1.21.0.tgz
* Copy the .so files of lib in /usr/local/lib64/
* Copy the .cmake files in /usr/local/lib64/cmake/onnxruntime/
* Copy the include/onnxruntime/ folder in /usr/local/include/
* update the libraries cache running ldconfig

**On WINDOWS**

## Running the project
Make sure to put the videos to use for testing in the data/test/ directory and the .onnx file of the model in data/model/ directory along with a .txt file containing the labels, one label per line.
To run the project:
1. cd into build/
2. cmake ..
3. make -> compiles (doesn't compile if there are compiler errors)
4. ./finalProject <image_to_test> (WIP, to define)

   
# Datasets
Some datasets of the proposal are used, with the addition of other datasets to have greater variety and robustness.
The program runs using YOLO, which means that the datasets have to follow YOLO's folder structure. See: [YOLO's Dataset Structure for YOLO Classification Tasks](https://docs.ultralytics.com/datasets/classify/).
In this specific case, the dataset structure is defined as:
```
<DATASET_PATH>/
├── Train/
│   └── images
│   │   └── <image1>.jpg
│   │   └──   ...
│   └── labels
│       └── <label1>.txt
│       └──   ...
└── Validation/
│   └── images
│   │   └── <image1>.jpg
│   │   └──   ...
│   └── labels
│       └── <label1>.txt
│       └──   ...
└──Test/
│   └── images
│   │   └── <image1>.jpg
│   │   └──   ...
│   └── labels
│       └── <label1>.txt
│       └──   ...
└── data.yaml
```
Where The data.yaml file is used by the YOLO model to find each part of the dataset and assign each class to its corresponding name. 

The labels are .txt files described in YOLO format. See: [Ultralytics YOLO format](https://docs.ultralytics.com/datasets/detect/) and https://labelformat.com/formats/object-detection/yolov11/. An example of a .txt file containing labels for multiple cards is:
```
<obj_class1> <xcenter1> <ycenter1> <width1> <height1>
<obj_class2> <xcenter2> <ycenter2> <width2> <height2>
...
```
Where each row is a bounding box enclosing the suit and the rank on the corners of a poker card. This means that at most 2 bounding boxes can be found for the same card, which makes it easier to find in case of partial occlusions.

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
### Training of the model
The YOLO model is trained on the dataset using Kaggle's 2 freely available T4 GPUs. YOLO11s is employed to deliver good performance with enough speed. The training process has been optimized as in the following code:

```python
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

model = YOLO("/kaggle/working/yolo11s.pt")
results = model.train(data="/kaggle/working/data.yaml", epochs=100, batch=16, augment=False, save_dir="/kaggle/working/output", patience=40, cache=True, workers=8, device=[0, 1])
```
The model detects a bounding box enclosing the suit and the rank on the corners of a poker card. This means that at most 2 bounding boxes can be found for the same card, which makes it easier to find in case of partial occlusions. The bounding boxes are then classified as the suit and the rank they enclose.

### Inference


## 2. Hi-Lo classification/Card Counting


## 3. Visual Overlay
Since YOLO is trained on bounding boxes enclosing only the suit and the rank of the card, it’s necessary to develop a method for extending these to cover the entire card.

### Finding the homography of the full card

### Color-coding by Hi-Lo class
As described in the [proposal](./Cv_final_proposal.pdf), each bounding box is color-coded:
    * **Green** boxes indicate cards valued at +1 (typically 2 through 6)
    * **Blue** boxes indicate neutral cards with a value of 0 (typically 7 through 9)
    * **Red** boxes indicate high-value cards that subtract from the count, assigned a value of -1 (typically 10, face cards and aces)

## 4. Occlusions management

### Short-term occlusions

### Partial occlusions


## 5. Metrics


# Results and Discussion
