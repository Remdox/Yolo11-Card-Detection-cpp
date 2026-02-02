# CV Final Project
CV Final Project, 2025.

Table of Contents
=================

   * [Introduction](#introduction)
   * [Instructions](#Instructions)
   * [Datasets](#Datasets)
      * [Training](#Training)
      * [Validation](#Validation)
      * [Test](#Test)
   * [Code](#code)
      * [Object Detection and Initial Classification](#Object-Detection-and-Initial-Classification)
      * [Hi-Lo classification and Card Counting](#Hi-Lo-classification-and-Card-Counting)
   * [Output & Metrics](#Output-&-Metrics)

# Introduction
[Read the full proposal](./Cv_final_proposal.pdf).
   
# Instructions
## Requirements
* CMake version: 4.0.0+
* OpenCV version: 4+
* ONNX Runtime version: 1.21.0. The binaries for the CPU version are already bundled inside the project in the [external](./external) folder and **CMake is already configured to find the binaries either in this folder or in the system's directories**. The GPU version can be downloaded from the most recent [Github Release](https://github.com/Remdox/Yolo11-Card-Detection-cpp/releases/latest) and it's compatible with CUDA 12.x versions. If both packages are present in the external folder, the GPU version will be selected first. 

\
The ONNX Runtime Library can be manually installed following the instructions below.

### On LINUX

Option 1 - Manual installation inside the [external](./external) folder:
* Extract the [onnxruntime-linux-x64-1.21.0.tgz](./external/onnxruntime-linux-x64-1.21.0.tgz) archive or download and extract the [onnxruntime-linux-x64-gpu-1.21.0.zip](https://github.com/Remdox/Yolo11-Card-Detection-cpp/releases/latest/download/onnxruntime-linux-x64-gpu-1.21.0.zip) archive from our most recent [Github Release](https://github.com/Remdox/Yolo11-Card-Detection-cpp/releases/latest);
* Move the extracted folder inside the external/ directory.

Option 2 - Automatic (system-wide) installation using the provided bash script:
* Run either [onnxruntime_Linux_install.sh](./external/onnxruntime_Linux_install.sh) or [onnxruntime_Linux_GPU_install.sh](./external/onnxruntime_Linux_GPU_install.sh);

Option 3 - Manual global installation:
* Follow the first step in option 1;
* Copy the .so files of lib in `/usr/local/lib64/` ;
* Copy the .cmake files in `/usr/local/lib64/cmake/onnxruntime/` ;
* Copy the `include/onnxruntime/` folder in `/usr/local/include/` ;
* update the libraries cache running `ldconfig` ;

### On WINDOWS

Please use Linux (...or you could also check out the official documentation for installing ONNX Runtime on the official website).

## Running the project
To run the project:
1. Build and compilation:
   ```
   cd build
   cmake ..
   make
   ```
4. You can run the project on a default pre-bundled image using:
   ```
   ./finalProject --default-image
   ```
   Run with a default pre-bundled video using:
   ```
   ./finalProject --default-video
   ```
   Alternatively, provide a custom file or the camera, using the main menu:
   ```
   ./finalProject
   ```
   When specifying the path of the custom file relative to the build directory, it's possible to use the `TAB` shortcut for autocompletition.
   
# Datasets
Some datasets of the proposal are used, with the addition of other datasets to have greater variety and robustness. No data augmentation has been used for the datasets, keeping a low memory overhead.

## Training
The program runs using YOLO, which means that the training dataset has to follow YOLO's folder structure. See: [YOLO's Dataset Structure for YOLO Classification Tasks](https://docs.ultralytics.com/datasets/classify/).
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

The datasets used are:
   * **The Complete Playing Card Dataset** by **Jay Pradip Shah** on Kaggle. See: https://www.kaggle.com/datasets/jaypradipshah/the-complete-playing-card-dataset.
   * **Playing Cards Object Detection Dataset** by **Andy8744** on Kaggle. See: https://www.kaggle.com/datasets/andy8744/playing-cards-object-detection-dataset.

The two have been merged into one single dataset. The labels have been [adjusted](./scripts/changeLabelsToMatchNewYaml.py) to the data.yaml file used for **The Complete Playing Card Dataset**.

## Validation
The **Playing Cards Object Detection Dataset** provides the images used for validation.

## Test
The model is applied on videos from Youtube.
In order to do this, some tools like [CVAT](https://www.cvat.ai/), [Label studio](https://github.com/HumanSignal/labelImg), [Roboflow](https://roboflow.com/) can be used for annotations on Youtube clips of poker games.


# Code
## Doxygen Documentation

## Object Detection and Initial Classification
### Training of the model and exporting to ONNX format
The YOLO model is trained on the dataset using two RTX 3090 GPUs inside the UniPD DEI cluster (see its [official documentation](https://docs.dei.unipd.it/en/CLUSTER) and the [slurm file](./scripts/yolo.slurm) used). YOLO11s is employed to deliver good performance with enough speed. 

The training dataset has been splitted on 3 smaller subsets such that:
 * Each subset contains images (and corresponding labels) which are different than the other datasets (preventing data leakage)
 * Each subset has a maximum size which allows each one of them to be loaded into RAM.

As a result, the training process is divided into three sequential steps: each step loads the model weights from the previous step and continues training on the next subset. This allows for a faster and more fault-tolerant training process with a negligent penalty on accuracy. The full [script](./scripts/yolo_training.py) is available. 
 
The model detects a bounding box enclosing the suit and the rank on the corners of a poker card. This means that at most 2 bounding boxes can be found for the same card, which makes it easier to find in case of partial occlusions. The bounding boxes are then classified as the suit and the rank they enclose.

Since the inference has to be performed in a C++ program, the model has to be [exported](./scripts/convertToONNX.ipynb) into a ONNX format and imported into the program using a library like ONNX Runtime.

### Inference
The inference is subdivided into three sections:
 * Pre-processing of the image: the image is formatted as a valid input of the Yolo11s model and the ONNX Runtime session. Letterbox padding of the input images is enabled by default, since YOLO models use it for the training phase, but the option can also be disabled when calling the function used for the inference.
 * Inference of the imported YOLO11s model, using a ONNX Runtime session. If the model used is static, YOLO11 generates 84000 detections.
 * Post-processing of the results. Each detection result consists of 4 values defining the object's bounding box (position and size), and a confidence score for every possible class which the object might correspond to. Non-maxima suppression is used to keep only 1 bounding box for each detected object.
 
 In order to handle detection of small cards in big images, a sliding window approach is used: the image is subdivided in smaller, slightly overlapping tiles, with size equal to the image size used for training the model (640px). The overhead introduced by this method is mitigated by using multi-threading, where each tile is processed by a separate thread in parallel.

 More details are available in the [source code](./src/marco_annunziata.cpp).

## Hi-Lo classification and video processing

### Color-coding by Hi-Lo class
As described in the [proposal](./Cv_final_proposal.pdf), each bounding box is color-coded:
    * **Green** boxes indicate cards valued at +1 (typically 2 through 6)
    * **Blue** boxes indicate neutral cards with a value of 0 (typically 7 through 9)
    * **Red** boxes indicate high-value cards that subtract from the count, assigned a value of -1 (typically 10, face cards and aces)

### Metrics
The metrics used are: Precision, Recall, F1-Score. 

For custom files, ground truths can be manually added inside the `data/test/ground_truths` subfolders. The content of the ground truths must be expressed in the Multi-Object Tracking (MOT) format in order to be correctly read. For more info on how to add ground truths for custom files, more specific [instructions](./data/test/ground_truths/README_adding_labels.txt) are provided. 

If the ground truths for a custom file are not provided, the program will still run the detection pipeline without computing metrics.

# Output
