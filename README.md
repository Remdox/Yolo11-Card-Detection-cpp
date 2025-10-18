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
      * [Visual Overlay](#Visual-Overlay)
      * [Occlusions management](#Occlusions-management)
   * [Output](output)

# Introduction
[Read the full proposal](./Cv_final_proposal.pdf).
   
# Instructions
## Requirements
* CMake version: 4.0.0+
* OpenCV version: 4+
* ONNXRuntime version: 1.21.0. The binaries are already bundled inside the project in the [external](./external) folder and **CMake is already configured to find the binaries either in this folder or in the system's directories**. If there are problems on using this library with CMake on Linux, you can manually install it:

**On LINUX**

Option 1 - Automatic (system-wide) installation using the bash script:
* Run [onnxruntime_Linux_install.sh](./external/onnxruntime_Linux_install.sh)
* See section: [Running the project](#Running-the-project)

Option 2 - Manual (global) installation:
* Extract onnxruntime-linux-x64-1.21.0.tgz
* Copy the .so files of lib in /usr/local/lib64/
* Copy the .cmake files in /usr/local/lib64/cmake/onnxruntime/
* Copy the include/onnxruntime/ folder in /usr/local/include/
* update the libraries cache running ldconfig

**On WINDOWS**
Use Linux (...well, you could also check out the official documentation for installing ONNXRuntime on their website).

## Running the project
Make sure to put the videos to use for testing in the data/test/ directory and the .onnx file of the model in data/model/ directory along with a .txt file containing the labels, one label per line.
To run the project:
1. cd into build/
2. cmake ..
3. make -> compiles (doesn't compile if there are compiler errors)
4. to start, test on an image with `./finalProject <image_to_test>` or see the help page with `./finalProject -h`

   
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
## 1.Object Detection and Initial Classification
### Training of the model and exporting to ONNX format
The YOLO model is trained on the dataset using Kaggle's 2 freely available T4 GPUs. YOLO11s is employed to deliver good performance with enough speed. The training process has been optimized as in the following python code:

```python
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

model = YOLO("/kaggle/working/yolo11s.pt")results = model.train(data="/kaggle/wresults = model.train(data="/kaggle/working/data.yaml", epochs=100, batch=16, augment=False, save_dir="/kaggle/working/output", patience=40, cache=True, workers=8, device=[0, 1])orking/data.yaml", epochs=100, batch=16, augment=False, save_dir="/kaggle/working/output", patience=40, cache=True, workers=8, device=[0, 1])
results = model.train(data="/kaggle/working/data_SPLIT1.yaml", epochs=40, batch=16, augment=False, project="/kaggle/working/output", name="RAM1", patience=15, cache='ram', workers=8, device=[0, 1])
```
Given the size of the training data and the high fail rate of Kaggle machines, the training dataset has been splitted on 3 smaller subsets such that:
 * Each subset contains images (and corresponding labels) which are different than the other datasets (preventing data leakage)
 * Each subset has a maximum size which allows each one of them to be loaded into RAM (Kaggle machines avail of GB of RAM)
As a result, the training process is divided into three sequential steps: each step loads the model weights from the previous step and continues training on the next subset. This allows for a faster and more fault-tolerant training process with a negligent penalty on accuracy. The full code is available as a [.ipynb file](./scripts/training.ipynb).
 
The model detects a bounding box enclosing the suit and the rank on the corners of a poker card. This means that at most 2 bounding boxes can be found for the same card, which makes it easier to find in case of partial occlusions. The bounding boxes are then classified as the suit and the rank they enclose.

Since the inference has to be performed in a C++ program, the model has to be [exported](./scripts/convertToONNX.ipynb) into a ONNX format and imported into the program using a library like ONNX Runtime.

### Inference
The inference is subdivided into three sections:
 * Pre-processing of the image: the image is formatted as a valid input of the Yolo11s model and the ONNX Runtime session. Letterbox padding of the input images is enabled by default, since YOLO models use it for the training phase, but the option can also be disabled when calling the function used for the inference.
 * Inference of the imported YOLO11s model, using a ONNX Runtime session. If the model used is static, YOLO11 generates 84000 detections.
 * Post-processing of the results. Each detection result consists of 4 values defining the object's bounding box (position and size), and a confidence score for every possible class which the object might correspond to. Non-maxima suppression is used to keep only 1 bounding box for each detected object.
 
 More details are available inside the [source code](./src/marco_annunziata.cpp).

## 2. Hi-Lo classification and video processing


## 3. Visual Overlay
Since YOLO11s is trained on bounding boxes enclosing only the suit and the rank of the card, it’s necessary to develop a method for extending these to cover the entire card.

### Finding the homography of the full card

### Color-coding by Hi-Lo class
As described in the [proposal](./Cv_final_proposal.pdf), each bounding box is color-coded:
    * **Green** boxes indicate cards valued at +1 (typically 2 through 6)
    * **Blue** boxes indicate neutral cards with a value of 0 (typically 7 through 9)
    * **Red** boxes indicate high-value cards that subtract from the count, assigned a value of -1 (typically 10, face cards and aces)

## 4. Occlusions management

### Short-term occlusions

### Partial occlusions


## 5. Output

### Metrics

### Results
