#!/usr/bin/env python
# coding: utf-8

# In[5]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from ultralytics import YOLO
import logging
import gc
import torch
import os
import logging

file = logging.FileHandler("consoleRAM1.log")
loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
for logger in loggers:
    logger.handlers = []
    logger.addHandler(file)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory


#for dirname, _, filenames in os.walk('/kaggle/input'):
#    print(dirname)


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

model = YOLO("./model/yolo11s.pt")

print("OKK!!")

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outsilsde of the current session


# Train on first split of the dataset, using RAM
# Change the device parameter depending on the number of GPUs used
results = model.train(data="./data/data_SPLIT1.yaml", epochs=40, batch=32, augment=True, hsv_h=0.6, hsv_s=0.7, hsv_v=0.4, flipud=0.5, fliplr=0.5, degrees=180.0, perspective=0.001, scale=0.5, translate=0.1, shear=5.0, project="./output", name="RAM1", patience=15, cache='ram', workers=4, device=[0, 1], imgsz=640, close_mosaic=10, exist_ok=True)


file = logging.FileHandler("consoleRAM2.log")
loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
for logger in loggers:
    logger.handlers = []
    logger.addHandler(file)

# This is used to prevent memory leaks from the last training session using RAM
gc.collect()
torch.cuda.empty_cache()

del model
del results
gc.collect()
torch.cuda.empty_cache()
# Train on second split of the dataset, using RAM
model = YOLO("./output/RAM1/weights/best.pt")
results = model.train(data="./data/data_SPLIT2.yaml", epochs=40, batch=32, augment=True, hsv_h=0.6, hsv_s=0.7, hsv_v=0.4, flipud=0.5, fliplr=0.5, degrees=180.0, perspective=0.001, scale=0.5, translate=0.1, shear=5.0, project="./output", name="RAM2", patience=15, cache='ram', workers=4, device=[0, 1], imgsz=640, close_mosaic=10, exist_ok=True)


# In[ ]:

file = logging.FileHandler("consoleFINAL.log")
loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
for logger in loggers:
    logger.handlers = []
    logger.addHandler(file)

# This is used to prevent memory leaks from the last training session using RAM
gc.collect()
torch.cuda.empty_cache()

del model
del results
gc.collect()
torch.cuda.empty_cache()
# Train on third split of the dataset, using RAM
model = YOLO("./output/RAM2/weights/best.pt")
results = model.train(data="./data/data_SPLIT3.yaml", epochs=40, batch=32, augment=True, hsv_h=0.6, hsv_s=0.7, hsv_v=0.4, flipud=0.5, fliplr=0.5, degrees=180.0, perspective=0.001, scale=0.5, translate=0.1, shear=5.0, project="./output", name="FINAL", patience=15, cache=False, workers=4, device=[0, 1], imgsz=640, close_mosaic=10, exist_ok=True)

