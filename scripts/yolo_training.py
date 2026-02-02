import numpy as np
import pandas as pd
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

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

model = YOLO("./model/yolo11s.pt")

print("OKK!!")

# Train on first split of the dataset, using RAM
# Change the device parameter depending on the number of GPUs used
results = model.train(data="./data/data_SPLIT1.yaml", epochs=40, batch=16, augment=False, project="./output", name="RAM1", patience=15, cache='ram', workers=4, device=[0, 1], exist_ok=True)


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
results = model.train(data="./data/data_SPLIT2.yaml", epochs=40, batch=8, augment=False, project="./output", name="RAM2", patience=15, cache='ram', workers=4, device=[0, 1], exist_ok=True)

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
results = model.train(data="./data/data_SPLIT3.yaml", epochs=40, batch=16, augment=True, project="./output", name="FINAL", patience=15, cache=False, workers=4, device=[0, 1], close_mosaic=10, exist_ok=True)

