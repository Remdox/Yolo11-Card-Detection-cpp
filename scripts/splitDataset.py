import os
import shutil
import random
from collections import defaultdict
from pathlib import Path
import yaml

# ======= CONFIGURATION =======
TRAIN_PATH = Path("dataset_manual_split/data/train_SPLIT1")
TOT_IMGs = 17000
TOT_IMG_SPLIT2 = 5700
TOT_IMG_SPLIT3 = 5700
JOKER_MIN_COUNT = 12
SEED = 42

IMG_PATH = TRAIN_PATH / "images"
LBL_PATH = TRAIN_PATH / "labels"
OUT_SPLIT2_IMG = TRAIN_PATH.parent / "train_SPLIT2/images"
OUT_SPLIT2_LBL = TRAIN_PATH.parent / "train_SPLIT2/labels"
OUT_SPLIT3_IMG = TRAIN_PATH.parent / "train_SPLIT3/images"
OUT_SPLIT3_LBL = TRAIN_PATH.parent / "train_SPLIT3/labels"
os.makedirs(OUT_SPLIT2_IMG, exist_ok=True)
os.makedirs(OUT_SPLIT2_LBL, exist_ok=True)
os.makedirs(OUT_SPLIT3_IMG, exist_ok=True)
os.makedirs(OUT_SPLIT3_LBL, exist_ok=True)

# ======= LOAD CLASS NAMES =======
with open("dataset_manual_split/data_SPLIT1.yaml", "r") as f:
    yml = yaml.safe_load(f)
    class_names = yml["names"]
    joker_class_id = [i for i, name in enumerate(class_names) if "joker" in name.lower()]
    if not joker_class_id:
        raise ValueError("Could not find a class containing 'joker' in data.yaml")
    joker_class_id = joker_class_id[0]

# ======= ANALYZE LABELS =======
label_files = list(LBL_PATH.glob("*.txt"))
random.seed(SEED)
random.shuffle(label_files)

img_to_classes = {}
class_to_imgs = defaultdict(list)

for lbl_file in label_files:
    with open(lbl_file, "r") as f:
        classes = {int(line.split()[0]) for line in f if line.strip()}
    if classes:
        img_name = lbl_file.stem + ".jpg"
        img_to_classes[img_name] = classes
        for c in classes:
            class_to_imgs[c].append(img_name)

# ======= STRATIFIED SPLIT =======
def pick_images(class_to_imgs, target_count, exclude=set()):
    selected = set()
    class_coverage = defaultdict(int)

    while len(selected) < target_count:
        for cls, imgs in class_to_imgs.items():
            candidates = [img for img in imgs if img not in selected and img not in exclude]
            if candidates:
                selected.add(random.choice(candidates))
                class_coverage[cls] += 1
            if len(selected) >= target_count:
                break
    return selected

# ======= HANDLE JOKERS FIRST =======
# Since jokers are present only in one of the two datasets to merge, there are less instances of them overall.
# This means that the split has to be done in such a way that the number of jokers is mantained balanced throughout the whole training process.
# In this code we provide a minimum count of jokers for each split
joker_imgs = list(set(class_to_imgs[joker_class_id]))
random.shuffle(joker_imgs)
joker_SPLIT2 = set(joker_imgs[:JOKER_MIN_COUNT])
joker_SPLIT3 = set(joker_imgs[JOKER_MIN_COUNT:2*JOKER_MIN_COUNT])
joker_used = joker_SPLIT2 | joker_SPLIT3

# ======= PICK REST =======
rest_SPLIT2 = pick_images(class_to_imgs, TOT_IMG_SPLIT2 - len(joker_SPLIT2), exclude=joker_used)
rest_SPLIT3 = pick_images(class_to_imgs, TOT_IMG_SPLIT3 - len(joker_SPLIT3), exclude=rest_SPLIT2 | joker_used)

train_ram_imgs = joker_SPLIT2 | rest_SPLIT2
train_disk_imgs = joker_SPLIT3 | rest_SPLIT3
train_used = train_ram_imgs | train_disk_imgs

# Remaining go in full train
train_imgs = set(img_to_classes.keys()) - train_used

# ======= MOVE IMAGES BETWEEN DIRECTORIES =======
def move(img_name, dst_img_dir, dst_lbl_dir):
    src_img = IMG_PATH / img_name
    src_lbl = LBL_PATH / (Path(img_name).stem + ".txt")
    if src_img.exists():
        shutil.move(str(src_img), str(dst_img_dir / img_name))
    if src_lbl.exists():
        shutil.move(str(src_lbl), str(dst_lbl_dir / src_lbl.name))

for img in train_ram_imgs:
    move(img, OUT_SPLIT2_IMG, OUT_SPLIT2_LBL)

for img in train_disk_imgs:
    move(img, OUT_SPLIT3_IMG, OUT_SPLIT3_LBL)

# ======= RESULTS FOR CHECKING CORRECT SPLITTING =======
print(f"Train_SPLIT2:  {len(train_ram_imgs)} images")
print(f"Train_SPLIT3: {len(train_disk_imgs)} images")
print(f"Train:      {len(train_imgs)} images (leftover)")
print(f"JOKERs → RAM: {len(joker_SPLIT2)}, split3: {len(joker_SPLIT3)}")
