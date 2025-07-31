import os
from pathlib import Path
import shutil
import yaml

labels_folder = Path("./data/data/valid/labels/")
output_folder = labels_folder.parent / (labels_folder.name + "_remapped")

data_yaml_path = "./data.yaml"
data2_yaml_path = "./data2.yaml"

def load_names(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)["names"]

names = load_names(data_yaml_path)
names2 = load_names(data2_yaml_path)

name_to_index = {name.upper(): idx for idx, name in enumerate(names)}
index_to_new_index = {}

for old_idx, name in enumerate(names2):
    name_up = name.upper()
    if name_up in name_to_index:
        index_to_new_index[old_idx] = name_to_index[name_up]
    else:
        print(f"Warning: '{name_up}' not found in target labels")

output_folder.mkdir(parents=True, exist_ok=True)

for txt_file in labels_folder.glob("*.txt"):
    new_lines = []
    with open(txt_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            old_idx = int(parts[0])
            if old_idx in index_to_new_index:
                parts[0] = str(index_to_new_index[old_idx])
                new_lines.append(" ".join(parts))
            else:
                print(f"⚠️ Index {old_idx} not in mapping, skipping line: {line.strip()}")

    new_path = output_folder / txt_file.name
    with open(new_path, "w") as f:
        f.write("\n".join(new_lines) + "\n")

print(f"Conversion complete. Remapped labels saved in: {output_folder}")
