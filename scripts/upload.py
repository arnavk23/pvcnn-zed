import os
import json

root = "data/shapenet_part"  # update this to your dataset root path

# Load category name <-> synset ID mappings
synset_to_cat = {}
cat_to_synset = {}
with open(os.path.join(root, 'synsetoffset2category.txt'), 'r') as f:
    for line in f:
        cat_name, synset = line.strip().split()
        synset_to_cat[synset] = cat_name
        cat_to_synset[cat_name] = synset

# Load train split JSON (example)
split = "train"
file_list_path = os.path.join(root, 'train_test_split', f'shuffled_{split}_file_list.json')
with open(file_list_path, 'r') as f:
    file_list = json.load(f)

file_paths = []
for file_path in file_list:
    file_path = file_path.strip().strip('"')
    parts = file_path.split('/')
    if len(parts) == 3:
        category_name, _, instance_id = parts
        if category_name not in cat_to_synset:
            print(f"[WARN] Unknown category name: {category_name}")
            continue
        synset = cat_to_synset[category_name]
        instance_id_padded = instance_id.zfill(8)  # adjust padding as needed
        full_file = os.path.join(root, f"{synset}-{instance_id_padded}.npy")
        if not os.path.exists(full_file):
            print(f"[MISSING] File not found: {full_file}")
            continue
        file_paths.append((full_file, synset_to_cat[synset]))
    else:
        print(f"[WARN] Malformed file path: {file_path}")

print(f"Total valid files found: {len(file_paths)}")

