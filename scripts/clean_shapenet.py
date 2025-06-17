import json
import os

def clean_json_file(filepath):
    print(f"Cleaning: {filepath}")
    with open(filepath, 'r') as f:
        data = json.load(f)

    # Only keep entries like "02691156/1a04e3e8e0b7c6f1a3f2462f0c3e2d1d"
    cleaned = [x for x in data if '/' in x and x.count('/') == 1 and not x.startswith("Bag/")]

    print(f"Original: {len(data)} entries")
    print(f"Cleaned:  {len(cleaned)} entries")

    with open(filepath, 'w') as f:
        json.dump(cleaned, f, indent=2)

# Path to your json files
base_dir = 'data/shapenet_part/train_test_split'
clean_json_file(os.path.join(base_dir, 'shuffled_train_file_list_clean.json'))
clean_json_file(os.path.join(base_dir, 'shuffled_test_file_list_clean.json'))

import json
import random

with open("data/shapenet_part/train_test_split/shuffled_train_file_list_clean.json", "r") as f:
    data = json.load(f)

random.shuffle(data)
test_data = data[:1000]
train_data = data[1000:]

with open("data/shapenet_part/train_test_split/shuffled_train_file_list_clean.json", "w") as f:
    json.dump(train_data, f, indent=2)

with open("data/shapenet_part/train_test_split/shuffled_test_file_list_clean.json", "w") as f:
    json.dump(test_data, f, indent=2)


