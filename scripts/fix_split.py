import os
import json

root_dir = "data/shapenet_part"
output_file = os.path.join(root_dir, "train_test_split/shuffled_train_file_list_fixed.json")

file_list = []
for subdir in os.listdir(root_dir):
    subpath = os.path.join(root_dir, subdir)
    if not os.path.isdir(subpath):
        continue
    for fname in os.listdir(subpath):
        if fname.endswith(".npy") and "-" in fname:
            try:
                synset, instance = fname.replace(".npy", "").split("-")
                file_list.append(f"{synset}/{instance}")
            except ValueError:
                print(f"Skipping malformed: {fname}")

file_list.sort()  # Optional: shuffle if needed

# Save
with open(output_file, "w") as f:
    json.dump(file_list, f, indent=2)

print(f"Saved fixed file list with {len(file_list)} entries to {output_file}")

