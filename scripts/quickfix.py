import json

def clean_file(file_path):
    with open(file_path) as f:
        data = json.load(f)

    cleaned = []
    bad = []
    for entry in data:
        if isinstance(entry, str) and entry.count('/') == 1 and len(entry.split('/')[0]) == 8:
            cleaned.append(entry)
        else:
            bad.append(entry)

    print(f"Cleaning {file_path}:")
    print(f"  Valid entries: {len(cleaned)}")
    print(f"  Invalid entries: {len(bad)}")
    if bad:
        print("  Sample bad entry:", bad[0])

    with open(file_path, 'w') as f:
        json.dump(cleaned, f, indent=2)

# Update the paths if your directory is different
clean_file('data/shapenet_part/train_test_split/shuffled_train_file_list.json')
clean_file('data/shapenet_part/train_test_split/shuffled_test_file_list.json')


