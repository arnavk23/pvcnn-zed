import json

with open("data/shapenet_part/train_test_split/shuffled_test_file_list.json", "r") as f:
    raw = json.load(f)

cleaned = []
for path in raw:
    parts = path.strip().split('/')
    if len(parts) == 3:
        category_map = {
            "Airplane": "02691156",
            "Bag": "02773838",
            "Cap": "02954340",
            "Car": "02958343",
            "Chair": "03001627",
            "Earphone": "03261776",
            "Guitar": "03467517",
            "Knife": "03624134",
            "Lamp": "03636649",
            "Laptop": "03642806",
            "Motorbike": "03790512",
            "Mug": "03797390",
            "Pistol": "03948459",
            "Rocket": "04099429",
            "Skateboard": "04225987",
            "Table": "04379243"
        }
        cat, _, file_id = parts
        if cat in category_map:
            cleaned.append(f"{category_map[cat]}/{file_id}")
        else:
            print(f"Unknown category: {cat}")
    else:
        print(f"Malformed path: {path}")

with open("data/shapenet_part/train_test_split/shuffled_test_file_list_clean.json", "w") as f:
    json.dump(cleaned, f, indent=2)

