import os
import pickle
import torch
from torch.utils.data import Dataset

class S3DISPKLDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data_files = []

        for area in os.listdir(root_dir):
            area_path = os.path.join(root_dir, area, 'hallway_1', 'archive/data.pkl')
            if os.path.isfile(area_path):
                self.data_files.append(area_path)

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        with open(self.data_files[idx], 'rb') as f:
            data = pickle.load(f)

        coord = data['coord']
        color = data['color']
        semantic_gt = data['semantic_gt'].squeeze()

        # Normalize and stack inputs
        point_features = torch.tensor(color / 255.0, dtype=torch.float32)
        coords = torch.tensor(coord, dtype=torch.float32)
        label = torch.tensor(semantic_gt, dtype=torch.long)

        return coords, point_features, label

