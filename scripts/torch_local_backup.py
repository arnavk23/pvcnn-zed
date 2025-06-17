import os
import glob
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset

class PKLS3DISDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.files = sorted(glob.glob(os.path.join(root_dir, '**', '*.pkl'), recursive=True))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        with open(self.files[idx], 'rb') as f:
            data = pickle.load(f)

        coord = data['coord']      # (N, 3)
        color = data['color']      # (N, 3)
        semantic_gt = data['semantic_gt'].reshape(-1)  # (N,)

        # Concatenate features: XYZ + RGB
        features = np.concatenate([coord, color], axis=-1)  # (N, 6)

        sample = {
            'features': torch.from_numpy(features).float(),
            'labels': torch.from_numpy(semantic_gt).long()
        }

        if self.transform:
            sample = self.transform(sample)

        return sample
