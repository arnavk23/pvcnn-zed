import os
import zipfile
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Set CUDA_HOME if needed (before importing any CUDA-dependent libs)
os.environ['CUDA_HOME'] = '/usr/local/cuda'

# Import your PVConv model - adjust this import path as needed
from modules import PVConv

# === Step 1: Unzip dataset ===
def unzip_dataset(zip_path, extract_to):
    if not os.path.exists(extract_to):
        print(f"Extracting dataset from {zip_path} to {extract_to}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print("Extraction complete.")
    else:
        print(f"Dataset already extracted at {extract_to}.")

# === Step 2: Dataset Class ===
class PointCloudDataset(Dataset):
    def __init__(self, data_dir):
        super().__init__()
        self.cloud_dir = os.path.join(data_dir, 'clouds')
        self.label_dir = os.path.join(data_dir, 'labels')
        self.cloud_files = sorted(os.listdir(self.cloud_dir))
        self.label_files = sorted(os.listdir(self.label_dir))
        assert len(self.cloud_files) == len(self.label_files), "Mismatch in clouds and labels count"

    def __len__(self):
        return len(self.cloud_files)

    def __getitem__(self, idx):
        cloud_path = os.path.join(self.cloud_dir, self.cloud_files[idx])
        label_path = os.path.join(self.label_dir, self.label_files[idx])

        # Adjust loaders based on your actual file format!
        # Example: loading numpy arrays (.npy)
        points = np.load(cloud_path)  # shape (N, 3) or (N, features)
        labels = np.load(label_path)  # shape (N, )

        points = torch.from_numpy(points).float()
        labels = torch.from_numpy(labels).long()

        return points, labels

# === Step 3: Main training function ===
def main():
    # Paths - change these to your files
    zip_path = '/home/intern/Downloads/data_object_label_2.zip'  # Your zip file path
    extract_to = 'dataset_extracted'
    unzip_dataset(zip_path, extract_to)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Dataset + DataLoader
    dataset = PointCloudDataset(extract_to)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)

    # Model params - adjust these based on your PVConv version
    in_channels = 3      # e.g. xyz only, or xyz+features
    out_channels = 32    # feature dimension inside model
    kernel_size = 3
    resolution = 0.1

    model = PVConv(in_channels, out_channels, kernel_size, resolution).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(1):  # just 1 epoch example
        for points, labels in dataloader:
            points = points.to(device)          # (B, N, 3 or features)
            labels = labels.to(device)          # (B, N)

            coords = points[..., :3]
            features = points[..., 3:] if points.shape[-1] > 3 else coords

            inputs = (features, coords)

            optimizer.zero_grad()
            outputs = model(inputs)              # forward pass

            # outputs shape depends on your model, example:
            # assume (B, num_classes, N)
            outputs = outputs.permute(0, 2, 1)  # to (B, N, num_classes) if needed
            loss = criterion(outputs.reshape(-1, outputs.shape[-1]), labels.reshape(-1))

            loss.backward()
            optimizer.step()

            print(f"Loss: {loss.item():.4f}")

    # Save checkpoint
    checkpoint_path = 'checkpoints/best_model.pth'
    os.makedirs('checkpoints', exist_ok=True)
    torch.save({'model_state_dict': model.state_dict()}, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

if __name__ == "__main__":
    main()

