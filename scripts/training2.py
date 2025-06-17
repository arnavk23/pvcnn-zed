import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import open3d as o3d

# Dataset for single PLY file point cloud
class PlyPointCloudDataset(Dataset):
    def __init__(self, ply_file):
        self.ply_file = ply_file
        pcd = o3d.io.read_point_cloud(ply_file)
        self.points = np.asarray(pcd.points, dtype=np.float32)
        # Colors ignored for model input

    def __len__(self):
        return 1  # Single sample (one point cloud)

    def __getitem__(self, idx):
        return torch.tensor(self.points)  # (N, 3)

# Updated PointNetSegmentation model
class PointNetSegmentation(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(64, 128, 1)
        self.bn2 = nn.BatchNorm1d(128)

        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn3 = nn.BatchNorm1d(1024)

        # fc1 input dim = local feat (128) + global feat (1024) = 1152
        self.fc1 = nn.Linear(128 + 1024, 512)
        self.bn4 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn5 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        # x: (batch_size, num_points, 3)
        x = x.permute(0, 2, 1)  # -> (batch_size, 3, num_points)
        x = F.relu(self.bn1(self.conv1(x)))  # (batch, 64, N)
        x = F.relu(self.bn2(self.conv2(x)))  # (batch, 128, N)

        local_feat = x  # save local features before global

        x = F.relu(self.bn3(self.conv3(x)))  # (batch, 1024, N)
        global_feat = torch.max(x, 2, keepdim=True)[0]  # (batch, 1024, 1)

        global_feat_expand = global_feat.repeat(1, 1, x.size(2))  # (batch, 1024, N)

        # concat local and global features
        x_concat = torch.cat([local_feat, global_feat_expand], dim=1)  # (batch, 1152, N)
        x_concat = x_concat.permute(0, 2, 1)  # (batch, N, 1152)

        x = F.relu(self.bn4(self.fc1(x_concat)))  # (batch, N, 512)
        x = F.relu(self.bn5(self.fc2(x)))         # (batch, N, 256)
        x = self.fc3(x)                           # (batch, N, num_classes)

        return x

def segment_ply(ply_path, model, device, class_names):
    model.eval()
    dataset = PlyPointCloudDataset(ply_path)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    with torch.no_grad():
        for points in loader:
            points = points.to(device)  # (1, N, 3)
            outputs = model(points)     # (1, N, num_classes)
            preds = outputs.argmax(dim=2).squeeze(0).cpu().numpy()  # (N,)

            print(f"Segmented {len(preds)} points.")
            for cls_idx in np.unique(preds):
                print(f"Class '{class_names[cls_idx]}' points: {(preds == cls_idx).sum()}")

            # Load original point cloud again to preserve colors
            pcd = o3d.io.read_point_cloud(ply_path)
            points_np = np.asarray(pcd.points)
            pcd_segmented = o3d.geometry.PointCloud()
            pcd_segmented.points = o3d.utility.Vector3dVector(points_np)

            color_map = {
                "DontCare": [0.5, 0.5, 0.5],
                "Pedestrian": [1.0, 0.0, 0.0],
                "Cyclist": [0.0, 1.0, 0.0],
                "Car": [0.0, 0.0, 1.0],
                "Truck": [1.0, 1.0, 0.0],
                "Misc": [1.0, 0.0, 1.0],
            }
            colors = np.array([color_map.get(class_names[p], [0, 0, 0]) for p in preds], dtype=np.float32)
            pcd_segmented.colors = o3d.utility.Vector3dVector(colors)

            output_path = os.path.splitext(ply_path)[0] + "_segmented.ply"
            o3d.io.write_point_cloud(output_path, pcd_segmented)
            print(f"Saved segmented point cloud to: {output_path}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    class_names = ["DontCare", "Pedestrian", "Cyclist", "Car", "Truck", "Misc"]
    num_classes = len(class_names)

    model = PointNetSegmentation(num_classes=num_classes).to(device)
    # Load pretrained weights if available:
    # model.load_state_dict(torch.load('path_to_checkpoint.pth'))

    ply_path = "/home/intern/Desktop/zed_out.ply"  # Update your path here
    segment_ply(ply_path, model, device, class_names)

if __name__ == "__main__":
    main()

