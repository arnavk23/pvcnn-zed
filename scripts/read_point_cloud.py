import open3d as o3d
import numpy as np
import torch as torch

def load_ply_vertices_only(ply_path):
    # Try reading as mesh and get vertices only
    try:
        mesh = o3d.io.read_triangle_mesh(ply_path)
        if not mesh.has_vertices():
            raise RuntimeError("No vertices found")
        points = np.asarray(mesh.vertices, dtype=np.float32)
        return points
    except Exception as e:
        print(f"Failed to load mesh vertices only: {e}")
        # fallback: try read_point_cloud ignoring colors
        pcd = o3d.geometry.PointCloud()
        with open(ply_path, 'r') as f:
            lines = f.readlines()

        # parse header to find number of vertices and properties, then parse manually skipping colors
        vertex_start = 0
        vertex_count = 0
        header_ended = False
        for i, line in enumerate(lines):
            if line.startswith("element vertex"):
                vertex_count = int(line.strip().split()[-1])
            if line.strip() == "end_header":
                vertex_start = i + 1
                header_ended = True
                break
        if not header_ended:
            raise RuntimeError("PLY header not found or incomplete")

        # Read only xyz skipping color columns
        points_list = []
        for i in range(vertex_start, vertex_start + vertex_count):
            parts = lines[i].strip().split()
            if len(parts) < 3:
                continue  # skip malformed vertex line
            try:
                x, y, z = map(float, parts[:3])
                points_list.append([x, y, z])
            except:
                continue  # skip lines with conversion errors

        points = np.array(points_list, dtype=np.float32)
        return points

# Replace the dataset __init__ with this function
class PlyPointCloudDataset(torch.utils.data.Dataset):
    def __init__(self, ply_file):
        self.points = load_ply_vertices_only(ply_file)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return torch.tensor(self.points)

