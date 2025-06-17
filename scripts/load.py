from plyfile import PlyData
import numpy as np
import open3d as o3d

ply_file = "/home/intern/Desktop/zed_out_segmented.ply"

plydata = PlyData.read(ply_file)
points = np.vstack([plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z']]).T
colors = np.vstack([plydata['vertex']['red'], plydata['vertex']['green'], plydata['vertex']['blue']]).T / 255.0  # normalize to [0,1]

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points.astype(np.float32))
pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float32))

print(f"Loaded {len(points)} points with colors")

o3d.visualization.draw_geometries([pcd])

