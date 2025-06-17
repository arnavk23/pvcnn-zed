import numpy as np
import open3d as o3d
from plyfile import PlyData, PlyElement
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from collections import defaultdict

class PointCloudSegmenter:
    def __init__(self, num_classes=6):
        self.num_classes = num_classes
        self.class_names = ["Ground", "Building", "Vehicle", "Pedestrian", "Vegetation", "Other"]
        self.color_map = plt.get_cmap('tab20')
        
    def read_ply(self, file_path):
        """Read PLY file and return points with optional features"""
        try:
            plydata = PlyData.read(file_path)
            vertex = plydata['vertex']
            
            # Get available properties
            properties = vertex.data.dtype.names
            
            # Always get coordinates
            points = np.vstack([vertex['x'], vertex['y'], vertex['z']]).T
            
            # Try to get additional features if available
            features = []
            for feat in ['red', 'green', 'blue', 'intensity']:
                if feat in properties:
                    features.append(vertex[feat])
            
            if features:
                features = np.vstack(features).T
                return points.astype(np.float32), features.astype(np.float32)
            
            return points.astype(np.float32), None
            
        except Exception as e:
            print(f"Error reading PLY file: {e}")
            return None, None

    def cluster_points(self, points, features=None):
        """Enhanced clustering with better class distribution"""
        # Adaptive DBSCAN parameters
        z_range = np.percentile(points[:, 2], 95) - np.percentile(points[:, 2], 5)
        eps = 0.03 * z_range
        min_samples = max(10, int(0.002 * len(points)))
        
        # Prepare feature vector
        if features is not None:
            features = StandardScaler().fit_transform(features)
            cluster_data = np.hstack([StandardScaler().fit_transform(points), features])
        else:
            cluster_data = StandardScaler().fit_transform(points)
        
        # First stage: DBSCAN for initial clustering
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(cluster_data)
        core_samples = db.core_sample_indices_
        labels = db.labels_
        
        # Filter out noise (will be classified separately)
        valid_mask = labels != -1
        filtered_points = points[valid_mask]
        filtered_labels = labels[valid_mask]
        
        if len(filtered_points) == 0:
            return np.zeros(len(points))  # All points are noise
            
        # Second stage: GMM for better class distribution
        n_clusters = min(50, len(np.unique(filtered_labels)))  # Limit number of clusters
        gmm = GaussianMixture(n_components=n_clusters, covariance_type='spherical')
        gmm.fit(filtered_points)
        gmm_labels = gmm.predict(filtered_points)
        
        # Analyze clusters and assign classes
        final_labels = np.zeros(len(points), dtype=int)
        
        # Class assignment rules (customize these thresholds for your data)
        for cluster_id in np.unique(gmm_labels):
            cluster_mask = gmm_labels == cluster_id
            cluster_pts = filtered_points[cluster_mask]
            
            # Calculate cluster properties
            z_mean = cluster_pts[:, 2].mean()
            z_std = cluster_pts[:, 2].std()
            xy_span = np.ptp(cluster_pts[:, :2], axis=0)  # Fixed: using np.ptp() instead of .ptp()
            area = xy_span[0] * xy_span[1]
            density = len(cluster_pts) / area
            
            # Enhanced classification logic
            if z_mean < 0.5 and area > 100 and density > 10:
                cls = 0  # Ground
            elif z_mean > 3 and area > 20 and z_std > 1.5:
                cls = 1  # Building
            elif 0.5 < z_mean < 2.5 and 1 < area < 20 and density > 5:
                cls = 2  # Vehicle
            elif 0.5 < z_mean < 2 and area < 1 and density > 20:
                cls = 3  # Pedestrian
            elif z_std > 1 and density < 5:
                cls = 4  # Vegetation
            else:
                cls = 5  # Other
            
            final_labels[valid_mask][cluster_mask] = cls
        
        # Classify noise points based on elevation
        noise_mask = labels == -1
        final_labels[noise_mask] = np.where(
            points[noise_mask, 2] > 1.0, 
            5,  # Other (above ground)
            0    # Ground (below threshold)
        )
        
        return final_labels

    def visualize(self, points, labels):
        """Visualize segmented point cloud"""
        # Assign colors based on labels
        colors = self.color_map(labels / (self.num_classes - 1))[:, :3]
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Add coordinate frame
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
        
        # Custom visualization
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name='Point Cloud Segmentation', width=1024, height=768)
        vis.add_geometry(pcd)
        vis.add_geometry(coord_frame)
        
        # Set view parameters
        ctr = vis.get_view_control()
        ctr.set_front([0, 0, -1])
        ctr.set_up([0, -1, 0])
        
        vis.run()
        vis.destroy_window()

    def save_segmented_ply(self, points, labels, output_path):
        """Save segmented point cloud to PLY file"""
        colors = (self.color_map(labels / (self.num_classes - 1))[:, :3] * 255).astype(np.uint8)
        
        vertex = np.array(
            [(points[i,0], points[i,1], points[i,2], colors[i,0], colors[i,1], colors[i,2]) 
            for i in range(points.shape[0])],
            dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                   ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
        )
        
        ply_el = PlyElement.describe(vertex, 'vertex')
        PlyData([ply_el], text=False).write(output_path)
        print(f"Saved segmented point cloud to {output_path}")

    def analyze_results(self, labels):
        """Print segmentation statistics"""
        unique, counts = np.unique(labels, return_counts=True)
        print("\nSegmentation Results:")
        print("--------------------")
        for cls, cnt in zip(unique, counts):
            print(f"{self.class_names[cls]:<15}: {cnt:>6} points ({cnt/len(labels):.1%})")

def main():
    segmenter = PointCloudSegmenter()
    
    # Load point cloud
    input_path = "/home/intern/Desktop/zed_out.ply"  # Change to your file path
    points, features = segmenter.read_ply(input_path)
    if points is None:
        return
    
    print(f"Loaded {points.shape[0]} points")
    if features is not None:
        print(f"Found {features.shape[1]} additional features")
    
    # Perform segmentation
    labels = segmenter.cluster_points(points, features)
    
    # Visualize results
    segmenter.visualize(points, labels)
    
    # Save results
    output_path = input_path.replace(".ply", "_segmented.ply")
    segmenter.save_segmented_ply(points, labels, output_path)
    
    # Show statistics
    segmenter.analyze_results(labels)

if __name__ == "__main__":
    main()
