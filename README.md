## PVCNN on ZED 2i Camera with ShapeNet Dataset

This project demonstrates how to run the PVCNN (Point-Voxel Convolutional Neural Network) model using real-world point cloud data from a **ZED 2i** camera. The pipeline uses the **ShapeNet Part Segmentation** dataset for training, and inference is done on live or recorded point clouds from the ZED 2i.

### Features

- Real-time or offline 3D point cloud inference using ZED 2i stereo depth.
- Pretrained PVCNN segmentation model support.
- Training on ShapeNet part segmentation dataset.
- Uses `zed_inference.py` and `zed_helpers.py` for camera integration and preprocessing.

### Requirements

- Python 3.8+
- PyTorch ≥ 1.7 with CUDA
- ZED SDK ≥ 3.8
- ZED Python API (`pyzed`)
- OpenCV, NumPy, tqdm
- PyTorch3D (optional for rendering)

#### Install Dependencies

```bash
pip install -r requirements.txt
````

Install ZED SDK and Python API from: [https://www.stereolabs.com/developers](https://www.stereolabs.com/developers)

### Setup Instructions

#### 1. Clone Repositories

```bash
# Clone arnavk23's PVCNN fork (which includes ZED support)
git clone https://github.com/arnavk23/pvcnn.git
cd pvcnn
```

> Use this version — it includes `zed_inference.py`, `zed_helpers.py`, and modified model wrappers.

#### 2. Download ShapeNet Part Dataset

```bash
bash data/download_shapenet_part.sh
```

This will download and extract the dataset to `data/ShapeNetPart`.

#### 3. Train or Use Pretrained Model

Train a PVCNN model on ShapeNet:

```bash
python train.py --config configs/shapenet/pvcnn.yaml
```

Or download a pretrained checkpoint and place it under `checkpoints/`.

### ZED 2i Point Cloud Inference

#### Live Inference with ZED Camera

Run real-time inference using the ZED 2i camera stream:

```bash
python zed_inference.py --ckpt checkpoints/pvcnn_shapenet.pth --num_points 2048
```

#### Optional Args:

* `--downsample`: Downsample input to 2048 points (default: True)
* `--save_output`: Save visualized output to disk

### Scripts Overview

Key scripts used :

* `zed_inference.py`: Captures point cloud from ZED camera and runs PVCNN inference.
* `zed_helpers.py`: Utilities to convert ZED depth map to 3D point cloud format and normalize.
* `train.py`: Train model on ShapeNet dataset.
* `models/pvcnn.py`: PVCNN model implementation.

### Model Architecture

PVCNN combines the strengths of **point-based** and **voxel-based** representations using a hybrid convolutional network. It achieves real-time inference performance with high accuracy.

Refer to the [original paper](https://arxiv.org/abs/2007.08500) for more on the architecture.

### Evaluation on ShapeNet

```bash
python evaluate.py --config configs/shapenet/pvcnn.yaml --ckpt checkpoints/pvcnn_shapenet.pth
```

### References

* [MIT HAN Lab PVCNN](https://github.com/mit-han-lab/pvcnn)
* [ZED SDK Documentation](https://www.stereolabs.com/docs/)
* [ShapeNet Dataset](https://www.shapenet.org/)

---
