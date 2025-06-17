from torch.utils.data import DataLoader
from s3dis_pkl_dataset import S3DISPKLDataset

dataset = S3DISPKLDataset('/home/intern/Downloads/data/s3dis/pointcnn')
loader = DataLoader(dataset, batch_size=1, shuffle=True)

for batch in loader:
    print(batch['coord'].shape, batch['features'].shape, batch['label'].shape)
    break

