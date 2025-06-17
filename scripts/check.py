import pickle

with open('/home/intern/Downloads/data/s3dis/pointcnn/Area_1/hallway_3/archive/data.pkl', 'rb') as f:
    data = pickle.load(f)

print(type(data))
if isinstance(data, dict):
    for k in data:
        print(f"{k}: {type(data[k])}, shape={getattr(data[k], 'shape', 'N/A')}")
else:
    print(data)

