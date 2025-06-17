# split_gen.py
import os, json, glob, random

random.seed(0)
root = 'data/shapenet_part'
h5_files = glob.glob(os.path.join(root, '*.h5'))
names = [os.path.basename(f).replace('.h5', '') for f in h5_files]
random.shuffle(names)
n = len(names)
train = names[:int(0.8*n)]
val = names[int(0.8*n):int(0.9*n)]
test = names[int(0.9*n):]

os.makedirs(os.path.join(root, 'train_test_split'), exist_ok=True)
for name, lst in zip(['train','val','test'], [train, val, test]):
    with open(os.path.join(root, 'train_test_split', f'shuffled_{name}_file_list.json'),'w') as f:
        json.dump([n + '.h5' for n in lst], f)
print("Train/Val/Test splits generated.")

