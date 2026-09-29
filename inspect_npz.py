import numpy as np
import os

file_path = 'Results_volco/Test_volco_voxels.npz'
if not os.path.exists(file_path):
    print(f"File not found: {file_path}")
else:
    data = np.load(file_path)
    print("Keys in npz file:", data.files)
    for key in data.files:
        print(f"Shape of {key}: {data[key].shape}")
