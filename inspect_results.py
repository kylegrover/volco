import numpy as np
import os

results_path = os.path.join("Results_volco", "Test_volco_voxels.npz")
if not os.path.exists(results_path):
    print(f"File not found: {results_path}")
    exit(1)

data = np.load(results_path)
print("Keys:", data.files)

if 'voxels' in data:
    voxels = data['voxels']
    print("Voxels shape:", voxels.shape)
    
    # Assuming voxels is a boolean or integer grid where > 0 means occupied
    # Or it might be a list of coordinates?
    # Let's check the type and shape first.
    
    if voxels.ndim == 3:
        # It's a grid
        indices = np.argwhere(voxels > 0)
        if len(indices) == 0:
            print("No occupied voxels found.")
        else:
            z_indices = indices[:, 2] # Assuming (x, y, z) or (z, y, x)? 
            # Usually numpy is (z, y, x) or (x, y, z). 
            # Let's print min/max for all dims to guess.
            print("Min indices:", indices.min(axis=0))
            print("Max indices:", indices.max(axis=0))
            
            # Let's assume the last dimension is Z for now, or check the bounds.
            # Bridge is at Z=2.2. Voxel size 0.05. Index ~44.
            
            # If it's (x, y, z)
            z_coords = indices[:, 2]
            print(f"Z indices (assuming dim 2): min={z_coords.min()}, max={z_coords.max()}, mean={z_coords.mean()}")
            
            # If it's (z, y, x)
            z_coords_0 = indices[:, 0]
            print(f"Z indices (assuming dim 0): min={z_coords_0.min()}, max={z_coords_0.max()}, mean={z_coords_0.mean()}")

    else:
        print("Voxels array is not 3D.")
else:
    print("No 'voxels' key found.")
