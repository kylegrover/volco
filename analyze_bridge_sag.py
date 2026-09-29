import numpy as np
# import matplotlib.pyplot as plt

def analyze_sag():
    file_path = 'Results_volco/Test_volco_voxels.npz'
    data = np.load(file_path)
    voxels = data['voxels']
    voxel_size = data['voxel_size']
    
    print(f"Voxel size: {voxel_size}")
    print(f"Grid shape: {voxels.shape}")

    # The bridge seems to be along Y=4.0 (approx).
    # Let's find the Y index corresponding to Y=4.0
    # Assuming grid starts at 0,0,0
    
    # We can project the max Z for each X, Y
    # But first let's just find where the mass is.
    
    # Sum along Z to find the XY footprint
    xy_projection = np.sum(voxels, axis=2)
    
    # Find the Y indices with significant mass
    y_indices = np.where(np.sum(xy_projection, axis=0) > 0)[0]
    print(f"Active Y indices: {y_indices}")
    
    if len(y_indices) == 0:
        print("No mass found in the grid.")
        return

    # Calculate max Z for each (X, Y)
    # voxels shape: (X, Y, Z)
    
    # Create a height map
    height_map = np.zeros((voxels.shape[0], voxels.shape[1]))
    
    for x in range(voxels.shape[0]):
        for y in range(voxels.shape[1]):
            col = voxels[x, y, :]
            if np.sum(col) > 0:
                # Find the highest filled voxel
                # We can use the center of mass of the highest cluster, or just the max index
                # Let's use the max index for simplicity
                z_indices = np.where(col > 0)[0]
                max_z_idx = z_indices[-1]
                height_map[x, y] = max_z_idx * voxel_size
            else:
                height_map[x, y] = 0

    print("\nHeight Map Analysis:")
    # Filter for the bridge area (Z > 0.5 to avoid base layer)
    bridge_mask = height_map > 0.5
    
    if not np.any(bridge_mask):
        print("No bridge detected above Z=0.5")
        return

    bridge_heights = height_map[bridge_mask]
    print(f"Max Height: {np.max(bridge_heights):.4f}")
    print(f"Min Height (on bridge): {np.min(bridge_heights):.4f}")
    print(f"Mean Height: {np.mean(bridge_heights):.4f}")
    
    # Let's try to visualize the bridge path
    # We know it goes from X~24 to X~4
    # X index range: 4/0.05 = 80 to 24/0.05 = 480
    
    print("\nBridge Profile (Projected Max Z along X):")
    # We project the max Z along Y for each X
    x_profile = []
    z_profile = []
    
    for x in range(height_map.shape[0]):
        col = height_map[x, :]
        if np.any(col > 0.5):
            max_z = np.max(col)
            x_profile.append(x * voxel_size)
            z_profile.append(max_z)
    
    # Binning for display
    bins = 20
    if not x_profile:
        print("No profile data.")
        return
        
    min_x = min(x_profile)
    max_x = max(x_profile)
    x_bins = np.linspace(min_x, max_x, bins)
    
    for i in range(bins-1):
        mask = (np.array(x_profile) >= x_bins[i]) & (np.array(x_profile) < x_bins[i+1])
        if np.any(mask):
            z_val = np.mean(np.array(z_profile)[mask])
            bar_len = int(z_val * 10)
            print(f"X={x_bins[i]:6.2f} | {'#' * bar_len} ({z_val:.4f})")


if __name__ == "__main__":
    analyze_sag()
