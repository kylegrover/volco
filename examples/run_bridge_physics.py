import os
import sys
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from volco import run_simulation

def main():
    print("Running Bridge Physics Simulation...")
    
    # Run VOLCO simulation
    output = run_simulation(
        gcode_path='examples/gcode_example.gcode',
        printer_config_path='examples/printer_settings.json',
        sim_config_path='examples/simulation_settings_physics_bridge.json'
    )
    
    print("Simulation complete.")
    print(f"Results saved to: {output._simulation.results_folder}")
    
    # Check if we have filled voxels
    voxel_matrix = output.cropped_voxel_space
    filled_voxels = np.count_nonzero(voxel_matrix)
    print(f"Filled voxels: {filled_voxels}")
    
    if filled_voxels == 0:
        print("WARNING: No voxels filled! Simulation might have failed or physics parameters are too extreme.")

if __name__ == "__main__":
    main()
