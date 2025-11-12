"""
Direct comparison: same G-code and settings, with and without physics
"""
import json
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from volco import run_simulation

# Load the same G-code
gcode_path = 'examples/gcode_example.gcode'
with open(gcode_path, 'r') as f:
    gcode_content = f.read()

printer_config = {
    "nozzle_jerk_speed": 8.0,
    "extruder_jerk_speed": 5.0,
    "nozzle_acceleration": 500.0,
    "extruder_acceleration": 1000.0,
    "feedstock_filament_diameter": 1.75,
    "nozzle_diameter": 0.4
}

# BASELINE: No physics
print("Running BASELINE (no physics)...")
baseline_config = {
    "voxel_size": 0.1,
    "step_size": 0.2,
    "x_offset": 2.0,
    "y_offset": 2.0,
    "z_offset": 0,
    "sphere_z_offset": 0.2,
    "simulation_name": "Baseline_NoPhysics",
    "results_folder": "Results_baseline",
    "radius_increment": 0.1,
    "solver_tolerance": 0.0001,
    "x_crop": ["all", "all"],
    "y_crop": ["all", "all"],
    "z_crop": [0.0, "all"],
    "consider_acceleration": False,
    "stl_ascii": True,
    # NO PHYSICS
    "enable_thermal_simulation": False,
    "enable_droop_simulation": False,
}

output_baseline = run_simulation(
    gcode=gcode_content,
    printer_config=printer_config,
    sim_config=baseline_config
)
stl_baseline = output_baseline.export_mesh_to_stl()
print(f"Baseline STL: {stl_baseline}\n")

# PHYSICS: With thermal and droop
print("Running WITH PHYSICS...")
physics_config = {
    "voxel_size": 0.1,
    "step_size": 0.2,
    "x_offset": 2.0,
    "y_offset": 2.0,
    "z_offset": 0,
    "sphere_z_offset": 0.2,
    "simulation_name": "Physics_Enabled",
    "results_folder": "Results_physics",
    "radius_increment": 0.1,
    "solver_tolerance": 0.0001,
    "x_crop": ["all", "all"],
    "y_crop": ["all", "all"],
    "z_crop": [0.0, "all"],
    "consider_acceleration": False,
    "stl_ascii": True,
    # WITH PHYSICS
    "enable_thermal_simulation": True,
    "enable_droop_simulation": True,
    "material_type": "PLA"
}

output_physics = run_simulation(
    gcode=gcode_content,
    printer_config=printer_config,
    sim_config=physics_config
)
stl_physics = output_physics.export_mesh_to_stl()
print(f"Physics STL: {stl_physics}\n")

# Compare file sizes
import os
baseline_size = os.path.getsize(stl_baseline)
physics_size = os.path.getsize(stl_physics)

print("=" * 70)
print("COMPARISON RESULTS")
print("=" * 70)
print(f"Baseline (no physics): {baseline_size:,} bytes")
print(f"Physics (thermal+droop): {physics_size:,} bytes")
print(f"Size ratio: {physics_size/baseline_size*100:.1f}%")
print(f"Difference: {abs(physics_size-baseline_size):,} bytes")
print("\nExpected: Files should be very similar in size (>90%) when droop is minimal")
print("=" * 70)
