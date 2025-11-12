"""
Test bridge example with and without physics
"""
import json
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from volco import run_simulation
from app.reporter.analysis import PrintAnalysis

# Load the bridge G-code
gcode_path = 'examples/bridge_example.gcode'
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

print("=" * 70)
print("BRIDGE SIMULATION TEST")
print("=" * 70)

# BASELINE: No physics
print("\n1. Running BASELINE (no physics)...")
baseline_config = {
    "voxel_size": 0.1,
    "step_size": 0.2,
    "x_offset": 2.0,
    "y_offset": 2.0,
    "z_offset": 0,
    "sphere_z_offset": 0.2,
    "simulation_name": "Bridge_NoPhysics",
    "results_folder": "Results_bridge_baseline",
    "radius_increment": 0.1,
    "solver_tolerance": 0.0001,
    "x_crop": ["all", "all"],
    "y_crop": ["all", "all"],
    "z_crop": [0.0, "all"],
    "consider_acceleration": False,
    "stl_ascii": True,
    "enable_thermal_simulation": False,
    "enable_droop_simulation": False,
}

output_baseline = run_simulation(
    gcode=gcode_content,
    printer_config=printer_config,
    sim_config=baseline_config
)
stl_baseline = output_baseline.export_mesh_to_stl()
print(f"   Baseline STL: {stl_baseline}")

# PHYSICS: With thermal and droop
print("\n2. Running WITH PHYSICS (thermal + droop)...")
physics_config = {
    "voxel_size": 0.1,
    "step_size": 0.2,
    "x_offset": 2.0,
    "y_offset": 2.0,
    "z_offset": 0,
    "sphere_z_offset": 0.2,
    "simulation_name": "Bridge_WithPhysics",
    "results_folder": "Results_bridge_physics",
    "radius_increment": 0.1,
    "solver_tolerance": 0.0001,
    "x_crop": ["all", "all"],
    "y_crop": ["all", "all"],
    "z_crop": [0.0, "all"],
    "consider_acceleration": False,
    "stl_ascii": True,
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
print(f"   Physics STL: {stl_physics}")

# Analyze physics data
physics_data = output_physics.voxel_space.get_physics_simulation_data()

if physics_data:
    print("\n" + "=" * 70)
    print("PHYSICS ANALYSIS")
    print("=" * 70)
    
    analyzer = PrintAnalysis(material_type="PLA")
    analysis = analyzer.analyze_simulation(physics_data)
    print(analyzer.generate_text_report(analysis))
    
    # Export detailed CSV
    csv_path = "Results_bridge_physics/bridge_segment_analysis.csv"
    os.makedirs("Results_bridge_physics", exist_ok=True)
    analyzer.export_detailed_csv(physics_data['segments'], csv_path)
    print(f"Detailed segment data: {csv_path}")
    
    # Show bridge details
    bridges = analysis['bridges']
    if bridges:
        print(f"\n{'='*70}")
        print("BRIDGE DETAILS")
        print(f"{'='*70}")
        print(f"Total bridges found: {len(bridges)}")
        print(f"Longest bridge: {analysis['max_bridge_length']:.2f} mm")
        
        # Find the longest bridge
        longest = max(bridges, key=lambda b: b['length'])
        print(f"\nLongest bridge segment #{longest['segment_id']}:")
        print(f"  Position: ({longest['start_pos'][0]:.1f}, {longest['start_pos'][1]:.1f}, {longest['start_pos'][2]:.2f})")
        print(f"           → ({longest['end_pos'][0]:.1f}, {longest['end_pos'][1]:.1f}, {longest['end_pos'][2]:.2f})")
        print(f"  Length: {longest['length']:.2f} mm")
        print(f"  Droop: {longest['droop']:.3f} mm")
        print(f"  Temperature: {longest['temperature']:.1f}°C")

# Compare file sizes
baseline_size = os.path.getsize(stl_baseline)
physics_size = os.path.getsize(stl_physics)

print("\n" + "=" * 70)
print("FILE SIZE COMPARISON")
print("=" * 70)
print(f"Baseline (no physics): {baseline_size:,} bytes")
print(f"Physics (thermal+droop): {physics_size:,} bytes")
print(f"Size ratio: {physics_size/baseline_size*100:.1f}%")
print(f"Difference: {abs(physics_size-baseline_size):,} bytes")
if abs(physics_size - baseline_size) / baseline_size > 0.05:
    print("\n⚠️  Significant difference detected - droop physics is affecting the mesh!")
else:
    print("\n✓ Similar sizes - minimal droop (expected for well-supported prints)")
print("=" * 70)
