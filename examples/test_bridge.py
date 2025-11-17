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
from app.instructions.gcode import Gcode
from app.configs.printer import Printer

# Load the proper bridge G-code (40mm bridge with proper gap)
gcode_path = 'examples/bridge_proper.gcode'

# First, check how many movements are parsed
print("\n=== Checking G-code parsing ===")
print(f"Testing with: {gcode_path}")
printer_config = Printer(config_path='examples/printer_settings.json')
gcode = Gcode(gcode_path=gcode_path, printer=printer_config)
gcode.read()
print(f"Total movements parsed: {len(gcode.movements)}")
print(f"Total filament coordinates: {len(gcode.filaments_coordinates)}")
print(f"\nLast 5 movements:")
for i, mov in enumerate(gcode.movements[-5:]):
    print(f"  {len(gcode.movements)-5+i}: X={mov[0]:.2f}, Y={mov[1]:.2f}, Z={mov[2]:.2f}, E={mov[3]:.6f}, V={mov[4]:.2f}")
print(f"\nLast 5 filament coordinates:")
for i, fil in enumerate(gcode.filaments_coordinates[-5:]):
    print(f"  {len(gcode.filaments_coordinates)-5+i}: start=({fil[0][0]:.2f}, {fil[0][1]:.2f}, {fil[0][2]:.2f}), end=({fil[1][0]:.2f}, {fil[1][1]:.2f}, {fil[1][2]:.2f}), vol={fil[2]:.6f}")
print("="*50 + "\n")

print("\n" + "="*70)
print("LONG BRIDGE SIMULATION TEST (40mm bridge)")
print("="*70)

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
    "simulation_name": "LongBridge_NoPhysics",
    "results_folder": "Results_longbridge_baseline",
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
    "simulation_name": "LongBridge_WithPhysics",
    "results_folder": "Results_longbridge_physics",
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
    
    # Export detailed CSV first (before text report that might have encoding issues)
    csv_path = "Results_longbridge_physics/longbridge_segment_analysis.csv"
    os.makedirs("Results_longbridge_physics", exist_ok=True)
    
    analyzer = PrintAnalysis(material_type="PLA")
    analysis = analyzer.analyze_simulation(physics_data)
    analyzer.export_detailed_csv(physics_data['segments'], csv_path)
    print(f"Detailed segment data exported: {csv_path}")
    
    # Try to print text report (may have encoding issues on Windows)
    try:
        print(analyzer.generate_text_report(analysis))
    except UnicodeEncodeError:
        print("(Text report skipped due to encoding issues - see CSV for details)")
    
    # Show bridge details from droop analysis
    bridge_stats = analysis.get('droop_analysis', {}).get('bridge_stats', {})
    if bridge_stats.get('total_bridges', 0) > 0:
        print(f"\n{'='*70}")
        print("BRIDGE DETAILS")
        print(f"{'='*70}")
        print(f"Total bridges found: {bridge_stats['total_bridges']}")
        print(f"Longest bridge: {bridge_stats['longest_bridge']:.2f} mm")
        print(f"Average bridge length: {bridge_stats['avg_bridge_length']:.2f} mm")
        
        # Show max strand length from stats
        stats = analysis.get('statistics', {})
        if 'max_strand_length' in stats:
            print(f"\nMaximum continuous unsupported strand: {stats['max_strand_length']:.2f} mm")

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
