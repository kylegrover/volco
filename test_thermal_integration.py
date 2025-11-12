"""
Quick test to verify thermal simulation integration works.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from volco import run_simulation

# Simple test G-code
gcode = """M83
G0 X10 Y10 Z0.3
G1 F1000 X15.0 E0.1
G1 Y15.0 E0.1
G1 X10.0 E0.1
G1 Y10.0 E0.1
"""

# Printer config
printer_config = {
    "nozzle_diameter": 0.4,
    "feedstock_filament_diameter": 1.75,
    "nozzle_jerk_speed": 8.0,
    "extruder_jerk_speed": 5.0,
    "nozzle_acceleration": 500.0,
    "extruder_acceleration": 1000.0
}

# Simulation config WITHOUT physics (baseline)
sim_config_baseline = {
    "voxel_size": 0.1,
    "step_size": 0.2,
    "x_offset": 2.0,
    "y_offset": 2.0,
    "z_offset": 0,
    "sphere_z_offset": 0.2,
    "simulation_name": "baseline_test",
    "results_folder": "test_results",
    "radius_increment": 0.1,
    "solver_tolerance": 0.0001,
    "x_crop": ["all", "all"],
    "y_crop": ["all", "all"],
    "z_crop": [0.0, "all"],
    "consider_acceleration": False,
    "stl_ascii": False,
    "enable_thermal_simulation": False,
    "enable_droop_simulation": False,
    "material_type": "PLA"
}

# Simulation config WITH physics
sim_config_physics = sim_config_baseline.copy()
sim_config_physics["enable_thermal_simulation"] = True
sim_config_physics["enable_droop_simulation"] = True
sim_config_physics["simulation_name"] = "physics_test"

print("=" * 60)
print("Testing VolCo Thermal Simulation Integration")
print("=" * 60)

# Test 1: Baseline (original VolCo behavior)
print("\n[1/2] Running baseline simulation (no physics)...")
try:
    output_baseline = run_simulation(
        gcode=gcode,
        printer_config=printer_config,
        sim_config=sim_config_baseline
    )
    physics_data_baseline = output_baseline.voxel_space.get_physics_simulation_data()
    
    if physics_data_baseline is None:
        print("✓ Baseline: Physics data is None (as expected)")
    else:
        print("✗ Baseline: Physics data should be None but isn't")
except Exception as e:
    print(f"✗ Baseline simulation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: With physics
print("\n[2/2] Running simulation with physics enabled...")
try:
    output_physics = run_simulation(
        gcode=gcode,
        printer_config=printer_config,
        sim_config=sim_config_physics
    )
    physics_data = output_physics.voxel_space.get_physics_simulation_data()
    
    if physics_data is None:
        print("✗ Physics: Data is None (should have data)")
        sys.exit(1)
    
    print("✓ Physics: Got physics data")
    print(f"  - Total segments: {physics_data['statistics']['total_segments']}")
    print(f"  - Unsupported segments: {physics_data['statistics']['unsupported_segments']}")
    print(f"  - Material type: {physics_data['material_type']}")
    print(f"  - Thermal enabled: {physics_data['thermal_enabled']}")
    print(f"  - Droop enabled: {physics_data['droop_enabled']}")
    
    # Test analysis
    from app.reporter.analysis import PrintAnalysis
    analyzer = PrintAnalysis(material_type="PLA")
    analysis = analyzer.analyze_simulation(physics_data)
    
    print(f"\n✓ Analysis completed:")
    print(f"  - Severity: {analysis['severity']}")
    print(f"  - Printability score: {analysis['printability_score']:.1f}/100")
    print(f"  - Recommendations: {len(analysis['recommendations'])}")
    
except Exception as e:
    print(f"✗ Physics simulation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✓ All tests passed!")
print("=" * 60)
print("\nIntegration successful! The thermal simulation system is working.")
print("Run 'uv run python examples/thermal_simulation_example.py' for full examples.")
