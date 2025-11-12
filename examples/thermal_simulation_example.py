"""
Thermal and Droop Simulation Example for VolCo

This script demonstrates the new thermal and droop simulation capabilities
added to VolCo for analyzing 3D printing physics.

Features demonstrated:
1. Running simulation with thermal physics
2. Running simulation with droop physics
3. Analyzing simulation results
4. Comparing with/without physics simulation
"""

import json
import sys
import os

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from volco import run_simulation
from app.reporter.analysis import PrintAnalysis


def example_1_thermal_simulation():
    """Example 1: Basic thermal simulation"""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Basic Thermal Simulation")
    print("=" * 70)
    
    # Create G-code with some bridging
    gcode_content = """M83
G0 X10 Y10 Z0.3
G1 F7200 X10.0 Y10.0 Z0.3
G1 F1000 X14.0 E0.133041
G0 F7200 X12.0 Y8.0 Z0.5
G1 F1000 Y12.0 E0.133041
G1 F7200 X10.0 Y10.0 Z0.7
G1 F1000 X12.0 E0.067
G1 F1000 X14 Y8.0 E0.1
; Add a bridge
G1 Z1.0
G1 X10.0 Y10.0
G1 X20.0 E0.3
"""
    
    # Printer configuration
    printer_config = {
        "nozzle_jerk_speed": 8.0,
        "extruder_jerk_speed": 5.0,
        "nozzle_acceleration": 500.0,
        "extruder_acceleration": 1000.0,
        "feedstock_filament_diameter": 1.75,
        "nozzle_diameter": 0.4
    }
    
    # Simulation configuration WITH thermal simulation
    sim_config = {
        "voxel_size": 0.1,
        "step_size": 0.2,
        "x_offset": 2.0,
        "y_offset": 2.0,
        "z_offset": 0,
        "sphere_z_offset": 0.2,
        "simulation_name": "Thermal_Test",
        "results_folder": "Results_thermal",
        "radius_increment": 0.1,
        "solver_tolerance": 0.0001,
        "x_crop": ["all", "all"],
        "y_crop": ["all", "all"],
        "z_crop": [0.0, "all"],
        "consider_acceleration": False,
        "stl_ascii": False,
        # NEW: Enable thermal simulation
        "enable_thermal_simulation": True,
        "enable_droop_simulation": False,
        "material_type": "PLA"
    }
    
    # Run simulation
    print("\nRunning simulation with thermal physics enabled...")
    output = run_simulation(
        gcode=gcode_content,
        printer_config=printer_config,
        sim_config=sim_config
    )
    
    # Get physics data
    physics_data = output.voxel_space.get_physics_simulation_data()
    
    if physics_data:
        print(f"\nPhysics simulation completed!")
        print(f"Total segments: {physics_data['statistics']['total_segments']}")
        print(f"Simulation time: {physics_data['statistics']['total_time']:.2f}s")
        
        # Analyze results
        analyzer = PrintAnalysis(material_type="PLA")
        analysis = analyzer.analyze_simulation(physics_data)
        
        # Print report
        print("\n" + analyzer.generate_text_report(analysis))
    
    return output, physics_data


def example_2_droop_simulation():
    """Example 2: Droop simulation with bridging"""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Droop Simulation with Bridging")
    print("=" * 70)
    
    # Create G-code with deliberate bridging
    gcode_content = """M83
; Build two towers
G0 X10 Y10 Z0.3
G1 F1000 X12.0 Y10.0 E0.1
G1 X12.0 Y12.0 E0.1
G1 X10.0 Y12.0 E0.1
G1 X10.0 Y10.0 E0.1

G0 X20.0 Y10.0
G1 X22.0 Y10.0 E0.1
G1 X22.0 Y12.0 E0.1
G1 X20.0 Y12.0 E0.1
G1 X20.0 Y10.0 E0.1

; Stack a few more layers
G0 X10.0 Y10.0 Z0.6
G1 F1000 X12.0 Y10.0 E0.1
G1 X12.0 Y12.0 E0.1
G1 X10.0 Y12.0 E0.1
G1 X10.0 Y10.0 E0.1

G0 X20.0 Y10.0
G1 X22.0 Y10.0 E0.1
G1 X22.0 Y12.0 E0.1
G1 X20.0 Y12.0 E0.1
G1 X20.0 Y10.0 E0.1

; Now bridge between towers (10mm bridge)
G0 X12.0 Y11.0 Z0.9
G1 F1000 X20.0 Y11.0 E0.3
"""
    
    printer_config = {
        "nozzle_jerk_speed": 8.0,
        "extruder_jerk_speed": 5.0,
        "nozzle_acceleration": 500.0,
        "extruder_acceleration": 1000.0,
        "feedstock_filament_diameter": 1.75,
        "nozzle_diameter": 0.4
    }
    
    # Simulation configuration WITH droop simulation
    sim_config = {
        "voxel_size": 0.1,
        "step_size": 0.2,
        "x_offset": 2.0,
        "y_offset": 2.0,
        "z_offset": 0,
        "sphere_z_offset": 0.2,
        "simulation_name": "Droop_Test",
        "results_folder": "Results_droop",
        "radius_increment": 0.1,
        "solver_tolerance": 0.0001,
        "x_crop": ["all", "all"],
        "y_crop": ["all", "all"],
        "z_crop": [0.0, "all"],
        "consider_acceleration": False,
        "stl_ascii": False,
        # NEW: Enable droop simulation
        "enable_thermal_simulation": False,
        "enable_droop_simulation": True,
        "material_type": "PLA"
    }
    
    # Run simulation
    print("\nRunning simulation with droop physics enabled...")
    output = run_simulation(
        gcode=gcode_content,
        printer_config=printer_config,
        sim_config=sim_config
    )
    
    # Get and analyze physics data
    physics_data = output.voxel_space.get_physics_simulation_data()
    
    if physics_data:
        analyzer = PrintAnalysis(material_type="PLA")
        analysis = analyzer.analyze_simulation(physics_data)
        print("\n" + analyzer.generate_text_report(analysis))
        
        # Show some detailed segment info
        print("\nDETAILED SEGMENT ANALYSIS:")
        print("(Showing first 5 segments with droop)")
        
        drooped_segments = [s for s in physics_data['segments'] if s.droop_offset > 0.01]
        for i, seg in enumerate(drooped_segments[:5]):
            print(f"\nSegment {seg.segment_id}:")
            print(f"  Position: {seg.start_pos} -> {seg.end_pos}")
            print(f"  Length: {seg.length:.2f} mm")
            print(f"  Droop: {seg.droop_offset:.3f} mm")
            print(f"  Temperature: {seg.temperature:.1f}°C")
            print(f"  Supported: {seg.support_below}")
            print(f"  Unsupported length: {seg.unsupported_length:.2f} mm")
    
    return output, physics_data


def example_3_combined_simulation():
    """Example 3: Both thermal and droop simulation"""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Combined Thermal + Droop Simulation")
    print("=" * 70)
    
    # Use existing G-code file if available
    gcode_path = 'gcode_example.gcode'
    if os.path.exists(gcode_path):
        print(f"Using G-code from {gcode_path}")
        with open(gcode_path, 'r') as f:
            gcode_content = f.read()
    else:
        # Fallback G-code
        print("Using generated G-code")
        gcode_content = """M83
G0 X10 Y10 Z0.3
G1 F1000 X15.0 E0.2
G1 Y15.0 E0.2
G1 X10.0 E0.2
G1 Y10.0 E0.2
G0 Z0.6
G1 X15.0 E0.2
G1 Y15.0 E0.2
G1 X10.0 E0.2
G1 Y10.0 E0.2
"""
    
    printer_config = {
        "nozzle_jerk_speed": 8.0,
        "extruder_jerk_speed": 5.0,
        "nozzle_acceleration": 500.0,
        "extruder_acceleration": 1000.0,
        "feedstock_filament_diameter": 1.75,
        "nozzle_diameter": 0.4
    }
    
    # Simulation configuration with BOTH enabled
    sim_config = {
        "voxel_size": 0.1,
        "step_size": 0.2,
        "x_offset": 2.0,
        "y_offset": 2.0,
        "z_offset": 0,
        "sphere_z_offset": 0.2,
        "simulation_name": "Combined_Test",
        "results_folder": "Results_combined",
        "radius_increment": 0.1,
        "solver_tolerance": 0.0001,
        "x_crop": ["all", "all"],
        "y_crop": ["all", "all"],
        "z_crop": [0.0, "all"],
        "consider_acceleration": False,
        "stl_ascii": True,
        # NEW: Enable BOTH thermal and droop
        "enable_thermal_simulation": True,
        "enable_droop_simulation": True,
        "material_type": "PLA"
    }
    
    print("\nRunning simulation with thermal + droop physics...")
    output = run_simulation(
        gcode=gcode_content,
        printer_config=printer_config,
        sim_config=sim_config
    )
    
    # Analyze and export
    physics_data = output.voxel_space.get_physics_simulation_data()
    
    if physics_data:
        analyzer = PrintAnalysis(material_type="PLA")
        analysis = analyzer.analyze_simulation(physics_data)
        print("\n" + analyzer.generate_text_report(analysis))
        
        # Export detailed CSV
        csv_path = "Results_combined/segment_analysis.csv"
        os.makedirs("Results_combined", exist_ok=True)
        analyzer.export_detailed_csv(physics_data['segments'], csv_path)
        print(f"\nDetailed segment data exported to {csv_path}")
        
        # Export STL
        stl_path = output.export_mesh_to_stl()
        print(f"Mesh exported to {stl_path}")
    
    return output, physics_data


def example_4_compare_materials():
    """Example 4: Compare different materials"""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Material Comparison (PLA vs ABS)")
    print("=" * 70)
    
    # Simple bridging test
    gcode_content = """M83
G0 X10 Y10 Z0.3
G1 F1000 X10.0 Y10.0 Z0.3
G1 X20.0 E0.3
G1 Y20.0 E0.3
G1 X10.0 E0.3
G1 Y10.0 E0.3
; Add bridge
G0 X10.0 Y15.0 Z0.6
G1 X20.0 E0.3
"""
    
    printer_config = {
        "nozzle_jerk_speed": 8.0,
        "extruder_jerk_speed": 5.0,
        "nozzle_acceleration": 500.0,
        "extruder_acceleration": 1000.0,
        "feedstock_filament_diameter": 1.75,
        "nozzle_diameter": 0.4
    }
    
    results = {}
    
    for material in ['PLA', 'ABS']:
        print(f"\n--- Testing with {material} ---")
        
        sim_config = {
            "voxel_size": 0.1,
            "step_size": 0.2,
            "x_offset": 2.0,
            "y_offset": 2.0,
            "z_offset": 0,
            "sphere_z_offset": 0.2,
            "simulation_name": f"{material}_Test",
            "results_folder": f"Results_{material}",
            "radius_increment": 0.1,
            "solver_tolerance": 0.0001,
            "x_crop": ["all", "all"],
            "y_crop": ["all", "all"],
            "z_crop": [0.0, "all"],
            "consider_acceleration": False,
            "stl_ascii": False,
            "enable_thermal_simulation": True,
            "enable_droop_simulation": True,
            "material_type": material
        }
        
        output = run_simulation(
            gcode=gcode_content,
            printer_config=printer_config,
            sim_config=sim_config
        )
        
        physics_data = output.voxel_space.get_physics_simulation_data()
        
        if physics_data:
            analyzer = PrintAnalysis(material_type=material)
            analysis = analyzer.analyze_simulation(physics_data)
            results[material] = analysis
            
            print(f"\n{material} Results:")
            print(f"  Printability Score: {analysis['printability_score']:.1f}/100")
            print(f"  Max Droop: {analysis['droop_analysis']['droop_stats']['max_droop']:.3f} mm")
            print(f"  Severity: {analysis['severity']}")
    
    # Compare results
    print("\n" + "=" * 70)
    print("MATERIAL COMPARISON SUMMARY:")
    print("=" * 70)
    for material, analysis in results.items():
        print(f"\n{material}:")
        print(f"  Printability: {analysis['printability_score']:.1f}/100")
        print(f"  Max Droop: {analysis['droop_analysis']['droop_stats']['max_droop']:.3f} mm")
        print(f"  Severity: {analysis['severity']}")
    
    return results


if __name__ == "__main__":
    print("=" * 70)
    print("VolCo Thermal & Droop Simulation Examples")
    print("=" * 70)
    print("\nThese examples demonstrate the new physics simulation capabilities.")
    print("Choose an example to run:")
    print("  1. Basic thermal simulation")
    print("  2. Droop simulation with bridging")
    print("  3. Combined thermal + droop simulation")
    print("  4. Material comparison (PLA vs ABS)")
    print("  5. Run all examples")
    
    choice = input("\nEnter choice (1-5): ").strip()
    
    try:
        if choice == "1":
            example_1_thermal_simulation()
        elif choice == "2":
            example_2_droop_simulation()
        elif choice == "3":
            example_3_combined_simulation()
        elif choice == "4":
            example_4_compare_materials()
        elif choice == "5":
            example_1_thermal_simulation()
            example_2_droop_simulation()
            example_3_combined_simulation()
            example_4_compare_materials()
        else:
            print("Invalid choice. Running example 3 (combined simulation)...")
            example_3_combined_simulation()
    except Exception as e:
        print(f"\nError running example: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("Examples completed!")
    print("=" * 70)
