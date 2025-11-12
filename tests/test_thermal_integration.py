"""
Basic integration test for thermal/droop simulation.

Run this to verify the new physics simulation features work correctly.
"""

import sys
import os

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from volco import run_simulation


def test_basic_thermal():
    """Test basic thermal simulation"""
    print("Testing basic thermal simulation...")
    
    gcode = """M83
G0 X10 Y10 Z0.3
G1 F1000 X12.0 E0.1
G1 Y12.0 E0.1
"""
    
    printer_config = {
        "nozzle_diameter": 0.4,
        "feedstock_filament_diameter": 1.75,
        "nozzle_jerk_speed": 8.0,
        "extruder_jerk_speed": 5.0,
        "nozzle_acceleration": 500.0,
        "extruder_acceleration": 1000.0,
    }
    
    sim_config = {
        "voxel_size": 0.2,
        "step_size": 0.5,
        "x_offset": 2.0,
        "y_offset": 2.0,
        "z_offset": 0,
        "sphere_z_offset": 0.2,
        "simulation_name": "Test",
        "results_folder": "test_results",
        "radius_increment": 0.1,
        "solver_tolerance": 0.0001,
        "x_crop": ["all", "all"],
        "y_crop": ["all", "all"],
        "z_crop": [0.0, "all"],
        "consider_acceleration": False,
        "stl_ascii": False,
        "enable_thermal_simulation": True,
        "enable_droop_simulation": False,
        "material_type": "PLA"
    }
    
    try:
        output = run_simulation(
            gcode=gcode,
            printer_config=printer_config,
            sim_config=sim_config
        )
        
        physics_data = output.voxel_space.get_physics_simulation_data()
        
        assert physics_data is not None, "Physics data should not be None"
        assert 'segments' in physics_data, "Should have segments"
        assert 'statistics' in physics_data, "Should have statistics"
        assert len(physics_data['segments']) > 0, "Should have at least one segment"
        
        print(f"✓ Thermal simulation works!")
        print(f"  - Generated {len(physics_data['segments'])} segments")
        print(f"  - Statistics: {physics_data['statistics']}")
        return True
        
    except Exception as e:
        print(f"✗ Thermal simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_basic_droop():
    """Test basic droop simulation"""
    print("\nTesting basic droop simulation...")
    
    # Create a bridge scenario
    gcode = """M83
G0 X10 Y10 Z0.3
G1 F1000 X12.0 E0.1
G0 X20.0 Y10.0
G1 X22.0 E0.1
G0 X12.0 Z0.6
G1 X20.0 E0.3
"""
    
    printer_config = {
        "nozzle_diameter": 0.4,
        "feedstock_filament_diameter": 1.75,
        "nozzle_jerk_speed": 8.0,
        "extruder_jerk_speed": 5.0,
        "nozzle_acceleration": 500.0,
        "extruder_acceleration": 1000.0,
    }
    
    sim_config = {
        "voxel_size": 0.2,
        "step_size": 0.5,
        "x_offset": 2.0,
        "y_offset": 2.0,
        "z_offset": 0,
        "sphere_z_offset": 0.2,
        "simulation_name": "Test",
        "results_folder": "test_results",
        "radius_increment": 0.1,
        "solver_tolerance": 0.0001,
        "x_crop": ["all", "all"],
        "y_crop": ["all", "all"],
        "z_crop": [0.0, "all"],
        "consider_acceleration": False,
        "stl_ascii": False,
        "enable_thermal_simulation": False,
        "enable_droop_simulation": True,
        "material_type": "PLA"
    }
    
    try:
        output = run_simulation(
            gcode=gcode,
            printer_config=printer_config,
            sim_config=sim_config
        )
        
        physics_data = output.voxel_space.get_physics_simulation_data()
        
        assert physics_data is not None, "Physics data should not be None"
        assert physics_data['statistics']['unsupported_segments'] > 0, "Should have unsupported segments in bridge"
        
        # Check for droop in unsupported segments
        drooped_count = sum(1 for seg in physics_data['segments'] if seg.droop_offset > 0)
        
        print(f"✓ Droop simulation works!")
        print(f"  - Unsupported segments: {physics_data['statistics']['unsupported_segments']}")
        print(f"  - Segments with droop: {drooped_count}")
        return True
        
    except Exception as e:
        print(f"✗ Droop simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_combined():
    """Test both thermal and droop together"""
    print("\nTesting combined thermal + droop simulation...")
    
    gcode = """M83
G0 X10 Y10 Z0.3
G1 F1000 X15.0 E0.2
G1 Y15.0 E0.2
"""
    
    printer_config = {
        "nozzle_diameter": 0.4,
        "feedstock_filament_diameter": 1.75,
        "nozzle_jerk_speed": 8.0,
        "extruder_jerk_speed": 5.0,
        "nozzle_acceleration": 500.0,
        "extruder_acceleration": 1000.0,
    }
    
    sim_config = {
        "voxel_size": 0.2,
        "step_size": 0.5,
        "x_offset": 2.0,
        "y_offset": 2.0,
        "z_offset": 0,
        "sphere_z_offset": 0.2,
        "simulation_name": "Test",
        "results_folder": "test_results",
        "radius_increment": 0.1,
        "solver_tolerance": 0.0001,
        "x_crop": ["all", "all"],
        "y_crop": ["all", "all"],
        "z_crop": [0.0, "all"],
        "consider_acceleration": False,
        "stl_ascii": False,
        "enable_thermal_simulation": True,
        "enable_droop_simulation": True,
        "material_type": "PLA"
    }
    
    try:
        output = run_simulation(
            gcode=gcode,
            printer_config=printer_config,
            sim_config=sim_config
        )
        
        physics_data = output.voxel_space.get_physics_simulation_data()
        
        assert physics_data is not None, "Physics data should not be None"
        assert physics_data['thermal_enabled'], "Thermal should be enabled"
        assert physics_data['droop_enabled'], "Droop should be enabled"
        
        print(f"✓ Combined simulation works!")
        print(f"  - Material: {physics_data['material_type']}")
        print(f"  - Thermal enabled: {physics_data['thermal_enabled']}")
        print(f"  - Droop enabled: {physics_data['droop_enabled']}")
        return True
        
    except Exception as e:
        print(f"✗ Combined simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_analysis():
    """Test analysis module"""
    print("\nTesting analysis module...")
    
    from app.reporter.analysis import PrintAnalysis
    
    gcode = """M83
G0 X10 Y10 Z0.3
G1 F1000 X20.0 E0.3
"""
    
    printer_config = {
        "nozzle_diameter": 0.4,
        "feedstock_filament_diameter": 1.75,
        "nozzle_jerk_speed": 8.0,
        "extruder_jerk_speed": 5.0,
        "nozzle_acceleration": 500.0,
        "extruder_acceleration": 1000.0,
    }
    
    sim_config = {
        "voxel_size": 0.2,
        "step_size": 0.5,
        "x_offset": 2.0,
        "y_offset": 2.0,
        "z_offset": 0,
        "sphere_z_offset": 0.2,
        "simulation_name": "Test",
        "results_folder": "test_results",
        "radius_increment": 0.1,
        "solver_tolerance": 0.0001,
        "x_crop": ["all", "all"],
        "y_crop": ["all", "all"],
        "z_crop": [0.0, "all"],
        "consider_acceleration": False,
        "stl_ascii": False,
        "enable_thermal_simulation": True,
        "enable_droop_simulation": True,
        "material_type": "PLA"
    }
    
    try:
        output = run_simulation(
            gcode=gcode,
            printer_config=printer_config,
            sim_config=sim_config
        )
        
        physics_data = output.voxel_space.get_physics_simulation_data()
        
        analyzer = PrintAnalysis(material_type="PLA")
        analysis = analyzer.analyze_simulation(physics_data)
        
        assert 'printability_score' in analysis, "Should have printability score"
        assert 'severity' in analysis, "Should have severity"
        assert 'recommendations' in analysis, "Should have recommendations"
        
        # Generate report
        report = analyzer.generate_text_report(analysis)
        assert len(report) > 0, "Report should not be empty"
        
        print(f"✓ Analysis works!")
        print(f"  - Printability score: {analysis['printability_score']:.1f}/100")
        print(f"  - Severity: {analysis['severity']}")
        print(f"  - Recommendations: {len(analysis['recommendations'])}")
        return True
        
    except Exception as e:
        print(f"✗ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 70)
    print("THERMAL/DROOP SIMULATION INTEGRATION TESTS")
    print("=" * 70)
    
    results = []
    
    results.append(("Thermal simulation", test_basic_thermal()))
    results.append(("Droop simulation", test_basic_droop()))
    results.append(("Combined simulation", test_combined()))
    results.append(("Analysis module", test_analysis()))
    
    print("\n" + "=" * 70)
    print("TEST RESULTS SUMMARY")
    print("=" * 70)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print("\n⚠️  Some tests failed. Please check the output above.")
        sys.exit(1)
