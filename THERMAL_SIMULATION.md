# Thermal and Droop Simulation in VolCo

## Overview

VolCo has been extended with physics-based simulation capabilities to model the thermal and mechanical behavior of extruded filament during 3D printing. This helps predict issues like:

- **Droop/Sagging**: Unsupported filament sagging under gravity
- **Poor Fusion**: Insufficient layer bonding due to temperature
- **Bridging Problems**: Long spans without support
- **Cooling Issues**: Material solidifying too quickly or slowly

## Quick Start

### Basic Usage

Enable thermal and/or droop simulation by adding parameters to your simulation configuration:

```python
from volco import run_simulation

sim_config = {
    # ... standard VolCo parameters ...
    "enable_thermal_simulation": True,  # Enable temperature tracking
    "enable_droop_simulation": True,    # Enable gravity/droop physics
    "material_type": "PLA"              # Material: PLA, ABS, or PETG
}

output = run_simulation(
    gcode_path='my_file.gcode',
    printer_config_path='printer.json',
    sim_config=sim_config  # Can pass dict directly
)

# Access physics simulation data
physics_data = output.voxel_space.get_physics_simulation_data()

# Analyze results
from app.reporter.analysis import PrintAnalysis
analyzer = PrintAnalysis(material_type="PLA")
analysis = analyzer.analyze_simulation(physics_data)
report = analyzer.generate_text_report(analysis)
print(report)
```

### Configuration Parameters

Add these to your simulation configuration JSON or dictionary:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_thermal_simulation` | bool | `false` | Enable temperature tracking and fusion detection |
| `enable_droop_simulation` | bool | `false` | Enable gravity-based droop calculation |
| `material_type` | string | `"PLA"` | Material type: `"PLA"`, `"ABS"`, or `"PETG"` |

**Example simulation_settings.json:**

```json
{
    "voxel_size": 0.1,
    "step_size": 0.2,
    "x_offset": 2.0,
    "y_offset": 2.0,
    "z_offset": 0,
    "sphere_z_offset": 0.2,
    "simulation_name": "Thermal_Test",
    "results_folder": "Results",
    "radius_increment": 0.1,
    "solver_tolerance": 0.0001,
    "x_crop": ["all", "all"],
    "y_crop": ["all", "all"],
    "z_crop": [0.0, "all"],
    "consider_acceleration": false,
    "stl_ascii": false,
    "enable_thermal_simulation": true,
    "enable_droop_simulation": true,
    "material_type": "PLA"
}
```

## Physics Models

### Thermal Model

The thermal simulation tracks:

1. **Cooling**: Material cools following Newton's law of cooling
   - T(t) = T_ambient + (T_extrusion - T_ambient) × e^(-k×t)
   - Material-specific cooling rates

2. **Material States**:
   - **Hot/Molten** (>150°C for PLA): Can fuse with other material
   - **Transitional** (60-150°C): Viscosity changes rapidly
   - **Solid** (<60°C): Below glass transition temperature

3. **Fusion Detection**: Checks if adjacent material is hot enough to bond

### Droop Model

The droop simulation calculates sagging for unsupported segments:

1. **Support Detection**: Checks voxel space below each segment
2. **Gravity Calculation**: Uses simplified catenary model
   - droop ≈ (ρ × g × L²) / (8 × η)
   - Where η (viscosity) depends on temperature
3. **Time-dependent**: Droop increases asymptotically over time

### Material Properties

Each material has specific properties:

#### PLA (Default)
- Extrusion temp: 200°C
- Glass transition: 60°C
- Fusion temp: 150°C
- Cooling rate: 0.15 s⁻¹

#### ABS
- Extrusion temp: 230°C
- Glass transition: 105°C
- Fusion temp: 180°C
- Cooling rate: 0.10 s⁻¹ (slower than PLA)

#### PETG
- Extrusion temp: 230°C
- Glass transition: 80°C
- Fusion temp: 160°C
- Cooling rate: 0.12 s⁻¹

## Analysis and Reporting

### Automated Analysis

```python
from app.reporter.analysis import PrintAnalysis

analyzer = PrintAnalysis(material_type="PLA")
analysis = analyzer.analyze_simulation(physics_data)

# Get printability score (0-100, higher is better)
score = analysis['printability_score']

# Get severity assessment
severity = analysis['severity']  # 'excellent', 'good', 'fair', 'poor', 'critical'

# Get recommendations
for rec in analysis['recommendations']:
    print(rec)
```

### Report Contents

The analysis report includes:

1. **Overall Assessment**
   - Printability score (0-100)
   - Severity level
   
2. **Statistics**
   - Total segments
   - Unsupported segments
   - Droop statistics
   - Bridge statistics

3. **Problems Detected**
   - Excessive droop locations
   - Poor fusion areas
   - Long unsupported bridges

4. **Recommendations**
   - Suggested fixes
   - Parameter adjustments
   - Design modifications

### Exporting Data

```python
# Export detailed CSV
analyzer.export_detailed_csv(
    physics_data['segments'],
    'segment_analysis.csv'
)

# Get segment colors for visualization
colors = analyzer.get_segment_colors_for_visualization(
    physics_data['segments']
)
```

## Examples

### Example 1: Detect Bridging Issues

```python
gcode_with_bridge = """M83
G0 X10 Y10 Z0.3
; Build two towers
G1 F1000 X12.0 E0.1
G0 X20.0 Y10.0
G1 X22.0 E0.1
; Bridge between them
G0 X12.0 Z0.6
G1 X20.0 E0.3
"""

sim_config = {
    # ... other settings ...
    "enable_droop_simulation": True,
    "material_type": "PLA"
}

output = run_simulation(gcode=gcode_with_bridge, ...)
physics_data = output.voxel_space.get_physics_simulation_data()

# Check droop
for seg in physics_data['segments']:
    if seg.droop_offset > 0.5:  # More than 0.5mm
        print(f"Excessive droop at segment {seg.segment_id}: {seg.droop_offset:.2f}mm")
```

### Example 2: Compare Materials

```python
for material in ['PLA', 'ABS', 'PETG']:
    sim_config['material_type'] = material
    output = run_simulation(...)
    physics_data = output.voxel_space.get_physics_simulation_data()
    
    analyzer = PrintAnalysis(material_type=material)
    analysis = analyzer.analyze_simulation(physics_data)
    
    print(f"{material}: Score = {analysis['printability_score']:.1f}/100")
```

### Example 3: Thermal Analysis

```python
sim_config = {
    # ... settings ...
    "enable_thermal_simulation": True,
    "enable_droop_simulation": False,  # Thermal only
    "material_type": "PLA"
}

output = run_simulation(...)
physics_data = output.voxel_space.get_physics_simulation_data()

# Check fusion quality
for seg in physics_data['segments']:
    if not seg.is_fused and seg.support_below:
        print(f"Poor fusion at segment {seg.segment_id}")
        print(f"  Temperature: {seg.temperature:.1f}°C")
        print(f"  Required: >{seg.material_props.fusion_temp}°C")
```

## Architecture

### New Modules

```
app/
├── physics/
│   ├── thermal/
│   │   ├── cooling_model.py       # Temperature calculations
│   │   └── material_properties.py # Material constants
│   ├── mechanics/
│   │   ├── gravity.py             # Droop calculations
│   │   ├── adhesion.py            # Fusion detection
│   │   └── droop_detection.py     # Problem analysis
│   └── material_state.py          # Hybrid state management
└── reporter/
    └── analysis.py                # Analysis and reporting
```

### Integration with VolCo

The new physics simulation is integrated at the `VoxelSpace.print()` level:

1. **Initialization**: `HybridMaterialState` is created when physics is enabled
2. **Deposition Loop**: Each segment is processed through physics before voxel deposition
3. **Data Access**: Physics data is available via `get_physics_simulation_data()`

The integration is **backward compatible** - existing code works unchanged when physics simulation is disabled.

## Performance

### Computational Cost

- **Thermal only**: ~10-20% overhead
- **Droop only**: ~15-25% overhead  
- **Both enabled**: ~25-40% overhead

The overhead is acceptable for most prints (still completes in seconds to minutes).

### Memory Usage

Additional memory per segment:
- ~200 bytes/segment for tracking
- Negligible compared to voxel space (which dominates memory)

## Limitations & Future Work

### Current Limitations

1. **Simplified Models**: Physics models are approximate, not FEA-level accurate
2. **No Nozzle Pressure**: Doesn't model extrusion pressure effects
3. **No Part Cooling Fan**: Cooling rate is constant (not fan-speed dependent yet)
4. **No Heat Transfer**: Doesn't model heat conduction between layers
5. **2D Support Detection**: Only checks directly below, not lateral support

### Planned Enhancements

- [ ] Variable cooling rates (fan speed, layer time)
- [ ] Heat transfer between adjacent material
- [ ] Nozzle pressure/force modeling
- [ ] String formation (oozing/stringing)
- [ ] Multi-material support
- [ ] GPU acceleration for large prints

## Troubleshooting

### Issue: Physics data is None

**Cause**: Physics simulation not enabled in configuration

**Solution**: Set `enable_thermal_simulation` or `enable_droop_simulation` to `true`

### Issue: Excessive droop everywhere

**Possible causes**:
1. Cooling rate too high (material staying hot too long)
2. Voxel size too large (support detection missing supports)
3. Step size too small (many short unsupported segments)

**Solutions**:
- Reduce voxel size for better support detection
- Adjust material properties if needed
- Check G-code for actual print issues

### Issue: No droop detected when expected

**Possible causes**:
1. Droop simulation not enabled
2. Material cooling too quickly (high viscosity = no droop)
3. Support detection finding false positives

**Solutions**:
- Verify `enable_droop_simulation: true`
- Check `material_type` is correct
- Inspect segment data to see support_below values

## API Reference

### HybridMaterialState

Main class managing physics simulation.

```python
HybridMaterialState(
    voxel_space,
    simulation_config,
    printer_config,
    material_type='PLA',
    enable_thermal=True,
    enable_droop=True
)
```

### FilamentSegment

Represents one extruded segment.

**Attributes:**
- `start_pos`, `end_pos`: Position (x,y,z)
- `length`: Segment length in mm
- `volume`: Extruded volume in mm³
- `temperature`: Current temperature in °C
- `droop_offset`: Vertical sag in mm
- `support_below`: Boolean, has support
- `is_solidified`: Boolean, below Tg
- `is_fused`: Boolean, bonded to substrate

### PrintAnalysis

Analysis and reporting class.

```python
analyzer = PrintAnalysis(material_type='PLA')
analysis = analyzer.analyze_simulation(physics_data)
report_text = analyzer.generate_text_report(analysis)
analyzer.export_detailed_csv(segments, 'output.csv')
```

## Contributing

To add new features:

1. **New Material**: Add constants to `material_properties.py`
2. **New Physics**: Create module in `app/physics/mechanics/`
3. **New Analysis**: Extend `PrintAnalysis` class
4. **Tests**: Add unit tests in `tests/physics/`

## References

- [VolCo Paper](https://www.sciencedirect.com/science/article/pii/S2214860417304852) - Original VolCo model
- Newton's Law of Cooling - Temperature modeling
- Catenary Curves - Droop approximation
- Glass Transition Temperature - Material state changes

## License

Same license as VolCo (see main LICENSE file).
