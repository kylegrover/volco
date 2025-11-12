# Quick Start: Thermal & Droop Simulation

## 5-Minute Quick Start

### 1. Enable Physics Simulation

Add to your simulation config:

```python
sim_config = {
    # ... existing settings ...
    "enable_thermal_simulation": True,
    "enable_droop_simulation": True,
    "material_type": "PLA"  # or "ABS" or "PETG"
}
```

### 2. Run Simulation

```python
from volco import run_simulation

output = run_simulation(
    gcode_path='your_file.gcode',
    sim_config=sim_config,
    printer_config=printer_config
)
```

### 3. Get Analysis

```python
from app.reporter.analysis import PrintAnalysis

# Get physics data
physics_data = output.voxel_space.get_physics_simulation_data()

# Analyze
analyzer = PrintAnalysis()
analysis = analyzer.analyze_simulation(physics_data)

# Print report
print(analyzer.generate_text_report(analysis))
```

## What You Get

- **Printability Score**: 0-100 (higher is better)
- **Problem Detection**: Excessive droop, poor fusion, long bridges
- **Recommendations**: Actionable suggestions to fix issues
- **Detailed Data**: CSV export with per-segment information

## When to Use

### Use Thermal Simulation When:
- ✅ Testing layer adhesion
- ✅ Optimizing nozzle temperature
- ✅ Analyzing cooling issues
- ✅ Checking fusion quality

### Use Droop Simulation When:
- ✅ Printing with bridges
- ✅ Testing overhangs
- ✅ Evaluating support needs
- ✅ Predicting print failures

## Example Output

```
VOLCO THERMAL & DROOP SIMULATION ANALYSIS
======================================================================

Material Type: PLA
Thermal Simulation: Enabled
Droop Simulation: Enabled

OVERALL ASSESSMENT:
  Severity Level: GOOD
  Printability Score: 87.5/100

STATISTICS:
  Total segments: 42
  Unsupported segments: 5
  Solidified segments: 37
  Active (hot) segments: 5
  Simulation time: 8.32s

DROOP ANALYSIS:
  Maximum droop: 0.234 mm
  Average droop: 0.012 mm
  Severe droop issues: 0
  Moderate droop issues: 1

BRIDGE ANALYSIS:
  Total bridges: 2
  Longest bridge: 8.45 mm
  Average bridge length: 6.23 mm

RECOMMENDATIONS:
  ℹ️ Long bridge detected (8.5mm). Ensure cooling fan at 100% for bridges.
  ✅ No major issues detected. Model appears printable.
```

## Common Patterns

### Pattern 1: Quick Check

```python
# Just want to know if it will print?
output = run_simulation(gcode_path='file.gcode', ...)
physics_data = output.voxel_space.get_physics_simulation_data()
score = PrintAnalysis().analyze_simulation(physics_data)['printability_score']
print(f"Printability: {score}/100")
```

### Pattern 2: Detailed Analysis

```python
# Need full details?
analyzer = PrintAnalysis()
analysis = analyzer.analyze_simulation(physics_data)
print(analyzer.generate_text_report(analysis))
analyzer.export_detailed_csv(physics_data['segments'], 'details.csv')
```

### Pattern 3: Material Comparison

```python
# Which material works best?
for material in ['PLA', 'ABS', 'PETG']:
    sim_config['material_type'] = material
    output = run_simulation(...)
    physics_data = output.voxel_space.get_physics_simulation_data()
    analysis = PrintAnalysis().analyze_simulation(physics_data)
    print(f"{material}: {analysis['printability_score']}/100")
```

### Pattern 4: Problem Debugging

```python
# Find specific problem areas
physics_data = output.voxel_space.get_physics_simulation_data()
for seg in physics_data['segments']:
    if seg.droop_offset > 0.5:  # More than 0.5mm droop
        print(f"Problem at segment {seg.segment_id}")
        print(f"  Position: {seg.start_pos} -> {seg.end_pos}")
        print(f"  Droop: {seg.droop_offset:.2f}mm")
        print(f"  Unsupported length: {seg.unsupported_length:.1f}mm")
```

## Configuration Cheat Sheet

| Setting | Effect | When to Use |
|---------|--------|-------------|
| `enable_thermal_simulation: true` | Track temperature & fusion | Always recommended |
| `enable_droop_simulation: true` | Calculate gravity sag | When bridges/overhangs present |
| `material_type: "PLA"` | PLA properties | Most common |
| `material_type: "ABS"` | ABS properties | Higher temp prints |
| `material_type: "PETG"` | PETG properties | Stronger parts |

## Performance Tips

- **Fast preview**: Use larger `voxel_size` (0.2) and `step_size` (0.5)
- **High accuracy**: Use smaller `voxel_size` (0.05) and `step_size` (0.1)
- **Large prints**: Enable only one simulation (thermal OR droop)
- **Quick check**: Disable physics entirely for fast geometry check

## Interpreting Results

### Printability Scores
- **90-100**: Excellent, print with confidence
- **75-89**: Good, minor issues possible
- **60-74**: Fair, may need tweaks
- **40-59**: Poor, likely problems
- **0-39**: Critical, will probably fail

### Droop Severity
- **< 0.1mm**: None/minor (usually fine)
- **0.1-0.3mm**: Moderate (visible but acceptable)
- **0.3-1.0mm**: Severe (quality issues)
- **> 1.0mm**: Critical (will fail)

## Troubleshooting

**Problem**: No physics data  
**Fix**: Set `enable_thermal_simulation: true` or `enable_droop_simulation: true`

**Problem**: Excessive droop everywhere  
**Fix**: Check voxel_size (smaller = better support detection)

**Problem**: No droop detected  
**Fix**: Verify bridges exist in G-code, check material_type

**Problem**: Slow simulation  
**Fix**: Increase voxel_size and step_size

## Full Documentation

- 📖 Complete Guide: [THERMAL_SIMULATION.md](THERMAL_SIMULATION.md)
- 💡 Examples: [examples/thermal_simulation_example.py](examples/thermal_simulation_example.py)
- 🔧 Implementation: [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)
- 📝 Main README: [README.md](README.md)

## Getting Help

1. Check the full documentation
2. Run the example scripts
3. Review the integration tests
4. Open an issue on GitHub
