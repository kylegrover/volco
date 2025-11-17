# Physics Simulation Calibration Guide

## Overview
The droop physics simulation has several tunable parameters. This guide explains how to calibrate them against real-world print tests.

## Quick Calibration Workflow

### 1. Print a Test Bridge
Create a simple bridge test print:
- Two towers 10mm apart, 5mm tall
- Single-line bridge between them
- Print at your normal settings (speed, temp, cooling)
- Measure actual droop with calipers or camera

### 2. Run Simulation with Same G-code
```python
# Use the EXACT same G-code you printed
output = run_simulation(
    gcode="path/to/bridge_test.gcode",
    printer_config=your_printer_config,
    sim_config=physics_config  # thermal + droop enabled
)
```

### 3. Compare Results
- Measure simulated droop from CSV: `bridge_segment_analysis.csv`
- Compare to real droop measurement
- Adjust parameters below

## Tunable Parameters

### A. Viscosity Model (Most Important!)

**File:** `app/physics/thermal/material_properties.py`

The viscosity directly controls droop. Lower viscosity = more droop.

**Current PLA viscosity at 200°C:** ~100 Pa·s

**To increase droop (make more realistic):**

1. **Reduce base viscosity** (Line 116):
```python
# Current:
return 100 * math.exp(-2.0 * delta_T / melt_range)

# Try 10x lower:
return 10 * math.exp(-2.0 * delta_T / melt_range)  # More droop
```

2. **Adjust exponential decay rate** (Line 116):
```python
# Slower decay = higher viscosity at lower temps
return 100 * math.exp(-1.0 * delta_T / melt_range)  # Less decay = stiffer
return 100 * math.exp(-3.0 * delta_T / melt_range)  # More decay = droopier
```

**Recommended starting values for PLA:**
- At extrusion temp (200°C): 10-50 Pa·s
- At fusion temp (150°C): 100-500 Pa·s
- At glass transition (60°C): 10,000+ Pa·s

### B. Cooling Rate

**File:** `app/physics/thermal/material_properties.py`, Line 40

```python
PLA_COOLING_RATE = 0.15  # 1/s
```

**Effect:**
- Higher value = faster cooling = less time to droop = less droop
- Lower value = slower cooling = more time to droop = more droop

**To match reality:**
- With 100% fan: try 0.2-0.3
- With 50% fan: try 0.15-0.2
- With 0% fan: try 0.08-0.12

**Calibration test:**
Print a single-wall tower and measure how long it takes to cool from 200°C to 60°C.
Then: `cooling_rate = -ln((60-25)/(200-25)) / measured_time`

### C. Droop Time Constant

**File:** `app/physics/mechanics/gravity.py`, Line 28

```python
DROOP_TIME_CONSTANT = 0.5  # seconds
```

**Effect:** How fast droop develops toward maximum.
- Smaller = droop develops faster
- Larger = droop develops slower

**Typical values:**
- Fast droop (thin bridges, hot material): 0.2-0.5s
- Slow droop (thick extrusions, cooler): 0.5-1.5s

### D. Material Density

**File:** `app/physics/mechanics/gravity.py`, Lines 15-17

```python
PLA_DENSITY = 1250.0  # kg/m³
```

**Effect:** Higher density = heavier = more droop

**Typical values:**
- PLA: 1210-1430 kg/m³ (varies by brand/color)
- ABS: 1030-1070 kg/m³
- PETG: 1230-1290 kg/m³

Measure by cutting a 100mm length of filament and weighing it:
`density = (weight_g / (π * (1.75/2)² * 100)) * 1000`

## Advanced Calibration

### Multi-Point Calibration

Test bridges of different lengths:
1. 5mm bridge
2. 10mm bridge  
3. 15mm bridge
4. 20mm bridge

For each, measure actual droop vs simulated droop.

**If droop scales wrong with length:**
- Issue is in the catenary formula (L² term)
- May need to adjust the coefficient in `calculate_droop()` (Line 81):
```python
# Current: divide by 8.0
max_droop_m = (linear_density * self.GRAVITY * length_m ** 2) / (8.0 * viscosity * area_m2)

# Try different coefficients:
max_droop_m = (linear_density * self.GRAVITY * length_m ** 2) / (4.0 * viscosity * area_m2)  # 2x more droop
max_droop_m = (linear_density * self.GRAVITY * length_m ** 2) / (16.0 * viscosity * area_m2)  # 2x less droop
```

### Temperature-Dependent Calibration

Print bridges at different temperatures:
- 190°C (cooler - should droop less)
- 200°C (normal)
- 210°C (hotter - should droop more)

If temperature effect is wrong, adjust viscosity curve in `get_viscosity()`.

## Example Calibration Session

```python
# 1. Print real bridge, measure 2.5mm droop

# 2. Run simulation - reports 0.5mm droop (5x too small!)

# 3. Reduce viscosity by 5x in material_properties.py:
return 20 * math.exp(-2.0 * delta_T / melt_range)  # Was 100

# 4. Re-run simulation - now reports 2.3mm droop (close!)

# 5. Fine-tune: try 18 Pa·s base viscosity
return 18 * math.exp(-2.0 * delta_T / melt_range)

# 6. Re-run - reports 2.6mm droop (within 5% - good!)
```

## What If Nothing Works?

If you can't match reality by adjusting parameters, the physics model itself may need enhancement:

1. **Add temperature gradient along strand** - far end cools faster
2. **Add strain-rate dependent viscosity** - PLA is shear-thinning
3. **Add partial solidification** - skin forms while core is liquid
4. **Add fan cooling effects** - dramatically changes cooling rate
5. **Use FEA-style beam bending** instead of catenary approximation

See `ADVANCED_PHYSICS.md` for implementation details (TODO).

## Validation Metrics

Good calibration should match:
- ✅ Absolute droop values (±20%)
- ✅ Droop vs length scaling (L²)
- ✅ Droop vs temperature trend
- ✅ Time to failure (when bridge breaks)

## Quick Parameter Suggestions by Observation

**"My sim shows way less droop than reality":**
→ Reduce viscosity 5-10x

**"My sim shows too much droop":**
→ Increase viscosity 2-5x OR increase cooling rate

**"Short bridges match but long bridges are way off":**
→ Adjust catenary coefficient (divide by smaller number)

**"Droop happens too slowly in sim":**
→ Reduce DROOP_TIME_CONSTANT (try 0.2)

**"Droop happens too fast in sim":**
→ Increase DROOP_TIME_CONSTANT (try 1.0)

## Contributing Calibrated Values

If you calibrate for your specific printer/material, please share!
Add a config file: `configs/calibrated/<material>_<printer>.json`

Example:
```json
{
  "material": "PLA",
  "brand": "Hatchbox",
  "printer": "Prusa MK3S",
  "nozzle_temp": 215,
  "cooling_fan": 100,
  "calibrated_params": {
    "base_viscosity_pa_s": 25,
    "cooling_rate_per_s": 0.22,
    "droop_time_constant_s": 0.4,
    "density_kg_m3": 1240
  },
  "test_results": {
    "10mm_bridge_droop_mm": 1.2,
    "15mm_bridge_droop_mm": 3.1,
    "20mm_bridge_droop_mm": 6.8
  }
}
```
