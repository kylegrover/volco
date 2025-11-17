# Bridge Droop Debug Notes

## Goal
Make 40mm bridge visibly sag in STL output

## Current Status
- Droop calculated: ~5mm for 43mm bridge ✓
- File sizes identical (0 bytes diff) ✗
- STL shows NO visible sag ✗
- CSV shows all bridge segments at Z=0.0 ✗

## Key Files
- Test: `examples/test_bridge.py`
- G-code: `examples/bridge_proper.gcode` (40mm bridge, X=110→70, Z=2.2)
- Physics STL: `Results_longbridge_physics/LongBridge_WithPhysics.stl`
- CSV: `Results_longbridge_physics/longbridge_segment_analysis.csv`

## What Works
1. ✓ Support detection: 43mm bridge detected as unsupported
2. ✓ Droop calculation: ~5mm calculated for 43mm span
3. ✓ Timing: Segments timestamped correctly based on feedrate
4. ✓ Age-based droop: Each segment droops based on its age
5. ✓ Build plate clamp: Z clamped to max(0.0, Z-droop)

## What Doesn't Work
**PRIMARY ISSUE**: All bridge segments in CSV show Z=0.0
- Segments 215-260: all have start_z=0.0, end_z=0.0
- Droop values: 1.5mm - 3.3mm (reasonable)
- Expected: Z should be around 2.2 - droop = ~1.9mm to -0.8mm → clamped to 0.0-1.9mm
- Actual: ALL at Z=0.0 (flat line, no curve)

## Root Cause Discovery 🔴

### THE BUG: Segment coordinates are 100x too small!

**Debug evidence** (segment 250):
```
orig_Z=0.233mm (WRONG! Should be ~2.2mm)
droop=2.758mm
before_clamp = 0.233 - 2.758 = -2.526mm
after_clamp = max(0.0, -2.526) = 0.0mm
```

**Why every segment is at Z=0.0**:
- Segments have Z ≈ 0.24mm instead of Z ≈ 2.2mm
- With ~3mm droop: 0.24 - 3.0 = -2.76mm → clamped to 0.0mm
- Every segment clamped flat because original Z is too small!

### Investigation Trail

1. ✅ **G-code correct**: `G1 X70 Y50 Z2.2 E7.0 F1800`
2. ✅ **Raw filament coords correct**:
   - Start: [110, 50, 2.2] → After translation: [112, 52, 2.2]
   - End: [70, 50, 2.2] → After translation: [72, 52, 2.2]
3. 🔴 **Filament length WRONG**:
   - Expected: sqrt((112-72)² + 0² + 0²) = 40mm
   - Actual: 0.40mm (100x too small!)
4. 🔴 **Number of steps WRONG**:
   - Expected: 40mm / 0.2mm step_size = 200 steps
   - Actual: 0.4mm / 0.2mm = 2 steps (!!)

### The Smoking Gun

Bridge divided into only **2 segments** instead of **200 segments**:
- Step 0: displacement = 0×0.2 = 0.0mm → center at 0.1mm from start
- Step 1: displacement = 1×0.2 = 0.2mm → center at 0.3mm from start
- Total span: 0.4mm instead of 40mm

**Where segments end up**:
- With 2 steps over 0.4mm, each segment is tiny fraction of actual bridge
- Segment centers at ~0.1mm and ~0.3mm displacement
- These get placed with Z coordinates scaled down 100x somehow

### ROOT CAUSE FOUND! 🎯

**The bug was a use-before-definition error!**

In `voxel_space.py`, debug code was printing `filament_length` BEFORE it was calculated:
- Line 113: Printed `filament_length` (showed 0.4mm from previous iteration)
- Line 120: THEN calculated `filament_length = distance(...)` (returned 40mm)

This caused:
1. Only 2 simulation steps instead of 200 (0.4/0.2 = 2)
2. Segments created with tiny Z increments
3. All segments ended up at Z ≈ 0.24mm instead of Z ≈ 2.2mm

**FIX**: Moved debug prints to AFTER the length calculation

## Verification

✅ After fix:
- `filament_length: 40.00mm` (was 0.40mm)
- `number_simulation_steps: 200` (was 2)
- Coordinates passed to deposit_filament_segment: Z=2.2mm ✓
- Segments initialized with Z=2.2mm ✓
- Bridge segments ARE at correct Z: **segments 550-560 at Z=2.2mm** ✓

## NEW PROBLEM DISCOVERED! 🔴

**Bridge segments have ZERO droop!**

Segment 555 (middle of bridge):
- `orig_Z`: 2.200mm ✓
- `droop`: 0.00mm ❌
- `has_support`: False ✓
- `unsupported_length`: **0.8mm** ❌❌❌ (should be 40mm!)

**Root cause**: Strand tracking isn't grouping bridge segments into a single 40mm unsupported strand!

Each segment thinks it's only 0.8mm unsupported (just itself + neighbors), not part of a 40mm bridge.

The droop formula with L=0.8mm gives ~0mm droop (correct for formula, wrong input).

**Next steps**: Debug strand tracking - why aren't bridge segments being grouped into one continuous strand?

---

# Investigation Summary & Action Plan

## What We've Discovered

### ✅ FIXED: Use-Before-Definition Bug
**Problem**: `filament_length` printed before calculation showed 0.4mm from previous iteration
**Impact**: Bridge divided into 2 steps instead of 200, segments at wrong Z
**Solution**: Moved debug prints after calculation
**Result**: Bridge now correctly has 200 steps, segments at Z=2.2mm

### ✅ CONFIRMED WORKING: Coordinate Pipeline
- G-code parsing: Bridge at Z=2.2mm ✓
- Raw filament coordinates: [110, 50, 2.2] → [70, 50, 2.2] ✓
- After translation: [112, 52, 2.2] → [72, 52, 2.2] ✓
- Distance calculation: 40.00mm ✓
- Starting/ending points per step: Z=2.2mm ✓
- FilamentSegment creation: Z=2.2mm ✓
- Bridge segments in CSV: **550-560 at Z=2.2mm** ✓

### 🔴 CURRENT PROBLEM: Strand Tracking Not Working

**Symptoms**:
- Bridge segments (550-560) have `droop_offset ≈ 0.00mm`
- Each segment reports `unsupported_length: 0.8mm` (should be 40mm)
- Droop formula correctly returns ~0mm for L=0.8mm
- File sizes identical (no geometry change)

**Root Cause**: Strand tracking isn't accumulating unsupported length across bridge segments

## Action Plan

### Phase 1: Understand Strand Tracking Logic
1. **Read** `_update_strand_physics()` - how are strands created/tracked?
2. **Read** `_apply_strand_droop()` - how is unsupported length accumulated?
3. **Identify** what conditions terminate a strand
4. **Check** what `unsupported_length` is supposed to represent (per-segment vs cumulative)

### Phase 2: Debug Strand Tracking
5. **Add logging** to `_update_strand_physics()`:
   - When does a new strand start?
   - When does a strand continue?
   - When does a strand end?
   - What's the accumulated unsupported length?
6. **Add logging** to strand droop application:
   - How many segments in each strand?
   - What's the total strand length?
   - Is the bridge recognized as one continuous strand?

### Phase 3: Fix Strand Accumulation
7. **Hypothesis A**: Strands are being terminated too early (maybe after each simulation step?)
8. **Hypothesis B**: `unsupported_length` is per-segment, not cumulative from strand
9. **Hypothesis C**: Support detection is incorrectly detecting support mid-bridge
10. **Apply fix** based on findings

### Phase 4: Verification
11. **Run test** and confirm bridge segments show proper unsupported_length (~40mm)
12. **Check** droop calculation gives ~5mm for 40mm bridge
13. **Verify** file sizes differ (geometry changed)
14. **Visual check** STL shows visible sag

## Key Findings to Remember

- Test runs TWO simulations: baseline (no physics) then physics (with physics)
- BRIDGE DROOP messages show segments 215-429 being processed
- But CSV from physics shows segments 550-560 with no droop!
- Segments 550-560 ARE at correct Z=2.2mm
- Support detection IS working (has_support=False)  
- Problem: Different segment IDs in droop processing vs CSV output

## CRITICAL ISSUE IDENTIFIED

**Segment ID mismatch between simulations!**

The "BRIDGE DROOP" messages show segments 215-429 (from first simulation).
The CSV shows segments 550-560 (from second simulation).

This suggests:
1. Segment IDs are NOT resetting between simulations
2. OR both simulations are writing to same HybridMaterialState
3. OR baseline is incorrectly running with physics enabled

**NEXT ACTION**: Verify that segment IDs reset between baseline and physics runs
