# Big Square Profiling Notes (Nov 16 2025)

## Run Setup
- Command: `uv run python` wrapping `run_simulation` for `examples/big_square.gcode`
- Configs: `examples/printer_settings.json`, `examples/simulation_settings.json`
- Profiler: `cProfile` with 420 s timer; actual runtime captured 3,535.6 s before graceful stop
- Output file: `profile_big_square_partial_run2.pstats`
- Simulation progressed through the first filament loop twice (2,000 steps total) before timeout

## High-Level Metrics
- Total function calls: 153,567,305 (153,560,330 primitive)
- Wall-clock recorded by profiler: 3,535.6 s
- Dominant operations: voxel-space deformation, bisection solver iterations, voxel filling math utilities

## Top Hotspots (cumulative time)
| Rank | Function | Calls | Cumulative (s) | Notes |
| --- | --- | --- | --- | --- |
| 1 | `sphere.py:118 _deposit_sphere` | 23,615 | 3,767 | Umbrella for every deposit; downstream costs listed below |
| 2 | `sphere.py:87 deform_voxel_space_for_big_spheres` | 23,615 | 2,648 | Runs before each deposit to grow arrays |
| 3 | `sphere.py:99 _maybe_expand_voxel_space` | 70,845 | 2,153 | Allocates+concats new NumPy slabs repeatedly |
| 4 | `bisection_method.py:32 _loop_fb` | 4,000 | 2,014 | Step-increase search for bisection upper bound |
| 5 | `bisection_method.py:47 _loop_fc` | 4,000 | 1,841 | Core bisection loop, tolerance boosts included |
| 6 | `sphere.py:39 fill_voxels` | 23,615 | 425 | Per-voxel scan + writes |
| 7 | `geometry_math.py:46 find_coordinates` | 49,865,529 | 284 | Heavy Python loop converting voxel index → XYZ |
| 8 | `geometry_math.py:50 calculate_filled_volume` → `numpy.count_nonzero` | 23,619 | 214 | Re-counts voxels to compute fill ratio |

## Root Causes & Hypotheses
1. **Voxel-space expansion thrash**
   - `_maybe_expand_voxel_space` fires ~3× per deposit (once per axis). Each call concatenates an entire NumPy array, so cost grows with volume.
   - No caching of required extents; solver keeps re-copying grids even if radius is unchanged.

2. **Solver churn**
   - `_loop_fb` + `_loop_fc` consume ~55% of total runtime combined (3.9 ks).
   - Initial radius guess + increment are conservative, so many iterations are needed to bracket volume. `fun_increase_tolerance` only triggers when radii nearly match, so tolerance rarely widens.

3. **Per-voxel Python work**
   - `fill_voxels` iterates through `GeometryMath.find_empty_voxels_in_space`, computing coordinates and distances one voxel at a time.
   - 50M calls to `find_coordinates` and 49M to `GeometryMath.distance` create pure-Python bottlenecks despite simple math.
   - `calculate_filled_volume` re-counts entire space after every deposit instead of maintaining a running tally.

## Potential Fix Directions (to evaluate later)
1. **Preallocate / cache voxel extents**
   - Predict max needed indices per axis from toolpath bounds + max extrusion radius; allocate once.
   - Alternatively keep a growth watermark per axis to skip `np.concatenate` when radius fits.

2. **Smarter bisection hints**
   - Reuse last converged radius as the next filament’s initial range.
   - Increase increment size dynamically when `_loop_fb` repeatedly undershoots.
   - Consider memoizing `(radius, volume)` pairs to avoid re-running `_deposit_sphere` for the same nozzle height.

3. **Vectorize voxel fill math**
   - Replace per-voxel coordinate calculations with NumPy grids or use distance transforms.
   - Maintain an incremental `filled_voxel_count` to avoid `count_nonzero` on every iteration.
   - Restrict search window tightly: compute bounding box once per neighborhood and reuse.

4. **Reduce deposit frequency**
   - Investigate whether filaments can batch multiple steps per deposit (e.g., accumulate motion until nozzle moves more than one voxel).

These notes capture the issues observed on Nov 16 so we can resume optimization later without rerunning the full profiler.



---

Next obvious options remain:
1) tackle voxel-space reallocations,
2) tune the bisection iteration pattern,
3) rework the per-voxel math loop