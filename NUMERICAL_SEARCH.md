# Bounded deposition search contract

This is a numerical safety contract, not a physical-accuracy claim. Coordinate-frame export is covered separately in `EXPORT_COORDINATES.md`; numerical and physical acceptance remain distinct.

## Domain and candidate selection

- Cells have centres `(index + 0.5) * voxel_size`. Negative XYZ indices are outside the allowed domain, never wrapped to the opposite array edge. X/Y offsets must provide the intended physical padding; clipping at zero does **not** simulate material beyond that domain.
- Candidate centres must be below or on the nozzle plane. A partially intersected cell whose centre is above the plane is excluded.
- Existing occupied voxels are excluded from all trials. Positive-side expansion is evaluated virtually. Rejected trials neither fill nor expand the grid. Accepted indices are committed once, with count parity.
- Within one uncommitted step, the most recent bounding box's empty indices and distance values may be reused across radius trials; radius inclusion is recalculated with the same finite-precision rule. The one-box cache is discarded between steps and on failure, and is never reused after occupancy changes. Existing candidate/grid budgets are checked before each trial. It reduces repeated allocations on measured early-layer real paths; it does not cap total Python/mesh memory or change deposition physics.
- Search brackets the required additional voxel count, then refines until there is an exact count, consecutive integer counts, or adjacent representable floating-point radii. A plateau alone is not a stopping condition. Compare both endpoints and choose the closest cumulative volume; a tie prefers underfill. Zero deposition is a legitimate candidate.
- No tolerance inflation. `solver_tolerance` remains a positive relative tolerance for **local diagnostic classification**, not a whole-print-relative early acceptance criterion. The search always seeks the closest representable count. Generic smooth-function `BisectionMethod` is separately bounded and no longer widens tolerance; its legacy tolerance-increase callback argument is retained but ignored.
- The inclusion allowance remains `voxel_size * 1e-8`. Very close floating-point shell thresholds can split geometrically symmetric shells. Tests compare against the actual finite-precision inclusion rule; no claim of exact-arithmetic symmetry is made. Changing the distance/tie convention requires separate characterization.

## Conservation and residuals

Cumulative targets persist across G-code movement boundaries. Previously every movement rebased on actual occupancy, erasing its predecessor's quantization debt. Each step now retains its nominal increment, while radius selection targets cumulative commanded material. Thus a later step may repay a prior deficit or skip deposition after an overshoot. This is numerical material accounting, not proof the resulting local material redistribution is physically correct.

Every successful step emits an INFO-level `Deposition volume` record and exposes `Sphere.last_deposition` / `VoxelSpace.last_deposition`, including:

- nominal requested increment and cumulative target (mm³);
- occupied before/after, actual increment, signed increment and cumulative residuals;
- local relative residual and whether it is within `solver_tolerance`;
- radius, added count, candidate bracket counts/residuals, evaluations, status and commit flag.

`VoxelSpace.volume_summary` retains step counts, maximum absolute local error, final cumulative target and residual. It does not retain an unbounded in-memory list of per-step records. Capture INFO logs for the full trace. `exact` refers to the cumulative voxel-volume target, not the local allocation or physical geometry. `quantized` means nearest-candidate error; `carried_overshoot` means earlier material already exceeds the target.

## Failure and operational budgets

Simulation JSON accepts these optional settings:

| Setting | Default | Meaning |
| --- | ---: | --- |
| `max_evaluations` | 128 | Candidate evaluations per deposition step |
| `max_candidate_voxels` | 250000 | Maximum trial bounding-box cells, checked before candidate-array allocation |
| `max_grid_voxels` | 64000000 | Maximum initial/expanded grid cells |
| `max_volume_error_mm3` | null | Optional absolute cumulative residual ceiling, checked before commit |
| `max_increment_error_mm3` | null | Optional absolute local increment residual ceiling, checked before commit |

These count/evaluation defaults are **explicit operational guardrails**, not experimentally derived bead-radius limits or accuracy thresholds. The candidate cap limits several temporary arrays; the grid cap limits grid cells, not whole-process peak bytes. Expansion can temporarily hold old and new grids. STL generation/cropping, Python overhead and other pipeline allocations are not covered by these budgets. Large valid workloads may require reviewed settings.

If the target cannot be bracketed/refined within a budget, no partial approximation is silently accepted. `DepositionError` includes the reason, target, best observed residual and evaluation diagnostics. A nozzle plane below the first cell centre fails immediately when material is required. Failure does not commit or expand the current step; previously completed steps remain. Invalid nonpositive/nonfinite simulation search parameters are rejected before allocation.

An explicit `max_volume_error_mm3` rejects an excessive nearest-shell cumulative residual before commit. `max_increment_error_mm3` separately limits local error, including repayment of earlier debt; this is a rejection ceiling, not a rule to choose a different, less conservative candidate. No universal default physical error percentage is invented: without these optional ceilings, a representationally unavoidable closest-shell residual is reported rather than treated as physical validation. In particular, a subvoxel command may yield zero material at one step and be repaid later. An empirically justified local redistribution limit and physical radius/domain bound remain open validation questions.

## Regression evidence

`tests/test_candidate_cache.py` differentially compares the cached enumeration against the frozen pre-cache implementation at three pitches, boundary/shell phases, occupancy and expansion changes, and guard failures, and checks single-box reuse. On macro-excluded early-layer real G-code, fixed-fidelity reference/candidate comparisons matched numeric diagnostics and binary STL hashes. The optimization changes measured performance, not the physical model. `tests/test_search_safety.py` covers negative slicing/index wrap, nested boundary candidates, nozzle-centre clipping, cumulative-normalization masking, subvoxel debt across moves, raw-array/object expansion parity, candidate/evaluation/grid budgets, residual-limit rejection, malformed configs and generic solver termination. Finite-lattice enumeration checks nearest candidates at two pitches and three lattice phases, including blocked boundary fixtures. End-to-end crossing-bead checks now run with both uniform and acceleration-aware deposition.

These checks demonstrate numerical behavior on these fixtures. They do not establish broad slicer support, physical prediction accuracy, monotonic resolution convergence, or GUI integration.
