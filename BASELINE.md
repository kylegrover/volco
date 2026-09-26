# Numerical baseline — development checkpoint

This branch starts from Volco `dev` (`8eb866a`) and repairs several *reproducible* regressions. It is an engineering test baseline, **not a validated physical prediction** or a claim to implement all of VOLCO-X. Keep the GUI submodule and any physics experiments separate until integration is explicitly approved. The agreed path toward trustworthy numerical output, a real bracket/art workload, and a candidate zigged-bridge physical comparison is in [VALIDATION_PLAN.md](VALIDATION_PLAN.md). The bridge G-code under `examples/validation/` is a reference candidate, **not a print-ready or physics-validated fixture**.

## Run tests

With the adjacent `volcogui` development project providing the uv-managed environment:

```sh
cd volco-baseline
uv run --project ../volcogui pytest -q tests
```

`pytest.ini` discovers both historical `*_test.py` and modern `test_*.py`, adding the engine source root to imports. Files in `tests/fixtures/` are resolved relative to their tests. The command also works from the parent directory as `uv run --project volcogui pytest -q volco-baseline/tests`. This checkout has no independent pinned lockfile; record the environment alongside any benchmark or comparison. Do not mistake the adjacent GUI dependency set for a formal engine dependency specification.

## Explicit contracts covered here

- `Volume.get_volumes_for_filament` returns **cumulative targets since the start of a filament**, in both uniform and acceleration-aware modes. The radius estimate uses the individual step increment; the requested total uses the cumulative value plus occupancy before the filament.
- Radius trials observe the same pre-step occupancy. Evaluating a trial may grow the empty grid but **cannot fill voxels or update the running count**; only the selected radius commits. An already-met cumulative target skips deposition.
- For a solid occupancy grid, the mesh-object, binary STL and ASCII STL paths expose the same faces. Corner positions derive from integer half-voxel grid coordinates, avoiding sub-voxel float32 cracks for non-binary pitches. STL coordinates place the *center* of voxel `[i,j,k]` at `[i,j,k] * voxel_size` in the cropped grid, with outer faces at `+/- voxel_size/2`. Deposition treats voxel occupancy centers as `(i+0.5)*voxel_size` in the *uncropped* space: these are distinct coordinate conventions; do not silently change exported placement without a frame decision.
- Preview Mode draws nozzle-radius capsules rather than conserving specified extruded volume; it is not a volume-reference workload.

## What the tests actually check

- Five obsolete acceleration test imports/usage and printer/parser fixture assumptions were updated rather than hidden. The old one-sided deposited-volume assertion is now a two-sided, fixture-specific bound against **parsed** material volume.
- Tiny parser examples exercise relative/absolute extrusion, position mode, G92 E0, travel, retraction/recovery and intentional stationary deposition. The candidate zigged bridge has a parser-only signature test; it is not run as a physical-prediction assertion.
- Independent radius trials and accepted counts, volume helper/caller contract, tiny exact-solid surfaces and slice boundaries on all axes, and cross-path watertightness at 0.1 mm.
- A short bead checks filled-count parity, requested/occupied volume error <= 0.03 mm³ at 0.1 mm voxels, and binary STL volume/voxel-volume parity; crossing beads at 0.1 and 0.05 mm voxel sizes use fixture-specific absolute error limits of 0.05 and 0.025 mm³. These are checks for these inputs, **not general error guarantees** or evidence of monotonic convergence.
- Test discovery includes unrelated FEA tests; a passing suite does not imply FEA has been independently validated.

## Known limits before calling this a trustworthy numerical reference

1. The bisection solver relaxes relative error tolerance as radii get close and has no explicit maximum search radius/iteration count. A bounded characterization found the configured tolerance can accept large per-increment errors, even when a closer voxel shell exists; the comparison is relative to cumulative occupied volume and can hide a locally skipped increment. `find_sphere_limits` clamps a negative Z minimum, but not negative X/Y minima; negative slice/index behavior can make candidate counts spatially invalid and wrap writes to the opposite array edge. An intentionally oversized probe was manually stopped after 60 evaluations while the grid had grown to `(194,194,139)`; that is evidence of unsafe unbounded work in this case, not proof every target loops forever. Cumulative requested volumes and occupied voxels can differ due to quantization; there is no per-step requested/achieved/residual record or general local error bound. Establish a finite best-candidate/failure policy and test boundary indexing before baseline promotion. Detailed methods and residual tables are kept in the local private baseline work notes.
2. G-code parsing and unsupported commands require wider coverage (including transitions between extrusion modes); these passing microcases do not establish broad slicer compatibility. Material semantics for stationary extrusion are intentional and must not be silently dropped.
3. Crop-origin/translation and output-frame behavior are not reconciled. Known solids are checked before crop; no assertion here says the exported STL is in world G-code coordinates. Adopt the cell/world transform and test plan in `VALIDATION_PLAN.md` before changing coordinates. Avoid compensating in the viewer.
4. Preview writes occupancy without maintaining `_filled_voxels_count`; do not use its counter to assert preview conservation.
5. No standardized performance measurements or memory bounds were collected. The candidate search still converts empty indices to Python lists, and rejected trials can grow the grid. Check memory/time before declaring the repairs performance-neutral.
6. Bundled GUI engine remains at old `dev`. GUI tests passing while this branch is separate verifies the GUI was not disturbed, **not** GUI compatibility with this new engine revision. Full packaged-GUI testing and clean-machine checks are separate release tasks.

Research evidence and plans live in private ignored `volcogui/notes/` (`VOLCO_RESEARCH_FINDINGS.md`, `VOLCO_NEXT_SPRINTS.md`, `VOLCO_BASELINE_WORK.md`). No branch/stash merges or physical-modeling claims are implied by this checkpoint.
