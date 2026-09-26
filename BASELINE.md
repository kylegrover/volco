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

- `Volume.get_volumes_for_filament` returns **cumulative targets since the start of a filament**, in both uniform and acceleration-aware modes. The radius estimate uses the individual step increment; the requested total uses the cumulative value plus commanded volume before the filament, preserving quantization debt across G-code moves.
- Radius trials observe the same pre-step occupancy and **neither expand nor fill the grid**. Only the selected nearest-volume candidate commits. An already-met cumulative target skips deposition. Search/evaluation/allocation guards fail before committing a step; negative indices are clipped on all axes. See [NUMERICAL_SEARCH.md](NUMERICAL_SEARCH.md) for budgets, residual reporting and remaining policy limitations.
- For a solid occupancy grid, the mesh-object, binary STL and ASCII STL paths expose the same faces. Corner positions derive from integer half-voxel grid coordinates, avoiding sub-voxel float32 cracks for non-binary pitches. STL coordinates place the *center* of voxel `[i,j,k]` at `[i,j,k] * voxel_size` in the cropped grid, with outer faces at `+/- voxel_size/2`. Deposition treats voxel occupancy centers as `(i+0.5)*voxel_size` in the *uncropped* space: these are distinct coordinate conventions; do not silently change exported placement without a frame decision.
- Preview Mode draws nozzle-radius capsules rather than conserving specified extruded volume; it is not a volume-reference workload.

## What the tests actually check

- Five obsolete acceleration test imports/usage and printer/parser fixture assumptions were updated rather than hidden. The old one-sided deposited-volume assertion is now a two-sided, fixture-specific bound against **parsed** material volume.
- Tiny parser examples exercise relative/absolute extrusion, position mode, G92 E0, travel, retraction/recovery and intentional stationary deposition. The candidate zigged bridge has a parser-only signature test; it is not run as a physical-prediction assertion.
- Independent radius trials and accepted counts, volume helper/caller contract, tiny exact-solid surfaces and slice boundaries on all axes, and cross-path watertightness at 0.1 mm.
- A short bead checks filled-count parity, requested/occupied volume error <= 0.03 mm³ at 0.1 mm voxels, and binary STL volume/voxel-volume parity; crossing beads at 0.1 and 0.05 mm voxel sizes use fixture-specific absolute error limits of 0.05 and 0.025 mm³. These are checks for these inputs, **not general error guarantees** or evidence of monotonic convergence.
- Crossing-bead integration now covers both uniform and acceleration-aware deposition. Search tests compare against exhaustive finite-lattice candidates at two pitches and multiple phases, including blocked boundaries, and verify failure without mutation on budget/residual-limit exhaustion. Current engine suite: **79 passed**.
- Test discovery includes unrelated FEA tests; a passing suite does not imply FEA has been independently validated.

## Known limits before calling this a trustworthy numerical reference

1. Search no longer relaxes tolerance or normalizes acceptance by whole-print volume; it selects the closest discrete candidate with finite operational budgets and explicit residual diagnostics. Optional cumulative and local absolute residual limits reject before commit. However, **nearest representable volume is not a physical accuracy guarantee**: no universal physical error ceiling, local redistribution limit or physically calibrated radius bound is established. Finite-precision shell splitting, lattice phase and resolution still matter. See `NUMERICAL_SEARCH.md`.
2. G-code parsing and unsupported commands require wider coverage (including transitions between extrusion modes); these passing microcases do not establish broad slicer compatibility. Material semantics for stationary extrusion are intentional and must not be silently dropped.
3. Crop-origin/translation and output-frame behavior are not reconciled. Known solids are checked before crop; no assertion here says the exported STL is in world G-code coordinates. Adopt the cell/world transform and test plan in `VALIDATION_PLAN.md` before changing coordinates. Avoid compensating in the viewer.
4. Preview writes occupancy without maintaining `_filled_voxels_count`; do not use its counter to assert preview conservation.
5. Candidate evaluation is vectorized and trials no longer grow the grid. Explicit cell/evaluation budgets bound search work, not whole-process peak memory. No standardized performance measurements were collected; nearest-shell refinement may cost more than the former early inaccurate exit. Check memory/time before declaring the repairs performance-neutral.
6. Bundled GUI engine remains at old `dev`. GUI tests passing while this branch is separate verifies the GUI was not disturbed, **not** GUI compatibility with this new engine revision. Full packaged-GUI testing and clean-machine checks are separate release tasks.

Research evidence and plans live in private ignored `volcogui/notes/` (`VOLCO_RESEARCH_FINDINGS.md`, `VOLCO_NEXT_SPRINTS.md`, `VOLCO_BASELINE_WORK.md`). No branch/stash merges or physical-modeling claims are implied by this checkpoint.
