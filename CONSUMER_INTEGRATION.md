# Local consumer handoff (integration/consumer-ready)

This branch starts from the `dev` ancestry and includes the bounded numerical search, world-coordinate export and candidate-cache repairs through `65a2e13`. It is an **engineering baseline**, not a physically calibrated deposition model, manifold-output guarantee, GUI release, or a merged experimental physics solver. Verify the specific commit SHA (`git rev-parse HEAD`) before integrating; a local branch name is not available remotely until separately published. No dev/main or GUI submodule has moved.

## Source-checkout Python entrypoint

This repo is not an independently packaged/pinned library. In a Python environment with its dependencies (see `requirements.txt` and the consumer's own lockfile), import `volco.run_simulation` from this checkout's root. Run it in an isolated process when possible: it uses top-level `app` imports, can produce substantial INFO output/CPU/memory usage and has no built-in whole-process deadline. The caller must supply resource limits, propagate errors and keep its UI responsive. The `volcogui` source application currently imports its **older bundled submodule**, not this branch; this handoff does not alter that integration.

```python
from volco import run_simulation

output = run_simulation(
    gcode_path="/path/to/explicit-supported.gcode",  # original numeric path, not arbitrary macros
    printer_config={
        "nozzle_diameter": 0.4,
        "feedstock_filament_diameter": 1.75,
        "nozzle_jerk_speed": 40, "extruder_jerk_speed": 5,
        "nozzle_acceleration": 1200, "extruder_acceleration": 1200,
    },
    sim_config={
        "voxel_size": 0.1, "step_size": 0.2,
        "simulation_name": "my-simulation", "results_folder": "/writable/output",
        "radius_increment": 0.1, "sphere_z_offset": 0.2,
        "consider_acceleration": False, "preview_mode": False,
        "stl_ascii": False,
        "max_evaluations": 128, "max_candidate_voxels": 250000,
        "max_grid_voxels": 64000000,
        "x_crop": ["all", "all"], "y_crop": ["all", "all"], "z_crop": ["all", "all"],
    },
)
path = output.export_mesh_to_stl()  # return is a path, not an auto-exported file
summary = output.voxel_space.volume_summary
```

`run_simulation(gcode=..., printer_config=..., sim_config=...)` also accepts string content; config-file path variants exist. File-path invocation, world coordinates and export are smoke-tested by `tests/test_consumer_entrypoint.py`. Exported STL includes all material in the input, including skirts; it is **not** automatically segmented into parts. `volume_summary` reports commanded-volume accounting and residuals, **not** pressure, ooze or physical extrusion success. Preview Mode is a different non-conserving algorithm, not a speed-equivalent substitute.

## Operational limits and caveats

- Engine parser rejects `START_PRINT BED_TEMP=... EXTRUDER_TEMP=...` and other unsupported macro syntax. A separately labeled, simulation-only projection can omit macro lines **only after verifying numeric command sequence and modal state**; never print such a derivative. The engine does not model bed mesh, thermal effects, pressure advance or motion hidden in macros. Keep exact original G-code and provenance.
- The repaired solver chooses nearest discrete voxel volume under explicit search/grid guards; a guard failure is **not** a printer-failure verdict. Use INFO logging and explicit external wall/memory caps for large inputs. Tested Windows Job limits of 180 seconds / 1 GiB committed memory and one-thread numerical libraries were **external** to this API; a consumer does not inherit them. At the tested 0.1 mm pitch the complete 23-layer camera-angle input took about 27.4 s / 194 MiB Job commitment on the measurement machine, not a general runtime promise.
- STL mesh world placement uses whole voxel cell corners; crop intervals are half-open whole-cell selections. See `EXPORT_COORDINATES.md`. Full-size exports of real paths can have **four-incident non-manifold edges** even with consistent winding and volume. Do not assume watertightness or use the mesh directly for CAD/boolean operations without a separate topology gate.
- Physical comparison so far is a single provisional ~25.03 mm recollection for a camera-angle part versus 25.00 mm nominal width and 25.10 mm voxel envelope; measurement contacts/uncertainty and slicer outer-wall envelope are not established. It is **not** quantitative physical validation. See `BASELINE.md`, `NUMERICAL_SEARCH.md`, and `VALIDATION_PLAN.md` for scope.

## Verification in this workspace

```sh
cd ../volco-baseline
uv run --project ../volcogui pytest -q tests
```

The adjacent GUI project's uv environment supplies dependencies for this checkout's tests; a downstream app must resolve/pin its own environment and verify native-platform performance. Do not silently increase fidelity or remove topology checks to make a run pass. Adoption by another app requires confirming its desired API, subprocess boundary, input types and expected STL topology before claiming end-to-end integration.
