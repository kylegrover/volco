"""Public file-path API smoke test for source-checkout consumers; not a physics test."""

from pathlib import Path

import pytest
import trimesh

from volco import run_simulation


def test_file_path_entrypoint_exports_world_placed_stl(tmp_path: Path, capsys):
    gcode = tmp_path / "bead.gcode"
    gcode.write_text("G21\nG90\nM82\nG92 E0\nG0 X10 Y20 Z0.4\nG1 X12 Y20 E0.1 F600\n", encoding="utf-8")
    output = run_simulation(
        gcode_path=str(gcode),
        printer_config={
            "nozzle_diameter": 0.4, "feedstock_filament_diameter": 1.75,
            "nozzle_jerk_speed": 40, "extruder_jerk_speed": 5,
            "nozzle_acceleration": 1200, "extruder_acceleration": 1200,
        },
        sim_config={
            "voxel_size": 0.1, "step_size": 0.2, "simulation_name": "bead",
            "results_folder": str(tmp_path), "radius_increment": 0.1,
            "sphere_z_offset": 0.2, "consider_acceleration": False,
            "preview_mode": False, "stl_ascii": False,
        },
    )
    capsys.readouterr()  # Engine currently prints a filament trace.
    path = Path(output.export_mesh_to_stl())
    assert path == tmp_path / "bead.stl"
    mesh = trimesh.load_mesh(path, force="mesh")
    assert mesh.bounds[0][0] == pytest.approx(10, abs=0.5)
    assert mesh.bounds[1][0] == pytest.approx(12, abs=0.5)
    assert mesh.bounds[0][1] == pytest.approx(20, abs=0.5)
    assert output.voxel_space.volume_summary["steps"] > 0
