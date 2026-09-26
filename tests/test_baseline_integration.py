"""Small parser and end-to-end checks; these do not validate fluid physics."""

import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import trimesh

from app.instructions.gcode import Gcode
from volco import run_simulation


AREA = math.pi * (1.75 / 2) ** 2


def test_zig_bridge_reference_parser_contract():
    path = Path(__file__).resolve().parents[1] / "examples" / "validation" / "zig_bridge_candidate.gcode"
    gcode = Gcode(gcode_path=str(path), printer=SimpleNamespace(feedstock_filament_diameter=1.75))
    gcode.read()
    assert gcode.number_printed_filaments == 344
    assert all(a[:3] != b[:3] for a, b, _ in gcode.filaments_coordinates)
    assert [(a[:3], b[:3]) for a, b, _ in gcode.filaments_coordinates[-2:]] == [
        ([90.25, 50.0, 2.2], [80.0, 52.0, 2.2]),
        ([80.0, 52.0, 2.2], [70.0, 50.0, 2.2]),
    ]
    assert [volume for _, _, volume in gcode.filaments_coordinates[-2:]] == pytest.approx(
        [0.347346 * AREA, 0.339188 * AREA]
    )


def test_absolute_extrusion_reset_travel_retraction_and_stationary_deposit():
    gcode = Gcode(gcode_content="""G21
G90
M82
G92 E0
G0 X1 Y1 Z0.4
G1 X2 E0.2 F600
G1 X3 E0.1
G0 X4
G1 X5 E0.3
G1 X5 E0.5
G92 E0
G1 X6 E0.1
""", printer=SimpleNamespace(feedstock_filament_diameter=1.75))
    gcode.read()
    assert gcode.number_printed_filaments == 4
    assert [(a[0], b[0]) for a, b, _ in gcode.filaments_coordinates] == [
        (1, 2), (4, 5), (5, 5), (5, 6)
    ]
    assert [v for _, _, v in gcode.filaments_coordinates] == pytest.approx(
        [0.2 * AREA, 0.1 * AREA, 0.2 * AREA, 0.1 * AREA]
    )


def test_relative_extrusion_and_position_modes():
    gcode = Gcode(gcode_content="""G21
G90
M83
G0 X1 Y1 Z0.4
G1 X2 E0.1
G91
G0 X1
G1 X1 E0.2
""", printer=SimpleNamespace(feedstock_filament_diameter=1.75))
    gcode.read()
    assert [(a[0], b[0]) for a, b, _ in gcode.filaments_coordinates] == [(1, 2), (3, 4)]
    assert [v for _, _, v in gcode.filaments_coordinates] == pytest.approx([0.1 * AREA, 0.2 * AREA])


def test_two_mm_bead_end_to_end(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    output = run_simulation(
        gcode="G21\nG90\nM82\nG92 E0\nG0 X0 Y0 Z0.4\nG1 X2 Y0 E0.1 F600\n",
        printer_config={"nozzle_diameter": 0.4, "feedstock_filament_diameter": 1.75,
                        "nozzle_jerk_speed": 40, "extruder_jerk_speed": 5,
                        "nozzle_acceleration": 1200, "extruder_acceleration": 1200},
        sim_config={"voxel_size": 0.1, "step_size": 0.2, "simulation_name": "bead",
                    "results_folder": "results", "radius_increment": 0.1,
                    "solver_tolerance": 0.001, "sphere_z_offset": 0.2,
                    "consider_acceleration": False, "stl_ascii": False},
    )
    capsys.readouterr()  # engine prints a movement trace today
    space = output.voxel_space
    actual_volume = int(np.count_nonzero(space.space)) * 0.1**3
    assert space._filled_voxels_count == np.count_nonzero(space.space)
    # Quantized volume is not exact; this bound is for this fixture/resolution,
    # not a general conservation guarantee or empirical physics calibration.
    assert abs(actual_volume - 0.1 * AREA) <= 0.03
    path = output.export_mesh_to_stl()
    mesh = trimesh.load_mesh(path, force="mesh")
    assert mesh.is_watertight
    assert len(mesh.faces) > 0
    assert mesh.volume == pytest.approx(actual_volume, abs=1e-5)


@pytest.mark.parametrize('acceleration', [False, True])
@pytest.mark.parametrize("voxel_size,step_size,allowed_error", [
    (0.1, 0.2, 0.05), (0.05, 0.1, 0.025),
])
def test_crossing_beads_material_accounting(voxel_size, step_size, allowed_error, acceleration, capsys):
    output = run_simulation(
        gcode="G21\nG90\nM82\nG92 E0\nG0 X0 Y0 Z0.4\n"
              "G1 X2 Y0 E0.1 F600\nG0 X1 Y-1 Z0.4\nG1 X1 Y1 E0.2\n",
        printer_config={"nozzle_diameter": 0.4, "feedstock_filament_diameter": 1.75,
                        "nozzle_jerk_speed": 40, "extruder_jerk_speed": 5,
                        "nozzle_acceleration": 1200, "extruder_acceleration": 1200},
        sim_config={"voxel_size": voxel_size, "step_size": step_size,
                    "simulation_name": "cross", "results_folder": "unused",
                    "radius_increment": 0.1, "solver_tolerance": 0.001,
                    "sphere_z_offset": 0.2, "consider_acceleration": acceleration},
    )
    capsys.readouterr()
    grid = output.voxel_space.space
    occupied = int(np.count_nonzero(grid))
    assert occupied == output.voxel_space._filled_voxels_count
    assert abs(occupied * voxel_size**3 - 0.2 * AREA) <= allowed_error
    summary = output.voxel_space.volume_summary
    assert summary['cumulative_target_mm3'] == pytest.approx(0.2 * AREA)
    assert summary['cumulative_residual_mm3'] == pytest.approx(occupied * voxel_size**3 - 0.2 * AREA)
    assert summary['steps'] > 0
    assert output.voxel_space.last_deposition['evaluations'] <= 128
