"""Small behavioral contracts for the numerical baseline, not physical validation."""

from types import SimpleNamespace

import numpy as np
import pytest
import trimesh

from app.geometry.sphere import Sphere
from app.geometry.voxel_space import VoxelSpace
from app.physics.volume import Volume
from app.reporter.mesh import (
    create_mesh_vectorized,
    export_voxel_stl_streaming_binary,
    export_voxel_stl_streaming_ascii,
)


def test_uniform_volume_targets_are_cumulative():
    assert Volume.get_volumes_for_filament(3, 0.06) == pytest.approx([0.02, 0.04, 0.06])


def test_acceleration_volume_targets_are_cumulative(monkeypatch):
    from app.physics.acceleration.volume import AccelerationVolume

    def sample(*args):
        return [0.01, 0.025, 0.06]

    monkeypatch.setattr(AccelerationVolume, "get_volumes_for_filament", sample)
    assert Volume.get_volumes_for_filament(3, 0.06, True, 1.0, 10.0, object()) == sample()


def test_filament_uses_cumulative_targets(monkeypatch):
    calls = []

    def capture(self, **kwargs):
        calls.append((kwargs["sphere_volume"], kwargs["voxel_space_target_volume"]))

    monkeypatch.setattr(Sphere, "deposit_sphere", capture)
    simulation = SimpleNamespace(voxel_size=0.1, sphere_z_offset=0.0,
                                 solver_tolerance=0.01, radius_increment=0.1)
    space = VoxelSpace.__new__(VoxelSpace)
    space._simulation = simulation
    space._filled_voxels_count = 10
    volumes = Volume.get_volumes_for_filament(3, 0.06)
    space._deposit_filament(3, 1.0, [1, 0, 0], [1, 1, 1], volumes)
    assert [target for _, target in calls] == pytest.approx([0.03, 0.05, 0.07])


def test_radius_trials_are_independent_and_do_not_change_counter():
    sphere = Sphere([1, 1, 0.5], 0.1)
    space = SimpleNamespace(space=np.zeros((30, 30, 20), dtype=np.int8),
                            _filled_voxels_count=0)
    large_error, large_radius = sphere._deposit_sphere(0.3, space, 0.8, 0.02)
    small_error, small_radius = sphere._deposit_sphere(0.15, space, 0.8, 0.02)
    assert (large_radius, small_radius) == (0.3, 0.15)
    assert space._filled_voxels_count == np.count_nonzero(space.space) == 0
    # The smaller radius must be evaluated against the original empty grid.
    assert large_error > small_error
    assert sphere._deposit_sphere(0.15, space, 0.8, 0.02)[0] == small_error


def test_already_exceeded_target_does_not_deposit_more(monkeypatch):
    sphere = Sphere([1, 1, 0.5], 0.1)
    space = SimpleNamespace(space=np.zeros((30, 30, 20), dtype=np.int8),
                            _filled_voxels_count=1)
    space.space[10, 10, 4] = 1
    def unnecessary_trial(*args):
        raise AssertionError("An already-met target must not search for a radius")
    monkeypatch.setattr(sphere, "_deposit_sphere", unnecessary_trial)
    assert sphere.deposit_sphere(space, 0.8, 0.02, 0.0005, 0.001, 0.05) is space
    assert space._filled_voxels_count == np.count_nonzero(space.space) == 1


def test_accepted_sphere_is_committed_once():
    sphere = Sphere([1, 1, 0.5], 0.1)
    space = SimpleNamespace(space=np.zeros((30, 30, 20), dtype=np.int8),
                            _filled_voxels_count=0)
    space.space[10, 10, 4] = 1
    space._filled_voxels_count = 1
    result = sphere.deposit_sphere(space, 0.8, 0.02, 0.021, 0.2, 0.05)
    assert result is space
    assert space._filled_voxels_count == np.count_nonzero(space.space)
    assert space._filled_voxels_count > 1


@pytest.mark.parametrize("occupied", [[(0, 0, 0)], [(0, 0, 0), (1, 0, 0)],
                                     list(np.ndindex(2, 2, 2))])
def test_mesh_paths_agree_on_known_solids(occupied, tmp_path):
    grid = np.zeros((3, 3, 3), dtype=np.int8)
    for index in occupied:
        grid[index] = 1
    expected_faces = {1: 12, 2: 20, 8: 48}[len(occupied)]
    meshes = [create_mesh_vectorized(grid, 1.0)]
    for kind, exporter in (("binary", export_voxel_stl_streaming_binary),
                           ("ascii", export_voxel_stl_streaming_ascii)):
        path = tmp_path / (kind + ".stl")
        exporter(grid, 1.0, str(path))
        meshes.append(trimesh.load_mesh(path, force="mesh"))
    for mesh in meshes:
        assert isinstance(mesh, trimesh.Trimesh)
        assert len(mesh.faces) == expected_faces
        assert mesh.is_watertight
        assert mesh.volume == pytest.approx(len(occupied), abs=1e-6)
        expected_bounds = [[0, 0, 0],
                           [max(i[0] for i in occupied) + 1,
                            max(i[1] for i in occupied) + 1,
                            max(i[2] for i in occupied) + 1]]
        np.testing.assert_allclose(mesh.bounds, expected_bounds, atol=1e-6)


@pytest.mark.parametrize("axis", range(3))
def test_binary_export_across_slice_boundary(axis, tmp_path):
    shape = [1, 1, 1]
    shape[axis] = 52
    grid = np.zeros(shape, dtype=np.int8)
    for position in (49, 50):
        index = [0, 0, 0]
        index[axis] = position
        grid[tuple(index)] = 1
    path = tmp_path / "seam.stl"
    origin = np.array([100.123, -20.456, 3.21])
    export_voxel_stl_streaming_binary(grid, 0.1, str(path), origin=origin)
    mesh = trimesh.load_mesh(path, force="mesh")
    expected_bounds = np.array([[0, 0, 0], [.1, .1, .1]])
    expected_bounds[:, axis] = [4.9, 5.1]
    np.testing.assert_allclose(mesh.bounds, expected_bounds + origin, atol=8e-6, rtol=0)
    np.testing.assert_array_equal(mesh.bounds, create_mesh_vectorized(grid, .1, origin=origin).bounds)
    assert len(mesh.faces) == 20
    assert mesh.is_watertight
    assert mesh.volume == pytest.approx(0.002, abs=1e-7)


def test_nonbinary_pitch_ascii_and_mesh_agree(tmp_path):
    grid = np.zeros((4, 3, 2), dtype=np.int8)
    grid[1:3, 1, 0] = 1
    path = tmp_path / "small.stl"
    export_voxel_stl_streaming_ascii(grid, 0.1, str(path))
    for mesh in (trimesh.load_mesh(path, force="mesh"),
                 create_mesh_vectorized(grid, 0.1)):
        assert len(mesh.faces) == 20
        assert mesh.is_watertight
        assert mesh.volume == pytest.approx(0.002, abs=1e-7)
