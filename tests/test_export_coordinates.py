"""Cell corners and crop/world transforms, not physical validation."""
from types import SimpleNamespace

import numpy as np
import pytest
import trimesh

from app.reporter.report import SimulationOutput
from app.reporter import mesh as mesher


def output_for(tmp_path, pitch=.1, crop=True):
    grid = np.zeros((8, 9, 7), dtype=np.int8)
    grid[2:5, 3:6, 1:4] = 1
    translation = np.array([-100.123, 20.456, 0])
    sim = SimpleNamespace(voxel_size=pitch, simulation_name='frame', results_folder=str(tmp_path),
                          stl_ascii=False)
    for axis, start, stop, shift in zip('xyz', [2, 3, 1], [5, 6, 4], translation):
        setattr(sim, axis+'_crop', [start*pitch-shift, stop*pitch-shift] if crop else ['all', 'all'])
    space = SimpleNamespace(space=grid, filament_translations=dict(x=translation[0], y=translation[1]))
    return SimulationOutput(space, sim), translation


@pytest.mark.parametrize('crop', [False, True])
def test_world_frame_parity(tmp_path, crop):
    output, translation = output_for(tmp_path, crop=crop)
    original = output.voxel_space.space.copy()
    output.crop_voxel_space()
    expected = np.array([[.2, .3, .1], [.5, .6, .4]]) - translation
    meshes = [output.generate_mesh()]
    for ascii_format in (False, True):
        output._simulation.stl_ascii = ascii_format
        path = output.export_mesh_to_stl()
        meshes.append(trimesh.load_mesh(path, force='mesh'))
    for mesh in meshes:
        np.testing.assert_allclose(mesh.bounds, expected, atol=8e-6, rtol=0)
        assert mesh.is_watertight and mesh.is_winding_consistent
        assert mesh.volume == pytest.approx(.027, abs=2e-6)
        assert len(mesh.faces) == 108
    np.testing.assert_array_equal(meshes[0].bounds, meshes[1].bounds)
    np.testing.assert_array_equal(meshes[1].bounds, meshes[2].bounds)
    np.testing.assert_array_equal(output.voxel_space.space, original)
    if crop:
        assert output.cropped_voxel_space.shape == (3, 3, 3)
        np.testing.assert_array_equal(output.crop_start, [2, 3, 1])


def test_crop_axis_is_half_open_cell_intersection(tmp_path):
    output, _ = output_for(tmp_path)
    assert output._crop_axis([.2, .5], 0, 8) == [2, 5]
    assert output._crop_axis([.21, .49], 0, 8) == [2, 5]
    assert output._crop_axis([-1, .21], 0, 8) == [0, 3]
    assert output._crop_axis(['all', 'all'], 0, 8) == [0, 8]
    for limits in ([.5, .2], [.2, .2], [-2, -1], [1, 2], [float('nan'), 1]):
        with pytest.raises(ValueError):
            output._crop_axis(limits, 0, 8)


def test_failed_recrop_invalidates_previous_geometry(tmp_path):
    output, _ = output_for(tmp_path)
    output.crop_voxel_space()
    output.generate_mesh()
    output._simulation.x_crop = [200, 201]
    with pytest.raises(ValueError):
        output.crop_voxel_space()
    assert output.cropped_voxel_space is None and output.mesh is None


def test_recrop_preserves_world_frame_and_closes_cut_surface(tmp_path):
    output, translation = output_for(tmp_path)
    output.crop_voxel_space()
    output.generate_mesh()
    output._simulation.x_crop = [.3 - translation[0], .4 - translation[0]]
    output.crop_voxel_space()
    assert output.mesh is None
    assert output.cropped_voxel_space.shape == (1, 3, 3)
    mesh = output.generate_mesh()
    assert mesh.is_watertight and mesh.is_winding_consistent
    assert mesh.volume == pytest.approx(.009, abs=1e-6)
    np.testing.assert_allclose(mesh.bounds, np.array([[.3, .3, .1], [.4, .6, .4]]) - translation,
                               atol=8e-6, rtol=0)
    output.crop_voxel_space()
    np.testing.assert_array_equal(output.generate_mesh().bounds, mesh.bounds)


def test_mesher_failure_does_not_silently_switch_geometry(monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError('deliberate mesher failure')
    monkeypatch.setattr(mesher, 'create_mesh_vectorized', broken)
    with pytest.raises(RuntimeError, match='deliberate'):
        mesher.generate_mesh_from_voxels(np.ones((2, 2, 2)), .1)
