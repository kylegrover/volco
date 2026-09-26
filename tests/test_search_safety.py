"""Numerical safety contracts; these do not validate physical predictions."""
from types import SimpleNamespace

import numpy as np
import pytest

from app.configs.simulation import Simulation
from app.geometry.geometry_math import GeometryMath
from app.geometry.sphere import Sphere
from app.geometry.voxel_space import VoxelSpace
from app.solvers.bisection_method import BisectionMethod


def container(shape=(30, 30, 12)):
    return SimpleNamespace(space=np.zeros(shape, dtype=np.int8), _filled_voxels_count=0)


def test_negative_slice_limits_are_clipped_not_wrapped():
    indices = np.asarray(GeometryMath.find_empty_voxels_in_space(
        np.zeros((5, 5, 5)), [-2, -2, -2], [1, 1, 1]))
    assert len(indices) == 8
    assert np.all(indices >= 0)
    assert GeometryMath.find_empty_voxels_in_space(np.zeros((5, 5, 5)), [0]*3, [-1]*3) == []


def test_boundary_candidates_are_nested_and_never_wrap():
    sphere = Sphere([0.05, 0.05, 0.15], 0.1)
    space = np.zeros((10, 10, 5), dtype=np.int8)
    previous = set()
    for radius in (0.1, 0.2, 0.4, 0.6):
        indices = sphere._new_voxel_indices(space, radius, *sphere.find_sphere_limits(radius, 0.3))
        current = set(map(tuple, indices))
        assert previous <= current
        assert np.all(indices >= 0)
        previous = current
    sphere.fill_voxels(space, 0.2, *sphere.find_sphere_limits(0.2, 0.3))
    assert not space[-1].any() and not space[:, -1].any()


def test_nozzle_plane_excludes_centres_above_it():
    sphere = Sphere([0.15, 0.15, 0.15], 0.1)
    indices = sphere._new_voxel_indices(np.zeros((5, 5, 5)), 0.3,
                                       *sphere.find_sphere_limits(0.3, 0.21))
    assert np.all((indices[:, 2] + 0.5) * 0.1 <= 0.21)


@pytest.mark.parametrize('prior', [0, 1000])
def test_local_target_not_hidden_by_cumulative_normalization(prior):
    space = container((120, 120, 100))
    if prior:
        space.space[20:120, :100, :100] = 1
        space._filled_voxels_count = 1000000
    sphere = Sphere([1, 1, 0.3], 0.1)
    sphere.deposit_sphere(space, 0.5, 0.02, prior + 0.02, 0.0001, 0.1)
    added = space._filled_voxels_count - prior * 1000
    assert abs(added * 0.001 - 0.02) <= 0.00101
    report = sphere.last_deposition
    assert report['increment_residual_mm3'] == pytest.approx(added * .001 - .02)
    assert report['evaluations'] <= 128
    assert space._filled_voxels_count == np.count_nonzero(space.space)


def test_resource_failure_does_not_mutate_or_expand_grid():
    space = container()
    before = space.space.copy()
    sphere = Sphere([1, 1, .3], .1)
    with pytest.raises(RuntimeError, match='candidate|budget'):
        sphere.deposit_sphere(space, .5, 1000, 1000, .0001, .1,
                              max_candidate_voxels=1000)
    np.testing.assert_array_equal(space.space, before)
    assert space._filled_voxels_count == 0


def test_empty_nozzle_domain_fails_without_growth():
    space = container()
    with pytest.raises(RuntimeError, match='nozzle'):
        Sphere([1, 1, -.1], .1).deposit_sphere(space, 0, .02, .02, .0001, .1)
    assert space.space.shape == (30, 30, 12)


@pytest.mark.parametrize('key,value', [('voxel_size', 0), ('step_size', float('nan')),
    ('radius_increment', -1), ('solver_tolerance', 0)])
def test_invalid_simulation_search_parameters_rejected(key, value):
    config = dict(voxel_size=.1, step_size=.2, radius_increment=.1,
                  solver_tolerance=.0001, simulation_name='test', results_folder='unused')
    config[key] = value
    with pytest.raises(ValueError, match=key):
        Simulation(config_dict=config)


def test_commanded_volume_debt_survives_move_boundaries(monkeypatch):
    targets = []
    def capture(self, **kwargs):
        targets.append(kwargs['voxel_space_target_volume'])
        return kwargs['voxel_space']
    monkeypatch.setattr(Sphere, 'deposit_sphere', capture)
    space = VoxelSpace.__new__(VoxelSpace)
    space._simulation = SimpleNamespace(voxel_size=.1, sphere_z_offset=0,
                                       solver_tolerance=.0001, radius_increment=.1)
    space._filled_voxels_count = 0
    for _ in range(2):
        space._deposit_filament(1, .2, [1, 0, 0], [1, 1, .3], [.0005])
    assert targets == pytest.approx([.0005, .001])


@pytest.mark.parametrize('pitch,target', [(.1, .05), (.05, .05), (.1, .0005)])
def test_nearest_volume_matches_exhaustive_small_domain(pitch, target):
    space = container((60, 60, 24))
    sphere = Sphere([1, 1, .3], pitch)
    sphere.deposit_sphere(space, .5, target, target, .0001, .1)
    # Independent finite lattice oracle: evaluate every cell-entry threshold,
    # including rounding neighbors of the radius + inclusion-epsilon test.
    coordinates = (np.indices((int(2/pitch),)*2 + (int(.5/pitch),)).reshape(3, -1).T + .5) * pitch
    distances = np.linalg.norm(coordinates - [1, 1, .3], axis=1)
    entries = np.unique(distances[distances < .4] - pitch * 1e-8)
    candidates = np.unique(np.concatenate([entries, np.nextafter(entries, -np.inf), np.nextafter(entries, np.inf)]))
    volumes = np.array([0] + [np.count_nonzero(distances <= r + pitch * 1e-8) for r in candidates]) * pitch**3
    best_error = np.min(np.abs(volumes - target))
    assert abs(space._filled_voxels_count * pitch**3 - target) == pytest.approx(best_error, abs=1e-12)


@pytest.mark.parametrize('budget', ['max_evaluations', 'max_grid_voxels'])
def test_exhausted_budget_has_diagnostics_and_no_commit(budget):
    space = container()
    sphere = Sphere([1, 1, .3], .1)
    with pytest.raises(RuntimeError, match='budget') as error:
        sphere.deposit_sphere(space, .5, .02, .02, .0001, .1, **{budget: 1})
    assert error.value.diagnostics['status'] == 'failed'
    assert 'best_residual_mm3' in error.value.diagnostics
    assert not space.space.any() and space._filled_voxels_count == 0


def test_explicit_volume_limit_rejects_before_commit():
    space = container()
    with pytest.raises(RuntimeError, match='max_volume_error_mm3'):
        Sphere([1, 1, .3], .1).deposit_sphere(space, .5, .0005, .0005, .0001, .1,
                                             max_volume_error_mm3=.0001)
    assert not space.space.any()


def test_explicit_local_limit_can_reject_debt_repayment():
    space = container()
    sphere = Sphere([1, 1, .3], .1)
    # The cumulative target includes old debt, not just this .001 increment.
    with pytest.raises(RuntimeError, match='max_increment_error_mm3'):
        sphere.deposit_sphere(space, .5, .001, .02, .0001, .1,
                              max_increment_error_mm3=.002)
    assert not space.space.any() and space._filled_voxels_count == 0
    assert sphere.last_deposition['committed'] is False


def test_raw_array_and_object_expansion_agree():
    raw = np.zeros((2, 2, 2), dtype=np.int8)
    obj = container((2, 2, 2))
    args = (.5, .02, .02, .0001, .1)
    result = Sphere([.3, .3, .3], .1).deposit_sphere(raw, *args)
    Sphere([.3, .3, .3], .1).deposit_sphere(obj, *args)
    assert not raw.any()  # expanded return must be used
    assert result.shape != raw.shape
    np.testing.assert_array_equal(result, obj.space)
    assert obj._filled_voxels_count == np.count_nonzero(result)


@pytest.mark.parametrize('phase', [0, .17, .5])
@pytest.mark.parametrize('pitch', [.05, .1])
def test_blocked_boundary_search_matches_finite_lattice_oracle(phase, pitch):
    space = container((16, 16, 8))
    space.space[:2, :3, :2] = 1
    space._filled_voxels_count = int(np.count_nonzero(space.space))
    before = space._filled_voxels_count
    centre = np.array([phase, .5, 1.5]) * pitch
    nozzle = 3.2 * pitch
    target_count = 10.3
    sphere = Sphere(centre, pitch)
    sphere.deposit_sphere(space, nozzle, target_count * pitch**3,
                          (before + target_count) * pitch**3, .0001, pitch)
    indices = np.indices((16, 16, 3)).reshape(3, -1).T
    indices = indices[~((indices[:, 0] < 2) & (indices[:, 1] < 3) & (indices[:, 2] < 2))]
    distances = np.linalg.norm((indices + .5) * pitch - centre, axis=1)
    entries = np.unique(distances[distances < 5 * pitch] - pitch * 1e-8)
    radii = np.unique(np.concatenate([entries, np.nextafter(entries, -np.inf), np.nextafter(entries, np.inf)]))
    counts = np.array([0] + [np.count_nonzero(distances <= r + pitch * 1e-8) for r in radii])
    assert abs(space._filled_voxels_count - before - target_count) == pytest.approx(np.min(abs(counts - target_count)))
    assert space._filled_voxels_count == np.count_nonzero(space.space)
    assert not space.space[-1].any() and not space.space[:, -1].any()


def test_actual_subvoxel_debt_is_not_discarded_between_moves():
    space = VoxelSpace.__new__(VoxelSpace)
    space._simulation = SimpleNamespace(voxel_size=.1, sphere_z_offset=0,
                                       solver_tolerance=.0001, radius_increment=.1)
    space.space = np.zeros((10, 10, 10), dtype=np.int8)
    space._filled_voxels_count = 0
    for _ in range(2):
        space._deposit_filament(1, 0, [0, 0, 0], [.15, .15, .15], [.0005])
    assert space._commanded_volume == pytest.approx(.001)
    assert space._filled_voxels_count == 1
    assert space.volume_summary['cumulative_residual_mm3'] == pytest.approx(0, abs=1e-12)


@pytest.mark.parametrize('function', [lambda r: (-1, r), lambda r: (float('nan'), r),
                                      lambda r: (-1 if r < .5 else 1, r)])
def test_generic_solver_is_also_finite(function):
    with pytest.raises((RuntimeError, ValueError)):
        BisectionMethod().execute(function, .1, 1e-6, .1, (), max_evaluations=80)

