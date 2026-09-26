"""Differential controls against the pre-optimization candidate enumeration."""
from types import SimpleNamespace

import numpy as np
import pytest

from app.geometry.sphere import DepositionError, Sphere


def reference_candidates(self, space, radius, lower, upper, cache=None):
    # Frozen 8a964fc implementation, independent of the optimized path.
    shape = tuple(max(0, hi - lo + 1) for lo, hi in zip(lower, upper))
    if not all(shape):
        return np.empty((0, 3), dtype=int)
    indices = np.indices(shape).reshape(3, -1).T + np.asarray(lower)
    distances = np.linalg.norm((indices + .5) * self.voxel_size - self.centre_coordinates, axis=1)
    indices = indices[distances <= radius + self.voxel_size * 1e-8]
    inside = np.all(indices < np.asarray(space.shape), axis=1)
    empty = np.ones(len(indices), dtype=bool)
    empty[inside] = space[tuple(indices[inside].T)] == 0
    return indices[empty]


class ReferenceSphere(Sphere):
    _virtual_candidates = reference_candidates


@pytest.mark.parametrize('pitch', [.05, .1, .2])
@pytest.mark.parametrize('centre_cells', [[0, 0, 1.5], [2.3, 3.7, 2.1], [7.5, 7.5, 4.5]])
def test_candidate_order_and_shell_boundaries(pitch, centre_cells):
    rng = np.random.default_rng(824)
    space = (rng.random((8, 8, 5)) < .5).astype(np.int8)
    sphere = Sphere(np.asarray(centre_cells) * pitch, pitch)
    cache = {}
    for radius in np.concatenate((np.linspace(0, 5 * pitch, 31), np.linspace(5 * pitch, 0, 31))):
        lower, upper = sphere.find_sphere_limits(radius, 4.5 * pitch)
        for r in [radius, np.nextafter(radius, 0), np.nextafter(radius, np.inf)]:
            expected = reference_candidates(sphere, space, r, lower, upper)
            actual = sphere._virtual_candidates(space, r, lower, upper, cache)
            np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('pitch', [.05, .1, .2])
@pytest.mark.parametrize('guard', [{}, {'max_evaluations': 1}, {'max_candidate_voxels': 1},
                                  {'max_grid_voxels': 256}, {'max_volume_error_mm3': 0},
                                  {'max_increment_error_mm3': 0}])
def test_complete_search_parity_including_failure_and_reuse(pitch, guard):
    rng = np.random.default_rng(123)
    initial = (rng.random((8, 8, 4)) < .3).astype(np.int8)
    objects = [SimpleNamespace(space=initial.copy(), _filled_voxels_count=int(initial.sum())) for _ in range(2)]
    spheres = [cls([.3 * pitch, 7.7 * pitch, 2.3 * pitch], pitch) for cls in (Sphere, ReferenceSphere)]
    target = int(initial.sum()) * pitch**3
    # Reuse Sphere instances across changes to occupancy and expansion; no stale cache.
    for cells in [7.3, .1, 15.7, 0, 4.2]:
        target += cells * pitch**3
        errors = []
        for obj, sphere in zip(objects, spheres):
            before = obj.space.copy()
            try:
                sphere.deposit_sphere(obj, 3.5 * pitch, cells * pitch**3, target, .0001, pitch, **guard)
                errors.append(None)
            except DepositionError as exc:
                np.testing.assert_array_equal(obj.space, before)
                errors.append((str(exc), exc.diagnostics))
        assert errors[0] == errors[1]
        assert spheres[0].last_deposition == spheres[1].last_deposition
        assert objects[0]._filled_voxels_count == objects[1]._filled_voxels_count
        np.testing.assert_array_equal(objects[0].space, objects[1].space)


def test_same_box_enumerated_once_and_cache_replaces_box(monkeypatch):
    sphere = Sphere([.25, .25, .15], .1)
    space = np.zeros((8, 8, 4), dtype=np.int8)
    original = np.indices
    calls = []
    def counted(shape):
        calls.append(shape)
        return original(shape)
    monkeypatch.setattr(np, 'indices', counted)
    cache = {}
    for radius in [.2, .21, .22]:
        sphere._virtual_candidates(space, radius, [0, 0, 0], [4, 4, 2], cache)
    assert len(calls) == 1
    sphere._virtual_candidates(space, .3, [0, 0, 0], [5, 5, 2], cache)
    assert len(calls) == 2
    sphere._virtual_candidates(space, .2, [0, 0, 0], [4, 4, 2], cache)
    assert len(calls) == 3  # Only one box retained, not an unbounded dictionary.
