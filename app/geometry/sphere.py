import math
import logging
import numpy as np

from app.geometry.geometry_math import GeometryMath
logger = logging.getLogger(__name__)


class DepositionError(RuntimeError):
    """No material was committed for this step; diagnostics describe why."""

    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.diagnostics = diagnostics


class Sphere:
    def __init__(self, centre_coordinates, voxel_size):
        if not math.isfinite(voxel_size) or voxel_size <= 0:
            raise ValueError('voxel_size must be finite and positive')
        if len(centre_coordinates) != 3 or not all(math.isfinite(c) for c in centre_coordinates):
            raise ValueError('centre_coordinates must contain three finite values')
        self.centre_coordinates = centre_coordinates
        self.voxel_size = voxel_size
        self.last_deposition = None

    def deposit_sphere(
        self,
        voxel_space,
        nozzle_height,
        sphere_volume,
        voxel_space_target_volume,
        solver_tolerance,
        radius_increment,
        *,
        max_evaluations=128,
        max_candidate_voxels=250000,
        max_grid_voxels=64000000,
        max_volume_error_mm3=None,
        max_increment_error_mm3=None,
    ):
        """Select the closest discrete volume without relaxing tolerance.

        The nonnegative grid is the allowed domain; positive expansion is
        virtual until commit. Budgets are operational guards, not physical
        radius limits. Ties choose underfill. Diagnostics retain both debts.
        """
        for name, value in [('sphere_volume', sphere_volume),
                            ('voxel_space_target_volume', voxel_space_target_volume)]:
            if not math.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        for name, value in [('solver_tolerance', solver_tolerance), ('radius_increment', radius_increment)]:
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
        for name, value in [('max_evaluations', max_evaluations),
                            ('max_candidate_voxels', max_candidate_voxels), ('max_grid_voxels', max_grid_voxels)]:
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f'{name} must be a positive integer')
        if not math.isfinite(nozzle_height):
            raise ValueError('nozzle_height must be finite')
        for name, value in [('max_volume_error_mm3', max_volume_error_mm3),
                            ('max_increment_error_mm3', max_increment_error_mm3)]:
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError(f'{name} must be finite and nonnegative')

        space = voxel_space.space if hasattr(voxel_space, 'space') else voxel_space
        cell_volume = self.voxel_size ** 3
        before = GeometryMath.calculate_filled_volume(voxel_space, self.voxel_size)
        needed = (voxel_space_target_volume - before) / cell_volume
        report = dict(requested_increment_mm3=sphere_volume,
                      cumulative_target_mm3=voxel_space_target_volume,
                      occupied_before_mm3=before, evaluations=0, status='searching',
                      committed=False, best_added_voxels=0)
        self.last_deposition = report

        def fail(reason):
            report['status'] = 'failed'
            report['reason'] = reason
            report['best_residual_mm3'] = before + report['best_added_voxels'] * cell_volume - voxel_space_target_volume
            raise DepositionError(f'{reason}; target={voxel_space_target_volume:g} mm3, '
                                  f'best residual={report["best_residual_mm3"]:g} mm3', dict(report))

        def evaluate(radius):
            if report['evaluations'] >= max_evaluations:
                fail('radius search evaluation budget exhausted')
            if not math.isfinite(radius):
                fail('radius search overflow')
            lower, upper = self.find_sphere_limits(radius, nozzle_height)
            lengths = [max(0, hi - lo + 1) for lo, hi in zip(lower, upper)]
            if math.prod(lengths) > max_candidate_voxels:
                fail('candidate voxel budget exhausted')
            shape = tuple(max(n, hi + 1) for n, hi in zip(space.shape, upper))
            if math.prod(shape) > max_grid_voxels:
                fail('grid voxel budget exhausted')
            report['evaluations'] += 1
            indices = self._virtual_candidates(space, radius, lower, upper)
            count = len(indices)
            if abs(count - needed) < abs(report['best_added_voxels'] - needed):
                report['best_added_voxels'] = count
            return count

        low, low_count = 0.0, 0
        high, high_count = 0.0, 0
        radius, count = 0.0, 0
        if needed > 1e-9:
            if nozzle_height < self.voxel_size * .5:
                fail('no voxel centre lies below the nozzle plane')
            high = max(self.estimate_initial_radius(max(sphere_volume, needed * cell_volume)), radius_increment)
            high_count = evaluate(high)
            while high_count < needed - 1e-9:
                low, low_count = high, high_count
                high = max(high * 2, high + radius_increment)
                high_count = evaluate(high)
            # Refine to adjacent representable radii (or adjacent counts).
            # A flat count does NOT imply there are no later shells.
            while high_count - low_count > 1 and abs(high_count - needed) > 1e-9:
                middle = low + (high - low) * .5
                if middle == low or middle == high:
                    break
                middle_count = evaluate(middle)
                if middle_count < needed - 1e-9:
                    low, low_count = middle, middle_count
                else:
                    high, high_count = middle, middle_count
            radius, count = min([(low, low_count), (high, high_count)],
                                key=lambda item: (abs(item[1] - needed), item[1]))

        after = before + count * cell_volume
        residual = after - voxel_space_target_volume
        report.update(selected_radius_mm=radius, added_voxels=count,
                      actual_increment_mm3=count * cell_volume,
                      occupied_after_mm3=after,
                      increment_residual_mm3=count * cell_volume - sphere_volume,
                      cumulative_residual_mm3=residual,
                      increment_relative_residual=(count * cell_volume / sphere_volume - 1) if sphere_volume else None,
                      bracket_added_voxels=[low_count, high_count],
                      bracket_residuals_mm3=[before + n * cell_volume - voxel_space_target_volume
                                             for n in (low_count, high_count)],
                      within_increment_tolerance=abs(count * cell_volume - sphere_volume) <= solver_tolerance * sphere_volume + cell_volume * 1e-9,
                      status=('exact' if abs(residual) <= cell_volume * 1e-9 else
                              'carried_overshoot' if needed < 0 else 'quantized'))
        if max_volume_error_mm3 is not None and abs(residual) > max_volume_error_mm3 + cell_volume * 1e-9:
            fail('cumulative volume residual exceeds max_volume_error_mm3')
        if max_increment_error_mm3 is not None and abs(report['increment_residual_mm3']) > max_increment_error_mm3 + cell_volume * 1e-9:
            fail('local increment residual exceeds max_increment_error_mm3')
        if count:
            lower, upper = self.find_sphere_limits(radius, nozzle_height)
            indices = self._virtual_candidates(space, radius, lower, upper)
            if len(indices) != count:
                fail('candidate count changed before commit')
            shape = tuple(max(n, int(indices[:, axis].max()) + 1) for axis, n in enumerate(space.shape))
            if shape != space.shape:
                expanded = np.zeros(shape, dtype=space.dtype)
                expanded[tuple(slice(0, n) for n in space.shape)] = space
                space = expanded
            space[tuple(indices.T)] = 1
            if hasattr(voxel_space, 'space'):
                voxel_space.space = space
                if hasattr(voxel_space, '_filled_voxels_count'):
                    voxel_space._filled_voxels_count += count
            else:
                voxel_space = space
        report['committed'] = True
        # INFO is the normal CLI log level: do not silently discard a local
        # miss just because its whole-print relative residual is small.
        logger.info('Deposition volume: %s', report)
        return voxel_space

    def _virtual_candidates(self, space, radius, lower, upper):
        """Count positive-side expansion without allocating an expanded grid."""
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

    def _new_voxel_indices(self, space, radius, lower_indexes, upper_indexes):
        empty_voxels = GeometryMath.find_empty_voxels_in_space(
            space, lower_indexes, upper_indexes
        )
        if not empty_voxels:
            return np.empty((0, 3), dtype=int)

        empty_voxels_np = np.asarray(empty_voxels)
        voxel_coords = (empty_voxels_np + 0.5) * self.voxel_size
        distances = np.linalg.norm(voxel_coords - self.centre_coordinates, axis=1)
        return empty_voxels_np[distances <= radius + self.voxel_size * 1e-8]

    def fill_voxels(self, voxel_space_obj, radius, lower_indexes, upper_indexes):
        # Accept either a VoxelSpace-like object (has `.space`) or a raw ndarray.
        is_voxel_space = hasattr(voxel_space_obj, "space")
        space = voxel_space_obj.space if is_voxel_space else voxel_space_obj

        fill_indices = self._new_voxel_indices(space, radius, lower_indexes, upper_indexes)
        if not len(fill_indices):
            return voxel_space_obj

        # Advanced index assignment (vectorized)
        xi = fill_indices[:, 0].astype(int)
        yj = fill_indices[:, 1].astype(int)
        zk = fill_indices[:, 2].astype(int)

        # Before setting, count how many of these are actually zero (defensive)
        current_vals = space[xi, yj, zk]
        new_mask = current_vals == 0
        n_new = int(np.count_nonzero(new_mask))
        if n_new > 0:
            # Assign into the ndarray `space` (in-place)
            space[xi[new_mask], yj[new_mask], zk[new_mask]] = 1
            # Update running counter only when we were given a VoxelSpace object
            if is_voxel_space and hasattr(voxel_space_obj, "_filled_voxels_count"):
                voxel_space_obj._filled_voxels_count += n_new

        # Return the same type we were given
        if is_voxel_space:
            return voxel_space_obj
        return space

    def estimate_initial_radius(self, volume):
        return (3.0 * volume / (4.0 * math.pi)) ** (1.0 / 3.0)

    def find_sphere_limits(self, radius, nozzle_height):
        # subtract 1 because the voxel space starts at position [0,0]
        min_indexes = [
            self._find_index(centre_coordinate - radius)
            for centre_coordinate in self.centre_coordinates
        ]

        min_indexes = [max(0, index) for index in min_indexes]

        max_indexes = [
            self._find_index(centre_coordinate + radius)
            for centre_coordinate in self.centre_coordinates
        ]

        _, _, z0 = self.centre_coordinates
        if z0 + radius > nozzle_height:
            max_indexes[2] = self._find_index(nozzle_height)
        else:
            max_indexes[2] = self._find_index(z0 + radius)

        # A cell belongs below the nozzle only when its centre does. The old
        # cell-overlap bound admitted a centre above a non-grid-aligned plane.
        max_indexes[2] = min(max_indexes[2], math.floor(nozzle_height / self.voxel_size - 0.5))
        return min_indexes, max_indexes

    """
    Checks if the sphere violates the boundaries of the voxel space. If it does,
    the voxel space is expanded to accommodate the sphere.
    """

    def deform_voxel_space_for_big_spheres(self, voxel_space, radius):
        # Accept either a VoxelSpace object or a raw ndarray and return the same type.
        max_indexes = [
            self._find_index(coord + radius) for coord in self.centre_coordinates
        ]

        vs = voxel_space
        for axis_number in range(0, 3):
            vs = self._maybe_expand_voxel_space(vs, max_indexes[axis_number], axis_number)

        return vs

    def _maybe_expand_voxel_space(self, voxel_space_obj, max_index, axis_number):
        # Accept either a VoxelSpace-like object (has `.space`) or a raw ndarray.
        is_voxel_space = hasattr(voxel_space_obj, "space")
        space = voxel_space_obj.space if is_voxel_space else voxel_space_obj

        size = space.shape
        index_size = size[axis_number]

        if max_index < index_size:
            return voxel_space_obj

        number_to_be_added = max_index - index_size + 1

        # Add a buffer to reduce the number of reallocations. Buffer is 20%
        # of current size (rounded), but at least the minimum required.
        buffer_layers = max(int(index_size * 0.2), 1)
        layers_to_add = max(number_to_be_added, buffer_layers)

        # Create only the additional block to concatenate
        mat_add_size = list(size)
        mat_add_size[axis_number] = layers_to_add
        mat_add = np.zeros(mat_add_size, dtype=np.int8)

        new_space = np.concatenate((space, mat_add), axis=axis_number)

        if is_voxel_space:
            voxel_space_obj.space = new_space
            return voxel_space_obj

        return new_space

    def _deposit_sphere(self, radius, voxel_space, nozzle_height, target_volume):
        # Compatibility helper for scalar trial callers: no expansion or fill.
        lower_indexes, upper_indexes = self.find_sphere_limits(radius, nozzle_height)
        if math.prod(max(0, hi - lo + 1) for lo, hi in zip(lower_indexes, upper_indexes)) > 250000:
            raise DepositionError('candidate voxel budget exhausted', {'radius_mm': radius})
        space = voxel_space.space if hasattr(voxel_space, 'space') else voxel_space
        n_new = len(self._virtual_candidates(space, radius, lower_indexes, upper_indexes))
        current_volume = GeometryMath.calculate_filled_volume(voxel_space, self.voxel_size)
        volume_overshoot = (current_volume + n_new * self.voxel_size**3) / target_volume - 1.0

        # Return the radius to the solver without committing this trial.
        return volume_overshoot, radius

    def _find_index(self, coordinate):
        return GeometryMath.find_index(coordinate, self.voxel_size)
