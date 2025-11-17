import math
import numpy as np

from app.geometry.geometry_math import GeometryMath
from app.solvers.bisection_method import BisectionMethod


class Sphere:
    def __init__(self, centre_coordinates, voxel_size):
        self.centre_coordinates = centre_coordinates
        self.voxel_size = voxel_size

    def deposit_sphere(
        self,
        voxel_space,
        nozzle_height,
        sphere_volume,
        voxel_space_target_volume,
        solver_tolerance,
        radius_increment,
    ):
        initial_radius = self.estimate_initial_radius(sphere_volume)

        # `voxel_space` is expected to be a VoxelSpace object here. The
        # bisection solver and inner functions will mutate and ultimately
        # return that object so callers can read `.space` and counters.
        _, voxel_space_out = BisectionMethod().execute(
            self._deposit_sphere,
            initial_point=initial_radius,
            tolerance=solver_tolerance,
            increment=radius_increment,
            fun_increase_tolerance=self._increase_solver_tolerance,
            args=(
                voxel_space,
                nozzle_height,
                voxel_space_target_volume,
            ),
        )

        return voxel_space_out

    def fill_voxels(self, voxel_space_obj, radius, lower_indexes, upper_indexes):
        # Accept either a VoxelSpace-like object (has `.space`) or a raw ndarray.
        is_voxel_space = hasattr(voxel_space_obj, "space")
        space = voxel_space_obj.space if is_voxel_space else voxel_space_obj

        empty_voxels = GeometryMath.find_empty_voxels_in_space(
            space, lower_indexes, upper_indexes
        )

        if not empty_voxels:
            return voxel_space_obj

        # Convert to numpy array for vectorized operations
        empty_voxels_np = np.array(empty_voxels)
        # Calculate coordinates for all voxels at once
        voxel_coords = self.voxel_size * (2 * (empty_voxels_np + 1) - 1) * 0.5
        # Calculate distances to centre for all voxels
        centre = np.array(self.centre_coordinates)
        dists = np.linalg.norm(voxel_coords - centre, axis=1)
        # Mask for voxels within radius
        mask = dists <= radius + self.voxel_size * 1e-8

        if not np.any(mask):
            return voxel_space_obj

        # Filter indices that should be filled
        fill_indices = empty_voxels_np[mask]

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

        if min_indexes[2] < 0:
            min_indexes[2] = 0

        max_indexes = [
            self._find_index(centre_coordinate + radius)
            for centre_coordinate in self.centre_coordinates
        ]

        _, _, z0 = self.centre_coordinates
        if z0 + radius > nozzle_height:
            max_indexes[2] = self._find_index(nozzle_height)
        else:
            max_indexes[2] = self._find_index(z0 + radius)

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
        # Accept and return either a VoxelSpace object or a raw ndarray.
        vs = self.deform_voxel_space_for_big_spheres(voxel_space, radius)

        lower_indexes, upper_indexes = self.find_sphere_limits(radius, nozzle_height)

        vs = self.fill_voxels(vs, radius, lower_indexes, upper_indexes)

        current_volume = GeometryMath.calculate_filled_volume(vs, self.voxel_size)

        volume_overshoot = current_volume / target_volume - 1.0

        # Return the computed overshoot and the (possibly mutated) voxel container
        return volume_overshoot, vs

    def _increase_solver_tolerance(self, radius_a, radius_b):
        return radius_b - radius_a < self.voxel_size * 0.5

    def _find_index(self, coordinate):
        return GeometryMath.find_index(coordinate, self.voxel_size)
