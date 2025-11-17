import math
import numpy as np


class GeometryMath:
    @staticmethod
    def distance(point_a, point_b):
        d = 0.0
        for (p1_i, p2_i) in zip(point_a, point_b):
            d += (p1_i - p2_i) ** 2
        return math.sqrt(d)

    @staticmethod
    def direction_vector(point_a, point_b):
        vec = list()
        norm = 0.0

        for (point_a_i, point_b_i) in zip(point_a, point_b):

            aux = point_b_i - point_a_i
            vec.append(aux)

            norm += aux**2

        norm = math.sqrt(norm)

        return [pi / norm for pi in vec]

    @staticmethod
    def find_empty_voxels_in_space(voxel_space, lower_indexes, upper_indexes):
        lower_i, lower_j, lower_k = lower_indexes
        upper_i, upper_j, upper_k = upper_indexes

        mat_aux = voxel_space[
            lower_i : upper_i + 1, lower_j : upper_j + 1, lower_k : upper_k + 1
        ]

        empty_voxels = np.transpose(np.where(mat_aux == 0))

        empty_voxels = [
            [elem[0] + lower_i, elem[1] + lower_j, elem[2] + lower_k]
            for elem in empty_voxels
        ]
        return empty_voxels

    @staticmethod
    def find_coordinates(indexes, voxel_size):
        return [voxel_size * (2 * (index + 1) - 1) * 0.5 for index in indexes]

    @staticmethod
    def calculate_filled_volume(voxel_space, voxel_size):
        # Accept either a raw ndarray or a VoxelSpace-like object with a
        # `_filled_voxels_count` running counter. Prefer the running counter
        # when available to avoid expensive full-array scans.
        if hasattr(voxel_space, "_filled_voxels_count"):
            return int(voxel_space._filled_voxels_count) * voxel_size ** 3

        # If an object with `.space` is provided, try to use its counter first.
        if hasattr(voxel_space, "space") and hasattr(voxel_space.space, "shape"):
            # If the container itself tracks a counter, use it.
            if hasattr(voxel_space, "_filled_voxels_count"):
                return int(voxel_space._filled_voxels_count) * voxel_size ** 3
            # Fallback to scanning the ndarray
            return np.count_nonzero(voxel_space.space) * voxel_size ** 3

        # Otherwise assume voxel_space is a numpy array
        return np.count_nonzero(voxel_space) * voxel_size**3

    def find_index(coordinate, voxel_size):
        return int(math.ceil(coordinate / voxel_size) - 1)
