import logging
import numpy as np
import math

from app.configs.simulation import Simulation
from app.configs.printer import Printer
from app.geometry.sphere import Sphere
from app.instructions.instruction import Instruction
from app.geometry.geometry_math import GeometryMath
from app.physics.volume import Volume


logger = logging.getLogger(__name__)


class VoxelSpace:
    def __init__(
        self, instruction: Instruction, simulation_config: Simulation, printer: Printer):
        coord_limits = instruction.coordinate_limits

        [x_min, x_max] = coord_limits["x"]
        [y_min, y_max] = coord_limits["y"]
        [z_min, z_max] = coord_limits["z"]

        x_offset = simulation_config.x_offset
        y_offset = simulation_config.y_offset
        z_offset = simulation_config.z_offset

        dim = [
            x_max - x_min + 2.0 * x_offset,
            y_max - y_min + 2.0 * y_offset,
            z_max - z_min + z_offset,
        ]

        dimensions = [int(dim_i / simulation_config.voxel_size) for dim_i in dim]
        self.dimensions = {
            "x": dimensions[0],
            "y": dimensions[1],
            "z": dimensions[2],
        }

        self.filament_translations = {"x": x_offset - x_min, "y": y_offset - y_min}

        self._instruction = instruction
        self._simulation = simulation_config
        self._printer = printer
        self._consider_acceleration = self._simulation.consider_acceleration
        # Running count of filled voxels (to avoid repeated `np.count_nonzero` calls)
        self._filled_voxels_count = 0

    def initialize_space(self):
        # Try to intelligently preallocate Z dimension to avoid repeated expansions.
        # Estimate the maximum per-step sphere radius from filaments and add a safety margin.
        z_dim = self.dimensions["z"]

        try:
            max_radius = 0.0
            step_size = self._simulation.step_size
            for filament in self._instruction.filaments_coordinates:
                coord_old = filament[0]
                coord_new = filament[1]
                volume = filament[2]

                # compute filament length and estimated number of steps
                length = GeometryMath.distance(coord_old, coord_new)
                n_steps = int(round(length / step_size))
                if n_steps < 1:
                    n_steps = 1

                per_step_volume = volume / n_steps
                # estimate radius of sphere that would contain this volume
                radius = (3.0 * per_step_volume / (4.0 * math.pi)) ** (1.0 / 3.0)
                if radius > max_radius:
                    max_radius = radius

            # include configured sphere z offset
            max_extra_z = int(math.ceil((max_radius + self._simulation.sphere_z_offset) / self._simulation.voxel_size))

            # safety margin (20% of current z) to reduce chance of further expansion
            safety_margin = int(max(1, z_dim * 0.2))

            if max_extra_z > 0:
                z_dim = z_dim + max_extra_z + safety_margin

            logging.getLogger(__name__).info(f"Preallocating voxel space z-dimension: base={self.dimensions['z']}, extra={max_extra_z}, safety={safety_margin}, final={z_dim}")
        except Exception:
            # Fallback to original allocation if any issue occurs during estimation
            z_dim = self.dimensions["z"]

        self.space = np.zeros(
            (self.dimensions["x"], self.dimensions["y"], z_dim),
            dtype=np.int8,
        )
        # reset the running counter when allocating space
        self._filled_voxels_count = 0

    def print(self):
        # Check for preview mode in simulation config
        if hasattr(self._simulation, "preview_mode") and self._simulation.preview_mode:
            self.print_preview_mode()
            return

        number_printed_filaments = 0
        number_printed_layers = 0
        initial_z_coordinate = 0.0
        print(self._instruction.filaments_coordinates)
        for filament_coordinates in self._instruction.filaments_coordinates:
            (
                initial_coordinate,
                final_coordinate,
            ) = self._find_initial_and_final_filament_coordinates(filament_coordinates)
            direction_vector = GeometryMath.direction_vector(
                initial_coordinate, final_coordinate
            )
            printing_speed = filament_coordinates[1][4]
            volume = filament_coordinates[2]  # Now this is volume, not E
            number_printed_filaments += 1
            if final_coordinate[2] > initial_z_coordinate:
                number_printed_layers += 1
                initial_z_coordinate = final_coordinate[2]
            filament_length = GeometryMath.distance(
                initial_coordinate, final_coordinate
            )
            number_simulation_steps, step_size = self._find_simulation_step_info(
                filament_length
            )
            volumes = Volume.get_volumes_for_filament(
                number_simulation_steps=number_simulation_steps,
                total_volume=volume,
                consider_acceleration=self._consider_acceleration,
                filament_length=filament_length,
                printing_speed=printing_speed,
                printer=self._printer,
            )
            self._deposit_filament(
                number_simulation_steps=number_simulation_steps,
                step_size=step_size,
                direction_vector=direction_vector,
                filament_initial_coordinates=initial_coordinate,
                volumes=volumes,
            )

    def print_preview_mode(self):
        """
        Preview mode: Fast, lightweight visualization. Traces G-code path and fills a capsule (cylinder with rounded endcaps) between last and current point if extruder is on. No physics, no overlap checks. Intended for quick feedback before running the full simulation.
        
        Optimized version with:
        - Direct voxel coordinate computation (no meshgrid)
        - Pre-allocated arrays for reuse
        - Efficient boolean indexing
        """
        import time
        t_start = time.perf_counter()
        
        nozzle_radius = self._printer.nozzle_diameter / 2.0
        voxel_size = self._simulation.voxel_size
        nozzle_radius_sq = nozzle_radius ** 2
        
        # Pre-allocate space for the largest possible bounding box
        # This avoids repeated allocations in the loop
        max_box_size = int(2 * nozzle_radius / voxel_size) + 10
        
        processed = 0
        
        # Use the same logic as the default mode: fill capsule for every filament segment
        for filament_coordinates in self._instruction.filaments_coordinates:
            initial, final = self._find_initial_and_final_filament_coordinates(filament_coordinates)
            volume = filament_coordinates[2]
            
            if volume <= 0:
                continue
            
            processed += 1
            
            # Compute bounding box
            min_x = min(initial[0], final[0]) - nozzle_radius
            max_x = max(initial[0], final[0]) + nozzle_radius
            min_y = min(initial[1], final[1]) - nozzle_radius
            max_y = max(initial[1], final[1]) + nozzle_radius
            min_z = min(initial[2], final[2]) - nozzle_radius
            max_z = max(initial[2], final[2]) + nozzle_radius
            
            i_min = max(0, int(min_x / voxel_size))
            i_max = min(self.space.shape[0] - 1, int(max_x / voxel_size))
            j_min = max(0, int(min_y / voxel_size))
            j_max = min(self.space.shape[1] - 1, int(max_y / voxel_size))
            k_min = max(0, int(min_z / voxel_size))
            k_max = min(self.space.shape[2] - 1, int(max_z / voxel_size))
            
            # Capsule parameters
            p1 = np.array(initial, dtype=np.float32)
            p2 = np.array(final, dtype=np.float32)
            ba = p2 - p1
            ba_dot_ba = np.dot(ba, ba)
            
            if ba_dot_ba < 1e-8:
                # Degenerate case: point sphere
                # Use simple sphere fill
                for i in range(i_min, i_max + 1):
                    x = (i + 0.5) * voxel_size
                    dx = x - p1[0]
                    dx_sq = dx * dx
                    if dx_sq > nozzle_radius_sq:
                        continue
                    for j in range(j_min, j_max + 1):
                        y = (j + 0.5) * voxel_size
                        dy = y - p1[1]
                        dy_sq = dy * dy
                        if dx_sq + dy_sq > nozzle_radius_sq:
                            continue
                        for k in range(k_min, k_max + 1):
                            z = (k + 0.5) * voxel_size
                            dz = z - p1[2]
                            if dx_sq + dy_sq + dz * dz <= nozzle_radius_sq:
                                self.space[i, j, k] = 1
            else:
                # Use vectorized capsule SDF for non-degenerate cases
                # Create coordinate arrays directly without meshgrid
                i_range = np.arange(i_min, i_max + 1, dtype=np.int32)
                j_range = np.arange(j_min, j_max + 1, dtype=np.int32)
                k_range = np.arange(k_min, k_max + 1, dtype=np.int32)
                
                # Compute voxel centers using broadcasting
                ii = i_range[:, None, None]
                jj = j_range[None, :, None]
                kk = k_range[None, None, :]
                
                x = (ii + 0.5) * voxel_size
                y = (jj + 0.5) * voxel_size
                z = (kk + 0.5) * voxel_size
                
                # Capsule SDF computation
                pa_x = x - p1[0]
                pa_y = y - p1[1]
                pa_z = z - p1[2]
                
                h = np.clip((pa_x * ba[0] + pa_y * ba[1] + pa_z * ba[2]) / ba_dot_ba, 0.0, 1.0)
                
                dx = pa_x - ba[0] * h
                dy = pa_y - ba[1] * h
                dz = pa_z - ba[2] * h
                
                dist_sq = dx*dx + dy*dy + dz*dz
                inside = dist_sq <= nozzle_radius_sq
                
                # Assign voxels using boolean indexing
                self.space[i_min + np.where(inside)[0], 
                          j_min + np.where(inside)[1], 
                          k_min + np.where(inside)[2]] = 1
        
        t_end = time.perf_counter()
        logger.info(f"[Profile]: Voxel filling complete. Time: {t_end-t_start:.2f}s, Segments: {processed}")

    def _find_initial_and_final_filament_coordinates(self, filament_coordinates):
        initial = [
            filament_coordinates[0][0] + self.filament_translations["x"],
            filament_coordinates[0][1] + self.filament_translations["y"],
            filament_coordinates[0][2],
        ]
        final = (
            filament_coordinates[1][0] + self.filament_translations["x"],
            filament_coordinates[1][1] + self.filament_translations["y"],
            filament_coordinates[1][2],
        )
        return initial, final

    def _find_simulation_step_info(self, filament_length):
        number_simulation_steps = int(
            round(filament_length / self._simulation.step_size)
        )

        if number_simulation_steps == 0:
            number_simulation_steps = 1

        step_size = filament_length / number_simulation_steps

        return number_simulation_steps, step_size

    def _deposit_filament(
        self,
        number_simulation_steps,
        step_size,
        direction_vector,
        filament_initial_coordinates,
        volumes,
    ):
        # Use the running counter to compute total deposited volume quickly
        total_deposited_volume = (
            self._filled_voxels_count * self._simulation.voxel_size ** 3
        )

        for step_n in range(0, number_simulation_steps):

            # o --------------------- X --------------------- o --------------------- X --------------------- o
            # starting_point      centre_coordinates
            #
            # |<--------------------------------------------->|
            #              step_size
            #
            # |<--------------------------------------------------------------------------------------------->|
            #                                           filament length

            displacement = step_n * step_size
            starting_point = [
                pi_i + dir_vec_i * displacement
                for (pi_i, dir_vec_i) in zip(
                    filament_initial_coordinates, direction_vector
                )
            ]

            displacement_to_centre = step_size * 0.5
            centre_coordinates = [
                pstart_i + dir_vec_i * displacement_to_centre
                for (pstart_i, dir_vec_i) in zip(starting_point, direction_vector)
            ]

            nozzle_height = centre_coordinates[2]

            # Correcting the centre of the sphere based on the configured z-offset
            centre_coordinates[2] = (
                centre_coordinates[2] - self._simulation.sphere_z_offset
            )

            sphere_volume = volumes[step_n]
            volume_target = total_deposited_volume + sphere_volume

            sphere = Sphere(
                centre_coordinates=centre_coordinates,
                voxel_size=self._simulation.voxel_size,
            )

            logger.info(
                f"Depositing filament: step = {step_n + 1}/{number_simulation_steps}"
            )

            # Pass the VoxelSpace instance so sphere code can update the
            # running counter and expand the `.space` ndarray in-place.
            voxel_space_out = sphere.deposit_sphere(
                voxel_space=self,
                nozzle_height=nozzle_height,
                sphere_volume=sphere_volume,
                voxel_space_target_volume=volume_target,
                solver_tolerance=self._simulation.solver_tolerance,
                radius_increment=self._simulation.radius_increment,
            )

            # Ensure our internal `.space` and counter reflect any mutations
            # returned by the sphere logic (voxel_space_out is a VoxelSpace)
            if hasattr(voxel_space_out, "space"):
                self.space = voxel_space_out.space
            if hasattr(voxel_space_out, "_filled_voxels_count"):
                self._filled_voxels_count = voxel_space_out._filled_voxels_count
