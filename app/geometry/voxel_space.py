import logging
import numpy as np
import math

from app.configs.simulation import Simulation
from app.configs.printer import Printer
from app.geometry.sphere import Sphere
from app.instructions.instruction import Instruction
from app.geometry.geometry_math import GeometryMath
from app.physics.volume import Volume
from app.physics.solver import PhysicsSolver
from app.physics.filament_solver import FilamentSolver


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

        # Initialize Physics Solver if enabled
        if self._simulation.physics_enabled:
            self.physics_solver = PhysicsSolver(
                shape=self.space.shape,
                voxel_size=self._simulation.voxel_size,
                config=self._simulation
            )
            self.filament_solver = FilamentSolver(
                config=self._simulation,
                physics_solver=self.physics_solver
            )
        else:
            self.physics_solver = None
            self.filament_solver = None

    def print(self):
        # Check for preview mode in simulation config
        if hasattr(self._simulation, "preview_mode") and self._simulation.preview_mode:
            self.print_preview_mode()
            return
        filament_coordinates_count = len(self._instruction.filaments_coordinates)
        number_printed_filaments = 0
        number_printed_layers = 0
        initial_z_coordinate = 0.0
        # print(self._instruction.filaments_coordinates)
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
                printing_speed=printing_speed,
            )
            logger.info(
                f"Deposited step {number_printed_filaments}/{filament_coordinates_count} (~layer: {number_printed_layers}): "
                f"from {initial_coordinate} to {final_coordinate}, "
            )
        
        # Final sync for physics mode
        if self.physics_solver:
            logger.info("Finalizing physics simulation...")
            
            # Flush any remaining nodes in the filament solver
            # We can just force-freeze them or let them settle
            # Let's run a few seconds of simulation to let the last strand settle
            settle_time = 2.0 # seconds
            settle_steps = 20
            dt = settle_time / settle_steps
            
            for _ in range(settle_steps):
                frozen_segments = self.filament_solver.step(dt)
                for i, segment in enumerate(frozen_segments):
                    p1, p2, radius, mass, temp = segment
                    
                    # Ensure bounds for p1
                    shift_vec = self._ensure_physics_bounds(p1, radius)
                    if shift_vec is not None:
                        p1 += shift_vec
                        p2 += shift_vec
                        # Shift remaining segments in the list
                        for j in range(i+1, len(frozen_segments)):
                            s_p1, s_p2, s_r, s_m, s_t = frozen_segments[j]
                            frozen_segments[j] = (s_p1 + shift_vec, s_p2 + shift_vec, s_r, s_m, s_t)

                    # Ensure bounds for p2
                    shift_vec = self._ensure_physics_bounds(p2, radius)
                    if shift_vec is not None:
                        p1 += shift_vec
                        p2 += shift_vec
                        # Shift remaining segments in the list
                        for j in range(i+1, len(frozen_segments)):
                            s_p1, s_p2, s_r, s_m, s_t = frozen_segments[j]
                            frozen_segments[j] = (s_p1 + shift_vec, s_p2 + shift_vec, s_r, s_m, s_t)

                    self.physics_solver.add_segment(p1, p2, radius, mass, temp)
                self.physics_solver.step(dt)
            
            # Force rasterize any remaining nodes as segments
            nodes = self.filament_solver.nodes
            if len(nodes) > 1:
                for i in range(len(nodes) - 1):
                    n1 = nodes[i]
                    n2 = nodes[i+1]
                    
                    # Ensure bounds (this updates n1.position and n2.position in-place via shift_nodes)
                    self._ensure_physics_bounds(n1.position, n1.radius)
                    self._ensure_physics_bounds(n2.position, n2.radius)
                    
                    self.physics_solver.add_segment(
                        p1=n1.position,
                        p2=n2.position,
                        radius=n2.radius,
                        amount=n2.mass,
                        temperature=n2.temperature
                    )

            self.space = self.physics_solver.get_voxel_grid()
            self._filled_voxels_count = np.count_nonzero(self.space)
            logger.info(f"Physics simulation complete. Filled voxels: {self._filled_voxels_count}")

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

    def _ensure_physics_bounds(self, position, radius):
        """
        Ensures that the given position (and radius) fits within the physics grid.
        Expands the grid (and shifts coordinates) if necessary.
        """
        if not self.physics_solver:
            return

        voxel_size = self._simulation.voxel_size
        
        # Calculate required bounds (indices)
        # position is in Grid Coordinates (mm)
        
        # Min/Max extent of the new material
        p_min = position - radius
        p_max = position + radius
        
        # Required indices
        # We use floor for min and ceil for max to be safe
        req_x_min = int(math.floor(p_min[0] / voxel_size))
        req_y_min = int(math.floor(p_min[1] / voxel_size))
        req_z_min = int(math.floor(p_min[2] / voxel_size))
        
        req_x_max = int(math.ceil(p_max[0] / voxel_size))
        req_y_max = int(math.ceil(p_max[1] / voxel_size))
        req_z_max = int(math.ceil(p_max[2] / voxel_size))
        
        # Current shape
        sx, sy, sz = self.physics_solver.shape
        
        # Check for expansion
        expand_neg_x = max(0, -req_x_min)
        expand_neg_y = max(0, -req_y_min)
        expand_neg_z = max(0, -req_z_min)
        
        expand_pos_x = max(0, req_x_max - sx)
        expand_pos_y = max(0, req_y_max - sy)
        expand_pos_z = max(0, req_z_max - sz)
        
        # SAFETY CHECK: Prevent massive expansions due to flying nodes
        max_expansion = 2000 # 2000 voxels = 200mm
        if (expand_neg_x > max_expansion or expand_neg_y > max_expansion or expand_neg_z > max_expansion or
            expand_pos_x > max_expansion or expand_pos_y > max_expansion or expand_pos_z > max_expansion):
            logger.error(f"Requested grid expansion too large! Neg: ({expand_neg_x}, {expand_neg_y}, {expand_neg_z}), Pos: ({expand_pos_x}, {expand_pos_y}, {expand_pos_z}). Ignoring expansion request.")
            return None

        if (expand_neg_x == 0 and expand_neg_y == 0 and expand_neg_z == 0 and
            expand_pos_x == 0 and expand_pos_y == 0 and expand_pos_z == 0):
            return
            
        # Calculate new shape
        new_sx = sx + expand_neg_x + expand_pos_x
        new_sy = sy + expand_neg_y + expand_pos_y
        new_sz = sz + expand_neg_z + expand_pos_z
        
        new_shape = (new_sx, new_sy, new_sz)
        
        # Offset for old data in new grid
        offset = (expand_neg_x, expand_neg_y, expand_neg_z)
        
        logger.info(f"Expanding Physics Grid: {self.physics_solver.shape} -> {new_shape}, Offset: {offset}")
        
        # Expand Physics Solver
        self.physics_solver.expand(new_shape, offset)
        
        # Expand VoxelSpace.space (int8 grid)
        # We need to do the same expansion
        new_space = np.zeros(new_shape, dtype=np.int8)
        new_space[offset[0]:offset[0]+sx, offset[1]:offset[1]+sy, offset[2]:offset[2]+sz] = self.space
        self.space = new_space
        
        # If we expanded negatively, we shifted the origin.
        # We must update:
        # 1. filament_translations (so future G-code maps to the new coordinate system)
        # 2. FilamentSolver nodes (so existing nodes map to the new coordinate system)
        
        if expand_neg_x > 0 or expand_neg_y > 0 or expand_neg_z > 0:
            shift_vec = np.array([
                expand_neg_x * voxel_size,
                expand_neg_y * voxel_size,
                expand_neg_z * voxel_size
            ])
            
            # Update translations
            self.filament_translations["x"] += shift_vec[0]
            self.filament_translations["y"] += shift_vec[1]
            
            if "z" not in self.filament_translations:
                self.filament_translations["z"] = 0.0
            self.filament_translations["z"] += shift_vec[2]
            
            # Shift FilamentSolver nodes
            self.filament_solver.shift_nodes(shift_vec)
            
            return shift_vec
            
        return None

    def _find_initial_and_final_filament_coordinates(self, filament_coordinates):
        z_offset = self.filament_translations.get("z", 0.0)
        initial = [
            filament_coordinates[0][0] + self.filament_translations["x"],
            filament_coordinates[0][1] + self.filament_translations["y"],
            filament_coordinates[0][2] + z_offset,
        ]
        final = (
            filament_coordinates[1][0] + self.filament_translations["x"],
            filament_coordinates[1][1] + self.filament_translations["y"],
            filament_coordinates[1][2] + z_offset,
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
        printing_speed=None,
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

            if self.physics_solver:
                # Physics Mode
                
                # 1. Ensure Grid Size
                injection_radius = self._printer.nozzle_diameter / 2.0
                # This expands self.space if needed
                sphere.deform_voxel_space_for_big_spheres(self, injection_radius)
                
                # Sync solver shape if needed
                if self.space.shape != self.physics_solver.shape:
                    self.physics_solver.expand(self.space.shape, (0, 0, 0))
                
                # 2. Add Material to Filament Solver (Lagrangian)
                dt = step_size / printing_speed if printing_speed and printing_speed > 0 else 0.0
                
                # Use a default extrusion temp if not specified (e.g. 220C)
                extrusion_temp = 220.0 
                
                # Add node to the active chain
                # We use the nozzle velocity as the initial velocity of the material?
                # Or just zero relative to nozzle?
                # Let's assume it comes out at the speed of the print head (roughly)
                # or we can calculate E-speed. For now, just use direction * printing_speed
                
                initial_velocity = [d * printing_speed for d in direction_vector]
                
                self.filament_solver.add_node(
                    position=centre_coordinates,
                    velocity=initial_velocity,
                    mass=sphere_volume,
                    temperature=extrusion_temp,
                    radius=injection_radius
                )
                
                # 3. Step Filament Physics
                # This returns segments that have collided/frozen and need to be rasterized
                frozen_segments = []
                if dt > 0:
                    # We might want to sub-step if dt is large
                    frozen_segments = self.filament_solver.step(dt)
                
                # 4. Rasterize Frozen Segments into Grid (Eulerian)
                for i, segment in enumerate(frozen_segments):
                    p1, p2, radius, mass, temp = segment
                    
                    # Ensure bounds for p1
                    shift_vec = self._ensure_physics_bounds(p1, radius)
                    if shift_vec is not None:
                        p1 += shift_vec
                        p2 += shift_vec
                        # Shift remaining segments in the list
                        for j in range(i+1, len(frozen_segments)):
                            s_p1, s_p2, s_r, s_m, s_t = frozen_segments[j]
                            frozen_segments[j] = (s_p1 + shift_vec, s_p2 + shift_vec, s_r, s_m, s_t)

                    # Ensure bounds for p2
                    shift_vec = self._ensure_physics_bounds(p2, radius)
                    if shift_vec is not None:
                        p1 += shift_vec
                        p2 += shift_vec
                        # Shift remaining segments in the list
                        for j in range(i+1, len(frozen_segments)):
                            s_p1, s_p2, s_r, s_m, s_t = frozen_segments[j]
                            frozen_segments[j] = (s_p1 + shift_vec, s_p2 + shift_vec, s_r, s_m, s_t)

                    self.physics_solver.add_segment(p1, p2, radius, mass, temp)
                
                # 5. Step Grid Physics (Cooling, Flow)
                if dt > 0:
                    self.physics_solver.step(dt)
                
            else:
                # Geometric Mode
                # logger.info(
                #     f"Depositing filament: step = {step_n + 1}/{number_simulation_steps}"
                # )

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
