import logging
import numpy as np
import math

from app.configs.simulation import Simulation
from app.configs.printer import Printer
from app.geometry.sphere import Sphere
from app.instructions.instruction import Instruction
from app.geometry.geometry_math import GeometryMath
from app.physics.volume import Volume
from app.physics.material_state import HybridMaterialState


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
        
        # Initialize hybrid material state if thermal/droop simulation enabled
        self._use_hybrid_simulation = (
            self._simulation.enable_thermal_simulation or 
            self._simulation.enable_droop_simulation
        )
        self._hybrid_state = None  # Will be initialized in print()

    def initialize_space(self):
        self.space = np.zeros(
            (self.dimensions["x"], self.dimensions["y"], self.dimensions["z"]),
            dtype=np.int8,
        )

    def print(self):
        number_printed_filaments = 0
        number_printed_layers = 0

        initial_z_coordinate = 0.0
        
        # Initialize hybrid state if physics simulation enabled
        if self._use_hybrid_simulation:
            print(f"\n[DEBUG] Initializing HybridMaterialState: thermal={self._simulation.enable_thermal_simulation}, droop={self._simulation.enable_droop_simulation}")
            self._hybrid_state = HybridMaterialState(
                voxel_space=self,
                simulation_config=self._simulation,
                printer_config=self._printer,
                material_type=self._simulation.material_type,
                enable_thermal=self._simulation.enable_thermal_simulation,
                enable_droop=self._simulation.enable_droop_simulation
            )
            logger.info("Hybrid material state initialized for physics simulation")
        else:
            print(f"\n[DEBUG] NOT using hybrid simulation: thermal={self._simulation.enable_thermal_simulation}, droop={self._simulation.enable_droop_simulation}")

        print(self._instruction.filaments_coordinates)

        for fil_idx, filament_coordinates in enumerate(self._instruction.filaments_coordinates):
            # DEBUG: Log raw filament coordinates for bridge
            if fil_idx == 10:
                print(f"\n[DEBUG RAW FILAMENT {fil_idx}] filament_coordinates[0]: {filament_coordinates[0]}")
                print(f"[DEBUG RAW FILAMENT {fil_idx}] filament_coordinates[1]: {filament_coordinates[1]}")
                print(f"[DEBUG RAW FILAMENT {fil_idx}] filament_translations: {self.filament_translations}")
            
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
            
            # Log bridge filament and segment count
            if fil_idx >= 340 and fil_idx <= 343:
                logger.info(f"Processing filament {fil_idx}: start=({initial_coordinate[0]:.2f},{initial_coordinate[1]:.2f},{initial_coordinate[2]:.2f}) end=({final_coordinate[0]:.2f},{final_coordinate[1]:.2f},{final_coordinate[2]:.2f}) vol={volume:.6f}")
                if self._hybrid_state:
                    logger.info(f"  Segments in history BEFORE processing: {len(self._hybrid_state.segment_history)}")
            
            # DEBUG: Log bridge filament coordinates (fil 10 is the 40mm bridge - 11th filament, 0-indexed)
            if fil_idx == 10:
                print(f"\n[DEBUG FILAMENT {fil_idx}] initial_coordinate: {initial_coordinate}")
                print(f"[DEBUG FILAMENT {fil_idx}] final_coordinate: {final_coordinate}")
                print(f"[DEBUG FILAMENT {fil_idx}] filament_length: {filament_length:.2f}mm")
                print(f"[DEBUG FILAMENT {fil_idx}] number_simulation_steps: {number_simulation_steps}")
                print(f"[DEBUG FILAMENT {fil_idx}] step_size: {step_size:.4f}mm\n")

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
            
            # Log segment count after processing bridge
            if fil_idx >= 340 and fil_idx <= 343:
                if self._hybrid_state:
                    logger.info(f"  Segments in history AFTER processing: {len(self._hybrid_state.segment_history)}")

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

        step_size = filament_length / number_simulation_steps

        return number_simulation_steps, step_size

    def _deposit_filament(
        self,
        number_simulation_steps,
        step_size,
        direction_vector,
        filament_initial_coordinates,
        volumes,
        printing_speed=50.0,
    ):
        total_deposited_volume = GeometryMath.calculate_filled_volume(
            self.space, self._simulation.voxel_size
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

            logger.info(
                f"Depositing filament: step = {step_n + 1}/{number_simulation_steps}"
            )
            
            # Use hybrid material state if physics simulation enabled
            if self._use_hybrid_simulation and self._hybrid_state is not None:
                # Calculate end point for this step
                end_displacement = (step_n + 1) * step_size
                end_point = [
                    pi_i + dir_vec_i * end_displacement
                    for (pi_i, dir_vec_i) in zip(
                        filament_initial_coordinates, direction_vector
                    )
                ]
                
                # Create sphere depositor function for hybrid state
                # IMPORTANT: Use volume_target from outer scope to match non-physics behavior
                def sphere_depositor(center, volume):
                    # Center comes from segment - this is at nozzle height
                    # Need to apply sphere_z_offset to match non-physics behavior
                    # BUT: nozzle_height should be BEFORE the offset
                    actual_nozzle_height = center[2]
                    adjusted_center = [center[0], center[1], center[2] - self._simulation.sphere_z_offset]
                    
                    sphere = Sphere(
                        centre_coordinates=adjusted_center,
                        voxel_size=self._simulation.voxel_size,
                    )
                    self.space = sphere.deposit_sphere(
                        voxel_space=self.space,
                        nozzle_height=actual_nozzle_height,
                        sphere_volume=volume,
                        voxel_space_target_volume=volume_target,
                        solver_tolerance=self._simulation.solver_tolerance,
                        radius_increment=self._simulation.radius_increment,
                    )
                
                # DEBUG: Log coordinates for bridge segments (Z ≈ 2.2)
                if abs(starting_point[2] - 2.2) < 0.01 and step_n == 100:
                    print(f"\n[DEBUG SEGMENT] step={step_n}")
                    print(f"[DEBUG SEGMENT] starting_point: {starting_point}")
                    print(f"[DEBUG SEGMENT] end_point: {end_point}\n")
                
                # Deposit through hybrid state (applies physics)
                self._hybrid_state.deposit_filament_segment(
                    start=starting_point,
                    end=end_point,
                    volume=sphere_volume,
                    step_number=step_n,
                    printing_speed=printing_speed,
                    sphere_depositor=sphere_depositor
                )
                
                # DON'T update total_deposited_volume inside loop - matches non-physics behavior
                # Non-physics path uses ONE initial volume + accumulated sphere volumes
                # total_deposited_volume stays constant throughout loop
            else:
                # Original VolCo behavior (no physics simulation)
                sphere = Sphere(
                    centre_coordinates=centre_coordinates,
                    voxel_size=self._simulation.voxel_size,
                )

                self.space = sphere.deposit_sphere(
                    voxel_space=self.space,
                    nozzle_height=nozzle_height,
                    sphere_volume=sphere_volume,
                    voxel_space_target_volume=volume_target,
                    solver_tolerance=self._simulation.solver_tolerance,
                    radius_increment=self._simulation.radius_increment,
                )
    
    def get_physics_simulation_data(self):
        """
        Get physics simulation data if hybrid state was used.
        
        Returns:
        --------
        dict or None
            Dictionary with segment history and statistics, or None if not using physics simulation
        """
        if self._hybrid_state is None:
            return None
        
        return {
            'segments': self._hybrid_state.get_segment_history(),
            'statistics': self._hybrid_state.get_statistics(),
            'material_type': self._simulation.material_type,
            'thermal_enabled': self._simulation.enable_thermal_simulation,
            'droop_enabled': self._simulation.enable_droop_simulation,
        }
