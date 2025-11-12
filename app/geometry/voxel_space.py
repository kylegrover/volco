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
            self._hybrid_state = HybridMaterialState(
                voxel_space=self,
                simulation_config=self._simulation,
                printer_config=self._printer,
                material_type=self._simulation.material_type,
                enable_thermal=self._simulation.enable_thermal_simulation,
                enable_droop=self._simulation.enable_droop_simulation
            )
            logger.info("Hybrid material state initialized for physics simulation")

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
                printing_speed=printing_speed,
            )

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
                def sphere_depositor(center, volume):
                    sphere = Sphere(
                        centre_coordinates=center,
                        voxel_size=self._simulation.voxel_size,
                    )
                    self.space = sphere.deposit_sphere(
                        voxel_space=self.space,
                        nozzle_height=nozzle_height,
                        sphere_volume=volume,
                        voxel_space_target_volume=volume_target,
                        solver_tolerance=self._simulation.solver_tolerance,
                        radius_increment=self._simulation.radius_increment,
                    )
                
                # Deposit through hybrid state (applies physics)
                self._hybrid_state.deposit_filament_segment(
                    start=starting_point,
                    end=end_point,
                    volume=sphere_volume,
                    step_number=step_n,
                    printing_speed=printing_speed,
                    sphere_depositor=sphere_depositor
                )
                
                # Update total volume from actual voxel space
                total_deposited_volume = GeometryMath.calculate_filled_volume(
                    self.space, self._simulation.voxel_size
                )
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
