import logging
import math
import os
import numpy as np
import trimesh

from app.configs.simulation import Simulation
from app.geometry.voxel_space import VoxelSpace
from app.reporter.visualization import color_mesh, visualize_with_trimesh, visualize_with_plotly
from app.reporter.mesh import generate_mesh_from_voxels, export_mesh_to_stl, generate_and_export_mesh


logger = logging.getLogger(__name__)


class SimulationOutput:
    """
    Processes and visualizes the output of a 3D printing simulation.
    
    This class handles post-processing of simulation data, including:
    - Cropping the voxel space
    - Generating meshes from voxel data
    - Exporting meshes to STL files
    - Visualizing the results
    """
    def __init__(self, voxel_space: VoxelSpace, simulation: Simulation):
        self.voxel_space = voxel_space
        self._simulation = simulation
        self.cropped_voxel_space = None
        self.mesh = None
        self.crop_start = None
        self.world_origin = None

    def crop_voxel_space(self):
        """
        Snapshot whole cells intersecting half-open world-space crop intervals.
        """
        # Invalidate old results even if a new crop fails validation.
        self.cropped_voxel_space = None
        self.mesh = None
        self.crop_start = None
        self.world_origin = None
        crop_coordinates_for_axes = [
            self._simulation.x_crop,
            self._simulation.y_crop,
            self._simulation.z_crop,
        ]

        filament_translations = [
            self.voxel_space.filament_translations["x"],
            self.voxel_space.filament_translations["y"],
            0.0,
        ]

        axes_length = list(self.voxel_space.space.shape)

        indexes_to_crop = list()

        for (crop_coordinates, filament_translation, axis_length) in zip(
            crop_coordinates_for_axes, filament_translations, axes_length
        ):
            indexes_for_axis = self._crop_axis(
                crop_coordinates, filament_translation, axis_length
            )
            indexes_to_crop.append(indexes_for_axis)

        self.crop_start = np.array([bounds[0] for bounds in indexes_to_crop], dtype=int)
        self.world_origin = self.crop_start * self._simulation.voxel_size - np.asarray(filament_translations)
        # Copy only the crop, retaining snapshot semantics without a full-grid copy.
        self.cropped_voxel_space = self.voxel_space.space[
            tuple(slice(start, stop) for start, stop in indexes_to_crop)
        ].copy()

    def generate_mesh(self):
        """
        Generate the exposed voxel faces in input G-code/world coordinates.
        Returns the mesh object for further use.
        """
        if self.cropped_voxel_space is None:
            logger.warning("[SimulationOutput]: No cropped voxel space available. Run crop_voxel_space() first.")
            return None

        # Get voxel size
        voxel_size = self._simulation.voxel_size

        # Create a mesh from the voxel data using the mesh module
        mesh = generate_mesh_from_voxels(self.cropped_voxel_space, voxel_size, origin=self.world_origin)

        # Store the mesh for later use
        self.mesh = mesh
        return mesh

    def color_mesh(self, mesh=None, color_scheme='cyan_blue'):
        """
        Apply colors to the mesh based on height (z-value).
        Returns the colored mesh.
        
        This method uses the color_mesh function from the visualization submodule.
        """
        if mesh is None:
            if self.mesh is None:
                logger.warning("[SimulationOutput]: No mesh available. Generate mesh first.")
                return None
            mesh = self.mesh

        # Check if mesh is a Scene object (from box representation)
        if isinstance(mesh, trimesh.Scene):
            # For box representation, we can't easily color individual vertices
            # Return the original scene without coloring
            return mesh
            
        # For Trimesh objects (from marching cubes)
        if not hasattr(mesh, 'vertices') or len(mesh.vertices) == 0:
            logger.warning("[SimulationOutput]: Mesh has no vertices to color.")
            return mesh
            
        return color_mesh(mesh, color_scheme)

    def export_mesh_to_stl(self, mesh=None):
        """
        Export the mesh to an STL file.
        If mesh is not provided, uses the stored mesh.
        If file_path is not provided, uses the default path based on simulation settings.
        Uses the stl_ascii setting from simulation configuration to determine the STL format.
        """
        # If preview_mode, or if a streaming export is desired, bypass building a mesh
        result_path = self._get_result_folder_path()
        stl_file_name = self._simulation.simulation_name + ".stl"
        file_path = os.path.join(result_path, stl_file_name)

        # Use streaming exporter for all STL exports
        # This avoids creating a massive mesh object in memory
        logger.info("[SimulationOutput]: Using streaming exporter for STL.")
        # Use the vectorized streaming exporter which accepts voxel_space directly
        return generate_and_export_mesh(self.cropped_voxel_space, self._simulation.voxel_size, file_path,
                                        binary=not self._simulation.stl_ascii, origin=self.world_origin)

        # Non-preview: expect a mesh object to be provided or generated previously
        if mesh is None:
            if self.mesh is None:
                logger.warning("[SimulationOutput]: No mesh available. Generate mesh first.")
                return
            mesh = self.mesh

        return export_mesh_to_stl(mesh, file_path, ascii_format=self._simulation.stl_ascii)

    def visualize_mesh(self, mesh=None, visualizer='trimesh', color_scheme='cyan_blue'):
        """
        Create a 3D visualization of the mesh with coloring applied.
        If mesh is not provided, uses the stored mesh.
        
        Parameters:
        -----------
        mesh : trimesh.Trimesh
            The mesh to visualize
        visualizer : str
            The visualizer to use ('trimesh' or 'plotly')
        color_scheme : str
            The color scheme to use ('cyan_blue', 'viridis', or None for no coloring)
            
        Returns:
        --------
        trimesh.Scene or plotly.graph_objects.Figure
            A visualization object that can be displayed
        """
        if mesh is None:
            if self.mesh is None:
                logger.warning("[SimulationOutput]: No mesh available. Generate mesh first.")
                return None
            mesh = self.mesh
        
        # Apply coloring if requested
        if color_scheme is not None:
            mesh = self.color_mesh(mesh, color_scheme=color_scheme)
            
        if visualizer.lower() == 'plotly':
            return visualize_with_plotly(mesh)
        else:  # default to trimesh
            return visualize_with_trimesh(mesh)

    def _crop_axis(self, crop_coordinates, filament_translation, axis_length):
        """World interval [lower, upper); retain intersecting whole cells.

        Snap roundoff within 1e-9 cell units of an integer before floor/ceil.
        This avoids admitting an extra cell after cancelling world translations.
        """
        if len(crop_coordinates) != 2:
            raise ValueError('Crop must contain lower and upper bounds')
        bounds = []
        h = self._simulation.voxel_size
        for side, value in enumerate(crop_coordinates):
            if value == 'all':
                position = 0.0 if side == 0 else float(axis_length)
            else:
                position = (float(value) + filament_translation) / h
                if not math.isfinite(position):
                    raise ValueError('Crop bounds must be finite or all')
                nearest = round(position)
                if abs(position - nearest) <= 1e-9:
                    position = float(nearest)
            bounds.append(position)
        if bounds[0] >= bounds[1]:
            raise ValueError('Crop lower bound must be below upper bound')
        start = max(0, min(axis_length, math.floor(bounds[0])))
        stop = max(0, min(axis_length, math.ceil(bounds[1])))
        if start >= stop:
            raise ValueError('Crop does not intersect the voxel grid')
        return [start, stop]

    def _get_result_folder_path(self):
        folder_name = self._simulation.results_folder

        cwd = os.getcwd()
        path = os.path.join(cwd, folder_name)

        if not os.path.exists(path):
            os.mkdir(path)
        return path
