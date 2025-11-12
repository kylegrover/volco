"""
Material state management for hybrid voxel-segment simulation.

This module bridges VolCo's voxel-based simulation with segment-level physics:
- FilamentSegment: Represents individual extruded segments with thermal/physical state
- HybridMaterialState: Manages active (hot) segments and their transition to solidified state
"""

import logging
import math
import numpy as np

from app.physics.thermal.cooling_model import CoolingModel
from app.physics.thermal.material_properties import MaterialProperties
from app.physics.mechanics.gravity import GravityModel
from app.physics.mechanics.adhesion import AdhesionModel
from app.geometry.geometry_math import GeometryMath


logger = logging.getLogger(__name__)


class FilamentSegment:
    """
    Represents a segment of extruded filament with physical and thermal state.
    
    A segment corresponds to one G-code move (from one point to another).
    It tracks temperature, position, deformation, and fusion state.
    """
    
    def __init__(self, start_pos, end_pos, volume, timestamp, 
                 initial_temp, material_props, nozzle_diameter=0.4):
        """
        Initialize a filament segment.
        
        Parameters:
        -----------
        start_pos : tuple/list
            Starting position (x, y, z) in mm
        end_pos : tuple/list
            Ending position (x, y, z) in mm
        volume : float
            Volume of extruded material in mm³
        timestamp : float
            Time when segment was extruded (in seconds from start)
        initial_temp : float
            Initial temperature in °C (typically extrusion temp)
        material_props : MaterialProperties
            Material properties object
        nozzle_diameter : float
            Nozzle diameter in mm (for estimating filament width)
        """
        self.start_pos = np.array(start_pos, dtype=float)
        self.end_pos = np.array(end_pos, dtype=float)
        self.volume = volume
        self.timestamp = timestamp
        self.nozzle_diameter = nozzle_diameter
        
        # Calculate segment properties
        displacement = self.end_pos - self.start_pos
        self.length = np.linalg.norm(displacement)
        self.direction = displacement / self.length if self.length > 0 else np.array([0, 0, 0])
        
        # Estimate filament width from volume and length
        if self.length > 0:
            # Assume roughly circular cross-section
            cross_section_area = volume / self.length
            self.width = math.sqrt(4 * cross_section_area / math.pi)
        else:
            self.width = nozzle_diameter
        
        # Thermal state
        self.temperature = initial_temp
        self.material_props = material_props
        
        # Physical state
        self.support_below = None  # Will be set by support detection
        self.unsupported_length = 0.0  # Length of segment without support
        self.droop_offset = 0.0  # Vertical displacement due to gravity (mm)
        self.deformed_center = None  # Modified center position after droop
        
        # Fusion state
        self.is_solidified = False
        self.is_fused = False
        self.fusion_quality = 'unknown'
        
        # For visualization/analysis
        self.segment_id = None
        self.layer_number = None
    
    def get_center_position(self):
        """Get the center position of the segment (accounting for droop)"""
        if self.deformed_center is not None:
            return self.deformed_center
        return (self.start_pos + self.end_pos) / 2.0
    
    def update_temperature(self, current_time, cooling_model):
        """
        Update segment temperature based on elapsed time.
        
        Parameters:
        -----------
        current_time : float
            Current simulation time in seconds
        cooling_model : CoolingModel
            Cooling model instance
        """
        time_elapsed = current_time - self.timestamp
        
        self.temperature = cooling_model.calculate_temperature(
            initial_temp=self.material_props.extrusion_temp,
            time_elapsed=time_elapsed,
            ambient_temp=self.material_props.ambient_temp,
            cooling_rate=self.material_props.cooling_rate
        )
        
        # Check if solidified
        if self.temperature < self.material_props.glass_transition:
            self.is_solidified = True
    
    def apply_droop(self, droop_amount):
        """
        Apply droop offset to segment position.
        
        Parameters:
        -----------
        droop_amount : float
            Vertical droop in mm (positive = down)
        """
        self.droop_offset = droop_amount
        
        # Update deformed center position
        center = (self.start_pos + self.end_pos) / 2.0
        self.deformed_center = center.copy()
        self.deformed_center[2] -= droop_amount  # Z-axis down
    
    def __repr__(self):
        return (f"FilamentSegment(start={self.start_pos}, end={self.end_pos}, "
                f"T={self.temperature:.1f}°C, droop={self.droop_offset:.3f}mm)")


class HybridMaterialState:
    """
    Manages transition between active (hot) and solidified material.
    
    This class bridges VolCo's voxel-based simulation with segment-level physics:
    1. Tracks recently extruded segments that are still hot
    2. Simulates thermal and mechanical behavior (cooling, droop, fusion)
    3. Deposits material into voxel space with physics-based modifications
    4. Maintains history for analysis and visualization
    """
    
    def __init__(self, voxel_space, simulation_config, printer_config, 
                 material_type='PLA', enable_thermal=True, enable_droop=True):
        """
        Initialize hybrid material state manager.
        
        Parameters:
        -----------
        voxel_space : VoxelSpace
            VolCo voxel space object
        simulation_config : Simulation
            Simulation configuration
        printer_config : Printer
            Printer configuration
        material_type : str
            Material type ('PLA', 'ABS', 'PETG')
        enable_thermal : bool
            Enable thermal simulation (cooling, fusion)
        enable_droop : bool
            Enable droop simulation (gravity effects)
        """
        self.voxel_space = voxel_space
        self.sim_config = simulation_config
        self.printer_config = printer_config
        self.material_type = material_type
        self.enable_thermal = enable_thermal
        self.enable_droop = enable_droop
        
        # Initialize physics models
        self.material_props = MaterialProperties(material_type)
        self.cooling_model = CoolingModel()
        self.gravity_model = GravityModel(material_type)
        self.adhesion_model = AdhesionModel(self.material_props)
        
        # Segment tracking
        self.active_segments = []  # Currently hot segments
        self.segment_history = []  # All segments (for analysis)
        self.current_time = 0.0  # Simulation time in seconds
        
        # Strand tracking - NEW: track continuous unsupported spans
        self.current_strand = []  # Current continuous unsupported strand
        self.last_anchor_point = None  # Last supported/solidified position
        self.strand_start_time = 0.0  # When current strand started
        
        # Statistics
        self.stats = {
            'total_segments': 0,
            'solidified_segments': 0,
            'drooped_segments': 0,
            'unsupported_segments': 0,
            'max_strand_length': 0.0,
        }
        
        logger.info(f"HybridMaterialState initialized: material={material_type}, "
                   f"thermal={enable_thermal}, droop={enable_droop}")
    
    def deposit_filament_segment(self, start, end, volume, step_number, 
                                 printing_speed, sphere_depositor):
        """
        Deposit a filament segment with physics simulation.
        
        This is the main integration point with VolCo's print loop.
        
        Parameters:
        -----------
        start : list/tuple
            Starting position (x, y, z) in mm
        end : list/tuple
            Ending position (x, y, z) in mm
        volume : float
            Volume to deposit in mm³
        step_number : int
            Step number in simulation (used as time proxy)
        printing_speed : float
            Printing speed in mm/s
        sphere_depositor : callable
            Function to deposit sphere into voxel space
            Signature: sphere_depositor(center, volume) -> modified_voxel_space
        """
        # Estimate time for this segment
        segment_length = GeometryMath.distance(start, end)
        segment_time = segment_length / printing_speed if printing_speed > 0 else 0.1
        self.current_time += segment_time
        
        # Create segment
        segment = FilamentSegment(
            start_pos=start,
            end_pos=end,
            volume=volume,
            timestamp=self.current_time,
            initial_temp=self.material_props.extrusion_temp,
            material_props=self.material_props,
            nozzle_diameter=self.printer_config.nozzle_diameter
        )
        segment.segment_id = self.stats['total_segments']
        
        # Update active segments first (cooling, solidification)
        if self.enable_thermal or self.enable_droop:
            self._update_active_segments(self.current_time)
        
        # Check for support
        if self.enable_droop:
            segment.support_below = self._check_support(segment)
            if not segment.support_below:
                self.stats['unsupported_segments'] += 1
        
        # NEW: Strand-based physics simulation
        if self.enable_droop:
            self._update_strand_physics(segment)
        
        # Thermal updates for this segment
        if self.enable_thermal:
            segment.update_temperature(self.current_time, self.cooling_model)
            self._check_fusion(segment)
        
        # Add to tracking
        self.active_segments.append(segment)
        self.segment_history.append(segment)
        self.stats['total_segments'] += 1
        
        # Deposit into voxel space
        center = segment.get_center_position()
        
        # Debug logging
        logger.debug(f"Segment {segment.segment_id}: start={segment.start_pos}, end={segment.end_pos}, center={center}, droop={segment.droop_offset:.3f}mm")
        
        sphere_depositor(center, volume)
        
        logger.debug(f"Deposited segment {segment.segment_id}: "
                    f"T={segment.temperature:.1f}°C, droop={segment.droop_offset:.3f}mm")
    
    def _check_support(self, segment):
        """
        Check if segment has material below it for support.
        
        Parameters:
        -----------
        segment : FilamentSegment
            Segment to check
            
        Returns:
        --------
        bool
            True if supported, False if unsupported
        """
        # Check center and endpoints
        positions_to_check = [
            segment.start_pos,
            segment.get_center_position(),
            segment.end_pos
        ]
        
        # Check a small distance below (half layer height)
        check_distance = self.sim_config.voxel_size * 2
        
        for pos in positions_to_check:
            check_pos = pos.copy()
            check_pos[2] -= check_distance  # Look down
            
            if check_pos[2] < 0:
                # On build plate
                return True
            
            # Convert to voxel indices
            voxel_idx = [
                GeometryMath.find_index(check_pos[0], self.sim_config.voxel_size),
                GeometryMath.find_index(check_pos[1], self.sim_config.voxel_size),
                GeometryMath.find_index(check_pos[2], self.sim_config.voxel_size),
            ]
            
            # Check if within bounds
            if (0 <= voxel_idx[0] < self.voxel_space.space.shape[0] and
                0 <= voxel_idx[1] < self.voxel_space.space.shape[1] and
                0 <= voxel_idx[2] < self.voxel_space.space.shape[2]):
                
                if self.voxel_space.space[tuple(voxel_idx)] > 0:
                    return True
        
        # If we checked multiple points and none had support, estimate unsupported length
        segment.unsupported_length = segment.length
        return False
    
    def _apply_droop_physics(self, segment):
        """
        Apply gravity-based droop to segment.
        
        Parameters:
        -----------
        segment : FilamentSegment
            Segment to apply droop to
        """
        if segment.support_below or segment.unsupported_length == 0:
            return
        
        # Get current viscosity based on temperature
        viscosity = self.material_props.get_viscosity(segment.temperature)
        
        # Calculate droop
        time_since_extrusion = self.current_time - segment.timestamp
        droop = self.gravity_model.calculate_droop(
            unsupported_length=segment.unsupported_length,
            viscosity=viscosity,
            time_elapsed=time_since_extrusion,
            filament_diameter=segment.width
        )
        
        segment.apply_droop(droop)
        
        if droop > 0.1:  # More than 0.1mm
            self.stats['drooped_segments'] += 1
    
    def _update_strand_physics(self, segment):
        """
        NEW: Strand-based physics simulation.
        
        Tracks continuous unsupported strands and applies physics to the entire
        span from the last anchor point (supported/solidified) to current nozzle.
        
        Parameters:
        -----------
        segment : FilamentSegment
            Newly deposited segment
        """
        # Determine if this segment is an anchor point
        is_anchor = False
        
        if segment.support_below:
            is_anchor = True
        elif self.enable_thermal and segment.is_solidified:
            is_anchor = True
        
        if is_anchor:
            # This segment is supported or solidified - it becomes an anchor
            if len(self.current_strand) > 0:
                # Apply physics to the completed strand before clearing
                self._apply_strand_droop(self.current_strand)
                
                # Calculate total strand length for stats
                total_length = sum(s.length for s in self.current_strand)
                self.stats['max_strand_length'] = max(
                    self.stats['max_strand_length'], 
                    total_length
                )
                
                logger.debug(f"Strand completed: {len(self.current_strand)} segments, "
                           f"{total_length:.2f}mm total length")
                
                # Clear strand
                self.current_strand = []
            
            # Set new anchor point
            self.last_anchor_point = segment.end_pos.copy()
            self.strand_start_time = self.current_time
            
        else:
            # Unsupported segment - add to current strand
            self.current_strand.append(segment)
            
            # Calculate unsupported length from last anchor
            if self.last_anchor_point is not None:
                # Distance from anchor to current segment end
                strand_length = np.linalg.norm(segment.end_pos - self.last_anchor_point)
            else:
                # No previous anchor (shouldn't happen, but handle gracefully)
                strand_length = segment.length
                self.last_anchor_point = segment.start_pos.copy()
            
            segment.unsupported_length = strand_length
            
            # Apply droop physics to entire current strand
            if len(self.current_strand) > 0:
                self._apply_strand_droop(self.current_strand)
    
    def _apply_strand_droop(self, strand_segments):
        """
        Apply droop physics to a continuous strand of segments.
        
        Parameters:
        -----------
        strand_segments : list[FilamentSegment]
            List of connected unsupported segments forming a strand
        """
        if not strand_segments:
            return
        
        # Calculate total strand properties
        total_length = sum(s.length for s in strand_segments)
        if total_length == 0:
            return
        
        # Use the most recent (hottest) segment's temperature for viscosity
        # In reality, we'd model temperature gradient along the strand
        latest_segment = strand_segments[-1]
        viscosity = self.material_props.get_viscosity(latest_segment.temperature)
        
        # Time since strand started forming
        time_elapsed = self.current_time - self.strand_start_time
        
        # Calculate maximum droop at the midpoint of the strand
        # (catenary has maximum sag at center)
        max_droop = self.gravity_model.calculate_droop(
            unsupported_length=total_length,
            viscosity=viscosity,
            time_elapsed=time_elapsed,
            filament_diameter=latest_segment.width
        )
        
        # Apply droop distribution along strand
        # Simple model: parabolic distribution with max at center
        cumulative_length = 0.0
        for i, segment in enumerate(strand_segments):
            cumulative_length += segment.length
            
            # Position along strand (0 to 1)
            position_ratio = cumulative_length / total_length
            
            # Parabolic droop profile: max at center (0.5)
            # droop_factor peaks at 1.0 when position_ratio = 0.5
            droop_factor = 1.0 - 4.0 * (position_ratio - 0.5) ** 2
            droop_factor = max(0.0, droop_factor)  # Clamp to non-negative
            
            # Apply proportional droop
            segment_droop = max_droop * droop_factor
            segment.apply_droop(segment_droop)
            segment.unsupported_length = total_length  # Update to reflect full strand
            
            if segment_droop > 0.1:
                self.stats['drooped_segments'] += 1
        
        logger.debug(f"Applied strand droop: length={total_length:.2f}mm, "
                   f"max_droop={max_droop:.3f}mm, segments={len(strand_segments)}")
    
    def _check_fusion(self, segment):
        """
        Check if segment can fuse with material below it.
        
        Parameters:
        -----------
        segment : FilamentSegment
            Segment to check
        """
        if not segment.support_below:
            segment.is_fused = False
            segment.fusion_quality = 'none'
            return
        
        # For now, simple model: check if temperature is above fusion temp
        if segment.temperature >= self.material_props.fusion_temp:
            segment.is_fused = True
            segment.fusion_quality = 'good'
        else:
            segment.is_fused = False
            segment.fusion_quality = 'poor'
    
    def _update_active_segments(self, current_time):
        """
        Update all active segments (cooling, physics).
        
        Parameters:
        -----------
        current_time : float
            Current simulation time
        """
        segments_to_remove = []
        
        for segment in self.active_segments:
            # Update temperature
            if self.enable_thermal:
                segment.update_temperature(current_time, self.cooling_model)
            
            # Check if solidified (below glass transition temp)
            if (self.enable_thermal and 
                segment.temperature < self.material_props.glass_transition and
                not segment.is_solidified):
                segment.is_solidified = True
                segments_to_remove.append(segment)
                self.stats['solidified_segments'] += 1
                
                # If this segment is in current strand, it becomes an anchor
                if segment in self.current_strand:
                    logger.debug(f"Segment {segment.segment_id} solidified in strand, "
                               f"becoming anchor at {segment.end_pos}")
                    # Note: The next deposited segment will trigger strand completion
        
        # Remove solidified segments from active list
        for segment in segments_to_remove:
            self.active_segments.remove(segment)
    
    def get_statistics(self):
        """Get simulation statistics"""
        return {
            **self.stats,
            'active_segments': len(self.active_segments),
            'total_time': self.current_time,
        }
    
    def get_segment_history(self):
        """Get all segments for analysis"""
        return self.segment_history
