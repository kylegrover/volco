"""
Gravity and droop physics for unsupported filament.

This module calculates how unsupported filament segments sag under gravity
based on material viscosity, temperature, and unsupported length.
"""

import math


class GravityModel:
    """Models gravity effects on hot, unsupported filament"""
    
    # Physical constants
    GRAVITY = 9.81  # m/s² (standard gravity)
    
    # Material densities (kg/m³)
    PLA_DENSITY = 1250.0  # kg/m³
    ABS_DENSITY = 1050.0  # kg/m³
    PETG_DENSITY = 1270.0  # kg/m³
    
    MATERIAL_DENSITIES = {
        'PLA': PLA_DENSITY,
        'ABS': ABS_DENSITY,
        'PETG': PETG_DENSITY,
    }
    
    # Time constants for droop (empirical values, can be tuned)
    DROOP_TIME_CONSTANT = 0.25  # seconds - how quickly droop approaches maximum
    
    def __init__(self, material_type='PLA'):
        """
        Initialize gravity model for a specific material.
        
        Parameters:
        -----------
        material_type : str
            Type of material ('PLA', 'ABS', 'PETG')
        """
        self.material_type = material_type
        self.density = self.MATERIAL_DENSITIES.get(material_type, self.PLA_DENSITY)
    
    def calculate_droop(self, unsupported_length, viscosity, time_elapsed, 
                       filament_diameter=0.4):
        """
        Calculate droop (sag) for unsupported filament segment.
        
        Uses simplified catenary model for hanging material.
        Droop depends on:
        - Unsupported length (longer = more droop)
        - Material viscosity (lower = more droop)
        - Time elapsed (more time = more droop, asymptotically)
        - Filament cross-sectional area
        
        Parameters:
        -----------
        unsupported_length : float
            Length of unsupported segment in mm
        viscosity : float
            Material viscosity in Pa·s
        time_elapsed : float
            Time since extrusion in seconds
        filament_diameter : float
            Diameter of extruded filament in mm
            
        Returns:
        --------
        float
            Vertical droop amount in mm (always positive)
            
        Notes:
        ------
        Simplified model based on balance of gravitational force and viscous resistance.
        For a hanging cable of uniform material:
        max_droop ≈ (ρ * g * L²) / (8 * η)
        
        Where:
        - ρ = density (kg/m³)
        - g = gravity (m/s²)
        - L = unsupported length (m)
        - η = viscosity (Pa·s)
        """
        if unsupported_length <= 0 or viscosity == float('inf'):
            return 0.0
        
        # Convert to meters for calculation
        length_m = unsupported_length / 1000.0
        diameter_m = filament_diameter / 1000.0
        
        # Cross-sectional area (circular)
        area_m2 = math.pi * (diameter_m / 2.0) ** 2
        
        # Mass per unit length (kg/m)
        linear_density = self.density * area_m2
        
        # Empirical droop formula calibrated to real FDM bridge behavior
        # Based on observations: ~0.75mm sag for 40mm PLA bridge at 193°C
        # 
        # Simplified model: droop ∝ (weight × span²) / (viscous_resistance)
        # where weight = ρ × g × area × length
        # and resistance = η (viscosity)
        
        # Calculate droop (meters)
        # Formula: droop = (ρ * g * area * L²) / (C * η)
        # where C is a dimensionless constant
        max_droop_m = (self.density * self.GRAVITY * area_m2 * length_m ** 2) / (8.0 * viscosity)
        
        # Empirical calibration: 40mm PLA bridge should sag ~0.75mm
        # Scale to achieve realistic values
        max_droop_m *= 800.0
        
        # Apply time-dependent approach to maximum droop
        # Droop increases asymptotically: d(t) = d_max * (1 - exp(-t/τ))
        time_factor = 1.0 - math.exp(-time_elapsed / self.DROOP_TIME_CONSTANT)
        current_droop_m = max_droop_m * time_factor
        
        # Convert back to mm
        current_droop_mm = current_droop_m * 1000.0
        
        return current_droop_mm
    
    def calculate_max_bridge_length(self, viscosity, max_acceptable_droop=0.1,
                                    filament_diameter=0.4):
        """
        Calculate maximum bridgeable length for given droop tolerance.
        
        Parameters:
        -----------
        viscosity : float
            Material viscosity in Pa·s
        max_acceptable_droop : float
            Maximum acceptable droop in mm
        filament_diameter : float
            Diameter of extruded filament in mm
            
        Returns:
        --------
        float
            Maximum bridge length in mm
        """
        if viscosity == float('inf'):
            return float('inf')
        
        # Convert to meters
        max_droop_m = max_acceptable_droop / 1000.0
        diameter_m = filament_diameter / 1000.0
        area_m2 = math.pi * (diameter_m / 2.0) ** 2
        linear_density = self.density * area_m2
        
        # Solve for L: L = sqrt(8 * viscosity * area * droop / (density * area * g))
        length_m = math.sqrt(
            (8.0 * viscosity * area_m2 * max_droop_m) / 
            (linear_density * self.GRAVITY)
        )
        
        return length_m * 1000.0  # Convert to mm
    
    def get_material_weight(self, volume_mm3):
        """
        Calculate weight of material given volume.
        
        Parameters:
        -----------
        volume_mm3 : float
            Volume in mm³
            
        Returns:
        --------
        float
            Weight in grams
        """
        # Convert mm³ to m³
        volume_m3 = volume_mm3 * 1e-9
        
        # Weight in kg
        weight_kg = self.density * volume_m3
        
        # Return in grams
        return weight_kg * 1000.0
