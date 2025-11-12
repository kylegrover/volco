"""
Material properties for common 3D printing filaments.

This module defines temperature-dependent material properties such as:
- Glass transition temperature (Tg)
- Melting/fusion temperature
- Viscosity as a function of temperature
- Thermal conductivity and cooling rates
"""

import math


class MaterialProperties:
    """Material properties for 3D printing filaments"""
    
    # PLA Material Constants (default)
    PLA_EXTRUSION_TEMP = 200.0  # °C - typical extrusion temperature
    PLA_GLASS_TRANSITION = 60.0  # °C - glass transition temperature (Tg)
    PLA_FUSION_TEMP = 150.0  # °C - minimum temperature for layer fusion
    PLA_AMBIENT_TEMP = 25.0  # °C - room temperature
    
    # ABS Material Constants
    ABS_EXTRUSION_TEMP = 230.0  # °C
    ABS_GLASS_TRANSITION = 105.0  # °C
    ABS_FUSION_TEMP = 180.0  # °C
    ABS_AMBIENT_TEMP = 25.0  # °C
    
    # PETG Material Constants
    PETG_EXTRUSION_TEMP = 230.0  # °C
    PETG_GLASS_TRANSITION = 80.0  # °C
    PETG_FUSION_TEMP = 160.0  # °C
    PETG_AMBIENT_TEMP = 25.0  # °C
    
    # Cooling constants (empirical, can be tuned)
    PLA_COOLING_RATE = 0.15  # 1/s - cooling rate constant
    ABS_COOLING_RATE = 0.10  # 1/s - ABS cools slower than PLA
    PETG_COOLING_RATE = 0.12  # 1/s
    
    # Material type enum
    MATERIALS = {
        'PLA': {
            'extrusion_temp': PLA_EXTRUSION_TEMP,
            'glass_transition': PLA_GLASS_TRANSITION,
            'fusion_temp': PLA_FUSION_TEMP,
            'ambient_temp': PLA_AMBIENT_TEMP,
            'cooling_rate': PLA_COOLING_RATE,
        },
        'ABS': {
            'extrusion_temp': ABS_EXTRUSION_TEMP,
            'glass_transition': ABS_GLASS_TRANSITION,
            'fusion_temp': ABS_FUSION_TEMP,
            'ambient_temp': ABS_AMBIENT_TEMP,
            'cooling_rate': ABS_COOLING_RATE,
        },
        'PETG': {
            'extrusion_temp': PETG_EXTRUSION_TEMP,
            'glass_transition': PETG_GLASS_TRANSITION,
            'fusion_temp': PETG_FUSION_TEMP,
            'ambient_temp': PETG_AMBIENT_TEMP,
            'cooling_rate': PETG_COOLING_RATE,
        }
    }
    
    def __init__(self, material_type='PLA'):
        """
        Initialize material properties for a specific material type.
        
        Parameters:
        -----------
        material_type : str
            Type of material ('PLA', 'ABS', 'PETG')
        """
        if material_type not in self.MATERIALS:
            raise ValueError(f"Unknown material type: {material_type}. Use one of {list(self.MATERIALS.keys())}")
        
        self.material_type = material_type
        props = self.MATERIALS[material_type]
        
        self.extrusion_temp = props['extrusion_temp']
        self.glass_transition = props['glass_transition']
        self.fusion_temp = props['fusion_temp']
        self.ambient_temp = props['ambient_temp']
        self.cooling_rate = props['cooling_rate']
    
    def get_viscosity(self, temperature):
        """
        Calculate material viscosity as a function of temperature.
        
        Parameters:
        -----------
        temperature : float
            Current temperature in °C
            
        Returns:
        --------
        float
            Viscosity in Pa·s
            
        Notes:
        ------
        This is a simplified model. Real viscosity curves are more complex.
        Below glass transition: effectively infinite (solid)
        Above extrusion temp: low viscosity (flows easily)
        In between: exponential transition
        """
        if temperature < self.glass_transition:
            # Below glass transition - effectively solid
            return float('inf')
        elif temperature < self.fusion_temp:
            # Transitional regime - rapid viscosity decrease
            # Using exponential model: η = η₀ * exp(-k * (T - Tg))
            delta_T = temperature - self.glass_transition
            transition_range = self.fusion_temp - self.glass_transition
            return 10000 * math.exp(-5.0 * delta_T / transition_range)
        else:
            # Molten regime - still temperature dependent but lower viscosity
            delta_T = temperature - self.fusion_temp
            melt_range = self.extrusion_temp - self.fusion_temp
            if melt_range > 0:
                return 100 * math.exp(-2.0 * delta_T / melt_range)
            else:
                return 100
    
    def is_solid(self, temperature):
        """Check if material is solid at given temperature"""
        return temperature < self.glass_transition
    
    def can_fuse(self, temperature):
        """Check if material is hot enough to fuse with other material"""
        return temperature >= self.fusion_temp
    
    def is_printable(self, temperature):
        """Check if material is in printable temperature range"""
        return temperature >= self.glass_transition and temperature <= self.extrusion_temp + 20
