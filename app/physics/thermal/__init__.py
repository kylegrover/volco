"""
Thermal physics module for simulating temperature-dependent behavior of extruded filament.

This module provides:
- Cooling models for extruded material
- Material properties (viscosity, glass transition temperature, etc.)
- Temperature tracking over time
"""

from app.physics.thermal.cooling_model import CoolingModel
from app.physics.thermal.material_properties import MaterialProperties

__all__ = ['CoolingModel', 'MaterialProperties']
