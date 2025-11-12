"""
Mechanics physics module for simulating physical behavior of extruded filament.

This module provides:
- Gravity and droop calculations
- Material adhesion and fusion detection
- Support detection
"""

from app.physics.mechanics.gravity import GravityModel
from app.physics.mechanics.adhesion import AdhesionModel
from app.physics.mechanics.droop_detection import DroopDetector

__all__ = ['GravityModel', 'AdhesionModel', 'DroopDetector']
