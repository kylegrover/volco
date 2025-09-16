"""
VOLCO FEA Module

This module provides a simplified interface to the Finite Element Analysis (FEA)
functionality in VOLCO. It allows users to perform structural analysis on voxel models,
visualize results, and save/load analysis data.

Basic usage:
    from volco_fea import analyze_voxel_matrix, Surface
    
    # Define boundary conditions
    boundary_conditions = {
        'constraints': {
            Surface.MINUS_Z: "fix",  # Fix bottom surface
            Surface.PLUS_Z: [None, None, -0.1, None, None, None]  # Apply displacement on top
        }
    }
    
    # Run analysis
    results = analyze_voxel_matrix(
        voxel_matrix=voxel_matrix,
        voxel_size=voxel_size,
        boundary_conditions=boundary_conditions
    )
"""

# Re-export all FEA functionality from the main module (no duplication!)
from app.postprocessing.fea import *