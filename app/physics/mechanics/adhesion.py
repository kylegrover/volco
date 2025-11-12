"""
Adhesion and fusion model for layer bonding.

This module determines whether extruded filament will properly fuse
with existing material based on temperature, contact area, and time.
"""

import math


class AdhesionModel:
    """Models material adhesion and fusion between layers"""
    
    # Fusion quality thresholds
    EXCELLENT_FUSION_TEMP_RATIO = 0.9  # T > 0.9 * T_fusion
    GOOD_FUSION_TEMP_RATIO = 0.7       # 0.7 < T/T_fusion < 0.9
    POOR_FUSION_TEMP_RATIO = 0.5       # 0.5 < T/T_fusion < 0.7
    # Below 0.5 * T_fusion: no fusion
    
    # Contact time requirements (seconds)
    MIN_CONTACT_TIME = 0.1  # Minimum time for any fusion
    OPTIMAL_CONTACT_TIME = 1.0  # Time for optimal fusion
    
    def __init__(self, material_properties):
        """
        Initialize adhesion model with material properties.
        
        Parameters:
        -----------
        material_properties : MaterialProperties
            Material properties object containing fusion temperature etc.
        """
        self.material_props = material_properties
    
    def check_fusion_potential(self, temp1, temp2, contact_time=None):
        """
        Check if two material segments can fuse together.
        
        Parameters:
        -----------
        temp1 : float
            Temperature of first material in °C
        temp2 : float
            Temperature of second material in °C
        contact_time : float, optional
            Time materials are in contact in seconds
            
        Returns:
        --------
        dict
            Dictionary with keys:
            - 'can_fuse': bool - whether fusion is possible
            - 'fusion_quality': str - 'excellent', 'good', 'poor', or 'none'
            - 'fusion_strength': float - 0.0 to 1.0
        """
        fusion_temp = self.material_props.fusion_temp
        
        # Use the lower of the two temperatures (limiting factor)
        effective_temp = min(temp1, temp2)
        
        # Calculate temperature ratio
        temp_ratio = effective_temp / fusion_temp
        
        # Determine fusion quality based on temperature
        if temp_ratio < self.POOR_FUSION_TEMP_RATIO:
            fusion_quality = 'none'
            fusion_strength = 0.0
            can_fuse = False
        elif temp_ratio < self.GOOD_FUSION_TEMP_RATIO:
            fusion_quality = 'poor'
            fusion_strength = (temp_ratio - self.POOR_FUSION_TEMP_RATIO) / \
                            (self.GOOD_FUSION_TEMP_RATIO - self.POOR_FUSION_TEMP_RATIO) * 0.5
            can_fuse = True
        elif temp_ratio < self.EXCELLENT_FUSION_TEMP_RATIO:
            fusion_quality = 'good'
            base_strength = 0.5 + (temp_ratio - self.GOOD_FUSION_TEMP_RATIO) / \
                           (self.EXCELLENT_FUSION_TEMP_RATIO - self.GOOD_FUSION_TEMP_RATIO) * 0.3
            fusion_strength = base_strength
            can_fuse = True
        else:
            fusion_quality = 'excellent'
            fusion_strength = 0.8 + min(0.2, (temp_ratio - self.EXCELLENT_FUSION_TEMP_RATIO) * 0.5)
            can_fuse = True
        
        # Apply contact time modifier if provided
        if contact_time is not None and can_fuse:
            if contact_time < self.MIN_CONTACT_TIME:
                # Insufficient contact time - no fusion
                can_fuse = False
                fusion_quality = 'none'
                fusion_strength = 0.0
            elif contact_time < self.OPTIMAL_CONTACT_TIME:
                # Partial contact time - reduce strength
                time_factor = contact_time / self.OPTIMAL_CONTACT_TIME
                fusion_strength *= time_factor
        
        return {
            'can_fuse': can_fuse,
            'fusion_quality': fusion_quality,
            'fusion_strength': fusion_strength,
            'temp_ratio': temp_ratio
        }
    
    def estimate_bond_strength(self, temp1, temp2, contact_area_mm2, 
                              contact_time=1.0, pressure_mpa=0.0):
        """
        Estimate bond strength between two material segments.
        
        Parameters:
        -----------
        temp1 : float
            Temperature of first material in °C
        temp2 : float
            Temperature of second material in °C
        contact_area_mm2 : float
            Contact area in mm²
        contact_time : float
            Time materials are in contact in seconds
        pressure_mpa : float
            Applied pressure in MPa (0 for simple contact)
            
        Returns:
        --------
        float
            Estimated bond strength in Newtons
        """
        fusion_info = self.check_fusion_potential(temp1, temp2, contact_time)
        
        if not fusion_info['can_fuse']:
            return 0.0
        
        # Base bond strength per unit area (MPa) for perfect fusion
        # This is a simplified model - real values depend on many factors
        base_strength_mpa = 30.0  # Typical for PLA at optimal conditions
        
        # Apply fusion quality factor
        strength_factor = fusion_info['fusion_strength']
        
        # Apply pressure bonus (pressure helps fusion)
        pressure_factor = 1.0 + min(0.5, pressure_mpa / 10.0)
        
        # Calculate total strength
        strength_mpa = base_strength_mpa * strength_factor * pressure_factor
        
        # Convert to force (N = MPa * mm²)
        bond_strength_n = strength_mpa * contact_area_mm2
        
        return bond_strength_n
    
    def check_interlayer_adhesion(self, new_segment_temp, substrate_temp,
                                  layer_time, nozzle_force=0.0):
        """
        Check adhesion between a new segment and the substrate below it.
        
        Parameters:
        -----------
        new_segment_temp : float
            Temperature of newly deposited material in °C
        substrate_temp : float
            Temperature of substrate/previous layer in °C
        layer_time : float
            Time since previous layer was deposited in seconds
        nozzle_force : float
            Force applied by nozzle in Newtons (helps adhesion)
            
        Returns:
        --------
        dict
            Adhesion information including quality and recommendations
        """
        fusion_info = self.check_fusion_potential(new_segment_temp, substrate_temp)
        
        result = {
            'fusion_info': fusion_info,
            'substrate_too_cold': substrate_temp < self.material_props.glass_transition,
            'new_material_too_cold': new_segment_temp < self.material_props.fusion_temp,
            'layer_time_ok': layer_time < 60.0,  # Less than 1 minute is good
            'recommendations': []
        }
        
        # Generate recommendations
        if result['substrate_too_cold']:
            result['recommendations'].append(
                f"Substrate temperature ({substrate_temp:.1f}°C) is below glass transition. "
                "Consider heated bed or shorter layer times."
            )
        
        if result['new_material_too_cold']:
            result['recommendations'].append(
                f"New material temperature ({new_segment_temp:.1f}°C) is below fusion temp. "
                "May indicate nozzle temperature too low or excessive cooling."
            )
        
        if not result['layer_time_ok']:
            result['recommendations'].append(
                f"Long layer time ({layer_time:.1f}s) allows substrate to cool too much. "
                "Consider printing multiple parts simultaneously or using a heated chamber."
            )
        
        if fusion_info['fusion_quality'] in ['poor', 'none']:
            result['recommendations'].append(
                "Poor interlayer adhesion detected. Check nozzle temperature, "
                "cooling fan settings, and layer times."
            )
        
        return result
