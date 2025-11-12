"""
Cooling model for extruded filament.

This module implements Newton's law of cooling and related thermal models
to simulate how extruded filament cools over time.
"""

import math


class CoolingModel:
    """Models filament cooling over time"""
    
    @staticmethod
    def calculate_temperature(initial_temp, time_elapsed, ambient_temp, cooling_rate):
        """
        Calculate temperature using Newton's law of cooling.
        
        Newton's law of cooling: T(t) = T_ambient + (T_initial - T_ambient) * e^(-k*t)
        
        Parameters:
        -----------
        initial_temp : float
            Initial temperature in °C (typically extrusion temperature)
        time_elapsed : float
            Time elapsed since extrusion in seconds
        ambient_temp : float
            Ambient temperature in °C (room temperature)
        cooling_rate : float
            Cooling rate constant k (1/s), depends on material and environment
            
        Returns:
        --------
        float
            Current temperature in °C
            
        Notes:
        ------
        Cooling rate depends on:
        - Material thermal properties
        - Part cooling fan speed
        - Ambient temperature
        - Filament diameter (surface area to volume ratio)
        
        Typical values for PLA with part cooling fan:
        - k ≈ 0.1 to 0.2 (1/s)
        """
        delta_T = initial_temp - ambient_temp
        current_temp = ambient_temp + delta_T * math.exp(-cooling_rate * time_elapsed)
        return current_temp
    
    @staticmethod
    def time_to_temperature(initial_temp, target_temp, ambient_temp, cooling_rate):
        """
        Calculate time required to cool from initial temperature to target temperature.
        
        Solving T(t) = T_ambient + (T_initial - T_ambient) * e^(-k*t) for t:
        t = -ln((T_target - T_ambient) / (T_initial - T_ambient)) / k
        
        Parameters:
        -----------
        initial_temp : float
            Initial temperature in °C
        target_temp : float
            Target temperature in °C
        ambient_temp : float
            Ambient temperature in °C
        cooling_rate : float
            Cooling rate constant k (1/s)
            
        Returns:
        --------
        float
            Time in seconds to reach target temperature
            Returns inf if target is below ambient or above initial temp
        """
        if target_temp <= ambient_temp or target_temp >= initial_temp:
            return float('inf')
        
        delta_T_initial = initial_temp - ambient_temp
        delta_T_target = target_temp - ambient_temp
        
        time = -math.log(delta_T_target / delta_T_initial) / cooling_rate
        return time
    
    @staticmethod
    def adjust_cooling_rate_for_fan(base_cooling_rate, fan_speed_percent):
        """
        Adjust cooling rate based on part cooling fan speed.
        
        Parameters:
        -----------
        base_cooling_rate : float
            Base cooling rate with no fan (1/s)
        fan_speed_percent : float
            Fan speed as percentage (0-100)
            
        Returns:
        --------
        float
            Adjusted cooling rate (1/s)
            
        Notes:
        ------
        Fan can increase cooling rate by 2-3x at full speed.
        This is a simplified linear model.
        """
        # Fan effect: can increase cooling rate by up to 2x
        fan_multiplier = 1.0 + (fan_speed_percent / 100.0)
        return base_cooling_rate * fan_multiplier
    
    @staticmethod
    def adjust_cooling_rate_for_layer_time(base_cooling_rate, layer_time):
        """
        Adjust cooling rate based on layer time.
        
        Longer layer times mean the previous layer has more time to cool,
        which can affect heat transfer to new layers.
        
        Parameters:
        -----------
        base_cooling_rate : float
            Base cooling rate (1/s)
        layer_time : float
            Time to complete a layer in seconds
            
        Returns:
        --------
        float
            Adjusted cooling rate (1/s)
            
        Notes:
        ------
        This is a simplified model. In reality, heat transfer is more complex.
        """
        # Faster layers have slightly increased cooling due to less heat accumulation
        if layer_time < 5:
            # Very fast layer - material doesn't have time to transfer heat to bed
            return base_cooling_rate * 1.2
        elif layer_time > 30:
            # Slow layer - previous material is already cool
            return base_cooling_rate * 0.9
        else:
            return base_cooling_rate
