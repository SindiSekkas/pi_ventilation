# simulation/simulation_environment.py
"""
SimulationEnvironment for ventilation control system.
Models room CO2, temperature, and humidity dynamics.
"""
import logging
import numpy as np
from typing import Dict, Tuple, Any, Optional

logger = logging.getLogger(__name__)

class SimulationEnvironment:
    """
    Physics-based room model for simulating indoor environment variables.
    
    Tracks and updates CO2 levels, temperature, and humidity based on 
    occupancy and ventilation actions. Uses configurable parameters to
    model natural infiltration and human-generated CO2.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize simulation environment with configuration parameters.
        
        Args:
            config: Dictionary of configuration parameters
        """
        # Set default parameters
        self.config = {
            # Room dimensions
            "ROOM_VOLUME_M3": 62.5,
            
            # CO2 parameters
            "EXTERNAL_CO2_PPM": 420,
            "NATURAL_INFILTRATION_RATE_PPM_PER_HOUR": 128.8,  # decay when empty, relative to external CO2
            "CO2_GENERATION_PER_PERSON_PPM_PER_HOUR": 243.1,  # awake
            "CO2_GENERATION_PER_PERSON_SLEEPING_PPM_PER_HOUR": 170.0,  # ~70% of awake
            
            # Ventilation rates
            "MECHANICAL_VENTILATION_RATES_PPM_PER_HOUR": {
                "off": 0,
                "low": 403.2,
                "medium": 768.5,
                "max": 922.1
            },
            
            # Power consumption
            "VENTILATION_POWER_CONSUMPTION_W": {
                "off": 0,
                "low": 24,
                "medium": 29,
                "max": 53
            },
            
            # Humidity parameters
            "BASE_HUMIDITY_PERCENT": 30.0,
            "HUMIDITY_INCREASE_VENTILATION_PERCENT_PER_HOUR": 1.0,
            "MAX_SIMULATED_HUMIDITY_PERCENT": 65.0,
            
            # Temperature parameters
            "EXTERNAL_TEMP_C": 15.0,
            "ROOM_INITIAL_TEMP_C": 21.0,
            "NATURAL_TEMP_CHANGE_C_PER_HOUR": 0.5,  # naturally moving toward external temp
            "PERSON_HEAT_GENERATION_C_PER_HOUR": 0.2,  # heat from a person
            "VENTILATION_COOLING_EFFECT_C_PER_HOUR": {
                "off": 0,
                "low": 0.1,
                "medium": 0.25,
                "max": 0.4
            }
        }
        
        # Override defaults with provided config
        if config:
            for key, value in config.items():
                self.config[key] = value
        
        # Initialize state
        self.reset()
    
    def reset(self) -> Dict[str, Any]:
        """
        Reset environment to initial state.
        
        Returns:
            dict: Current state after reset
        """
        self.co2_ppm = self.config["EXTERNAL_CO2_PPM"] + 200  # Start slightly above external
        self.temperature_c = self.config["ROOM_INITIAL_TEMP_C"]
        self.humidity_percent = self.config["BASE_HUMIDITY_PERCENT"]
        self.current_energy_consumption_w = 0
        self.total_energy_consumption_wh = 0
        
        return self.get_current_state()
    
    def get_current_state(self) -> Dict[str, Any]:
        """
        Get current environment state.
        
        Returns:
            dict: Current state (CO2, temperature, humidity)
        """
        return {
            "co2_ppm": self.co2_ppm,
            "temperature_c": self.temperature_c,
            "humidity_percent": self.humidity_percent,
            "current_energy_w": self.current_energy_consumption_w,
            "total_energy_wh": self.total_energy_consumption_wh
        }
    
    def step(self, action: str, num_occupants_awake: int = 0, num_occupants_sleeping: int = 0, 
             time_step_minutes: int = 5) -> Tuple[Dict[str, Any], float]:
        """
        Update environment for one time step based on action and occupancy.
        
        Args:
            action: Ventilation action ('off', 'low', 'medium', 'max')
            num_occupants_awake: Number of awake people in the room
            num_occupants_sleeping: Number of sleeping people in the room
            time_step_minutes: Time step in minutes
            
        Returns:
            tuple: (new_state, energy_consumed_wh)
        """
        time_step_hours = time_step_minutes / 60.0
        
        # Validate action
        if action not in self.config["MECHANICAL_VENTILATION_RATES_PPM_PER_HOUR"]:
            logger.warning(f"Invalid action: {action}. Using 'off' instead.")
            action = "off"
        
        # Calculate CO2 change
        co2_change = self._calculate_co2_change(
            action, num_occupants_awake, num_occupants_sleeping, time_step_hours
        )
        
        # Calculate temperature change
        temp_change = self._calculate_temperature_change(
            action, num_occupants_awake, num_occupants_sleeping, time_step_hours
        )
        
        # Calculate humidity change
        humidity_change = self._calculate_humidity_change(
            action, num_occupants_awake, num_occupants_sleeping, time_step_hours
        )
        
        # Apply changes
        self.co2_ppm = max(self.config["EXTERNAL_CO2_PPM"], self.co2_ppm + co2_change)
        self.temperature_c += temp_change
        self.humidity_percent = np.clip(
            self.humidity_percent + humidity_change,
            self.config["BASE_HUMIDITY_PERCENT"],
            self.config["MAX_SIMULATED_HUMIDITY_PERCENT"]
        )
        
        # Calculate energy consumption
        self.current_energy_consumption_w = self.config["VENTILATION_POWER_CONSUMPTION_W"][action]
        energy_consumed_wh = self.current_energy_consumption_w * time_step_hours
        self.total_energy_consumption_wh += energy_consumed_wh
        
        return self.get_current_state(), energy_consumed_wh
    
    def _calculate_co2_change(self, action: str, num_occupants_awake: int, 
                              num_occupants_sleeping: int, time_step_hours: float) -> float:
        """
        Calculate CO2 level change for current time step.
        
        Accounts for:
        - Natural decay toward external CO2 level
        - CO2 generation from occupants (different rates for awake vs sleeping)
        - Mechanical ventilation removal based on speed
        
        Args:
            action: Ventilation setting
            num_occupants_awake: Number of awake people
            num_occupants_sleeping: Number of sleeping people
            time_step_hours: Time step in hours
            
        Returns:
            float: Net change in CO2 (ppm)
        """
        # Natural decay toward external CO2
        natural_decay = (self.config["EXTERNAL_CO2_PPM"] - self.co2_ppm) * (
            1 - np.exp(-self.config["NATURAL_INFILTRATION_RATE_PPM_PER_HOUR"] * time_step_hours / 
                       (self.co2_ppm - self.config["EXTERNAL_CO2_PPM"] + 1e-6))
        )
        
        # CO2 generation from people
        co2_generation = (
            num_occupants_awake * self.config["CO2_GENERATION_PER_PERSON_PPM_PER_HOUR"] +
            num_occupants_sleeping * self.config["CO2_GENERATION_PER_PERSON_SLEEPING_PPM_PER_HOUR"]
        ) * time_step_hours
        
        # Mechanical ventilation CO2 removal
        ventilation_removal = (
            self.config["MECHANICAL_VENTILATION_RATES_PPM_PER_HOUR"][action] * 
            (self.co2_ppm - self.config["EXTERNAL_CO2_PPM"]) / 
            (self.co2_ppm - self.config["EXTERNAL_CO2_PPM"] + 1e-6) * 
            time_step_hours
        )
        
        return co2_generation + natural_decay - ventilation_removal
    
    def _calculate_temperature_change(self, action: str, num_occupants_awake: int,
                                      num_occupants_sleeping: int, time_step_hours: float) -> float:
        """
        Calculate temperature change for current time step.
        
        Accounts for:
        - Natural tendency toward external temperature
        - Heat generated by occupants
        - Cooling effect of ventilation
        
        Args:
            action: Ventilation setting
            num_occupants_awake: Number of awake people
            num_occupants_sleeping: Number of sleeping people
            time_step_hours: Time step in hours
            
        Returns:
            float: Net change in temperature (°C)
        """
        # Natural change toward external temperature
        natural_change = (
            (self.config["EXTERNAL_TEMP_C"] - self.temperature_c) * 
            self.config["NATURAL_TEMP_CHANGE_C_PER_HOUR"] * 
            time_step_hours
        )
        
        # Heat from people (assume sleeping people generate less heat)
        people_heat = (
            num_occupants_awake * self.config["PERSON_HEAT_GENERATION_C_PER_HOUR"] +
            num_occupants_sleeping * self.config["PERSON_HEAT_GENERATION_C_PER_HOUR"] * 0.7
        ) * time_step_hours
        
        # Cooling from ventilation
        ventilation_cooling = (
            self.config["VENTILATION_COOLING_EFFECT_C_PER_HOUR"][action] * 
            time_step_hours
        )
        
        return natural_change + people_heat - ventilation_cooling
    
    def _calculate_humidity_change(self, action: str, num_occupants_awake: int,
                                   num_occupants_sleeping: int, time_step_hours: float) -> float:
        """
        Calculate humidity change for current time step.
        
        Accounts for:
        - Humidity increase from occupants
        - Humidity decrease from ventilation
        
        Args:
            action: Ventilation setting
            num_occupants_awake: Number of awake people
            num_occupants_sleeping: Number of sleeping people
            time_step_hours: Time step in hours
            
        Returns:
            float: Net change in humidity (%)
        """
        # Base humidity is controlled by climate control systems (not modeled in detail)
        # We'll model a simple increase from occupants and decrease from ventilation
        
        # People add humidity
        total_occupants = num_occupants_awake + num_occupants_sleeping
        humidity_increase = total_occupants * 1.5 * time_step_hours
        
        # Ventilation can decrease humidity if it's high
        ventilation_factor = {
            "off": 0,
            "low": 0.3,
            "medium": 0.7,
            "max": 1.0
        }
        
        # Only reduce humidity if it's above base level
        if self.humidity_percent > self.config["BASE_HUMIDITY_PERCENT"]:
            humidity_decrease = (
                ventilation_factor[action] * 
                self.config["HUMIDITY_INCREASE_VENTILATION_PERCENT_PER_HOUR"] * 
                time_step_hours
            )
        else:
            humidity_decrease = 0
        
        return humidity_increase - humidity_decrease