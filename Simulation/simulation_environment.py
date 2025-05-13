# simulation/simulation_environment.py
"""
SimulationEnvironment for ventilation control system.
Models room CO2, temperature, and humidity dynamics.
"""
import logging
import numpy as np
from typing import Dict, Tuple, List, Any, Optional

logger = logging.getLogger(__name__)

class SimulationEnvironment:
    """
    Physics-based room model for simulating indoor environment variables.
    
    This class simulates the dynamics of CO2, temperature, and humidity in an
    indoor space based on occupancy, ventilation actions, and external conditions.
    It provides a realistic environment for training and evaluating ventilation
    control algorithms.
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
            "HUMIDITY_INCREASE_PER_PERSON_PERCENT_PER_HOUR": 0.8,
            "HUMIDITY_INCREASE_PER_SLEEPING_PERSON_PERCENT_PER_HOUR": 0.5,
            "HUMIDITY_DECREASE_VENTILATION_PERCENT_PER_HOUR": {
                "off": 0,
                "low": 0.3,
                "medium": 0.7,
                "max": 1.0
            },
            "MAX_SIMULATED_HUMIDITY_PERCENT": 65.0,
            "MIN_SIMULATED_HUMIDITY_PERCENT": 25.0,
            
            # Temperature parameters
            "EXTERNAL_TEMP_C": 15.0,
            "ROOM_INITIAL_TEMP_C": 21.0,
            "NATURAL_TEMP_CHANGE_C_PER_HOUR": 0.5,  # naturally moving toward external temp
            "PERSON_HEAT_GENERATION_C_PER_HOUR": 0.2,  # heat from a person
            "SLEEPING_PERSON_HEAT_GENERATION_C_PER_HOUR": 0.15,  # heat from a sleeping person
            "VENTILATION_COOLING_EFFECT_C_PER_HOUR": {
                "off": 0,
                "low": 0.1,
                "medium": 0.25,
                "max": 0.4
            },
            "MAX_SIMULATED_TEMP_C": 30.0,
            "MIN_SIMULATED_TEMP_C": 15.0
        }
        
        # Override defaults with provided config
        if config:
            for key, value in config.items():
                if isinstance(value, dict) and key in self.config and isinstance(self.config[key], dict):
                    # Merge nested dictionaries
                    self.config[key].update(value)
                else:
                    # Replace or add top-level keys
                    self.config[key] = value
        
        # Initialize environment state
        self.reset()
        
        # Store simulation history for analysis
        self.history = []
        
        logger.info(f"Initialized SimulationEnvironment with room volume {self.config['ROOM_VOLUME_M3']}m³")
        logger.info(f"CO2 generation rates: {self.config['CO2_GENERATION_PER_PERSON_PPM_PER_HOUR']}ppm/h (awake), "
                   f"{self.config['CO2_GENERATION_PER_PERSON_SLEEPING_PPM_PER_HOUR']}ppm/h (sleeping)")
    
    def reset(self) -> Dict[str, Any]:
        """
        Reset environment to initial state.
        
        Returns:
            dict: Current state after reset
        """
        # Initialize to slightly above external conditions
        self.co2_ppm = self.config["EXTERNAL_CO2_PPM"] + 200
        self.temperature_c = self.config["ROOM_INITIAL_TEMP_C"]
        self.humidity_percent = self.config["BASE_HUMIDITY_PERCENT"]
        
        # Reset energy tracking
        self.current_energy_consumption_w = 0
        self.total_energy_consumption_wh = 0
        
        # Clear history
        self.history = []
        
        logger.info(f"Environment reset: CO2={self.co2_ppm}ppm, Temperature={self.temperature_c}°C, "
                   f"Humidity={self.humidity_percent}%")
        
        return self.get_current_state()
    
    def get_current_state(self) -> Dict[str, Any]:
        """
        Get current environment state.
        
        Returns:
            dict: Current environmental conditions
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
        
        Simulates the changes in CO2, temperature, and humidity based on occupancy,
        ventilation action, and natural environmental dynamics.
        
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
        
        # Log starting state for troubleshooting
        logger.debug(f"Step input: Action={action}, AwakeOcc={num_occupants_awake}, "
                    f"SleepingOcc={num_occupants_sleeping}, TimeStep={time_step_minutes}min")
        logger.debug(f"State before step: CO2={self.co2_ppm:.1f}ppm, Temp={self.temperature_c:.1f}°C, "
                    f"Humidity={self.humidity_percent:.1f}%")
        
        # Calculate and apply changes to all environmental variables
        # CO2 change
        co2_change = self._calculate_co2_change(
            action, num_occupants_awake, num_occupants_sleeping, time_step_hours
        )
        new_co2 = max(self.config["EXTERNAL_CO2_PPM"], self.co2_ppm + co2_change)
        
        # Temperature change
        temp_change = self._calculate_temperature_change(
            action, num_occupants_awake, num_occupants_sleeping, time_step_hours
        )
        new_temp = np.clip(
            self.temperature_c + temp_change,
            self.config["MIN_SIMULATED_TEMP_C"],
            self.config["MAX_SIMULATED_TEMP_C"]
        )
        
        # Humidity change
        humidity_change = self._calculate_humidity_change(
            action, num_occupants_awake, num_occupants_sleeping, time_step_hours
        )
        new_humidity = np.clip(
            self.humidity_percent + humidity_change,
            self.config["MIN_SIMULATED_HUMIDITY_PERCENT"],
            self.config["MAX_SIMULATED_HUMIDITY_PERCENT"]
        )
        
        # Calculate energy consumption
        energy_rate_w = self.config["VENTILATION_POWER_CONSUMPTION_W"][action]
        energy_consumed_wh = energy_rate_w * time_step_hours
        
        # Apply changes
        self.co2_ppm = new_co2
        self.temperature_c = new_temp
        self.humidity_percent = new_humidity
        self.current_energy_consumption_w = energy_rate_w
        self.total_energy_consumption_wh += energy_consumed_wh
        
        # Record this state in history
        self.history.append({
            "action": action,
            "num_occupants_awake": num_occupants_awake,
            "num_occupants_sleeping": num_occupants_sleeping,
            "co2_ppm": self.co2_ppm,
            "temperature_c": self.temperature_c,
            "humidity_percent": self.humidity_percent,
            "energy_consumed_wh": energy_consumed_wh
        })
        
        # Log resulting state for troubleshooting
        logger.debug(f"State after step: CO2={self.co2_ppm:.1f}ppm, Temp={self.temperature_c:.1f}°C, "
                    f"Humidity={self.humidity_percent:.1f}%, Energy={energy_consumed_wh:.2f}Wh")
        
        return self.get_current_state(), energy_consumed_wh
    
    def _calculate_co2_change(self, action: str, num_occupants_awake: int, 
                              num_occupants_sleeping: int, time_step_hours: float) -> float:
        """
        Calculate CO2 level change for current time step.
        
        Models:
        1. CO2 generation by occupants (higher for awake vs sleeping)
        2. Natural decay toward external CO2 levels
        3. Mechanical ventilation removal based on speed setting
        
        Args:
            action: Ventilation setting
            num_occupants_awake: Number of awake people
            num_occupants_sleeping: Number of sleeping people
            time_step_hours: Time step in hours
            
        Returns:
            float: Net change in CO2 (ppm)
        """
        # Calculate current CO2 excess above external level
        co2_excess = max(0, self.co2_ppm - self.config["EXTERNAL_CO2_PPM"])
        
        # Natural decay toward external CO2 (exponential decay)
        natural_decay_rate = self.config["NATURAL_INFILTRATION_RATE_PPM_PER_HOUR"]
        natural_decay = -co2_excess * (1 - np.exp(-natural_decay_rate * time_step_hours / co2_excess)) if co2_excess > 0 else 0
        
        # CO2 generation from people
        co2_generation = (
            num_occupants_awake * self.config["CO2_GENERATION_PER_PERSON_PPM_PER_HOUR"] +
            num_occupants_sleeping * self.config["CO2_GENERATION_PER_PERSON_SLEEPING_PPM_PER_HOUR"]
        ) * time_step_hours
        
        # Mechanical ventilation CO2 removal (proportional to excess CO2)
        vent_rate = self.config["MECHANICAL_VENTILATION_RATES_PPM_PER_HOUR"][action]
        ventilation_removal = vent_rate * co2_excess / 1000 * time_step_hours if co2_excess > 0 else 0
        
        # Total change
        total_change = co2_generation + natural_decay - ventilation_removal
        
        logger.debug(f"CO2 change: Generation={co2_generation:.1f}ppm, NaturalDecay={natural_decay:.1f}ppm, "
                    f"Ventilation={ventilation_removal:.1f}ppm, Total={total_change:.1f}ppm")
        
        return total_change
    
    def _calculate_temperature_change(self, action: str, num_occupants_awake: int,
                                      num_occupants_sleeping: int, time_step_hours: float) -> float:
        """
        Calculate temperature change for current time step.
        
        Models:
        1. Natural tendency toward external temperature
        2. Heat generated by occupants (less when sleeping)
        3. Cooling effect of ventilation
        
        Args:
            action: Ventilation setting
            num_occupants_awake: Number of awake people
            num_occupants_sleeping: Number of sleeping people
            time_step_hours: Time step in hours
            
        Returns:
            float: Net change in temperature (°C)
        """
        # Natural tendency toward external temperature (slower change in better insulated rooms)
        natural_rate = self.config["NATURAL_TEMP_CHANGE_C_PER_HOUR"]
        temp_diff = self.config["EXTERNAL_TEMP_C"] - self.temperature_c
        natural_change = temp_diff * natural_rate * time_step_hours
        
        # Heat from people (less for sleeping people)
        heat_generation = (
            num_occupants_awake * self.config["PERSON_HEAT_GENERATION_C_PER_HOUR"] +
            num_occupants_sleeping * self.config["SLEEPING_PERSON_HEAT_GENERATION_C_PER_HOUR"]
        ) * time_step_hours
        
        # Cooling from ventilation (stronger effect at higher speeds)
        ventilation_cooling = self.config["VENTILATION_COOLING_EFFECT_C_PER_HOUR"][action] * time_step_hours
        
        # Total change (natural movement + people heat - ventilation cooling)
        total_change = natural_change + heat_generation - ventilation_cooling
        
        logger.debug(f"Temperature change: Natural={natural_change:.2f}°C, People={heat_generation:.2f}°C, "
                    f"Ventilation={ventilation_cooling:.2f}°C, Total={total_change:.2f}°C")
        
        return total_change
    
    def _calculate_humidity_change(self, action: str, num_occupants_awake: int,
                                   num_occupants_sleeping: int, time_step_hours: float) -> float:
        """
        Calculate humidity change for current time step.
        
        Models:
        1. Humidity increase from occupants (breathing, perspiration)
        2. Humidity decrease from ventilation
        3. Natural migration toward baseline (from structural materials)
        
        Args:
            action: Ventilation setting
            num_occupants_awake: Number of awake people
            num_occupants_sleeping: Number of sleeping people
            time_step_hours: Time step in hours
            
        Returns:
            float: Net change in humidity (%)
        """
        # Humidity increase from people (less when sleeping)
        humidity_increase = (
            num_occupants_awake * self.config["HUMIDITY_INCREASE_PER_PERSON_PERCENT_PER_HOUR"] +
            num_occupants_sleeping * self.config["HUMIDITY_INCREASE_PER_SLEEPING_PERSON_PERCENT_PER_HOUR"]
        ) * time_step_hours
        
        # Natural migration toward baseline (materials absorb/release moisture)
        base_humidity = self.config["BASE_HUMIDITY_PERCENT"]
        natural_rate = 0.1  # Rate of natural humidity change
        humidity_diff = base_humidity - self.humidity_percent
        natural_change = humidity_diff * natural_rate * time_step_hours
        
        # Ventilation effect (removes excess moisture when humidity is high)
        humidity_excess = max(0, self.humidity_percent - base_humidity)
        ventilation_factor = self.config["HUMIDITY_DECREASE_VENTILATION_PERCENT_PER_HOUR"][action]
        ventilation_effect = ventilation_factor * humidity_excess * time_step_hours
        
        # Total change
        total_change = humidity_increase + natural_change - ventilation_effect
        
        logger.debug(f"Humidity change: People={humidity_increase:.2f}%, Natural={natural_change:.2f}%, "
                    f"Ventilation={ventilation_effect:.2f}%, Total={total_change:.2f}%")
        
        return total_change
    
    def get_history(self) -> List[Dict[str, Any]]:
        """
        Get the full simulation history.
        
        Returns:
            list: History of environmental states and actions
        """
        return self.history
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the current configuration parameters.
        
        Returns:
            dict: Configuration parameters
        """
        return self.config.copy()
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """
        Update configuration parameters.
        
        Args:
            new_config: New parameters to update
        """
        for key, value in new_config.items():
            if isinstance(value, dict) and key in self.config and isinstance(self.config[key], dict):
                # Merge nested dictionaries
                self.config[key].update(value)
            else:
                # Replace or add top-level keys
                self.config[key] = value
        
        logger.info(f"Updated environment configuration: {list(new_config.keys())}")