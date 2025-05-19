# simulation/environment.py
"""
Environment simulator for adaptive ventilation system.
Models indoor air quality dynamics including CO2 concentration and temperature.
"""
import numpy as np
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class EnvironmentSimulator:
    """
    Simulates the indoor environment dynamics including CO2 concentration,
    temperature, and ventilation effects based on empirical measurements.
    """
    
    def __init__(self, 
                 initial_co2=420,           # ppm (outdoor level)
                 initial_temp=20.0,         # °C
                 initial_humidity=40.0,     # %
                 room_volume=62.5,          # m³ (typical bedroom)
                 co2_natural_decay=128.8,   # ppm/hour (from thesis)
                 temp_natural_decay=0.5,    # °C/hour
                 outdoor_co2=420,           # ppm
                 outdoor_temp_min=15.0,      # °C
                 outdoor_temp_max=25.0,     # °C
                 outdoor_humidity=40.0,     # %
                 co2_per_person=243.1):     # ppm/hour/person (from thesis)
        """
        Initialize the environment simulator with default values based on research.
        
        Args:
            initial_co2: Starting CO2 concentration in ppm
            initial_temp: Starting temperature in °C
            initial_humidity: Starting humidity in %
            room_volume: Room volume in cubic meters
            co2_natural_decay: Natural CO2 decrease rate in ppm/hour
            temp_natural_decay: Natural temperature equilibration rate in °C/hour
            outdoor_co2: Outdoor CO2 level in ppm
            outdoor_temp_min: Minimum outdoor temperature in °C
            outdoor_temp_max: Maximum outdoor temperature in °C
            outdoor_humidity: Outdoor humidity level in %
            co2_per_person: CO2 generation rate per person in ppm/hour
        """
        # Current environmental state
        self.co2 = initial_co2
        self.temperature = initial_temp
        self.humidity = initial_humidity
        
        # Room parameters
        self.room_volume = room_volume
        self.co2_natural_decay = co2_natural_decay
        self.temp_natural_decay = temp_natural_decay
        
        # Outdoor conditions
        self.outdoor_co2 = outdoor_co2
        self.outdoor_temp_min = outdoor_temp_min
        self.outdoor_temp_max = outdoor_temp_max
        self.outdoor_humidity = outdoor_humidity
        
        # Human factors
        self.co2_per_person = co2_per_person
        
        # Ventilation effect rates from thesis (ppm/hour)
        self.ventilation_rates = {
            "natural": 610.3,      # 2 windows in tilt mode
            "mechanical_low": 403.2,    # Fan 1
            "mechanical_medium": 768.5, # Fan 2
            "mechanical_max": 922.1     # Both fans
        }
        
        # Temperature effects of ventilation (°C/hour)
        self.temp_ventilation_effects = {
            "natural": 3.0,         # Natural ventilation effect on temperature
            "mechanical_low": 1.0,   # Low fan effect
            "mechanical_medium": 1.5, # Medium fan effect
            "mechanical_max": 2.0     # Max fan effect
        }
        
        # Humidity effects
        self.humidity_ventilation_effects = {
            "natural": 10.0,        # %/hour
            "mechanical_low": 3.0,
            "mechanical_medium": 6.0,
            "mechanical_max": 8.0
        }
        
        # Tracking
        self.current_time = datetime(2023, 1, 1, 0, 0, 0)  # Simulation start time
        self.history = []
        
        # Record initial state
        self._record_state()
    
    def _get_outdoor_temperature(self):
        """
        Calculate outdoor temperature based on time of day and yearly patterns.
        
        Returns:
            float: Current outdoor temperature in °C
        """
        # Daily cycle: coldest at 3 AM, warmest at 3 PM
        hour = self.current_time.hour
        minute = self.current_time.minute
        
        # Day of year affects temperature range
        day_of_year = self.current_time.timetuple().tm_yday
        yearly_factor = np.sin(2 * np.pi * (day_of_year - 172) / 365)  # Peak in summer
        
        # Daily cycle with peak at 15:00
        daily_factor = np.sin(2 * np.pi * ((hour + minute/60) - 3) / 24)
        
        # Calculate temperature with both cycles
        temp_range = (self.outdoor_temp_max - self.outdoor_temp_min) / 2
        temp_avg = (self.outdoor_temp_max + self.outdoor_temp_min) / 2
        
        # Combine effects: yearly cycle has 80% effect, daily cycle has 20% effect
        outdoor_temp = temp_avg + temp_range * (0.8 * yearly_factor + 0.2 * daily_factor)
        
        return max(self.outdoor_temp_min, min(self.outdoor_temp_max, outdoor_temp))
    
    def update(self, time_step_minutes, occupants, ventilation_mode=None, ventilation_speed=None):
        """
        Update the environment state for the given time step.
        
        Args:
            time_step_minutes: Time step in minutes
            occupants: Number of people in the room
            ventilation_mode: 'off', 'natural', or 'mechanical'
            ventilation_speed: 'low', 'medium', or 'max' for mechanical ventilation
            
        Returns:
            dict: Current environment state
        """
        # Convert time step to hours for calculations
        time_step_hours = time_step_minutes / 60.0
        
        # Update current time
        self.current_time += timedelta(minutes=time_step_minutes)
        
        # Get current outdoor temperature
        outdoor_temp = self._get_outdoor_temperature()
        
        # Calculate CO2 changes
        # 1. Natural decay
        co2_decay = self.co2_natural_decay * time_step_hours
        
        # 2. Human contribution
        co2_human = occupants * self.co2_per_person * time_step_hours
        
        # 3. Ventilation effect
        co2_ventilation = 0
        temp_ventilation = 0
        humidity_ventilation = 0
        
        if ventilation_mode == 'natural':
            co2_ventilation = self.ventilation_rates["natural"] * time_step_hours
            temp_ventilation = self.temp_ventilation_effects["natural"] * time_step_hours
            humidity_ventilation = self.humidity_ventilation_effects["natural"] * time_step_hours
        
        elif ventilation_mode == 'mechanical' and ventilation_speed in ['low', 'medium', 'max']:
            vent_key = f"mechanical_{ventilation_speed}"
            co2_ventilation = self.ventilation_rates[vent_key] * time_step_hours
            temp_ventilation = self.temp_ventilation_effects[vent_key] * time_step_hours
            humidity_ventilation = self.humidity_ventilation_effects[vent_key] * time_step_hours
        
        # Update CO2 concentration (ensures CO2 never goes below outdoor level)
        co2_net_change = co2_human - co2_decay - co2_ventilation
        self.co2 = max(self.outdoor_co2, self.co2 + co2_net_change)
        
        # Update temperature
        # Natural tendency toward outdoor temperature
        temp_net_change = (outdoor_temp - self.temperature) * self.temp_natural_decay * time_step_hours
        
        # Ventilation makes temperature closer to outdoor faster
        if ventilation_mode in ['natural', 'mechanical']:
            if outdoor_temp < self.temperature:
                temp_net_change -= temp_ventilation
            else:
                temp_net_change += temp_ventilation
        
        # Human body heat
        human_heat = 0.2 * occupants * time_step_hours  # Each person raises temp by ~0.2°C per hour
        
        self.temperature += temp_net_change + human_heat
        
        # Update humidity (simplified model)
        humidity_outdoor_diff = (self.outdoor_humidity - self.humidity) * 0.1 * time_step_hours
        humidity_human = 2.0 * occupants * time_step_hours  # People increase humidity
        
        if ventilation_mode in ['natural', 'mechanical']:
            if self.outdoor_humidity < self.humidity:
                humidity_ventilation = -humidity_ventilation
        
        self.humidity += humidity_outdoor_diff + humidity_human + humidity_ventilation
        self.humidity = max(10, min(95, self.humidity))  # Enforce realistic bounds
        
        # Record current state
        self._record_state(occupants=occupants, 
                          ventilation_mode=ventilation_mode, 
                          ventilation_speed=ventilation_speed)
        
        return self.get_current_state()
    
    def _record_state(self, occupants=0, ventilation_mode=None, ventilation_speed=None):
        """Record the current state to history."""
        self.history.append({
            'timestamp': self.current_time.isoformat(),
            'co2': round(self.co2, 1),
            'temperature': round(self.temperature, 1),
            'humidity': round(self.humidity, 1),
            'occupants': occupants,
            'ventilation_mode': ventilation_mode,
            'ventilation_speed': ventilation_speed
        })
        
        # Keep history to a reasonable size
        if len(self.history) > 10000:
            self.history = self.history[-5000:]
    
    def get_current_state(self):
        """
        Get the current environment state.
        
        Returns:
            dict: Current state with all environmental variables
        """
        return {
            'timestamp': self.current_time.isoformat(),
            'co2': round(self.co2, 1),
            'temperature': round(self.temperature, 1),
            'humidity': round(self.humidity, 1),
            'outdoor_temperature': round(self._get_outdoor_temperature(), 1),
            'outdoor_humidity': self.outdoor_humidity,
            'outdoor_co2': self.outdoor_co2
        }
    
    def get_sensor_data(self):
        """
        Get data formatted like real sensors for compatibility with existing systems.
        
        Returns:
            dict: Sensor-formatted data
        """
        return {
            "timestamp": self.current_time.strftime("%Y-%m-%d %H:%M:%S"),
            "scd41": {
                "co2": round(self.co2),
                "temperature": round(self.temperature, 1),
                "humidity": round(self.humidity, 1)
            },
            "bmp280": {
                "temperature": round(self.temperature, 1), 
                "pressure": 1013.25  # Standard pressure, not critical for simulation
            }
        }
    
    def reset(self, initial_co2=None, initial_temp=None, initial_humidity=None):
        """
        Reset the environment to initial conditions.
        
        Args:
            initial_co2: Optional new initial CO2 level
            initial_temp: Optional new initial temperature
            initial_humidity: Optional new initial humidity
        """
        if initial_co2 is not None:
            self.co2 = initial_co2
        if initial_temp is not None:
            self.temperature = initial_temp
        if initial_humidity is not None:
            self.humidity = initial_humidity
        
        self.history = []
        self._record_state()
        
        logger.info(f"Environment reset: CO2={self.co2}ppm, "
                   f"Temp={self.temperature}°C, "
                   f"Humidity={self.humidity}%")
    
    def export_history(self):
        """
        Export the environmental history for analysis.
        
        Returns:
            list: Complete environment history
        """
        return self.history