# simulation/simulation_managers.py
"""
SimulationDataManager and SimulationPicoManager for ventilation system.
Provides mock interfaces compatible with the real system components.
"""
import logging
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class SimulationDataManager:
    """
    Mock version of DataManager for simulation.
    
    Provides the same interface as the real DataManager but works
    with simulated data instead of real sensors.
    """
    
    def __init__(self):
        """Initialize simulation data manager with default values."""
        self.latest_data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "scd41": {"co2": 450, "temperature": 21.0, "humidity": 40.0},
            "bmp280": {"temperature": 21.0, "pressure": 1013.0},
            "room": {"occupants": 0, "ventilated": False, "ventilation_speed": "off"},
            "initialization": {
                "status": False,
                "current": 5,
                "total": 5,
                "time_remaining": 0
            }
        }
    
    def update_sensor_data(self, env_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update sensor data with simulated readings.
        
        Args:
            env_state: Current environment state
            
        Returns:
            dict: Updated sensor data
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        self.latest_data.update({
            "timestamp": timestamp,
            "scd41": {
                "co2": round(env_state.get("co2_ppm", 450), 1),
                "temperature": round(env_state.get("temperature_c", 21.0), 1),
                "humidity": round(env_state.get("humidity_percent", 40.0), 1)
            },
            "bmp280": {
                "temperature": round(env_state.get("temperature_c", 21.0), 1),
                "pressure": 1013.0  # Fixed value as not modeled in simulation
            }
        })
        
        logger.debug(f"Updated simulated sensor data: CO2={self.latest_data['scd41']['co2']} ppm, " 
                   f"Temp={self.latest_data['scd41']['temperature']}°C, "
                   f"Humidity={self.latest_data['scd41']['humidity']}%")
        return self.latest_data
    
    def update_room_data(self, occupants: Optional[int] = None, 
                        ventilated: Optional[bool] = None, 
                        ventilation_speed: Optional[str] = None) -> Dict[str, Any]:
        """
        Update room occupancy and ventilation status.
        
        Args:
            occupants: Number of people in room
            ventilated: Whether ventilation is active
            ventilation_speed: Ventilation speed setting
            
        Returns:
            dict: Updated room data
        """
        if occupants is not None:
            self.latest_data["room"]["occupants"] = occupants
        if ventilated is not None:
            self.latest_data["room"]["ventilated"] = ventilated
        if ventilation_speed is not None:
            self.latest_data["room"]["ventilation_speed"] = ventilation_speed
        
        return self.latest_data["room"]
    
    def save_measurement_to_csv(self, ventilation_status: bool, ventilation_speed: str = "off") -> bool:
        """
        Mock saving to CSV for simulation (does nothing).
        
        Args:
            ventilation_status: Whether ventilation is active
            ventilation_speed: Ventilation speed setting
            
        Returns:
            bool: Success indicator (always True)
        """
        # In simulation, we don't actually write to disk
        return True


class SimulationPicoManager:
    """
    Mock version of PicoManager for simulation.
    
    Provides the same interface as the real PicoManager but works
    with the simulation environment instead of real hardware.
    """
    
    def __init__(self):
        """Initialize simulation PicoManager."""
        self.ventilation_status = False
        self.ventilation_speed = "off"
        self.simulation_env = None  # Will be set by the SimulationRunner
    
    def set_simulation_env(self, simulation_env):
        """
        Link to the simulation environment.
        
        Args:
            simulation_env: Reference to SimulationEnvironment
        """
        self.simulation_env = simulation_env
    
    def get_ventilation_status(self) -> bool:
        """
        Get current ventilation status.
        
        Returns:
            bool: Whether ventilation is active
        """
        return self.ventilation_status
    
    def get_ventilation_speed(self) -> str:
        """
        Get current ventilation speed.
        
        Returns:
            str: Ventilation speed ('off', 'low', 'medium', 'max')
        """
        return self.ventilation_speed
    
    def control_ventilation(self, state: str, speed: Optional[str] = None) -> bool:
        """
        Control ventilation state and speed.
        
        Args:
            state: 'on' or 'off'
            speed: 'low', 'medium', or 'max' (required if state is 'on')
            
        Returns:
            bool: Success indicator
        """
        if state not in ['on', 'off']:
            logger.error(f"Invalid ventilation state: {state}")
            return False
        
        if state == 'on':
            if speed not in ['low', 'medium', 'max']:
                logger.error(f"Invalid ventilation speed: {speed}")
                return False
            
            self.ventilation_status = True
            self.ventilation_speed = speed
            
        else:  # 'off'
            self.ventilation_status = False
            self.ventilation_speed = "off"
        
        logger.debug(f"Ventilation set to {state} ({speed if state == 'on' else 'off'})")
        return True