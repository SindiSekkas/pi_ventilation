# simulation/reward_function.py
"""
Reward function for ventilation controller in simulation.
Calculates reward signals for reinforcement learning.
"""
import logging
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

class RewardFunction:
    """
    Calculates reward signals for the Markov ventilation controller.
    
    Generates reward values based on multiple weighted factors:
    - CO2 comfort (preferred vs. actual levels)
    - Temperature comfort (preferred range vs. actual)
    - Humidity comfort (preferred range vs. actual)
    - Energy cost (power consumption)
    - Action switching penalties (to avoid rapid changes)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize reward function with configuration parameters.
        
        Args:
            config: Dictionary of configuration parameters
        """
        # Set default reward parameters
        self.config = {
            # CO2 rewards/penalties
            "REWARD_CO2_COMFORT": 1.0,
            "PENALTY_CO2_SLIGHT_DISCOMFORT": -0.5,  # CO2 > threshold but < threshold * 1.2
            "PENALTY_CO2_HIGH_DISCOMFORT": -2.0,    # CO2 > threshold * 1.2
            
            # Temperature rewards/penalties
            "REWARD_TEMP_COMFORT": 0.8,
            "PENALTY_TEMP_DISCOMFORT": -1.0,
            
            # Humidity rewards/penalties (optional)
            "REWARD_HUMIDITY_COMFORT": 0.5,
            "PENALTY_HUMIDITY_DISCOMFORT": -0.8,
            
            # Energy and switching costs
            "ENERGY_COST_MULTIPLIER": -0.01,  # to scale energy Wh to reward units
            "SWITCHING_ACTION_PENALTY": -0.2,
            
            # Reward component weights
            "WEIGHT_CO2": 1.0,
            "WEIGHT_TEMP": 0.7,
            "WEIGHT_HUMIDITY": 0.5,
            "WEIGHT_ENERGY": 1.0,
            "WEIGHT_SWITCHING": 0.3
        }
        
        # Override defaults with provided config
        if config:
            for key, value in config.items():
                self.config[key] = value
                
        logger.info("Initialized reward function with weights: "
                  f"CO2={self.config['WEIGHT_CO2']}, "
                  f"Temp={self.config['WEIGHT_TEMP']}, "
                  f"Energy={self.config['WEIGHT_ENERGY']}, "
                  f"Switching={self.config['WEIGHT_SWITCHING']}")
    
    def calculate_reward(self, current_env_state: Dict[str, Any], 
                         compromise_preferences: Dict[str, Any],
                         action_taken: str, energy_consumed: float,
                         previous_action: Optional[str] = None) -> Tuple[float, Dict[str, float]]:
        """
        Calculate the reward for the current state, preferences and action.
        
        Args:
            current_env_state: Current environment state (CO2, temperature, humidity)
            compromise_preferences: User preferences from PreferenceManager
            action_taken: Ventilation action that was taken
            energy_consumed: Energy used in this step (Wh)
            previous_action: Previous ventilation action (for switching penalty)
            
        Returns:
            tuple: (total_reward, reward_components)
        """
        # Extract current environment state
        co2_level = current_env_state.get("co2_ppm", 0)
        temperature = current_env_state.get("temperature_c", 21.0)
        humidity = current_env_state.get("humidity_percent", 50.0)
        
        # Extract user preferences
        co2_threshold = compromise_preferences.get("co2_threshold", 1000)
        temp_min = compromise_preferences.get("temp_min", 20.0)
        temp_max = compromise_preferences.get("temp_max", 24.0)
        humidity_min = compromise_preferences.get("humidity_min", 30.0)
        humidity_max = compromise_preferences.get("humidity_max", 60.0)
        
        # Initialize reward components
        reward_components = {
            "co2": 0.0,
            "temperature": 0.0,
            "humidity": 0.0,
            "energy": 0.0,
            "switching": 0.0
        }
        
        # Calculate CO2 comfort reward/penalty
        if co2_level <= co2_threshold:
            reward_components["co2"] = self.config["REWARD_CO2_COMFORT"]
        elif co2_level < co2_threshold * 1.2:
            reward_components["co2"] = self.config["PENALTY_CO2_SLIGHT_DISCOMFORT"]
        else:
            reward_components["co2"] = self.config["PENALTY_CO2_HIGH_DISCOMFORT"]
        
        # Calculate temperature comfort reward/penalty
        if temp_min <= temperature <= temp_max:
            reward_components["temperature"] = self.config["REWARD_TEMP_COMFORT"]
        else:
            # Scale penalty by how far outside comfortable range
            temp_deviation = min(abs(temperature - temp_min), abs(temperature - temp_max))
            scaled_penalty = self.config["PENALTY_TEMP_DISCOMFORT"] * (1 + temp_deviation / 2)
            reward_components["temperature"] = scaled_penalty
        
        # Calculate humidity comfort reward/penalty
        if humidity_min <= humidity <= humidity_max:
            reward_components["humidity"] = self.config["REWARD_HUMIDITY_COMFORT"]
        else:
            # Scale penalty by how far outside comfortable range
            humidity_deviation = min(abs(humidity - humidity_min), abs(humidity - humidity_max))
            scaled_penalty = self.config["PENALTY_HUMIDITY_DISCOMFORT"] * (1 + humidity_deviation / 10)
            reward_components["humidity"] = scaled_penalty
        
        # Calculate energy cost penalty
        reward_components["energy"] = self.config["ENERGY_COST_MULTIPLIER"] * energy_consumed
        
        # Calculate action switching penalty
        if previous_action and previous_action != action_taken:
            reward_components["switching"] = self.config["SWITCHING_ACTION_PENALTY"]
        
        # Calculate weighted total reward
        total_reward = (
            self.config["WEIGHT_CO2"] * reward_components["co2"] +
            self.config["WEIGHT_TEMP"] * reward_components["temperature"] +
            self.config["WEIGHT_HUMIDITY"] * reward_components["humidity"] +
            self.config["WEIGHT_ENERGY"] * reward_components["energy"] +
            self.config["WEIGHT_SWITCHING"] * reward_components["switching"]
        )
        
        logger.debug(
            f"Reward calculation: CO2={reward_components['co2']:.2f}, "
            f"Temp={reward_components['temperature']:.2f}, "
            f"Energy={reward_components['energy']:.2f}, "
            f"Total={total_reward:.2f}"
        )
        
        return total_reward, reward_components