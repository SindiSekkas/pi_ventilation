# simulation/mock_controllers.py
"""
Mock module for MarkovController and related components.
Provides compatibility layer for the simulation framework.
"""
import os
import json
import logging
import numpy as np
import threading
import time
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)

class MockMarkovController:
    """
    Mock implementation of MarkovController for simulation.
    
    Provides the same interface as the real MarkovController but
    with simplified internal logic for training/evaluation.
    """
    
    def __init__(self, data_manager, pico_manager, preference_manager=None, model_dir="data/markov", 
                 scan_interval=60, occupancy_analyzer=None, enable_exploration=True):
        """Initialize mock MarkovController."""
        self.data_manager = data_manager
        self.pico_manager = pico_manager
        self.preference_manager = preference_manager
        self.occupancy_analyzer = occupancy_analyzer
        self.model_dir = model_dir
        self.scan_interval = scan_interval
        self.enable_exploration = enable_exploration
        
        # Ensure model directory exists
        os.makedirs(model_dir, exist_ok=True)
        
        # Q-learning parameters
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.exploration_rate = 0.1 if enable_exploration else 0.0
        self.MIN_EXPLORATION_RATE = 0.01
        self.MAX_EXPLORATION_RATE = 0.5
        self.epsilon_decay = 0.995
        
        # State tracking
        self.current_state = None
        self.last_action = "off"
        self.last_action_time = None
        
        # Night mode settings
        self.night_mode_enabled = True
        self.night_mode_start_hour = 23
        self.night_mode_end_hour = 7
        
        # Thresholds
        self.co2_thresholds = {
            "low_max": 800,    # Upper bound for LOW
            "medium_max": 1200  # Upper bound for MEDIUM
        }
        
        self.temp_thresholds = {
            "low_max": 20,     # Upper bound for LOW
            "medium_max": 24    # Upper bound for MEDIUM
        }
        
        # Auto mode
        self.auto_mode = True
        
        # Initialize Q-values
        self.q_values = {}
        self.model_file = os.path.join(model_dir, "markov_model.json")
        self.load_q_values(self.model_file)
    
    def _evaluate_state(self):
        """Determine current state based on sensor data."""
        try:
            # Get occupancy
            occupants = self.data_manager.latest_data["room"]["occupants"]
            
            # Get CO2 level
            co2 = self.data_manager.latest_data["scd41"]["co2"]
            if co2 is None:
                logger.warning("Missing CO2 data")
                return None
            
            # Determine CO2 level category
            if co2 < self.co2_thresholds["low_max"]:
                co2_level = "low"
            elif co2 < self.co2_thresholds["medium_max"]:
                co2_level = "medium"
            else:
                co2_level = "high"
            
            # Get temperature
            temp = self.data_manager.latest_data["scd41"]["temperature"]
            if temp is None:
                logger.warning("Missing temperature data")
                return None
            
            # Determine temperature level
            if temp < self.temp_thresholds["low_max"]:
                temp_level = "low"
            elif temp < self.temp_thresholds["medium_max"]:
                temp_level = "medium"
            else:
                temp_level = "high"
            
            # Get occupancy state
            occupancy = "occupied" if occupants > 0 else "empty"
            
            # Get time of day (simplified)
            hour = datetime.now().hour
            time_of_day = "night" if (hour < 5 or hour >= 22) else "day"
            
            # Create state key
            state_key = f"{co2_level}_{temp_level}_{occupancy}_{time_of_day}"
            return state_key
        
        except Exception as e:
            logger.error(f"Error evaluating state: {e}")
            return None
    
    def _decide_action(self):
        """Decide action based on current state."""
        # Update current state
        self.current_state = self._evaluate_state()
        
        if not self.current_state:
            logger.warning("Current state is None, defaulting to TURN_OFF")
            return "off"
        
        # Check if night mode is active
        if self._is_night_mode_active():
            return "off"
        
        # Get possible actions
        actions = ["off", "low", "medium", "max"]
        
        # Use epsilon-greedy for exploration/exploitation
        if self.enable_exploration and np.random.random() < self.exploration_rate:
            # Exploration - random action
            return np.random.choice(actions)
        
        # Exploitation - use Q-values
        if self.current_state in self.q_values:
            q_values = self.q_values[self.current_state]
            if q_values:
                best_action = max(q_values, key=q_values.get)
                return best_action
        
        # Default action based on CO2 level
        co2_level = self.current_state.split('_')[0]
        if co2_level == "low":
            return "off"
        elif co2_level == "medium":
            return "low"
        else:  # high
            return "medium"
    
    def _update_q_value(self, state_key, action, reward, next_state_key):
        """Update Q-values using Q-learning algorithm."""
        if not state_key or not next_state_key:
            return
        
        # Initialize if needed
        if state_key not in self.q_values:
            self.q_values[state_key] = {}
        
        if action not in self.q_values[state_key]:
            self.q_values[state_key][action] = 0.0
        
        # Get current Q-value
        current_q = self.q_values[state_key][action]
        
        # Get max Q-value for next state
        max_next_q = 0.0
        if next_state_key in self.q_values:
            next_q_values = self.q_values[next_state_key]
            if next_q_values:
                max_next_q = max(next_q_values.values())
        
        # Update Q-value using Q-learning formula
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_values[state_key][action] = new_q
    
    def _is_night_mode_active(self):
        """Check if night mode is currently active."""
        if not self.night_mode_enabled:
            return False
        
        current_hour = datetime.now().hour
        
        # Handle case where night mode crosses midnight
        if self.night_mode_start_hour > self.night_mode_end_hour:
            return current_hour >= self.night_mode_start_hour or current_hour < self.night_mode_end_hour
        else:
            return self.night_mode_start_hour <= current_hour < self.night_mode_end_hour
    
    def load_q_values(self, filepath):
        """Load Q-values from file."""
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    self.q_values = json.load(f)
                logger.info(f"Loaded {len(self.q_values)} Q-values from {filepath}")
                return True
            except Exception as e:
                logger.error(f"Error loading Q-values: {e}")
        
        logger.info("No Q-values file found, starting with empty Q-table")
        return False
    
    def save_q_values(self, filepath):
        """Save Q-values to file."""
        try:
            with open(filepath, 'w') as f:
                json.dump(self.q_values, f, indent=2)
            logger.info(f"Saved {len(self.q_values)} Q-values to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving Q-values: {e}")
            return False
    
    def set_auto_mode(self, enabled):
        """Enable/disable auto mode."""
        self.auto_mode = enabled
        return True
    
    def set_night_mode(self, enabled, start_hour=None, end_hour=None):
        """Configure night mode settings."""
        self.night_mode_enabled = enabled
        if start_hour is not None:
            self.night_mode_start_hour = start_hour
        if end_hour is not None:
            self.night_mode_end_hour = end_hour
        return True
    
    def get_status(self):
        """Get controller status information."""
        return {
            "auto_mode": self.auto_mode,
            "co2_thresholds": self.co2_thresholds,
            "temp_thresholds": self.temp_thresholds,
            "current_state": self.current_state,
            "last_action": self.last_action,
            "exploration_rate": self.exploration_rate,
            "ventilation_status": self.pico_manager.get_ventilation_status(),
            "ventilation_speed": self.pico_manager.get_ventilation_speed(),
            "night_mode": {
                "enabled": self.night_mode_enabled,
                "start_hour": self.night_mode_start_hour,
                "end_hour": self.night_mode_end_hour,
                "currently_active": self._is_night_mode_active()
            }
        }