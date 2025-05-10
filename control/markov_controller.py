# control/markov_controller.py
"""Markov Decision Process based ventilation controller."""
import os
import json
import logging
import threading
import time
import random
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)

# Define state and action enums
class CO2Level(Enum):
    """CO2 concentration levels."""
    LOW = "low"       # Good air quality
    MEDIUM = "medium" # Acceptable air quality
    HIGH = "high"     # Poor air quality

class TemperatureLevel(Enum):
    """Temperature levels."""
    LOW = "low"          # < 20°C
    MEDIUM = "medium"    # 20-24°C
    HIGH = "high"        # > 24°C

class TimeOfDay(Enum):
    """Time of day periods."""
    MORNING = "morning"  # 5:00-12:00
    DAY = "day"          # 12:00-18:00
    EVENING = "evening"  # 18:00-22:00
    NIGHT = "night"      # 22:00-5:00

class Occupancy(Enum):
    """Room occupancy levels."""
    EMPTY = "empty"      # No people
    OCCUPIED = "occupied"  # At least one person

class Action(Enum):
    """Ventilation actions."""
    TURN_OFF = "off"
    TURN_ON_LOW = "low"
    TURN_ON_MEDIUM = "medium"
    TURN_ON_MAX = "max"


class MarkovController:
    """Markov Decision Process based ventilation controller."""
    
    def __init__(self, data_manager, pico_manager, preference_manager=None, model_dir="data/markov", scan_interval=60):
        """
        Initialize the Markov controller.
        
        Args:
            data_manager: Data manager instance for sensor readings
            pico_manager: Pico manager for controlling ventilation
            preference_manager: Preference manager for user preferences (optional)
            model_dir: Directory to store the Markov model
            scan_interval: How often to check conditions (seconds)
        """
        self.data_manager = data_manager
        self.pico_manager = pico_manager
        self.preference_manager = preference_manager  # New: store preference manager
        self.model_dir = model_dir
        self.scan_interval = scan_interval
        os.makedirs(model_dir, exist_ok=True)
        
        # Control thread
        self.running = False
        self.thread = None
        
        # MDP model file
        self.model_file = os.path.join(model_dir, "markov_model.json")
        
        # Default CO2 thresholds (will be updated dynamically)
        self.co2_thresholds = {
            "low_max": 800,    # Upper bound for LOW
            "medium_max": 1200  # Upper bound for MEDIUM
        }
        
        # Default temperature thresholds (will be updated dynamically)
        self.temp_thresholds = {
            "low_max": 20,     # Upper bound for LOW
            "medium_max": 24    # Upper bound for MEDIUM
        }
        
        # Default thresholds for empty home (more energy-saving)
        self.default_empty_home_co2_thresholds = {
            "low_max": 850,
            "medium_max": 1300
        }
        
        self.default_empty_home_temp_thresholds = {
            "low_max": 18,
            "medium_max": 26
        }
        
        # State tracking
        self.current_state = None
        self.last_action = None
        self.last_action_time = None
        self.min_action_interval = 300  # Min time between action changes (seconds)
        
        # Control state
        self.auto_mode = True
        self.learning_rate = 0.1  # How quickly model adapts to new information
        self.exploration_rate = 0.2  # Chance of trying random action
        self.random_action_decay = 0.9999  # Decay rate for exploration
        
        # Night mode settings
        self.night_mode_enabled = True
        self.night_mode_start_hour = 23
        self.night_mode_end_hour = 7
        
        # Initialize with -1 to ensure first update is logged
        self.last_applied_occupants = -1
        
        # Load night mode settings from file
        self._load_night_mode_settings()
        
        # Load or initialize transition model
        self.transition_model = self._load_or_initialize_model()
    
    def _load_night_mode_settings(self):
        """Load night mode settings from file."""
        night_settings_file = os.path.join(self.model_dir, "night_mode_settings.json")
        try:
            if os.path.exists(night_settings_file):
                with open(night_settings_file, 'r') as f:
                    settings = json.load(f)
                    self.night_mode_enabled = settings.get("enabled", True)
                    self.night_mode_start_hour = settings.get("start_hour", 23)
                    self.night_mode_end_hour = settings.get("end_hour", 7)
                    logger.info(f"Loaded night mode settings: {self.night_mode_start_hour}:00 - {self.night_mode_end_hour}:00")
        except Exception as e:
            logger.error(f"Error loading night mode settings: {e}")
    
    def _save_night_mode_settings(self):
        """Save night mode settings to file."""
        night_settings_file = os.path.join(self.model_dir, "night_mode_settings.json")
        try:
            settings = {
                "enabled": self.night_mode_enabled,
                "start_hour": self.night_mode_start_hour,
                "end_hour": self.night_mode_end_hour
            }
            with open(night_settings_file, 'w') as f:
                json.dump(settings, f, indent=2)
            logger.info("Saved night mode settings")
        except Exception as e:
            logger.error(f"Error saving night mode settings: {e}")
    
    def _is_night_mode_active(self):
        """Check if night mode is currently active."""
        if not self.night_mode_enabled:
            return False
        
        current_hour = datetime.now().hour
        
        # Handle case where night mode crosses midnight
        if self.night_mode_start_hour > self.night_mode_end_hour:
            # Night mode spans midnight (e.g., 23:00 - 7:00)
            return current_hour >= self.night_mode_start_hour or current_hour < self.night_mode_end_hour
        else:
            # Night mode does not span midnight
            return self.night_mode_start_hour <= current_hour < self.night_mode_end_hour
    
    def _create_state_key(self, co2_level, temp_level, occupancy, time_of_day):
        """Create a unique key for a state."""
        return f"{co2_level}_{temp_level}_{occupancy}_{time_of_day}"
    
    def _load_or_initialize_model(self):
        """Load existing model or initialize a new one."""
        if os.path.exists(self.model_file):
            try:
                with open(self.model_file, 'r') as f:
                    model = json.load(f)
                logger.info(f"Loaded existing Markov model from {self.model_file}")
                return model
            except Exception as e:
                logger.error(f"Error loading Markov model: {e}")
        
        # Create a new model
        model = {}
        logger.info("Initializing new Markov model")
        
        # Populate model with all state-action combinations
        for co2 in CO2Level:
            for temp in TemperatureLevel:
                for occupancy in Occupancy:
                    for time_of_day in TimeOfDay:
                        state_key = self._create_state_key(
                            co2.value, temp.value, occupancy.value, time_of_day.value
                        )
                        
                        # Initialize actions with probabilities
                        model[state_key] = {}
                        
                        for action in Action:
                            action_key = action.value
                            model[state_key][action_key] = {}
                            
                            # Initialize next state transitions
                            for next_co2 in CO2Level:
                                for next_temp in TemperatureLevel:
                                    for next_occupancy in Occupancy:
                                        for next_time in TimeOfDay:
                                            next_state = self._create_state_key(
                                                next_co2.value, 
                                                next_temp.value,
                                                next_occupancy.value, 
                                                next_time.value
                                            )
                                            
                                            # Initial probabilities - higher for staying in same state
                                            if state_key == next_state:
                                                prob = 0.6
                                            else:
                                                prob = 0.0001  # Very small probability for other transitions
                                            
                                            model[state_key][action_key][next_state] = prob
        
        # Normalize probabilities
        self._normalize_model(model)
        
        # Save the model
        try:
            with open(self.model_file, 'w') as f:
                json.dump(model, f, indent=2)
            logger.info("Saved initial Markov model")
        except Exception as e:
            logger.error(f"Error saving initial model: {e}")
        
        return model
        
    def _normalize_model(self, model):
        """Normalize transition probabilities to sum to 1."""
        for state in model:
            for action in model[state]:
                total = sum(model[state][action].values())
                if total > 0:
                    for next_state in model[state][action]:
                        model[state][action][next_state] /= total
    
    def start(self):
        """Start the Markov controller."""
        if self.thread is not None and self.thread.is_alive():
            logger.warning("Markov controller already running")
            return False
        
        self.running = True
        self.thread = threading.Thread(target=self._control_loop, daemon=True)
        self.thread.start()
        logger.info("Started Markov controller")
        return True
    
    def stop(self):
        """Stop the Markov controller."""
        self.running = False
        logger.info("Stopped Markov controller")
    
    def _control_loop(self):
        """Main control loop for Markov controller."""
        while self.running:
            try:
                # Skip if auto mode is disabled
                if not self.auto_mode:
                    time.sleep(self.scan_interval)
                    continue
                
                # Check if night mode is active
                if self._is_night_mode_active():
                    # During night mode, only allow turning off ventilation
                    current_status = self.pico_manager.get_ventilation_status()
                    if current_status:
                        logger.info("Night mode active - turning off ventilation")
                        self._execute_action(Action.TURN_OFF.value)
                    time.sleep(self.scan_interval)
                    continue
                
                # Get current state
                previous_state = self.current_state
                self.current_state = self._evaluate_state()
                
                # Skip if we couldn't determine the state
                if not self.current_state:
                    time.sleep(self.scan_interval)
                    continue
                
                # Check if minimum time has passed since last action
                current_time = datetime.now()
                time_since_last_action = float('inf')
                if self.last_action_time:
                    time_since_last_action = (current_time - self.last_action_time).total_seconds()
                
                if time_since_last_action < self.min_action_interval:
                    logger.debug(f"Skipping action change - minimum interval not reached ({time_since_last_action:.1f}s < {self.min_action_interval}s)")
                    time.sleep(self.scan_interval)
                    continue
                
                # Decide on action
                action = self._decide_action()
                
                # Execute action
                success = self._execute_action(action)
                
                if success:
                    # Update last action time
                    self.last_action = action
                    self.last_action_time = current_time
                    
                    # Update model based on previous state transition (if applicable)
                    if previous_state and self.last_action:
                        self._update_model(previous_state, self.last_action, self.current_state)
                
                # Wait for next check
                time.sleep(self.scan_interval)
                
            except Exception as e:
                logger.error(f"Error in Markov control loop: {e}")
                time.sleep(self.scan_interval)
    
    def _update_thresholds_for_occupancy(self, occupants):
        """
        Update thresholds based on current occupancy level.
        
        Args:
            occupants: Number of people currently in the room
        """
        if self.preference_manager is None:
            logger.warning("No preference manager available, using default thresholds")
            return
        
        # Log only when occupancy changes
        if self.last_applied_occupants != occupants:
            logger.info(f"Updating thresholds for {occupants} occupants")
            self.last_applied_occupants = occupants
        
        if occupants == 0:
            # Empty home - use energy-saving thresholds
            self.co2_thresholds = self.default_empty_home_co2_thresholds.copy()
            self.temp_thresholds = self.default_empty_home_temp_thresholds.copy()
            logger.info(f"Using empty home thresholds: CO2={self.co2_thresholds}, Temp={self.temp_thresholds}")
            
        else:
            # Home occupied - use compromise preferences from all registered users
            try:
                # Get all user preferences
                all_user_preferences = self.preference_manager.get_all_user_preferences()
                
                if all_user_preferences:
                    # Calculate compromise based on all registered users
                    all_user_ids = list(all_user_preferences.keys())
                    compromise = self.preference_manager.calculate_compromise_preference(all_user_ids)
                    
                    # Update CO2 thresholds
                    self.co2_thresholds = {
                        "low_max": int(compromise.co2_threshold * 0.8),  # Low threshold at 80% of compromise
                        "medium_max": compromise.co2_threshold           # Medium threshold at compromise value
                    }
                    
                    # Update temperature thresholds
                    self.temp_thresholds = {
                        "low_max": compromise.temp_min,
                        "medium_max": compromise.temp_max
                    }
                    
                    logger.info(f"Using compromise thresholds: CO2={self.co2_thresholds}, "
                                f"Temp={self.temp_thresholds}, Effectiveness={compromise.effectiveness_score:.2f}")
                    
                else:
                    # No registered users - use default thresholds
                    logger.warning("No registered users found, using default thresholds")
                    
            except Exception as e:
                logger.error(f"Error calculating compromise preferences: {e}")
                logger.warning("Falling back to default thresholds")

    def _evaluate_state(self):
        """
        Determine current state based on sensor data.
        
        Returns:
            str: State key or None if state cannot be determined
        """
        try:
            # Get current occupancy
            occupants = self.data_manager.latest_data["room"]["occupants"]
            
            # Update thresholds based on occupancy
            self._update_thresholds_for_occupancy(occupants)
            
            # Get CO2 level
            co2 = self.data_manager.latest_data["scd41"]["co2"]
            if co2 is None:
                logger.warning("Missing CO2 data")
                return None
            
            # Determine CO2 level category
            if co2 < self.co2_thresholds["low_max"]:
                co2_level = CO2Level.LOW.value
            elif co2 < self.co2_thresholds["medium_max"]:
                co2_level = CO2Level.MEDIUM.value
            else:
                co2_level = CO2Level.HIGH.value
            
            # Get temperature
            temp = self.data_manager.latest_data["scd41"]["temperature"]
            if temp is None:
                logger.warning("Missing temperature data")
                return None
                
            # Determine temperature level
            if temp < self.temp_thresholds["low_max"]:
                temp_level = TemperatureLevel.LOW.value
            elif temp < self.temp_thresholds["medium_max"]:
                temp_level = TemperatureLevel.MEDIUM.value
            else:
                temp_level = TemperatureLevel.HIGH.value
            
            # Get occupancy state
            occupancy = Occupancy.OCCUPIED.value if occupants > 0 else Occupancy.EMPTY.value
            
            # Determine time of day
            hour = datetime.now().hour
            if 5 <= hour < 12:
                time_of_day = TimeOfDay.MORNING.value
            elif 12 <= hour < 18:
                time_of_day = TimeOfDay.DAY.value
            elif 18 <= hour < 22:
                time_of_day = TimeOfDay.EVENING.value
            else:
                time_of_day = TimeOfDay.NIGHT.value
            
            # Create state key
            state_key = self._create_state_key(co2_level, temp_level, occupancy, time_of_day)
            return state_key
            
        except Exception as e:
            logger.error(f"Error evaluating state: {e}")
            return None
    
    def _decide_action(self):
        """
        Decide what action to take based on the current state.
        
        Returns:
            str: Action key
        """
        # If no current state, default to off
        if not self.current_state:
            return Action.TURN_OFF.value
        
        # Check if state exists in model
        if self.current_state not in self.transition_model:
            logger.warning(f"Unknown state: {self.current_state}")
            return Action.TURN_OFF.value
        
        # Random exploration (occasionally try random actions)
        if random.random() < self.exploration_rate:
            random_action = random.choice(list(Action)).value
            logger.info(f"Exploring random action: {random_action}")
            return random_action
        
        # Find the best action for this state (highest predicted next state value)
        best_action = None
        best_value = float('-inf')
        
        for action in self.transition_model[self.current_state]:
            # Calculate expected value of this action
            expected_value = 0
            
            for next_state, probability in self.transition_model[self.current_state][action].items():
                # Simple value function: reward CO2 level improvements
                if "_low_" in next_state:
                    state_value = 1.0
                elif "_medium_" in next_state:
                    state_value = 0.5
                else:  # high CO2
                    state_value = 0.0
                
                # Higher value for occupied states
                if "_occupied_" in next_state:
                    state_value += 0.2
                
                # Penalize energy usage
                if action == Action.TURN_ON_MAX.value:
                    state_value -= 0.4
                elif action == Action.TURN_ON_MEDIUM.value:
                    state_value -= 0.2
                elif action == Action.TURN_ON_LOW.value:
                    state_value -= 0.1
                
                expected_value += probability * state_value
            
            if expected_value > best_value:
                best_value = expected_value
                best_action = action
        
        # Reduce exploration rate over time
        self.exploration_rate *= self.random_action_decay
        
        logger.info(f"Selected action: {best_action} for state: {self.current_state}")
        return best_action
    
    def _execute_action(self, action):
        """
        Execute the selected action.
        
        Args:
            action: Action key
            
        Returns:
            bool: Success status
        """
        current_status = self.pico_manager.get_ventilation_status()
        current_speed = self.pico_manager.get_ventilation_speed()
        
        # Check if action is already in effect
        if (action == Action.TURN_OFF.value and not current_status) or \
           (action != Action.TURN_OFF.value and current_status and current_speed == action):
            return True
        
        # Execute action
        if action == Action.TURN_OFF.value:
            success = self.pico_manager.control_ventilation("off")
            if success:
                logger.info("Turned ventilation OFF")
        else:
            success = self.pico_manager.control_ventilation("on", action)
            if success:
                logger.info(f"Turned ventilation ON at {action} speed")
        
        return success
    
    def _update_model(self, previous_state, action, current_state):
        """
        Update the transition model based on observed state transition.
        
        Args:
            previous_state: Previous state key
            action: Action taken
            current_state: Resulting state key
        """
        try:
            # Check if states and action exist in model
            if previous_state not in self.transition_model:
                logger.warning(f"Unknown previous state: {previous_state}")
                return
            
            if action not in self.transition_model[previous_state]:
                logger.warning(f"Unknown action: {action}")
                return
            
            # Initialize if destination state doesn't exist
            if current_state not in self.transition_model[previous_state][action]:
                self.transition_model[previous_state][action][current_state] = 0
            
            # Update probability with learning rate
            # Increase probability of observed transition
            self.transition_model[previous_state][action][current_state] += self.learning_rate
            
            # Normalize probabilities
            total = sum(self.transition_model[previous_state][action].values())
            for state in self.transition_model[previous_state][action]:
                self.transition_model[previous_state][action][state] /= total
            
            # Save updated model periodically (every ~50 updates, randomly)
            if random.random() < 0.02:
                try:
                    with open(self.model_file, 'w') as f:
                        json.dump(self.transition_model, f, indent=2)
                    logger.debug("Saved updated Markov model")
                except Exception as e:
                    logger.error(f"Error saving model: {e}")
            
        except Exception as e:
            logger.error(f"Error updating model: {e}")
    
    def set_auto_mode(self, enabled):
        """Enable or disable automatic control."""
        self.auto_mode = enabled
        logger.info(f"Automatic control {'enabled' if enabled else 'disabled'}")
        return True
    
    def set_night_mode(self, enabled, start_hour=None, end_hour=None):
        """Configure night mode settings."""
        self.night_mode_enabled = enabled
        if start_hour is not None:
            self.night_mode_start_hour = start_hour
        if end_hour is not None:
            self.night_mode_end_hour = end_hour
        
        self._save_night_mode_settings()
        logger.info(f"Night mode {'enabled' if enabled else 'disabled'}: {self.night_mode_start_hour}:00 - {self.night_mode_end_hour}:00")
        return True
    
    def get_status(self):
        """Get controller status information."""
        # Get current occupancy for status report
        occupants = self.data_manager.latest_data["room"]["occupants"]
        
        return {
            "auto_mode": self.auto_mode,
            "co2_thresholds": self.co2_thresholds,
            "temp_thresholds": self.temp_thresholds,
            "current_state": self.current_state,
            "last_action": self.last_action,
            "exploration_rate": self.exploration_rate,
            "ventilation_status": self.pico_manager.get_ventilation_status(),
            "ventilation_speed": self.pico_manager.get_ventilation_speed(),
            "last_action_time": self.last_action_time.isoformat() if self.last_action_time else None,
            "night_mode": {
                "enabled": self.night_mode_enabled,
                "start_hour": self.night_mode_start_hour,
                "end_hour": self.night_mode_end_hour,
                "currently_active": self._is_night_mode_active()
            },
            "current_occupants": occupants,
            "active_thresholds": "empty_home" if occupants == 0 else "compromise"
        }
    
    def set_thresholds(self, co2_low_max=None, co2_medium_max=None, temp_low_max=None, temp_medium_max=None):
        """Update threshold settings."""
        if co2_low_max is not None:
            self.co2_thresholds["low_max"] = co2_low_max
        if co2_medium_max is not None:
            self.co2_thresholds["medium_max"] = co2_medium_max
        if temp_low_max is not None:
            self.temp_thresholds["low_max"] = temp_low_max
        if temp_medium_max is not None:
            self.temp_thresholds["medium_max"] = temp_medium_max
        
        logger.info(f"Updated thresholds: CO2={self.co2_thresholds}, Temp={self.temp_thresholds}")
        return True