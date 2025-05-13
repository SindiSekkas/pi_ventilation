# control/markov_controller.py
"""Markov Decision Process based ventilation controller."""
import os
import json
import logging
import threading
import time
import random
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)

class CO2Level(Enum):
    """CO₂ concentration categories."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class TemperatureLevel(Enum):
    """Indoor temperature categories."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

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
    """Possible ventilation commands."""
    TURN_OFF = "off"
    TURN_ON_LOW = "low"
    TURN_ON_MEDIUM = "medium"
    TURN_ON_MAX = "max"


class MarkovController:
    """
    Uses a Markov Decision Process to choose ventilation settings.
    Balances air‐quality rewards and energy costs under varying states.
    """

    MIN_EXPLORATION_RATE = 0.01
    MAX_EXPLORATION_RATE = 0.5
    
    def __init__(self, data_manager, pico_manager, preference_manager=None, model_dir="data/markov", scan_interval=60, occupancy_analyzer=None, enable_exploration=True):
        """
        Initialize the Markov controller.
        
        Args:
            data_manager: Provides latest sensor readings.
            pico_manager: Interface to ventilation hardware.
            preference_manager: Manages user comfort preferences.
            model_dir: Path to store model files.
            scan_interval: Poll interval (seconds).
            occupancy_analyzer: Predicts occupancy patterns if available.
            enable_exploration: Whether to enable random actions for exploration.
        """
        self.data_manager = data_manager
        self.pico_manager = pico_manager
        self.preference_manager = preference_manager  # Store preference manager
        self.occupancy_analyzer = occupancy_analyzer  # Store occupancy analyzer
        self.model_dir = model_dir
        self.scan_interval = scan_interval
        self.enable_exploration = enable_exploration  # Control exploration behavior
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
        
        # Very energy-saving thresholds for long absence
        self.VERY_LOW_ENERGY_THRESHOLDS_CO2 = {
            "low_max": 900,
            "medium_max": 1400
        }
        
        self.VERY_LOW_ENERGY_THRESHOLDS_TEMP = {
            "low_max": 17,
            "medium_max": 27
        }
        
        # Thresholds for preparing for return
        self.PREPARE_FOR_RETURN_THRESHOLDS_CO2 = {
            "low_max": 750,
            "medium_max": 1100
        }
        
        self.PREPARE_FOR_RETURN_THRESHOLDS_TEMP = {
            "low_max": 19,
            "medium_max": 25
        }
        
        # State tracking
        self.current_state = None
        self.last_action = None
        self.last_action_time = None
        self.min_action_interval = 300  # Min time between action changes (seconds)
        
        # Control state
        self.auto_mode = True
        
        # Q-learning parameters
        self.learning_rate = 0.1  # Alpha - initial learning rate (0.01-0.5 recommended)
        self.discount_factor = 0.9  # Gamma - future reward discount (0.9-0.99 recommended)
        self.exploration_rate = 0.1  # Epsilon - exploration rate (0.1-1.0 recommended)
        
        # Decay parameters for Q-learning
        self.epsilon_decay = 0.995  # Rate at which exploration decreases (0.9-0.999 recommended)
        self.min_epsilon = 0.01  # Minimum exploration rate (0.01-0.1 recommended)
        self.alpha_decay = 0.99  # Learning rate decay (0.9-0.999 recommended)
        self.min_alpha = 0.01  # Minimum learning rate (0.01-0.1 recommended)
        
        # Night mode settings
        self.night_mode_enabled = True
        self.night_mode_start_hour = 23
        self.night_mode_end_hour = 7
        
        # Initialize with -1 to ensure first update is logged
        self.last_applied_occupants = -1 # Initialize with a value that won't match any real occupancy
        
        # Load night mode settings from file
        self._load_night_mode_settings()
        
        # Initialize Q-values and try to load from file
        self.q_values = {}
        self.load_q_values(self.model_file)
    
    def _load_night_mode_settings(self):
        """Retrieve night‐mode configuration from JSON or use defaults."""
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
        """Persist night‐mode configuration to JSON."""
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
    
    def save_q_values(self, filepath):
        """
        Save the Q-values dictionary to a JSON file.
        
        Args:
            filepath: Path where the Q-values will be saved
            
        Returns:
            bool: Success indicator
        """
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save Q-values
            with open(filepath, 'w') as f:
                json.dump(self.q_values, f, indent=2)
                
            logger.info(f"Q-values saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving Q-values to {filepath}: {e}")
            return False
    
    def load_q_values(self, filepath):
        """
        Load Q-values from a JSON file.
        
        Args:
            filepath: Path to the JSON file containing Q-values
            
        Returns:
            bool: Success indicator
        """
        if not os.path.exists(filepath):
            logger.info(f"Q-values file not found at {filepath}. Starting with empty table.")
            return False
            
        try:
            with open(filepath, 'r') as f:
                self.q_values = json.load(f)
                
            # Count loaded values for logging
            state_count = len(self.q_values)
            action_count = sum(len(actions) for actions in self.q_values.values())
            
            logger.info(f"Loaded Q-values from {filepath}: {state_count} states, {action_count} state-action pairs")
            return True
        except Exception as e:
            logger.error(f"Error loading Q-values from {filepath}: {e}")
            # Ensure q_values is initialized as empty dict if loading fails
            self.q_values = {}
            return False
    
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
    
    def _parse_state_key(self, state_key_str: str) -> dict:
        """
        Parse a state key string into its components.
        
        Args:
            state_key_str: State key string in format "co2_level_temp_level_occupancy_timeofday"
        
        Returns:
            dict: Dictionary with state components
        """
        parts = state_key_str.split('_')
        if len(parts) < 4:
            logger.warning(f"Invalid state key format: {state_key_str}")
            return {}
        
        return {
            "co2_level": parts[0],
            "temp_level": parts[1],
            "occupancy": parts[2],
            "time_of_day": parts[3] if len(parts) > 3 else "day"
        }
    
    def _get_q_value(self, state_key, action):
        """
        Safely retrieve the Q-value for a state-action pair.
        
        Args:
            state_key: State identifier
            action: Action identifier
            
        Returns:
            float: Q-value for the state-action pair (0.0 if not found)
        """
        if state_key not in self.q_values:
            return 0.0
        
        if action not in self.q_values.get(state_key, {}):
            return 0.0
        
        value = self.q_values[state_key][action]
        
        # Ensure we return a numeric value
        if isinstance(value, (int, float)):
            return value
        
        # If we have a non-numeric value, return 0.0
        return 0.0
    
    def _get_max_q_value(self, state_key):
        """
        Find the maximum Q-value across all possible actions for a state.
        
        Args:
            state_key: State identifier
            
        Returns:
            float: Maximum Q-value for the state (0.0 if state is unknown)
        """
        if state_key not in self.q_values or not self.q_values[state_key]:
            return 0.0
        
        # Extract numeric values only, skip non-numeric ones
        values = [q for q in self.q_values[state_key].values() if isinstance(q, (int, float))]
        
        # Return max of numeric values, or 0.0 if none
        return max(values) if values else 0.0

    
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
                        # For simplicity in this sprint, we'll use a default reward
                        # In a real implementation, reward should be calculated based on state transition
                        reward = 0.0  # Placeholder for now
                        self._update_q_value(previous_state, self.last_action, reward, self.current_state)
                
                # Wait for next check
                time.sleep(self.scan_interval)
                
            except Exception as e:
                logger.error(f"Error in Markov control loop: {e}")
                time.sleep(self.scan_interval)
    
    def _get_current_target_thresholds(self, occupants: int) -> tuple[dict, dict]:
        """
        Get current target thresholds based on occupancy level.
        
        Args:
            occupants: Number of people currently in the room
        
        Returns:
            tuple: (active_co2_thresholds, active_temp_thresholds)
        """
        if occupants == 0:
            # Empty home - use adaptive thresholds based on pattern analysis
            if self.occupancy_analyzer:
                expected_duration = self.occupancy_analyzer.get_expected_empty_duration(datetime.now())
                next_return = self.occupancy_analyzer.get_next_expected_return_time(datetime.now())
                
                if expected_duration and expected_duration > timedelta(hours=3):
                    # Long absence expected - use very energy-saving thresholds
                    active_co2_thr = self.VERY_LOW_ENERGY_THRESHOLDS_CO2.copy()
                    active_temp_thr = self.VERY_LOW_ENERGY_THRESHOLDS_TEMP.copy()
                    logger.debug(f"Using very energy-saving thresholds - expected absence: {expected_duration}")
                    
                elif next_return and next_return - datetime.now() < timedelta(hours=1):
                    # Return expected soon - prepare the environment
                    active_co2_thr = self.PREPARE_FOR_RETURN_THRESHOLDS_CO2.copy()
                    active_temp_thr = self.PREPARE_FOR_RETURN_THRESHOLDS_TEMP.copy()
                    logger.debug(f"Using return-prep thresholds - expected return: {next_return}")
                    
                else:
                    # Uncertain prediction - use standard empty home thresholds
                    active_co2_thr = self.default_empty_home_co2_thresholds.copy()
                    active_temp_thr = self.default_empty_home_temp_thresholds.copy()
                    logger.debug("Using standard empty home thresholds")
            else:
                # No analyzer - use standard empty home thresholds
                active_co2_thr = self.default_empty_home_co2_thresholds.copy()
                active_temp_thr = self.default_empty_home_temp_thresholds.copy()
                logger.debug("Using standard empty home thresholds")
        
        else:
            # Home occupied - use compromise preferences from all registered users
            try:
                # Get all user preferences
                all_user_preferences = self.preference_manager.get_all_user_preferences()
                
                if all_user_preferences and self.preference_manager:
                    # Calculate compromise based on all registered users
                    all_user_ids = list(all_user_preferences.keys())
                    compromise = self.preference_manager.calculate_compromise_preference(all_user_ids)
                    
                    # Update CO2 thresholds
                    active_co2_thr = {
                        "low_max": int(compromise.co2_threshold * 0.8),  # Low threshold at 80% of compromise
                        "medium_max": compromise.co2_threshold           # Medium threshold at compromise value
                    }
                    
                    # Update temperature thresholds
                    active_temp_thr = {
                        "low_max": compromise.temp_min,
                        "medium_max": compromise.temp_max
                    }
                    
                    logger.debug(f"Using compromise thresholds: CO2={active_co2_thr}, "
                                f"Temp={active_temp_thr}, Effectiveness={compromise.effectiveness_score:.2f}")
                    
                else:
                    # No registered users - use default thresholds
                    active_co2_thr = self.co2_thresholds.copy()
                    active_temp_thr = self.temp_thresholds.copy()
                    logger.debug("No registered users found, using default thresholds")
                    
            except Exception as e:
                logger.error(f"Error calculating compromise preferences: {e}")
                logger.debug("Falling back to default thresholds")
                active_co2_thr = self.co2_thresholds.copy()
                active_temp_thr = self.temp_thresholds.copy()
        
        return active_co2_thr, active_temp_thr
    
    def _update_thresholds_for_occupancy(self, occupants):
        """
        Adjust controller thresholds in place when occupancy changes.

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
            # Empty home - use adaptive thresholds based on pattern analysis
            if self.occupancy_analyzer:
                expected_duration = self.occupancy_analyzer.get_expected_empty_duration(datetime.now())
                next_return = self.occupancy_analyzer.get_next_expected_return_time(datetime.now())
                
                if expected_duration and expected_duration > timedelta(hours=3):
                    # Long absence expected - use very energy-saving thresholds
                    self.co2_thresholds = self.VERY_LOW_ENERGY_THRESHOLDS_CO2.copy()
                    self.temp_thresholds = self.VERY_LOW_ENERGY_THRESHOLDS_TEMP.copy()
                    logger.info(f"Using very energy-saving thresholds - expected absence: {expected_duration}")
                    
                elif next_return and next_return - datetime.now() < timedelta(hours=1):
                    # Return expected soon - prepare the environment
                    self.co2_thresholds = self.PREPARE_FOR_RETURN_THRESHOLDS_CO2.copy()
                    self.temp_thresholds = self.PREPARE_FOR_RETURN_THRESHOLDS_TEMP.copy()
                    logger.info(f"Using return-prep thresholds - expected return: {next_return}")
                    
                else:
                    # Uncertain prediction - use standard empty home thresholds
                    self.co2_thresholds = self.default_empty_home_co2_thresholds.copy()
                    self.temp_thresholds = self.default_empty_home_temp_thresholds.copy()
                    logger.info("Using standard empty home thresholds")
            else:
                # No analyzer - use standard empty home thresholds
                self.co2_thresholds = self.default_empty_home_co2_thresholds.copy()
                self.temp_thresholds = self.default_empty_home_temp_thresholds.copy()
                logger.info("Using standard empty home thresholds")
            
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
        Decide what action to take based on the current state using epsilon-greedy policy.
        
        Returns:
            str: Action key
        """
        if not self.current_state:
            logger.warning("Current state is None, defaulting to TURN_OFF")
            return Action.TURN_OFF.value

        # Check if occupancy analyzer predicts return soon - special case
        current_occupants = self.data_manager.latest_data["room"]["occupants"]
        active_co2_thr, active_temp_thr = self._get_current_target_thresholds(current_occupants)

        if self.occupancy_analyzer and self.auto_mode:
            next_return = self.occupancy_analyzer.get_next_expected_return_time(datetime.now())
            if next_return:
                time_until_return = next_return - datetime.now()
                if timedelta(minutes=30) <= time_until_return <= timedelta(minutes=60):
                    co2 = self.data_manager.latest_data["scd41"]["co2"]
                    if co2 and co2 > active_co2_thr.get("medium_max", 1100):
                        current_status = self.pico_manager.get_ventilation_status()
                        current_speed = self.pico_manager.get_ventilation_speed()
                        if not current_status or current_speed == Action.TURN_ON_LOW.value:
                            logger.info(f"Pre-arrival ventilation: CO2={co2}, return in {time_until_return}")
                            return Action.TURN_ON_MEDIUM.value

        # Epsilon-greedy exploration strategy
        if self.enable_exploration and random.random() < self.exploration_rate:
            # Exploration: choose a random action
            action = random.choice([a.value for a in Action])
            logger.info(f"Exploring random action: {action} (exploration_rate={self.exploration_rate:.3f})")
            return action
        
        # Exploitation: choose action with highest Q-value
        action_q_values = {}
        for action in [a.value for a in Action]:
            action_q_values[action] = self._get_q_value(self.current_state, action)
        
        if not action_q_values:
            logger.warning(f"No Q-values found for state {self.current_state}, defaulting to TURN_OFF")
            return Action.TURN_OFF.value
        
        # Find best action (with highest Q-value)
        max_q_value = max(action_q_values.values())
        best_actions = [action for action, q_value in action_q_values.items() if q_value == max_q_value]
        
        # If multiple actions have same max Q-value, choose one randomly
        best_action = random.choice(best_actions)
        
        logger.info(f"Selected action: {best_action} for state: {self.current_state} (Q-value: {max_q_value:.2f})")
        return best_action
    
    def _execute_action(self, action):
        """
        Send command to ventilation hardware if it differs from current state.

        Args:
            action: Target action key.

        Returns:
            bool: True if command succeeded or was already set.
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
    
    def _update_q_value(self, state_key, action, reward, next_state_key):
        """
        Update Q-value for a state-action pair using the Q-learning formula.
        
        Args:
            state_key: Current state
            action: Action taken
            reward: Reward received
            next_state_key: Resulting state
        """
        # Get current Q-value
        current_q = self._get_q_value(state_key, action)
        
        # Get maximum Q-value for next state
        max_next_q = self._get_max_q_value(next_state_key)
        
        # Calculate target value using Q-learning formula
        target = reward + self.discount_factor * max_next_q
        
        # Calculate TD error
        td_error = target - current_q
        
        # Ensure state_key exists in q_values dictionary
        if state_key not in self.q_values:
            self.q_values[state_key] = {}
        
        # Ensure action exists for this state
        if action not in self.q_values[state_key]:
            self.q_values[state_key][action] = 0.0
        
        # Update Q-value
        self.q_values[state_key][action] = current_q + self.learning_rate * td_error
        
        logger.debug(f"Updated Q-value for state: {state_key}, action: {action}, new value: {self.q_values[state_key][action]:.4f}")
        
        # Periodically save Q-values (very low frequency to avoid I/O overhead)
        if random.random() < 0.01:  # ~1% chance to save on each update
            self.save_q_values(self.model_file)
    
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
            "learning_rate": self.learning_rate,
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
    
    def set_thresholds(self, co2_low_max=None, co2_medium_max=None,
                       temp_low_max=None, temp_medium_max=None):
        """
        Update manual CO₂ and temperature threshold overrides.

        Args:
            co2_low_max: Upper bound for CO₂ LOW category.
            co2_medium_max: Upper bound for CO₂ MEDIUM category.
            temp_low_max: Upper bound for temperature LOW category.
            temp_medium_max: Upper bound for temperature MEDIUM category.
        """
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