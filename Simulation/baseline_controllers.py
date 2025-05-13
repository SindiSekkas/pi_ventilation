# simulation/baseline_controllers.py
"""
Baseline ventilation control algorithms for comparison.
Provides simple rule-based, timer-based, and oracle controllers.
"""
import logging
from datetime import datetime, time
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class BaselineController:
    """Base class for all baseline controllers."""
    
    def __init__(self, name: str = "BaseController"):
        """
        Initialize base controller.
        
        Args:
            name: Controller name for logging
        """
        self.name = name
        self.last_action = "off"
    
    def decide_action(self, 
                       sensor_data: Dict[str, Any],
                       preferences: Optional[Dict[str, Any]] = None) -> str:
        """
        Decide ventilation action based on current data.
        
        Args:
            sensor_data: Current sensor readings
            preferences: Optional user preferences
            
        Returns:
            str: Ventilation action ('off', 'low', 'medium', 'max')
        """
        # Base class returns a default action - should be overridden
        return "off"
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get controller status information.
        
        Returns:
            dict: Status information
        """
        return {
            "name": self.name,
            "last_action": self.last_action
        }


class RuleBasedController(BaselineController):
    """
    Simple threshold-based ventilation controller.
    
    Decides ventilation actions based on CO2 thresholds.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize rule-based controller.
        
        Args:
            config: Controller configuration
        """
        super().__init__(name="RuleBasedController")
        
        # Default thresholds
        self.config = {
            "CO2_LOW_THRESHOLD": 800,   # Below this, turn off
            "CO2_MEDIUM_THRESHOLD": 1000,  # Above this, use medium
            "CO2_HIGH_THRESHOLD": 1200,  # Above this, use max
            "MIN_ON_TIME_MINUTES": 10,  # Minimum time to keep ventilation on
            "MIN_OFF_TIME_MINUTES": 10   # Minimum time to keep ventilation off
        }
        
        # Override defaults with provided config
        if config:
            self.config.update(config)
        
        self.last_action_time = datetime.now()
    
    def decide_action(self, 
                      sensor_data: Dict[str, Any],
                      preferences: Optional[Dict[str, Any]] = None) -> str:
        """
        Decide ventilation action based on CO2 thresholds.
        
        Args:
            sensor_data: Current sensor readings
            preferences: Optional user preferences
            
        Returns:
            str: Ventilation action ('off', 'low', 'medium', 'max')
        """
        # Extract current CO2 level
        co2_level = sensor_data.get("co2_ppm", 0)
        
        # Get time since last action change
        current_time = datetime.now()
        time_since_last_action = (current_time - self.last_action_time).total_seconds() / 60
        
        # Check if minimum time has passed
        if (self.last_action == "off" and 
            time_since_last_action < self.config["MIN_OFF_TIME_MINUTES"]):
            return "off"
        
        if (self.last_action != "off" and 
            time_since_last_action < self.config["MIN_ON_TIME_MINUTES"]):
            return self.last_action
        
        # Apply threshold rules
        if co2_level <= self.config["CO2_LOW_THRESHOLD"]:
            action = "off"
        elif co2_level <= self.config["CO2_MEDIUM_THRESHOLD"]:
            action = "low"
        elif co2_level <= self.config["CO2_HIGH_THRESHOLD"]:
            action = "medium"
        else:
            action = "max"
        
        # Update last action if it changed
        if action != self.last_action:
            self.last_action_time = current_time
            self.last_action = action
            logger.info(f"RuleBasedController changed action to {action} based on CO2={co2_level}ppm")
        
        return action


class TimerBasedController(BaselineController):
    """
    Timer-based ventilation controller.
    
    Operates ventilation on a fixed schedule, regardless of sensor readings.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize timer-based controller.
        
        Args:
            config: Controller configuration
        """
        super().__init__(name="TimerBasedController")
        
        # Default schedule
        self.config = {
            # Format: {day_of_week: [(start_hour, end_hour, action), ...]}
            # 0=Monday, 6=Sunday
            "schedule": {
                0: [(7, 9, "low"), (17, 19, "medium")],  # Monday
                1: [(7, 9, "low"), (17, 19, "medium")],  # Tuesday
                2: [(7, 9, "low"), (17, 19, "medium")],  # Wednesday
                3: [(7, 9, "low"), (17, 19, "medium")],  # Thursday
                4: [(7, 9, "low"), (17, 19, "medium")],  # Friday
                5: [(10, 12, "low"), (17, 19, "low")],   # Saturday
                6: [(10, 12, "low"), (17, 19, "low")]    # Sunday
            }
        }
        
        # Override defaults with provided config
        if config:
            self.config.update(config)
    
    def decide_action(self, 
                      sensor_data: Dict[str, Any],
                      preferences: Optional[Dict[str, Any]] = None) -> str:
        """
        Decide ventilation action based on schedule.
        
        Args:
            sensor_data: Current sensor readings (not used)
            preferences: Optional user preferences (not used)
            
        Returns:
            str: Ventilation action ('off', 'low', 'medium', 'max')
        """
        # Get current time
        current_time = datetime.now()
        day_of_week = current_time.weekday()  # 0=Monday, 6=Sunday
        current_hour = current_time.hour
        
        # Check if current time falls within a scheduled period
        day_schedule = self.config["schedule"].get(day_of_week, [])
        
        for start_hour, end_hour, action in day_schedule:
            if start_hour <= current_hour < end_hour:
                self.last_action = action
                return action
        
        # Default to off if no scheduled period is active
        self.last_action = "off"
        return "off"


class OracleRuleBasedController(RuleBasedController):
    """
    Enhanced rule-based controller with perfect occupancy knowledge.
    
    Uses the same CO2 threshold approach as RuleBasedController but
    turns off ventilation when the space is empty.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize oracle rule-based controller.
        
        Args:
            config: Controller configuration
        """
        super().__init__(config)
        self.name = "OracleRuleBasedController"
    
    def decide_action(self, 
                      sensor_data: Dict[str, Any],
                      preferences: Optional[Dict[str, Any]] = None) -> str:
        """
        Decide ventilation action based on CO2 thresholds and occupancy.
        
        Args:
            sensor_data: Current sensor readings and occupancy
            preferences: Optional user preferences
            
        Returns:
            str: Ventilation action ('off', 'low', 'medium', 'max')
        """
        # First check occupancy - turn off if empty
        occupants = sensor_data.get("occupants", 1)
        
        if occupants == 0:
            if self.last_action != "off":
                logger.info("OracleRuleBasedController turned off ventilation - space is empty")
                self.last_action = "off"
                self.last_action_time = datetime.now()
            return "off"
        
        # If occupied, use normal rule-based logic
        return super().decide_action(sensor_data, preferences)