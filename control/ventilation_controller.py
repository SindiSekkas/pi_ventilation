# control/ventilation_controller.py
"""Simple automatic ventilation controller.
Mainly used for manual testing and debugging."""
import logging
import threading
import time
import json
import os
from datetime import datetime

logger = logging.getLogger(__name__)

class VentilationController:
    """Controller for automatic ventilation based on sensor data and presence."""
    
    def __init__(self, data_manager, pico_manager, scan_interval=60):
        """
        Initialize the ventilation controller.
        
        Args:
            data_manager: Data manager instance for sensor readings
            pico_manager: Pico manager for controlling ventilation
            scan_interval: How often to check conditions (seconds)
        """
        self.data_manager = data_manager
        self.pico_manager = pico_manager
        self.scan_interval = scan_interval
        
        # Control thread
        self.running = False
        self.thread = None
        
        # Controller settings
        self.co2_thresholds = {
            "low": 800,    # Below this is considered good
            "medium": 1000, # Below this is acceptable
            "high": 1200    # Above this is poor
        }
        
        self.temp_thresholds = {
            "low": 18,     # Below this is too cold
            "medium": 20,   # Below this is cool
            "high": 25      # Above this is too warm
        }
        
        # Control state
        self.auto_mode = True
        self.last_action_time = None
        self.min_ventilation_time = 300  # Minimum time to run ventilation (seconds)
        self.min_off_time = 600  # Minimum time to leave ventilation off (seconds)
        
        # Night mode settings
        self.night_mode_enabled = True
        self.night_mode_start_hour = 23
        self.night_mode_end_hour = 7
        
        # Settings dir for saving configuration
        self.settings_dir = "data/ventilation"
        os.makedirs(self.settings_dir, exist_ok=True)
        
        # Load night mode settings
        self._load_night_mode_settings()
    
    def _load_night_mode_settings(self):
        """Load night mode settings from file."""
        night_settings_file = os.path.join(self.settings_dir, "night_mode_settings.json")
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
        night_settings_file = os.path.join(self.settings_dir, "night_mode_settings.json")
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
    
    def start(self):
        """Start the ventilation controller."""
        if self.thread is not None and self.thread.is_alive():
            logger.warning("Ventilation controller already running")
            return False
        
        self.running = True
        self.thread = threading.Thread(target=self._control_loop, daemon=True)
        self.thread.start()
        logger.info("Started ventilation controller")
        return True
    
    def stop(self):
        """Stop the ventilation controller."""
        self.running = False
        logger.info("Stopped ventilation controller")
    
    def _control_loop(self):
        """Main control loop for ventilation."""
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
                        self._turn_ventilation_off("Night mode active")
                    time.sleep(self.scan_interval)
                    continue
                
                # Get current data
                sensor_data = self._get_current_data()
                
                # Determine appropriate ventilation action
                action, speed, reason = self._determine_action(sensor_data)
                
                # Execute action if needed
                if action == "on":
                    self._turn_ventilation_on(speed, reason)
                elif action == "off":
                    self._turn_ventilation_off(reason)
                
                # Wait for next cycle
                time.sleep(self.scan_interval)
                
            except Exception as e:
                logger.error(f"Error in ventilation control loop: {e}")
                time.sleep(self.scan_interval)
    
    def _get_current_data(self):
        """Get current sensor data and system state."""
        data = {}
        
        # Get CO2, temperature, and humidity
        data["co2"] = self.data_manager.latest_data["scd41"]["co2"]
        data["temperature"] = self.data_manager.latest_data["scd41"]["temperature"]
        data["humidity"] = self.data_manager.latest_data["scd41"]["humidity"]
        
        # Get occupancy information
        data["occupants"] = self.data_manager.latest_data["room"]["occupants"]
        
        # Get current ventilation state
        data["ventilated"] = self.data_manager.latest_data["room"]["ventilated"]
        data["ventilation_speed"] = self.data_manager.latest_data["room"]["ventilation_speed"]
        
        # Calculate how long current state has been active
        if self.last_action_time:
            data["time_since_last_action"] = (datetime.now() - self.last_action_time).total_seconds()
        else:
            data["time_since_last_action"] = float('inf')  # A very long time
        
        return data
    
    def _determine_action(self, data):
        """
        Determine appropriate ventilation action based on sensor data.
        
        Args:
            data: Current sensor and system data
            
        Returns:
            tuple: (action, speed, reason)
                action: 'on', 'off', or 'maintain'
                speed: 'low', 'medium', 'max', or None
                reason: String explaining the decision
        """
        # Default response is maintain current state
        action = "maintain"
        speed = data["ventilation_speed"]
        reason = "No change needed"
        
        # If data is missing, maintain current state
        if data["co2"] is None:
            return action, speed, "Missing CO2 data"
        
        # Check minimum on/off time constraints
        if data["ventilated"] and data["time_since_last_action"] < self.min_ventilation_time:
            return "maintain", speed, f"Minimum ventilation time not reached ({data['time_since_last_action']:.0f}s < {self.min_ventilation_time}s)"
        
        if not data["ventilated"] and data["time_since_last_action"] < self.min_off_time:
            return "maintain", speed, f"Minimum off time not reached ({data['time_since_last_action']:.0f}s < {self.min_off_time}s)"
        
        # Adjust thresholds based on occupancy
        co2_threshold_adjustment = 0
        if data["occupants"] == 0:
            # Be more conservative when no one is home
            co2_threshold_adjustment = 200  # Higher threshold when empty
        
        # Decision tree based on CO2 levels
        co2 = data["co2"]
        
        # Very high CO2 - max ventilation
        if co2 > self.co2_thresholds["high"] + co2_threshold_adjustment:
            return "on", "max", f"High CO2 level: {co2} ppm > {self.co2_thresholds['high'] + co2_threshold_adjustment} ppm"
        
        # Medium-high CO2 - medium ventilation
        elif co2 > self.co2_thresholds["medium"] + co2_threshold_adjustment:
            return "on", "medium", f"Elevated CO2 level: {co2} ppm > {self.co2_thresholds['medium'] + co2_threshold_adjustment} ppm"
        
        # Slightly elevated CO2 - low ventilation
        elif co2 > self.co2_thresholds["low"] + co2_threshold_adjustment:
            return "on", "low", f"Slightly elevated CO2 level: {co2} ppm > {self.co2_thresholds['low'] + co2_threshold_adjustment} ppm"
        
        # Low CO2 - turn off ventilation
        else:
            return "off", None, f"Low CO2 level: {co2} ppm < {self.co2_thresholds['low'] + co2_threshold_adjustment} ppm"
    
    def _turn_ventilation_on(self, speed, reason):
        """Turn on ventilation at specified speed."""
        current_status = self.pico_manager.get_ventilation_status()
        current_speed = self.pico_manager.get_ventilation_speed()
        
        # If already on at desired speed, no action needed
        if current_status and current_speed == speed:
            return
        
        # Turn on ventilation at specified speed
        success = self.pico_manager.control_ventilation("on", speed)
        
        if success:
            self.last_action_time = datetime.now()
            logger.info(f"Turned ventilation ON at {speed} speed. Reason: {reason}")
        else:
            logger.error(f"Failed to turn ventilation on at {speed} speed")
    
    def _turn_ventilation_off(self, reason):
        """Turn off ventilation."""
        current_status = self.pico_manager.get_ventilation_status()
        
        # If already off, no action needed
        if not current_status:
            return
        
        # Turn off ventilation
        success = self.pico_manager.control_ventilation("off")
        
        if success:
            self.last_action_time = datetime.now()
            logger.info(f"Turned ventilation OFF. Reason: {reason}")
        else:
            logger.error("Failed to turn ventilation off")
    
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
        return {
            "auto_mode": self.auto_mode,
            "co2_thresholds": self.co2_thresholds,
            "temp_thresholds": self.temp_thresholds,
            "ventilation_status": self.pico_manager.get_ventilation_status(),
            "ventilation_speed": self.pico_manager.get_ventilation_speed(),
            "last_action_time": self.last_action_time.isoformat() if self.last_action_time else None,
            "night_mode": {
                "enabled": self.night_mode_enabled,
                "start_hour": self.night_mode_start_hour,
                "end_hour": self.night_mode_end_hour,
                "currently_active": self._is_night_mode_active()
            }
        }
    
    def set_thresholds(self, co2_low=None, co2_medium=None, co2_high=None):
        """Update CO2 threshold settings."""
        if co2_low is not None:
            self.co2_thresholds["low"] = co2_low
        if co2_medium is not None:
            self.co2_thresholds["medium"] = co2_medium
        if co2_high is not None:
            self.co2_thresholds["high"] = co2_high
        
        logger.info(f"Updated CO2 thresholds: {self.co2_thresholds}")
        return True