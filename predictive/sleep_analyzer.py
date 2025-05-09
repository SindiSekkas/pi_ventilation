"""
Adaptive CO2-based Sleep Pattern Analyzer for ventilation system.
Works with user-provided night mode settings and gradually refines them.
"""
import os
import json
import logging
import numpy as np
from datetime import datetime, timedelta, time
from collections import defaultdict

logger = logging.getLogger(__name__)

class AdaptiveSleepAnalyzer:
    """
    Analyzes CO2 patterns to detect sleep and wake events,
    then gradually adapts user-provided night mode settings.
    """
    
    def __init__(self, data_manager, controller):
        """
        Initialize the adaptive sleep analyzer.
        
        Args:
            data_manager: Interface to access sensor data
            controller: Ventilation controller with night mode settings
        """
        self.data_manager = data_manager
        self.controller = controller
        
        # Directory for storing sleep pattern data
        self.data_dir = "data/sleep_patterns"
        os.makedirs(self.data_dir, exist_ok=True)
        self.sleep_patterns_file = os.path.join(self.data_dir, "adaptive_sleep_patterns.json")
        
        # Load or initialize sleep patterns data
        self.sleep_patterns = self._load_or_initialize_patterns()
        
        # Runtime variables
        self.daily_co2_readings = []
        self.max_daily_readings = 288  # 5-minute intervals for 24 hours
        self.current_day = datetime.now().day
        
        # Algorithm parameters
        self.min_co2_change_rate = 2.0  # ppm per minute - threshold for activity change
        self.stability_window = 6  # How many readings to consider for stability
        self.min_sleep_duration = 4 * 60  # 4 hours in minutes
        self.max_sleep_duration = 12 * 60  # 12 hours in minutes
        
        # Confidence and adjustment parameters
        self.min_confidence_threshold = 0.7  # Minimum confidence to update night mode
        self.adjustment_limit_minutes = 15  # Maximum minutes to adjust per detection
        self.learning_rate = 0.2  # How quickly to adjust to new patterns
        self.required_detections = 3  # Minimum detections before making adjustments
        
        # Anti-false positive parameters
        self.last_sleep_start_time = None
        self.last_wake_up_time = None
        self.min_sleep_event_interval = 6  # Hours between sleep detection events
        self.min_wake_event_interval = 8  # Hours between wake detection events
        self.sleep_detection_confidence_threshold = 0.75  # Higher threshold for sleep events
        
        # Sleep state tracking
        self.current_sleep_state = "awake"  # "awake" or "sleeping"
        self.state_changed_at = datetime.now()
        
        # Initialize the daily recording
        self._initialize_daily_tracking()

    def _load_or_initialize_patterns(self):
        """Load existing sleep patterns or initialize a new data structure."""
        if os.path.exists(self.sleep_patterns_file):
            try:
                with open(self.sleep_patterns_file, 'r') as f:
                    patterns = json.load(f)
                logger.info(f"Loaded sleep patterns from {self.sleep_patterns_file}")
                return patterns
            except Exception as e:
                logger.error(f"Error loading sleep patterns: {e}")
        
        # Initialize new sleep patterns structure
        patterns = {
            "version": 1.0,
            "last_updated": datetime.now().isoformat(),
            "daily_patterns": {},
            "weekday_patterns": {
                "0": {"sleep": None, "wake": None, "confidence": 0, "detections": 0},
                "1": {"sleep": None, "wake": None, "confidence": 0, "detections": 0},
                "2": {"sleep": None, "wake": None, "confidence": 0, "detections": 0},
                "3": {"sleep": None, "wake": None, "confidence": 0, "detections": 0},
                "4": {"sleep": None, "wake": None, "confidence": 0, "detections": 0},
                "5": {"sleep": None, "wake": None, "confidence": 0, "detections": 0},
                "6": {"sleep": None, "wake": None, "confidence": 0, "detections": 0}
            },
            "detected_events": [],
            "night_mode_adjustments": []
        }
        
        logger.info("Initialized new adaptive sleep patterns structure")
        return patterns
    
    def _initialize_daily_tracking(self):
        """Initialize or reset the daily CO2 tracking."""
        self.daily_co2_readings = []
        self.current_day = datetime.now().day
    
    def save_patterns(self):
        """Save sleep patterns to file."""
        try:
            # Update last updated timestamp
            self.sleep_patterns["last_updated"] = datetime.now().isoformat()
            
            with open(self.sleep_patterns_file, 'w') as f:
                json.dump(self.sleep_patterns, f, indent=2)
            
            logger.debug("Saved sleep patterns")
            return True
        except Exception as e:
            logger.error(f"Error saving sleep patterns: {e}")
            return False
    
    def update_co2_data(self):
        """
        Update CO2 data with latest reading.
        This should be called regularly, e.g., every 5 minutes.
        """
        try:
            now = datetime.now()
            
            # Check if day has changed, if so start a new daily record
            if now.day != self.current_day:
                self._process_daily_data()
                self._initialize_daily_tracking()
            
            # Get latest CO2 reading
            co2 = self.data_manager.latest_data["scd41"]["co2"]
            if co2 is None:
                logger.warning("No CO2 reading available")
                return False
            
            # Add to daily readings
            self.daily_co2_readings.append({
                "timestamp": now.isoformat(),
                "co2": co2,
                "hour": now.hour,
                "minute": now.minute
            })
            
            # Trim if needed
            if len(self.daily_co2_readings) > self.max_daily_readings:
                self.daily_co2_readings = self.daily_co2_readings[-self.max_daily_readings:]
            
            # Analyze in real-time if we have enough data
            if len(self.daily_co2_readings) >= self.stability_window * 2:
                self._real_time_pattern_analysis()
            
            return True
        except Exception as e:
            logger.error(f"Error updating CO2 data: {e}")
            return False
    
    def _real_time_pattern_analysis(self):
        """
        Perform real-time analysis of CO2 patterns.
        Detects significant changes that might indicate sleep/wake transitions.
        """
        try:
            # Get the most recent readings
            recent_readings = self.daily_co2_readings[-self.stability_window*2:]
            
            # Skip if we don't have enough data
            if len(recent_readings) < self.stability_window * 2:
                return
                
            # Split into two windows
            window1 = recent_readings[:self.stability_window]
            window2 = recent_readings[self.stability_window:]
            
            # Calculate average rate of change in each window (ppm per minute)
            rates1 = []
            rates2 = []
            co2_levels1 = []
            co2_levels2 = []
            
            for i in range(1, len(window1)):
                prev = window1[i-1]
                curr = window1[i]
                try:
                    prev_time = datetime.fromisoformat(prev["timestamp"])
                    curr_time = datetime.fromisoformat(curr["timestamp"])
                    time_diff = (curr_time - prev_time).total_seconds() / 60  # minutes
                    if time_diff > 0:
                        rate = (curr["co2"] - prev["co2"]) / time_diff
                        rates1.append(rate)
                        co2_levels1.append(curr["co2"])
                except:
                    continue
            
            for i in range(1, len(window2)):
                prev = window2[i-1]
                curr = window2[i]
                try:
                    prev_time = datetime.fromisoformat(prev["timestamp"])
                    curr_time = datetime.fromisoformat(curr["timestamp"])
                    time_diff = (curr_time - prev_time).total_seconds() / 60  # minutes
                    if time_diff > 0:
                        rate = (curr["co2"] - prev["co2"]) / time_diff
                        rates2.append(rate)
                        co2_levels2.append(curr["co2"])
                except:
                    continue
            
            # Skip if we don't have enough rate data
            if not rates1 or not rates2:
                return
            
            # Calculate average rates and levels
            avg_rate1 = np.mean(rates1)
            avg_rate2 = np.mean(rates2)
            avg_co2_1 = np.mean(co2_levels1) if co2_levels1 else 0
            avg_co2_2 = np.mean(co2_levels2) if co2_levels2 else 0
            
            # Calculate variability (standard deviation)
            var1 = np.std(rates1) if len(rates1) > 1 else 0
            var2 = np.std(rates2) if len(rates2) > 1 else 0
            
            # Get current time
            now = datetime.now()
            current_time = now.time()
            
            # Check if enough time has passed since last sleep event
            enough_time_passed_sleep = True
            if self.last_sleep_start_time:
                time_since_last_sleep = (now - self.last_sleep_start_time).total_seconds() / 3600  # hours
                enough_time_passed_sleep = time_since_last_sleep >= self.min_sleep_event_interval
            
            # Check if enough time has passed since last wake event
            enough_time_passed_wake = True
            if self.last_wake_up_time:
                time_since_last_wake = (now - self.last_wake_up_time).total_seconds() / 3600  # hours
                enough_time_passed_wake = time_since_last_wake >= self.min_wake_event_interval
            
            # Get current night mode settings from controller
            night_mode_info = self.controller.get_status()["night_mode"]
            night_start_hour = night_mode_info.get("start_hour", 23)
            night_end_hour = night_mode_info.get("end_hour", 7)
            
            # Only detect sleep patterns around expected sleep time
            # Allow 2 hours before and 1 hour after configured night start time
            expected_sleep_time_start = (night_start_hour - 2) % 24
            expected_sleep_time_end = (night_start_hour + 1) % 24
            
            # Handle cases where the range crosses midnight
            is_sleep_time = False
            if expected_sleep_time_start > expected_sleep_time_end:
                is_sleep_time = (current_time.hour >= expected_sleep_time_start or 
                                current_time.hour <= expected_sleep_time_end)
            else:
                is_sleep_time = (expected_sleep_time_start <= current_time.hour <= 
                                expected_sleep_time_end)
            
            # Only detect wake patterns around expected wake time
            # Allow 1 hour before and 2 hours after configured night end time
            expected_wake_time_start = (night_end_hour - 1) % 24
            expected_wake_time_end = (night_end_hour + 2) % 24
            
            # Handle cases where the range crosses midnight
            is_wake_time = False
            if expected_wake_time_start > expected_wake_time_end:
                is_wake_time = (current_time.hour >= expected_wake_time_start or 
                               current_time.hour <= expected_wake_time_end)
            else:
                is_wake_time = (expected_wake_time_start <= current_time.hour <= 
                               expected_wake_time_end)
            
            # Get ventilation status to avoid false detection during ventilation changes
            ventilation_status = self.data_manager.latest_data["room"]["ventilated"]
            
            # For detecting sleep start:
            # - We should be in awake state
            # - CO2 rate should decrease (people breathing less when falling asleep)
            # - Variability should decrease (less movement)
            # - It should be appropriate sleep time
            # - Enough time has passed since last sleep detection
            # - Ventilation should not have recently changed
            
            # Sleep detection - CO2 decreasing, variability decreasing, around night start time
            if (self.current_sleep_state == "awake" and
                avg_rate1 > avg_rate2 + self.min_co2_change_rate and 
                var1 > var2 and
                is_sleep_time and
                enough_time_passed_sleep and
                not ventilation_status):
                
                # Calculate confidence for this event
                confidence = min(1.0, abs(avg_rate1 - avg_rate2) / self.min_co2_change_rate)
                
                # Only log high confidence sleep events
                if confidence >= self.sleep_detection_confidence_threshold:
                    # This indicates going to sleep
                    self._log_sleep_event("sleep_start", now, {
                        "rate_before": avg_rate1,
                        "rate_after": avg_rate2,
                        "var_before": var1,
                        "var_after": var2,
                        "confidence": confidence
                    })
                    # Update state tracking
                    self.current_sleep_state = "sleeping"
                    self.state_changed_at = now
                    # Update last sleep event time
                    self.last_sleep_start_time = now
            
            # For detecting wake up:
            # - We should be in sleeping state
            # - CO2 rate should increase (people breathing more when waking up)
            # - It should be appropriate wake time
            # - Enough time has passed since last wake detection
            # - Check for significant CO2 increase when ventilation is OFF
            
            # Wake detection - CO2 increasing significantly, around night end time
            elif (self.current_sleep_state == "sleeping" and
                  is_wake_time and
                  enough_time_passed_wake):
                  
                # Primary detection method - significant CO2 increase when ventilation is OFF
                if (not ventilation_status and 
                    avg_rate2 > avg_rate1 + self.min_co2_change_rate and
                    var2 > var1):
                    
                    # Calculate confidence for this event
                    confidence = min(1.0, abs(avg_rate2 - avg_rate1) / self.min_co2_change_rate)
                    
                    # Only log high confidence wake events
                    if confidence >= self.sleep_detection_confidence_threshold:
                        self._log_sleep_event("wake_up", now, {
                            "rate_before": avg_rate1,
                            "rate_after": avg_rate2,
                            "var_before": var1,
                            "var_after": var2,
                            "confidence": confidence,
                            "detection_method": "rate_increase"
                        })
                        # Update state tracking
                        self.current_sleep_state = "awake"
                        self.state_changed_at = now
                        # Update last wake event time
                        self.last_wake_up_time = now
                
                # Alternative detection - rapid CO2 increase after ventilation turned OFF
                # This catches cases where someone wakes up, turns ventilation ON, then OFF
                elif (not ventilation_status and 
                      avg_co2_2 > avg_co2_1 + 50 and  # Rapid CO2 increase of at least 50 ppm
                      enough_time_passed_wake):
                    
                    confidence = min(1.0, (avg_co2_2 - avg_co2_1) / 100)
                    
                    if confidence >= 0.65:  # Slightly lower threshold for this method
                        self._log_sleep_event("wake_up", now, {
                            "co2_before": avg_co2_1,
                            "co2_after": avg_co2_2,
                            "confidence": confidence,
                            "detection_method": "level_increase"
                        })
                        # Update state tracking
                        self.current_sleep_state = "awake"
                        self.state_changed_at = now
                        # Update last wake event time
                        self.last_wake_up_time = now
            
        except Exception as e:
            logger.error(f"Error in real-time pattern analysis: {e}")
    
    def _log_sleep_event(self, event_type, timestamp, details):
        """
        Log a detected sleep-related event for further analysis.
        
        Args:
            event_type (str): Type of event ('sleep_start' or 'wake_up')
            timestamp (datetime): When the event was detected
            details (dict): Additional details about the event
        """
        try:
            # For sleep_start events with low confidence - don't register them
            if event_type == "sleep_start" and details['confidence'] < 0.7:
                logger.debug(f"Ignoring low confidence sleep event: {details['confidence']:.2f}")
                return
                
            event = {
                "type": event_type,
                "timestamp": timestamp.isoformat(),
                "weekday": timestamp.weekday(),
                "details": details
            }
            
            # Add to event history
            self.sleep_patterns["detected_events"].append(event)
            
            # Trim history if needed (keep only last 100 events)
            if len(self.sleep_patterns["detected_events"]) > 100:
                self.sleep_patterns["detected_events"] = self.sleep_patterns["detected_events"][-100:]
            
            # Log the event
            logger.info(
                f"Detected potential {event_type} at {timestamp.strftime('%H:%M')} "
                f"(confidence: {details['confidence']:.2f})"
            )
            
            # Update weekday pattern
            weekday = timestamp.weekday()
            weekday_key = str(weekday)
            
            # Format time as string
            time_str = timestamp.strftime("%H:%M")
            
            # Update pattern for this weekday
            pattern = self.sleep_patterns["weekday_patterns"][weekday_key]
            
            # Update sleep or wake time and increment detection count
            if event_type == "sleep_start":
                if pattern["sleep"] is None:
                    pattern["sleep"] = time_str
                else:
                    # Average with existing value, weighted by confidence
                    current = self._time_str_to_minutes(pattern["sleep"])
                    new = self._time_str_to_minutes(time_str)
                    updated = (current * (1 - self.learning_rate) + new * self.learning_rate)
                    pattern["sleep"] = self._minutes_to_time_str(updated)
                
                pattern["detections"] += 1
                pattern["confidence"] = max(pattern["confidence"], details["confidence"])
                
                # If we have enough detections, update night start time
                if pattern["detections"] >= self.required_detections and pattern["confidence"] >= self.min_confidence_threshold:
                    self._adjust_night_start_time(timestamp, details["confidence"])
                
            elif event_type == "wake_up":
                if pattern["wake"] is None:
                    pattern["wake"] = time_str
                else:
                    # Average with existing value, weighted by confidence
                    current = self._time_str_to_minutes(pattern["wake"])
                    new = self._time_str_to_minutes(time_str)
                    updated = (current * (1 - self.learning_rate) + new * self.learning_rate)
                    pattern["wake"] = self._minutes_to_time_str(updated)
                
                pattern["detections"] += 1
                pattern["confidence"] = max(pattern["confidence"], details["confidence"])
                
                # If we have enough detections, update night end time
                if pattern["detections"] >= self.required_detections and pattern["confidence"] >= self.min_confidence_threshold:
                    self._adjust_night_end_time(timestamp, details["confidence"])
            
            # Save patterns after each event
            self.save_patterns()
            
        except Exception as e:
            logger.error(f"Error logging sleep event: {e}")
    
    def _adjust_night_start_time(self, detected_time, confidence):
        """
        Gradually adjust night mode start time based on detected sleep start.
        
        Args:
            detected_time: Detected sleep start time
            confidence: Confidence level of detection
        """
        try:
            # Get current night mode settings
            night_mode_info = self.controller.get_status()["night_mode"]
            if not night_mode_info.get("enabled", False):
                logger.debug("Night mode is disabled, not adjusting start time")
                return False
            
            current_start_hour = night_mode_info.get("start_hour", 23)
            
            # Convert detected time to hour
            detected_hour = detected_time.hour
            detected_minute = detected_time.minute
            detected_time_minutes = detected_hour * 60 + detected_minute
            
            # Convert current setting to minutes
            current_start_minutes = current_start_hour * 60
            
            # Calculate difference (in minutes)
            # Handle cases that cross midnight
            if detected_hour < 12 and current_start_hour > 12:
                # Detected time is after midnight, current time is before
                detected_time_minutes += 24 * 60
            elif detected_hour > 12 and current_start_hour < 12:
                # Current time is after midnight, detected time is before
                current_start_minutes += 24 * 60
            
            diff_minutes = detected_time_minutes - current_start_minutes
            
            # Only adjust if difference is significant but not extreme
            if abs(diff_minutes) < 5:
                logger.debug(f"Difference too small ({diff_minutes} min), not adjusting night start time")
                return False
            
            # Limit adjustment to prevent drastic changes
            adjustment = max(-self.adjustment_limit_minutes, 
                           min(self.adjustment_limit_minutes, diff_minutes))
            
            # Scale adjustment by confidence
            adjustment = int(adjustment * confidence * self.learning_rate)
            
            # Calculate new hour, keeping it in 0-23 range
            new_minutes = (current_start_minutes + adjustment) % (24 * 60)
            new_hour = new_minutes // 60
            
            # Don't adjust if the new hour is the same
            if new_hour == current_start_hour:
                logger.debug("Adjustment too small, would result in same hour")
                return False
            
            # Log the adjustment
            self.sleep_patterns["night_mode_adjustments"].append({
                "timestamp": datetime.now().isoformat(),
                "type": "start_time",
                "from": current_start_hour,
                "to": new_hour,
                "detected_time": detected_time.strftime("%H:%M"),
                "confidence": confidence,
                "adjustment_minutes": adjustment
            })
            
            # Apply the adjustment
            self.controller.set_night_mode(
                enabled=night_mode_info.get("enabled", True),
                start_hour=new_hour,
                end_hour=None  # Don't change end hour
            )
            
            logger.info(f"Adjusted night mode start time from {current_start_hour}:00 to {new_hour}:00 based on detected sleep at {detected_time.strftime('%H:%M')}")
            return True
            
        except Exception as e:
            logger.error(f"Error adjusting night start time: {e}")
            return False
    
    def _adjust_night_end_time(self, detected_time, confidence):
        """
        Gradually adjust night mode end time based on detected wake up.
        
        Args:
            detected_time: Detected wake up time
            confidence: Confidence level of detection
        """
        try:
            # Get current night mode settings
            night_mode_info = self.controller.get_status()["night_mode"]
            if not night_mode_info.get("enabled", False):
                logger.debug("Night mode is disabled, not adjusting end time")
                return False
            
            current_end_hour = night_mode_info.get("end_hour", 7)
            
            # Convert detected time to hour
            detected_hour = detected_time.hour
            detected_minute = detected_time.minute
            detected_time_minutes = detected_hour * 60 + detected_minute
            
            # Convert current setting to minutes
            current_end_minutes = current_end_hour * 60
            
            # Calculate difference (in minutes)
            # Handle cases that cross midnight
            if detected_hour < 12 and current_end_hour > 12:
                # Detected time is after midnight, current time is before
                detected_time_minutes += 24 * 60
            elif detected_hour > 12 and current_end_hour < 12:
                # Current time is after midnight, detected time is before
                current_end_minutes += 24 * 60
            
            diff_minutes = detected_time_minutes - current_end_minutes
            
            # Only adjust if difference is significant but not extreme
            if abs(diff_minutes) < 5:
                logger.debug(f"Difference too small ({diff_minutes} min), not adjusting night end time")
                return False
            
            # Limit adjustment to prevent drastic changes
            adjustment = max(-self.adjustment_limit_minutes, 
                           min(self.adjustment_limit_minutes, diff_minutes))
            
            # Scale adjustment by confidence
            adjustment = int(adjustment * confidence * self.learning_rate)
            
            # Calculate new hour, keeping it in 0-23 range
            new_minutes = (current_end_minutes + adjustment) % (24 * 60)
            new_hour = new_minutes // 60
            
            # Don't adjust if the new hour is the same
            if new_hour == current_end_hour:
                logger.debug("Adjustment too small, would result in same hour")
                return False
            
            # Additional Safety: Ensure we don't set wake-up time too early
            if new_hour < 5 and current_end_hour >= 5:
                logger.warning(f"Rejecting adjustment to {new_hour}:00 as it's too early. Minimum is 5:00")
                return False
            
            # Log the adjustment
            self.sleep_patterns["night_mode_adjustments"].append({
                "timestamp": datetime.now().isoformat(),
                "type": "end_time",
                "from": current_end_hour,
                "to": new_hour,
                "detected_time": detected_time.strftime("%H:%M"),
                "confidence": confidence,
                "adjustment_minutes": adjustment
            })
            
            # Apply the adjustment
            self.controller.set_night_mode(
                enabled=night_mode_info.get("enabled", True),
                start_hour=None,  # Don't change start hour
                end_hour=new_hour
            )
            
            logger.info(f"Adjusted night mode end time from {current_end_hour}:00 to {new_hour}:00 based on detected wake up at {detected_time.strftime('%H:%M')}")
            return True
            
        except Exception as e:
            logger.error(f"Error adjusting night end time: {e}")
            return False
    
    def _process_daily_data(self):
        """
        Process the collected daily data to extract sleep patterns.
        This is called once per day (when the day changes).
        """
        try:
            if len(self.daily_co2_readings) < 24:  # Need at least 24 readings (2 hours at 5-min intervals)
                logger.warning("Not enough CO2 readings to process daily data")
                return False
            
            # Analysis logic is similar to the real-time version but 
            # processes the entire day of data at once.
            # This can be valuable for confirming and refining patterns.
            
            # Get the date for the data
            try:
                first_reading = self.daily_co2_readings[0]
                data_date = datetime.fromisoformat(first_reading["timestamp"]).date().isoformat()
            except:
                data_date = datetime.now().date().isoformat()
            
            # Extract CO2 rates of change
            timestamps = []
            co2_values = []
            rates = []
            
            for i in range(1, len(self.daily_co2_readings)):
                prev = self.daily_co2_readings[i-1]
                curr = self.daily_co2_readings[i]
                
                try:
                    prev_time = datetime.fromisoformat(prev["timestamp"])
                    curr_time = datetime.fromisoformat(curr["timestamp"])
                    time_diff = (curr_time - prev_time).total_seconds() / 60  # minutes
                    
                    if time_diff > 0 and time_diff < 30:  # Skip large gaps
                        timestamps.append(curr_time)
                        co2_values.append(curr["co2"])
                        rate = (curr["co2"] - prev["co2"]) / time_diff
                        rates.append(rate)
                except:
                    continue
            
            if len(timestamps) < 12:  # Need at least 12 valid rate calculations
                logger.warning("Not enough valid CO2 rate calculations")
                return False
            
            # Calculate moving average of rates
            window_size = 3  # 3 readings (15 minutes at 5-min intervals)
            smoothed_rates = []
            
            for i in range(len(rates)):
                start = max(0, i - window_size + 1)
                end = i + 1
                window = rates[start:end]
                smoothed_rates.append(sum(window) / len(window))
            
            # Find potential sleep and wake times
            # Sleep: Transition from higher to lower rate in evening
            # Wake: Transition from lower to higher rate in morning
            sleep_candidates = []
            wake_candidates = []
            
            # Get night mode settings from controller
            night_mode_info = self.controller.get_status()["night_mode"]
            night_start_hour = night_mode_info.get("start_hour", 23)
            night_end_hour = night_mode_info.get("end_hour", 7)
            
            # Search for sleep onset around night start time ±3 hours
            sleep_min_hour = (night_start_hour - 3) % 24
            sleep_max_hour = (night_start_hour + 3) % 24
            
            # Search for wake around night end time ±3 hours
            wake_min_hour = (night_end_hour - 3) % 24
            wake_max_hour = (night_end_hour + 3) % 24
            
            for i in range(window_size, len(timestamps) - window_size):
                before_window = smoothed_rates[i-window_size:i]
                after_window = smoothed_rates[i:i+window_size]
                
                before_avg = sum(before_window) / len(before_window)
                after_avg = sum(after_window) / len(after_window)
                
                timestamp = timestamps[i]
                hour = timestamp.hour
                
                # Check if hour is in sleep search range, handling midnight crossing
                in_sleep_range = False
                if sleep_min_hour > sleep_max_hour:  # Range crosses midnight
                    in_sleep_range = hour >= sleep_min_hour or hour <= sleep_max_hour
                else:
                    in_sleep_range = sleep_min_hour <= hour <= sleep_max_hour
                
                # Check if hour is in wake search range, handling midnight crossing
                in_wake_range = False
                if wake_min_hour > wake_max_hour:  # Range crosses midnight
                    in_wake_range = hour >= wake_min_hour or hour <= wake_max_hour
                else:
                    in_wake_range = wake_min_hour <= hour <= wake_max_hour
                
                # Significant rate decrease in evening (around night start time)
                if (before_avg - after_avg > self.min_co2_change_rate and in_sleep_range):
                    sleep_candidates.append({
                        "timestamp": timestamp,
                        "rate_change": before_avg - after_avg,
                        "confidence": min(1.0, (before_avg - after_avg) / self.min_co2_change_rate)
                    })
                
                # Significant rate increase in morning (around night end time)
                elif (after_avg - before_avg > self.min_co2_change_rate and in_wake_range):
                    wake_candidates.append({
                        "timestamp": timestamp,
                        "rate_change": after_avg - before_avg,
                        "confidence": min(1.0, (after_avg - before_avg) / self.min_co2_change_rate)
                    })
            
            # Select best candidates based on confidence
            selected_sleep = max(sleep_candidates, key=lambda x: x["confidence"]) if sleep_candidates else None
            selected_wake = max(wake_candidates, key=lambda x: x["confidence"]) if wake_candidates else None
            
            # Validate the pair makes sense
            valid_pair = False
            if selected_sleep and selected_wake:
                sleep_time = datetime.fromisoformat(selected_sleep["timestamp"].isoformat())
                wake_time = datetime.fromisoformat(selected_wake["timestamp"].isoformat())
                
                # Handle overnight case
                if wake_time < sleep_time:
                    wake_time += timedelta(days=1)
                
                # Calculate duration in minutes
                duration_minutes = (wake_time - sleep_time).total_seconds() / 60
                
                # Check if duration is reasonable
                valid_pair = (self.min_sleep_duration <= duration_minutes <= self.max_sleep_duration)
            
            # Store the validated pattern for this day
            if valid_pair:
                sleep_time_str = selected_sleep["timestamp"].strftime("%H:%M")
                wake_time_str = selected_wake["timestamp"].strftime("%H:%M")
                
                weekday = selected_sleep["timestamp"].weekday()
                
                # Store in daily patterns
                self.sleep_patterns["daily_patterns"][data_date] = {
                    "sleep": sleep_time_str,
                    "wake": wake_time_str,
                    "weekday": weekday,
                    "sleep_confidence": selected_sleep["confidence"],
                    "wake_confidence": selected_wake["confidence"]
                }
                
                logger.info(
                    f"Processed daily sleep pattern for {data_date}: "
                    f"Sleep at {sleep_time_str} (conf: {selected_sleep['confidence']:.2f}), "
                    f"Wake at {wake_time_str} (conf: {selected_wake['confidence']:.2f})"
                )
                
                # Save patterns
                self.save_patterns()
                
                return True
            else:
                logger.info(f"No valid sleep pattern detected for {data_date}")
                return False
                
        except Exception as e:
            logger.error(f"Error processing daily data: {e}")
            return False
    
    def _time_str_to_minutes(self, time_str):
        """Convert a time string (HH:MM) to minutes since midnight."""
        try:
            hours, minutes = map(int, time_str.split(':'))
            return hours * 60 + minutes
        except:
            return 0
    
    def _minutes_to_time_str(self, minutes):
        """Convert minutes since midnight to a time string (HH:MM)."""
        minutes = int(minutes)
        hours = (minutes // 60) % 24
        mins = minutes % 60
        return f"{hours:02d}:{mins:02d}"
    
    def get_sleep_pattern_summary(self):
        """Get a summary of detected sleep patterns and night mode adjustments."""
        try:
            summary = {
                "weekday_patterns": {},
                "recent_events": [],
                "recent_adjustments": [],
                "confidence_levels": {},
                "current_night_mode": {}
            }
            
            # Get current night mode settings
            night_mode_info = self.controller.get_status()["night_mode"]
            summary["current_night_mode"] = {
                "enabled": night_mode_info.get("enabled", False),
                "start": f"{night_mode_info.get('start_hour', 23)}:00",
                "end": f"{night_mode_info.get('end_hour', 7)}:00",
                "active": night_mode_info.get("currently_active", False)
            }
            
            # Format weekday patterns
            for day_key, pattern in self.sleep_patterns["weekday_patterns"].items():
                day_name = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][int(day_key)]
                
                if pattern["sleep"] and pattern["wake"]:
                    summary["weekday_patterns"][day_name] = {
                        "sleep": pattern["sleep"],
                        "wake": pattern["wake"],
                        "confidence": f"{pattern['confidence']:.2f}",
                        "detections": pattern["detections"]
                    }
                    summary["confidence_levels"][day_name] = pattern["confidence"]
            
            # Get recent events (last 5)
            events = self.sleep_patterns["detected_events"][-5:]
            for event in events:
                try:
                    event_time = datetime.fromisoformat(event["timestamp"]).strftime("%Y-%m-%d %H:%M")
                    summary["recent_events"].append({
                        "time": event_time,
                        "type": event["type"],
                        "confidence": f"{event['details']['confidence']:.2f}"
                    })
                except:
                    continue
            
            # Get recent adjustments (last 5)
            adjustments = self.sleep_patterns.get("night_mode_adjustments", [])[-5:]
            for adj in adjustments:
                try:
                    adj_time = datetime.fromisoformat(adj["timestamp"]).strftime("%Y-%m-%d")
                    summary["recent_adjustments"].append({
                        "date": adj_time,
                        "type": adj["type"],
                        "from": f"{adj['from']}:00",
                        "to": f"{adj['to']}:00",
                        "detected_time": adj["detected_time"]
                    })
                except:
                    continue
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting sleep pattern summary: {e}")
            return {"error": str(e)}