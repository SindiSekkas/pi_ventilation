"""
Adaptive CO2-based Sleep Pattern Analyzer for ventilation system.
Detects sleep patterns from CO2 data and gradually refines night mode settings.
"""
import os
import json
import logging
import numpy as np
import threading
import time as time_module
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

class AdaptiveSleepAnalyzer:
    """
    Analyzes CO2 patterns to detect sleep and wake events and adapts ventilation settings.
    
    Uses CO2 concentration changes as indicators of human activity to identify
    when occupants typically fall asleep and wake up, then gradually refines
    the ventilation system's night mode settings.
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
        self.current_sim_time = None  # Added for simulation support
        
        # Storage configuration
        self.data_dir = "data/sleep_patterns"
        os.makedirs(self.data_dir, exist_ok=True)
        self.sleep_patterns_file = os.path.join(self.data_dir, "adaptive_sleep_patterns.json")
        
        # Load or initialize sleep patterns data
        self.sleep_patterns = self._load_or_initialize_patterns()
        
        # Runtime variables
        self.daily_co2_readings = []
        self.max_daily_readings = 288  # 5-minute intervals for 24 hours
        self.current_day = datetime.now().day
        
        # Detection sensitivity parameters
        self.min_co2_change_rate = 2.0  # ppm per minute threshold for activity change
        self.stability_window = 6  # Number of readings to consider for stability
        self.min_sleep_duration = 4 * 60  # Minimum sleep duration in minutes
        self.max_sleep_duration = 12 * 60  # Maximum sleep duration in minutes
        
        # Learning parameters
        self.min_confidence_threshold = 0.7  # Confidence threshold for night mode updates
        self.adjustment_limit_minutes = 15  # Maximum adjustment per detection in minutes
        self.learning_rate = 0.2  # Adaptation rate for new patterns
        self.required_detections = 3  # Detections needed before making adjustments
        
        # False positive prevention
        self.last_sleep_start_time = None
        self.last_wake_up_time = None
        self.min_sleep_event_interval = 6  # Hours between sleep detection events
        self.min_wake_event_interval = 8  # Hours between wake detection events
        self.sleep_detection_confidence_threshold = 0.75  # Threshold for sleep events
        
        # Sleep state tracking
        self.current_sleep_state = "awake"  # Current state: "awake" or "sleeping"
        self.state_changed_at = datetime.now()
        
        # Initialize daily tracking
        self._initialize_daily_tracking()
        
        # Thread control
        self.running = False
        self.thread = None

    def _load_or_initialize_patterns(self):
        """
        Load existing sleep patterns or create a new data structure.
        
        Returns:
            dict: Sleep patterns data structure
        """
        if os.path.exists(self.sleep_patterns_file):
            try:
                with open(self.sleep_patterns_file, 'r') as f:
                    patterns = json.load(f)
                logger.info(f"Loaded sleep patterns from {self.sleep_patterns_file}")
                return patterns
            except Exception as e:
                logger.error(f"Error loading sleep patterns: {e}")
        
        # Default structure for new sleep patterns
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
    
    def _initialize_daily_tracking(self, current_time_source=None):
        """
        Reset daily CO2 tracking for a new day.
        
        Args:
            current_time_source: Optional datetime source for simulation
        """
        self.daily_co2_readings = []
        current_time = current_time_source or (self.current_sim_time or datetime.now())
        self.current_day = current_time.day
    
    def save_patterns(self):
        """
        Save sleep patterns to persistent storage.
        
        Returns:
            bool: Success or failure of save operation
        """
        try:
            self.sleep_patterns["last_updated"] = datetime.now().isoformat()
            
            with open(self.sleep_patterns_file, 'w') as f:
                json.dump(self.sleep_patterns, f, indent=2)
            
            logger.debug("Saved sleep patterns")
            return True
        except Exception as e:
            logger.error(f"Error saving sleep patterns: {e}")
            return False
    
    def update_co2_data(self, current_sim_time: datetime = None):
        """
        Record latest CO2 reading and analyze patterns.
        
        Should be called regularly (approximately every 5 minutes).
        
        Args:
            current_sim_time: Optional simulation time for time-stepped simulation
            
        Returns:
            bool: Success or failure of the update operation
        """
        try:
            now = current_sim_time or datetime.now()
            
            # Update simulation time if provided
            if current_sim_time:
                self.current_sim_time = current_sim_time
            
            # Start new daily record if day has changed
            if now.day != self.current_day:
                self._process_daily_data()
                self._initialize_daily_tracking(now)
            
            # Get latest CO2 reading
            co2 = self.data_manager.latest_data["scd41"]["co2"]
            if co2 is None:
                logger.warning("No CO2 reading available")
                return False
            
            # Record the reading
            self.daily_co2_readings.append({
                "timestamp": now.isoformat(),
                "co2": co2,
                "hour": now.hour,
                "minute": now.minute
            })
            
            # Maintain maximum readings limit
            if len(self.daily_co2_readings) > self.max_daily_readings:
                self.daily_co2_readings = self.daily_co2_readings[-self.max_daily_readings:]
            
            # Analyze if we have sufficient data
            if len(self.daily_co2_readings) >= self.stability_window * 2:
                self._real_time_pattern_analysis()
            
            return True
        except Exception as e:
            logger.error(f"Error updating CO2 data: {e}")
            return False

    def get_predicted_sleep_time_for_day(self, day_of_week: int) -> Tuple[Optional[datetime], float]:
        """
        Predict sleep time with confidence level for a specific weekday.
        
        Args:
            day_of_week: Day of week (0=Monday, 6=Sunday)
            
        Returns:
            Tuple containing predicted sleep time and confidence level (0.0-1.0)
        """
        weekday_key = str(day_of_week)
        pattern = self.sleep_patterns["weekday_patterns"].get(weekday_key, {})
        
        if not pattern.get("sleep"):
            return None, 0.0
        
        # Calculate confidence from multiple factors
        detections = pattern.get("detections", 0)
        base_confidence = pattern.get("confidence", 0)
        
        # Historical observations factor
        detection_factor = min(1.0, detections / 10.0)
        
        # Recent events impact factor
        recent_events = self._get_recent_events_for_weekday(day_of_week, "sleep_start")
        recent_factor = self._calculate_recent_event_factor(recent_events, pattern["sleep"])
        
        # Time decay factor
        time_factor = self._calculate_time_decay_factor()
        
        # Historical variance factor
        variance_factor = self._calculate_variance_factor(day_of_week, "sleep")
        
        # Combined weighted confidence
        confidence = min(0.95, max(0.1, 
            base_confidence * 0.3 +
            detection_factor * 0.3 +
            recent_factor * 0.2 +
            time_factor * 0.1 +
            variance_factor * 0.1
        ))
        
        # Convert time string to datetime
        try:
            today = (self.current_sim_time or datetime.now()).date()
            time_parts = pattern["sleep"].split(":")
            sleep_time = datetime.combine(today, datetime.min.time().replace(
                hour=int(time_parts[0]),
                minute=int(time_parts[1])
            ))
            
            # Adjust if sleep time is after midnight
            if sleep_time.hour < 12:
                sleep_time += timedelta(days=1)
                
            return sleep_time, confidence
        except (ValueError, IndexError) as e:
            logger.error(f"Error parsing sleep time: {e}")
            return None, 0.0

    def get_predicted_wake_time_for_day(self, day_of_week: int) -> Tuple[Optional[datetime], float]:
        """
        Predict wake time with confidence level for a specific weekday.
        
        Args:
            day_of_week: Day of week (0=Monday, 6=Sunday)
            
        Returns:
            Tuple containing predicted wake time and confidence level (0.0-1.0)
        """
        weekday_key = str(day_of_week)
        pattern = self.sleep_patterns["weekday_patterns"].get(weekday_key, {})
        
        if not pattern.get("wake"):
            return None, 0.0
        
        # Calculate confidence from multiple factors
        detections = pattern.get("detections", 0)
        base_confidence = pattern.get("confidence", 0)
        
        # Historical observations factor
        detection_factor = min(1.0, detections / 10.0)
        
        # Recent events impact factor
        recent_events = self._get_recent_events_for_weekday(day_of_week, "wake_up")
        recent_factor = self._calculate_recent_event_factor(recent_events, pattern["wake"])
        
        # Time decay factor
        time_factor = self._calculate_time_decay_factor()
        
        # Historical variance factor
        variance_factor = self._calculate_variance_factor(day_of_week, "wake")
        
        # Combined weighted confidence
        confidence = min(0.95, max(0.1, 
            base_confidence * 0.3 +
            detection_factor * 0.3 +
            recent_factor * 0.2 +
            time_factor * 0.1 +
            variance_factor * 0.1
        ))
        
        # Convert time string to datetime
        try:
            today = (self.current_sim_time or datetime.now()).date()
            time_parts = pattern["wake"].split(":")
            wake_time = datetime.combine(today, datetime.min.time().replace(
                hour=int(time_parts[0]),
                minute=int(time_parts[1])
            ))
            
            # Adjust if wake time is in the evening (unusual)
            if wake_time.hour > 12:
                wake_time += timedelta(days=1)
                
            return wake_time, confidence
        except (ValueError, IndexError) as e:
            logger.error(f"Error parsing wake time: {e}")
            return None, 0.0

    def _get_recent_events_for_weekday(self, day_of_week: int, event_type: str, days_back: int = 7) -> list:
        """
        Get recent events matching specific weekday and type.
        
        Args:
            day_of_week: Target day of week (0-6)
            event_type: Event type to filter for
            days_back: Number of days to look back
            
        Returns:
            List of matching events
        """
        recent_events = []
        cutoff_date = (self.current_sim_time or datetime.now()) - timedelta(days=days_back)
        
        for event in self.sleep_patterns.get("detected_events", []):
            try:
                event_time = datetime.fromisoformat(event["timestamp"])
                if (event_time > cutoff_date and 
                    event["weekday"] == day_of_week and 
                    event["type"] == event_type):
                    recent_events.append(event)
            except:
                continue
        
        return recent_events

    def _calculate_recent_event_factor(self, recent_events: list, pattern_time: str) -> float:
        """
        Calculate confidence factor based on consistency of recent events.
        
        Args:
            recent_events: List of recent events
            pattern_time: Pattern time string to compare against
            
        Returns:
            Confidence factor (0.0-1.0)
        """
        if not recent_events:
            return 0.8  # Default when no recent events
        
        try:
            pattern_minutes = self._time_str_to_minutes(pattern_time)
            event_minutes = []
            
            for event in recent_events:
                event_time = datetime.fromisoformat(event["timestamp"])
                event_mins = event_time.hour * 60 + event_time.minute
                event_minutes.append(event_mins)
            
            if event_minutes:
                variance = np.var(event_minutes)
                # Higher consistency (lower variance) increases confidence
                return max(0.3, min(1.0, 1.0 - (variance / 1800)))
        except:
            pass
        
        return 0.8  # Default factor

    def _calculate_time_decay_factor(self) -> float:
        """
        Calculate confidence decay based on time since last data update.
        
        Returns:
            Confidence factor (0.0-1.0)
        """
        try:
            last_updated = datetime.fromisoformat(self.sleep_patterns.get("last_updated", datetime.now().isoformat()))
            days_since_update = ((self.current_sim_time or datetime.now()) - last_updated).days
            
            # Step decay based on elapsed time
            if days_since_update <= 1:
                return 1.0
            elif days_since_update <= 7:
                return 0.9
            elif days_since_update <= 30:
                return 0.7
            else:
                return 0.5
        except:
            return 0.8  # Default factor

    def _calculate_variance_factor(self, day_of_week: int, time_type: str) -> float:
        """
        Calculate confidence factor based on historical pattern consistency.
        
        Args:
            day_of_week: Target day of week (0-6)
            time_type: Pattern type ('sleep' or 'wake')
            
        Returns:
            Confidence factor (0.0-1.0)
        """
        day_patterns = []
        for date_str, pattern in self.sleep_patterns.get("daily_patterns", {}).items():
            try:
                date = datetime.fromisoformat(date_str.replace(' ', 'T') if ' ' in date_str else date_str).date()
                if pattern.get("weekday", -1) == day_of_week and pattern.get(time_type):
                    day_patterns.append(pattern[time_type])
            except:
                continue
        
        if len(day_patterns) < 3:
            return 0.8  # Insufficient data for variance calculation
        
        try:
            pattern_minutes = [self._time_str_to_minutes(t) for t in day_patterns]
            variance = np.var(pattern_minutes)
            
            # Lower variance indicates higher confidence
            return max(0.4, min(1.0, 1.0 - (variance / 1800)))
        except:
            return 0.8  # Default factor

    def _real_time_pattern_analysis(self):
        """
        Analyze recent CO2 data to detect sleep/wake transitions in real-time.
        
        Uses sliding windows to identify significant changes in CO2 patterns
        that correlate with activity state changes.
        """
        try:
            # Get recent readings for analysis
            recent_readings = self.daily_co2_readings[-self.stability_window*2:]
            
            if len(recent_readings) < self.stability_window * 2:
                return
                
            # Split into before and after windows
            window1 = recent_readings[:self.stability_window]
            window2 = recent_readings[self.stability_window:]
            
            # Calculate CO2 rate of change in each window
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
                    time_diff = (curr_time - prev_time).total_seconds() / 60
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
                    time_diff = (curr_time - prev_time).total_seconds() / 60
                    if time_diff > 0:
                        rate = (curr["co2"] - prev["co2"]) / time_diff
                        rates2.append(rate)
                        co2_levels2.append(curr["co2"])
                except:
                    continue
            
            if not rates1 or not rates2:
                return
            
            # Calculate statistics for comparison
            avg_rate1 = np.mean(rates1)
            avg_rate2 = np.mean(rates2)
            avg_co2_1 = np.mean(co2_levels1) if co2_levels1 else 0
            avg_co2_2 = np.mean(co2_levels2) if co2_levels2 else 0
            var1 = np.std(rates1) if len(rates1) > 1 else 0
            var2 = np.std(rates2) if len(rates2) > 1 else 0
            
            now = self.current_sim_time or datetime.now()
            current_time = now.time()
            
            # Verify enough time has passed since last events
            enough_time_passed_sleep = True
            if self.last_sleep_start_time:
                time_since_last_sleep = (now - self.last_sleep_start_time).total_seconds() / 3600
                enough_time_passed_sleep = time_since_last_sleep >= self.min_sleep_event_interval
            
            enough_time_passed_wake = True
            if self.last_wake_up_time:
                time_since_last_wake = (now - self.last_wake_up_time).total_seconds() / 3600
                enough_time_passed_wake = time_since_last_wake >= self.min_wake_event_interval
            
            # Get current night mode settings
            night_mode_info = self.controller.get_status()["night_mode"]
            night_start_hour = night_mode_info.get("start_hour", 23)
            night_end_hour = night_mode_info.get("end_hour", 7)
            
            # Determine if current time is within expected sleep time window
            expected_sleep_time_start = (night_start_hour - 2) % 24
            expected_sleep_time_end = (night_start_hour + 1) % 24
            
            is_sleep_time = False
            if expected_sleep_time_start > expected_sleep_time_end:
                is_sleep_time = (current_time.hour >= expected_sleep_time_start or 
                                current_time.hour <= expected_sleep_time_end)
            else:
                is_sleep_time = (expected_sleep_time_start <= current_time.hour <= 
                                expected_sleep_time_end)
            
            # Determine if current time is within expected wake time window
            expected_wake_time_start = (night_end_hour - 1) % 24
            expected_wake_time_end = (night_end_hour + 2) % 24
            
            is_wake_time = False
            if expected_wake_time_start > expected_wake_time_end:
                is_wake_time = (current_time.hour >= expected_wake_time_start or 
                               current_time.hour <= expected_wake_time_end)
            else:
                is_wake_time = (expected_wake_time_start <= current_time.hour <= 
                               expected_wake_time_end)
            
            # Check ventilation status to avoid false detections
            ventilation_status = self.data_manager.latest_data["room"]["ventilated"]
            
            # Sleep detection conditions
            if (self.current_sleep_state == "awake" and
                avg_rate1 > avg_rate2 + self.min_co2_change_rate and 
                var1 > var2 and
                is_sleep_time and
                enough_time_passed_sleep and
                not ventilation_status):
                
                confidence = min(1.0, abs(avg_rate1 - avg_rate2) / self.min_co2_change_rate)
                
                if confidence >= self.sleep_detection_confidence_threshold:
                    self._log_sleep_event("sleep_start", now, {
                        "rate_before": avg_rate1,
                        "rate_after": avg_rate2,
                        "var_before": var1,
                        "var_after": var2,
                        "confidence": confidence
                    })
                    self.current_sleep_state = "sleeping"
                    self.state_changed_at = now
                    self.last_sleep_start_time = now
            
            # Wake detection conditions
            elif (self.current_sleep_state == "sleeping" and
                  is_wake_time and
                  enough_time_passed_wake):
                  
                # Primary detection method - CO2 increase when ventilation is OFF
                if (not ventilation_status and 
                    avg_rate2 > avg_rate1 + self.min_co2_change_rate and
                    var2 > var1):
                    
                    confidence = min(1.0, abs(avg_rate2 - avg_rate1) / self.min_co2_change_rate)
                    
                    if confidence >= self.sleep_detection_confidence_threshold:
                        self._log_sleep_event("wake_up", now, {
                            "rate_before": avg_rate1,
                            "rate_after": avg_rate2,
                            "var_before": var1,
                            "var_after": var2,
                            "confidence": confidence,
                            "detection_method": "rate_increase"
                        })
                        self.current_sleep_state = "awake"
                        self.state_changed_at = now
                        self.last_wake_up_time = now
                
                # Alternative detection - rapid CO2 level increase
                elif (not ventilation_status and 
                      avg_co2_2 > avg_co2_1 + 50 and
                      enough_time_passed_wake):
                    
                    confidence = min(1.0, (avg_co2_2 - avg_co2_1) / 100)
                    
                    if confidence >= 0.65:
                        self._log_sleep_event("wake_up", now, {
                            "co2_before": avg_co2_1,
                            "co2_after": avg_co2_2,
                            "confidence": confidence,
                            "detection_method": "level_increase"
                        })
                        self.current_sleep_state = "awake"
                        self.state_changed_at = now
                        self.last_wake_up_time = now
            
        except Exception as e:
            logger.error(f"Error in real-time pattern analysis: {e}")
    
    def _log_sleep_event(self, event_type, timestamp, details):
        """
        Record a detected sleep or wake event.
        
        Args:
            event_type: Type of event ('sleep_start' or 'wake_up')
            timestamp: When the event was detected
            details: Additional data about the event
        """
        try:
            # Skip low confidence sleep events
            if event_type == "sleep_start" and details['confidence'] < 0.7:
                logger.debug(f"Ignoring low confidence sleep event: {details['confidence']:.2f}")
                return
                
            event = {
                "type": event_type,
                "timestamp": timestamp.isoformat(),
                "weekday": timestamp.weekday(),
                "details": details
            }
            
            # Record event and limit history size
            self.sleep_patterns["detected_events"].append(event)
            if len(self.sleep_patterns["detected_events"]) > 100:
                self.sleep_patterns["detected_events"] = self.sleep_patterns["detected_events"][-100:]
            
            logger.info(
                f"Detected potential {event_type} at {timestamp.strftime('%H:%M')} "
                f"(confidence: {details['confidence']:.2f})"
            )
            
            # Update weekday pattern data
            weekday = timestamp.weekday()
            weekday_key = str(weekday)
            time_str = timestamp.strftime("%H:%M")
            pattern = self.sleep_patterns["weekday_patterns"][weekday_key]
            
            if event_type == "sleep_start":
                if pattern["sleep"] is None:
                    pattern["sleep"] = time_str
                else:
                    # Weighted average with existing value
                    current = self._time_str_to_minutes(pattern["sleep"])
                    new = self._time_str_to_minutes(time_str)
                    updated = (current * (1 - self.learning_rate) + new * self.learning_rate)
                    pattern["sleep"] = self._minutes_to_time_str(updated)
                
                pattern["detections"] += 1
                pattern["confidence"] = max(pattern["confidence"], details["confidence"])
                
                # Update night start time if we have sufficient confidence
                if pattern["detections"] >= self.required_detections and pattern["confidence"] >= self.min_confidence_threshold:
                    self._adjust_night_start_time(timestamp, details["confidence"])
                
            elif event_type == "wake_up":
                if pattern["wake"] is None:
                    pattern["wake"] = time_str
                else:
                    # Weighted average with existing value
                    current = self._time_str_to_minutes(pattern["wake"])
                    new = self._time_str_to_minutes(time_str)
                    updated = (current * (1 - self.learning_rate) + new * self.learning_rate)
                    pattern["wake"] = self._minutes_to_time_str(updated)
                
                pattern["detections"] += 1
                pattern["confidence"] = max(pattern["confidence"], details["confidence"])
                
                # Update night end time if we have sufficient confidence
                if pattern["detections"] >= self.required_detections and pattern["confidence"] >= self.min_confidence_threshold:
                    self._adjust_night_end_time(timestamp, details["confidence"])
            
            # Persist changes
            self.save_patterns()
            
        except Exception as e:
            logger.error(f"Error logging sleep event: {e}")
    
    def _adjust_night_start_time(self, detected_time, confidence):
        """
        Update ventilation night mode start time based on detected sleep pattern.
        
        Args:
            detected_time: When sleep was detected
            confidence: Confidence level of the detection
            
        Returns:
            bool: Whether an adjustment was made
        """
        try:
            # Get current night mode settings
            night_mode_info = self.controller.get_status()["night_mode"]
            if not night_mode_info.get("enabled", False):
                logger.debug("Night mode is disabled, not adjusting start time")
                return False
            
            current_start_hour = night_mode_info.get("start_hour", 23)
            
            # Convert times to minutes for comparison
            detected_hour = detected_time.hour
            detected_minute = detected_time.minute
            detected_time_minutes = detected_hour * 60 + detected_minute
            current_start_minutes = current_start_hour * 60
            
            # Handle midnight crossover cases
            if detected_hour < 12 and current_start_hour > 12:
                detected_time_minutes += 24 * 60
            elif detected_hour > 12 and current_start_hour < 12:
                current_start_minutes += 24 * 60
            
            diff_minutes = detected_time_minutes - current_start_minutes
            
            # Skip minor differences
            if abs(diff_minutes) < 5:
                logger.debug(f"Difference too small ({diff_minutes} min), not adjusting night start time")
                return False
            
            # Limit and scale adjustment
            adjustment = max(-self.adjustment_limit_minutes, 
                           min(self.adjustment_limit_minutes, diff_minutes))
            adjustment = int(adjustment * confidence * self.learning_rate)
            
            # Calculate new time
            new_minutes = (current_start_minutes + adjustment) % (24 * 60)
            new_hour = new_minutes // 60
            
            # Skip if adjustment wouldn't change the hour
            if new_hour == current_start_hour:
                logger.debug("Adjustment too small, would result in same hour")
                return False
            
            # Record the adjustment
            self.sleep_patterns["night_mode_adjustments"].append({
                "timestamp": (self.current_sim_time or datetime.now()).isoformat(),
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
                end_hour=None  # Keep existing end hour
            )
            
            logger.info(f"Adjusted night mode start time from {current_start_hour}:00 to {new_hour}:00 based on detected sleep at {detected_time.strftime('%H:%M')}")
            return True
            
        except Exception as e:
            logger.error(f"Error adjusting night start time: {e}")
            return False
    
    def _adjust_night_end_time(self, detected_time, confidence):
        """
        Update ventilation night mode end time based on detected wake pattern.
        
        Args:
            detected_time: When waking was detected
            confidence: Confidence level of the detection
            
        Returns:
            bool: Whether an adjustment was made
        """
        try:
            # Get current night mode settings
            night_mode_info = self.controller.get_status()["night_mode"]
            if not night_mode_info.get("enabled", False):
                logger.debug("Night mode is disabled, not adjusting end time")
                return False
            
            current_end_hour = night_mode_info.get("end_hour", 7)
            
            # Convert times to minutes for comparison
            detected_hour = detected_time.hour
            detected_minute = detected_time.minute
            detected_time_minutes = detected_hour * 60 + detected_minute
            current_end_minutes = current_end_hour * 60
            
            # Handle midnight crossover cases
            if detected_hour < 12 and current_end_hour > 12:
                detected_time_minutes += 24 * 60
            elif detected_hour > 12 and current_end_hour < 12:
                current_end_minutes += 24 * 60
            
            diff_minutes = detected_time_minutes - current_end_minutes
            
            # Skip minor differences
            if abs(diff_minutes) < 5:
                logger.debug(f"Difference too small ({diff_minutes} min), not adjusting night end time")
                return False
            
            # Limit and scale adjustment
            adjustment = max(-self.adjustment_limit_minutes, 
                           min(self.adjustment_limit_minutes, diff_minutes))
            adjustment = int(adjustment * confidence * self.learning_rate)
            
            # Calculate new time
            new_minutes = (current_end_minutes + adjustment) % (24 * 60)
            new_hour = new_minutes // 60
            
            # Skip if adjustment wouldn't change the hour
            if new_hour == current_end_hour:
                logger.debug("Adjustment too small, would result in same hour")
                return False
            
            # Safety check for unreasonably early wake times
            if new_hour < 5 and current_end_hour >= 5:
                logger.warning(f"Rejecting adjustment to {new_hour}:00 as it's too early. Minimum is 5:00")
                return False
            
            # Record the adjustment
            self.sleep_patterns["night_mode_adjustments"].append({
                "timestamp": (self.current_sim_time or datetime.now()).isoformat(),
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
                start_hour=None,  # Keep existing start hour
                end_hour=new_hour
            )
            
            logger.info(f"Adjusted night mode end time from {current_end_hour}:00 to {new_hour}:00 based on detected wake up at {detected_time.strftime('%H:%M')}")
            return True
            
        except Exception as e:
            logger.error(f"Error adjusting night end time: {e}")
            return False
    
    def _process_daily_data(self):
        """
        Analyze collected daily CO2 data for sleep pattern extraction.
        Called at the end of each day to validate real-time detections.
        """
        try:
            if len(self.daily_co2_readings) < 24:
                logger.warning("Not enough CO2 readings to process daily data")
                return False
            
            # Determine date for the collected data
            try:
                first_reading = self.daily_co2_readings[0]
                data_date = datetime.fromisoformat(first_reading["timestamp"]).date().isoformat()
            except:
                data_date = (self.current_sim_time or datetime.now()).date().isoformat()
            
            # Extract and calculate CO2 change rates
            timestamps = []
            co2_values = []
            rates = []
            
            for i in range(1, len(self.daily_co2_readings)):
                prev = self.daily_co2_readings[i-1]
                curr = self.daily_co2_readings[i]
                
                try:
                    prev_time = datetime.fromisoformat(prev["timestamp"])
                    curr_time = datetime.fromisoformat(curr["timestamp"])
                    time_diff = (curr_time - prev_time).total_seconds() / 60
                    
                    if time_diff > 0 and time_diff < 30:  # Exclude large gaps
                        timestamps.append(curr_time)
                        co2_values.append(curr["co2"])
                        rate = (curr["co2"] - prev["co2"]) / time_diff
                        rates.append(rate)
                except:
                    continue
            
            if len(timestamps) < 12:
                logger.warning("Not enough valid CO2 rate calculations")
                return False
            
            # Smooth rates with moving average
            window_size = 3
            smoothed_rates = []
            
            for i in range(len(rates)):
                start = max(0, i - window_size + 1)
                end = i + 1
                window = rates[start:end]
                smoothed_rates.append(sum(window) / len(window))
            
            # Get reference night mode times
            night_mode_info = self.controller.get_status()["night_mode"]
            night_start_hour = night_mode_info.get("start_hour", 23)
            night_end_hour = night_mode_info.get("end_hour", 7)
            
            # Define search windows for sleep and wake events
            sleep_min_hour = (night_start_hour - 3) % 24
            sleep_max_hour = (night_start_hour + 3) % 24
            wake_min_hour = (night_end_hour - 3) % 24
            wake_max_hour = (night_end_hour + 3) % 24
            
            # Find potential events by analyzing rate changes
            sleep_candidates = []
            wake_candidates = []
            
            for i in range(window_size, len(timestamps) - window_size):
                before_window = smoothed_rates[i-window_size:i]
                after_window = smoothed_rates[i:i+window_size]
                
                before_avg = sum(before_window) / len(before_window)
                after_avg = sum(after_window) / len(after_window)
                
                timestamp = timestamps[i]
                hour = timestamp.hour
                
                # Check if within sleep search window (handling midnight crossing)
                in_sleep_range = False
                if sleep_min_hour > sleep_max_hour:
                    in_sleep_range = hour >= sleep_min_hour or hour <= sleep_max_hour
                else:
                    in_sleep_range = sleep_min_hour <= hour <= sleep_max_hour
                
                # Check if within wake search window (handling midnight crossing)
                in_wake_range = False
                if wake_min_hour > wake_max_hour:
                    in_wake_range = hour >= wake_min_hour or hour <= wake_max_hour
                else:
                    in_wake_range = wake_min_hour <= hour <= wake_max_hour
                
                # Detect significant rate decreases during sleep time
                if (before_avg - after_avg > self.min_co2_change_rate and in_sleep_range):
                    sleep_candidates.append({
                        "timestamp": timestamp,
                        "rate_change": before_avg - after_avg,
                        "confidence": min(1.0, (before_avg - after_avg) / self.min_co2_change_rate)
                    })
                
                # Detect significant rate increases during wake time
                elif (after_avg - before_avg > self.min_co2_change_rate and in_wake_range):
                    wake_candidates.append({
                        "timestamp": timestamp,
                        "rate_change": after_avg - before_avg,
                        "confidence": min(1.0, (after_avg - before_avg) / self.min_co2_change_rate)
                    })
            
            # Select most reliable events
            selected_sleep = max(sleep_candidates, key=lambda x: x["confidence"]) if sleep_candidates else None
            selected_wake = max(wake_candidates, key=lambda x: x["confidence"]) if wake_candidates else None
            
            # Validate the event pair
            valid_pair = False
            if selected_sleep and selected_wake:
                sleep_time = datetime.fromisoformat(selected_sleep["timestamp"].isoformat())
                wake_time = datetime.fromisoformat(selected_wake["timestamp"].isoformat())
                
                # Handle overnight sleep periods
                if wake_time < sleep_time:
                    wake_time += timedelta(days=1)
                
                duration_minutes = (wake_time - sleep_time).total_seconds() / 60
                valid_pair = (self.min_sleep_duration <= duration_minutes <= self.max_sleep_duration)
            
            # Record validated sleep pattern
            if valid_pair:
                sleep_time_str = selected_sleep["timestamp"].strftime("%H:%M")
                wake_time_str = selected_wake["timestamp"].strftime("%H:%M")
                weekday = selected_sleep["timestamp"].weekday()
                
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
                
                self.save_patterns()
                return True
            else:
                logger.info(f"No valid sleep pattern detected for {data_date}")
                return False
                
        except Exception as e:
            logger.error(f"Error processing daily data: {e}")
            return False
    
    def _time_str_to_minutes(self, time_str):
        """
        Convert time string to minutes since midnight.
        
        Args:
            time_str: Time string in format "HH:MM"
            
        Returns:
            int: Minutes since midnight
        """
        try:
            hours, minutes = map(int, time_str.split(':'))
            return hours * 60 + minutes
        except:
            return 0
    
    def _minutes_to_time_str(self, minutes):
        """
        Convert minutes since midnight to time string.
        
        Args:
            minutes: Minutes since midnight
            
        Returns:
            str: Time string in format "HH:MM"
        """
        minutes = int(minutes)
        hours = (minutes // 60) % 24
        mins = minutes % 60
        return f"{hours:02d}:{mins:02d}"
    
    def get_sleep_pattern_summary(self):
        """
        Create a summary of detected sleep patterns and settings.
        
        Returns:
            dict: Summary of sleep patterns and confidence levels
        """
        try:
            summary = {
                "weekday_patterns": {},
                "recent_events": [],
                "recent_adjustments": [],
                "confidence_levels": {},
                "current_night_mode": {}
            }
            
            # Include current night mode settings
            night_mode_info = self.controller.get_status()["night_mode"]
            summary["current_night_mode"] = {
                "enabled": night_mode_info.get("enabled", False),
                "start": f"{night_mode_info.get('start_hour', 23)}:00",
                "end": f"{night_mode_info.get('end_hour', 7)}:00",
                "active": night_mode_info.get("currently_active", False)
            }
            
            # Summarize weekday patterns
            for day_key, pattern in self.sleep_patterns["weekday_patterns"].items():
                day_name = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][int(day_key)]
                
                if pattern["sleep"] and pattern["wake"]:
                    sleep_time, sleep_confidence = self.get_predicted_sleep_time_for_day(int(day_key))
                    wake_time, wake_confidence = self.get_predicted_wake_time_for_day(int(day_key))
                    
                    summary["weekday_patterns"][day_name] = {
                        "sleep": pattern["sleep"],
                        "wake": pattern["wake"],
                        "sleep_confidence": sleep_confidence,
                        "wake_confidence": wake_confidence,
                        "detections": pattern["detections"]
                    }
                    summary["confidence_levels"][day_name] = max(sleep_confidence, wake_confidence)
            
            # Include recent events
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
            
            # Include recent adjustments
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
            
    def start(self):
        """
        Start the sleep pattern analyzer in a background thread.
        
        Returns:
            bool: Success or failure
        """
        # If in simulation mode, don't start the thread
        if self.current_sim_time is not None:
            logger.warning("Adaptive sleep analyzer is in simulation mode, not starting analysis thread")
            return False
            
        if self.thread is not None and self.thread.is_alive():
            logger.warning("Adaptive sleep analyzer already running")
            return False
            
        self.running = True
        self.thread = threading.Thread(target=self._analysis_loop, daemon=True)
        self.thread.start()
        logger.info("Started adaptive sleep analyzer")
        return True
        
    def stop(self):
        """
        Stop the sleep pattern analyzer.
        
        Returns:
            bool: Success or failure
        """
        self.running = False
        logger.info("Stopped adaptive sleep analyzer")
        return True
        
    def _analysis_loop(self):
        """
        Main execution loop that runs in background thread.
        Updates and analyzes CO2 data at regular intervals.
        """
        try:
            while self.running:
                # Process new CO2 data
                self.update_co2_data()
                
                # Sleep in short intervals to allow clean shutdown
                for _ in range(100):
                    if not self.running:
                        break
                    time_module.sleep(3)
        except Exception as e:
            logger.error(f"Error in adaptive sleep analyzer loop: {e}")