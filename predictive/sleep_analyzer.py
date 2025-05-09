"""
Enhanced CO2-based Sleep Pattern Analyzer for ventilation system.
Includes user confirmation, baseline analysis, and predictive models.
"""
import os
import json
import logging
import numpy as np
from datetime import datetime, timedelta, time
from collections import defaultdict

logger = logging.getLogger(__name__)

class EnhancedCO2SleepAnalyzer:
    """
    Enhanced sleep pattern analyzer with user confirmation and predictive capabilities.
    """
    
    def __init__(self, data_manager, controller, data_dir="data/predictive"):
        """
        Initialize the enhanced CO2 sleep analyzer.
        
        Args:
            data_manager: Interface to access sensor data
            controller: Ventilation controller to apply sleep patterns
            data_dir: Directory for storing sleep pattern data
        """
        self.data_manager = data_manager
        self.controller = controller
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.sleep_patterns_file = os.path.join(data_dir, "enhanced_sleep_patterns.json")
        self.baseline_file = os.path.join(data_dir, "co2_baseline.json")
        self.user_confirmations_file = os.path.join(data_dir, "user_confirmations.json")
        
        # Load or initialize data structures
        self.sleep_patterns = self._load_or_initialize_patterns()
        self.baseline_data = self._load_baseline_data()
        self.user_confirmations = self._load_user_confirmations()
        
        # Runtime variables
        self.daily_co2_readings = []
        self.max_daily_readings = 288  # 5-minute intervals for 24 hours
        self.current_day = datetime.now().day
        
        # Enhanced algorithm parameters
        self.min_co2_change_rate = 2.0  # Base threshold
        self.adaptive_factor = 0.7  # Factor for baseline adjustment
        self.stability_window = 6
        self.min_sleep_duration = 4 * 60
        self.max_sleep_duration = 12 * 60
        self.min_confidence_threshold = 0.6
        
        # Anti-false positive parameters
        self.last_sleep_start_time = None
        self.last_wake_up_time = None
        self.min_sleep_event_interval = 6
        self.min_wake_event_interval = 8
        self.sleep_detection_confidence_threshold = 0.7
        
        # Sleep state tracking
        self.current_sleep_state = "awake"
        self.state_changed_at = datetime.now()
        
        # Predictive model parameters
        self.pre_sleep_ventilation_minutes = 15
        self.pre_wake_ventilation_minutes = 15
        self.prediction_lookahead_days = 7
        
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
        
        patterns = {
            "version": 2.0,
            "last_updated": datetime.now().isoformat(),
            "daily_patterns": {},
            "weekday_patterns": {
                str(i): {
                    "sleep": None, 
                    "wake": None, 
                    "confidence": 0,
                    "daytype": "workday" if i < 5 else "weekend",
                    "confirmed_count": 0,
                    "total_count": 0
                } for i in range(7)
            },
            "detected_events": [],
            "prediction_model": {
                "sleep_time_variance": {},
                "wake_time_variance": {},
                "recent_accuracy": 0.0
            }
        }
        
        logger.info("Initialized new enhanced sleep patterns structure")
        return patterns
    
    def _load_baseline_data(self):
        """Load CO2 baseline data for adaptive thresholds."""
        if os.path.exists(self.baseline_file):
            try:
                with open(self.baseline_file, 'r') as f:
                    baseline = json.load(f)
                logger.info(f"Loaded baseline data from {self.baseline_file}")
                return baseline
            except Exception as e:
                logger.error(f"Error loading baseline data: {e}")
        
        baseline = {
            "awake_baseline": {
                "morning": [],
                "afternoon": [],
                "evening": []
            },
            "sleep_baseline": {
                "early_night": [],
                "late_night": [],
                "early_morning": []
            },
            "last_updated": datetime.now().isoformat(),
            "adaptive_thresholds": {
                "sleep_detection": self.min_co2_change_rate,
                "wake_detection": self.min_co2_change_rate
            }
        }
        
        logger.info("Initialized new baseline data structure")
        return baseline
    
    def _load_user_confirmations(self):
        """Load user confirmation data for events."""
        if os.path.exists(self.user_confirmations_file):
            try:
                with open(self.user_confirmations_file, 'r') as f:
                    confirmations = json.load(f)
                logger.info(f"Loaded user confirmations from {self.user_confirmations_file}")
                return confirmations
            except Exception as e:
                logger.error(f"Error loading user confirmations: {e}")
        
        confirmations = {
            "pending_confirmations": [],
            "confirmed_events": [],
            "rejected_events": [],
            "last_updated": datetime.now().isoformat()
        }
        
        logger.info("Initialized new user confirmations structure")
        return confirmations
    
    def save_all_data(self):
        """Save all data structures to files."""
        try:
            # Update timestamps
            self.sleep_patterns["last_updated"] = datetime.now().isoformat()
            self.baseline_data["last_updated"] = datetime.now().isoformat()
            self.user_confirmations["last_updated"] = datetime.now().isoformat()
            
            # Save files
            with open(self.sleep_patterns_file, 'w') as f:
                json.dump(self.sleep_patterns, f, indent=2)
            
            with open(self.baseline_file, 'w') as f:
                json.dump(self.baseline_data, f, indent=2)
            
            with open(self.user_confirmations_file, 'w') as f:
                json.dump(self.user_confirmations, f, indent=2)
            
            logger.debug("Saved all data structures")
            return True
        except Exception as e:
            logger.error(f"Error saving data: {e}")
            return False
    
    def _initialize_daily_tracking(self):
        """Initialize or reset the daily CO2 tracking."""
        self.daily_co2_readings = []
        self.current_day = datetime.now().day
    
    def update_baseline_data(self, co2_value, current_time=None):
        """
        Update baseline CO2 data for adaptive thresholds.
        
        Args:
            co2_value: Current CO2 reading
            current_time: Current time (default: now)
        """
        now = current_time or datetime.now()
        hour = now.hour
        
        try:
            # Determine time period and sleep state
            if self.current_sleep_state == "awake":
                if 6 <= hour < 12:
                    period = "morning"
                elif 12 <= hour < 18:
                    period = "afternoon"
                else:
                    period = "evening"
                
                self.baseline_data["awake_baseline"][period].append({
                    "co2": co2_value,
                    "timestamp": now.isoformat()
                })
                
                # Keep only last 100 readings per period
                if len(self.baseline_data["awake_baseline"][period]) > 100:
                    self.baseline_data["awake_baseline"][period] = self.baseline_data["awake_baseline"][period][-100:]
            
            else:  # sleeping
                if 22 <= hour or hour < 2:
                    period = "early_night"
                elif 2 <= hour < 5:
                    period = "late_night"
                else:
                    period = "early_morning"
                
                self.baseline_data["sleep_baseline"][period].append({
                    "co2": co2_value,
                    "timestamp": now.isoformat()
                })
                
                # Keep only last 50 readings per period
                if len(self.baseline_data["sleep_baseline"][period]) > 50:
                    self.baseline_data["sleep_baseline"][period] = self.baseline_data["sleep_baseline"][period][-50:]
            
            # Update adaptive thresholds periodically
            if len(self.baseline_data["awake_baseline"]["evening"]) > 10:
                self._update_adaptive_thresholds()
            
        except Exception as e:
            logger.error(f"Error updating baseline data: {e}")
    
    def _update_adaptive_thresholds(self):
        """Update detection thresholds based on baseline data."""
        try:
            # Calculate average awake CO2 levels
            awake_co2_values = []
            for period in ["morning", "afternoon", "evening"]:
                if len(self.baseline_data["awake_baseline"][period]) > 0:
                    recent_values = [item["co2"] for item in self.baseline_data["awake_baseline"][period][-10:]]
                    awake_co2_values.extend(recent_values)
            
            # Calculate average sleep CO2 levels
            sleep_co2_values = []
            for period in ["early_night", "late_night", "early_morning"]:
                if len(self.baseline_data["sleep_baseline"][period]) > 0:
                    recent_values = [item["co2"] for item in self.baseline_data["sleep_baseline"][period][-10:]]
                    sleep_co2_values.extend(recent_values)
            
            if len(awake_co2_values) > 5 and len(sleep_co2_values) > 5:
                avg_awake = np.mean(awake_co2_values)
                avg_sleep = np.mean(sleep_co2_values)
                
                # Adjust thresholds based on typical difference
                typical_difference = avg_awake - avg_sleep
                
                # Adaptive threshold calculation
                self.baseline_data["adaptive_thresholds"]["sleep_detection"] = max(
                    self.min_co2_change_rate, 
                    typical_difference * self.adaptive_factor * 0.5
                )
                
                self.baseline_data["adaptive_thresholds"]["wake_detection"] = max(
                    self.min_co2_change_rate,
                    typical_difference * self.adaptive_factor * 0.3
                )
                
                logger.debug(f"Updated adaptive thresholds: sleep={self.baseline_data['adaptive_thresholds']['sleep_detection']:.2f}, wake={self.baseline_data['adaptive_thresholds']['wake_detection']:.2f}")
                
        except Exception as e:
            logger.error(f"Error updating adaptive thresholds: {e}")
    
    def update_co2_data(self):
        """
        Update CO2 data with latest reading and run analysis.
        This should be called regularly, e.g., every 5 minutes.
        """
        try:
            now = datetime.now()
            
            # Check if day has changed
            if now.day != self.current_day:
                self._process_daily_data()
                self._initialize_daily_tracking()
            
            # Get latest CO2 reading
            co2 = self.data_manager.latest_data["scd41"]["co2"]
            if co2 is None:
                logger.warning("No CO2 reading available")
                return False
            
            # Update baseline data
            self.update_baseline_data(co2, now)
            
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
            
            # Check for predictions
            self._check_predictive_actions()
            
            return True
        except Exception as e:
            logger.error(f"Error updating CO2 data: {e}")
            return False
    
    def _real_time_pattern_analysis(self):
        """Enhanced real-time analysis of CO2 patterns with adaptive thresholds."""
        try:
            # Get the most recent readings
            recent_readings = self.daily_co2_readings[-self.stability_window*2:]
            
            if len(recent_readings) < self.stability_window * 2:
                return
                
            # Split into two windows
            window1 = recent_readings[:self.stability_window]
            window2 = recent_readings[self.stability_window:]
            
            # Calculate average rate of change in each window
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
            
            # Calculate average rates and levels
            avg_rate1 = np.mean(rates1)
            avg_rate2 = np.mean(rates2)
            avg_co2_1 = np.mean(co2_levels1) if co2_levels1 else 0
            avg_co2_2 = np.mean(co2_levels2) if co2_levels2 else 0
            
            # Calculate variability
            var1 = np.std(rates1) if len(rates1) > 1 else 0
            var2 = np.std(rates2) if len(rates2) > 1 else 0
            
            # Get current time and adaptive thresholds
            now = datetime.now()
            current_time = now.time()
            
            sleep_threshold = self.baseline_data["adaptive_thresholds"]["sleep_detection"]
            wake_threshold = self.baseline_data["adaptive_thresholds"]["wake_detection"]
            
            # Check if enough time has passed since last events
            enough_time_passed_sleep = True
            if self.last_sleep_start_time:
                time_since_last_sleep = (now - self.last_sleep_start_time).total_seconds() / 3600
                enough_time_passed_sleep = time_since_last_sleep >= self.min_sleep_event_interval
            
            enough_time_passed_wake = True
            if self.last_wake_up_time:
                time_since_last_wake = (now - self.last_wake_up_time).total_seconds() / 3600
                enough_time_passed_wake = time_since_last_wake >= self.min_wake_event_interval
            
            # Time checks
            is_sleep_time = (current_time >= time(19, 0) or current_time <= time(4, 0))
            is_wake_time = time(5, 0) <= current_time <= time(14, 0)
            
            # Get ventilation status
            ventilation_status = self.data_manager.latest_data["room"]["ventilated"]
            
            # Sleep detection with adaptive threshold
            if (self.current_sleep_state == "awake" and
                avg_rate1 > avg_rate2 + sleep_threshold and 
                var1 > var2 and
                is_sleep_time and
                enough_time_passed_sleep and
                not ventilation_status):
                
                confidence = min(1.0, abs(avg_rate1 - avg_rate2) / sleep_threshold)
                
                if confidence >= self.sleep_detection_confidence_threshold:
                    self._log_sleep_event("sleep_start", now, {
                        "rate_before": avg_rate1,
                        "rate_after": avg_rate2,
                        "var_before": var1,
                        "var_after": var2,
                        "confidence": confidence,
                        "threshold_used": sleep_threshold
                    })
                    self.current_sleep_state = "sleeping"
                    self.state_changed_at = now
                    self.last_sleep_start_time = now
            
            # Wake detection with adaptive threshold
            elif (self.current_sleep_state == "sleeping" and
                  is_wake_time and
                  enough_time_passed_wake):
                  
                if (not ventilation_status and 
                    avg_rate2 > avg_rate1 + wake_threshold and
                    var2 > var1):
                    
                    confidence = min(1.0, abs(avg_rate2 - avg_rate1) / wake_threshold)
                    
                    if confidence >= self.sleep_detection_confidence_threshold:
                        self._log_sleep_event("wake_up", now, {
                            "rate_before": avg_rate1,
                            "rate_after": avg_rate2,
                            "var_before": var1,
                            "var_after": var2,
                            "confidence": confidence,
                            "threshold_used": wake_threshold,
                            "detection_method": "rate_increase"
                        })
                        self.current_sleep_state = "awake"
                        self.state_changed_at = now
                        self.last_wake_up_time = now
                
                elif (not ventilation_status and 
                      avg_co2_2 > avg_co2_1 + 50 and
                      enough_time_passed_wake):
                    
                    confidence = min(1.0, (avg_co2_2 - avg_co2_1) / 100)
                    
                    if confidence >= 0.6:
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
        Log a detected sleep-related event and create pending confirmation.
        
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
                "details": details,
                "event_id": f"{event_type}_{timestamp.strftime('%Y%m%d_%H%M%S')}"
            }
            
            # Add to detected events
            self.sleep_patterns["detected_events"].append(event)
            
            # Create pending confirmation
            pending_confirmation = {
                "event_id": event["event_id"],
                "event": event,
                "created_at": datetime.now().isoformat(),
                "status": "pending"
            }
            
            self.user_confirmations["pending_confirmations"].append(pending_confirmation)
            
            # Trim history if needed
            if len(self.sleep_patterns["detected_events"]) > 100:
                self.sleep_patterns["detected_events"] = self.sleep_patterns["detected_events"][-100:]
            
            # Log the event
            logger.info(
                f"Detected potential {event_type} at {timestamp.strftime('%H:%M')} "
                f"(confidence: {details['confidence']:.2f}) - Awaiting user confirmation"
            )
            
            # Save data
            self.save_all_data()
            
        except Exception as e:
            logger.error(f"Error logging sleep event: {e}")
    
    def confirm_event(self, event_id, confirmed=True):
        """
        Process user confirmation for a detected event.
        
        Args:
            event_id (str): ID of the event to confirm
            confirmed (bool): Whether event is confirmed or rejected
        
        Returns:
            bool: Success status
        """
        try:
            # Find pending confirmation
            pending_confirmation = None
            for i, confirmation in enumerate(self.user_confirmations["pending_confirmations"]):
                if confirmation["event_id"] == event_id:
                    pending_confirmation = confirmation
                    self.user_confirmations["pending_confirmations"].pop(i)
                    break
            
            if not pending_confirmation:
                logger.warning(f"Event {event_id} not found in pending confirmations")
                return False
            
            # Process confirmation
            confirmation_record = {
                "event_id": event_id,
                "event": pending_confirmation["event"],
                "confirmed": confirmed,
                "confirmed_at": datetime.now().isoformat()
            }
            
            if confirmed:
                self.user_confirmations["confirmed_events"].append(confirmation_record)
                
                # Update patterns with confirmed event
                self._update_patterns_with_confirmed_event(pending_confirmation["event"])
                
                logger.info(f"Event {event_id} confirmed by user")
            else:
                self.user_confirmations["rejected_events"].append(confirmation_record)
                logger.info(f"Event {event_id} rejected by user")
            
            # Save data
            self.save_all_data()
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing confirmation: {e}")
            return False
    
    def _update_patterns_with_confirmed_event(self, event):
        """Update sleep patterns with a user-confirmed event."""
        try:
            event_type = event["type"]
            timestamp = datetime.fromisoformat(event["timestamp"])
            weekday = timestamp.weekday()
            confidence = event["details"]["confidence"]
            
            # Update weekday pattern
            weekday_key = str(weekday)
            pattern = self.sleep_patterns["weekday_patterns"][weekday_key]
            
            # Track confirmation statistics
            pattern["total_count"] = pattern.get("total_count", 0) + 1
            pattern["confirmed_count"] = pattern.get("confirmed_count", 0) + 1
            
            # Update pattern times with higher weight for confirmed events
            if event_type == "sleep_start":
                time_str = timestamp.strftime("%H:%M")
                if pattern["sleep"] is None:
                    pattern["sleep"] = time_str
                    pattern["confidence"] = confidence
                else:
                    # Use exponential moving average with higher weight for confirmed events
                    alpha = 0.5  # Higher learning rate for confirmed events
                    current_minutes = self._time_str_to_minutes(pattern["sleep"])
                    new_minutes = self._time_str_to_minutes(time_str)
                    updated_minutes = (1 - alpha) * current_minutes + alpha * new_minutes
                    pattern["sleep"] = self._minutes_to_time_str(updated_minutes)
                    
                    # Update confidence
                    pattern["confidence"] = 0.5 * pattern["confidence"] + 0.5 * confidence
                    
            elif event_type == "wake_up":
                time_str = timestamp.strftime("%H:%M")
                if pattern["wake"] is None:
                    pattern["wake"] = time_str
                    pattern["confidence"] = confidence
                else:
                    # Use exponential moving average with higher weight for confirmed events
                    alpha = 0.5  # Higher learning rate for confirmed events
                    current_minutes = self._time_str_to_minutes(pattern["wake"])
                    new_minutes = self._time_str_to_minutes(time_str)
                    updated_minutes = (1 - alpha) * current_minutes + alpha * new_minutes
                    pattern["wake"] = self._minutes_to_time_str(updated_minutes)
                    
                    # Update confidence
                    pattern["confidence"] = 0.5 * pattern["confidence"] + 0.5 * confidence
            
            # Update prediction model accuracy
            self._update_prediction_model_accuracy()
            
        except Exception as e:
            logger.error(f"Error updating patterns with confirmed event: {e}")
    
    def _update_prediction_model_accuracy(self):
        """Update the accuracy metrics of the prediction model."""
        try:
            # Calculate recent prediction accuracy based on confirmed events
            recent_confirmations = self.user_confirmations["confirmed_events"][-20:]
            
            if len(recent_confirmations) > 0:
                confirmed_count = len([c for c in recent_confirmations if c["confirmed"]])
                accuracy = confirmed_count / len(recent_confirmations)
                
                self.sleep_patterns["prediction_model"]["recent_accuracy"] = accuracy
                
                # Update variance data for better predictions
                self._update_variance_data(recent_confirmations)
            
        except Exception as e:
            logger.error(f"Error updating prediction model accuracy: {e}")
    
    def _update_variance_data(self, recent_confirmations):
        """Update variance data for prediction models."""
        try:
            sleep_times = []
            wake_times = []
            
            for confirmation in recent_confirmations:
                if confirmation["confirmed"]:
                    event = confirmation["event"]
                    timestamp = datetime.fromisoformat(event["timestamp"])
                    weekday = timestamp.weekday()
                    
                    if event["type"] == "sleep_start":
                        sleep_times.append((weekday, timestamp.hour * 60 + timestamp.minute))
                    elif event["type"] == "wake_up":
                        wake_times.append((weekday, timestamp.hour * 60 + timestamp.minute))
            
            # Calculate variance by weekday
            for weekday in range(7):
                weekday_key = str(weekday)
                
                # Sleep time variance
                weekday_sleep_times = [t[1] for t in sleep_times if t[0] == weekday]
                if len(weekday_sleep_times) > 1:
                    variance = np.var(weekday_sleep_times)
                    self.sleep_patterns["prediction_model"]["sleep_time_variance"][weekday_key] = variance
                
                # Wake time variance
                weekday_wake_times = [t[1] for t in wake_times if t[0] == weekday]
                if len(weekday_wake_times) > 1:
                    variance = np.var(weekday_wake_times)
                    self.sleep_patterns["prediction_model"]["wake_time_variance"][weekday_key] = variance
            
        except Exception as e:
            logger.error(f"Error updating variance data: {e}")
    
    def _check_predictive_actions(self):
        """Check if predictive ventilation actions should be taken."""
        try:
            now = datetime.now()
            weekday = now.weekday()
            weekday_key = str(weekday)
            
            pattern = self.sleep_patterns["weekday_patterns"][weekday_key]
            
            # Only proceed if we have established patterns
            if not pattern["sleep"] or not pattern["wake"] or pattern["confidence"] < 0.6:
                return
            
            # Calculate predicted times
            sleep_time_minutes = self._time_str_to_minutes(pattern["sleep"])
            wake_time_minutes = self._time_str_to_minutes(pattern["wake"])
            
            # Current time in minutes
            current_minutes = now.hour * 60 + now.minute
            
            # Check for pre-sleep ventilation
            pre_sleep_start = sleep_time_minutes - self.pre_sleep_ventilation_minutes
            if pre_sleep_start < 0:
                pre_sleep_start += 24 * 60  # Handle midnight crossover
            
            if abs(current_minutes - pre_sleep_start) < 5:  # Within 5 minutes
                self._trigger_pre_sleep_ventilation(sleep_time_minutes)
            
            # Check for pre-wake ventilation
            pre_wake_start = wake_time_minutes - self.pre_wake_ventilation_minutes
            if pre_wake_start < 0:
                pre_wake_start += 24 * 60  # Handle midnight crossover
            
            if abs(current_minutes - pre_wake_start) < 5:  # Within 5 minutes
                self._trigger_pre_wake_ventilation(wake_time_minutes)
            
        except Exception as e:
            logger.error(f"Error checking predictive actions: {e}")
    
    def _trigger_pre_sleep_ventilation(self, sleep_time_minutes):
        """Trigger ventilation before predicted sleep time."""
        try:
            # Check if already triggered today
            today = datetime.now().date().isoformat()
            
            # Implement ventilation logic here
            # Example: Gradually reduce ventilation intensity
            logger.info(f"Triggering pre-sleep ventilation for predicted sleep at {self._minutes_to_time_str(sleep_time_minutes)}")
            
            # You would call controller methods here
            # self.controller.set_pre_sleep_mode()
            
        except Exception as e:
            logger.error(f"Error triggering pre-sleep ventilation: {e}")
    
    def _trigger_pre_wake_ventilation(self, wake_time_minutes):
        """Trigger ventilation before predicted wake time."""
        try:
            # Check if already triggered today
            today = datetime.now().date().isoformat()
            
            # Implement ventilation logic here
            # Example: Gradually increase ventilation intensity
            logger.info(f"Triggering pre-wake ventilation for predicted wake at {self._minutes_to_time_str(wake_time_minutes)}")
            
            # You would call controller methods here
            # self.controller.set_pre_wake_mode()
            
        except Exception as e:
            logger.error(f"Error triggering pre-wake ventilation: {e}")
    
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
    
    def get_pending_confirmations(self):
        """Get list of events pending user confirmation."""
        return self.user_confirmations["pending_confirmations"]
    
    def get_sleep_pattern_summary(self):
        """Get enhanced summary of detected sleep patterns."""
        try:
            summary = {
                "weekday_patterns": {},
                "recent_events": [],
                "confidence_levels": {},
                "baseline_info": {},
                "prediction_accuracy": self.sleep_patterns["prediction_model"]["recent_accuracy"],
                "pending_confirmations": len(self.user_confirmations["pending_confirmations"])
            }
            
            # Format weekday patterns
            for day_key, pattern in self.sleep_patterns["weekday_patterns"].items():
                day_name = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][int(day_key)]
                
                if pattern["sleep"] and pattern["wake"]:
                    summary["weekday_patterns"][day_name] = {
                        "sleep": pattern["sleep"],
                        "wake": pattern["wake"],
                        "confidence": f"{pattern['confidence']:.2f}",
                        "confirmed_ratio": f"{pattern.get('confirmed_count', 0)}/{pattern.get('total_count', 0)}"
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
                        "confidence": f"{event['details']['confidence']:.2f}",
                        "status": self._get_event_confirmation_status(event["event_id"])
                    })
                except:
                    continue
            
            # Add baseline info
            summary["baseline_info"] = {
                "sleep_threshold": self.baseline_data["adaptive_thresholds"]["sleep_detection"],
                "wake_threshold": self.baseline_data["adaptive_thresholds"]["wake_detection"]
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting sleep pattern summary: {e}")
            return {"error": str(e)}
    
    def _get_event_confirmation_status(self, event_id):
        """Get confirmation status for an event."""
        for conf in self.user_confirmations["confirmed_events"]:
            if conf["event_id"] == event_id:
                return "confirmed"
        
        for conf in self.user_confirmations["rejected_events"]:
            if conf["event_id"] == event_id:
                return "rejected"
        
        for conf in self.user_confirmations["pending_confirmations"]:
            if conf["event_id"] == event_id:
                return "pending"
        
        return "unknown"