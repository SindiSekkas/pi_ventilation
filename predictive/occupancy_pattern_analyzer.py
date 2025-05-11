"""
Occupancy pattern analyzer for smart ventilation system.
Uses historical data to predict when spaces are likely to be occupied or empty.
"""
import os
import json
import pandas as pd
import logging
import csv
from datetime import datetime, timedelta
from typing import Dict, Optional, Any, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)

class OccupancyPatternAnalyzer:
    """
    Analyzes occupancy patterns and provides probability-based predictions.
    
    Processes historical occupancy data to build time-based probability models,
    with special weighting for user-confirmed feedback. Enables smart ventilation
    decisions based on predicted occupancy states.
    """
    
    def __init__(self, occupancy_history_file: str):
        """
        Initialize the occupancy pattern analyzer.
        
        Args:
            occupancy_history_file: Path to CSV file containing occupancy history
        """
        self.history_file = occupancy_history_file
        self.probabilities_file = os.path.join(
            os.path.dirname(occupancy_history_file), 
            "occupancy_probabilities.json"
        )
        
        # Storage for calculated probabilities
        self.empty_probabilities = {}  # {(day_of_week, hour): probability}
        self.hourly_patterns = {}  # {(day_of_week, hour): {'total': count, 'empty': count}}
        self.last_load_time = None
        
        # Load existing probabilities if available
        self._load_probabilities()
    
    def _load_probabilities(self):
        """
        Load previously calculated probabilities from JSON file.
        
        Restores the probability model from persistent storage to avoid
        recalculating on each restart.
        """
        if os.path.exists(self.probabilities_file):
            try:
                with open(self.probabilities_file, 'r') as f:
                    data = json.load(f)
                    self.empty_probabilities = {
                        tuple(map(int, key.split(','))): value 
                        for key, value in data.get('probabilities', {}).items()
                    }
                    self.hourly_patterns = {
                        tuple(map(int, key.split(','))): value 
                        for key, value in data.get('patterns', {}).items()
                    }
                    self.last_load_time = datetime.now()
                logger.info("Loaded occupancy probabilities from file")
            except Exception as e:
                logger.error(f"Error loading probabilities: {e}")
    
    def _save_probabilities(self):
        """
        Persist probability model to JSON file.
        
        Converts tuple keys to string format for JSON serialization.
        """
        try:
            # Convert tuple keys to strings for JSON serialization
            data = {
                'probabilities': {
                    f"{key[0]},{key[1]}": value 
                    for key, value in self.empty_probabilities.items()
                },
                'patterns': {
                    f"{key[0]},{key[1]}": value 
                    for key, value in self.hourly_patterns.items()
                }
            }
            
            with open(self.probabilities_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info("Saved occupancy probabilities to file")
        except Exception as e:
            logger.error(f"Error saving probabilities: {e}")
    
    def _load_and_process_history(self):
        """
        Process historical occupancy data to build probability model.
        
        Prioritizes user feedback over automatically detected states and applies
        appropriate weighting to generate reliable probability estimates.
        """
        try:
            # Check file existence
            if not os.path.exists(self.history_file):
                logger.warning(f"History file does not exist: {self.history_file}")
                return
            
            df = pd.read_csv(self.history_file)
            if df.empty:
                logger.warning("Empty history file")
                return
            
            # Prepare data
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['day_of_week'] = df['timestamp'].dt.dayofweek  # 0=Monday, 6=Sunday
            df['hour'] = df['timestamp'].dt.hour
            
            # Handle backward compatibility
            if 'people_count' not in df.columns:
                df['people_count'] = 0
            
            # Temporary storage
            temp_hourly_patterns = {}
            temp_empty_probabilities = {}
            
            # Process user feedback with higher priority
            feedback_rows = df[df['status'].str.startswith('USER_CONFIRMED_', na=False)]
            grouped_feedback = feedback_rows.groupby(['day_of_week', 'hour'])
            
            feedback_weight = 10  # User feedback is weighted higher for reliability
            
            for (day, hour), group in grouped_feedback:
                key = (day, hour)
                
                # Count feedback types
                confirmed_home = len(group[group['status'] == 'USER_CONFIRMED_HOME'])
                confirmed_away = len(group[group['status'] == 'USER_CONFIRMED_AWAY'])
                
                # Apply feedback weighting
                total_count = (confirmed_home + confirmed_away) * feedback_weight
                empty_count = confirmed_away * feedback_weight
                
                temp_hourly_patterns[key] = {
                    'total': total_count,
                    'empty': empty_count
                }
                
                # Initial probability calculation
                probability = empty_count / total_count if total_count > 0 else 0.5
                temp_empty_probabilities[key] = probability
                
                logger.debug(f"Feedback for Day {day}, Hour {hour}: P(EMPTY) = {probability:.3f} "
                            f"(confirmed_home={confirmed_home}, confirmed_away={confirmed_away})")
            
            # Process automatic detection data
            regular_rows = df[(df['status'] == 'EMPTY') | (df['status'] == 'OCCUPIED')]
            grouped_regular = regular_rows.groupby(['day_of_week', 'hour'])
            
            for (day, hour), group in grouped_regular:
                key = (day, hour)
                
                empty_count = len(group[group['status'] == 'EMPTY'])
                total_count = len(group)
                
                # Combine with existing feedback data if available
                if key in temp_hourly_patterns:
                    temp_hourly_patterns[key]['total'] += total_count
                    temp_hourly_patterns[key]['empty'] += empty_count
                else:
                    temp_hourly_patterns[key] = {
                        'total': total_count,
                        'empty': empty_count
                    }
                
                # Calculate final probability
                pattern = temp_hourly_patterns[key]
                probability = pattern['empty'] / pattern['total'] if pattern['total'] > 0 else 0.5
                temp_empty_probabilities[key] = probability
                
                logger.debug(f"Combined data for Day {day}, Hour {hour}: P(EMPTY) = {probability:.3f} "
                            f"(empty={pattern['empty']}/{pattern['total']})")
            
            # Update class data
            self.hourly_patterns = temp_hourly_patterns
            self.empty_probabilities = temp_empty_probabilities
            
            self.last_load_time = datetime.now()
            self._save_probabilities()
            logger.info(f"Processed {len(df)} history records into {len(self.empty_probabilities)} patterns")
            
        except Exception as e:
            logger.error(f"Error processing history: {e}")
    
    def get_predicted_empty_probability(self, target_datetime: datetime) -> float:
        """
        Calculate probability that a space will be empty at a specific time.
        
        Args:
            target_datetime: The datetime to predict for
            
        Returns:
            float: Probability of being empty (0.0-1.0)
        """
        # Reload data if necessary
        if self._should_reload_history():
            self._load_and_process_history()
        
        day_of_week = target_datetime.weekday()
        hour = target_datetime.hour
        
        # Default to 0.5 (uncertain) if no data available
        probability = self.empty_probabilities.get((day_of_week, hour), 0.5)
        
        logger.debug(f"Predicted P(EMPTY) for {target_datetime}: {probability:.3f}")
        return probability
    
    def get_next_significant_event(self, current_datetime: datetime = None) -> Tuple[Optional[datetime], Optional[str], float]:
        """
        Find the next expected arrival or departure event.
        
        Analyzes probability transitions to identify when occupancy state is
        likely to change significantly.
        
        Args:
            current_datetime: Starting time for prediction (default: now)
            
        Returns:
            Tuple containing:
                - Time of next significant event (or None)
                - Event type ("EXPECTED_ARRIVAL" or "EXPECTED_DEPARTURE")
                - Confidence level (0.0-1.0)
        """
        if self._should_reload_history():
            self._load_and_process_history()
        
        now = current_datetime or datetime.now()
        max_hours_ahead = 48
        
        # Determine current state
        current_prob = self.get_predicted_empty_probability(now)
        current_state = "EMPTY" if current_prob > 0.5 else "OCCUPIED"
        
        # Thresholds for reliable state detection
        stable_threshold_empty = 0.7
        stable_threshold_occupied = 0.3
        min_stable_hours = 2
        
        # Scan future hours for state changes
        for hours_ahead in range(max_hours_ahead):
            check_time = now + timedelta(hours=hours_ahead)
            
            # Analyze stability of upcoming hours
            sequence_probs = []
            for hour_offset in range(min_stable_hours):
                seq_time = check_time + timedelta(hours=hour_offset)
                seq_prob = self.get_predicted_empty_probability(seq_time)
                sequence_probs.append(seq_prob)
            
            # Evaluate sequence stability
            avg_prob = sum(sequence_probs) / len(sequence_probs)
            prob_variance = sum((p - avg_prob) ** 2 for p in sequence_probs) / len(sequence_probs)
            
            # Low variance indicates stable state
            if prob_variance < 0.1:
                # Determine state from average probability
                new_state = None
                if avg_prob > stable_threshold_empty:
                    new_state = "EMPTY"
                elif avg_prob < stable_threshold_occupied:
                    new_state = "OCCUPIED"
                
                # If state changes, we found an event
                if new_state and new_state != current_state:
                    event_type = "EXPECTED_ARRIVAL" if new_state == "OCCUPIED" else "EXPECTED_DEPARTURE"
                    
                    # Calculate confidence from multiple factors
                    pattern_key = (check_time.weekday(), check_time.hour)
                    pattern = self.hourly_patterns.get(pattern_key, {'total': 0})
                    
                    confidence = min(0.9, max(0.1, 
                        (len(sequence_probs) / 3) * 0.3 +  # Sequence length factor
                        (abs(avg_prob - 0.5) * 2) * 0.4 +  # Probability strength
                        min(1.0, pattern.get('total', 0) / 10) * 0.3  # Historical data volume
                    ))
                    
                    return check_time, event_type, confidence
        
        # No significant event found
        return None, None, 0.0
    
    def get_predicted_current_period(self, current_datetime: datetime = None) -> Tuple[Optional[datetime], Optional[datetime], Optional[str], float]:
        """
        Analyze the current occupancy period including when it started and when it will end.
        
        Args:
            current_datetime: Reference time (default: now)
            
        Returns:
            Tuple containing:
                - Period start datetime
                - Expected period end datetime
                - Current status ("EXPECTED_EMPTY" or "EXPECTED_OCCUPIED")
                - Confidence level (0.0-1.0)
        """
        if self._should_reload_history():
            self._load_and_process_history()
        
        now = current_datetime or datetime.now()
        
        # Determine current state
        current_prob = self.get_predicted_empty_probability(now)
        current_state = "EXPECTED_EMPTY" if current_prob > 0.5 else "EXPECTED_OCCUPIED"
        
        # Thresholds for reliable state detection
        stable_threshold_empty = 0.7
        stable_threshold_occupied = 0.3
        
        # Look backward to find period start
        period_start = now
        for hours_back in range(24):
            check_time = now - timedelta(hours=hours_back)
            prob = self.get_predicted_empty_probability(check_time)
            
            # Detect state transition
            check_state = None
            if prob > stable_threshold_empty:
                check_state = "EXPECTED_EMPTY"
            elif prob < stable_threshold_occupied:
                check_state = "EXPECTED_OCCUPIED"
            
            if check_state and check_state != current_state:
                period_start = check_time + timedelta(hours=1)
                break
        
        # Find period end using next event
        next_event = self.get_next_significant_event(now)
        period_end = next_event[0] if next_event[0] else None
        
        # Calculate confidence based on period stability
        if period_start and period_end:
            duration_hours = (period_end - now).total_seconds() / 3600
            past_hours = (now - period_start).total_seconds() / 3600
            
            # Analyze probabilities throughout the period
            period_probs = []
            for hour in range(int(max(1, past_hours)), int(duration_hours) + 1):
                check_time = period_start + timedelta(hours=hour)
                if check_time <= period_end:
                    prob = self.get_predicted_empty_probability(check_time)
                    period_probs.append(prob)
            
            if period_probs:
                # Calculate statistics about period stability
                avg_prob = sum(period_probs) / len(period_probs)
                prob_variance = sum((p - avg_prob) ** 2 for p in period_probs) / len(period_probs)
                
                # Weighted confidence calculation
                confidence = min(0.9, max(0.1,
                    (1.0 - prob_variance) * 0.5 +  # Lower variance = higher confidence
                    (abs(avg_prob - 0.5) * 2) * 0.3 +  # Stronger probability = higher confidence
                    min(1.0, len(period_probs) / 6) * 0.2  # Longer stable period = higher confidence
                ))
            else:
                confidence = 0.3
        else:
            confidence = 0.1
        
        return period_start, period_end, current_state, confidence
    
    def record_user_feedback(self, feedback_timestamp: datetime, actual_status: str):
        """
        Incorporate user feedback to improve future predictions.
        
        User feedback is considered high-value data and is given extra weight
        when calculating probabilities.
        
        Args:
            feedback_timestamp: When the feedback applies to
            actual_status: "USER_CONFIRMED_HOME" or "USER_CONFIRMED_AWAY"
        """
        if actual_status not in ["USER_CONFIRMED_HOME", "USER_CONFIRMED_AWAY"]:
            logger.error(f"Invalid feedback status: {actual_status}")
            return
        
        # Convert feedback to empty/occupied state
        is_empty = (actual_status == "USER_CONFIRMED_AWAY")
        
        day_of_week = feedback_timestamp.weekday()
        hour = feedback_timestamp.hour
        key = (day_of_week, hour)
        
        # Initialize or update pattern data
        if key not in self.hourly_patterns:
            self.hourly_patterns[key] = {'total': 0, 'empty': 0}
        
        self.hourly_patterns[key]['total'] += 1
        if is_empty:
            self.hourly_patterns[key]['empty'] += 1
        
        # Calculate new raw probability
        pattern = self.hourly_patterns[key]
        new_probability = pattern['empty'] / pattern['total']
        
        # Apply weighted learning
        learning_rate = 0.3  # Controls adaptation speed
        old_probability = self.empty_probabilities.get(key, 0.5)
        
        self.empty_probabilities[key] = (
            old_probability * (1 - learning_rate) + 
            new_probability * learning_rate
        )
        
        logger.info(f"Updated occupancy pattern for day {day_of_week}, hour {hour}: "
                   f"P(EMPTY) = {self.empty_probabilities[key]:.3f} "
                   f"(feedback: {actual_status})")
        
        # Persist updated model
        self._save_probabilities()
        
        # Record feedback in history file for future processing
        self._save_feedback_to_csv(feedback_timestamp, actual_status)
    
    def _save_feedback_to_csv(self, feedback_timestamp: datetime, actual_status: str):
        """
        Save user feedback to history CSV for long-term storage.
        
        Args:
            feedback_timestamp: Time of the feedback
            actual_status: "USER_CONFIRMED_HOME" or "USER_CONFIRMED_AWAY"
        """
        try:
            feedback_row = {
                'timestamp': feedback_timestamp.isoformat(),
                'status': actual_status,
                'people_count': 1 if actual_status == "USER_CONFIRMED_HOME" else 0
            }
            
            file_exists = os.path.exists(self.history_file)
            
            with open(self.history_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['timestamp', 'status', 'people_count'])
                
                if not file_exists:
                    writer.writeheader()
                
                writer.writerow(feedback_row)
            
            logger.debug(f"Saved feedback to CSV: {feedback_row}")
            
        except Exception as e:
            logger.error(f"Error saving feedback to CSV: {e}")
    
    def get_next_expected_return_time(self, current_datetime: datetime) -> Optional[datetime]:
        """
        Predict when occupants are expected to return to an empty space.
        
        Args:
            current_datetime: Reference time
            
        Returns:
            datetime or None: Expected return time, or None if undetermined
        """
        next_event = self.get_next_significant_event(current_datetime)
        
        if next_event[0] and next_event[1] == "EXPECTED_ARRIVAL":
            logger.info(f"Expected return time: {next_event[0]} (confidence: {next_event[2]:.3f})")
            return next_event[0]
        
        logger.debug("No confident return time found")
        return None
    
    def get_next_expected_departure_time(self, current_datetime: datetime) -> Optional[datetime]:
        """
        Predict when occupants are expected to leave an occupied space.
        
        Args:
            current_datetime: Reference time
            
        Returns:
            datetime or None: Expected departure time, or None if undetermined
        """
        next_event = self.get_next_significant_event(current_datetime)
        
        if next_event[0] and next_event[1] == "EXPECTED_DEPARTURE":
            logger.info(f"Expected departure time: {next_event[0]} (confidence: {next_event[2]:.3f})")
            return next_event[0]
        
        logger.debug("No confident departure time found")
        return None
    
    def get_expected_empty_duration(self, current_datetime: datetime) -> Optional[timedelta]:
        """
        Estimate how long a space will remain empty.
        
        Useful for determining if ventilation can be reduced for an extended period.
        
        Args:
            current_datetime: Reference time
            
        Returns:
            timedelta or None: Expected duration of emptiness, or None if undetermined
        """
        return_time = self.get_next_expected_return_time(current_datetime)
        
        if return_time:
            duration = return_time - current_datetime
            logger.info(f"Expected empty duration: {duration}")
            return duration
        
        logger.debug("Cannot determine expected empty duration")
        return None
    
    def _should_reload_history(self) -> bool:
        """
        Determine if historical data needs to be reprocessed.
        
        Prevents unnecessary processing by checking file modification time
        and enforcing minimum intervals between updates.
        
        Returns:
            bool: True if reload is needed, False otherwise
        """
        # First-time load
        if self.last_load_time is None:
            return True
        
        # Fallback reload for stale data
        if datetime.now() - self.last_load_time > timedelta(hours=72):
            logger.warning("Emergency reload triggered - data is more than 72 hours old")
            return True
        
        # Regular update check - file modified and sufficient time elapsed
        try:
            if os.path.exists(self.history_file):
                file_mtime = datetime.fromtimestamp(os.path.getmtime(self.history_file))
                time_since_last_load = datetime.now() - self.last_load_time
                
                if file_mtime > self.last_load_time and time_since_last_load > timedelta(hours=6):
                    logger.info("Reloading history due to file changes and sufficient time elapsed")
                    return True
        except Exception as e:
            logger.error(f"Error checking file modification time: {e}")
        
        return False
    
    def update_patterns(self, force: bool = True) -> bool:
        """
        Trigger a manual update of occupancy patterns.
        
        Typically called periodically by the system scheduler.
        
        Args:
            force: Whether to bypass time-based update restrictions
            
        Returns:
            bool: True if patterns were updated, False otherwise
        """
        if force or self._should_reload_history():
            logger.info("Performing scheduled update of occupancy patterns")
            self._load_and_process_history()
            return True
        return False
    
    def get_pattern_summary(self) -> Dict[str, Any]:
        """
        Generate a human-readable summary of occupancy patterns.
        
        Provides an overview of when spaces are typically occupied or empty,
        organized by day of week.
        
        Returns:
            Dict containing pattern summary data
        """
        if self._should_reload_history():
            self._load_and_process_history()
        
        summary = {
            "total_patterns": len(self.empty_probabilities),
            "last_update": self.last_load_time.isoformat() if self.last_load_time else None,
            "day_patterns": {},
            "empty_hour_ranges": {}
        }
        
        # Organize by day of week
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        for day_idx in range(7):
            day_name = days[day_idx]
            summary["day_patterns"][day_name] = {}
            
            # Collect hours with high empty probability
            empty_hours = []
            for hour in range(24):
                prob = self.empty_probabilities.get((day_idx, hour), 0.5)
                summary["day_patterns"][day_name][hour] = prob
                if prob > 0.7:  # High probability threshold
                    empty_hours.append(hour)
            
            # Identify continuous time ranges
            if empty_hours:
                ranges = []
                start = empty_hours[0]
                end = empty_hours[0]
                
                for i in range(1, len(empty_hours)):
                    if empty_hours[i] == end + 1:
                        end = empty_hours[i]
                    else:
                        ranges.append((start, end))
                        start = empty_hours[i]
                        end = empty_hours[i]
                ranges.append((start, end))
                
                summary["empty_hour_ranges"][day_name] = ranges
        
        return summary