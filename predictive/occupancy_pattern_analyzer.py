# predictive/occupancy_pattern_analyzer.py
"""Analyzer for occupancy patterns and predictions."""
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
    """Analyzes occupancy patterns to predict empty/occupied states."""
    
    def __init__(self, occupancy_history_file: str):
        """
        Initialize the occupancy pattern analyzer.
        
        Args:
            occupancy_history_file: Path to occupancy_history.csv file
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
        """Load saved probabilities from JSON file."""
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
        """Save calculated probabilities to JSON file."""
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
        """Load history from CSV and calculate empty probabilities with feedback priority."""
        try:
            # Read the CSV file
            if not os.path.exists(self.history_file):
                logger.warning(f"History file does not exist: {self.history_file}")
                return
            
            df = pd.read_csv(self.history_file)
            if df.empty:
                logger.warning("Empty history file")
                return
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['day_of_week'] = df['timestamp'].dt.dayofweek  # 0=Monday, 6=Sunday
            df['hour'] = df['timestamp'].dt.hour
            
            # Ensure people_count column exists (for backward compatibility)
            if 'people_count' not in df.columns:
                df['people_count'] = 0
            
            # Initialize temporary patterns storage
            temp_hourly_patterns = {}
            temp_empty_probabilities = {}
            
            # First pass: Process feedback records with high priority
            feedback_rows = df[df['status'].str.startswith('USER_CONFIRMED_', na=False)]
            grouped_feedback = feedback_rows.groupby(['day_of_week', 'hour'])
            
            feedback_weight = 10  # Weight for feedback records
            
            for (day, hour), group in grouped_feedback:
                key = (day, hour)
                
                # Count feedback types
                confirmed_home = len(group[group['status'] == 'USER_CONFIRMED_HOME'])
                confirmed_away = len(group[group['status'] == 'USER_CONFIRMED_AWAY'])
                
                # Initialize with feedback data
                total_count = (confirmed_home + confirmed_away) * feedback_weight
                empty_count = confirmed_away * feedback_weight
                
                temp_hourly_patterns[key] = {
                    'total': total_count,
                    'empty': empty_count
                }
                
                # Calculate initial probability from feedback
                probability = empty_count / total_count if total_count > 0 else 0.5
                temp_empty_probabilities[key] = probability
                
                logger.debug(f"Feedback for Day {day}, Hour {hour}: P(EMPTY) = {probability:.3f} "
                            f"(confirmed_home={confirmed_home}, confirmed_away={confirmed_away})")
            
            # Second pass: Process all regular occupancy records
            regular_rows = df[(df['status'] == 'EMPTY') | (df['status'] == 'OCCUPIED')]
            grouped_regular = regular_rows.groupby(['day_of_week', 'hour'])
            
            for (day, hour), group in grouped_regular:
                key = (day, hour)
                
                # Count empty vs occupied states
                empty_count = len(group[group['status'] == 'EMPTY'])
                total_count = len(group)
                
                # If we have feedback for this slot, add to existing counts
                if key in temp_hourly_patterns:
                    temp_hourly_patterns[key]['total'] += total_count
                    temp_hourly_patterns[key]['empty'] += empty_count
                else:
                    # No feedback, initialize with regular data
                    temp_hourly_patterns[key] = {
                        'total': total_count,
                        'empty': empty_count
                    }
                
                # Recalculate probability
                pattern = temp_hourly_patterns[key]
                probability = pattern['empty'] / pattern['total'] if pattern['total'] > 0 else 0.5
                temp_empty_probabilities[key] = probability
                
                logger.debug(f"Combined data for Day {day}, Hour {hour}: P(EMPTY) = {probability:.3f} "
                            f"(empty={pattern['empty']}/{pattern['total']})")
            
            # Update class attributes with new data
            self.hourly_patterns = temp_hourly_patterns
            self.empty_probabilities = temp_empty_probabilities
            
            self.last_load_time = datetime.now()
            self._save_probabilities()
            logger.info(f"Processed {len(df)} history records into {len(self.empty_probabilities)} patterns")
            
        except Exception as e:
            logger.error(f"Error processing history: {e}")
    
    def get_predicted_empty_probability(self, target_datetime: datetime) -> float:
        """
        Get predicted probability of being empty at target datetime.
        
        Args:
            target_datetime: The datetime to predict for
            
        Returns:
            float: Probability of being empty (0.0-1.0)
        """
        # Update probabilities only if really needed
        if self._should_reload_history():
            self._load_and_process_history()
        
        day_of_week = target_datetime.weekday()
        hour = target_datetime.hour
        
        # Get probability from stored data
        probability = self.empty_probabilities.get((day_of_week, hour), 0.5)
        
        logger.debug(f"Predicted P(EMPTY) for {target_datetime}: {probability:.3f}")
        return probability
    
    def get_next_significant_event(self, current_datetime: datetime = None) -> Tuple[Optional[datetime], Optional[str], float]:
        """
        Get the next significant occupancy event (arrival or departure).
        
        Args:
            current_datetime: Starting time for prediction (default: now)
            
        Returns:
            Tuple[datetime, str, float]: (event_datetime, event_type, confidence)
                - event_datetime: Time of next significant event
                - event_type: "EXPECTED_ARRIVAL" or "EXPECTED_DEPARTURE"
                - confidence: Confidence level (0.0-1.0)
        """
        # Update probabilities if needed
        if self._should_reload_history():
            self._load_and_process_history()
        
        now = current_datetime or datetime.now()
        
        # Look ahead for 48 hours
        max_hours_ahead = 48
        
        # Get current state and find when it changes
        current_prob = self.get_predicted_empty_probability(now)
        current_state = "EMPTY" if current_prob > 0.5 else "OCCUPIED"
        
        # Look for stable sequences
        stable_threshold_empty = 0.7
        stable_threshold_occupied = 0.3
        min_stable_hours = 2
        
        for hours_ahead in range(max_hours_ahead):
            check_time = now + timedelta(hours=hours_ahead)
            
            # Get sequence of probabilities for next few hours
            sequence_probs = []
            for hour_offset in range(min_stable_hours):
                seq_time = check_time + timedelta(hours=hour_offset)
                seq_prob = self.get_predicted_empty_probability(seq_time)
                sequence_probs.append(seq_prob)
            
            # Check if we have a stable sequence
            avg_prob = sum(sequence_probs) / len(sequence_probs)
            prob_variance = sum((p - avg_prob) ** 2 for p in sequence_probs) / len(sequence_probs)
            
            if prob_variance < 0.1:  # Stable sequence
                new_state = "EMPTY" if avg_prob > stable_threshold_empty else "OCCUPIED" if avg_prob < stable_threshold_occupied else None
                
                if new_state and new_state != current_state:
                    # Found a significant change
                    event_type = "EXPECTED_ARRIVAL" if new_state == "OCCUPIED" else "EXPECTED_DEPARTURE"
                    
                    # Calculate confidence based on:
                    # 1. Length of stable sequence
                    # 2. Average probability strength
                    # 3. Historical data volume
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
        Get the predicted current occupancy period (start, end, status).
        
        Args:
            current_datetime: Reference time (default: now)
            
        Returns:
            Tuple[datetime, datetime, str, float]: (start_datetime, end_datetime, status, confidence)
                - start_datetime: When current period began
                - end_datetime: When current period is expected to end
                - status: "EXPECTED_EMPTY" or "EXPECTED_OCCUPIED"
                - confidence: Confidence level (0.0-1.0)
        """
        # Update probabilities if needed
        if self._should_reload_history():
            self._load_and_process_history()
        
        now = current_datetime or datetime.now()
        
        # Get current state
        current_prob = self.get_predicted_empty_probability(now)
        current_state = "EXPECTED_EMPTY" if current_prob > 0.5 else "EXPECTED_OCCUPIED"
        
        # Look back to find period start
        stable_threshold_empty = 0.7
        stable_threshold_occupied = 0.3
        
        period_start = now
        for hours_back in range(24):  # Look back up to 24 hours
            check_time = now - timedelta(hours=hours_back)
            prob = self.get_predicted_empty_probability(check_time)
            
            # Check if state was different
            check_state = "EXPECTED_EMPTY" if prob > stable_threshold_empty else "EXPECTED_OCCUPIED" if prob < stable_threshold_occupied else None
            
            if check_state and check_state != current_state:
                period_start = check_time + timedelta(hours=1)  # First hour in current period
                break
        
        # Get the next significant event as period end
        next_event = self.get_next_significant_event(now)
        period_end = next_event[0] if next_event[0] else None
        
        # Calculate confidence based on stability of the period
        if period_start and period_end:
            duration_hours = (period_end - now).total_seconds() / 3600
            past_hours = (now - period_start).total_seconds() / 3600
            
            # Check stability of the period
            period_probs = []
            for hour in range(int(max(1, past_hours)), int(duration_hours) + 1):
                check_time = period_start + timedelta(hours=hour)
                if check_time <= period_end:
                    prob = self.get_predicted_empty_probability(check_time)
                    period_probs.append(prob)
            
            if period_probs:
                avg_prob = sum(period_probs) / len(period_probs)
                prob_variance = sum((p - avg_prob) ** 2 for p in period_probs) / len(period_probs)
                
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
        Record user feedback about occupancy status to improve predictions.
        Saves feedback to both memory and CSV for persistence.
        
        Args:
            feedback_timestamp: Timestamp of the feedback
            actual_status: "USER_CONFIRMED_HOME" or "USER_CONFIRMED_AWAY"
        """
        if actual_status not in ["USER_CONFIRMED_HOME", "USER_CONFIRMED_AWAY"]:
            logger.error(f"Invalid feedback status: {actual_status}")
            return
        
        # Convert feedback status to our internal format
        is_empty = (actual_status == "USER_CONFIRMED_AWAY")
        
        day_of_week = feedback_timestamp.weekday()
        hour = feedback_timestamp.hour
        key = (day_of_week, hour)
        
        # Initialize pattern if not exists
        if key not in self.hourly_patterns:
            self.hourly_patterns[key] = {'total': 0, 'empty': 0}
        
        # Update pattern counts
        self.hourly_patterns[key]['total'] += 1
        if is_empty:
            self.hourly_patterns[key]['empty'] += 1
        
        # Recalculate probability for this hour
        pattern = self.hourly_patterns[key]
        new_probability = pattern['empty'] / pattern['total']
        
        # Apply learning rate to smooth the update
        learning_rate = 0.3  # Weight of new feedback
        old_probability = self.empty_probabilities.get(key, 0.5)
        
        # Weighted average update
        self.empty_probabilities[key] = (
            old_probability * (1 - learning_rate) + 
            new_probability * learning_rate
        )
        
        logger.info(f"Updated occupancy pattern for day {day_of_week}, hour {hour}: "
                   f"P(EMPTY) = {self.empty_probabilities[key]:.3f} "
                   f"(feedback: {actual_status})")
        
        # Save the updated probabilities
        self._save_probabilities()
        
        # Save feedback to CSV for persistence
        self._save_feedback_to_csv(feedback_timestamp, actual_status)
    
    def _save_feedback_to_csv(self, feedback_timestamp: datetime, actual_status: str):
        """
        Save user feedback to CSV file for long-term persistence.
        
        Args:
            feedback_timestamp: Timestamp of the feedback
            actual_status: "USER_CONFIRMED_HOME" or "USER_CONFIRMED_AWAY"
        """
        try:
            # Create feedback entry
            feedback_row = {
                'timestamp': feedback_timestamp.isoformat(),
                'status': actual_status,
                'people_count': 1 if actual_status == "USER_CONFIRMED_HOME" else 0
            }
            
            # Check if file exists to determine if we need headers
            file_exists = os.path.exists(self.history_file)
            
            # Append feedback to CSV
            with open(self.history_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['timestamp', 'status', 'people_count'])
                
                # Write header if file doesn't exist
                if not file_exists:
                    writer.writeheader()
                
                writer.writerow(feedback_row)
            
            logger.debug(f"Saved feedback to CSV: {feedback_row}")
            
        except Exception as e:
            logger.error(f"Error saving feedback to CSV: {e}")
    
    def get_next_expected_return_time(self, current_datetime: datetime) -> Optional[datetime]:
        """
        Predict when people are expected to return if currently empty.
        Updated to use the new significant event method.
        
        Args:
            current_datetime: Current datetime
            
        Returns:
            datetime or None: Expected return time, None if uncertain
        """
        next_event = self.get_next_significant_event(current_datetime)
        
        if next_event[0] and next_event[1] == "EXPECTED_ARRIVAL":
            logger.info(f"Expected return time: {next_event[0]} (confidence: {next_event[2]:.3f})")
            return next_event[0]
        
        logger.debug("No confident return time found")
        return None
    
    def get_next_expected_departure_time(self, current_datetime: datetime) -> Optional[datetime]:
        """
        Predict when people are expected to leave if currently occupied.
        Updated to use the new significant event method.
        
        Args:
            current_datetime: Current datetime
            
        Returns:
            datetime or None: Expected departure time, None if uncertain
        """
        next_event = self.get_next_significant_event(current_datetime)
        
        if next_event[0] and next_event[1] == "EXPECTED_DEPARTURE":
            logger.info(f"Expected departure time: {next_event[0]} (confidence: {next_event[2]:.3f})")
            return next_event[0]
        
        logger.debug("No confident departure time found")
        return None
    
    def get_expected_empty_duration(self, current_datetime: datetime) -> Optional[timedelta]:
        """
        Predict how long the space will remain empty.
        Updated to use the new significant event method.
        
        Args:
            current_datetime: Current datetime
            
        Returns:
            timedelta or None: Expected duration of emptiness, None if uncertain
        """
        # Find the next expected return time
        return_time = self.get_next_expected_return_time(current_datetime)
        
        if return_time:
            duration = return_time - current_datetime
            logger.info(f"Expected empty duration: {duration}")
            return duration
        
        logger.debug("Cannot determine expected empty duration")
        return None
    
    def _should_reload_history(self) -> bool:
        """Check if history should be reloaded based on time or file changes."""
        # Reload if we haven't loaded before
        if self.last_load_time is None:
            return True
        
        # Emergency reload if it's been more than 72 hours (fallback if periodic updates fail)
        if datetime.now() - self.last_load_time > timedelta(hours=72):
            logger.warning("Emergency reload triggered - data is more than 72 hours old")
            return True
        
        # Reload if the history file has been modified AND it's been at least 6 hours since last load
        # This prevents frequent reloads when getter methods are called in succession
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
        Update occupancy patterns, typically called periodically from main.py.
        
        Args:
            force: Whether to force an update regardless of time conditions
            
        Returns:
            bool: True if patterns were updated, False otherwise
        """
        if force or self._should_reload_history():
            logger.info("Performing scheduled update of occupancy patterns")
            self._load_and_process_history()
            return True
        return False
    
    def get_pattern_summary(self) -> Dict[str, Any]:
        """Get a summary of the learned patterns."""
        if self._should_reload_history():
            self._load_and_process_history()
        
        summary = {
            "total_patterns": len(self.empty_probabilities),
            "last_update": self.last_load_time.isoformat() if self.last_load_time else None,
            "day_patterns": {},
            "empty_hour_ranges": {}
        }
        
        # Organize patterns by day of week
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        for day_idx in range(7):
            day_name = days[day_idx]
            summary["day_patterns"][day_name] = {}
            
            # Find typical empty hours for this day
            empty_hours = []
            for hour in range(24):
                prob = self.empty_probabilities.get((day_idx, hour), 0.5)
                summary["day_patterns"][day_name][hour] = prob
                if prob > 0.7:  # High probability of being empty
                    empty_hours.append(hour)
            
            # Identify continuous ranges of empty hours
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