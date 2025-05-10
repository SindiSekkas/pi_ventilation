# predictive/occupancy_pattern_analyzer.py
"""Analyzer for occupancy patterns and predictions."""
import os
import json
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Any
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
                        for key, value in data.items()
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
                f"{key[0]},{key[1]}": value 
                for key, value in self.empty_probabilities.items()
            }
            
            with open(self.probabilities_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info("Saved occupancy probabilities to file")
        except Exception as e:
            logger.error(f"Error saving probabilities: {e}")
    
    def _load_and_process_history(self):
        """Load history from CSV and calculate empty probabilities."""
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
            
            # Calculate probabilities for each (day_of_week, hour) combination
            self.empty_probabilities = {}
            
            # Group by day_of_week and hour
            grouped = df.groupby(['day_of_week', 'hour'])
            
            for (day, hour), group in grouped:
                # Count empty vs occupied states
                empty_count = len(group[group['status'] == 'EMPTY'])
                total_count = len(group)
                
                # Calculate probability of being empty
                probability = empty_count / total_count if total_count > 0 else 0.5
                self.empty_probabilities[(day, hour)] = probability
                
                logger.debug(f"Day {day}, Hour {hour}: P(EMPTY) = {probability:.3f} ({empty_count}/{total_count})")
            
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
    
    def get_next_expected_return_time(self, current_datetime: datetime) -> Optional[datetime]:
        """
        Predict when people are expected to return if currently empty.
        
        Args:
            current_datetime: Current datetime
            
        Returns:
            datetime or None: Expected return time, None if uncertain
        """
        # Update probabilities if needed
        if self._should_reload_history():
            self._load_and_process_history()
        
        # Start from current time and look forward for significant occupancy probability
        search_datetime = current_datetime
        max_search_hours = 24  # Don't search beyond 24 hours
        
        for hours_ahead in range(max_search_hours + 1):
            check_datetime = search_datetime + timedelta(hours=hours_ahead)
            empty_prob = self.get_predicted_empty_probability(check_datetime)
            
            # If probability of being empty drops below 0.3, likely return time
            if empty_prob < 0.3:
                logger.info(f"Expected return time: {check_datetime} (P(EMPTY)={empty_prob:.3f})")
                return check_datetime
        
        logger.debug("No confident return time found within 24 hours")
        return None
    
    def get_next_expected_departure_time(self, current_datetime: datetime) -> Optional[datetime]:
        """
        Predict when people are expected to leave if currently occupied.
        
        Args:
            current_datetime: Current datetime
            
        Returns:
            datetime or None: Expected departure time, None if uncertain
        """
        # Update probabilities if needed
        if self._should_reload_history():
            self._load_and_process_history()
        
        # Start from current time and look forward for significant emptiness probability
        search_datetime = current_datetime
        max_search_hours = 24  # Don't search beyond 24 hours
        
        for hours_ahead in range(max_search_hours + 1):
            check_datetime = search_datetime + timedelta(hours=hours_ahead)
            empty_prob = self.get_predicted_empty_probability(check_datetime)
            
            # If probability of being empty rises above 0.7, likely departure time
            if empty_prob > 0.7:
                logger.info(f"Expected departure time: {check_datetime} (P(EMPTY)={empty_prob:.3f})")
                return check_datetime
        
        logger.debug("No confident departure time found within 24 hours")
        return None
    
    def get_expected_empty_duration(self, current_datetime: datetime) -> Optional[timedelta]:
        """
        Predict how long the space will remain empty.
        
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