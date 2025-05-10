# preferences/preference_manager.py
"""Preference manager for user ventilation preferences."""
import os
import json
import logging
import math
from datetime import datetime
from typing import Dict, List, Optional, Union
from .models import UserPreference, FeedbackRecord, CompromisePreference

logger = logging.getLogger(__name__)


class PreferenceManager:
    """Manages user preferences for ventilation settings."""
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize preference manager.
        
        Args:
            data_dir: Directory to store preference data
        """
        self.data_dir = data_dir
        self.preference_dir = os.path.join(data_dir, "preferences")
        self.preferences_file = os.path.join(self.preference_dir, "user_preferences.json")
        self.feedback_file = os.path.join(self.preference_dir, "user_feedback.json")
        
        # Create directory if not exists
        os.makedirs(self.preference_dir, exist_ok=True)
        
        # Load existing preferences and feedback
        self.preferences = self._load_preferences()
        self.feedback_history = self._load_feedback()
    
    def _load_preferences(self) -> Dict[int, UserPreference]:
        """Load preferences from file."""
        if os.path.exists(self.preferences_file):
            try:
                with open(self.preferences_file, 'r') as f:
                    data = json.load(f)
                    preferences = {}
                    for user_id, pref_data in data.items():
                        preferences[int(user_id)] = UserPreference.from_dict(pref_data)
                    logger.info(f"Loaded preferences for {len(preferences)} users")
                    return preferences
            except Exception as e:
                logger.error(f"Error loading preferences: {e}")
        return {}
    
    def _save_preferences(self):
        """Save preferences to file."""
        try:
            data = {}
            for user_id, preference in self.preferences.items():
                data[str(user_id)] = preference.to_dict()
            
            with open(self.preferences_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.debug("Saved preferences to file")
        except Exception as e:
            logger.error(f"Error saving preferences: {e}")
    
    def _load_feedback(self) -> List[FeedbackRecord]:
        """Load feedback history from file."""
        if os.path.exists(self.feedback_file):
            try:
                with open(self.feedback_file, 'r') as f:
                    data = json.load(f)
                    feedback = [FeedbackRecord.from_dict(record) for record in data]
                    logger.info(f"Loaded {len(feedback)} feedback records")
                    return feedback
            except Exception as e:
                logger.error(f"Error loading feedback: {e}")
        return []
    
    def _save_feedback(self):
        """Save feedback history to file."""
        try:
            data = [record.to_dict() for record in self.feedback_history]
            with open(self.feedback_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.debug("Saved feedback to file")
        except Exception as e:
            logger.error(f"Error saving feedback: {e}")
    
    def get_user_preference(self, user_id: int, username: str = None) -> UserPreference:
        """Get or create user preference."""
        if user_id not in self.preferences:
            logger.info(f"Creating new preferences for user {user_id}")
            self.preferences[user_id] = UserPreference(user_id=user_id, username=username)
            self._save_preferences()
        elif username and self.preferences[user_id].username != username:
            # Update username if changed
            self.preferences[user_id].username = username
            self._save_preferences()
        
        return self.preferences[user_id]
    
    def set_user_preference(self, user_id: int, **kwargs) -> bool:
        """Update user preferences."""
        try:
            preference = self.get_user_preference(user_id)
            preference.update(**kwargs)
            self._save_preferences()
            logger.info(f"Updated preferences for user {user_id}")
            return True
        except Exception as e:
            logger.error(f"Error updating preferences for user {user_id}: {e}")
            return False
    
    def get_all_user_preferences(self) -> Dict[int, UserPreference]:
        """Get all user preferences."""
        return self.preferences.copy()
    
    def calculate_compromise_preference(self, list_of_user_ids: List[int]) -> CompromisePreference:
        """
        Calculate compromise preferences for a group of users using an optimized approach.
        
        Args:
            list_of_user_ids: List of user IDs to calculate compromise for
            
        Returns:
            CompromisePreference: Calculated compromise settings
        """
        if not list_of_user_ids:
            # Return default preferences if no users
            return CompromisePreference(
                user_count=0,
                temp_min=20.0,
                temp_max=24.0,
                co2_threshold=1000,
                humidity_min=30.0,
                humidity_max=60.0,
                effectiveness_score=1.0
            )
        
        # Filter valid users
        valid_preferences = []
        for user_id in list_of_user_ids:
            if user_id in self.preferences:
                valid_preferences.append(self.preferences[user_id])
        
        if not valid_preferences:
            # Return default if no valid preferences
            return CompromisePreference(
                user_count=0,
                temp_min=20.0,
                temp_max=24.0,
                co2_threshold=1000,
                humidity_min=30.0,
                humidity_max=60.0,
                effectiveness_score=1.0
            )
        
        # Step 1: Find intersection ranges for each parameter
        temp_intersections = self._find_range_intersection(
            [(p.temp_min, p.temp_max) for p in valid_preferences]
        )
        
        humidity_intersections = self._find_range_intersection(
            [(p.humidity_min, p.humidity_max) for p in valid_preferences]
        )
        
        # Step 2: If intersections exist, use them
        if temp_intersections:
            temp_min = (temp_intersections[0] + temp_intersections[1]) / 2 - 1
            temp_max = (temp_intersections[0] + temp_intersections[1]) / 2 + 1
        else:
            # No intersection - find weighted average with sensitivity
            temp_min, temp_max = self._calculate_weighted_range(
                [(p.temp_min, p.temp_max, p.sensitivity_temp) for p in valid_preferences]
            )
        
        if humidity_intersections:
            humidity_min = (humidity_intersections[0] + humidity_intersections[1]) / 2 - 5
            humidity_max = (humidity_intersections[0] + humidity_intersections[1]) / 2 + 5
        else:
            # No intersection - find weighted average
            humidity_min, humidity_max = self._calculate_weighted_range(
                [(p.humidity_min, p.humidity_max, p.sensitivity_humidity) for p in valid_preferences]
            )
        
        # For CO2, use weighted average with higher priority for sensitive users
        co2_values = [(p.co2_threshold, p.sensitivity_co2) for p in valid_preferences]
        co2_threshold = self._calculate_weighted_average(co2_values)
        
        # Calculate effectiveness score
        effectiveness_score = self._calculate_effectiveness_score(
            valid_preferences,
            temp_min, temp_max,
            co2_threshold,
            humidity_min, humidity_max
        )
        
        return CompromisePreference(
            user_count=len(valid_preferences),
            temp_min=round(temp_min, 1),
            temp_max=round(temp_max, 1),
            co2_threshold=round(co2_threshold),
            humidity_min=round(humidity_min, 1),
            humidity_max=round(humidity_max, 1),
            effectiveness_score=effectiveness_score
        )
    
    def _find_range_intersection(self, ranges: List[tuple]) -> tuple:
        """Find intersection of multiple ranges."""
        if not ranges:
            return None
        
        max_mins = [r[0] for r in ranges]
        min_maxs = [r[1] for r in ranges]
        
        intersection_min = max(max_mins)
        intersection_max = min(min_maxs)
        
        if intersection_min <= intersection_max:
            return (intersection_min, intersection_max)
        return None
    
    def _calculate_weighted_range(self, preferences: List[tuple]) -> tuple:
        """Calculate weighted average range with sensitivity."""
        if not preferences:
            return (0, 0)
        
        # Extract mins and maxs with weights
        mins = [(p[0], p[2]) for p in preferences]
        maxs = [(p[1], p[2]) for p in preferences]
        
        weighted_min = self._calculate_weighted_average(mins)
        weighted_max = self._calculate_weighted_average(maxs)
        
        # Ensure min < max
        if weighted_min >= weighted_max:
            center = (weighted_min + weighted_max) / 2
            weighted_min = center - 1
            weighted_max = center + 1
        
        return (weighted_min, weighted_max)
    
    def _calculate_weighted_average(self, values_with_weights: List[tuple]) -> float:
        """Calculate weighted average."""
        if not values_with_weights:
            return 0
        
        total_weight = sum(weight for _, weight in values_with_weights)
        if total_weight == 0:
            return sum(value for value, _ in values_with_weights) / len(values_with_weights)
        
        weighted_sum = sum(value * weight for value, weight in values_with_weights)
        return weighted_sum / total_weight
    
    def _calculate_effectiveness_score(self, preferences: List[UserPreference],
                                     temp_min: float, temp_max: float,
                                     co2_threshold: int,
                                     humidity_min: float, humidity_max: float) -> float:
        """Calculate how effective the compromise is for all users."""
        if not preferences:
            return 1.0
        
        total_dissatisfaction = 0
        max_possible_dissatisfaction = 0
        
        for pref in preferences:
            # Calculate dissatisfaction for each parameter
            # Temperature
            temp_center = (temp_min + temp_max) / 2
            user_temp_center = (pref.temp_min + pref.temp_max) / 2
            temp_dissatisfaction = abs(temp_center - user_temp_center) * pref.sensitivity_temp
            
            # CO2
            co2_dissatisfaction = abs(co2_threshold - pref.co2_threshold) * pref.sensitivity_co2
            
            # Humidity
            humidity_center = (humidity_min + humidity_max) / 2
            user_humidity_center = (pref.humidity_min + pref.humidity_max) / 2
            humidity_dissatisfaction = abs(humidity_center - user_humidity_center) * pref.sensitivity_humidity
            
            # Priority weights
            priority_weights = {
                'temp': 0.7,
                'co2': 1.0,
                'humidity': 0.5
            }
            
            # Calculate weighted dissatisfaction
            weighted_dissatisfaction = (
                temp_dissatisfaction * priority_weights['temp'] +
                co2_dissatisfaction / 50 * priority_weights['co2'] +  # Scale CO2 to similar range
                humidity_dissatisfaction / 5 * priority_weights['humidity']
            )
            
            total_dissatisfaction += weighted_dissatisfaction
            
            # Calculate maximum possible dissatisfaction (worst case)
            max_temp_diff = max(abs(15 - user_temp_center), abs(30 - user_temp_center))
            max_co2_diff = max(abs(400 - pref.co2_threshold), abs(1500 - pref.co2_threshold))
            max_humidity_diff = max(abs(10 - user_humidity_center), abs(80 - user_humidity_center))
            
            max_possible = (
                max_temp_diff * pref.sensitivity_temp * priority_weights['temp'] +
                max_co2_diff / 50 * pref.sensitivity_co2 * priority_weights['co2'] +
                max_humidity_diff / 5 * pref.sensitivity_humidity * priority_weights['humidity']
            )
            max_possible_dissatisfaction += max_possible
        
        # Convert dissatisfaction to effectiveness score (0-1)
        if max_possible_dissatisfaction == 0:
            return 1.0
        
        effectiveness = 1 - (total_dissatisfaction / max_possible_dissatisfaction)
        return max(0.0, min(1.0, effectiveness))
    
    def get_average_preferences(self, list_of_user_ids: List[int] = None) -> UserPreference:
        """Calculate average preferences for a group of users (legacy method)."""
        if list_of_user_ids is None:
            list_of_user_ids = list(self.preferences.keys())
        
        if not list_of_user_ids:
            return UserPreference(user_id=0, username="default")
        
        # Calculate averages
        avg_temp_min = 0
        avg_temp_max = 0
        avg_co2_threshold = 0
        avg_humidity_min = 0
        avg_humidity_max = 0
        avg_sensitivity_temp = 0
        avg_sensitivity_co2 = 0
        avg_sensitivity_humidity = 0
        count = 0
        
        for user_id in list_of_user_ids:
            if user_id in self.preferences:
                pref = self.preferences[user_id]
                avg_temp_min += pref.temp_min
                avg_temp_max += pref.temp_max
                avg_co2_threshold += pref.co2_threshold
                avg_humidity_min += pref.humidity_min
                avg_humidity_max += pref.humidity_max
                avg_sensitivity_temp += pref.sensitivity_temp
                avg_sensitivity_co2 += pref.sensitivity_co2
                avg_sensitivity_humidity += pref.sensitivity_humidity
                count += 1
        
        if count == 0:
            return UserPreference(user_id=0, username="default")
        
        # Create and return average preference
        return UserPreference(
            user_id=0,
            username="average",
            temp_min=round(avg_temp_min / count, 1),
            temp_max=round(avg_temp_max / count, 1),
            co2_threshold=round(avg_co2_threshold / count),
            humidity_min=round(avg_humidity_min / count, 1),
            humidity_max=round(avg_humidity_max / count, 1),
            sensitivity_temp=round(avg_sensitivity_temp / count, 2),
            sensitivity_co2=round(avg_sensitivity_co2 / count, 2),
            sensitivity_humidity=round(avg_sensitivity_humidity / count, 2)
        )
    
    def add_feedback(self, user_id: int, feedback_type: str, sensor_data: Dict):
        """Add user feedback record."""
        feedback = FeedbackRecord(
            user_id=user_id,
            feedback_type=feedback_type,
            sensor_data=sensor_data.copy(),
            timestamp=datetime.now().isoformat()
        )
        
        self.feedback_history.append(feedback)
        
        # Keep only last 1000 records
        if len(self.feedback_history) > 1000:
            self.feedback_history = self.feedback_history[-1000:]
        
        self._save_feedback()
        logger.info(f"Added feedback from user {user_id}: {feedback_type}")
    
    def update_preference_from_feedback(self, user_id: int, feedback_type: str, current_sensor_data: Dict):
        """Update user preferences based on feedback."""
        try:
            preference = self.get_user_preference(user_id)
            
            # Get current sensor values
            current_temp = current_sensor_data.get("scd41", {}).get("temperature")
            current_co2 = current_sensor_data.get("scd41", {}).get("co2")
            current_humidity = current_sensor_data.get("scd41", {}).get("humidity")
            
            # Adjust preferences based on feedback
            if feedback_type in ["too_hot", "too_cold"] and current_temp is not None:
                preference.adjust_temp_preference(feedback_type, current_temp)
            elif feedback_type == "stuffy" and current_co2 is not None:
                preference.adjust_co2_preference(feedback_type, current_co2)
            elif feedback_type in ["too_dry", "too_humid"] and current_humidity is not None:
                preference.adjust_humidity_preference(feedback_type, current_humidity)
            elif feedback_type == "comfortable":
                if current_temp is not None:
                    preference.adjust_temp_preference("comfortable", current_temp)
                if current_co2 is not None:
                    preference.adjust_co2_preference("comfortable", current_co2)
                if current_humidity is not None:
                    preference.adjust_humidity_preference("comfortable", current_humidity)
            
            # Save updated preferences
            self._save_preferences()
            
            # Log the adjustment
            logger.info(f"Updated preferences for user {user_id} based on {feedback_type} feedback")
            
            # Add feedback to history
            self.add_feedback(user_id, feedback_type, current_sensor_data)
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating preference from feedback: {e}")
            return False
    
    def get_user_feedback_history(self, user_id: int, limit: int = 10) -> List[FeedbackRecord]:
        """Get recent feedback history for a user."""
        user_feedback = [f for f in self.feedback_history if f.user_id == user_id]
        return user_feedback[-limit:]
    
    def get_preference_summary(self, user_id: int) -> Dict:
        """Get a summary of user's preferences and recent feedback."""
        preference = self.get_user_preference(user_id)
        recent_feedback = self.get_user_feedback_history(user_id, 5)
        
        return {
            "preferences": preference.to_dict(),
            "recent_feedback": [f.to_dict() for f in recent_feedback],
            "feedback_count": len([f for f in self.feedback_history if f.user_id == user_id])
        }