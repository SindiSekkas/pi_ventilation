"""Preference manager for user ventilation preferences."""
import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Union
from .models import UserPreference, FeedbackRecord

logger = logging.getLogger(__name__)


class PreferenceManager:
    """Manages user preferences for ventilation settings."""
    
    def __init__(self, data_dir: str = "data/preferences"):
        """
        Initialize preference manager.
        
        Args:
            data_dir: Directory to store preference data
        """
        self.data_dir = data_dir
        self.preferences_file = os.path.join(data_dir, "user_preferences.json")
        self.feedback_file = os.path.join(data_dir, "user_feedback.json")
        
        # Create directory if not exists
        os.makedirs(data_dir, exist_ok=True)
        
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
    
    def get_average_preferences(self, list_of_user_ids: List[int] = None) -> UserPreference:
        """Calculate average preferences for a group of users."""
        if list_of_user_ids is None:
            # Use all users if no list provided
            list_of_user_ids = list(self.preferences.keys())
        
        if not list_of_user_ids:
            # Return default preferences if no users
            return UserPreference(user_id=0, username="default")
        
        # Calculate averages
        avg_temp_min = 0
        avg_temp_max = 0
        avg_co2_threshold = 0
        avg_humidity_min = 0
        avg_humidity_max = 0
        avg_sensitivity_temp = 0
        avg_sensitivity_co2 = 0
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
            sensitivity_co2=round(avg_sensitivity_co2 / count, 2)
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
    
    def update_preference_from_feedback(self, user_id: int, discomfort_type: str, current_sensor_data: Dict):
        """Update user preferences based on feedback."""
        try:
            preference = self.get_user_preference(user_id)
            
            # Get current sensor values
            current_temp = current_sensor_data.get("scd41", {}).get("temperature")
            current_co2 = current_sensor_data.get("scd41", {}).get("co2")
            
            # Adjust preferences based on feedback
            if discomfort_type in ["too_hot", "too_cold"] and current_temp is not None:
                preference.adjust_temp_preference(discomfort_type, current_temp)
            elif discomfort_type == "stuffy" and current_co2 is not None:
                preference.adjust_co2_preference(discomfort_type, current_co2)
            elif discomfort_type == "comfortable":
                if current_temp is not None:
                    preference.adjust_temp_preference("comfortable", current_temp)
                if current_co2 is not None:
                    preference.adjust_co2_preference("comfortable", current_co2)
            
            # Save updated preferences
            self._save_preferences()
            
            # Log the adjustment
            logger.info(f"Updated preferences for user {user_id} based on {discomfort_type} feedback")
            
            # Add feedback to history
            self.add_feedback(user_id, discomfort_type, current_sensor_data)
            
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