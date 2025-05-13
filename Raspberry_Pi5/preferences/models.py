# preferences/models.py
"""User preference models for the ventilation system."""
from dataclasses import dataclass, asdict
from datetime import datetime
import json


@dataclass
class UserPreference:
    """User preference model for ventilation settings."""
    user_id: int
    username: str = None
    temp_min: float = 20.0
    temp_max: float = 24.0
    co2_threshold: int = 1000
    humidity_min: float = 30.0
    humidity_max: float = 60.0
    sensitivity_temp: float = 1.0  # Normal (1.0), reduced (0.5), or enhanced (2.0) sensitivity
    sensitivity_co2: float = 1.0
    sensitivity_humidity: float = 1.0
    created_at: str = None
    updated_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        self.updated_at = datetime.now().isoformat()
    
    def to_dict(self):
        """Convert preference to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data):
        """Create preference from dictionary."""
        return cls(**data)
    
    def update(self, **kwargs):
        """Update preference values."""
        for key, value in kwargs.items():
            if hasattr(self, key) and key != 'user_id':
                setattr(self, key, value)
        self.updated_at = datetime.now().isoformat()
    
    def adjust_temp_preference(self, feedback_type, current_temp):
        """Adjust temperature preferences based on user feedback."""
        adjustment = 0.5 if self.sensitivity_temp >= 1.0 else 0.25
        
        if feedback_type == "too_hot":
            self.temp_max = max(self.temp_max - adjustment, 15.0)
            self.temp_min = min(self.temp_min, self.temp_max - 1.0)
        elif feedback_type == "too_cold":
            self.temp_min = min(self.temp_min + adjustment, 30.0)
            self.temp_max = max(self.temp_max, self.temp_min + 1.0)
        elif feedback_type == "comfortable" and self.temp_min < current_temp < self.temp_max:
            # Expand comfort zone when user is satisfied with current conditions
            range_adjustment = 0.2
            self.temp_min = max(self.temp_min - range_adjustment, 15.0)
            self.temp_max = min(self.temp_max + range_adjustment, 30.0)
    
    def adjust_co2_preference(self, feedback_type, current_co2):
        """Adjust CO2 threshold based on user feedback."""
        adjustment = 50 if self.sensitivity_co2 >= 1.0 else 25
        
        if feedback_type == "stuffy":
            self.co2_threshold = max(self.co2_threshold - adjustment, 400)
        elif feedback_type == "comfortable" and current_co2 <= self.co2_threshold:
            # Increase threshold when user is comfortable with current CO2 levels
            self.co2_threshold = min(self.co2_threshold + adjustment, 1500)
    
    def adjust_humidity_preference(self, feedback_type, current_humidity):
        """Adjust humidity preferences based on user feedback."""
        adjustment = 5.0 if self.sensitivity_humidity >= 1.0 else 2.5
        
        if feedback_type == "too_dry":
            self.humidity_min = min(self.humidity_min + adjustment, 70.0)
            self.humidity_max = max(self.humidity_max, self.humidity_min + 5.0)
        elif feedback_type == "too_humid":
            self.humidity_max = max(self.humidity_max - adjustment, 20.0)
            self.humidity_min = min(self.humidity_min, self.humidity_max - 5.0)
        elif feedback_type == "comfortable" and self.humidity_min < current_humidity < self.humidity_max:
            # Expand acceptable humidity range when user is comfortable
            range_adjustment = 2.0
            self.humidity_min = max(self.humidity_min - range_adjustment, 10.0)
            self.humidity_max = min(self.humidity_max + range_adjustment, 80.0)


@dataclass
class FeedbackRecord:
    """Record of user comfort feedback."""
    user_id: int
    timestamp: str
    feedback_type: str  # Comfort states: comfortable, too_hot, too_cold, stuffy, too_dry, too_humid
    sensor_data: dict   # Environmental conditions at feedback time
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
    
    def to_dict(self):
        """Convert feedback to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data):
        """Create feedback from dictionary."""
        return cls(**data)


@dataclass
class CompromisePreference:
    """Calculated compromise preference based on multiple users' comfort needs."""
    user_count: int
    temp_min: float
    temp_max: float
    co2_threshold: int
    humidity_min: float
    humidity_max: float
    effectiveness_score: float  # Satisfaction metric from 0.0 (poor) to 1.0 (ideal)
    
    def to_dict(self):
        """Convert compromise preference to dictionary."""
        return asdict(self)


