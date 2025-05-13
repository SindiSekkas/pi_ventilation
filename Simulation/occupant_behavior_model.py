# simulation/occupant_behavior_model.py
"""
OccupantBehaviorModel for ventilation simulation system.
Generates realistic occupant presence and sleep schedules.
"""
import random
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional

logger = logging.getLogger(__name__)

class OccupantBehaviorModel:
    """
    Generates realistic occupant presence and sleep schedules over time.
    
    Provides schedules for both training and testing with configurable
    parameters for weekday/weekend patterns, sleep times, and guest visits.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize occupant behavior model with configuration parameters.
        
        Args:
            config: Dictionary of configuration parameters
        """
        # Set default parameters
        self.config = {
            # Simulation duration
            "SIMULATION_DAYS_TRAINING": 60,
            "SIMULATION_DAYS_TESTING": 30,
            "NUM_OCCUPANTS": 2,
            
            # Weekday Schedule
            "WEEKDAY_SLEEP_START_MEAN_HOUR": 23.0,
            "WEEKDAY_SLEEP_START_STD_MIN": 10,
            "WEEKDAY_WAKE_UP_MEAN_HOUR": 7.0,
            "WEEKDAY_WAKE_UP_STD_MIN": 10,
            "WEEKDAY_AWAY_START_MEAN_HOUR": 7.5,
            "WEEKDAY_AWAY_START_STD_MIN": 5,
            "WEEKDAY_AWAY_END_MEAN_HOUR": 17.5,
            "WEEKDAY_AWAY_END_STD_MIN": 15,
            
            # Weekend Schedule
            "WEEKEND_SLEEP_START_MEAN_HOUR": 0.5,  # 00:30
            "WEEKEND_SLEEP_START_STD_MIN": 20,
            "WEEKEND_WAKE_UP_MEAN_HOUR": 9.0,
            "WEEKEND_WAKE_UP_STD_MIN": 30,
            "WEEKEND_AWAY_PROBABILITY": 0.5,
            "WEEKEND_AWAY_DURATION_HOURS_MIN": 2,
            "WEEKEND_AWAY_DURATION_HOURS_MAX": 5,
            "WEEKEND_AWAY_TIME_WINDOW_START_HOUR": 12,
            "WEEKEND_AWAY_TIME_WINDOW_END_HOUR": 18,
            
            # Guest Visits
            "WEEKEND_GUEST_VISIT_PROBABILITY": 0.25,
            "WEEKEND_GUESTS_NUM_MIN": 1,
            "WEEKEND_GUESTS_NUM_MAX": 3,
            "WEEKEND_GUEST_VISIT_DURATION_HOURS_MIN": 2, 
            "WEEKEND_GUEST_VISIT_DURATION_HOURS_MAX": 4,
            "WEEKEND_GUEST_VISIT_TIME_WINDOW_START_HOUR": 18,
            "WEEKEND_GUEST_VISIT_TIME_WINDOW_END_HOUR": 22,
            
            # Time step in minutes for detailed schedule
            "TIME_STEP_MINUTES": 5
        }
        
        # Override defaults with provided config
        if config:
            for key, value in config.items():
                self.config[key] = value
        
        # Initialize schedules
        self.training_schedule = []
        self.testing_schedule = []
        self.current_time_idx = 0
        
        # Generate schedules
        self._generate_schedules()
    
    def _generate_schedules(self):
        """Generate both training and testing schedules."""
        self.training_schedule = self._generate_schedule(self.config["SIMULATION_DAYS_TRAINING"])
        self.testing_schedule = self._generate_schedule(self.config["SIMULATION_DAYS_TESTING"])
        
        logger.info(f"Generated {len(self.training_schedule)} training time steps and "
                   f"{len(self.testing_schedule)} testing time steps")
    
    def _generate_schedule(self, num_days: int) -> List[Dict[str, Any]]:
        """
        Generate occupancy schedule for a specific number of days.
        
        Args:
            num_days: Number of days to generate
            
        Returns:
            list: Schedule entries with timestamp, awake and sleeping occupants
        """
        schedule = []
        
        # Initialize start date/time
        start_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Generate per-occupant detailed schedules
        occupant_schedules = []
        for i in range(self.config["NUM_OCCUPANTS"]):
            occupant_schedules.append(self._generate_occupant_schedule(start_time, num_days, i))
        
        # Guest visits schedule (for weekends only)
        guest_schedule = self._generate_guest_visits(start_time, num_days)
        
        # Create complete schedule by combining all occupants and guests
        time_step = timedelta(minutes=self.config["TIME_STEP_MINUTES"])
        current_time = start_time
        
        for day in range(num_days):
            for hour in range(24):
                for minute in range(0, 60, self.config["TIME_STEP_MINUTES"]):
                    current_time = start_time + day * timedelta(days=1) + hour * timedelta(hours=1) + minute * timedelta(minutes=1)
                    
                    # Count occupants who are present and awake or sleeping
                    awake_count = 0
                    sleeping_count = 0
                    
                    for occ_schedule in occupant_schedules:
                        state = self._get_occupant_state(occ_schedule, current_time)
                        if state == "awake":
                            awake_count += 1
                        elif state == "sleeping":
                            sleeping_count += 1
                    
                    # Add guests if any
                    guest_count = self._get_guests_count(guest_schedule, current_time)
                    awake_count += guest_count
                    
                    # Create schedule entry
                    entry = {
                        "timestamp": current_time,
                        "day_of_week": current_time.weekday(),  # 0=Monday, 6=Sunday
                        "hour": current_time.hour,
                        "minute": current_time.minute,
                        "awake": awake_count,
                        "sleeping": sleeping_count,
                        "total": awake_count + sleeping_count
                    }
                    
                    schedule.append(entry)
        
        return schedule
    
    def _generate_occupant_schedule(self, start_time: datetime, num_days: int, 
                                   occupant_id: int) -> List[Dict[str, Any]]:
        """
        Generate detailed schedule for a single occupant.
        
        Args:
            start_time: Start time of schedule generation
            num_days: Number of days to generate
            occupant_id: Identifier for the occupant
            
        Returns:
            list: List of occupant's presence/sleep events
        """
        events = []
        current_date = start_time.date()
        
        for day in range(num_days):
            current_date = start_time.date() + timedelta(days=day)
            is_weekend = current_date.weekday() >= 5  # 5=Saturday, 6=Sunday
            
            # Sleep end (wake up) event
            if is_weekend:
                wake_mean = self.config["WEEKEND_WAKE_UP_MEAN_HOUR"]
                wake_std = self.config["WEEKEND_WAKE_UP_STD_MIN"] / 60
            else:
                wake_mean = self.config["WEEKDAY_WAKE_UP_MEAN_HOUR"]
                wake_std = self.config["WEEKDAY_WAKE_UP_STD_MIN"] / 60
            
            wake_hour = max(5, min(11, random.normalvariate(wake_mean, wake_std)))
            wake_minute = int((wake_hour % 1) * 60)
            wake_hour = int(wake_hour)
            
            wake_time = datetime.combine(
                current_date, 
                datetime.min.time().replace(hour=wake_hour, minute=wake_minute)
            )
            
            events.append({
                "type": "wake_up",
                "time": wake_time,
                "occupant": occupant_id
            })
            
            # Going away and returning events (weekdays - work)
            if not is_weekend:
                # Leave for work
                away_start_mean = self.config["WEEKDAY_AWAY_START_MEAN_HOUR"]
                away_start_std = self.config["WEEKDAY_AWAY_START_STD_MIN"] / 60
                
                away_hour = max(wake_hour + 0.3, min(10, random.normalvariate(away_start_mean, away_start_std)))
                away_minute = int((away_hour % 1) * 60)
                away_hour = int(away_hour)
                
                away_time = datetime.combine(
                    current_date, 
                    datetime.min.time().replace(hour=away_hour, minute=away_minute)
                )
                
                events.append({
                    "type": "leave",
                    "time": away_time,
                    "occupant": occupant_id
                })
                
                # Return from work
                return_mean = self.config["WEEKDAY_AWAY_END_MEAN_HOUR"]
                return_std = self.config["WEEKDAY_AWAY_END_STD_MIN"] / 60
                
                return_hour = max(away_hour + 6, min(20, random.normalvariate(return_mean, return_std)))
                return_minute = int((return_hour % 1) * 60)
                return_hour = int(return_hour)
                
                return_time = datetime.combine(
                    current_date, 
                    datetime.min.time().replace(hour=return_hour, minute=return_minute)
                )
                
                events.append({
                    "type": "return",
                    "time": return_time,
                    "occupant": occupant_id
                })
            
            # Weekend away time (optional)
            elif random.random() < self.config["WEEKEND_AWAY_PROBABILITY"]:
                # Random time window for going out
                window_start = self.config["WEEKEND_AWAY_TIME_WINDOW_START_HOUR"]
                window_end = self.config["WEEKEND_AWAY_TIME_WINDOW_END_HOUR"]
                
                away_hour = random.randint(window_start, window_end - 1)
                away_minute = random.choice([0, 15, 30, 45])
                
                away_time = datetime.combine(
                    current_date, 
                    datetime.min.time().replace(hour=away_hour, minute=away_minute)
                )
                
                # Random duration
                duration_hours = random.uniform(
                    self.config["WEEKEND_AWAY_DURATION_HOURS_MIN"],
                    self.config["WEEKEND_AWAY_DURATION_HOURS_MAX"]
                )
                
                return_time = away_time + timedelta(hours=duration_hours)
                
                # Ensure return time is before sleep time
                if is_weekend:
                    sleep_hour = self.config["WEEKEND_SLEEP_START_MEAN_HOUR"]
                else:
                    sleep_hour = self.config["WEEKDAY_SLEEP_START_MEAN_HOUR"]
                
                # Cap return time to 1 hour before typical sleep time
                max_return_hour = min(int(sleep_hour) - 1, 22)
                if return_time.hour > max_return_hour:
                    return_time = return_time.replace(hour=max_return_hour)
                
                events.append({
                    "type": "leave",
                    "time": away_time,
                    "occupant": occupant_id
                })
                
                events.append({
                    "type": "return",
                    "time": return_time,
                    "occupant": occupant_id
                })
            
            # Sleep start event
            if is_weekend:
                sleep_mean = self.config["WEEKEND_SLEEP_START_MEAN_HOUR"]
                sleep_std = self.config["WEEKEND_SLEEP_START_STD_MIN"] / 60
            else:
                sleep_mean = self.config["WEEKDAY_SLEEP_START_MEAN_HOUR"]
                sleep_std = self.config["WEEKDAY_SLEEP_START_STD_MIN"] / 60
            
            sleep_hour = max(21, min(25, random.normalvariate(sleep_mean, sleep_std)))
            sleep_minute = int((sleep_hour % 1) * 60)
            sleep_hour = int(sleep_hour % 24)  # Handle after midnight
            
            # If after midnight, it's the next day
            sleep_date = current_date
            if sleep_hour < 12 and sleep_mean > 12:
                sleep_date = current_date + timedelta(days=1)
            
            sleep_time = datetime.combine(
                sleep_date, 
                datetime.min.time().replace(hour=sleep_hour, minute=sleep_minute)
            )
            
            events.append({
                "type": "sleep",
                "time": sleep_time,
                "occupant": occupant_id
            })
        
        # Sort events by time
        events.sort(key=lambda x: x["time"])
        return events
    
    def _generate_guest_visits(self, start_time: datetime, num_days: int) -> List[Dict[str, Any]]:
        """
        Generate schedule of guest visits (mainly for weekends).
        
        Args:
            start_time: Start time of schedule generation
            num_days: Number of days to generate
            
        Returns:
            list: List of guest visit events (arrival, departure)
        """
        events = []
        current_date = start_time.date()
        
        for day in range(num_days):
            current_date = start_time.date() + timedelta(days=day)
            is_weekend = current_date.weekday() >= 5  # 5=Saturday, 6=Sunday
            
            # Weekends have higher chance of guest visits
            if is_weekend and random.random() < self.config["WEEKEND_GUEST_VISIT_PROBABILITY"]:
                # Generate number of guests
                num_guests = random.randint(
                    self.config["WEEKEND_GUESTS_NUM_MIN"],
                    self.config["WEEKEND_GUESTS_NUM_MAX"]
                )
                
                # Visit time window
                window_start = self.config["WEEKEND_GUEST_VISIT_TIME_WINDOW_START_HOUR"]
                window_end = self.config["WEEKEND_GUEST_VISIT_TIME_WINDOW_END_HOUR"]
                
                visit_hour = random.randint(window_start, window_end - 2)
                visit_minute = random.choice([0, 15, 30, 45])
                
                visit_time = datetime.combine(
                    current_date, 
                    datetime.min.time().replace(hour=visit_hour, minute=visit_minute)
                )
                
                # Visit duration
                duration_hours = random.uniform(
                    self.config["WEEKEND_GUEST_VISIT_DURATION_HOURS_MIN"],
                    self.config["WEEKEND_GUEST_VISIT_DURATION_HOURS_MAX"]
                )
                
                leave_time = visit_time + timedelta(hours=duration_hours)
                
                # Ensure leave time is before midnight
                if leave_time.hour >= 23:
                    leave_time = leave_time.replace(hour=23, minute=0)
                
                events.append({
                    "type": "guests_arrive",
                    "time": visit_time,
                    "num_guests": num_guests
                })
                
                events.append({
                    "type": "guests_leave",
                    "time": leave_time,
                    "num_guests": num_guests
                })
        
        # Sort events by time
        events.sort(key=lambda x: x["time"])
        return events
    
    def _get_occupant_state(self, occupant_schedule: List[Dict[str, Any]], 
                           current_time: datetime) -> str:
        """
        Determine occupant state at given time.
        
        Args:
            occupant_schedule: List of occupant events
            current_time: Time to check
            
        Returns:
            str: State ('away', 'awake', or 'sleeping')
        """
        # Get most recent event before current_time
        relevant_events = [
            event for event in occupant_schedule 
            if event["time"] <= current_time
        ]
        
        if not relevant_events:
            # Assume occupant starts sleeping (from the night before)
            return "sleeping"
        
        last_event = relevant_events[-1]
        
        if last_event["type"] == "leave":
            return "away"
        elif last_event["type"] == "sleep":
            return "sleeping"
        else:  # wake_up or return
            return "awake"
    
    def _get_guests_count(self, guest_schedule: List[Dict[str, Any]], 
                         current_time: datetime) -> int:
        """
        Determine number of guests at given time.
        
        Args:
            guest_schedule: List of guest events
            current_time: Time to check
            
        Returns:
            int: Number of guests
        """
        # Get most recent arrive and leave events
        guests_present = 0
        
        for event in guest_schedule:
            if event["time"] <= current_time:
                if event["type"] == "guests_arrive":
                    guests_present += event["num_guests"]
                elif event["type"] == "guests_leave":
                    guests_present -= event["num_guests"]
        
        return max(0, guests_present)  # Ensure non-negative
    
    def reset(self, mode="training"):
        """
        Reset the schedule to the beginning.
        
        Args:
            mode: 'training' or 'testing'
        """
        self.current_time_idx = 0
        logger.info(f"Reset occupant behavior model to beginning of {mode} schedule")
    
    def get_occupancy_status_at_time(self, timestamp: datetime) -> Tuple[int, int]:
        """
        Get occupancy status at a specific time.
        
        Args:
            timestamp: Time to check
            
        Returns:
            tuple: (num_awake, num_sleeping)
        """
        # Find closest time in schedule
        schedule = self.training_schedule
        if not schedule:
            return 0, 0
        
        # Find closest entry by date/time
        closest_entry = min(
            schedule, 
            key=lambda x: abs((x["timestamp"] - timestamp).total_seconds())
        )
        
        return closest_entry["awake"], closest_entry["sleeping"]
    
    def get_next_timestep(self, mode="training") -> Dict[str, Any]:
        """
        Advance to next time step and return occupancy data.
        
        Args:
            mode: 'training' or 'testing'
            
        Returns:
            dict: Timestep data with occupancy information
        """
        schedule = self.training_schedule if mode == "training" else self.testing_schedule
        
        if self.current_time_idx >= len(schedule):
            logger.warning(f"End of {mode} schedule reached. Resetting to beginning.")
            self.current_time_idx = 0
        
        data = schedule[self.current_time_idx]
        self.current_time_idx += 1
        
        return data
    
    def peek_ahead(self, steps: int = 1, mode="training") -> Dict[str, Any]:
        """
        Look ahead a certain number of time steps without advancing.
        
        Args:
            steps: Number of steps to look ahead
            mode: 'training' or 'testing'
            
        Returns:
            dict: Future timestep data
        """
        schedule = self.training_schedule if mode == "training" else self.testing_schedule
        future_idx = min(self.current_time_idx + steps, len(schedule) - 1)
        
        return schedule[future_idx]