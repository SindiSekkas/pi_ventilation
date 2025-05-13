# simulation/occupants.py
"""
Occupant behavior model for adaptive ventilation system simulation.
Models realistic human behavior patterns including daily routines and sleep patterns.
"""
import logging
import random
from datetime import datetime, timedelta, time
import numpy as np
from enum import Enum, auto

logger = logging.getLogger(__name__)

class ActivityType(Enum):
    """Types of occupant activities."""
    SLEEPING = auto()
    AT_HOME = auto()
    AT_WORK = auto()
    AWAY = auto()  # Not at home or work
    GUEST = auto()  # Someone visiting

class OccupantBehaviorModel:
    """
    Models realistic human behavior patterns for ventilation simulation.
    
    Features:
    - Weekday vs weekend schedules
    - Sleep schedules (23:00-7:00 on weekdays, later on weekends)
    - Work schedules (8:00-17:00 on weekdays)
    - Weekend activities (going out, having guests)
    - Probabilistic behavior
    """
    
    def __init__(self, num_residents=2, start_date=None):
        """
        Initialize the occupant behavior model.
        
        Args:
            num_residents: Number of people living in the simulated space
            start_date: Custom start date (defaults to Jan 1, 2023)
        """
        self.num_residents = num_residents
        self.current_time = start_date or datetime(2023, 1, 1, 0, 0, 0)
        self.current_occupants = num_residents  # Start with everyone home
        self.num_guests = 0
        
        # Activity tracking
        self.resident_activities = [ActivityType.AT_HOME] * num_residents
        
        # Schedule parameters (with some randomness)
        self.schedules = []
        
        # Generate schedule for each resident
        for i in range(num_residents):
            # Base wake/sleep times with some variation between residents
            variation_minutes = random.randint(-15, 15)
            
            weekday_sleep_time = time(hour=23, minute=max(0, min(59, 0 + variation_minutes)))
            weekday_wake_time = time(hour=7, minute=max(0, min(59, 0 + variation_minutes)))
            
            weekend_sleep_time = time(hour=0, minute=max(0, min(59, 30 + variation_minutes)))
            weekend_wake_time = time(hour=9, minute=max(0, min(59, 0 + variation_minutes)))
            
            work_start_time = time(hour=8, minute=max(0, min(59, 0 + variation_minutes)))
            work_end_time = time(hour=17, minute=max(0, min(59, 0 + variation_minutes)))
            
            self.schedules.append({
                'weekday_sleep': weekday_sleep_time,
                'weekday_wake': weekday_wake_time,
                'weekend_sleep': weekend_sleep_time,
                'weekend_wake': weekend_wake_time,
                'work_start': work_start_time,
                'work_end': work_end_time,
                'work_commute_minutes': 30,
                'weekend_go_out_prob': 0.5,  # 50% chance to go out on weekends
                'weekend_guests_prob': 0.25  # 25% chance to have guests if staying home
            })
        
        # Event history tracking
        self.occupancy_history = []
        self.event_log = []
        
        # Variable to track weekends with guests
        self.weekend_has_guests = False
        self.weekend_out_of_home = False
        self.weekend_out_time = None
        self.weekend_return_time = None
        self.guest_arrival_time = None
        self.guest_departure_time = None
        
        # Record initial state
        self._record_state()
        
        logger.info(f"Initialized behavior model with {num_residents} residents")
    
    def _record_state(self):
        """Record current occupancy state."""
        total_occupants = self.current_occupants + self.num_guests
        
        # Store in occupancy history
        self.occupancy_history.append({
            'timestamp': self.current_time.isoformat(),
            'residents': self.current_occupants,
            'guests': self.num_guests,
            'total': total_occupants,
            'activities': [activity.name for activity in self.resident_activities],
            'weekday': self.current_time.weekday(),
            'is_weekend': self.current_time.weekday() >= 5
        })
        
        # Limit history size
        if len(self.occupancy_history) > 10000:
            self.occupancy_history = self.occupancy_history[-5000:]
    
    def _record_event(self, event_type, description, resident_idx=None):
        """
        Record a notable event in the simulation.
        
        Args:
            event_type: Type of event (e.g., 'sleep', 'wake', 'work')
            description: Text description of what happened
            resident_idx: Index of resident involved (if applicable)
        """
        self.event_log.append({
            'timestamp': self.current_time.isoformat(),
            'type': event_type,
            'description': description,
            'resident_idx': resident_idx
        })
        
        # Limit event log size
        if len(self.event_log) > 1000:
            self.event_log = self.event_log[-500:]
        
        logger.debug(f"[{self.current_time.strftime('%Y-%m-%d %H:%M')}] {description}")
    
    def is_weekend(self):
        """Check if current day is a weekend."""
        return self.current_time.weekday() >= 5  # 5=Saturday, 6=Sunday
    
    def is_weekday(self):
        """Check if current day is a weekday."""
        return self.current_time.weekday() < 5  # 0-4 = Monday-Friday
    
    def _add_time_variation(self, base_time, variance_minutes=15):
        """
        Add random variation to a schedule time.
        
        Args:
            base_time: Base time object
            variance_minutes: Maximum variation in minutes
            
        Returns:
            time: Time with added variation
        """
        # Random variation within range
        minutes_variation = random.randint(-variance_minutes, variance_minutes)
        
        # Convert to total minutes and add variation
        base_minutes = base_time.hour * 60 + base_time.minute
        new_minutes = base_minutes + minutes_variation
        
        # Convert back to hours and minutes
        new_hour = (new_minutes // 60) % 24
        new_minute = new_minutes % 60
        
        return time(hour=new_hour, minute=new_minute)
    
    def _plan_weekend_activities(self):
        """Plan weekend activities for all residents."""
        # Reset weekend variables 
        self.weekend_has_guests = False
        self.weekend_out_of_home = False
        self.weekend_out_time = None
        self.weekend_return_time = None
        self.guest_arrival_time = None
        self.guest_departure_time = None
        
        # Decide if going out as a group (50% chance)
        if random.random() < 0.5:
            self.weekend_out_of_home = True
            
            # Determine leaving and return times
            leave_hour = random.randint(10, 14)  # Leave between 10 AM and 2 PM
            duration_hours = random.randint(3, 6)  # Stay out for 3-6 hours
            
            self.weekend_out_time = time(hour=leave_hour, minute=random.randint(0, 59))
            return_hour = (leave_hour + duration_hours) % 24
            self.weekend_return_time = time(hour=return_hour, minute=random.randint(0, 59))
            
            logger.info(f"Weekend plan: Going out from {self.weekend_out_time} to {self.weekend_return_time}")
        
        # If staying home, decide if having guests (25% chance)
        elif random.random() < 0.25:
            self.weekend_has_guests = True
            self.num_guests = random.randint(1, 3)  # 1-3 guests
            
            # Determine arrival and departure times
            arrival_hour = random.randint(13, 18)  # Arrive between 1 PM and 6 PM
            duration_hours = random.randint(2, 5)  # Stay for 2-5 hours
            
            self.guest_arrival_time = time(hour=arrival_hour, minute=random.randint(0, 59))
            departure_hour = (arrival_hour + duration_hours) % 24
            self.guest_departure_time = time(hour=departure_hour, minute=random.randint(0, 59))
            
            logger.info(f"Weekend plan: Having {self.num_guests} guests from {self.guest_arrival_time} to {self.guest_departure_time}")
        
        else:
            logger.info("Weekend plan: Staying home, no guests")
    
    def update(self, time_step_minutes=1):
        """
        Update occupancy based on schedules and time progression.
        
        Args:
            time_step_minutes: Time step for the simulation in minutes
            
        Returns:
            dict: Current occupancy information
        """
        # Track occupancy changes
        old_occupants = self.current_occupants + self.num_guests
        
        # Advance the time
        self.current_time += timedelta(minutes=time_step_minutes)
        
        # Handle day transitions - plan weekend activities for the next day
        if self.current_time.hour == 0 and self.current_time.minute < time_step_minutes:
            # Plan weekend activities on Friday night for the weekend
            if self.current_time.weekday() == 5 and self.current_time.hour == 0:
                self._plan_weekend_activities()
        
        # Current time as minutes for easier comparison
        current_hour = self.current_time.hour
        current_minute = self.current_time.minute
        current_time_mins = current_hour * 60 + current_minute
        
        # Sunday night reset for new workweek
        if self.current_time.weekday() == 6 and current_hour >= 20:
            # Clear weekend flags for the upcoming workweek
            self.weekend_has_guests = False
            self.weekend_out_of_home = False
            self.weekend_out_time = None
            self.weekend_return_time = None
            self.guest_arrival_time = None
            self.guest_departure_time = None
            
            # Ensure all residents will be home for the night
            for i in range(self.num_residents):
                if self.resident_activities[i] != ActivityType.SLEEPING:
                    self.resident_activities[i] = ActivityType.AT_HOME
            
            self.current_occupants = sum(1 for activity in self.resident_activities 
                                        if activity in [ActivityType.AT_HOME, ActivityType.SLEEPING])
            self.num_guests = 0
            
        # Process each resident's schedule
        for i in range(self.num_residents):
            schedule = self.schedules[i]
            current_activity = self.resident_activities[i]
            
            # Get relevant schedule times with small daily variations
            if self.is_weekend():
                sleep_time = self._add_time_variation(schedule['weekend_sleep'])
                wake_time = self._add_time_variation(schedule['weekend_wake'])
                # No work on weekends
                work_start = None
                work_end = None
            else:
                sleep_time = self._add_time_variation(schedule['weekday_sleep'])
                wake_time = self._add_time_variation(schedule['weekday_wake'])
                work_start = self._add_time_variation(schedule['work_start'])
                work_end = self._add_time_variation(schedule['work_end'])
            
            # Convert times to minutes for easier comparison
            sleep_time_mins = sleep_time.hour * 60 + sleep_time.minute
            wake_time_mins = wake_time.hour * 60 + wake_time.minute
            
            # Handle overnight sleep time (sleep_time > wake_time)
            is_sleep_time = False
            if sleep_time_mins > wake_time_mins:
                # Sleep time crosses midnight
                is_sleep_time = current_time_mins >= sleep_time_mins or current_time_mins < wake_time_mins
            else:
                # Sleep time within same day
                is_sleep_time = sleep_time_mins <= current_time_mins < wake_time_mins
            
            # Process sleep/wake transition
            if is_sleep_time and current_activity != ActivityType.SLEEPING:
                # Go to sleep
                self.resident_activities[i] = ActivityType.SLEEPING
                self._record_event('sleep', f"Resident {i+1} went to sleep", i)
                
            elif not is_sleep_time and current_activity == ActivityType.SLEEPING:
                # Wake up
                self.resident_activities[i] = ActivityType.AT_HOME
                self._record_event('wake', f"Resident {i+1} woke up", i)
            
            # Process work schedule on weekdays
            if self.is_weekday() and not is_sleep_time:
                work_start_mins = work_start.hour * 60 + work_start.minute
                work_end_mins = work_end.hour * 60 + work_end.minute
                
                # Commute time
                commute_mins = schedule['work_commute_minutes']
                
                # Check if it's time to go to work
                if (current_activity == ActivityType.AT_HOME and 
                    current_time_mins >= (work_start_mins - commute_mins) and 
                    current_time_mins < work_end_mins):
                    
                    self.resident_activities[i] = ActivityType.AT_WORK
                    self._record_event('work', f"Resident {i+1} left for work", i)
                
                # Check if it's time to return from work
                elif (current_activity == ActivityType.AT_WORK and 
                     current_time_mins >= work_end_mins):
                    
                    self.resident_activities[i] = ActivityType.AT_HOME
                    self._record_event('return', f"Resident {i+1} returned from work", i)
            
            # Process weekend activities
            if self.is_weekend() and not is_sleep_time:
                # Going out
                if (self.weekend_out_of_home and self.weekend_out_time and self.weekend_return_time):
                    out_time_mins = self.weekend_out_time.hour * 60 + self.weekend_out_time.minute
                    return_time_mins = self.weekend_return_time.hour * 60 + self.weekend_return_time.minute
                    
                    # Handle overnight activities
                    is_out_time = False
                    if return_time_mins < out_time_mins:  # Crosses midnight
                        is_out_time = (current_time_mins >= out_time_mins or 
                                      current_time_mins < return_time_mins)
                    else:
                        is_out_time = out_time_mins <= current_time_mins < return_time_mins
                    
                    if is_out_time and current_activity == ActivityType.AT_HOME:
                        self.resident_activities[i] = ActivityType.AWAY
                        self._record_event('weekend', f"Resident {i+1} went out for weekend activity", i)
                    
                    elif not is_out_time and current_activity == ActivityType.AWAY:
                        self.resident_activities[i] = ActivityType.AT_HOME
                        self._record_event('weekend', f"Resident {i+1} returned home from weekend activity", i)
        
        # Process guests on weekends
        if self.is_weekend() and self.weekend_has_guests:
            if self.guest_arrival_time and self.guest_departure_time:
                arrival_mins = self.guest_arrival_time.hour * 60 + self.guest_arrival_time.minute
                departure_mins = self.guest_departure_time.hour * 60 + self.guest_departure_time.minute
                
                # Guests arrive
                if (current_time_mins == arrival_mins and self.num_guests > 0):
                    self._record_event('guests', f"{self.num_guests} guests arrived")
                
                # Guests leave
                if (current_time_mins == departure_mins and self.num_guests > 0):
                    self._record_event('guests', f"All guests departed")
                    self.num_guests = 0
        
        # Count residents physically at home
        self.current_occupants = sum(1 for activity in self.resident_activities 
                                    if activity in [ActivityType.AT_HOME, ActivityType.SLEEPING])
        
        # Record new state
        self._record_state()
        
        # Log significant occupancy changes
        new_occupants = self.current_occupants + self.num_guests
        if new_occupants != old_occupants:
            logger.info(f"Occupancy changed: {old_occupants} → {new_occupants} "
                      f"({self.current_time.strftime('%H:%M')})")
        
        return self.get_current_state()
    
    def get_current_state(self):
        """
        Get current occupancy state.
        
        Returns:
            dict: Occupancy information
        """
        return {
            'timestamp': self.current_time.isoformat(),
            'residents': self.current_occupants,
            'guests': self.num_guests,
            'total_occupants': self.current_occupants + self.num_guests,
            'activities': [activity.name for activity in self.resident_activities],
            'is_weekend': self.is_weekend()
        }
    
    def get_occupancy_history(self):
        """
        Get occupancy history data.
        
        Returns:
            list: History of occupancy states
        """
        return self.occupancy_history
    
    def get_occupancy_for_room_data(self):
        """
        Get occupancy data in format compatible with room data manager.
        
        Returns:
            dict: Room-formatted occupancy data
        """
        return {
            "occupants": self.current_occupants + self.num_guests
        }
    
    def get_event_log(self):
        """
        Get log of significant occupancy events.
        
        Returns:
            list: Event log entries
        """
        return self.event_log