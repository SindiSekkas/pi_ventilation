#!/usr/bin/env python3
"""
Improved ventilation system data simulator with realistic occupancy patterns.
"""
import csv
import random
import math
import logging
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("data_generator")

# === SIMULATION PARAMETERS ===
SIMULATION_DAYS = 7
TIME_STEP_MINUTES = 2
MAX_OCCUPANTS = 2
OUTPUT_FILE = "simulated_ventilation_history.csv"

# CO2 dynamics - adjusted for less frequent ventilation
CO2_GENERATION_PER_PERSON_PER_HOUR = 250    # ppm per person per hour (slightly reduced)
CO2_NATURAL_DECAY_RATE_PER_HOUR = 300       # ppm per hour natural decrease
CO2_BASELINE = 420                          # outdoor baseline CO2 level

# Ventilation effectiveness (CO2 reduction rates)
VENTILATION_EFFECTIVENESS = {
    "low": 200,      # ppm reduction per hour
    "medium": 890,   # ppm reduction per hour
    "max": 1236      # ppm reduction per hour
}

# Temperature dynamics
INITIAL_TEMP = 21.5  # starting indoor temperature
VENTILATION_TEMP_IMPACT = {
    "low": 0.2,      # °C decrease per hour
    "medium": 0.8,   # °C decrease per hour
    "max": 1.2       # °C decrease per hour
}
TEMP_RISE_RATE = 0.1  # °C per hour ambient rise (heating)
TEMP_RISE_PER_PERSON = 0.05  # Additional °C per hour per person

# Ventilation thresholds
CO2_THRESHOLD_HIGH = 1200     # ppm - high threshold
CO2_THRESHOLD_MEDIUM = 950    # ppm - medium threshold
CO2_THRESHOLD_LOW = 800       # ppm - low threshold
TEMP_THRESHOLD_HIGH = 25.0    # °C - high temperature

# Occupancy management
MIN_OCCUPANCY_DURATION = 180  # minutes (3 hours)

# === OCCUPANCY TRACKING ===
class OccupancyManager:
    """Manages realistic occupancy patterns with minimum stay duration."""
    
    def __init__(self, max_occupants, min_duration_minutes):
        self.max_occupants = max_occupants
        self.min_duration_minutes = min_duration_minutes
        self.current_occupants = 0
        self.occupancy_changes = {}  # Key: person_id, Value: next allowed change time
        self.last_update_time = None

    def update(self, current_datetime):
        """Update occupancy levels based on time and previous state."""
        # Initialize on first call
        if self.last_update_time is None:
            # Start with everyone home for overnight period
            self.current_occupants = self.max_occupants
            for i in range(self.max_occupants):
                self.occupancy_changes[i] = current_datetime
            self.last_update_time = current_datetime
            return self.current_occupants
            
        # Check if we've passed any change time thresholds
        hour = current_datetime.hour
        day_of_week = current_datetime.weekday()  # 0=Monday, 6=Sunday
        is_weekend = day_of_week >= 5
        
        # Deep night: everyone is home (2 AM to 7 AM)
        if 2 <= hour < 7:
            # Everyone who's not already home should return
            for i in range(self.max_occupants):
                if i >= self.current_occupants:  # This person is "away"
                    if current_datetime >= self.occupancy_changes.get(i, current_datetime):
                        self.current_occupants += 1
                        self.occupancy_changes[i] = current_datetime + timedelta(minutes=self.min_duration_minutes)
            return self.current_occupants
            
        # Weekday work hours: typically empty (7 AM to 5 PM)
        if not is_weekend and 7 <= hour < 17:
            # Only allow people to leave during this period if they've met minimum stay
            if self.current_occupants > 0:
                for i in range(self.current_occupants-1, -1, -1):  # Check current occupants from highest to lowest
                    # Check if this person is allowed to leave
                    if current_datetime >= self.occupancy_changes.get(i, current_datetime):
                        # 75% chance of leaving during work hours if allowed
                        if random.random() < 0.75:
                            self.current_occupants -= 1
                            # Mark this person as away for at least the minimum duration
                            next_allowed = current_datetime + timedelta(minutes=self.min_duration_minutes)
                            # During work hours, extend to at least 5 PM
                            if next_allowed.hour < 17:
                                next_allowed = next_allowed.replace(hour=17, minute=0)
                            self.occupancy_changes[i] = next_allowed
            return self.current_occupants
            
        # Evening time (5 PM to 2 AM) - people return and may leave again
        if 17 <= hour < 24 or 0 <= hour < 2:
            # People away may return
            for i in range(self.max_occupants):
                # If this "slot" is available (person is away)
                if i >= self.current_occupants:
                    # Check if they're allowed to return yet
                    if current_datetime >= self.occupancy_changes.get(i, current_datetime):
                        # Higher probability of return in evening
                        p_return = 0.4 if 17 <= hour < 21 else 0.2
                        if random.random() < p_return:
                            self.current_occupants += 1
                            self.occupancy_changes[i] = current_datetime + timedelta(minutes=self.min_duration_minutes)
            
            # People home may leave (less likely)
            if self.current_occupants > 0:
                for i in range(self.current_occupants-1, -1, -1):
                    if current_datetime >= self.occupancy_changes.get(i, current_datetime):
                        # Lower probability of leaving in evening 
                        p_leave = 0.1
                        if random.random() < p_leave:
                            self.current_occupants -= 1
                            self.occupancy_changes[i] = current_datetime + timedelta(minutes=self.min_duration_minutes)
            return self.current_occupants
        
        # Weekend patterns
        if is_weekend:
            # On weekends, 50% chance someone leaves if they've been home long enough
            if self.current_occupants > 0:
                for i in range(self.current_occupants-1, -1, -1):
                    if current_datetime >= self.occupancy_changes.get(i, current_datetime):
                        # Only 10% chance per update that someone leaves
                        if random.random() < 0.1:  
                            self.current_occupants -= 1
                            self.occupancy_changes[i] = current_datetime + timedelta(minutes=self.min_duration_minutes)
            
            # People away may return
            for i in range(self.max_occupants):
                if i >= self.current_occupants:  # This person is away
                    if current_datetime >= self.occupancy_changes.get(i, current_datetime):
                        # Higher return probability on weekends
                        if random.random() < 0.3:
                            self.current_occupants += 1
                            self.occupancy_changes[i] = current_datetime + timedelta(minutes=self.min_duration_minutes)
            return self.current_occupants
            
        # Default: maintain current occupancy but respect duration constraints
        self.last_update_time = current_datetime
        return self.current_occupants


def determine_ventilation_action(current_co2, current_temp, current_occupants, night_mode=False, current_action="off"):
    """Determine appropriate ventilation action based on conditions."""
    # If no one is home, don't run ventilation
    if current_occupants == 0:
        return "off"
        
    # Emergency ventilation for very high CO2
    if current_co2 > 1500:
        return "max"
        
    # Night mode is more conservative - virtually never ventilate at night
    if night_mode:
        if current_co2 > CO2_THRESHOLD_HIGH + 200:  # Much higher tolerance at night
            return "low"  # Just low at night unless emergency
        return "off"
    
    # Normal occupied mode - but be more aggressive about turning off
    if current_co2 > CO2_THRESHOLD_HIGH:
        return "max"
    elif current_co2 > CO2_THRESHOLD_MEDIUM:
        # Only use medium if CO2 is actually high, or if we're already ventilating
        return "medium"
    elif current_co2 > CO2_THRESHOLD_LOW:
        # For LOW mode, add some randomness to prevent constant low operation
        if current_action == "off":
            # Only 50% chance to turn on low if we're currently off
            if random.random() < 0.5:
                return "low"
            return "off"
        return "low"  # Keep low if already running
    
    # CO2 is fine, check temperature
    if current_temp > TEMP_THRESHOLD_HIGH + 1.5:  # Higher threshold
        return "medium"  # Ventilate to cool if too warm
    elif current_temp > TEMP_THRESHOLD_HIGH + 0.5:  # Higher threshold
        # 50% chance for low if we're currently off
        if current_action == "off" and random.random() < 0.5:
            return "off"
        return "low"
        
    # If already ventilating but CO2 is now good, be more aggressive about stopping
    if current_action != "off" and current_co2 < CO2_THRESHOLD_LOW:
        if random.random() < 0.9:  # 90% chance to turn off (was 70%)
            return "off"
        return current_action
        
    # Default to off - all parameters are good
    return "off"


def main():
    """Run the ventilation data simulation with realistic patterns."""
    logger.info(f"Starting improved simulation for {SIMULATION_DAYS} days...")
    
    # Initialize CSV file
    with open(OUTPUT_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'co2', 'temperature', 'humidity', 'occupants', 'ventilation_action'])
    
    # Initialize simulation state
    start_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=SIMULATION_DAYS)
    current_datetime = start_date
    end_datetime = start_date + timedelta(days=SIMULATION_DAYS)
    
    # Initial environmental conditions
    current_co2 = 700.0  # Starting CO2 level
    current_temp = INITIAL_TEMP
    current_humidity = 45.0
    current_vent_action = "off"
    
    # Initialize occupancy manager
    occupancy_manager = OccupancyManager(MAX_OCCUPANTS, MIN_OCCUPANCY_DURATION)
    
    # Simulation loop
    while current_datetime < end_datetime:
        # Determine time-dependent parameters
        hour = current_datetime.hour
        night_mode = (hour >= 23 or hour < 7)  # Night mode hours
        
        # Update occupancy based on time
        occupants = occupancy_manager.update(current_datetime)
        
        # Calculate CO2 dynamics for this time step
        time_step_hours = TIME_STEP_MINUTES / 60.0
        
        # 1. CO2 generation from occupants
        co2_generation = CO2_GENERATION_PER_PERSON_PER_HOUR * occupants * time_step_hours
        
        # 2. CO2 reduction from ventilation (if active)
        co2_ventilation_reduction = 0
        if current_vent_action != "off":
            co2_ventilation_reduction = VENTILATION_EFFECTIVENESS[current_vent_action] * time_step_hours
        
        # 3. Natural CO2 decay toward baseline
        co2_natural_decay = min(
            CO2_NATURAL_DECAY_RATE_PER_HOUR * time_step_hours,  # Max possible decay
            max(0, current_co2 - CO2_BASELINE) * 0.3 * time_step_hours  # Proportional to difference from baseline
        )
        
        # Calculate new CO2 level
        current_co2 = max(
            CO2_BASELINE,  # Can't go below baseline
            current_co2 + co2_generation - co2_ventilation_reduction - co2_natural_decay
        )
        
        # Calculate temperature dynamics
        # 1. Natural temperature rise (from heating/environment)
        temp_natural_rise = TEMP_RISE_RATE * time_step_hours
        
        # 2. Additional heat from occupants
        temp_occupant_rise = TEMP_RISE_PER_PERSON * occupants * time_step_hours
        
        # 3. Temperature decrease from ventilation (if active)
        temp_ventilation_decrease = 0
        if current_vent_action != "off":
            temp_ventilation_decrease = VENTILATION_TEMP_IMPACT[current_vent_action] * time_step_hours
        
        # Calculate new temperature
        current_temp = max(
            18.0,  # Minimum temperature
            min(
                27.0,  # Maximum temperature
                current_temp + temp_natural_rise + temp_occupant_rise - temp_ventilation_decrease
            )
        )
        
        # Calculate humidity (simplified model)
        # Base humidity changes
        if current_vent_action != "off":
            # Ventilation tends to lower humidity
            humidity_change = -2 * time_step_hours
        else:
            # With people, humidity rises, otherwise stays similar
            humidity_change = (0.5 * occupants) * time_step_hours
        
        # Add some randomness
        humidity_change += random.uniform(-0.5, 0.5) * time_step_hours
        
        # Apply humidity change
        current_humidity = max(30, min(70, current_humidity + humidity_change))
        
        # Round values for output
        output_co2 = round(current_co2)
        output_temp = round(current_temp, 1)
        output_humidity = round(current_humidity, 1)
        
        # Write current state to CSV
        with open(OUTPUT_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                current_datetime.isoformat(),
                output_co2,
                output_temp,
                output_humidity,
                occupants,
                current_vent_action
            ])
        
        # Determine ventilation action for the next time step
        next_vent_action = determine_ventilation_action(
            current_co2, current_temp, occupants, 
            night_mode=night_mode,
            current_action=current_vent_action
        )
        
        # Update ventilation action for next iteration
        current_vent_action = next_vent_action
        
        # Progress logging
        if current_datetime.hour == 0 and current_datetime.minute == 0:
            logger.info(f"Simulating: {current_datetime.date()} | Occupants: {occupants} | CO2: {output_co2} ppm | Temp: {output_temp}°C")
        
        # Advance time
        current_datetime += timedelta(minutes=TIME_STEP_MINUTES)
    
    logger.info(f"Simulation complete. Data saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()