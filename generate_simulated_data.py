#!/usr/bin/env python3
# generate_simulated_data.py
"""
Generate simulated historical data for ventilation system.

This script creates a CSV file with simulated measurements of indoor conditions
and ventilation system actions over time. The data includes timestamp, CO2 levels,
temperature, humidity, occupancy, and ventilation speed.

The simulation includes:
- Daily and weekly patterns of human activity
- Seasonal variations in temperature and humidity
- Realistic CO2 dynamics based on occupancy and ventilation
- Simple ventilation control logic based on environmental conditions

Usage:
    python generate_simulated_data.py

Output:
    Creates a file named 'simulated_ventilation_history.csv' in the current directory.

Configuration:
    Adjust the simulation parameters at the top of this file to customize the simulation.
"""
import csv
import random
import math
from datetime import datetime, timedelta

# Simulation parameters
SIMULATION_DAYS = 365
TIME_STEP_MINUTES = 10
MAX_OCCUPANTS = 3
CO2_GENERATION_PER_PERSON_PER_HOUR = 300  # ppm per person per hour
CO2_NATURAL_DECAY_RATE_PER_HOUR = 75  # ppm per hour due to natural infiltration
VENTILATION_EFFECTIVENESS = {
    "low": 200,      # ppm reduction per hour
    "medium": 600, 
    "max": 1056
}
INITIAL_CO2 = 450  # ppm
INITIAL_TEMP = 21.5  # °C
BASE_OUTDOOR_TEMP_DAY = 22.0  # °C - will be adjusted for seasonal variation
BASE_OUTDOOR_TEMP_NIGHT = 15.0  # °C - will be adjusted for seasonal variation
TEMP_CHANGE_RATE_VENT_ON = 0.3  # °C per hour
TEMP_CHANGE_RATE_VENT_OFF = 0.1  # °C per hour
OUTPUT_FILE = "simulated_ventilation_history.csv"

# Ventilation threshold parameters
CO2_THRESHOLD_HIGH = 1200  # ppm
CO2_THRESHOLD_MEDIUM = 900  # ppm 
CO2_THRESHOLD_LOW = 700  # ppm
TEMP_THRESHOLD_HIGH = 25.0  # °C
NIGHT_MODE_START_HOUR = 23
NIGHT_MODE_END_HOUR = 7

def get_seasonal_temperature_adjustment(day_of_year, base_temp):
    """
    Adjust temperature based on season.
    
    Args:
        day_of_year: Day of year (0-365)
        base_temp: Base temperature to adjust
        
    Returns:
        float: Adjusted temperature
    """
    # Simple sine wave seasonal variation with peak in summer (day ~180)
    seasonal_variation = 6.0 * math.sin((day_of_year / 365.0) * 2 * math.pi - math.pi/2)
    return base_temp + seasonal_variation

def get_occupancy(current_datetime, max_occupants):
    """
    Determine occupancy based on time of day and day of week.
    
    Args:
        current_datetime: Current datetime in simulation
        max_occupants: Maximum number of occupants
        
    Returns:
        int: Number of occupants
    """
    hour = current_datetime.hour
    day_of_week = current_datetime.weekday()  # 0=Monday, 6=Sunday
    is_weekend = day_of_week >= 5
    
    # Check for vacation periods or holidays (simplified)
    day_of_year = current_datetime.timetuple().tm_yday
    is_holiday_season = (350 <= day_of_year <= 365) or (1 <= day_of_year <= 7)  # Winter holidays
    is_summer_vacation = 180 <= day_of_year <= 220  # Summer vacation
    
    # Probability modifiers
    vacation_modifier = 0.7 if is_holiday_season or is_summer_vacation else 1.0
    
    # Base probabilities of being occupied
    if is_weekend:
        # Weekend pattern
        if 0 <= hour < 7:  # Night
            base_occupancy = random.choices([0, 1], weights=[0.8, 0.2])[0]
        elif 7 <= hour < 10:  # Morning
            base_occupancy = random.choices([0, 1, 2], weights=[0.1, 0.6, 0.3])[0]
        elif 10 <= hour < 20:  # Day
            base_occupancy = random.choices(
                list(range(max_occupants + 1)), 
                weights=[0.1] + [0.9/max_occupants] * max_occupants
            )[0]
        elif 20 <= hour < 23:  # Evening
            base_occupancy = random.choices([0, 1, 2], weights=[0.2, 0.5, 0.3])[0]
        else:  # Late night
            base_occupancy = random.choices([0, 1], weights=[0.7, 0.3])[0]
    else:
        # Weekday pattern
        if 0 <= hour < 7:  # Night
            base_occupancy = random.choices([0, 1], weights=[0.8, 0.2])[0]
        elif 7 <= hour < 9:  # Morning preparation
            base_occupancy = random.choices([0, 1, 2], weights=[0.2, 0.6, 0.2])[0]
        elif 9 <= hour < 17:  # Work hours
            base_occupancy = random.choices([0, 1], weights=[0.9, 0.1])[0]  # Mostly empty, sometimes someone stays
        elif 17 <= hour < 20:  # Evening return
            base_occupancy = random.choices([0, 1, 2, 3], weights=[0.1, 0.3, 0.4, 0.2])[0]
        elif 20 <= hour < 23:  # Evening
            base_occupancy = random.choices([0, 1, 2], weights=[0.2, 0.5, 0.3])[0]
        else:  # Late night
            base_occupancy = random.choices([0, 1], weights=[0.7, 0.3])[0]
    
    # Apply vacation/holiday modifier
    if random.random() > vacation_modifier:
        base_occupancy = 0  # Away during vacation periods
    
    # Add some randomness - occasionally have unexpected occupancy
    if random.random() < 0.03:  # 3% chance of deviation
        base_occupancy = random.randint(0, max_occupants)
    
    return min(base_occupancy, max_occupants)

def determine_ventilation_action(current_co2, current_temp, current_occupants, current_datetime):
    """
    Determine ventilation action based on CO2, temperature, and occupancy.
    
    Args:
        current_co2: Current CO2 level in ppm
        current_temp: Current temperature in °C
        current_occupants: Current number of occupants
        current_datetime: Current datetime
        
    Returns:
        str: Ventilation action ("off", "low", "medium", "max")
    """
    # Check if it's night mode
    hour = current_datetime.hour
    night_mode = (hour >= NIGHT_MODE_START_HOUR or hour < NIGHT_MODE_END_HOUR)
    
    # Emergency ventilation - high CO2 regardless of other factors
    if current_co2 > 1400:
        return "max"
    
    # During night mode, ventilation usually off unless CO2 is high
    if night_mode:
        if current_co2 > CO2_THRESHOLD_HIGH:
            return "medium"
        elif current_co2 > CO2_THRESHOLD_MEDIUM:
            return "low"
        else:
            return "off"
    
    # If no occupants, only ventilate if necessary
    if current_occupants == 0:
        if current_co2 > CO2_THRESHOLD_HIGH or current_temp > TEMP_THRESHOLD_HIGH + 1:
            return "medium"
        elif current_co2 > CO2_THRESHOLD_MEDIUM or current_temp > TEMP_THRESHOLD_HIGH:
            return "low"
        else:
            return "off"
    
    # With occupants, determine action based on CO2 and temperature
    if current_co2 > CO2_THRESHOLD_HIGH:
        return "max"
    elif current_co2 > CO2_THRESHOLD_MEDIUM:
        return "medium"
    elif current_co2 > CO2_THRESHOLD_LOW or current_temp > TEMP_THRESHOLD_HIGH:
        return "low"
    else:
        # Sometimes ventilate at low speed even if not strictly necessary
        if random.random() < 0.1:  # 10% chance
            return "low"
        return "off"

def main():
    """Run the ventilation data simulation."""
    # Initialize CSV file
    with open(OUTPUT_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'co2', 'temperature', 'humidity', 'occupants', 'ventilation_action'])
    
    # Initialize simulation state
    start_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=SIMULATION_DAYS)
    current_datetime = start_date
    current_co2 = INITIAL_CO2
    current_temp = INITIAL_TEMP
    
    # Main simulation loop
    end_datetime = start_date + timedelta(days=SIMULATION_DAYS)
    
    print(f"Starting simulation from {start_date.isoformat()} to {end_datetime.isoformat()}")
    
    while current_datetime < end_datetime:
        # Get day of year for seasonal adjustments
        day_of_year = current_datetime.timetuple().tm_yday
        
        # Determine occupancy
        current_occupants = get_occupancy(current_datetime, MAX_OCCUPANTS)
        
        # Determine ventilation action
        ventilation_action = determine_ventilation_action(
            current_co2, current_temp, current_occupants, current_datetime
        )
        
        # Calculate CO2 change
        time_factor = TIME_STEP_MINUTES / 60  # Convert minutes to hours
        
        # CO2 generated by occupants
        delta_co2_generation = CO2_GENERATION_PER_PERSON_PER_HOUR * time_factor * current_occupants
        
        # Natural CO2 decay (proportional to difference from baseline)
        delta_co2_decay = CO2_NATURAL_DECAY_RATE_PER_HOUR * time_factor * (current_co2 - 400) / 1000
        
        # CO2 reduction from ventilation (proportional to difference from baseline)
        delta_co2_ventilation = 0
        if ventilation_action != "off":
            delta_co2_ventilation = VENTILATION_EFFECTIVENESS[ventilation_action] * time_factor * (current_co2 - 400) / 1000
        
        # Update CO2
        current_co2 += delta_co2_generation - delta_co2_decay - delta_co2_ventilation
        
        # CO2 has a floor based on outdoor levels
        current_co2 = max(400, min(5000, current_co2))
        
        # Calculate temperature change
        # Determine target outdoor temperature based on time of day and season
        hour = current_datetime.hour
        is_daytime = 7 <= hour < 19
        
        base_outdoor_temp = BASE_OUTDOOR_TEMP_DAY if is_daytime else BASE_OUTDOOR_TEMP_NIGHT
        outdoor_temp = get_seasonal_temperature_adjustment(day_of_year, base_outdoor_temp)
        
        # Temperature change based on ventilation state
        if ventilation_action == "off":
            # When ventilation is off, temperature changes slowly
            temp_change_rate = TEMP_CHANGE_RATE_VENT_OFF
        else:
            # When ventilation is on, temperature changes faster toward outdoor temperature
            temp_change_rate = TEMP_CHANGE_RATE_VENT_ON
            
            # More aggressive ventilation speeds cause faster temperature change
            if ventilation_action == "medium":
                temp_change_rate *= 1.5
            elif ventilation_action == "max":
                temp_change_rate *= 2.0
        
        # Calculate temperature change
        temp_change = (outdoor_temp - current_temp) * temp_change_rate * time_factor
        
        # Add heat from occupants (each person adds about 0.1°C per hour)
        temp_change += current_occupants * 0.1 * time_factor
        
        # Small random variation
        temp_change += random.uniform(-0.05, 0.05) * time_factor
        
        # Update temperature
        current_temp += temp_change
        
        # Generate humidity (somewhat realistic but simplified)
        # Humidity tends to be higher at night and lower during the day
        base_humidity = 50  # Base humidity level
        time_variation = 5 if is_daytime else 10  # Day/night variation
        occupant_variation = current_occupants * 2  # Each person adds humidity
        ventilation_effect = -5 if ventilation_action != "off" else 0  # Ventilation reduces humidity
        
        # Seasonal variation - higher in summer, lower in winter
        seasonal_variation = 10 * math.sin((day_of_year / 365.0) * 2 * math.pi - math.pi/2)
        
        # Random variation
        humidity_noise = random.uniform(-5, 5)
        
        current_humidity = (
            base_humidity + 
            occupant_variation + 
            ventilation_effect - 
            time_variation + 
            seasonal_variation * 0.3 +  # Reduce the impact of seasonal variation
            humidity_noise
        )
        
        # Keep humidity within reasonable bounds
        current_humidity = max(30, min(70, current_humidity))
        
        # Round values for output
        output_co2 = round(current_co2)
        output_temp = round(current_temp, 1)
        output_humidity = round(current_humidity, 1)
        
        # Write to CSV
        with open(OUTPUT_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                current_datetime.isoformat(),
                output_co2,
                output_temp,
                output_humidity,
                current_occupants,
                ventilation_action
            ])
        
        # Advance time
        current_datetime += timedelta(minutes=TIME_STEP_MINUTES)
        
        # Progress indication (every 7 days)
        days_completed = (current_datetime - start_date).days
        if days_completed % 7 == 0 and current_datetime.hour == 0 and current_datetime.minute == 0:
            print(f"Simulation progress: {days_completed}/{SIMULATION_DAYS} days completed")

if __name__ == "__main__":
    main()
    print(f"Simulation complete. Data saved to {OUTPUT_FILE}")