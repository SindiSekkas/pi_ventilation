"""Ventilation simulation environment for comparing control algorithms."""
import os
import csv
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("ventilation_simulator")

class EnvironmentState:
    """Current state of the simulated environment."""
    
    def __init__(self, co2: float = 800.0, temperature: float = 21.5, 
                 humidity: float = 45.0, occupants: int = 0, timestamp: datetime = None):
        """Initialize environment state."""
        self.co2 = co2
        self.temperature = temperature
        self.humidity = humidity
        self.occupants = occupants
        self.ventilated = False
        self.ventilation_speed = "off"
        self.timestamp = timestamp or datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for logging."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "co2": self.co2,
            "temperature": self.temperature,
            "humidity": self.humidity,
            "occupants": self.occupants,
            "ventilated": self.ventilated,
            "ventilation_speed": self.ventilation_speed
        }
    
    def __str__(self) -> str:
        """String representation of state."""
        return (f"State[CO2={self.co2:.1f}ppm, Temp={self.temperature:.1f}°C, "
                f"Humidity={self.humidity:.1f}%, Occupants={self.occupants}, "
                f"Vent={'ON-'+self.ventilation_speed if self.ventilated else 'OFF'}]")


class VentilationSimulator:
    """Simulator for ventilation control algorithms."""
    
    # CO2 dynamics parameters
    CO2_GENERATION_PER_PERSON_PER_HOUR = 250    # ppm per person per hour
    CO2_NATURAL_DECAY_RATE_PER_HOUR = 300       # ppm per hour natural decrease
    CO2_BASELINE = 420                          # outdoor baseline CO2 level
    
    # Ventilation effectiveness (CO2 reduction rates)
    VENTILATION_EFFECTIVENESS = {
        "low": 200,      # ppm reduction per hour
        "medium": 890,   # ppm reduction per hour
        "max": 1236      # ppm reduction per hour
    }
    
    # Temperature dynamics
    VENTILATION_TEMP_IMPACT = {
        "low": 0.2,      # °C decrease per hour
        "medium": 0.8,   # °C decrease per hour
        "max": 1.2       # °C decrease per hour
    }
    TEMP_RISE_RATE = 0.1  # °C per hour ambient rise (heating)
    TEMP_RISE_PER_PERSON = 0.05  # Additional °C per hour per person
    
    # Humidity dynamics
    HUMIDITY_BASELINE = 40.0  # Baseline outdoor humidity
    
    # Energy consumption in watts
    ENERGY_CONSUMPTION = {
        "off": 0,
        "low": 15,      # 15W for low speed
        "medium": 30,   # 30W for medium speed
        "max": 60       # 60W for max speed
    }
    
    def __init__(self, output_dir: str = "simulation_results", time_step_minutes: int = 5):
        """
        Initialize the ventilation simulator.
        
        Args:
            output_dir: Directory for storing simulation results
            time_step_minutes: Simulation time step in minutes
        """
        self.output_dir = output_dir
        self.time_step_minutes = time_step_minutes
        self.time_step_hours = time_step_minutes / 60.0
        
        # Prepare output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize environment state
        self.state = EnvironmentState()
        
        # Logs for each controller
        self.controller_logs = {}
        
        # Performance metrics
        self.metrics = {}
    
    def reset(self, initial_state: Optional[EnvironmentState] = None) -> EnvironmentState:
        """
        Reset the simulation to initial state.
        
        Args:
            initial_state: Optional custom initial state
            
        Returns:
            EnvironmentState: Initial state
        """
        if initial_state:
            self.state = initial_state
        else:
            self.state = EnvironmentState()
            
        # Clear logs
        self.controller_logs = {}
        
        # Clear metrics
        self.metrics = {}
        
        return self.state
    
    def step(self, action: str, controller_name: str) -> EnvironmentState:
        """
        Advance simulation one time step with given action.
        
        Args:
            action: Ventilation action ("off", "low", "medium", "max")
            controller_name: Name of controller for logging
            
        Returns:
            EnvironmentState: New state
        """
        # Validate action
        if action not in ["off", "low", "medium", "max"]:
            logger.warning(f"Invalid action: {action}, defaulting to 'off'")
            action = "off"
        
        # Update timestamp
        self.state.timestamp += timedelta(minutes=self.time_step_minutes)
        
        # Store previous state for logging
        prev_state = self.state.to_dict()
        
        # Update ventilation status based on action
        self.state.ventilated = action != "off"
        self.state.ventilation_speed = action
        
        # Apply dynamics
        self._update_co2()
        self._update_temperature()
        self._update_humidity()
        
        # Log state transition for this controller
        if controller_name not in self.controller_logs:
            self.controller_logs[controller_name] = []
        
        # Calculate energy consumption for this step
        energy_consumed = self.ENERGY_CONSUMPTION[action] * (self.time_step_minutes / 60.0)
        
        # Add to controller log
        log_entry = {
            **self.state.to_dict(),
            "action": action,
            "energy_consumed": energy_consumed
        }
        self.controller_logs[controller_name].append(log_entry)
        
        return self.state
    
    def _update_co2(self):
        """Update CO2 level based on occupancy and ventilation."""
        # 1. CO2 generation from occupants
        co2_generation = self.CO2_GENERATION_PER_PERSON_PER_HOUR * self.state.occupants * self.time_step_hours
        
        # 2. CO2 reduction from ventilation (if active)
        co2_ventilation_reduction = 0
        if self.state.ventilated:
            co2_ventilation_reduction = self.VENTILATION_EFFECTIVENESS[self.state.ventilation_speed] * self.time_step_hours
        
        # 3. Natural CO2 decay toward baseline
        co2_natural_decay = min(
            self.CO2_NATURAL_DECAY_RATE_PER_HOUR * self.time_step_hours,  # Max possible decay
            max(0, self.state.co2 - self.CO2_BASELINE) * 0.3 * self.time_step_hours  # Proportional to difference from baseline
        )
        
        # Calculate new CO2 level
        self.state.co2 = max(
            self.CO2_BASELINE,  # Can't go below baseline
            self.state.co2 + co2_generation - co2_ventilation_reduction - co2_natural_decay
        )
    
    def _update_temperature(self):
        """Update temperature based on occupancy and ventilation."""
        # 1. Natural temperature rise (from heating/environment)
        temp_natural_rise = self.TEMP_RISE_RATE * self.time_step_hours
        
        # 2. Additional heat from occupants
        temp_occupant_rise = self.TEMP_RISE_PER_PERSON * self.state.occupants * self.time_step_hours
        
        # 3. Temperature decrease from ventilation (if active)
        temp_ventilation_decrease = 0
        if self.state.ventilated:
            temp_ventilation_decrease = self.VENTILATION_TEMP_IMPACT[self.state.ventilation_speed] * self.time_step_hours
        
        # Calculate new temperature with constraints
        self.state.temperature = max(
            18.0,  # Minimum temperature
            min(
                27.0,  # Maximum temperature
                self.state.temperature + temp_natural_rise + temp_occupant_rise - temp_ventilation_decrease
            )
        )
    
    def _update_humidity(self):
        """Update humidity based on occupancy and ventilation."""
        if self.state.ventilated:
            # Ventilation tends to lower humidity
            humidity_change = -2 * self.time_step_hours
        else:
            # With people, humidity rises, otherwise stays similar
            humidity_change = (0.5 * self.state.occupants) * self.time_step_hours
        
        # Add some randomness
        humidity_change += np.random.uniform(-0.5, 0.5) * self.time_step_hours
        
        # Apply humidity change
        self.state.humidity = max(30, min(70, self.state.humidity + humidity_change))
    
    def load_occupancy_pattern(self, pattern_file: str) -> bool:
        """
        Load occupancy pattern from generated CSV file.
        
        Args:
            pattern_file: Path to occupancy pattern CSV
            
        Returns:
            bool: Success indicator
        """
        try:
            # Load the CSV data
            df = pd.read_csv(pattern_file)
            
            if 'timestamp' not in df.columns or 'occupants' not in df.columns:
                logger.error(f"Invalid pattern file: {pattern_file} - missing required columns")
                return False
            
            # Convert timestamps
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Calculate occupancy for each time step 
            self.occupancy_pattern = df[['timestamp', 'occupants']].copy()
            
            logger.info(f"Loaded occupancy pattern with {len(df)} entries")
            return True
        
        except Exception as e:
            logger.error(f"Error loading occupancy pattern: {e}")
            return False
    
    def run_simulation(self, controllers: Dict[str, Any], days: int = 90) -> Dict[str, Dict[str, float]]:
        """
        Run simulation with multiple controllers over specified period.
        
        Args:
            controllers: Dictionary mapping controller names to controller objects
            days: Number of days to simulate
            
        Returns:
            Dict: Performance metrics for each controller
        """
        # Prepare for new simulation run
        start_datetime = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        end_datetime = start_datetime + timedelta(days=days)
        
        logger.info(f"Starting simulation from {start_datetime} to {end_datetime} ({days} days)")
        
        # Reset state and logs
        self.reset(EnvironmentState(timestamp=start_datetime))
        
        # Track each controller's actions and metrics
        for controller_name in controllers:
            self.controller_logs[controller_name] = []
            self.metrics[controller_name] = {
                "total_energy": 0.0,
                "avg_co2": 0.0,
                "time_co2_high": 0,
                "avg_temp": 0.0,
                "time_outside_comfort": 0,
                "comfort_score": 0.0,
                "high_co2_events": 0,
                "vent_cycles": 0,
                "vent_time": 0,
            }
        
        # Run simulation for each controller independently 
        for controller_name, controller in controllers.items():
            logger.info(f"Simulating controller: {controller_name}")
            
            # Reset state for this controller
            self.reset(EnvironmentState(timestamp=start_datetime))
            
            # Track previous action for counting ventilation cycles
            prev_action = "off"
            
            # Run through all time steps
            current_datetime = start_datetime
            step_count = 0
            
            while current_datetime < end_datetime:
                # Set current state timestamp
                self.state.timestamp = current_datetime
                
                # Update occupancy based on pattern
                if hasattr(self, 'occupancy_pattern'):
                    # Find closest timestamp in occupancy pattern
                    closest_idx = (self.occupancy_pattern['timestamp'] - current_datetime).abs().idxmin()
                    self.state.occupants = int(self.occupancy_pattern.iloc[closest_idx]['occupants'])
                
                # Get action from controller
                try:
                    data_for_controller = {
                        "scd41": {"co2": self.state.co2, "temperature": self.state.temperature, "humidity": self.state.humidity},
                        "room": {"occupants": self.state.occupants, "ventilated": self.state.ventilated, "ventilation_speed": self.state.ventilation_speed},
                    }
                    action = controller.decide_action(data_for_controller)
                except Exception as e:
                    logger.error(f"Error getting action from {controller_name}: {e}")
                    action = "off"
                
                # Apply action and update state
                self.step(action, controller_name)
                
                # Check for ventilation cycle change (off->on)
                if prev_action == "off" and action != "off":
                    self.metrics[controller_name]["vent_cycles"] += 1
                
                # Track ventilation time
                if action != "off":
                    self.metrics[controller_name]["vent_time"] += self.time_step_minutes
                
                # Track high CO2 events
                if self.state.co2 > 1200 and self.state.occupants > 0:
                    self.metrics[controller_name]["high_co2_events"] += 1
                
                # Track comfort metrics
                if (self.state.co2 > 1000 or self.state.temperature < 19.0 or 
                    self.state.temperature > 25.0) and self.state.occupants > 0:
                    self.metrics[controller_name]["time_outside_comfort"] += self.time_step_minutes
                
                prev_action = action
                current_datetime += timedelta(minutes=self.time_step_minutes)
                step_count += 1
                
                # Progress logging
                if step_count % 1000 == 0:
                    logger.info(f"Controller: {controller_name}, Steps: {step_count}, "
                                f"Date: {current_datetime.strftime('%Y-%m-%d')}, "
                                f"CO2: {self.state.co2:.1f}, Temp: {self.state.temperature:.1f}, "
                                f"Action: {action}")
            
            # Calculate metrics for this controller
            logs = self.controller_logs[controller_name]
            
            if logs:
                # Basic metrics
                self.metrics[controller_name]["total_energy"] = sum(log["energy_consumed"] for log in logs)
                self.metrics[controller_name]["avg_co2"] = sum(log["co2"] for log in logs) / len(logs)
                self.metrics[controller_name]["avg_temp"] = sum(log["temperature"] for log in logs) / len(logs)
                
                # Calculate time CO2 was high with people present
                high_co2_occupied_count = sum(1 for log in logs if log["co2"] > 1200 and log["occupants"] > 0)
                self.metrics[controller_name]["time_co2_high"] = high_co2_occupied_count * self.time_step_minutes
                
                # Calculate time outside comfort range with people present
                outside_comfort_count = sum(1 for log in logs if 
                    (log["co2"] > 1000 or log["temperature"] < 19.0 or log["temperature"] > 25.0) and log["occupants"] > 0)
                self.metrics[controller_name]["time_outside_comfort"] = outside_comfort_count * self.time_step_minutes
                
                # Calculate overall comfort score (lower is better)
                occupied_logs = [log for log in logs if log["occupants"] > 0]
                if occupied_logs:
                    co2_discomfort = sum(max(0, (log["co2"] - 800) / 400) for log in occupied_logs) / len(occupied_logs)
                    temp_discomfort = sum(
                        min(abs(log["temperature"] - 21) / 4, 1.0) for log in occupied_logs
                    ) / len(occupied_logs)
                    
                    self.metrics[controller_name]["comfort_score"] = (
                        (co2_discomfort * 0.7) + (temp_discomfort * 0.3)
                    )
            
            logger.info(f"Simulation complete for controller: {controller_name}")
            logger.info(f"Metrics: {self.metrics[controller_name]}")
            
            # Save logs to CSV
            self._save_controller_logs(controller_name)
        
        # Save overall metrics
        self._save_metrics()
        
        return self.metrics
    
    def _save_controller_logs(self, controller_name: str) -> bool:
        """
        Save controller logs to CSV file.
        
        Args:
            controller_name: Name of the controller
            
        Returns:
            bool: Success indicator
        """
        try:
            logs = self.controller_logs[controller_name]
            if not logs:
                return False
            
            output_file = os.path.join(self.output_dir, f"{controller_name}_logs.csv")
            
            # Extract all keys for headers
            headers = set()
            for log in logs:
                headers.update(log.keys())
            
            with open(output_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=sorted(headers))
                writer.writeheader()
                writer.writerows(logs)
            
            logger.info(f"Saved {len(logs)} logs for {controller_name} to {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving controller logs: {e}")
            return False
    
    def _save_metrics(self) -> bool:
        """
        Save performance metrics to JSON file.
        
        Returns:
            bool: Success indicator
        """
        try:
            output_file = os.path.join(self.output_dir, "performance_metrics.json")
            
            with open(output_file, 'w') as f:
                json.dump(self.metrics, f, indent=2)
            
            logger.info(f"Saved performance metrics to {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")
            return False