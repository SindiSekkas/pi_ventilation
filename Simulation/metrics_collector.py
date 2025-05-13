# simulation/metrics_collector.py
"""
Metrics collection for ventilation simulation.
Provides tools to track and compute performance metrics.
"""
import logging
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class MetricsCollector:
    """
    Collects and processes performance metrics during simulation.
    
    Tracks comfort, energy consumption, and prediction accuracy metrics
    for evaluating ventilation control strategies.
    """
    
    def __init__(self, simulation_duration_days: int = 1):
        """
        Initialize metrics collector.
        
        Args:
            simulation_duration_days: Duration of simulation in days
        """
        # Comfort metrics
        self.co2_values = []
        self.co2_comfort_periods = []
        self.co2_slight_discomfort_periods = []
        self.co2_high_discomfort_periods = []
        
        self.temp_values = []
        self.temp_comfort_periods = []
        self.temp_discomfort_periods = []
        
        self.humidity_values = []
        self.humidity_comfort_periods = []
        self.humidity_discomfort_periods = []
        
        # Energy metrics
        self.energy_consumption = 0.0
        self.energy_consumption_per_day = {}
        self.energy_consumption_per_hour = {}
        
        # Ventilation actions
        self.actions = []
        self.action_durations = {"off": 0, "low": 0, "medium": 0, "max": 0}
        self.action_switches = 0
        
        # Predictions
        self.occupancy_predictions = []
        self.sleep_predictions = []
        
        # Reward metrics
        self.reward_components = []
        self.total_rewards = []
        
        # Time tracking
        self.timestamps = []
        self.simulation_duration_days = simulation_duration_days
        self.time_step_minutes = 0  # Will be set when first data point is received
        
        # Occupancy data for reference
        self.actual_occupancy = []
    
    def reset(self):
        """Reset all metrics to initial values."""
        self.__init__(self.simulation_duration_days)
        logger.info("Metrics collector reset")
    
    def record_timestep(self, 
                       timestamp: datetime,
                       env_state: Dict[str, Any],
                       preferences: Dict[str, Any],
                       action: str,
                       reward_breakdown: Optional[Dict[str, float]] = None,
                       total_reward: Optional[float] = None,
                       occupancy_prediction: Optional[Dict[str, Any]] = None,
                       sleep_prediction: Optional[Dict[str, Any]] = None,
                       actual_occupants: int = 0):
        """
        Record metrics for a single simulation time step.
        
        Args:
            timestamp: Current timestamp
            env_state: Environment state (CO2, temp, humidity)
            preferences: User preferences
            action: Ventilation action taken
            reward_breakdown: Component-wise reward breakdown
            total_reward: Total reward value
            occupancy_prediction: Occupancy prediction data
            sleep_prediction: Sleep prediction data
            actual_occupants: Ground truth occupancy
        """
        # Record time step
        if not self.timestamps:
            # First time step
            self.timestamps.append(timestamp)
        else:
            # Calculate time step if not first point
            self.time_step_minutes = int((timestamp - self.timestamps[-1]).total_seconds() / 60)
            self.timestamps.append(timestamp)
        
        # Extract environmental values
        co2 = env_state.get("co2_ppm", 0)
        temp = env_state.get("temperature_c", 21.0)
        humidity = env_state.get("humidity_percent", 50.0)
        
        # Extract preference values
        co2_threshold = preferences.get("co2_threshold", 1000)
        temp_min = preferences.get("temp_min", 20.0)
        temp_max = preferences.get("temp_max", 24.0)
        humidity_min = preferences.get("humidity_min", 30.0)
        humidity_max = preferences.get("humidity_max", 60.0)
        
        # Record environmental values
        self.co2_values.append(co2)
        self.temp_values.append(temp)
        self.humidity_values.append(humidity)
        
        # Record comfort periods
        self._record_comfort_periods(co2, temp, humidity, 
                                  co2_threshold, temp_min, temp_max,
                                  humidity_min, humidity_max, self.time_step_minutes)
        
        # Record action and action switches
        if self.actions and self.actions[-1] != action:
            self.action_switches += 1
        self.actions.append(action)
        
        # Update action durations
        if self.time_step_minutes > 0:
            self.action_durations[action] += self.time_step_minutes
        
        # Record energy consumption
        energy = env_state.get("current_energy_w", 0) * (self.time_step_minutes / 60)
        self.energy_consumption += energy
        
        # Record energy by day and hour
        day_key = timestamp.strftime("%Y-%m-%d")
        hour_key = timestamp.strftime("%Y-%m-%d %H")
        
        if day_key not in self.energy_consumption_per_day:
            self.energy_consumption_per_day[day_key] = 0
        self.energy_consumption_per_day[day_key] += energy
        
        if hour_key not in self.energy_consumption_per_hour:
            self.energy_consumption_per_hour[hour_key] = 0
        self.energy_consumption_per_hour[hour_key] += energy
        
        # Record predictions if available
        if occupancy_prediction:
            self.occupancy_predictions.append(occupancy_prediction)
        
        if sleep_prediction:
            self.sleep_predictions.append(sleep_prediction)
        
        # Record rewards
        if reward_breakdown:
            self.reward_components.append(reward_breakdown)
        
        if total_reward is not None:
            self.total_rewards.append(total_reward)
        
        # Record actual occupancy
        self.actual_occupancy.append(actual_occupants)
    
    def _record_comfort_periods(self, co2: float, temp: float, humidity: float,
                             co2_threshold: float, temp_min: float, temp_max: float,
                             humidity_min: float, humidity_max: float, time_minutes: int):
        """
        Update comfort period metrics.
        
        Args:
            co2: Current CO2 level
            temp: Current temperature
            humidity: Current humidity
            co2_threshold: CO2 comfort threshold
            temp_min: Minimum comfortable temperature
            temp_max: Maximum comfortable temperature
            humidity_min: Minimum comfortable humidity
            humidity_max: Maximum comfortable humidity
            time_minutes: Duration of current time step
        """
        # CO2 comfort
        if co2 <= co2_threshold:
            self.co2_comfort_periods.append(time_minutes)
            logger.debug(f"CO2 comfort: {co2} <= {co2_threshold}")
        elif co2 <= co2_threshold * 1.2:
            self.co2_slight_discomfort_periods.append(time_minutes)
            logger.debug(f"CO2 slight discomfort: {co2} > {co2_threshold}")
        else:
            self.co2_high_discomfort_periods.append(time_minutes)
            logger.debug(f"CO2 high discomfort: {co2} > {co2_threshold * 1.2}")
        
        # Temperature comfort
        if temp_min <= temp <= temp_max:
            self.temp_comfort_periods.append(time_minutes)
            logger.debug(f"Temperature comfort: {temp_min} <= {temp} <= {temp_max}")
        else:
            self.temp_discomfort_periods.append(time_minutes)
            logger.debug(f"Temperature discomfort: temperature {temp} outside range [{temp_min}, {temp_max}]")
        
        # Humidity comfort
        if humidity_min <= humidity <= humidity_max:
            self.humidity_comfort_periods.append(time_minutes)
        else:
            self.humidity_discomfort_periods.append(time_minutes)
    
    def compute_metrics(self) -> Dict[str, Any]:
        """
        Compute final performance metrics.
        
        Returns:
            dict: Comprehensive metrics report
        """
        # Calculate comfort percentages
        total_minutes = sum(self.co2_comfort_periods + self.co2_slight_discomfort_periods + self.co2_high_discomfort_periods)
        if total_minutes > 0:
            co2_comfort_percent = 100 * sum(self.co2_comfort_periods) / total_minutes
            co2_slight_discomfort_percent = 100 * sum(self.co2_slight_discomfort_periods) / total_minutes
            co2_high_discomfort_percent = 100 * sum(self.co2_high_discomfort_periods) / total_minutes
        else:
            co2_comfort_percent = 0
            co2_slight_discomfort_percent = 0
            co2_high_discomfort_percent = 0
        
        total_temp_minutes = sum(self.temp_comfort_periods + self.temp_discomfort_periods)
        if total_temp_minutes > 0:
            temp_comfort_percent = 100 * sum(self.temp_comfort_periods) / total_temp_minutes
        else:
            temp_comfort_percent = 0
        
        total_humidity_minutes = sum(self.humidity_comfort_periods + self.humidity_discomfort_periods)
        if total_humidity_minutes > 0:
            humidity_comfort_percent = 100 * sum(self.humidity_comfort_periods) / total_humidity_minutes
        else:
            humidity_comfort_percent = 0
        
        # Calculate energy metrics
        daily_energy = self.energy_consumption / max(1, self.simulation_duration_days)
        energy_per_switch = self.energy_consumption / max(1, self.action_switches)
        
        # Calculate action percentages
        total_action_minutes = sum(self.action_durations.values())
        action_percentages = {}
        for action, duration in self.action_durations.items():
            if total_action_minutes > 0:
                action_percentages[action] = 100 * duration / total_action_minutes
            else:
                action_percentages[action] = 0
        
        # Calculate prediction accuracy metrics
        prediction_metrics = self._compute_prediction_metrics()
        
        # Calculate reward metrics
        reward_metrics = self._compute_reward_metrics()
        
        # CO2 and temperature statistics
        co2_stats = {
            "min": min(self.co2_values) if self.co2_values else 0,
            "max": max(self.co2_values) if self.co2_values else 0,
            "mean": np.mean(self.co2_values) if self.co2_values else 0,
            "median": np.median(self.co2_values) if self.co2_values else 0,
            "std": np.std(self.co2_values) if self.co2_values else 0,
            "pct95": np.percentile(self.co2_values, 95) if self.co2_values else 0
        }
        
        temp_stats = {
            "min": min(self.temp_values) if self.temp_values else 0,
            "max": max(self.temp_values) if self.temp_values else 0,
            "mean": np.mean(self.temp_values) if self.temp_values else 0,
            "median": np.median(self.temp_values) if self.temp_values else 0,
            "std": np.std(self.temp_values) if self.temp_values else 0
        }
        
        # Compile all metrics
        return {
            "comfort": {
                "co2": {
                    "comfort_percent": co2_comfort_percent,
                    "slight_discomfort_percent": co2_slight_discomfort_percent,
                    "high_discomfort_percent": co2_high_discomfort_percent,
                    "stats": co2_stats
                },
                "temperature": {
                    "comfort_percent": temp_comfort_percent,
                    "stats": temp_stats
                },
                "humidity": {
                    "comfort_percent": humidity_comfort_percent
                }
            },
            "energy": {
                "total_wh": self.energy_consumption,
                "daily_avg_wh": daily_energy,
                "wh_per_switch": energy_per_switch
            },
            "actions": {
                "total_switches": self.action_switches,
                "switches_per_day": self.action_switches / max(1, self.simulation_duration_days),
                "percentages": action_percentages
            },
            "predictions": prediction_metrics,
            "rewards": reward_metrics
        }
    
    def _compute_prediction_metrics(self) -> Dict[str, Any]:
        """
        Compute accuracy metrics for predictive models.
        
        Returns:
            dict: Prediction accuracy metrics
        """
        # Occupancy prediction accuracy
        occupancy_accuracy = 0
        if self.occupancy_predictions and self.actual_occupancy:
            correct_predictions = 0
            total_predictions = min(len(self.occupancy_predictions), len(self.actual_occupancy))
            
            for i in range(total_predictions):
                predicted = self.occupancy_predictions[i].get("predicted_occupancy", 0) > 0
                actual = self.actual_occupancy[i] > 0
                if predicted == actual:
                    correct_predictions += 1
            
            if total_predictions > 0:
                occupancy_accuracy = 100 * correct_predictions / total_predictions
        
        # Sleep prediction metrics (placeholder - would need reference data)
        sleep_accuracy = 0
        
        return {
            "occupancy": {
                "accuracy_percent": occupancy_accuracy
            },
            "sleep": {
                "accuracy_percent": sleep_accuracy  # Placeholder
            }
        }
    
    def _compute_reward_metrics(self) -> Dict[str, Any]:
        """
        Compute reward-related metrics.
        
        Returns:
            dict: Reward metrics
        """
        if not self.total_rewards:
            return {
                "total": 0,
                "mean": 0,
                "components": {}
            }
        
        # Calculate total and average rewards
        total_reward = sum(self.total_rewards)
        mean_reward = np.mean(self.total_rewards)
        
        # Calculate component-wise reward statistics
        component_metrics = {}
        
        if self.reward_components:
            # Collect all components
            components = {}
            for reward_dict in self.reward_components:
                for key, value in reward_dict.items():
                    if key not in components:
                        components[key] = []
                    components[key].append(value)
            
            # Calculate statistics for each component
            for key, values in components.items():
                component_metrics[key] = {
                    "total": sum(values),
                    "mean": np.mean(values),
                    "min": min(values),
                    "max": max(values)
                }
        
        return {
            "total": total_reward,
            "mean": mean_reward,
            "components": component_metrics
        }
    
    def get_hourly_energy_profile(self) -> Dict[str, float]:
        """
        Get energy consumption by hour of day.
        
        Returns:
            dict: Hour of day (0-23) to average energy consumption
        """
        hourly_profile = {}
        for hour in range(24):
            hourly_profile[hour] = 0
            count = 0
            
            # Sum energy for this hour across all days
            for hour_key, energy in self.energy_consumption_per_hour.items():
                if hour_key[-2:] == f"{hour:02d}":
                    hourly_profile[hour] += energy
                    count += 1
            
            # Calculate average for this hour
            if count > 0:
                hourly_profile[hour] /= count
        
        return hourly_profile
    
    def print_summary(self):
        """Print a summary of key metrics to the console."""
        metrics = self.compute_metrics()
        
        print("\n==== SIMULATION METRICS SUMMARY ====")
        print(f"Simulation Duration: {self.simulation_duration_days} days")
        
        print("\n--- Comfort Metrics ---")
        print(f"CO2 Comfort: {metrics['comfort']['co2']['comfort_percent']:.1f}%")
        print(f"Temperature Comfort: {metrics['comfort']['temperature']['comfort_percent']:.1f}%")
        print(f"Humidity Comfort: {metrics['comfort']['humidity']['comfort_percent']:.1f}%")
        
        print("\n--- Energy Metrics ---")
        print(f"Total Energy: {metrics['energy']['total_wh']:.1f} Wh")
        print(f"Daily Average: {metrics['energy']['daily_avg_wh']:.1f} Wh/day")
        
        print("\n--- Ventilation Actions ---")
        print(f"Total Action Switches: {metrics['actions']['total_switches']}")
        print(f"Action Distribution:")
        for action, percentage in metrics['actions']['percentages'].items():
            print(f"  {action}: {percentage:.1f}%")
        
        print("\n--- Prediction Accuracy ---")
        print(f"Occupancy Prediction: {metrics['predictions']['occupancy']['accuracy_percent']:.1f}%")
        
        print("\n--- Reward Metrics ---")
        print(f"Total Reward: {metrics['rewards']['total']:.1f}")
        print(f"Mean Reward: {metrics['rewards']['mean']:.2f}")
        
        print("\n--- Environmental Statistics ---")
        print(f"CO2 Range: {metrics['comfort']['co2']['stats']['min']:.0f} - " 
              f"{metrics['comfort']['co2']['stats']['max']:.0f} ppm " 
              f"(Mean: {metrics['comfort']['co2']['stats']['mean']:.0f} ppm)")
        print(f"Temperature Range: {metrics['comfort']['temperature']['stats']['min']:.1f} - " 
              f"{metrics['comfort']['temperature']['stats']['max']:.1f} °C " 
              f"(Mean: {metrics['comfort']['temperature']['stats']['mean']:.1f} °C)")
        
        print("\n==================================\n")