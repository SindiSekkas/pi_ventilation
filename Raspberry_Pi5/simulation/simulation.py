# simulation/simulation.py
"""
Main simulation coordinator for adaptive ventilation system.
Integrates environment, occupants, and ventilation components to run experiments.
"""
import os
import json
import logging
import csv
import time
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum

# Import simulation components
from simulation.environment import EnvironmentSimulator
from simulation.occupants import OccupantBehaviorModel
from simulation.ventilation import VentilationSystem, ControlStrategy, VentilationMode, VentilationSpeed

# Try to import actual system components for integration
try:
    from control.markov_controller import MarkovController
    from predictive.occupancy_pattern_analyzer import OccupancyPatternAnalyzer
    from predictive.adaptive_sleep_analyzer import AdaptiveSleepAnalyzer
    from preferences.preference_manager import PreferenceManager
    REAL_COMPONENTS_AVAILABLE = True
except ImportError:
    REAL_COMPONENTS_AVAILABLE = False
    print("Warning: Real system components not available. Using simplified simulation.")

logger = logging.getLogger(__name__)

class Simulation:
    """
    Main simulation coordinator for the adaptive ventilation system.
    
    Brings together environment, occupants, and ventilation components
    to run comprehensive simulations for comparing different strategies.
    """
    
    def __init__(self, 
                 output_dir: str = "simulation_results",
                 initial_date: datetime = None,
                 time_step_minutes: int = 5,
                 use_pretrained_markov: bool = True):
        """
        Initialize the simulation.
        
        Args:
            output_dir: Directory for storing simulation results
            initial_date: Starting date for simulation (default: Jan 1, 2023)
            time_step_minutes: Simulation time step in minutes
            use_pretrained_markov: Whether to use a pre-trained Markov model
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.initial_date = initial_date or datetime(2023, 1, 1, 0, 0, 0)
        self.time_step_minutes = time_step_minutes
        self.use_pretrained_markov = use_pretrained_markov
        
        # Create simulation components
        self.environment = EnvironmentSimulator()
        self.occupants = OccupantBehaviorModel(start_date=self.initial_date)
        self.ventilation = VentilationSystem(self.environment)
        
        # Simulation control
        self.running = False
        self.current_step = 0
        self.max_steps = 0
        
        # External components
        self.markov_controller = None
        self.occupancy_analyzer = None
        self.sleep_analyzer = None
        
        # Markov training parameters (used during setup_real_components)
        self.markov_explore_rate = 0.1  # Lower for evaluation, higher for training
        self.markov_learning_rate = 0.1
        
        # Experiment tracking
        self.experiments = []
        self.current_experiment = None
        
        # Data collection
        self.data_buffer = []
        self.buffer_max_size = 10000  # Flush to disk when buffer exceeds this size
        
        # Mock data manager for integration with real components
        self.mock_data_manager = None
        
        logger.info("Simulation initialized")
    
    def _prepare_for_json(self, obj):
        """Recursively convert objects to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._prepare_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._prepare_for_json(i) for i in obj]
        elif isinstance(obj, Enum):
            return obj.value  # Convert Enum to its string value
        elif hasattr(obj, 'isoformat'):  # Handle datetime objects
            return obj.isoformat()
        # Handle numpy numeric types
        elif hasattr(obj, 'dtype') and hasattr(obj, 'item'):  # NumPy scalar types
            return obj.item()  # Convert to native Python type
        elif str(type(obj)).startswith("<class 'numpy."):  # Fallback for other NumPy types
            return obj.tolist() if hasattr(obj, 'tolist') else float(obj)
        else:
            return obj
    
    def setup_experiment(self, 
                        name: str, 
                        strategy: ControlStrategy,
                        duration_days: float = 7.0,
                        initial_co2: float = 400.0,
                        initial_temp: float = 20.0,
                        description: str = None):
        """
        Configure a new simulation experiment.
        
        Args:
            name: Experiment identifier
            strategy: Ventilation control strategy to use
            duration_days: Duration in days
            initial_co2: Starting CO2 level in ppm
            initial_temp: Starting temperature in °C
            description: Optional experiment description
            
        Returns:
            dict: Experiment configuration
        """
        # Calculate total steps
        total_mins = duration_days * 24 * 60
        total_steps = int(total_mins / self.time_step_minutes)
        
        # Create experiment configuration
        experiment = {
            'id': len(self.experiments) + 1,
            'name': name,
            'strategy': strategy.name if isinstance(strategy, ControlStrategy) else strategy,
            'strategy_params': self.ventilation.parameters.get(
                f"{strategy.name.lower()}_strategy", {}
            ) if isinstance(strategy, ControlStrategy) else {},
            'duration_days': duration_days,
            'time_step_minutes': self.time_step_minutes,
            'total_steps': total_steps,
            'start_time': self.initial_date.isoformat(),
            'initial_conditions': {
                'co2': initial_co2,
                'temperature': initial_temp
            },
            'description': description or f"Testing {strategy} strategy",
            'use_real_components': REAL_COMPONENTS_AVAILABLE,
            'results': None
        }
        
        self.experiments.append(experiment)
        self.current_experiment = experiment
        
        # Create output directory for this experiment
        experiment_dir = os.path.join(self.output_dir, f"experiment_{experiment['id']}")
        os.makedirs(experiment_dir, exist_ok=True)
        experiment['output_dir'] = experiment_dir
        
        # Create CSV file for results
        csv_path = os.path.join(experiment_dir, "data.csv")
        experiment['csv_path'] = csv_path
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'co2', 'temperature', 'humidity',
                'occupants', 'ventilation_mode', 'ventilation_speed',
                'energy_consumption', 'outdoor_temp', 'step'
            ])
        
        # Save experiment configuration
        config_path = os.path.join(experiment_dir, "config.json")
        experiment['config_path'] = config_path
        
        with open(config_path, 'w') as f:
            # Create a copy without circular references
            config_data = experiment.copy()
            config_data.pop('results', None)  # Remove results field if present
            
            # Convert all Enum values to strings
            config_data = self._prepare_for_json(config_data)
            
            json.dump(config_data, f, indent=2)
        
        logger.info(f"Experiment {name} set up with {strategy} strategy "
                   f"for {duration_days} days ({total_steps} steps)")
        
        return experiment
    
    def _initialize_markov_q_values(self, controller):
        """
        Initialize Q-values for Markov controller with sensible defaults.
        
        This provides initial guidance to the controller before learning,
        especially important for states that are rarely encountered.
        
        Args:
            controller: MarkovController instance to initialize
        """
        # Create meaningful state-action pairs for common scenarios
        # Format: state_key: { action: q_value, ... }
        initial_q_values = {
            # Low CO2, Low Temp scenarios
            "low_low_empty_morning": {
                "off": 0.9,
                "low": 0.2,
                "medium": 0.1,
                "max": 0.05
            },
            "low_low_empty_day": {
                "off": 0.9,
                "low": 0.2,
                "medium": 0.1,
                "max": 0.05
            },
            "low_low_empty_evening": {
                "off": 0.9,
                "low": 0.2,
                "medium": 0.1,
                "max": 0.05
            },
            "low_low_empty_night": {
                "off": 0.95,
                "low": 0.1,
                "medium": 0.05,
                "max": 0.01
            },
            "low_low_occupied_morning": {
                "off": 0.8,
                "low": 0.5,
                "medium": 0.2,
                "max": 0.1
            },
            "low_low_occupied_day": {
                "off": 0.8,
                "low": 0.5,
                "medium": 0.2,
                "max": 0.1
            },
            "low_low_occupied_evening": {
                "off": 0.8,
                "low": 0.5,
                "medium": 0.2,
                "max": 0.1
            },
            "low_low_occupied_night": {
                "off": 0.95,
                "low": 0.2,
                "medium": 0.1,
                "max": 0.05
            },
            
            # Medium CO2, Medium Temp scenarios
            "medium_medium_empty_morning": {
                "off": 0.6,
                "low": 0.8,
                "medium": 0.4,
                "max": 0.2
            },
            "medium_medium_empty_day": {
                "off": 0.6,
                "low": 0.8,
                "medium": 0.4,
                "max": 0.2
            },
            "medium_medium_empty_evening": {
                "off": 0.6,
                "low": 0.8,
                "medium": 0.4,
                "max": 0.2
            },
            "medium_medium_empty_night": {
                "off": 0.95,
                "low": 0.3,
                "medium": 0.1,
                "max": 0.05
            },
            "medium_medium_occupied_morning": {
                "off": 0.3,
                "low": 0.9,
                "medium": 0.6,
                "max": 0.3
            },
            "medium_medium_occupied_day": {
                "off": 0.3,
                "low": 0.9,
                "medium": 0.6,
                "max": 0.3
            },
            "medium_medium_occupied_evening": {
                "off": 0.3,
                "low": 0.9,
                "medium": 0.6,
                "max": 0.3
            },
            "medium_medium_occupied_night": {
                "off": 0.8,
                "low": 0.4,
                "medium": 0.2,
                "max": 0.1
            },
            
            # High CO2, Medium Temp scenarios
            "high_medium_empty_morning": {
                "off": 0.1,
                "low": 0.6,
                "medium": 0.9,
                "max": 0.7
            },
            "high_medium_empty_day": {
                "off": 0.1,
                "low": 0.6,
                "medium": 0.9,
                "max": 0.7
            },
            "high_medium_empty_evening": {
                "off": 0.1,
                "low": 0.6,
                "medium": 0.9,
                "max": 0.7
            },
            "high_medium_empty_night": {
                "off": 0.5,
                "low": 0.8,
                "medium": 0.6,
                "max": 0.3
            },
            "high_medium_occupied_morning": {
                "off": 0.0,
                "low": 0.3,
                "medium": 0.8,
                "max": 0.9
            },
            "high_medium_occupied_day": {
                "off": 0.0,
                "low": 0.3,
                "medium": 0.8,
                "max": 0.9
            },
            "high_medium_occupied_evening": {
                "off": 0.0,
                "low": 0.3,
                "medium": 0.8,
                "max": 0.9
            },
            "high_medium_occupied_night": {
                "off": 0.4,
                "low": 0.7,
                "medium": 0.5,
                "max": 0.3
            },
            
            # High CO2, High Temp scenarios - prioritize max ventilation
            "high_high_occupied_morning": {
                "off": 0.0,
                "low": 0.1,
                "medium": 0.5,
                "max": 0.95
            },
            "high_high_occupied_day": {
                "off": 0.0,
                "low": 0.1,
                "medium": 0.5,
                "max": 0.95
            },
            "high_high_occupied_evening": {
                "off": 0.0,
                "low": 0.1,
                "medium": 0.5,
                "max": 0.95
            },
            "high_high_occupied_night": {
                "off": 0.3,
                "low": 0.5,
                "medium": 0.8,
                "max": 0.6
            }
        }
        
        # Add initial Q-values to the controller
        for state_key, action_values in initial_q_values.items():
            for action, q_value in action_values.items():
                # Add or update Q-value if not already set or is zero
                if state_key not in controller.q_values:
                    controller.q_values[state_key] = {}
                    
                if action not in controller.q_values.get(state_key, {}) or controller.q_values[state_key].get(action, 0.0) == 0.0:
                    controller.q_values[state_key][action] = q_value
        
        logger.info(f"Initialized Markov controller with {len(initial_q_values)} base state configurations")
    
    def _setup_real_components(self, experiment):
        """
        Set up integration with actual system components if available.
        
        Args:
            experiment: Current experiment configuration
        """
        if REAL_COMPONENTS_AVAILABLE:
            # Set up mock data manager for real components
            class MockDataManager:
                def __init__(self, simulation):
                    self.simulation = simulation
                    self.latest_data = {
                        "timestamp": simulation.environment.current_time.strftime("%Y-%m-%d %H:%M:%S"),
                        "scd41": {
                            "co2": simulation.environment.co2,
                            "temperature": simulation.environment.temperature,
                            "humidity": simulation.environment.humidity
                        },
                        "bmp280": {
                            "temperature": simulation.environment.temperature,
                            "pressure": 1013.25  # Standard pressure
                        },
                        "room": {
                            "occupants": simulation.occupants.current_occupants + simulation.occupants.num_guests,
                            "ventilated": simulation.ventilation.mode.value != "off",
                            "ventilation_speed": simulation.ventilation.speed.value
                        }
                    }
                
                def update_room_data(self, **kwargs):
                    """Mock room data update."""
                    for key, value in kwargs.items():
                        self.latest_data["room"][key] = value
                    return self.latest_data["room"]
            
            self.mock_data_manager = MockDataManager(self)
            
            # Create simulation directory structure
            sim_data_dir = os.path.join(self.output_dir, "sim_data")
            os.makedirs(sim_data_dir, exist_ok=True)
            
            # Initialize occupancy pattern analyzer if needed
            if experiment['strategy'] in [ControlStrategy.PREDICTIVE.name, ControlStrategy.MARKOV.name]:
                # Create history file
                occupancy_history_dir = os.path.join(sim_data_dir, "occupancy_history")
                os.makedirs(occupancy_history_dir, exist_ok=True)
                occupancy_history_file = os.path.join(occupancy_history_dir, "occupancy_history.csv")
                
                # Create empty file if it doesn't exist
                if not os.path.exists(occupancy_history_file):
                    with open(occupancy_history_file, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(['timestamp', 'status', 'people_count'])
                
                # Initialize OccupancyPatternAnalyzer
                self.occupancy_analyzer = OccupancyPatternAnalyzer(
                    occupancy_history_file=occupancy_history_file
                )
                
                logger.info("Integrated real OccupancyPatternAnalyzer")
            
            # Create preference manager for simulation
            preference_dir = os.path.join(sim_data_dir, "preferences")
            os.makedirs(preference_dir, exist_ok=True)
            sim_preference_manager = PreferenceManager(data_dir=preference_dir)
            
            # Add mock users for testing
            sim_preference_manager.set_user_preference(
                user_id=1, 
                username="SimUser1",
                temp_min=20.0,
                temp_max=24.0,
                co2_threshold=1000
            )
            sim_preference_manager.set_user_preference(
                user_id=2, 
                username="SimUser2",
                temp_min=21.0,
                temp_max=25.0,
                co2_threshold=950
            )
            
            # Only initialize components if using Markov strategy
            if experiment['strategy'] == ControlStrategy.MARKOV.name:
                # Create a real MarkovController
                try:
                    class MockPicoManager:
                        def __init__(self, ventilation_system):
                            self.ventilation = ventilation_system
                        
                        def get_ventilation_status(self):
                            return self.ventilation.mode.value != "off"
                        
                        def get_ventilation_speed(self):
                            return self.ventilation.speed.value
                        
                        def control_ventilation(self, state, speed=None):
                            if state == "on":
                                self.ventilation.set_mode(VentilationMode.MECHANICAL) # Use Enum
                                if speed:
                                    self.ventilation.set_speed(VentilationSpeed(speed)) # Use Enum
                            else:
                                self.ventilation.set_mode(VentilationMode.OFF) # Use Enum
                            return True
                    
                    # Create mock PicoManager for hardware control
                    mock_pico = MockPicoManager(self.ventilation)
                    
                    # Create MarkovController with higher exploration rate 
                    # and initialize with some Q-values
                    self.markov_controller = MarkovController(
                        data_manager=self.mock_data_manager,
                        pico_manager=mock_pico,
                        preference_manager=sim_preference_manager,
                        occupancy_analyzer=self.occupancy_analyzer,
                        model_dir=os.path.join(sim_data_dir, "markov"),
                        scan_interval=30,  # Faster updates for simulation
                        enable_exploration=self.markov_explore_rate > 0  # Enable if rate > 0
                    )
                    
                    # Set exploration rate higher to encourage more learning
                    self.markov_controller.exploration_rate = self.markov_explore_rate
                    self.markov_controller.learning_rate = self.markov_learning_rate
                    
                    # Initialize with basic Q-values to encourage ventilation
                    if self.use_pretrained_markov:
                        # Attempt to load a pre-trained model if specified
                        pretrained_model_path = os.path.join("trained_models", "markov_model_latest.json")
                        if os.path.exists(pretrained_model_path):
                            self.markov_controller.load_q_values(pretrained_model_path)
                            logger.info(f"Loaded pre-trained Markov model from {pretrained_model_path}")
                        else:
                            logger.warning(f"Pre-trained model not found at {pretrained_model_path}. Initializing with basic Q-values.")
                            self._initialize_markov_q_values(self.markov_controller)
                    else:
                        self._initialize_markov_q_values(self.markov_controller)
                    
                    # Initialize sleep analyzer only for Markov controller
                    try:
                        self.sleep_analyzer = AdaptiveSleepAnalyzer(
                            data_manager=self.mock_data_manager,
                            controller=self.markov_controller
                        )
                        # DO NOT START THE THREAD: self.sleep_analyzer.start()
                        logger.info("Integrated AdaptiveSleepAnalyzer with Markov controller")
                    except Exception as e:
                        logger.error(f"Failed to initialize AdaptiveSleepAnalyzer: {e}")
                        self.sleep_analyzer = None
                    
                    # DO NOT START THE THREAD: self.markov_controller.start()
                    logger.info(f"Integrated real MarkovController with exploration={self.markov_controller.exploration_rate}")

                except Exception as e:
                    logger.error(f"Failed to initialize MarkovController: {e}")
                    self.markov_controller = None
            else:
                # For other strategies, don't use Markov/SleepAnalyzer
                self.markov_controller = None
                self.sleep_analyzer = None
    
    def _update_mock_data_manager(self):
        """Update mock data manager with current system state."""
        if self.mock_data_manager:
            # Check if we have data to update with
            if not hasattr(self.environment, 'co2') or not hasattr(self.environment, 'temperature'):
                return
                
            try:
                self.mock_data_manager.latest_data = {
                    "timestamp": self.environment.current_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "scd41": {
                        "co2": self.environment.co2,
                        "temperature": self.environment.temperature,
                        "humidity": self.environment.humidity
                    },
                    "bmp280": {
                        "temperature": self.environment.temperature,
                        "pressure": 1013.25  # Standard pressure
                    },
                    "room": {
                        "occupants": self.occupants.current_occupants + self.occupants.num_guests,
                        "ventilated": self.ventilation.mode.value != "off",
                        "ventilation_speed": self.ventilation.speed.value
                    }
                }
            except Exception as e:
                logger.error(f"Error updating mock data manager: {e}")
                # Don't raise to avoid breaking the simulation
    
    def run_experiment(self, experiment=None):
        """
        Run the specified or current experiment.
        
        Args:
            experiment: Experiment configuration or ID (defaults to current)
            
        Returns:
            dict: Experiment results
        """
        # Select experiment to run
        if experiment is None:
            if self.current_experiment is None:
                raise ValueError("No experiment configured. Call setup_experiment first.")
            experiment = self.current_experiment
        elif isinstance(experiment, int):
            if experiment < 1 or experiment > len(self.experiments):
                raise ValueError(f"Invalid experiment ID: {experiment}")
            experiment = self.experiments[experiment - 1]
        
        logger.info(f"Starting experiment: {experiment['name']}")
        
        # Initialize environment
        self.environment.reset(
            initial_co2=experiment['initial_conditions']['co2'],
            initial_temp=experiment['initial_conditions']['temperature']
        )
        
        # Set experiment start date
        start_date = datetime.fromisoformat(experiment['start_time'])
        self.environment.current_time = start_date
        self.occupants = OccupantBehaviorModel(
            start_date=start_date,
            num_residents=2  # Fixed for this simulation
        )
        
        # Set up ventilation strategy
        self.ventilation = VentilationSystem(
            self.environment,
            strategy=ControlStrategy[experiment['strategy']]
        )
        
        # Set up real components if available
        self._setup_real_components(experiment)
        
        # Clear data buffer
        self.data_buffer = []
        
        # Run simulation
        self.running = True
        self.current_step = 0
        self.max_steps = experiment['total_steps']
        
        start_time = time.time()
        last_flush_time = start_time
        last_report_time = start_time
        
        try:
            while self.running and self.current_step < self.max_steps:
                # Update components
                self._simulate_step()
                
                # Report progress periodically
                current_time = time.time()
                if current_time - last_report_time > 5.0:  # Report every 5 seconds
                    progress = (self.current_step / self.max_steps) * 100
                    elapsed = current_time - start_time
                    
                    # Estimate completion time
                    if self.current_step > 0:
                        time_per_step = elapsed / self.current_step
                        remaining_steps = self.max_steps - self.current_step
                        estimated_remaining = time_per_step * remaining_steps
                        
                        logger.info(f"Experiment progress: {progress:.1f}% complete. "
                                   f"Step {self.current_step}/{self.max_steps}. "
                                   f"ETA: {timedelta(seconds=estimated_remaining)}")
                    
                    last_report_time = current_time
                
                # Flush data buffer periodically
                if len(self.data_buffer) >= self.buffer_max_size or (current_time - last_flush_time > 30.0):
                    self._flush_data_buffer(experiment['csv_path'])
                    last_flush_time = current_time
                
                # Check for early termination
                if not self.running:
                    logger.info("Experiment terminated early")
                    break
                
                self.current_step += 1
            
            # Ensure all data is flushed
            self._flush_data_buffer(experiment['csv_path'])
            
            # Calculate experiment results
            results = self._calculate_results(experiment['csv_path'])
            experiment['results'] = results
            
            # Save results
            results_path = os.path.join(experiment['output_dir'], "results.json")
            try:
                with open(results_path, 'w') as f:
                    # Handle Enum serialization in results
                    json_safe_results = self._prepare_for_json(results)
                    json.dump(json_safe_results, f, indent=2)
            except TypeError as e:
                logger.error(f"TypeError during JSON serialization: {e}")
                # Fall back to simpler result output without detailed data
                fallback_results = {
                    "energy_consumption": float(results.get('energy_consumption', 0)),
                    "avg_co2": float(results.get('avg_co2', 0)),
                    "max_co2": float(results.get('max_co2', 0)),
                    "note": "Some detailed data was omitted due to serialization error"
                }
                with open(results_path, 'w') as f:
                    json.dump(fallback_results, f, indent=2)
            
            # Generate plots
            self._generate_plots(experiment)
            
            logger.info(f"Experiment completed: {experiment['name']}")
            logger.info(f"Total energy consumption: {results['energy_consumption']:.2f} kWh")
            logger.info(f"Average CO2: {results['avg_co2']:.1f} ppm")
            
            return results
            
        except Exception as e:
            logger.error(f"Error during experiment: {e}", exc_info=True)
            return {"error": str(e)}
        finally:
            self.running = False
    
    def _simulate_step(self):
        """Execute a single simulation step."""
        # Update occupancy
        occupancy_data = self.occupants.update(self.time_step_minutes)
        
        # Current simulation time
        current_simulation_time = self.environment.current_time

        # Update mock data manager
        self._update_mock_data_manager()
        
        # Update external components that track data over time
        if self.sleep_analyzer:
            try:
                self.sleep_analyzer.update_co2_data(current_simulation_time) # Pass simulation time
            except Exception as e:
                logger.warning(f"Error in sleep analyzer during simulation step: {e}")
                # Capture error but continue simulation        
        
        if self.occupancy_analyzer:
            try:
                # Create status string
                status = "EMPTY" if occupancy_data['total_occupants'] == 0 else "OCCUPIED"
                
                # Record to occupancy history file
                occupancy_history_file = self.occupancy_analyzer.history_file
                with open(occupancy_history_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        self.environment.current_time.isoformat(),
                        status,
                        occupancy_data['total_occupants']
                    ])
            except Exception as e:
                logger.error(f"Error updating occupancy history: {e}")
        
        # Get sensor data
        sensor_data = self.environment.get_sensor_data()
        
        # If Markov strategy, let MarkovController make its decision first
        if self.current_experiment and self.current_experiment['strategy'] == ControlStrategy.MARKOV.name and self.markov_controller:
            try:
                self.markov_controller.make_step_decision(current_simulation_time) # Pass simulation time
            except Exception as e:
                logger.error(f"Error during MarkovController step decision: {e}")

        # Update ventilation system (it will use the state set by Markov if applicable,
        # or apply its own strategy otherwise)
        ventilation_state = self.ventilation.update(
            sensor_data=sensor_data,
            occupancy_data=occupancy_data,
            time_step_minutes=self.time_step_minutes,
            markov_controller=self.markov_controller, # Pass controller for logging/awareness
            occupancy_analyzer=self.occupancy_analyzer
        )
        
        # Update environment based on ventilation and occupancy
        self.environment.update(
            time_step_minutes=self.time_step_minutes,
            occupants=occupancy_data['total_occupants'],
            ventilation_mode=ventilation_state['mode'],
            ventilation_speed=ventilation_state['speed']
        )
        
        # Record data point
        self._record_data_point(sensor_data, occupancy_data, ventilation_state)
    
    def _record_data_point(self, sensor_data, occupancy_data, ventilation_state):
        """Record data point to buffer."""
        self.data_buffer.append({
            'timestamp': self.environment.current_time.isoformat(),
            'co2': sensor_data['scd41']['co2'],
            'temperature': sensor_data['scd41']['temperature'],
            'humidity': sensor_data['scd41']['humidity'],
            'occupants': occupancy_data['total_occupants'],
            'ventilation_mode': ventilation_state['mode'],
            'ventilation_speed': ventilation_state['speed'],
            'energy_consumption': ventilation_state['total_energy_consumption'],
            'outdoor_temp': self.environment._get_outdoor_temperature(),
            'step': self.current_step
        })
    
    def _flush_data_buffer(self, csv_path):
        """Flush data buffer to CSV file."""
        if not self.data_buffer:
            return
        
        try:
            with open(csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                for data in self.data_buffer:
                    writer.writerow([
                        data['timestamp'],
                        data['co2'],
                        data['temperature'],
                        data['humidity'],
                        data['occupants'],
                        data['ventilation_mode'],
                        data['ventilation_speed'],
                        data['energy_consumption'],
                        data['outdoor_temp'],
                        data['step']
                    ])
            
            logger.debug(f"Flushed {len(self.data_buffer)} data points to {csv_path}")
            self.data_buffer = []
        except Exception as e:
            logger.error(f"Error flushing data buffer: {e}")
    
    def _calculate_results(self, csv_path):
        """
        Calculate experiment results from data.
        
        Args:
            csv_path: Path to experiment CSV data
            
        Returns:
            dict: Calculated results
        """
        try:
            # Load data from CSV
            df = pd.read_csv(csv_path)
            
            # Ensure timestamps are parsed
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Calculate basic statistics
            results = {
                'energy_consumption': df['energy_consumption'].iloc[-1],  # Final value
                'avg_co2': df['co2'].mean(),
                'max_co2': df['co2'].max(),
                'min_co2': df['co2'].min(),
                'avg_temperature': df['temperature'].mean(),
                'co2_over_1000_pct': (df['co2'] > 1000).mean() * 100,  # % of time CO2 > 1000 ppm
                'co2_over_1200_pct': (df['co2'] > 1200).mean() * 100,  # % of time CO2 > 1200 ppm
                'time_with_occupants_pct': (df['occupants'] > 0).mean() * 100,  # % of time occupied
                'ventilation_on_pct': (df['ventilation_mode'] != 'off').mean() * 100,  # % of time ventilation on
                'ventilation_on_occupied_pct': df[df['occupants'] > 0]['ventilation_mode'].apply(
                    lambda x: x != 'off').mean() * 100,  # % of occupied time ventilation on
                'ventilation_on_empty_pct': df[df['occupants'] == 0]['ventilation_mode'].apply(
                    lambda x: x != 'off').mean() * 100,  # % of empty time ventilation on
                
                # Hourly patterns
                'hourly_co2_avg': df.groupby(df['timestamp'].dt.hour)['co2'].mean().to_dict(),
                'hourly_occupancy_avg': df.groupby(df['timestamp'].dt.hour)['occupants'].mean().to_dict(),
                'hourly_ventilation_pct': df.groupby(df['timestamp'].dt.hour)['ventilation_mode'].apply(
                    lambda x: (x != 'off').mean() * 100).to_dict(),
                
                # Daily patterns
                'daily_energy': df.groupby(df['timestamp'].dt.day)['energy_consumption'].max().diff().fillna(0).to_dict()
            }
            
            # Convert hourly data to strings for JSON serialization
            results['hourly_co2_avg'] = {str(k): v for k, v in results['hourly_co2_avg'].items()}
            results['hourly_occupancy_avg'] = {str(k): v for k, v in results['hourly_occupancy_avg'].items()}
            results['hourly_ventilation_pct'] = {str(k): v for k, v in results['hourly_ventilation_pct'].items()}
            results['daily_energy'] = {str(k): v for k, v in results['daily_energy'].items()}
            
            return results
        except Exception as e:
            logger.error(f"Error calculating results: {e}", exc_info=True)
            return {"error": str(e)}
    
    def _generate_plots(self, experiment):
        """
        Generate visualization plots for experiment results.
        
        Args:
            experiment: Experiment configuration
        """
        try:
            # Load data from CSV
            csv_path = experiment['csv_path']
            df = pd.read_csv(csv_path)
            
            # Ensure timestamps are parsed
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Create output directory for plots
            plots_dir = os.path.join(experiment['output_dir'], "plots")
            os.makedirs(plots_dir, exist_ok=True)
            
            # Set plot style
            plt.style.use('ggplot')
            
            # Plot 1: CO2 over time with ventilation and occupancy
            self._plot_co2_and_ventilation(df, os.path.join(plots_dir, "co2_ventilation.png"))
            
            # Plot 2: Temperature over time
            self._plot_temperature(df, os.path.join(plots_dir, "temperature.png"))
            
            # Plot 3: Hourly patterns
            self._plot_hourly_patterns(df, os.path.join(plots_dir, "hourly_patterns.png"))
            
            # Plot 4: Energy consumption
            self._plot_energy_consumption(df, os.path.join(plots_dir, "energy.png"))
            
            logger.info(f"Generated plots for experiment {experiment['name']}")
        except Exception as e:
            logger.error(f"Error generating plots: {e}", exc_info=True)
    
    def _plot_co2_and_ventilation(self, df, output_path):
        """Generate CO2, ventilation and occupancy plot."""
        fig, ax1 = plt.subplots(figsize=(14, 7))
        
        # Create helper axes sharing x-axis
        ax2 = ax1.twinx()
        ax3 = ax1.twinx()
        ax3.spines["right"].set_position(("axes", 1.1))  # Offset third y-axis
        
        # Plot CO2 levels
        ax1.plot(df['timestamp'], df['co2'], 'r-', alpha=0.7, linewidth=2, label='CO2 (ppm)')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('CO2 (ppm)', color='r')
        ax1.tick_params(axis='y', labelcolor='r')
        
        # Add CO2 threshold lines
        ax1.axhline(y=800, color='y', linestyle='--', alpha=0.5, label='Low CO2 Threshold (800 ppm)')
        ax1.axhline(y=1000, color='orange', linestyle='--', alpha=0.5, label='Medium CO2 Threshold (1000 ppm)')
        ax1.axhline(y=1200, color='r', linestyle='--', alpha=0.5, label='High CO2 Threshold (1200 ppm)')
        
        # Plot occupancy
        ax2.plot(df['timestamp'], df['occupants'], 'g-', alpha=0.7, linewidth=2, label='Occupants')
        ax2.set_ylabel('Occupants', color='g')
        ax2.tick_params(axis='y', labelcolor='g')
        ax2.set_ylim(0, max(df['occupants']) + 1)
        
        # Create ventilation color mapping
        ventilation_colors = {
            'off': 'gray',
            'low': 'blue',
            'medium': 'purple',
            'max': 'red'
        }
        
        # Create ventilation status indicator
        vent_status = []
        for mode, speed in zip(df['ventilation_mode'], df['ventilation_speed']):
            if mode == 'off':
                vent_status.append(0)
            elif speed == 'low':
                vent_status.append(1)
            elif speed == 'medium':
                vent_status.append(2)
            elif speed == 'max':
                vent_status.append(3)
            else:
                vent_status.append(0)
        
        # Plot ventilation status
        ax3.plot(df['timestamp'], vent_status, 'b-', alpha=0.5, linewidth=1.5, label='Ventilation Status')
        ax3.set_ylabel('Ventilation', color='b')
        ax3.tick_params(axis='y', labelcolor='b')
        ax3.set_yticks([0, 1, 2, 3])
        ax3.set_yticklabels(['Off', 'Low', 'Medium', 'Max'])
        
        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines3, labels3 = ax3.get_legend_handles_labels()
        ax1.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc='upper left')
        
        plt.title('CO2 Concentration, Occupancy, and Ventilation Status Over Time')
        fig.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close(fig)
    
    def _plot_temperature(self, df, output_path):
        """Generate temperature plot."""
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Create helper axes sharing x-axis
        ax2 = ax1.twinx()
        
        # Plot indoor temperature
        ax1.plot(df['timestamp'], df['temperature'], 'r-', alpha=0.7, linewidth=2, label='Indoor Temp (°C)')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Temperature (°C)', color='r')
        ax1.tick_params(axis='y', labelcolor='r')
        
        # Plot outdoor temperature
        ax1.plot(df['timestamp'], df['outdoor_temp'], 'b-', alpha=0.5, linewidth=1.5, label='Outdoor Temp (°C)')
        
        # Plot ventilation status on second y-axis
        vent_status = []
        for mode, speed in zip(df['ventilation_mode'], df['ventilation_speed']):
            if mode == 'off':
                vent_status.append(0)
            elif speed == 'low':
                vent_status.append(1)
            elif speed == 'medium':
                vent_status.append(2)
            elif speed == 'max':
                vent_status.append(3)
            else:
                vent_status.append(0)
        
        ax2.plot(df['timestamp'], vent_status, 'g-', alpha=0.5, linewidth=1, label='Ventilation Status')
        ax2.set_ylabel('Ventilation', color='g')
        ax2.tick_params(axis='y', labelcolor='g')
        ax2.set_yticks([0, 1, 2, 3])
        ax2.set_yticklabels(['Off', 'Low', 'Medium', 'Max'])
        
        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.title('Indoor and Outdoor Temperature with Ventilation Status')
        fig.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close(fig)
    
    def _plot_hourly_patterns(self, df, output_path):
        """Generate hourly patterns plot."""
        # Group data by hour
        hourly_data = df.groupby(df['timestamp'].dt.hour).agg({
            'co2': 'mean',
            'occupants': 'mean',
            'ventilation_mode': lambda x: (x != 'off').mean() * 100  # % time ventilation on
        }).reset_index()
        
        hourly_data.rename(columns={'timestamp': 'hour'}, inplace=True)
        
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Create helper axes sharing x-axis
        ax2 = ax1.twinx()
        ax3 = ax1.twinx()
        ax3.spines["right"].set_position(("axes", 1.1))  # Offset third y-axis
        
        # Plot hourly CO2
        ax1.plot(hourly_data['hour'], hourly_data['co2'], 'r-', marker='o', linewidth=2, label='Avg CO2 (ppm)')
        ax1.set_xlabel('Hour of Day')
        ax1.set_ylabel('CO2 (ppm)', color='r')
        ax1.tick_params(axis='y', labelcolor='r')
        
        # Plot hourly occupancy
        ax2.plot(hourly_data['hour'], hourly_data['occupants'], 'g-', marker='s', linewidth=2, label='Avg Occupants')
        ax2.set_ylabel('Occupants', color='g')
        ax2.tick_params(axis='y', labelcolor='g')
        
        # Plot ventilation percentage
        ax3.plot(hourly_data['hour'], hourly_data['ventilation_mode'], 'b-', marker='^', linewidth=2, 
                label='Ventilation On (%)')
        ax3.set_ylabel('Ventilation On (%)', color='b')
        ax3.tick_params(axis='y', labelcolor='b')
        
        # Set x-axis ticks to hours
        ax1.set_xticks(range(24))
        ax1.set_xlim(-0.5, 23.5)
        
        # Add hour labels in 4-hour increments
        for h in range(0, 24, 4):
            ax1.axvline(x=h, color='gray', linestyle='--', alpha=0.3)
        
        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines3, labels3 = ax3.get_legend_handles_labels()
        ax1.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc='upper left')
        
        plt.title('Hourly Patterns: CO2, Occupancy, and Ventilation')
        fig.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close(fig)
    
    def _plot_energy_consumption(self, df, output_path):
        """Generate energy consumption plot."""
        # Create a copy with datetime index
        df_energy = df.copy()
        df_energy.set_index('timestamp', inplace=True)
        
        # Resample to daily totals
        daily_energy = df_energy['energy_consumption'].resample('D').max().diff().fillna(0)
        
        # Create cumulative energy series
        cumulative_energy = df_energy['energy_consumption'] - df_energy['energy_consumption'].iloc[0]
        
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Create helper axes sharing x-axis
        ax2 = ax1.twinx()
        
        # Plot daily energy consumption
        ax1.bar(daily_energy.index, daily_energy.values, width=0.8, alpha=0.7, color='steelblue', 
               label='Daily Energy (kWh)')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Daily Energy Consumption (kWh)', color='steelblue')
        ax1.tick_params(axis='y', labelcolor='steelblue')
        
        # Plot cumulative energy consumption
        ax2.plot(cumulative_energy.index, cumulative_energy.values, 'r-', linewidth=2, 
                label='Cumulative Energy (kWh)')
        ax2.set_ylabel('Cumulative Energy (kWh)', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        
        # Add mean line to daily plot
        mean_daily = daily_energy.mean()
        ax1.axhline(y=mean_daily, color='navy', linestyle='--', alpha=0.7, 
                   label=f'Mean Daily: {mean_daily:.2f} kWh')
        
        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.title('Energy Consumption Over Time')
        fig.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close(fig)
    
    def compare_experiments(self, experiment_ids=None):
        """
        Generate comparison charts for multiple experiments.
        
        Args:
            experiment_ids: List of experiment IDs to compare (default: all with results)
        """
        # Select experiments to compare
        if experiment_ids is None:
            experiments = [exp for exp in self.experiments if exp.get('results') is not None]
        else:
            experiments = [self.experiments[idx-1] for idx in experiment_ids 
                          if 1 <= idx <= len(self.experiments) and 
                          self.experiments[idx-1].get('results') is not None]
        
        if not experiments:
            logger.error("No experiments with results to compare")
            return
        
        logger.info(f"Comparing {len(experiments)} experiments")
        
        # Create output directory
        comparison_dir = os.path.join(self.output_dir, "comparisons")
        os.makedirs(comparison_dir, exist_ok=True)
        
        # Generate comparison charts
        try:
            # Chart 1: Energy consumption comparison
            self._plot_energy_comparison(experiments, os.path.join(comparison_dir, "energy_comparison.png"))
            
            # Chart 2: CO2 levels comparison
            self._plot_co2_comparison(experiments, os.path.join(comparison_dir, "co2_comparison.png"))
            
            # Chart 3: Ventilation usage comparison
            self._plot_ventilation_comparison(experiments, os.path.join(comparison_dir, "ventilation_comparison.png"))
            
            # Save comparison data to JSON
            comparison_data = self._create_comparison_summary(experiments)
            try:
                json_safe_comparison = self._prepare_for_json(comparison_data)
                with open(os.path.join(comparison_dir, "comparison_summary.json"), 'w') as f:
                    json.dump(json_safe_comparison, f, indent=2)
            except TypeError as e:
                logger.error(f"Error serializing comparison data: {e}")
                # Create a simplified version with basic metrics
                simplified_comparison = {
                    'timestamp': datetime.now().isoformat(),
                    'experiments': [exp['name'] for exp in experiments],
                    'note': "Detailed data omitted due to serialization error"
                }
                with open(os.path.join(comparison_dir, "comparison_summary_simplified.json"), 'w') as f:
                    json.dump(simplified_comparison, f, indent=2)
            
            logger.info("Experiment comparison completed")
        except Exception as e:
            logger.error(f"Error comparing experiments: {e}", exc_info=True)
    
    def _plot_energy_comparison(self, experiments, output_path):
        """Generate energy consumption comparison chart."""
        # Extract energy consumption data
        experiment_names = [exp['name'] for exp in experiments]
        energy_values = [exp['results']['energy_consumption'] for exp in experiments]
        
        # Create the figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot bar chart
        bars = ax.bar(experiment_names, energy_values, alpha=0.7, color='steelblue')
        
        # Add value labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom')
        
        ax.set_ylabel('Total Energy Consumption (kWh)')
        ax.set_title('Energy Consumption Comparison')
        plt.xticks(rotation=45, ha='right')
        
        fig.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close(fig)
    
    def _plot_co2_comparison(self, experiments, output_path):
        """Generate CO2 levels comparison chart."""
        # Extract CO2 data
        experiment_names = [exp['name'] for exp in experiments]
        avg_co2 = [exp['results']['avg_co2'] for exp in experiments]
        max_co2 = [exp['results']['max_co2'] for exp in experiments]
        co2_over_1000 = [exp['results']['co2_over_1000_pct'] for exp in experiments]
        
        # Create the figure
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Create helper axis
        ax2 = ax1.twinx()
        
        # Set width and positions
        x = np.arange(len(experiment_names))
        width = 0.35
        
        # Plot average and max CO2
        ax1.bar(x - width/2, avg_co2, width, alpha=0.7, color='steelblue', label='Avg CO2 (ppm)')
        ax1.bar(x + width/2, max_co2, width, alpha=0.7, color='darkorange', label='Max CO2 (ppm)')
        
        # Add CO2 threshold lines
        ax1.axhline(y=800, color='green', linestyle='--', alpha=0.5, label='Low Threshold (800 ppm)')
        ax1.axhline(y=1000, color='orange', linestyle='--', alpha=0.5, label='Medium Threshold (1000 ppm)')
        ax1.axhline(y=1200, color='red', linestyle='--', alpha=0.5, label='High Threshold (1200 ppm)')
        
        # Plot % time over 1000 ppm on second axis
        ax2.plot(x, co2_over_1000, 'r-', marker='o', linewidth=2, label='Time CO2 > 1000 ppm (%)')
        
        # Set labels and legends
        ax1.set_xlabel('Strategy')
        ax1.set_ylabel('CO2 (ppm)', color='steelblue')
        ax2.set_ylabel('Time Above Threshold (%)', color='r')
        
        ax1.set_xticks(x)
        ax1.set_xticklabels(experiment_names, rotation=45, ha='right')
        
        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.title('CO2 Levels Comparison')
        fig.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close(fig)
    
    def _plot_ventilation_comparison(self, experiments, output_path):
        """Generate ventilation usage comparison chart."""
        # Extract ventilation data
        experiment_names = [exp['name'] for exp in experiments]
        vent_total = [exp['results']['ventilation_on_pct'] for exp in experiments]
        vent_occupied = [exp['results']['ventilation_on_occupied_pct'] for exp in experiments]
        vent_empty = [exp['results']['ventilation_on_empty_pct'] for exp in experiments]
        
        # Create the figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Set width and positions
        x = np.arange(len(experiment_names))
        width = 0.25
        
        # Plot ventilation percentages
        ax.bar(x - width, vent_total, width, alpha=0.7, color='steelblue', label='Total Time')
        ax.bar(x, vent_occupied, width, alpha=0.7, color='green', label='Occupied Time')
        ax.bar(x + width, vent_empty, width, alpha=0.7, color='darkorange', label='Empty Time')
        
        # Set labels and legend
        ax.set_xlabel('Strategy')
        ax.set_ylabel('Ventilation Active (%)')
        ax.set_xticks(x)
        ax.set_xticklabels(experiment_names, rotation=45, ha='right')
        ax.legend()
        
        plt.title('Ventilation Usage Comparison')
        fig.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close(fig)
    
    def _create_comparison_summary(self, experiments):
        """
        Create a detailed comparison summary.
        
        Args:
            experiments: List of experiment configurations
            
        Returns:
            dict: Comparison summary data
        """
        comparison = {
            'timestamp': datetime.now().isoformat(),
            'experiments': [exp['name'] for exp in experiments],
            'energy_comparison': {
                'values': [exp['results']['energy_consumption'] for exp in experiments],
                'unit': 'kWh',
                'best_strategy': experiments[np.argmin([exp['results']['energy_consumption'] 
                                                      for exp in experiments])]['name']
            },
            'co2_comparison': {
                'avg_values': [exp['results']['avg_co2'] for exp in experiments],
                'max_values': [exp['results']['max_co2'] for exp in experiments],
                'time_over_1000': [exp['results']['co2_over_1000_pct'] for exp in experiments],
                'time_over_1200': [exp['results']['co2_over_1200_pct'] for exp in experiments],
                'unit': 'ppm',
                'best_avg_strategy': experiments[np.argmin([exp['results']['avg_co2'] 
                                                          for exp in experiments])]['name'],
                'best_threshold_strategy': experiments[np.argmin([exp['results']['co2_over_1000_pct'] 
                                                                for exp in experiments])]['name']
            },
            'ventilation_comparison': {
                'total_time': [exp['results']['ventilation_on_pct'] for exp in experiments],
                'occupied_time': [exp['results']['ventilation_on_occupied_pct'] for exp in experiments],
                'empty_time': [exp['results']['ventilation_on_empty_pct'] for exp in experiments],
                'unit': '%',
                'most_efficient_strategy': experiments[np.argmin([exp['results']['ventilation_on_pct'] 
                                                                for exp in experiments])]['name']
            }
        }
        
        # Calculate efficiency scores (lower is better)
        energy_normalized = np.array([exp['results']['energy_consumption'] for exp in experiments])
        energy_normalized = energy_normalized / np.max(energy_normalized) if np.max(energy_normalized) > 0 else energy_normalized
        
        co2_normalized = np.array([exp['results']['co2_over_1000_pct'] for exp in experiments])
        co2_normalized = co2_normalized / np.max(co2_normalized) if np.max(co2_normalized) > 0 else co2_normalized
        
        # Combined score (50% energy, 50% CO2)
        combined_scores = 0.5 * energy_normalized + 0.5 * co2_normalized
        best_overall_idx = np.argmin(combined_scores)
        
        comparison['overall_scores'] = {
            'values': [float(score) for score in combined_scores],  # Convert to native Python floats
            'best_overall_strategy': experiments[best_overall_idx]['name'],
            'normalized_energy_scores': [float(score) for score in energy_normalized],  # Convert to native Python floats
            'normalized_co2_scores': [float(score) for score in co2_normalized]  # Convert to native Python floats
        }
        
        return comparison
    
    def stop(self):
        """Stop the simulation."""
        self.running = False
        logger.info("Simulation stopped")

# Utility function to help set up and run a complete simulation
def run_complete_simulation(duration_days=14, output_dir="simulation_results"):
    """
    Set up and run a comprehensive simulation comparing all strategies.
    
    Args:
        duration_days: Duration of each experiment in days
        output_dir: Directory for storing simulation results
        
    Returns:
        Simulation: The simulation instance
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(output_dir, "simulation.log")),
            logging.StreamHandler()
        ]
    )
    
    # Create simulation
    sim = Simulation(output_dir=output_dir)
    
    # Configure experiments for each strategy
    strategies = [
        (ControlStrategy.CONSTANT, "Constant Low Ventilation"),
        (ControlStrategy.THRESHOLD, "Threshold-Based Control"),
        (ControlStrategy.SCHEDULED, "Scheduled Ventilation"),
        (ControlStrategy.MARKOV, "Markov-Based Control")
    ]
    
    # Set up and run each experiment
    for strategy, name in strategies:
        experiment = sim.setup_experiment(
            name=name,
            strategy=strategy,
            duration_days=duration_days,
            description=f"Testing {name} strategy for {duration_days} days"
        )
        
        sim.run_experiment(experiment)
    
    # Generate comparison
    sim.compare_experiments()
    
    logger.info("Complete simulation finished")
    return sim

if __name__ == "__main__":
    # Run complete simulation when script is executed directly
    sim = run_complete_simulation()