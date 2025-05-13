# simulation/simulation_runner.py
"""
Main simulation runner for ventilation control system.
Orchestrates training and evaluation of ventilation controllers.
"""
import os
import logging
import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple

# Import simulation components
from Simulation.simulation_environment import SimulationEnvironment
from Simulation.occupant_behavior_model import OccupantBehaviorModel
from Simulation.reward_function import RewardFunction
from Simulation.simulation_managers import SimulationDataManager, SimulationPicoManager
from Simulation.baseline_controllers import RuleBasedController, TimerBasedController, OracleRuleBasedController
from Simulation.metrics_collector import MetricsCollector

# Import real system components
from control.markov_controller import MarkovController
from predictive.occupancy_pattern_analyzer import OccupancyPatternAnalyzer
from predictive.adaptive_sleep_analyzer import AdaptiveSleepAnalyzer
from preferences.preference_manager import PreferenceManager
from preferences.models import CompromisePreference

logger = logging.getLogger(__name__)

class SimulationRunner:
    """
    Main orchestrator for training and evaluating ventilation controllers.
    
    Manages the simulation lifecycle, data flow between components,
    training of learning-based controllers, and performance evaluation.
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize the simulation runner.
        
        Args:
            config_file: Path to simulation configuration file
        """
        # Load configuration
        self.config = self._load_config(config_file)
        
        # Create simulation directory structure
        self._setup_directories()
        
        # Initialize simulation flags
        self.mode = "training"  # 'training' or 'evaluation'
        self.running = False
        self.current_step = 0
        self.is_initialized = False
        
        # Initialize components
        self._initialize_components()
        
        logger.info("SimulationRunner initialized")
    
    def _load_config(self, config_file: Optional[str]) -> Dict[str, Any]:
        """
        Load configuration from file or use defaults.
        
        Args:
            config_file: Path to configuration file
            
        Returns:
            dict: Configuration parameters
        """
        default_config = {
            # Simulation parameters
            "data_dir": "Simulation/data",
            "time_step_minutes": 5,  # Simulation time step in minutes
            "training_days": 60,
            "evaluation_days": 30,
            
            # Simulator components configuration
            "environment": {
                "ROOM_VOLUME_M3": 62.5,
                "EXTERNAL_CO2_PPM": 420,
                "NATURAL_INFILTRATION_RATE_PPM_PER_HOUR": 128.8,
                "CO2_GENERATION_PER_PERSON_PPM_PER_HOUR": 243.1,
                "CO2_GENERATION_PER_PERSON_SLEEPING_PPM_PER_HOUR": 170.0
            },
            
            "occupants": {
                "NUM_OCCUPANTS": 2,
                "WEEKDAY_SLEEP_START_MEAN_HOUR": 23.0,
                "WEEKDAY_WAKE_UP_MEAN_HOUR": 7.0
            },
            
            "reward": {
                "WEIGHT_CO2": 1.0,
                "WEIGHT_TEMP": 0.7,
                "WEIGHT_ENERGY": 1.0,
                "WEIGHT_SWITCHING": 0.3
            },
            
            # Controller configurations
            "markov_controller": {
                "scan_interval": 5,  # Minutes between control decisions
                "enable_exploration": True
            },
            
            "rule_based_controller": {
                "CO2_LOW_THRESHOLD": 800,
                "CO2_MEDIUM_THRESHOLD": 1000,
                "CO2_HIGH_THRESHOLD": 1200
            },
            
            "timer_based_controller": {
                "schedule": {
                    "0": [[7, 9, "low"], [17, 19, "medium"]],
                    "1": [[7, 9, "low"], [17, 19, "medium"]],
                    "2": [[7, 9, "low"], [17, 19, "medium"]],
                    "3": [[7, 9, "low"], [17, 19, "medium"]],
                    "4": [[7, 9, "low"], [17, 19, "medium"]],
                    "5": [[10, 12, "low"], [17, 19, "low"]],
                    "6": [[10, 12, "low"], [17, 19, "low"]]
                }
            },
            
            # User preferences
            "preferences": {
                "co2_threshold": 1000,
                "temp_min": 20.0,
                "temp_max": 24.0,
                "humidity_min": 30.0,
                "humidity_max": 60.0
            }
        }
        
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                
                # Merge with defaults (keeping custom values)
                for section, values in config.items():
                    if section in default_config and isinstance(values, dict):
                        default_config[section].update(values)
                    else:
                        default_config[section] = values
                
                logger.info(f"Loaded configuration from {config_file}")
            except Exception as e:
                logger.error(f"Failed to load configuration: {e}")
        
        return default_config
    
    def _setup_directories(self):
        """Create necessary directories for simulation data."""
        # Create data directories
        os.makedirs(self.config["data_dir"], exist_ok=True)
        os.makedirs(os.path.join(self.config["data_dir"], "models"), exist_ok=True)
        os.makedirs(os.path.join(self.config["data_dir"], "results"), exist_ok=True)
        os.makedirs(os.path.join(self.config["data_dir"], "plots"), exist_ok=True)
        
        # Create logs directory if needed
        log_dir = os.path.join(self.config["data_dir"], "logs")
        os.makedirs(log_dir, exist_ok=True)
    
    def _initialize_components(self):
        """Initialize all simulation components."""
        # Create simulation environment
        self.env = SimulationEnvironment(self.config["environment"])
        
        # Create occupant behavior model
        self.occupant_model = OccupantBehaviorModel(self.config["occupants"])
        
        # Create simulation interfaces
        self.sim_data_manager = SimulationDataManager()
        self.sim_pico_manager = SimulationPicoManager()
        self.sim_pico_manager.set_simulation_env(self.env)
        
        # Create reward function
        self.reward_function = RewardFunction(self.config["reward"])
        
        # Create metrics collector
        self.metrics = MetricsCollector(self.config["training_days"])
        
        # Create PreferenceManager
        self.preference_manager = self._create_preference_manager()
        
        # Initialize controller variables (to be set later)
        self.controller = None
        self.occupancy_analyzer = None
        self.sleep_analyzer = None
        
        # Create preferred path for Markov controller model save
        self.markov_model_path = os.path.join(self.config["data_dir"], "models", "markov_model.json")
        
        # Simulation state
        self.current_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        self.time_step_minutes = self.config["time_step_minutes"]
        
        # Initialized flag
        self.is_initialized = True
    
    def _create_preference_manager(self) -> PreferenceManager:
        """
        Create and configure preference manager.
        
        Returns:
            PreferenceManager: Configured preference manager
        """
        pref_manager = PreferenceManager(
            data_dir=os.path.join(self.config["data_dir"], "preferences")
        )
        
        # Add a simulated user with default preferences
        user_id = 1001  # Simulation user ID
        pref_manager.set_user_preference(
            user_id,
            temp_min=self.config["preferences"]["temp_min"],
            temp_max=self.config["preferences"]["temp_max"],
            co2_threshold=self.config["preferences"]["co2_threshold"],
            humidity_min=self.config["preferences"]["humidity_min"],
            humidity_max=self.config["preferences"]["humidity_max"]
        )
        
        return pref_manager
    
    def _initialize_markov_controller(self) -> "MarkovController":
        """
        Initialize the Markov controller.
        
        Returns:
            MarkovController: Initialized controller
        """
        # Create model directory if needed
        model_dir = os.path.dirname(self.markov_model_path)
        os.makedirs(model_dir, exist_ok=True)
        
        # Try to import real MarkovController
        try:
            from control.markov_controller import MarkovController
            logger.info("Using real MarkovController from control module")
            
            # Initialize Markov controller
            controller = MarkovController(
                data_manager=self.sim_data_manager,
                pico_manager=self.sim_pico_manager,
                preference_manager=self.preference_manager,
                model_dir=model_dir,
                scan_interval=60 * self.config["markov_controller"]["scan_interval"],
                occupancy_analyzer=self.occupancy_analyzer,
                enable_exploration=self.config["markov_controller"]["enable_exploration"]
            )
            
            # Ensure initial thresholds are set
            controller.co2_thresholds = {
                "low_max": 800,    # Upper bound for LOW
                "medium_max": 1200  # Upper bound for MEDIUM
            }
            
            controller.temp_thresholds = {
                "low_max": 20,     # Upper bound for LOW
                "medium_max": 24    # Upper bound for MEDIUM
            }
            
        except ImportError:
            # Fall back to mock implementation
            from Simulation.mock_controllers import MockMarkovController
            logger.info("Using MockMarkovController as fallback")
            
            controller = MockMarkovController(
                data_manager=self.sim_data_manager,
                pico_manager=self.sim_pico_manager,
                preference_manager=self.preference_manager,
                model_dir=model_dir,
                scan_interval=60 * self.config["markov_controller"]["scan_interval"],
                occupancy_analyzer=self.occupancy_analyzer,
                enable_exploration=self.config["markov_controller"]["enable_exploration"]
            )
        
        logger.info(f"Initialized Markov controller with thresholds: CO2={controller.co2_thresholds}, Temp={controller.temp_thresholds}")
        return controller
    
    def _initialize_occupancy_analyzer(self) -> OccupancyPatternAnalyzer:
        """
        Initialize the occupancy pattern analyzer.
        
        Returns:
            OccupancyPatternAnalyzer: Initialized analyzer
        """
        occupancy_history_file = os.path.join(
            self.config["data_dir"], "occupancy_history.csv"
        )
        
        # Create a minimal occupancy history file if it doesn't exist
        if not os.path.exists(occupancy_history_file):
            os.makedirs(os.path.dirname(occupancy_history_file), exist_ok=True)
            with open(occupancy_history_file, 'w') as f:
                f.write("timestamp,status,people_count\n")
        
        return OccupancyPatternAnalyzer(occupancy_history_file)
    
    def _initialize_sleep_analyzer(self) -> AdaptiveSleepAnalyzer:
        """
        Initialize the adaptive sleep analyzer.
        
        Returns:
            AdaptiveSleepAnalyzer: Initialized analyzer
        """
        return AdaptiveSleepAnalyzer(
            data_manager=self.sim_data_manager,
            controller=self.controller
        )
    
    def _get_compromise_preferences(self) -> Dict[str, Any]:
        """
        Get compromise preferences for multiple users.
        
        In simulation, we use a single user's preferences, but this
        interface allows for extending to multiple user scenarios.
        
        Returns:
            dict: Compromise preferences
        """
        user_ids = list(self.preference_manager.preferences.keys())
        if user_ids:
            compromise = self.preference_manager.calculate_compromise_preference(user_ids)
            return compromise.to_dict()
        else:
            # Default preferences if no users available
            return {
                "user_count": 1,
                "temp_min": self.config["preferences"]["temp_min"],
                "temp_max": self.config["preferences"]["temp_max"],
                "co2_threshold": self.config["preferences"]["co2_threshold"],
                "humidity_min": self.config["preferences"]["humidity_min"],
                "humidity_max": self.config["preferences"]["humidity_max"],
                "effectiveness_score": 1.0
            }
    
    def train_markov_controller(self) -> Tuple[MarkovController, Dict[str, Any]]:
        """
        Train the Markov controller through simulation.
        
        Returns:
            tuple: (Trained controller, training metrics)
        """
        logger.info("Starting Markov controller training")
        
        # Set training mode
        self.mode = "training"
        
        # Reset environment and models
        self.env.reset()
        self.occupant_model.reset(mode="training")
        self.metrics.reset()
        
        # Initialize occupancy analyzer
        self.occupancy_analyzer = self._initialize_occupancy_analyzer()
        
        # Initialize Markov controller
        self.controller = self._initialize_markov_controller()
        
        # Initialize sleep analyzer
        self.sleep_analyzer = self._initialize_sleep_analyzer()
        
        # Reset time and step counter
        self.current_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        self.current_step = 0
        
        # Set exploration on for training
        if isinstance(self.controller, MarkovController):
            self.controller.enable_exploration = True
            self.controller.exploration_rate = self.controller.MAX_EXPLORATION_RATE
            logger.info(f"Set exploration rate to {self.controller.exploration_rate}")
        
        # Run training simulation
        total_steps = int(self.config["training_days"] * 24 * 60 / self.time_step_minutes)
        
        logger.info(f"Starting training for {self.config['training_days']} days "
                  f"({total_steps} steps of {self.time_step_minutes} minutes)")
        
        # Simulation loop
        try:
            prev_action = "off"
            
            for step in range(total_steps):
                self.current_step = step
                
                # Get occupancy for this time step
                occupancy_data = self.occupant_model.get_next_timestep(mode="training")
                num_awake = occupancy_data["awake"]
                num_sleeping = occupancy_data["sleeping"]
                self.current_time = occupancy_data["timestamp"]
                
                # Update environment
                env_state, energy_consumed = self.env.step(
                    prev_action, num_awake, num_sleeping, self.time_step_minutes
                )
                
                # Update simulated sensor data
                self.sim_data_manager.update_sensor_data(env_state)
                self.sim_data_manager.update_room_data(
                    occupants=num_awake + num_sleeping,
                    ventilated=(prev_action != "off"),
                    ventilation_speed=prev_action
                )
                
                # Debug output
                if step % 100 == 0:
                    logger.debug(f"Step {step}: CO2={env_state['co2_ppm']:.1f}ppm, " 
                                f"Temp={env_state['temperature_c']:.1f}°C, " 
                                f"Occupants={num_awake+num_sleeping}")
                
                # Update predictive models
                if step % 10 == 0:  # Update every 10 steps to simulate real-world delay
                    self._update_predictive_models()
                
                # Use controller to decide action
                preferences = self._get_compromise_preferences()
                action = self.controller._decide_action()
                
                # Calculate reward
                reward, reward_components = self.reward_function.calculate_reward(
                    env_state, preferences, action, energy_consumed, prev_action
                )
                
                # Update controller's Q-values
                state_key = self.controller.current_state
                self.controller._update_q_value(state_key, action, reward, state_key)
                
                # Execute action
                self.sim_pico_manager.control_ventilation(
                    "on" if action != "off" else "off", 
                    action if action != "off" else None
                )
                
                # Record metrics
                occupancy_prediction = {
                    "predicted_occupancy": num_awake + num_sleeping
                }
                
                self.metrics.record_timestep(
                    self.current_time, env_state, preferences, action,
                    reward_components, reward, occupancy_prediction,
                    None, num_awake + num_sleeping
                )
                
                # Save action for next step
                prev_action = action
                
                # Progress logging
                if step % 1000 == 0:
                    days_completed = step * self.time_step_minutes / (24 * 60)
                    logger.info(f"Training progress: {days_completed:.1f}/{self.config['training_days']} days "
                              f"({100 * step / total_steps:.1f}%)")
                
                # Gradual exploration rate decay
                if isinstance(self.controller, MarkovController) and step % 500 == 0:
                    # Decay exploration rate
                    self.controller.exploration_rate = max(
                        self.controller.MIN_EXPLORATION_RATE,
                        self.controller.exploration_rate * self.controller.epsilon_decay
                    )
                    
                    if step % (total_steps // 10) == 0:
                        logger.info(f"Exploration rate decayed to {self.controller.exploration_rate:.3f}")
            
            logger.info("Training completed")
            
            # Save trained model
            if isinstance(self.controller, MarkovController):
                self.controller.save_q_values(self.markov_model_path)
                logger.info(f"Saved Markov controller model to {self.markov_model_path}")
            
            # Compute and return training metrics
            metrics = self.metrics.compute_metrics()
            return self.controller, metrics
            
        except Exception as e:
            logger.error(f"Error during training: {e}", exc_info=True)
            return self.controller, {}
    
    def evaluate_controller(self, controller, controller_name: str) -> Dict[str, Any]:
        """
        Evaluate a controller through simulation.
        
        Args:
            controller: Controller to evaluate
            controller_name: Name of the controller
            
        Returns:
            dict: Evaluation metrics
        """
        logger.info(f"Starting evaluation of {controller_name}")
        
        # Set evaluation mode
        self.mode = "evaluation"
        
        # Reset environment and models
        self.env.reset()
        self.occupant_model.reset(mode="testing")
        self.metrics.reset()
        
        # Set simulation duration for metrics
        self.metrics.simulation_duration_days = self.config["evaluation_days"]
        
        # Assign controller
        self.controller = controller
        
        # Turn off exploration for MarkovController during evaluation
        if isinstance(self.controller, MarkovController):
            self.controller.enable_exploration = False
            self.controller.exploration_rate = 0.0
            logger.info("Disabled exploration for evaluation")
        
        # Reset time and step counter
        self.current_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        self.current_step = 0
        
        # Run evaluation simulation
        total_steps = int(self.config["evaluation_days"] * 24 * 60 / self.time_step_minutes)
        
        logger.info(f"Starting evaluation for {self.config['evaluation_days']} days "
                  f"({total_steps} steps of {self.time_step_minutes} minutes)")
        
        # Simulation loop
        try:
            prev_action = "off"
            
            for step in range(total_steps):
                self.current_step = step
                
                # Get occupancy for this time step
                occupancy_data = self.occupant_model.get_next_timestep(mode="testing")
                num_awake = occupancy_data["awake"]
                num_sleeping = occupancy_data["sleeping"]
                self.current_time = occupancy_data["timestamp"]
                
                # Update environment
                env_state, energy_consumed = self.env.step(
                    prev_action, num_awake, num_sleeping, self.time_step_minutes
                )
                
                # Update simulated sensor data
                self.sim_data_manager.update_sensor_data(env_state)
                self.sim_data_manager.update_room_data(
                    occupants=num_awake + num_sleeping,
                    ventilated=(prev_action != "off"),
                    ventilation_speed=prev_action
                )
                
                # Different handling based on controller type
                preferences = self._get_compromise_preferences()
                
                # For rule-based controllers, prepare sensor data format
                if isinstance(controller, (RuleBasedController, TimerBasedController, OracleRuleBasedController)):
                    sensor_data = {
                        "co2_ppm": env_state["co2_ppm"],
                        "temperature_c": env_state["temperature_c"],
                        "humidity_percent": env_state["humidity_percent"],
                        "occupants": num_awake + num_sleeping
                    }
                    action = controller.decide_action(sensor_data, preferences)
                else:
                    # For MarkovController, just call decide action which will
                    # use the updated sim_data_manager
                    action = controller._decide_action()
                
                # Calculate reward (for metrics, not for learning)
                reward, reward_components = self.reward_function.calculate_reward(
                    env_state, preferences, action, energy_consumed, prev_action
                )
                
                # Execute action
                self.sim_pico_manager.control_ventilation(
                    "on" if action != "off" else "off", 
                    action if action != "off" else None
                )
                
                # Record metrics
                occupancy_prediction = {
                    "predicted_occupancy": num_awake + num_sleeping
                }
                
                self.metrics.record_timestep(
                    self.current_time, env_state, preferences, action,
                    reward_components, reward, occupancy_prediction,
                    None, num_awake + num_sleeping
                )
                
                # Save action for next step
                prev_action = action
                
                # Progress logging
                if step % 1000 == 0:
                    days_completed = step * self.time_step_minutes / (24 * 60)
                    logger.info(f"Evaluation progress: {days_completed:.1f}/{self.config['evaluation_days']} days "
                              f"({100 * step / total_steps:.1f}%)")
            
            logger.info(f"Evaluation of {controller_name} completed")
            
            # Compute and return evaluation metrics
            metrics = self.metrics.compute_metrics()
            
            # Save evaluation results
            results_file = os.path.join(
                self.config["data_dir"], 
                "results", 
                f"{controller_name}_evaluation.json"
            )
            
            with open(results_file, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            logger.info(f"Saved evaluation results to {results_file}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error during evaluation: {e}", exc_info=True)
            return {}
    
    def compare_controllers(self) -> Dict[str, Dict[str, Any]]:
        """
        Train and compare multiple controllers.
        
        Returns:
            dict: Comparison results with metrics for each controller
        """
        results = {}
        
        # First, train Markov controller
        logger.info("Starting controller comparison")
        logger.info("First: Training Markov controller")
        
        markov_controller, training_metrics = self.train_markov_controller()
        
        # Initialize baseline controllers
        rule_based = RuleBasedController(self.config["rule_based_controller"])
        timer_based = TimerBasedController(self.config["timer_based_controller"])
        oracle_rule_based = OracleRuleBasedController(self.config["rule_based_controller"])
        
        # Initialize predictive models
        self.occupancy_analyzer = self._initialize_occupancy_analyzer()
        
        # Evaluate all controllers
        controllers = [
            (markov_controller, "MarkovController"),
            (rule_based, "RuleBasedController"),
            (timer_based, "TimerBasedController"),
            (oracle_rule_based, "OracleRuleBasedController")
        ]
        
        for controller, name in controllers:
            logger.info(f"Evaluating {name}")
            metrics = self.evaluate_controller(controller, name)
            results[name] = metrics
        
        # Generate comparison report
        self._generate_comparison_report(results)
        
        return results
    
    def _update_predictive_models(self):
        """Update occupancy and sleep analyzers with recent data."""
        # Record occupancy data
        if self.occupancy_analyzer:
            total_occupants = self.sim_data_manager.latest_data["room"]["occupants"]
            status = "OCCUPIED" if total_occupants > 0 else "EMPTY"
            
            # Record to history file
            occupancy_history_file = self.occupancy_analyzer.history_file
            with open(occupancy_history_file, 'a') as f:
                f.write(f"{self.current_time.isoformat()},{status},{total_occupants}\n")
        
        # Occasionally update pattern analysis
        if self.current_step % 1000 == 0 and self.occupancy_analyzer:
            self.occupancy_analyzer.update_patterns(force=True)
            logger.debug("Updated occupancy patterns")
        
        # Update sleep analyzer
        if self.sleep_analyzer:
            self.sleep_analyzer.update_co2_data()
    
    def _generate_comparison_report(self, results: Dict[str, Dict[str, Any]]):
        """
        Generate comparison report and visualizations.
        
        Args:
            results: Metrics for each controller
        """
        # Create comparison summary
        summary = {
            "comfort": {},
            "energy": {},
            "actions": {}
        }
        
        # Extract key metrics for each controller
        for controller, metrics in results.items():
            if not metrics:
                continue
                
            # Comfort metrics
            if "comfort" in metrics:
                if "comfort" not in summary:
                    summary["comfort"] = {}
                
                summary["comfort"][controller] = {
                    "co2_comfort": metrics["comfort"]["co2"]["comfort_percent"],
                    "temp_comfort": metrics["comfort"]["temperature"]["comfort_percent"],
                    "humidity_comfort": metrics["comfort"]["humidity"]["comfort_percent"]
                }
            
            # Energy metrics
            if "energy" in metrics:
                if "energy" not in summary:
                    summary["energy"] = {}
                
                summary["energy"][controller] = {
                    "total_wh": metrics["energy"]["total_wh"],
                    "daily_avg_wh": metrics["energy"]["daily_avg_wh"]
                }
            
            # Action metrics
            if "actions" in metrics:
                if "actions" not in summary:
                    summary["actions"] = {}
                
                summary["actions"][controller] = {
                    "total_switches": metrics["actions"]["total_switches"],
                    "switches_per_day": metrics["actions"]["switches_per_day"],
                }
        
        # Save summary
        summary_file = os.path.join(
            self.config["data_dir"], 
            "results", 
            "controller_comparison.json"
        )
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Generate plots
        self._generate_comparison_plots(results)
        
        logger.info(f"Saved comparison report to {summary_file}")
    
    def _generate_comparison_plots(self, results: Dict[str, Dict[str, Any]]):
        """
        Generate comparative visualizations.
        
        Args:
            results: Metrics for each controller
        """
        # Plotting setup
        plt.figure(figsize=(12, 8))
        
        # CO2 comfort comparison
        plt.subplot(2, 2, 1)
        controllers = []
        co2_metrics = []
        
        for controller, metrics in results.items():
            if "comfort" in metrics and "co2" in metrics["comfort"]:
                controllers.append(controller)
                co2_metrics.append(metrics["comfort"]["co2"]["comfort_percent"])
        
        if controllers and co2_metrics:
            plt.bar(controllers, co2_metrics)
            plt.title('CO2 Comfort (%)')
            plt.ylim(0, 100)
            plt.xticks(rotation=45)
        
        # Temperature comfort comparison
        plt.subplot(2, 2, 2)
        controllers = []
        temp_metrics = []
        
        for controller, metrics in results.items():
            if "comfort" in metrics and "temperature" in metrics["comfort"]:
                controllers.append(controller)
                temp_metrics.append(metrics["comfort"]["temperature"]["comfort_percent"])
        
        if controllers and temp_metrics:
            plt.bar(controllers, temp_metrics)
            plt.title('Temperature Comfort (%)')
            plt.ylim(0, 100)
            plt.xticks(rotation=45)
        
        # Energy consumption comparison
        plt.subplot(2, 2, 3)
        controllers = []
        energy_metrics = []
        
        for controller, metrics in results.items():
            if "energy" in metrics:
                controllers.append(controller)
                energy_metrics.append(metrics["energy"]["daily_avg_wh"])
        
        if controllers and energy_metrics:
            plt.bar(controllers, energy_metrics)
            plt.title('Daily Energy Consumption (Wh)')
            plt.xticks(rotation=45)
        
        # Action switches comparison
        plt.subplot(2, 2, 4)
        controllers = []
        action_metrics = []
        
        for controller, metrics in results.items():
            if "actions" in metrics:
                controllers.append(controller)
                action_metrics.append(metrics["actions"]["switches_per_day"])
        
        if controllers and action_metrics:
            plt.bar(controllers, action_metrics)
            plt.title('Action Switches per Day')
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(
            self.config["data_dir"], 
            "plots", 
            "controller_comparison.png"
        )
        plt.savefig(plot_path)
        
        logger.info(f"Saved comparison plots to {plot_path}")