# simulation/run_simulation.py
"""
Main entry point for running ventilation system simulations.
Provides command-line interface for different simulation modes.
"""
import os
import sys
import argparse
import logging
import json
from datetime import datetime
import matplotlib.pyplot as plt

# Import simulation components
from Simulation.simulation_runner import SimulationRunner
from Simulation.baseline_controllers import RuleBasedController, TimerBasedController, OracleRuleBasedController

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Ventilation System Simulation')
    
    # Main operation mode
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate', 'compare'], default='compare',
                       help='Simulation mode (train, evaluate, compare)')
    
    # Configuration file
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration file')
    
    # Training specific options
    parser.add_argument('--training-days', type=int, default=60,
                       help='Number of days for training simulation')
    
    # Evaluation specific options
    parser.add_argument('--evaluation-days', type=int, default=30,
                       help='Number of days for evaluation simulation')
    parser.add_argument('--controller', type=str, 
                       choices=['markov', 'rule', 'timer', 'oracle'], default='markov',
                       help='Controller to evaluate')
    
    # Visualization options
    parser.add_argument('--plot', action='store_true',
                       help='Generate plots of simulation results')
    parser.add_argument('--no-display', action='store_true',
                       help='Do not display plots (save only)')
    
    # Model loading/saving
    parser.add_argument('--load-model', type=str, default=None,
                       help='Path to load pre-trained Markov model')
    parser.add_argument('--save-model', type=str, default=None,
                       help='Path to save trained Markov model')
    
    return parser.parse_args()

def create_default_config():
    """Create default configuration."""
    config = {
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
            "CO2_GENERATION_PER_PERSON_SLEEPING_PPM_PER_HOUR": 170.0,
            "MECHANICAL_VENTILATION_RATES_PPM_PER_HOUR": {
                "off": 0,
                "low": 403.2,
                "medium": 768.5,
                "max": 922.1
            },
            "VENTILATION_POWER_CONSUMPTION_W": {
                "off": 0,
                "low": 24,
                "medium": 29,
                "max": 53
            }
        },
        
        "occupants": {
            "NUM_OCCUPANTS": 2,
            "WEEKDAY_SLEEP_START_MEAN_HOUR": 23.0,
            "WEEKDAY_WAKE_UP_MEAN_HOUR": 7.0,
            "WEEKDAY_AWAY_START_MEAN_HOUR": 7.5,
            "WEEKDAY_AWAY_END_MEAN_HOUR": 17.5,
            "WEEKEND_SLEEP_START_MEAN_HOUR": 0.5,  # 00:30
            "WEEKEND_WAKE_UP_MEAN_HOUR": 9.0
        },
        
        "reward": {
            "REWARD_CO2_COMFORT": 1.0,
            "PENALTY_CO2_SLIGHT_DISCOMFORT": -0.5,
            "PENALTY_CO2_HIGH_DISCOMFORT": -2.0,
            "REWARD_TEMP_COMFORT": 0.8,
            "PENALTY_TEMP_DISCOMFORT": -1.0,
            "ENERGY_COST_MULTIPLIER": -0.01,
            "SWITCHING_ACTION_PENALTY": -0.2,
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
                "0": [[7, 9, "low"], [17, 19, "medium"]],  # Monday
                "1": [[7, 9, "low"], [17, 19, "medium"]],  # Tuesday
                "2": [[7, 9, "low"], [17, 19, "medium"]],  # Wednesday
                "3": [[7, 9, "low"], [17, 19, "medium"]],  # Thursday
                "4": [[7, 9, "low"], [17, 19, "medium"]],  # Friday
                "5": [[10, 12, "low"], [17, 19, "low"]],   # Saturday
                "6": [[10, 12, "low"], [17, 19, "low"]]    # Sunday
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
    
    return config

def save_config(config, filename="simulation_config.json"):
    """Save configuration to JSON file."""
    with open(filename, 'w') as f:
        json.dump(config, f, indent=2)
    logger.info(f"Saved configuration to {filename}")

def main():
    """Main execution function."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Create default config if none provided
    if args.config is None:
        config = create_default_config()
        save_config(config)
        config_file = "simulation_config.json"
    else:
        config_file = args.config
    
    # Update config with command line arguments
    if args.training_days:
        with open(config_file, 'r') as f:
            config = json.load(f)
        config["training_days"] = args.training_days
        save_config(config, config_file)
    
    if args.evaluation_days:
        with open(config_file, 'r') as f:
            config = json.load(f)
        config["evaluation_days"] = args.evaluation_days
        save_config(config, config_file)
    
    # Initialize simulation runner
    sim_runner = SimulationRunner(config_file)
    
    # Execute based on selected mode
    if args.mode == 'train':
        # Train Markov controller
        logger.info("Starting Markov controller training")
        controller, metrics = sim_runner.train_markov_controller()
        
        # Save model if requested
        if args.save_model:
            if hasattr(controller, 'save_q_values'):
                controller.save_q_values(args.save_model)
                logger.info(f"Saved trained model to {args.save_model}")
            else:
                logger.warning("Controller does not support saving")
        
        # Print summary
        logger.info("Training completed. Summary of results:")
        if metrics and 'comfort' in metrics:
            logger.info(f"CO2 Comfort: {metrics['comfort']['co2']['comfort_percent']:.1f}%")
            logger.info(f"Temperature Comfort: {metrics['comfort']['temperature']['comfort_percent']:.1f}%")
        
        if metrics and 'energy' in metrics:
            logger.info(f"Energy Consumption: {metrics['energy']['total_wh']:.1f} Wh")
        
        # Generate plots if requested
        if args.plot and metrics:
            plot_training_results(metrics, display=(not args.no_display))
    
    elif args.mode == 'evaluate':
        # Load or create controller based on selection
        if args.controller == 'markov':
            # Initialize Markov controller
            if args.load_model and os.path.exists(args.load_model):
                # Create a MarkovController and load the model
                from control.markov_controller import MarkovController
                controller = MarkovController(
                    data_manager=sim_runner.sim_data_manager,
                    pico_manager=sim_runner.sim_pico_manager,
                    preference_manager=sim_runner.preference_manager,
                    model_dir=os.path.dirname(args.load_model),
                    enable_exploration=False
                )
                controller.load_q_values(args.load_model)
                logger.info(f"Loaded Markov controller from {args.load_model}")
            else:
                # Train a new controller
                logger.info("No pre-trained model specified. Training a new Markov controller.")
                controller, _ = sim_runner.train_markov_controller()
        
        elif args.controller == 'rule':
            controller = RuleBasedController(sim_runner.config["rule_based_controller"])
            logger.info("Using rule-based controller for evaluation")
        
        elif args.controller == 'timer':
            controller = TimerBasedController(sim_runner.config["timer_based_controller"])
            logger.info("Using timer-based controller for evaluation")
        
        elif args.controller == 'oracle':
            controller = OracleRuleBasedController(sim_runner.config["rule_based_controller"])
            logger.info("Using oracle rule-based controller for evaluation")
        
        # Run evaluation
        logger.info(f"Starting evaluation of {args.controller} controller")
        metrics = sim_runner.evaluate_controller(controller, args.controller)
        
        # Print summary
        logger.info("Evaluation completed. Summary of results:")
        if metrics and 'comfort' in metrics:
            logger.info(f"CO2 Comfort: {metrics['comfort']['co2']['comfort_percent']:.1f}%")
            logger.info(f"Temperature Comfort: {metrics['comfort']['temperature']['comfort_percent']:.1f}%")
        
        if metrics and 'energy' in metrics:
            logger.info(f"Energy Consumption: {metrics['energy']['total_wh']:.1f} Wh")
        
        # Generate plots if requested
        if args.plot and metrics:
            plot_evaluation_results(metrics, args.controller, display=(not args.no_display))
    
    elif args.mode == 'compare':
        # Run comparison of all controllers
        logger.info("Starting comparison of all controllers")
        results = sim_runner.compare_controllers()
        
        # Print comparison summary
        logger.info("Comparison completed. Summary of results:")
        for controller, metrics in results.items():
            if not metrics or 'comfort' not in metrics:
                continue
            
            logger.info(f"\n{controller}:")
            logger.info(f"  CO2 Comfort: {metrics['comfort']['co2']['comfort_percent']:.1f}%")
            logger.info(f"  Temperature Comfort: {metrics['comfort']['temperature']['comfort_percent']:.1f}%")
            
            if 'energy' in metrics:
                logger.info(f"  Energy Consumption: {metrics['energy']['daily_avg_wh']:.1f} Wh/day")
            
            if 'actions' in metrics:
                logger.info(f"  Action Switches: {metrics['actions']['switches_per_day']:.1f} per day")
        
        # Generate plots if requested
        if args.plot and results:
            plot_comparison_results(results, display=(not args.no_display))

def plot_training_results(metrics, display=True):
    """Generate plots for training results."""
    plt.figure(figsize=(15, 10))
    
    # Plot CO2 levels
    if 'comfort' in metrics and 'co2' in metrics['comfort'] and 'stats' in metrics['comfort']['co2']:
        co2_stats = metrics['comfort']['co2']['stats']
        plt.subplot(2, 2, 1)
        plt.bar(['Min', 'Max', 'Mean', 'Median', '95th Percentile'], 
               [co2_stats['min'], co2_stats['max'], co2_stats['mean'], 
                co2_stats['median'], co2_stats['pct95']])
        plt.title('CO2 Level Statistics (ppm)')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot comfort percentages
    if 'comfort' in metrics:
        plt.subplot(2, 2, 2)
        comfort_data = [
            metrics['comfort']['co2']['comfort_percent'],
            metrics['comfort']['temperature']['comfort_percent'],
            metrics['comfort']['humidity']['comfort_percent']
        ]
        plt.bar(['CO2', 'Temperature', 'Humidity'], comfort_data)
        plt.title('Comfort Percentages')
        plt.ylim(0, 100)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot action distribution
    if 'actions' in metrics and 'percentages' in metrics['actions']:
        plt.subplot(2, 2, 3)
        actions = list(metrics['actions']['percentages'].keys())
        percentages = list(metrics['actions']['percentages'].values())
        plt.bar(actions, percentages)
        plt.title('Action Distribution')
        plt.ylim(0, 100)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot reward components if available
    if 'rewards' in metrics and 'components' in metrics['rewards']:
        plt.subplot(2, 2, 4)
        components = list(metrics['rewards']['components'].keys())
        values = [metrics['rewards']['components'][c]['total'] for c in components]
        plt.bar(components, values)
        plt.title('Reward Components')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('training_results.png')
    logger.info("Saved training results plot to training_results.png")
    
    if display:
        plt.show()

def plot_evaluation_results(metrics, controller_name, display=True):
    """Generate plots for evaluation results."""
    plt.figure(figsize=(15, 10))
    
    # Plot CO2 levels
    if 'comfort' in metrics and 'co2' in metrics['comfort'] and 'stats' in metrics['comfort']['co2']:
        co2_stats = metrics['comfort']['co2']['stats']
        plt.subplot(2, 2, 1)
        plt.bar(['Min', 'Max', 'Mean', 'Median', '95th Percentile'], 
               [co2_stats['min'], co2_stats['max'], co2_stats['mean'], 
                co2_stats['median'], co2_stats['pct95']])
        plt.title(f'{controller_name}: CO2 Level Statistics (ppm)')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot comfort percentages
    if 'comfort' in metrics:
        plt.subplot(2, 2, 2)
        comfort_data = [
            metrics['comfort']['co2']['comfort_percent'],
            metrics['comfort']['temperature']['comfort_percent'],
            metrics['comfort']['humidity']['comfort_percent']
        ]
        plt.bar(['CO2', 'Temperature', 'Humidity'], comfort_data)
        plt.title(f'{controller_name}: Comfort Percentages')
        plt.ylim(0, 100)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot action distribution
    if 'actions' in metrics and 'percentages' in metrics['actions']:
        plt.subplot(2, 2, 3)
        actions = list(metrics['actions']['percentages'].keys())
        percentages = list(metrics['actions']['percentages'].values())
        plt.bar(actions, percentages)
        plt.title(f'{controller_name}: Action Distribution')
        plt.ylim(0, 100)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot energy consumption
    if 'energy' in metrics:
        plt.subplot(2, 2, 4)
        plt.bar(['Total Energy (Wh)', 'Daily Average (Wh/day)'], 
               [metrics['energy']['total_wh'], metrics['energy']['daily_avg_wh']])
        plt.title(f'{controller_name}: Energy Consumption')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(f'{controller_name}_evaluation.png')
    logger.info(f"Saved evaluation results plot to {controller_name}_evaluation.png")
    
    if display:
        plt.show()

def plot_comparison_results(results, display=True):
    """Generate plots for controller comparison."""
    plt.figure(figsize=(15, 10))
    
    # Plot CO2 comfort comparison
    plt.subplot(2, 2, 1)
    controllers = []
    co2_comfort = []
    
    for controller, metrics in results.items():
        if 'comfort' in metrics and 'co2' in metrics['comfort']:
            controllers.append(controller)
            co2_comfort.append(metrics['comfort']['co2']['comfort_percent'])
    
    plt.bar(controllers, co2_comfort)
    plt.title('CO2 Comfort Comparison (%)')
    plt.ylim(0, 100)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    
    # Plot temperature comfort comparison
    plt.subplot(2, 2, 2)
    controllers = []
    temp_comfort = []
    
    for controller, metrics in results.items():
        if 'comfort' in metrics and 'temperature' in metrics['comfort']:
            controllers.append(controller)
            temp_comfort.append(metrics['comfort']['temperature']['comfort_percent'])
    
    plt.bar(controllers, temp_comfort)
    plt.title('Temperature Comfort Comparison (%)')
    plt.ylim(0, 100)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    
    # Plot energy consumption comparison
    plt.subplot(2, 2, 3)
    controllers = []
    energy = []
    
    for controller, metrics in results.items():
        if 'energy' in metrics:
            controllers.append(controller)
            energy.append(metrics['energy']['daily_avg_wh'])
    
    plt.bar(controllers, energy)
    plt.title('Daily Energy Consumption (Wh/day)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    
    # Plot action switches comparison
    plt.subplot(2, 2, 4)
    controllers = []
    switches = []
    
    for controller, metrics in results.items():
        if 'actions' in metrics:
            controllers.append(controller)
            switches.append(metrics['actions']['switches_per_day'])
    
    plt.bar(controllers, switches)
    plt.title('Action Switches per Day')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('controller_comparison.png')
    logger.info("Saved comparison results plot to controller_comparison.png")
    
    if display:
        plt.show()

if __name__ == "__main__":
    main()