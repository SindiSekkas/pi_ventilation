# run_simulation.py
"""
Main script for running ventilation simulation experiments.
"""
import os
import sys
import logging
import json
import argparse
from datetime import datetime

# Add the parent directory to system path to import modules correctly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation import (
    Simulation, 
    ControlStrategy, 
    run_complete_simulation,
    REAL_COMPONENTS_AVAILABLE
)

def setup_logging(output_dir):
    """Configure logging for the simulation."""
    os.makedirs(output_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(output_dir, "simulation.log")),
            logging.StreamHandler()
        ]
    )

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run ventilation simulation experiments")
    
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="simulation_results",
        help="Directory for storing simulation output"
    )
    
    parser.add_argument(
        "--duration", 
        type=float, 
        default=7.0,
        help="Duration of each experiment in days"
    )
    
    parser.add_argument(
        "--time-step", 
        type=int, 
        default=5,
        help="Simulation time step in minutes"
    )
    
    parser.add_argument(
        "--strategies", 
        type=str, 
        nargs="+",
        choices=["all", "threshold", "constant", "scheduled", "interval", "markov", "predictive"], # Added "interval"
        default=["all"],
        help="Control strategies to evaluate"
    )
    
    parser.add_argument(
        "--config", 
        type=str,
        help="Path to JSON configuration file"
    )
    
    parser.add_argument(
        "--use-pretrained", 
        action="store_true",
        help="Use pre-trained Markov model for evaluation"
    )
    
    parser.add_argument(
        "--training-mode", 
        action="store_true",
        help="Run in training mode with higher exploration rate"
    )
    
    parser.add_argument(
        "--compare-only",
        action="store_true",
        help="Only compare existing experiments without running new ones"
    )
    
    return parser.parse_args()

def load_config(config_path):
    """Load configuration from JSON file."""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Error loading configuration: {e}")
        return None

def main():
    """Main function to run simulation experiments."""
    args = parse_arguments()
    
    # Set up logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, timestamp)
    setup_logging(output_dir)
    
    logging.info("Starting ventilation simulation")
    logging.info(f"Output directory: {output_dir}")
    
    # Load configuration if provided
    config = None
    if args.config:
        config = load_config(args.config)
        if config:
            logging.info(f"Loaded configuration from {args.config}")
    else:
        # Use default configuration
        default_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                          "simulation", "config.json")
        if os.path.exists(default_config_path):
            config = load_config(default_config_path)
            if config:
                logging.info(f"Loaded default configuration from {default_config_path}")
        else:
            config = {}  # Empty config if no file exists    
    # If compare-only mode is active
    if args.compare_only:
        if not config or "experiment_dirs" not in config:
            logging.error("Cannot compare experiments: no experiment directories specified in config")
            return
        
        # Create simulation object
        sim = Simulation(output_dir=output_dir, time_step_minutes=args.time_step)
        
        # Load experiments from specified directories
        experiment_ids = []
        for exp_dir in config["experiment_dirs"]:
            try:
                with open(os.path.join(exp_dir, "config.json"), 'r') as f:
                    exp_config = json.load(f)
                
                with open(os.path.join(exp_dir, "results.json"), 'r') as f:
                    exp_results = json.load(f)
                
                # Add to experiments list
                exp_config["results"] = exp_results
                exp_config["output_dir"] = exp_dir
                sim.experiments.append(exp_config)
                experiment_ids.append(len(sim.experiments))
                
                logging.info(f"Loaded experiment from {exp_dir}")
            except Exception as e:
                logging.error(f"Error loading experiment from {exp_dir}: {e}")
        
        # Compare loaded experiments
        if experiment_ids:
            sim.compare_experiments(experiment_ids)
            logging.info("Comparison generated successfully")
        else:
            logging.error("No valid experiments found to compare")
        
        return
    
    # Determine strategies to evaluate
    strategies_to_run = []
    if "all" in args.strategies:
        strategies_to_run = [
            (ControlStrategy.CONSTANT, "Constant Low Ventilation"),
            (ControlStrategy.THRESHOLD, "Threshold-Based Control"),
            (ControlStrategy.SCHEDULED, "Scheduled Ventilation"),
            (ControlStrategy.INTERVAL, "Regular Interval Ventilation"), # Added INTERVAL
            (ControlStrategy.MARKOV, "Markov-Based Control")
        ]
        
        if REAL_COMPONENTS_AVAILABLE:
            strategies_to_run.append((ControlStrategy.PREDICTIVE, "Occupancy Prediction"))
    else:
        strategy_mapping = {
            "constant": (ControlStrategy.CONSTANT, "Constant Low Ventilation"),
            "threshold": (ControlStrategy.THRESHOLD, "Threshold-Based Control"),
            "scheduled": (ControlStrategy.SCHEDULED, "Scheduled Ventilation"),
            "interval": (ControlStrategy.INTERVAL, "Regular Interval Ventilation"), # Added INTERVAL
            "markov": (ControlStrategy.MARKOV, "Markov-Based Control"),
            "predictive": (ControlStrategy.PREDICTIVE, "Occupancy Prediction")
        }
        
        for strategy_name in args.strategies:
            if strategy_name in strategy_mapping:
                strategies_to_run.append(strategy_mapping[strategy_name])
    
    # Check if predictive strategy is requested but not available
    if not REAL_COMPONENTS_AVAILABLE and any(s[0] == ControlStrategy.PREDICTIVE for s in strategies_to_run):
        logging.warning("Predictive strategy requested but real components not available. Skipping.")
        strategies_to_run = [s for s in strategies_to_run if s[0] != ControlStrategy.PREDICTIVE]
    
    # Create and run simulation
    sim = Simulation(
        output_dir=output_dir, 
        time_step_minutes=args.time_step,
        use_pretrained_markov=args.use_pretrained
    )
    
    # Set Markov parameters based on mode
    if args.training_mode:
        sim.markov_explore_rate = 0.5  # Higher for training
        sim.markov_learning_rate = 0.3  # Higher for training
    else:
        sim.markov_explore_rate = 0.1  # Lower for evaluation
        sim.markov_learning_rate = 0.1  # Lower for evaluation
    
    # Run selected strategies
    for strategy, name in strategies_to_run:
        logging.info(f"Running experiment: {name}")
        
        # Get strategy-specific configuration
        strategy_config = config.get(strategy.name.lower(), {}) if config else {}
        description = f"Testing {name} strategy for {args.duration} days"
        if strategy == ControlStrategy.INTERVAL:
            description = "10 minutes of ventilation every 60 minutes" # Updated description
            print(f"Setting up INTERVAL experiment, ControlStrategy.INTERVAL value is: {ControlStrategy.INTERVAL}") # DEBUG

        # Set up experiment
        experiment = sim.setup_experiment(
            name=name,
            strategy=strategy,
            duration_days=args.duration,
            description=description # Use potentially updated description
        )
        
        if strategy == ControlStrategy.INTERVAL:
            print(f"INTERVAL experiment set up: {experiment}") # DEBUG

        # Apply configuration overrides
        if strategy_config:
            strategy_key = f"{strategy.name.lower()}_strategy"
            if strategy_key in sim.ventilation.parameters:
                sim.ventilation.parameters[strategy_key].update(strategy_config)
                logging.info(f"Applied custom configuration for {strategy.name}")
        
        # Run experiment
        sim.run_experiment(experiment)
    
    # Compare results if multiple experiments were run
    if len(strategies_to_run) > 1:
        sim.compare_experiments()
    
    logging.info("Simulation completed successfully")

if __name__ == "__main__":
    main()