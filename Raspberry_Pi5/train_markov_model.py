# train_markov_model.py
"""
Script to train the Markov controller for adaptive ventilation.
Runs an extended simulation to let the controller learn naturally.
"""
import os
import sys
import logging
import json
import argparse
from datetime import datetime, timedelta

# Add parent directory to system path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation import (
    Simulation, 
    ControlStrategy, 
    REAL_COMPONENTS_AVAILABLE
)

def setup_logging(output_dir):
    """Configure logging for the training."""
    os.makedirs(output_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(output_dir, "training.log")),
            logging.StreamHandler()
        ]
    )

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Markov model for ventilation control")
    
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="markov_training",
        help="Directory for storing training output"
    )
    
    parser.add_argument(
        "--duration", 
        type=float, 
        default=90.0,  # 90 days = ~3 months
        help="Training duration in days"
    )
    
    parser.add_argument(
        "--time-step", 
        type=int, 
        default=5,
        help="Simulation time step in minutes"
    )
    
    parser.add_argument(
        "--exploration-rate", 
        type=float, 
        default=0.5,
        help="Initial exploration rate (0.0-1.0)"
    )
    
    parser.add_argument(
        "--learning-rate", 
        type=float, 
        default=0.3,
        help="Learning rate (0.0-1.0)"
    )
    
    return parser.parse_args()

def main():
    """Run Markov model training."""
    args = parse_arguments()
    
    # Set up logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, timestamp)
    setup_logging(output_dir)
    
    # Create trained_models directory in advance
    trained_models_dir = os.path.join("trained_models")
    os.makedirs(trained_models_dir, exist_ok=True)
    
    if not REAL_COMPONENTS_AVAILABLE:
        logging.error("Real components not available. Cannot train Markov model.")
        return 1
    
    logging.info("Starting Markov model training")
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"Training duration: {args.duration} days")
    
    # Create simulation for training
    sim = Simulation(output_dir=output_dir, time_step_minutes=args.time_step)
    
    # Set up training experiment
    experiment = sim.setup_experiment(
        name="Markov Training",
        strategy=ControlStrategy.MARKOV,
        duration_days=args.duration,
        description=f"Training Markov controller for {args.duration} days"
    )
    
    # Set up Markov controller with custom parameters (happens in run_experiment)
    sim.markov_explore_rate = args.exploration_rate
    sim.markov_learning_rate = args.learning_rate
    
    # Run extended training simulation
    logging.info("Starting training simulation...")
    
    # Make sure model directory exists before running simulation
    model_dir = os.path.join(output_dir, "sim_data", "markov")
    os.makedirs(model_dir, exist_ok=True)
    
    # Create empty model file if it doesn't exist
    model_file = os.path.join(model_dir, "markov_model.json")
    if not os.path.exists(model_file):
        try:
            with open(model_file, 'w') as f:
                json.dump({}, f)
            logging.info(f"Created empty model file at {model_file}")
        except Exception as e:
            logging.error(f"Error creating empty model file: {e}")
    
    result = sim.run_experiment(experiment)
    
    # Get markov model path
    model_dir = os.path.join(output_dir, "sim_data", "markov")
    model_file = os.path.join(model_dir, "markov_model.json")
    
    if os.path.exists(model_file):
        # Save a copy of the trained model to a standard location
        standard_model_dir = os.path.join("trained_models")
        os.makedirs(standard_model_dir, exist_ok=True)
        
        # Save with timestamp
        model_copy_path = os.path.join(standard_model_dir, f"markov_model_{timestamp}.json")
        
        # Also save as latest
        latest_model_path = os.path.join(standard_model_dir, "markov_model_latest.json")
        
        # Copy the model files
        import shutil
        shutil.copy2(model_file, model_copy_path)
        shutil.copy2(model_file, latest_model_path)
        
        logging.info(f"Trained model saved to: {model_copy_path}")
        logging.info(f"Also saved as latest model: {latest_model_path}")
        
        # Report some statistics
        try:
            with open(model_file, 'r') as f:
                q_values = json.load(f)
            
            num_states = len(q_values)
            total_values = sum(len(actions) for state, actions in q_values.items())
            
            logging.info(f"Model statistics: {num_states} states, {total_values} state-action pairs")
        except Exception as e:
            logging.error(f"Error reading model statistics: {e}")
    else:
        logging.error(f"Training completed but model file not found at {model_file}")
        return 1
    
    logging.info("Training completed successfully")
    return 0

if __name__ == "__main__":
    sys.exit(main())