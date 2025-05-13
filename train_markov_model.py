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
            logging.info(f"Checking model file at: {model_file}")
            if os.path.exists(model_file):
                file_size = os.path.getsize(model_file)
                logging.info(f"Model file exists, size: {file_size} bytes")
                
                with open(model_file, 'r') as f:
                    file_content = f.read()
                    logging.info(f"File content: {file_content[:200]}..." if len(file_content) > 200 else file_content)
                    
                    # Reset file pointer and load JSON
                    f.seek(0)
                    q_values = json.load(f)
                
                num_states = len(q_values)
                total_values = sum(len(actions) for state, actions in q_values.items())
                
                logging.info(f"Model statistics: {num_states} states, {total_values} state-action pairs")
                
                # Check the keys in the model
                if num_states > 0:
                    first_state = next(iter(q_values.keys()))
                    logging.info(f"Sample state key: {first_state}")
                    
                    # Check values for the first state to see if learning occurred
                    if first_state in q_values:
                        first_state_actions = q_values[first_state]
                        logging.info(f"Q-values for sample state: {first_state_actions}")
                        
                        # Check if all values are the same initial values or if learning occurred
                        values = list(first_state_actions.values())
                        if all(v in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] for v in values):
                            logging.warning("Q-values appear to be initial values only, learning may not have occurred")
                        else:
                            logging.info("Q-values show variability, learning has likely occurred")
            else:
                logging.error(f"Model file doesn't exist at: {model_file}")
                
            # Also check temp directory where model might be during simulation
            sim_model_dir = os.path.join(output_dir, "sim_data", "markov")
            sim_model_file = os.path.join(sim_model_dir, "markov_model.json")
            if os.path.exists(sim_model_file):
                logging.info(f"Found simulation model file at: {sim_model_file}")
                with open(sim_model_file, 'r') as f:
                    sim_q_values = json.load(f)
                sim_states = len(sim_q_values)
                logging.info(f"Simulation model has {sim_states} states")
        except Exception as e:
            logging.error(f"Error during model diagnostics: {e}", exc_info=True)
    else:
        logging.error(f"Training completed but model file not found at {model_file}")
        return 1
    
    logging.info("Training completed successfully")
    return 0

if __name__ == "__main__":
    sys.exit(main())