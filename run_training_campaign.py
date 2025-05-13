#!/usr/bin/env python3
"""
Run multiple independent training sessions for the Markov Controller.
"""
import os
import sys
import logging
import argparse
import shutil
from datetime import datetime

# Add parent directory to system path to import modules correctly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation import (
    Simulation, 
    ControlStrategy, 
    REAL_COMPONENTS_AVAILABLE
)

def setup_logging(output_dir):
    """Configure logging for the training campaign."""
    os.makedirs(output_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(output_dir, "training_campaign.log")),
            logging.StreamHandler()
        ]
    )

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run multiple Markov model training sessions")
    
    parser.add_argument(
        "--sessions", 
        type=int, 
        default=3,
        help="Number of training sessions to run"
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="training_campaign",
        help="Base directory for storing training session outputs"
    )
    
    parser.add_argument(
        "--duration", 
        type=float, 
        default=30.0,
        help="Duration of each training session in days"
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
    """Run multiple Markov model training sessions."""
    args = parse_arguments()
    
    # Create base directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = os.path.join(args.output_dir, timestamp)
    setup_logging(base_output_dir)
    
    # Create evaluation directory
    eval_dir = os.path.join(base_output_dir, "trained_models_for_evaluation")
    os.makedirs(eval_dir, exist_ok=True)
    
    if not REAL_COMPONENTS_AVAILABLE:
        logging.error("Real components not available. Cannot train Markov model.")
        return 1
    
    logging.info(f"Starting training campaign with {args.sessions} sessions")
    logging.info(f"Base output directory: {base_output_dir}")
    logging.info(f"Each session duration: {args.duration} days")
    
    # Store paths to all trained models for evaluation
    trained_model_paths = []
    
    # Run each training session
    for session_num in range(1, args.sessions + 1):
        logging.info(f"Starting training session {session_num}/{args.sessions}")
        
        # Create session-specific output directory
        session_dir = os.path.join(base_output_dir, f"session_{session_num}")
        os.makedirs(session_dir, exist_ok=True)
        
        # Create simulation for this session
        sim = Simulation(output_dir=session_dir, time_step_minutes=args.time_step)
        
        # Set Markov parameters
        sim.markov_explore_rate = args.exploration_rate
        sim.markov_learning_rate = args.learning_rate
        
        # Set up training experiment
        experiment = sim.setup_experiment(
            name=f"Markov Training Session {session_num}",
            strategy=ControlStrategy.MARKOV,
            duration_days=args.duration,
            description=f"Training Markov controller for {args.duration} days (Session {session_num})"
        )
        
        # Run training simulation
        logging.info(f"Running training simulation for session {session_num}...")
        result = sim.run_experiment(experiment)
        
        # Copy trained model to evaluation directory
        source_model_path = os.path.join(session_dir, "sim_data", "markov", "markov_model.json")
        if os.path.exists(source_model_path):
            # Create session-specific directory in evaluation folder
            eval_session_dir = os.path.join(eval_dir, f"run_{session_num}")
            os.makedirs(eval_session_dir, exist_ok=True)
            
            # Copy the model
            target_model_path = os.path.join(eval_session_dir, "markov_model.json")
            shutil.copy2(source_model_path, target_model_path)
            
            trained_model_paths.append(target_model_path)
            logging.info(f"Session {session_num} trained model saved to: {target_model_path}")
        else:
            logging.error(f"Training completed for session {session_num} but model file not found at {source_model_path}")
    
    # Log summary of all trained models
    logging.info("Training campaign completed")
    logging.info(f"Total models trained: {len(trained_model_paths)}")
    logging.info("Models ready for evaluation:")
    for path in trained_model_paths:
        logging.info(f"  - {path}")
    
    # Save list of model paths to a file for easy access by evaluation script
    models_list_path = os.path.join(base_output_dir, "trained_model_paths.txt")
    with open(models_list_path, 'w') as f:
        for path in trained_model_paths:
            f.write(f"{path}\n")
    
    logging.info(f"List of model paths saved to: {models_list_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main())