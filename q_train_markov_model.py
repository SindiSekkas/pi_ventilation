#!/usr/bin/env python3
# train_markov_model.py
"""
Train the MarkovController's Q-values using simulated historical data.
"""
import os
import sys
import csv
import json
import logging
import argparse
from datetime import datetime

# Add parent directory to the python path to find project modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from control.markov_controller import MarkovController, Action, CO2Level, TemperatureLevel, Occupancy, TimeOfDay

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("markov_trainer")

# --- Configuration ---
# Default hyperparameters - can be overridden via command line
SIMULATED_DATA_CSV_PATH = "simulated_ventilation_history.csv"  # Path to training data CSV
MARKOV_MODEL_DIR = "data/markov_trained"  # Directory to save the trained model
MARKOV_MODEL_FILENAME = "markov_model.json"  # Filename for the trained model
TRAINING_EPOCHS = 5  # Number of times to iterate over the entire dataset (1-20 recommended)
TRAINING_LEARNING_RATE = 0.2  # Initial learning rate for Q-value updates (0.01-0.5 recommended)
DISCOUNT_FACTOR = 0.9  # Future reward discount factor (0.9-0.99 recommended)
EXPLORATION_RATE = 0.3  # Initial exploration rate (0.1-1.0 recommended)
EPSILON_DECAY = 0.95  # Rate at which exploration decreases (0.9-0.99 recommended)
MIN_EPSILON = 0.01  # Minimum exploration rate (0.01-0.1 recommended)
ALPHA_DECAY = 0.99  # Learning rate decay factor (0.9-0.999 recommended)
MIN_ALPHA = 0.01  # Minimum learning rate (0.01-0.1 recommended)

# --- Mock Classes ---
class MockDataManager:
    """Simplified DataManager for training purposes."""
    def __init__(self):
        self.latest_data = {
            "scd41": {"co2": 400, "temperature": 20.0, "humidity": 50.0},
            "room": {"occupants": 0, "ventilated": False, "ventilation_speed": "off"},
            "bmp280": {"temperature": 20.0, "pressure": 1000.0},
            "timestamp": datetime.now().isoformat()
        }

    def update_sensor_data_from_row(self, csv_row: dict):
        """Updates latest_data based on a row from the simulated CSV."""
        try:
            self.latest_data["scd41"]["co2"] = float(csv_row.get('co2', self.latest_data["scd41"]["co2"]))
            self.latest_data["scd41"]["temperature"] = float(csv_row.get('temperature', self.latest_data["scd41"]["temperature"]))
            self.latest_data["scd41"]["humidity"] = float(csv_row.get('humidity', self.latest_data["scd41"]["humidity"]))
            self.latest_data["room"]["occupants"] = int(csv_row.get('occupants', self.latest_data["room"]["occupants"]))
            
            # Update ventilation status based on action
            action_str = csv_row.get('ventilation_action', 'off')
            if action_str == 'off':
                self.latest_data["room"]["ventilated"] = False
                self.latest_data["room"]["ventilation_speed"] = "off"
            else:
                self.latest_data["room"]["ventilated"] = True
                self.latest_data["room"]["ventilation_speed"] = action_str

            self.latest_data["timestamp"] = csv_row.get('timestamp', datetime.now().isoformat())
        except ValueError as e:
            logger.error(f"Error parsing data from CSV row: {csv_row}. Error: {e}")
        except Exception as e:
            logger.error(f"Unexpected error updating mock data: {e}")


class MockPicoManager:
    """Simplified PicoManager, not actively used for training transitions."""
    def get_ventilation_status(self):
        return False
    def get_ventilation_speed(self):
        return "off"
    def control_ventilation(self, state, speed=None):
        return True

class MockPreferenceManager:
    """Simplified PreferenceManager."""
    def get_all_user_preferences(self):
        return {}  # No specific users for this general training
    def calculate_compromise_preference(self, user_ids):
        # Return a default compromise that leads to standard thresholds in MarkovController
        from preferences.models import CompromisePreference  # Assuming this model exists
        return CompromisePreference(
            user_count=0, temp_min=20.0, temp_max=24.0, co2_threshold=1200,
            humidity_min=30.0, humidity_max=60.0, effectiveness_score=1.0
        )

class MockOccupancyAnalyzer:
    """Simplified OccupancyPatternAnalyzer."""
    def get_next_expected_return_time(self, current_datetime):
        return None
    def get_expected_empty_duration(self, current_datetime):
        return None

# --- Reward Calculation Function ---
def _calculate_reward(previous_state_key, action_taken_str, current_state_key, current_csv_row_dict):
    """
    Calculate reward for a state-action-next_state transition.
    
    Args:
        previous_state_key: Starting state key string
        action_taken_str: Action taken (off, low, medium, max)
        current_state_key: Resulting state key string
        current_csv_row_dict: CSV row data for the current state
    
    Returns:
        float: Calculated reward
    """
    # Reward depends on result (current_state) and costs (action)
    
    # CO2 comfort in the resulting state
    co2_level = current_state_key.split('_')[0]  # low, medium, high
    co2_reward = 0
    if co2_level == 'low': co2_reward = 1.0
    elif co2_level == 'medium': co2_reward = 0.2
    elif co2_level == 'high': co2_reward = -2.0  # Penalty for high CO2

    # Temperature comfort in the resulting state
    temp_level = current_state_key.split('_')[1]  # low, medium, high
    temp_reward = 0
    if temp_level == 'medium': temp_reward = 0.5
    elif temp_level == 'low' or temp_level == 'high': temp_reward = -0.5  # Penalty for uncomfortable temperature

    # Energy cost penalty
    energy_cost = 0
    if action_taken_str == 'low': energy_cost = -0.1
    elif action_taken_str == 'medium': energy_cost = -0.2
    elif action_taken_str == 'max': energy_cost = -0.4

    # Additional penalty if high CO2 AND room is occupied
    occupancy = current_state_key.split('_')[2]  # empty, occupied
    if co2_level == 'high' and occupancy == 'occupied':
        co2_reward -= 3.0  # Increase penalty

    total_reward = co2_reward + temp_reward + energy_cost
    return total_reward

# --- Parse Command-line Arguments ---
def parse_args():
    """Parse command-line arguments for training parameters."""
    parser = argparse.ArgumentParser(description='Train MarkovController Q-values from data.')
    
    parser.add_argument('--csv', 
                        default=SIMULATED_DATA_CSV_PATH,
                        help=f'Path to CSV training data (default: {SIMULATED_DATA_CSV_PATH})')
    
    parser.add_argument('--model-dir', 
                        default=MARKOV_MODEL_DIR,
                        help=f'Directory to save model (default: {MARKOV_MODEL_DIR})')
    
    parser.add_argument('--model-file', 
                        default=MARKOV_MODEL_FILENAME,
                        help=f'Filename for the model (default: {MARKOV_MODEL_FILENAME})')
    
    parser.add_argument('--epochs', 
                        type=int, 
                        default=TRAINING_EPOCHS,
                        help=f'Number of training epochs (default: {TRAINING_EPOCHS})')
    
    parser.add_argument('--learning-rate', 
                        type=float, 
                        default=TRAINING_LEARNING_RATE,
                        help=f'Initial learning rate (default: {TRAINING_LEARNING_RATE})')
    
    parser.add_argument('--discount-factor', 
                        type=float, 
                        default=DISCOUNT_FACTOR,
                        help=f'Discount factor (gamma) (default: {DISCOUNT_FACTOR})')
    
    parser.add_argument('--exploration-rate', 
                        type=float, 
                        default=EXPLORATION_RATE,
                        help=f'Initial exploration rate (default: {EXPLORATION_RATE})')
    
    parser.add_argument('--epsilon-decay', 
                        type=float, 
                        default=EPSILON_DECAY,
                        help=f'Epsilon decay rate (default: {EPSILON_DECAY})')
    
    parser.add_argument('--min-epsilon', 
                        type=float, 
                        default=MIN_EPSILON,
                        help=f'Minimum exploration rate (default: {MIN_EPSILON})')
    
    parser.add_argument('--alpha-decay', 
                        type=float, 
                        default=ALPHA_DECAY,
                        help=f'Learning rate decay (default: {ALPHA_DECAY})')
    
    parser.add_argument('--min-alpha', 
                        type=float, 
                        default=MIN_ALPHA,
                        help=f'Minimum learning rate (default: {MIN_ALPHA})')
    
    return parser.parse_args()

# --- Training Function ---
def train_model(args):
    """Trains the MarkovController's Q-values using simulated data."""
    logger.info("Starting Markov model Q-learning training...")

    # Ensure output directory exists
    os.makedirs(args.model_dir, exist_ok=True)
    output_model_path = os.path.join(args.model_dir, args.model_file)

    # Initialize mock components
    mock_data_manager = MockDataManager()
    mock_pico_manager = MockPicoManager()
    mock_preference_manager = MockPreferenceManager()
    mock_occupancy_analyzer = MockOccupancyAnalyzer()

    # Initialize MarkovController with Q-learning parameters
    markov_controller = MarkovController(
        data_manager=mock_data_manager,
        pico_manager=mock_pico_manager,
        preference_manager=mock_preference_manager,
        occupancy_analyzer=mock_occupancy_analyzer,
        model_dir=args.model_dir,
        scan_interval=60  # Not relevant for training script
    )
    
    # Set Q-learning hyperparameters
    markov_controller.learning_rate = args.learning_rate
    markov_controller.discount_factor = args.discount_factor
    markov_controller.exploration_rate = args.exploration_rate
    markov_controller.epsilon_decay = args.epsilon_decay
    markov_controller.min_epsilon = args.min_epsilon
    markov_controller.alpha_decay = args.alpha_decay
    markov_controller.min_alpha = args.min_alpha
    
    # Load existing Q-values if available to continue training
    markov_controller.load_q_values(output_model_path)
    
    logger.info(f"MarkovController initialized with:")
    logger.info(f"  - learning_rate: {markov_controller.learning_rate}")
    logger.info(f"  - discount_factor: {markov_controller.discount_factor}")
    logger.info(f"  - exploration_rate: {markov_controller.exploration_rate}")
    logger.info(f"  - epsilon_decay: {markov_controller.epsilon_decay}")
    logger.info(f"  - min_epsilon: {markov_controller.min_epsilon}")

    if not os.path.exists(args.csv):
        logger.error(f"Training data CSV file not found: {args.csv}")
        return

    logger.info(f"Reading training data from: {args.csv}")

    for epoch in range(args.epochs):
        logger.info(f"--- Starting Epoch {epoch + 1}/{args.epochs} ---")
        previous_csv_row = None
        rows_processed = 0

        with open(args.csv, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for current_csv_row in reader:
                if previous_csv_row is None:
                    previous_csv_row = current_csv_row
                    # Evaluate initial state to have a starting point for the first transition
                    mock_data_manager.update_sensor_data_from_row(previous_csv_row)
                    markov_controller.current_state = markov_controller._evaluate_state()
                    continue

                # 1. Get action taken that led from previous_csv_row to current_csv_row
                action_taken_str = previous_csv_row.get('ventilation_action')
                if not action_taken_str:
                    logger.warning(f"Missing 'ventilation_action' in row: {previous_csv_row}")
                    previous_csv_row = current_csv_row
                    mock_data_manager.update_sensor_data_from_row(previous_csv_row)
                    markov_controller.current_state = markov_controller._evaluate_state()
                    continue

                # 2. Determine previous_state_key
                temp_data_holder = mock_data_manager.latest_data.copy()
                mock_data_manager.update_sensor_data_from_row(previous_csv_row)
                previous_state_key = markov_controller._evaluate_state()
                mock_data_manager.latest_data = temp_data_holder

                # 3. Determine current_state_key (result of the action)
                mock_data_manager.update_sensor_data_from_row(current_csv_row)
                current_state_key = markov_controller._evaluate_state()

                # 4. Calculate reward for the transition
                try:
                    calculated_reward = _calculate_reward(
                        previous_state_key,
                        action_taken_str,
                        current_state_key,
                        current_csv_row
                    )
                except Exception as e:
                    logger.warning(f"Error calculating reward: {e}")
                    calculated_reward = 0.0

                # 5. Update Q-value if all components are valid
                if previous_state_key and current_state_key and action_taken_str:
                    logger.debug(f"Q-update: {previous_state_key} --({action_taken_str})--> {current_state_key} (Reward: {calculated_reward:.2f})")
                    markov_controller._update_q_value(previous_state_key, action_taken_str, calculated_reward, current_state_key)
                    rows_processed += 1
                else:
                    logger.warning(f"Missing components for Q-update. Prev_State: {previous_state_key}, Action: {action_taken_str}, Curr_State: {current_state_key}")

                previous_csv_row = current_csv_row
                markov_controller.current_state = current_state_key

        logger.info(f"Epoch {epoch + 1} completed. Processed {rows_processed} transitions.")
        
        # Update hyperparameters after each epoch for gradual decay
        old_epsilon = markov_controller.exploration_rate
        markov_controller.exploration_rate = max(markov_controller.min_epsilon, 
                                              markov_controller.exploration_rate * markov_controller.epsilon_decay)
        
        old_alpha = markov_controller.learning_rate
        markov_controller.learning_rate = max(markov_controller.min_alpha, 
                                           markov_controller.learning_rate * markov_controller.alpha_decay)
        
        logger.info(f"Updated exploration rate: {old_epsilon:.4f} -> {markov_controller.exploration_rate:.4f}")
        logger.info(f"Updated learning rate: {old_alpha:.4f} -> {markov_controller.learning_rate:.4f}")

    # Save the final trained Q-values
    markov_controller.save_q_values(output_model_path)
    logger.info(f"Trained Q-values saved to: {output_model_path}")
    logger.info("Q-learning training finished.")

# --- Main Execution ---
if __name__ == "__main__":
    args = parse_args()
    if not os.path.exists(args.csv):
        logger.error(f"Training data file '{args.csv}' not found.")
        logger.error("Please provide a valid CSV file with historical data.")
        sys.exit(1)
    else:
        train_model(args)