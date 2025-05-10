#!/usr/bin/env python3
# train_markov_model.py
"""
Train (pre-fill) the MarkovController's transition model using simulated data.
"""
import os
import sys
import csv
import json
import logging
import random
from datetime import datetime

# Add parent directory to the python path to find project modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from control.markov_controller import MarkovController, Action, CO2Level, TemperatureLevel, Occupancy, TimeOfDay # Assuming enums are here
# If enums are in a different models.py, adjust the import path accordingly.
# from control.models import Action, CO2Level, TemperatureLevel, Occupancy, TimeOfDay

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("markov_trainer")

# --- Configuration ---
SIMULATED_DATA_CSV_PATH = "simulated_ventilation_history.csv" # Path to your generated CSV
MARKOV_MODEL_DIR = "data/markov_trained" # Directory to save the trained model
MARKOV_MODEL_FILENAME = "markov_model.json" # Filename for the trained model
TRAINING_EPOCHS = 2  # Number of times to iterate over the entire dataset
TRAINING_LEARNING_RATE = 0.2 # Learning rate for model updates during training

# --- Mock Classes ---
class MockDataManager:
    """Simplified DataManager for training purposes."""
    def __init__(self):
        self.latest_data = {
            "scd41": {"co2": 400, "temperature": 20.0, "humidity": 50.0},
            "room": {"occupants": 0, "ventilated": False, "ventilation_speed": "off"}, # Added ventilation info
            "bmp280": {"temperature": 20.0, "pressure": 1000.0}, # Added bmp280 for completeness
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
        return {} # No specific users for this general training
    def calculate_compromise_preference(self, user_ids):
        # Return a default compromise that leads to standard thresholds in MarkovController
        from preferences.models import CompromisePreference # Assuming this model exists
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

# --- Training Function ---
def train_model():
    """Trains the MarkovController's transition model using simulated data."""
    logger.info("Starting Markov model training...")

    # Ensure output directory exists
    os.makedirs(MARKOV_MODEL_DIR, exist_ok=True)
    output_model_path = os.path.join(MARKOV_MODEL_DIR, MARKOV_MODEL_FILENAME)

    # Initialize mock components
    mock_data_manager = MockDataManager()
    mock_pico_manager = MockPicoManager()
    mock_preference_manager = MockPreferenceManager()
    mock_occupancy_analyzer = MockOccupancyAnalyzer()

    # Initialize MarkovController
    # Pass the output model directory for saving the trained model
    markov_controller = MarkovController(
        data_manager=mock_data_manager,
        pico_manager=mock_pico_manager,
        preference_manager=mock_preference_manager,
        occupancy_analyzer=mock_occupancy_analyzer,
        model_dir=MARKOV_MODEL_DIR, # Controller will save to its own model_dir/markov_model.json
        scan_interval=60 # Not relevant for training script
    )
    markov_controller.learning_rate = TRAINING_LEARNING_RATE
    # Load existing model if you want to continue training,
    # or let it initialize a new one. The controller's __init__ handles this.
    logger.info(f"Initial model loaded/initialized by MarkovController from/to {markov_controller.model_file}")


    if not os.path.exists(SIMULATED_DATA_CSV_PATH):
        logger.error(f"Simulated data CSV file not found: {SIMULATED_DATA_CSV_PATH}")
        return

    logger.info(f"Reading simulated data from: {SIMULATED_DATA_CSV_PATH}")

    for epoch in range(TRAINING_EPOCHS):
        logger.info(f"--- Starting Epoch {epoch + 1}/{TRAINING_EPOCHS} ---")
        previous_csv_row = None
        rows_processed = 0

        with open(SIMULATED_DATA_CSV_PATH, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for current_csv_row in reader:
                if previous_csv_row is None:
                    previous_csv_row = current_csv_row
                    # Evaluate initial state to have a starting point for the first transition
                    mock_data_manager.update_sensor_data_from_row(previous_csv_row)
                    markov_controller.current_state = markov_controller._evaluate_state()
                    continue

                # 1. Get action taken that led from previous_csv_row to current_csv_row
                # This action was recorded IN the previous_csv_row as 'ventilation_action'
                action_taken_str = previous_csv_row.get('ventilation_action')
                if not action_taken_str:
                    logger.warning(f"Missing 'ventilation_action' in row: {previous_csv_row}")
                    previous_csv_row = current_csv_row
                    continue
                try:
                    action_taken_enum = Action(action_taken_str)
                except ValueError:
                    logger.warning(f"Invalid action string '{action_taken_str}' in row: {previous_csv_row}")
                    previous_csv_row = current_csv_row
                    continue

                # 2. Determine previous_state_key
                mock_data_manager.update_sensor_data_from_row(previous_csv_row)
                # _evaluate_state updates thresholds, which is fine for getting the state category
                previous_state_key = markov_controller._evaluate_state()

                # 3. Determine current_state_key (result of the action)
                mock_data_manager.update_sensor_data_from_row(current_csv_row)
                current_state_key = markov_controller._evaluate_state()

                if previous_state_key and current_state_key:
                    logger.debug(f"Transition: {previous_state_key} --({action_taken_enum.value})--> {current_state_key}")
                    # Ensure the _update_model uses the .value of the enum
                    markov_controller._update_model(previous_state_key, action_taken_enum.value, current_state_key)
                    rows_processed += 1
                else:
                    logger.warning(f"Could not determine states for transition. Prev: {previous_state_key}, Curr: {current_state_key}")

                previous_csv_row = current_csv_row
        
        logger.info(f"Epoch {epoch + 1} completed. Processed {rows_processed} transitions.")

    # Save the final trained model
    # The MarkovController's _update_model might save periodically,
    # but we ensure a final save here.
    # We need a public method in MarkovController to save its transition_model
    # or save it directly here if that's simpler for now.
    
    # Option 1: Add a save method to MarkovController (Preferred)
    # if hasattr(markov_controller, 'save_transition_model'):
    #     markov_controller.save_transition_model(output_model_path)
    # else:
    # Option 2: Save directly (Less ideal encapsulation-wise)
    try:
        with open(output_model_path, 'w') as f:
            json.dump(markov_controller.transition_model, f, indent=2)
        logger.info(f"Trained Markov model saved to: {output_model_path}")
    except Exception as e:
        logger.error(f"Error saving trained model: {e}")

    logger.info("Markov model training finished.")

# --- Main Execution ---
if __name__ == "__main__":
    if not os.path.exists(SIMULATED_DATA_CSV_PATH):
        logger.error(f"Simulated data file '{SIMULATED_DATA_CSV_PATH}' not found.")
        logger.error("Please run 'generate_simulated_data.py' first or provide the correct path.")
    else:
        train_model()