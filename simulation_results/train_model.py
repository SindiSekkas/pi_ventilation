#!/usr/bin/env python3
import os
import sys
import json
import logging
import csv
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from control.markov_controller import MarkovController, Action, CO2Level, TemperatureLevel, Occupancy, TimeOfDay

# Configuration
SIMULATED_DATA_CSV_PATH = "simulation_results/simulated_data.csv"
MARKOV_MODEL_DIR = "simulation_results/markov_model"
MARKOV_MODEL_FILENAME = "markov_model.json"
TRAINING_EPOCHS = 3
TRAINING_LEARNING_RATE = 0.2

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("markov_trainer")

class MockDataManager:
    def __init__(self):
        self.latest_data = {
            "scd41": {"co2": 400, "temperature": 20.0, "humidity": 50.0},
            "room": {"occupants": 0, "ventilated": False, "ventilation_speed": "off"},
            "bmp280": {"temperature": 20.0, "pressure": 1000.0},
            "timestamp": datetime.now().isoformat()
        }
        
    def update_sensor_data_from_row(self, csv_row):
        try:
            self.latest_data["scd41"]["co2"] = float(csv_row.get('co2', self.latest_data["scd41"]["co2"]))
            self.latest_data["scd41"]["temperature"] = float(csv_row.get('temperature', self.latest_data["scd41"]["temperature"]))
            self.latest_data["scd41"]["humidity"] = float(csv_row.get('humidity', self.latest_data["scd41"]["humidity"]))
            self.latest_data["room"]["occupants"] = int(csv_row.get('occupants', self.latest_data["room"]["occupants"]))
            
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
    def get_ventilation_status(self):
        return False
    def get_ventilation_speed(self):
        return "off"
    def control_ventilation(self, state, speed=None):
        return True

class MockPreferenceManager:
    def get_all_user_preferences(self):
        return {}
    def calculate_compromise_preference(self, user_ids):
        from preferences.models import CompromisePreference
        return CompromisePreference(
            user_count=0, temp_min=20.0, temp_max=24.0, co2_threshold=1200,
            humidity_min=30.0, humidity_max=60.0, effectiveness_score=1.0
        )

class MockOccupancyAnalyzer:
    def get_next_expected_return_time(self, current_datetime):
        return None
    def get_expected_empty_duration(self, current_datetime):
        return None

def _calculate_reward(state_key, action, co2_level, temp_level, occupants):
    components = state_key.split('_')
    if len(components) < 4:
        return 0.0
    
    next_co2_level, next_temp_level, occupancy, time_of_day = components
    
    co2_rewards = {
        "low": 2.0,
        "medium": 0.5,
        "high": -6.0
    }
    
    temp_rewards = {
        "low": -0.5,
        "medium": 1.0,
        "high": -0.5
    }
    
    energy_costs = {
        "off": 0,
        "low": -0.1,
        "medium": -0.3,
        "max": -0.5
    }
    
    co2_reward = co2_rewards.get(next_co2_level, 0)
    
    if next_co2_level == "high" and occupancy == "occupied":
        co2_reward = -12.0
    
    temp_reward = temp_rewards.get(next_temp_level, 0)
    energy_cost = energy_costs.get(action, 0)
    
    total_reward = co2_reward + temp_reward + energy_cost
    
    return total_reward

def train_model():
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
    markov_controller = MarkovController(
        data_manager=mock_data_manager,
        pico_manager=mock_pico_manager,
        preference_manager=mock_preference_manager,
        occupancy_analyzer=mock_occupancy_analyzer,
        model_dir=MARKOV_MODEL_DIR,
        scan_interval=60
    )
    markov_controller.learning_rate = TRAINING_LEARNING_RATE
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
                    mock_data_manager.update_sensor_data_from_row(previous_csv_row)
                    markov_controller.current_state = markov_controller._evaluate_state()
                    continue

                action_taken_str = previous_csv_row.get('ventilation_action')
                if not action_taken_str:
                    logger.warning(f"Missing 'ventilation_action' in row: {previous_csv_row}")
                    previous_csv_row = current_csv_row
                    mock_data_manager.update_sensor_data_from_row(previous_csv_row)
                    markov_controller.current_state = markov_controller._evaluate_state()
                    continue
                try:
                    action_taken_enum = Action(action_taken_str)
                except ValueError:
                    logger.warning(f"Invalid action string '{action_taken_str}' in row: {previous_csv_row}")
                    previous_csv_row = current_csv_row
                    mock_data_manager.update_sensor_data_from_row(previous_csv_row)
                    markov_controller.current_state = markov_controller._evaluate_state()
                    continue

                temp_data_holder = mock_data_manager.latest_data.copy()
                mock_data_manager.update_sensor_data_from_row(previous_csv_row)
                previous_state_key = markov_controller._evaluate_state()
                mock_data_manager.latest_data = temp_data_holder

                mock_data_manager.update_sensor_data_from_row(current_csv_row)
                current_state_key = markov_controller._evaluate_state()

                try:
                    current_co2_val = float(current_csv_row['co2'])
                    current_temp_val = float(current_csv_row['temperature'])
                    current_occupants_val = int(current_csv_row['occupants'])
                    
                    if current_co2_val < markov_controller.co2_thresholds["low_max"]:
                        co2_level = "low"
                    elif current_co2_val < markov_controller.co2_thresholds["medium_max"]:
                        co2_level = "medium"
                    else:
                        co2_level = "high"
                        
                    if current_temp_val < markov_controller.temp_thresholds["low_max"]:
                        temp_level = "low"
                    elif current_temp_val < markov_controller.temp_thresholds["medium_max"]:
                        temp_level = "medium"
                    else:
                        temp_level = "high"
                    
                    calculated_reward = _calculate_reward(
                        current_state_key,
                        action_taken_str,
                        co2_level,
                        temp_level,
                        current_occupants_val
                    )
                        
                except KeyError as e:
                    logger.warning(f"Missing data for reward calculation in row {current_csv_row}: {e}")
                    calculated_reward = 0.0
                except ValueError as e:
                    logger.warning(f"Invalid data type for reward calculation in row {current_csv_row}: {e}")
                    calculated_reward = 0.0

                if previous_state_key and current_state_key:
                    logger.debug(f"Transition: {previous_state_key} --({action_taken_enum.value})--> {current_state_key} (Reward: {calculated_reward:.2f})")
                    markov_controller._update_model(previous_state_key, action_taken_enum.value, current_state_key, reward=calculated_reward)
                    rows_processed += 1
                else:
                    logger.warning(f"Could not determine states for transition. Prev_Key: {previous_state_key}, Curr_Key: {current_state_key}, Action: {action_taken_enum.value}")

                previous_csv_row = current_csv_row
                markov_controller.current_state = current_state_key


        logger.info(f"Epoch {epoch + 1} completed. Processed {rows_processed} transitions.")

    try:
        with open(output_model_path, 'w') as f:
            json.dump(markov_controller.transition_model, f, indent=2)
        logger.info(f"Trained Markov model saved to: {output_model_path}")
    except Exception as e:
        logger.error(f"Error saving trained model: {e}")

    logger.info("Markov model training finished.")

if __name__ == "__main__":
    train_model()
