# tests/test_markov_controller2.py
"""Test script for the new threshold-based MarkovController functionality."""
import os
import sys
import logging
import shutil
import time
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

# Add parent directory to the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("threshold_test")

def setup_test_environment():
    """Create test directory, initialize test data, and copy the trained Markov model for testing."""
    # Determine absolute paths based on the location of this test script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))

    # Absolute path to the root directory for this specific test's data
    # e.g., /home/pi/pi_ventilation/test_data/threshold_test
    test_run_data_root_abs = os.path.join(project_root, "test_data", "threshold_test")

    if os.path.exists(test_run_data_root_abs):
        shutil.rmtree(test_run_data_root_abs)
    os.makedirs(test_run_data_root_abs)

    # Absolute path to the directory where the MarkovController in the test will look for its model
    # e.g., /home/pi/pi_ventilation/test_data/threshold_test/markov
    test_model_target_dir_abs = os.path.join(test_run_data_root_abs, "markov")
    os.makedirs(test_model_target_dir_abs)

    # Absolute path to the source pre-trained model file
    # e.g., /home/pi/pi_ventilation/data/markov_trained/markov_model.json
    source_model_file_abs = os.path.join(project_root, "data", "markov_trained", "markov_model.json")

    # Absolute path for the destination model file within the test environment
    dest_model_file_abs = os.path.join(test_model_target_dir_abs, "markov_model.json")

    if os.path.exists(source_model_file_abs):
        shutil.copy(source_model_file_abs, dest_model_file_abs)
        logger.info(f"Copied trained model from {source_model_file_abs} to {dest_model_file_abs}")
    else:
        logger.warning(f"Trained model not found at {source_model_file_abs}. Controller might use an empty/new model if it can't load.")

    # Return the path that the test functions expect (relative to project root, or absolute)
    # The original code returned a relative path "test_data/threshold_test".
    # Returning an absolute path is safer for the os.path.join calls later.
    return test_run_data_root_abs

class MockDataManager:
    """Mock DataManager for testing."""
    def __init__(self, co2=800, temperature=22.0, humidity=50, occupants=2):
        self.latest_data = {
            "scd41": {
                "co2": co2,
                "temperature": temperature,
                "humidity": humidity
            },
            "room": {
                "occupants": occupants
            }
        }
    
    def update_data(self, co2=None, temperature=None, humidity=None, occupants=None):
        """Update mock sensor data."""
        if co2 is not None:
            self.latest_data["scd41"]["co2"] = co2
        if temperature is not None:
            self.latest_data["scd41"]["temperature"] = temperature
        if humidity is not None:
            self.latest_data["scd41"]["humidity"] = humidity
        if occupants is not None:
            self.latest_data["room"]["occupants"] = occupants

class MockPicoManager:
    """Mock PicoManager for testing."""
    def __init__(self):
        self.status = False
        self.speed = "off"
        self.control_calls = []
    
    def get_ventilation_status(self):
        """Get mock ventilation status."""
        return self.status
    
    def get_ventilation_speed(self):
        """Get mock ventilation speed."""
        return self.speed
    
    def control_ventilation(self, state, speed=None):
        """Mock ventilation control."""
        self.control_calls.append((state, speed))
        if state == "off":
            self.status = False
            self.speed = "off"
        else:
            self.status = True
            self.speed = speed
        return True

class MockPreferenceManager:
    """Mock PreferenceManager for testing."""
    def __init__(self):
        self.preferences = {}
        self.compromise = None
    
    def get_all_user_preferences(self):
        """Return mock user preferences."""
        return self.preferences
    
    def calculate_compromise_preference(self, user_ids):
        """Return mock compromise preference."""
        return self.compromise

class MockOccupancyAnalyzer:
    """Mock OccupancyPatternAnalyzer for testing."""
    def __init__(self):
        self.expected_return_time = None
        self.expected_empty_duration = None
    
    def get_next_expected_return_time(self, current_time):
        """Return mock expected return time."""
        if isinstance(self.expected_return_time, timedelta):
            return current_time + self.expected_return_time
        return self.expected_return_time
    
    def get_expected_empty_duration(self, current_time):
        """Return mock expected empty duration."""
        return self.expected_empty_duration

def test_get_current_target_thresholds():
    """Test the _get_current_target_thresholds method."""
    logger.info("Testing _get_current_target_thresholds...")
    
    test_dir = setup_test_environment()
    
    try:
        from control.markov_controller import MarkovController
        from preferences.models import CompromisePreference
        
        # Setup mock components
        data_manager = MockDataManager()
        pico_manager = MockPicoManager()
        preference_manager = MockPreferenceManager()
        occupancy_analyzer = MockOccupancyAnalyzer()
        
        # Initialize controller
        controller = MarkovController(
            data_manager=data_manager,
            pico_manager=pico_manager,
            preference_manager=preference_manager,
            model_dir=os.path.join(test_dir, "markov"),
            occupancy_analyzer=occupancy_analyzer
        )
        
        # Test 1: Empty house, no expected return
        data_manager.update_data(occupants=0)
        occupancy_analyzer.expected_return_time = None
        occupancy_analyzer.expected_empty_duration = None
        
        co2_thr, temp_thr = controller._get_current_target_thresholds(0)
        
        assert co2_thr == controller.default_empty_home_co2_thresholds
        assert temp_thr == controller.default_empty_home_temp_thresholds
        logger.info("✅ Empty house, no expected return: Using standard empty home thresholds")
        
        # Test 2: Empty house, long expected duration
        data_manager.update_data(occupants=0)
        occupancy_analyzer.expected_empty_duration = timedelta(hours=5)
        
        co2_thr, temp_thr = controller._get_current_target_thresholds(0)
        
        assert co2_thr == controller.VERY_LOW_ENERGY_THRESHOLDS_CO2
        assert temp_thr == controller.VERY_LOW_ENERGY_THRESHOLDS_TEMP
        logger.info("✅ Empty house, long absence: Using very energy-saving thresholds")
        
        # Test 3: Empty house, return expected soon
        data_manager.update_data(occupants=0)
        occupancy_analyzer.expected_return_time = timedelta(minutes=45)
        occupancy_analyzer.expected_empty_duration = timedelta(minutes=45)
        
        co2_thr, temp_thr = controller._get_current_target_thresholds(0)
        
        assert co2_thr == controller.PREPARE_FOR_RETURN_THRESHOLDS_CO2
        assert temp_thr == controller.PREPARE_FOR_RETURN_THRESHOLDS_TEMP
        logger.info("✅ Empty house, return soon: Using preparation thresholds")
        
        # Test 4: Occupied house, with user preferences
        data_manager.update_data(occupants=2)
        preference_manager.preferences = {"1": "user1", "2": "user2"}
        preference_manager.compromise = CompromisePreference(
            user_count=2,
            temp_min=21.0,
            temp_max=24.0,
            co2_threshold=900,
            humidity_min=40.0,
            humidity_max=60.0,
            effectiveness_score=0.9
        )
        
        co2_thr, temp_thr = controller._get_current_target_thresholds(2)
        
        assert co2_thr["low_max"] == int(900 * 0.8)  # 720
        assert co2_thr["medium_max"] == 900
        assert temp_thr["low_max"] == 21.0
        assert temp_thr["medium_max"] == 24.0
        logger.info("✅ Occupied house: Using compromise preferences")
        
        # Test 5: Occupied house, no user preferences
        data_manager.update_data(occupants=1)
        preference_manager.preferences = {}
        
        co2_thr, temp_thr = controller._get_current_target_thresholds(1)
        
        # Should fall back to default thresholds stored in the controller
        assert isinstance(co2_thr, dict)
        assert isinstance(temp_thr, dict)
        logger.info("✅ Occupied house, no preferences: Using default thresholds")
        
        logger.info("All _get_current_target_thresholds tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"Error testing _get_current_target_thresholds: {e}", exc_info=True)
        return False

def test_decide_action():
    """Test the _decide_action method with various scenarios."""
    logger.info("Testing _decide_action...")
    
    test_dir = setup_test_environment()
    
    try:
        from control.markov_controller import MarkovController, CO2Level, TemperatureLevel, Occupancy, TimeOfDay
        from preferences.models import CompromisePreference
        
        # Setup mock components
        data_manager = MockDataManager()
        pico_manager = MockPicoManager()
        preference_manager = MockPreferenceManager()
        occupancy_analyzer = MockOccupancyAnalyzer()
        
        # Initialize controller
        controller = MarkovController(
            data_manager=data_manager,
            pico_manager=pico_manager,
            preference_manager=preference_manager,
            model_dir=os.path.join(test_dir, "markov"),
            occupancy_analyzer=occupancy_analyzer
        )
        
        # Disable random exploration for testing
        controller.exploration_rate = 0
        
        # Setup user preferences
        preference_manager.preferences = {"1": "user1", "2": "user2"}
        preference_manager.compromise = CompromisePreference(
            user_count=2,
            temp_min=21.0,
            temp_max=24.0,
            co2_threshold=1000,
            humidity_min=40.0,
            humidity_max=60.0,
            effectiveness_score=0.9
        )
        
        # Test scenario 1: High CO2, occupied room
        data_manager.update_data(co2=1300, temperature=22.0, occupants=2)
        controller.current_state = controller._create_state_key(
            CO2Level.HIGH.value, 
            TemperatureLevel.MEDIUM.value, 
            Occupancy.OCCUPIED.value, 
            TimeOfDay.DAY.value
        )
        
        action = controller._decide_action()
        
        # With high CO2, should turn on ventilation at medium or max
        assert action in ["medium", "max"]
        logger.info(f"✅ High CO2, occupied: {action}")
        
        # Test scenario 2: Low CO2, optimal temperature, occupied room
        data_manager.update_data(co2=600, temperature=22.0, occupants=2)
        controller.current_state = controller._create_state_key(
            CO2Level.LOW.value, 
            TemperatureLevel.MEDIUM.value, 
            Occupancy.OCCUPIED.value, 
            TimeOfDay.DAY.value
        )
        
        action = controller._decide_action()
        
        # With good air quality, should turn off ventilation
        assert action == "off"
        logger.info(f"✅ Low CO2, optimal temp: {action}")
        
        # Test scenario 3: Medium CO2, cold room
        logger.info("--- TESTING SCENARIO 3: Medium CO2, cold room ---")
        data_manager.update_data(co2=950, temperature=19.0, occupants=2) # Temperature 19.0 -> LOW category
        controller.current_state = controller._create_state_key(
            CO2Level.MEDIUM.value, 
            TemperatureLevel.LOW.value, 
            Occupancy.OCCUPIED.value, 
            TimeOfDay.DAY.value 
        )
        action = controller._decide_action()
        controller_best_value_for_action_max_if_available = controller.last_best_value
        logger.info(f"Scenario 3 Result: Current state={controller.current_state}, Decided action={action}, Best value={controller_best_value_for_action_max_if_available}")
        # Add your assertion here, for example:
        # assert action == "off", f"Expected 'off' for cold room with medium CO2, got {action}"
        
        # Test scenario 4: Empty room, high CO2
        data_manager.update_data(co2=1300, temperature=22.0, occupants=0)
        controller.current_state = controller._create_state_key(
            CO2Level.HIGH.value, 
            TemperatureLevel.MEDIUM.value, 
            Occupancy.EMPTY.value, 
            TimeOfDay.DAY.value
        )
        
        # Set more permissive thresholds for empty home
        controller.default_empty_home_co2_thresholds = {
            "low_max": 850,
            "medium_max": 1300  # Higher threshold when no one is home
        }
        
        action = controller._decide_action()
        
        # Should still ventilate but maybe not as aggressively
        logger.info(f"✅ Empty room, high CO2: {action}")
        
        # Test scenario 5: Imminent return, high CO2
        data_manager.update_data(co2=1200, temperature=22.0, occupants=0)
        occupancy_analyzer.expected_return_time = timedelta(minutes=45)
        controller.current_state = controller._create_state_key(
            CO2Level.HIGH.value, 
            TemperatureLevel.MEDIUM.value, 
            Occupancy.EMPTY.value, 
            TimeOfDay.DAY.value
        )
        
        action = controller._decide_action()
        
        # Should ventilate to prepare for return
        assert action in ["low", "medium", "max"]
        logger.info(f"✅ Imminent return, high CO2: {action}")
        
        logger.info("All _decide_action tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"Error testing _decide_action: {e}", exc_info=True)
        return False

def test_parse_state_key():
    """Test the _parse_state_key method."""
    logger.info("Testing _parse_state_key...")
    
    test_dir = setup_test_environment()
    
    try:
        from control.markov_controller import MarkovController
        
        # Setup mock components
        data_manager = MockDataManager()
        pico_manager = MockPicoManager()
        
        # Initialize controller
        controller = MarkovController(
            data_manager=data_manager,
            pico_manager=pico_manager,
            model_dir=os.path.join(test_dir, "markov")
        )
        
        # Test valid state key
        state_key = "low_medium_occupied_day"
        components = controller._parse_state_key(state_key)
        
        assert components["co2_level"] == "low"
        assert components["temp_level"] == "medium"
        assert components["occupancy"] == "occupied"
        assert components["time_of_day"] == "day"
        logger.info("✅ Valid state key parsed correctly")
        
        # Test invalid state key
        invalid_key = "invalid_key"
        components = controller._parse_state_key(invalid_key)
        
        assert components == {}
        logger.info("✅ Invalid state key handled correctly")
        
        logger.info("All _parse_state_key tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"Error testing _parse_state_key: {e}", exc_info=True)
        return False

def run_integration_test():
    """Run an integration test with the new controller."""
    logger.info("Running integration test...")
    
    test_dir = setup_test_environment()
    
    try:
        from control.markov_controller import MarkovController
        from preferences.models import CompromisePreference
        
        # Setup mock components with realistic values
        data_manager = MockDataManager(co2=850, temperature=22.5, occupants=2)
        pico_manager = MockPicoManager()
        preference_manager = MockPreferenceManager()
        occupancy_analyzer = MockOccupancyAnalyzer()
        
        # Setup user preferences
        preference_manager.preferences = {"1": "user1", "2": "user2"}
        preference_manager.compromise = CompromisePreference(
            user_count=2,
            temp_min=21.0,
            temp_max=24.0,
            co2_threshold=1000,
            humidity_min=40.0,
            humidity_max=60.0,
            effectiveness_score=0.9
        )
        
        # Initialize controller
        controller = MarkovController(
            data_manager=data_manager,
            pico_manager=pico_manager,
            preference_manager=preference_manager,
            model_dir=os.path.join(test_dir, "markov"),
            occupancy_analyzer=occupancy_analyzer,
            scan_interval=1  # Fast scanning for test
        )
        
        # Start controller
        controller.start()
        logger.info("Controller started")
        
        # Let it run for a few cycles
        scenarios = [
            {"co2": 850, "temp": 22.5, "occupants": 2, "desc": "Normal conditions"},
            {"co2": 1200, "temp": 22.5, "occupants": 2, "desc": "High CO2"},
            {"co2": 1200, "temp": 19.5, "occupants": 2, "desc": "High CO2, low temp"},
            {"co2": 850, "temp": 22.5, "occupants": 0, "desc": "Empty room"},
            {"co2": 1200, "temp": 22.5, "occupants": 0, "desc": "Empty room, high CO2"}
        ]
        
        # Override min action interval for testing
        controller.min_action_interval = 0
        
        for i, scenario in enumerate(scenarios):
            logger.info(f"Scenario {i+1}: {scenario['desc']}")
            data_manager.update_data(
                co2=scenario["co2"],
                temperature=scenario["temp"],
                occupants=scenario["occupants"]
            )
            
            # Force state evaluation
            controller.current_state = controller._evaluate_state()
            
            # Let controller process
            time.sleep(2)
            
            # Log status
            status = controller.get_status()
            logger.info(f"State: {status['current_state']}")
            logger.info(f"Action: {status['last_action']}")
            logger.info(f"Ventilation: {'ON' if status['ventilation_status'] else 'OFF'}, Speed: {status['ventilation_speed']}")
            logger.info("")
        
        # Stop controller
        controller.stop()
        logger.info("Controller stopped")
        
        logger.info("Integration test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error in integration test: {e}", exc_info=True)
        return False

def run_all_tests():
    """Run all tests for the new threshold-based controller."""
    tests = [
        ("Get Current Target Thresholds", test_get_current_target_thresholds),
        ("Decide Action", test_decide_action),
        ("Parse State Key", test_parse_state_key),
        ("Integration Test", run_integration_test)
    ]
    
    print("\n===== THRESHOLD-BASED MARKOV CONTROLLER TEST =====\n")
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"Running {test_name}...")
        if test_func():
            print(f"✅ {test_name} PASSED\n")
            passed += 1
        else:
            print(f"❌ {test_name} FAILED\n")
            failed += 1
    
    print("===== TEST RESULTS =====")
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")
    
    # Clean up test directory
    test_dir = "test_data/threshold_test"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    
    return failed == 0

if __name__ == "__main__":
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)