"""Test script for MarkovController with dynamic exploration rate.

For interactive menu:
python test_markov_controller.py

For automated test:
python test_markov_controller.py --auto

To use a different model directory:
python test_markov_controller.py --model-dir /path/to/model/dir
"""

# tests/test_markov_controller.py

import time
import logging
import sys
import argparse
import tempfile
import os
import json
import random
from datetime import datetime, timedelta
from unittest.mock import MagicMock, Mock

# Add parent directory to the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("markov_test")

# Import components
try:
    from control.markov_controller import MarkovController, Action, CO2Level, TemperatureLevel, Occupancy, TimeOfDay
    from preferences.models import UserPreference, CompromisePreference
except ImportError as e:
    logger.error(f"Unable to import required modules: {e}. Please check your installation.")
    sys.exit(1)

# Mock classes for testing
class MockDataManager:
    """Mock data manager for testing."""
    def __init__(self):
        self.latest_data = {
            "scd41": {"co2": 800, "temperature": 22.0, "humidity": 50.0},
            "room": {"occupants": 2, "ventilated": False, "ventilation_speed": "off"},
            "timestamp": datetime.now().isoformat()
        }
    
    def set_conditions(self, co2=None, temperature=None, occupants=None, ventilated=None, ventilation_speed=None):
        """Set specific conditions for testing."""
        if co2 is not None:
            self.latest_data["scd41"]["co2"] = co2
        if temperature is not None:
            self.latest_data["scd41"]["temperature"] = temperature
        if occupants is not None:
            self.latest_data["room"]["occupants"] = occupants
        if ventilated is not None:
            self.latest_data["room"]["ventilated"] = ventilated
        if ventilation_speed is not None:
            self.latest_data["room"]["ventilation_speed"] = ventilation_speed

class MockPicoManager:
    """Mock PicoWH manager for testing."""
    def __init__(self):
        self.ventilation_status = False
        self.ventilation_speed = "off"
    
    def get_ventilation_status(self):
        return self.ventilation_status
    
    def get_ventilation_speed(self):
        return self.ventilation_speed if self.ventilation_status else "off"
    
    def control_ventilation(self, state, speed=None):
        if state == "off":
            self.ventilation_status = False
            self.ventilation_speed = "off"
        elif state == "on" and speed in ["low", "medium", "max"]:
            self.ventilation_status = True
            self.ventilation_speed = speed
        return True

class MockPreferenceManager:
    """Mock preference manager for testing."""
    def __init__(self):
        self.users = {}
        # Add some test users
        self.users[100] = UserPreference(100, "Test User 1", temp_min=19, temp_max=23, co2_threshold=800)
        self.users[200] = UserPreference(200, "Test User 2", temp_min=21, temp_max=25, co2_threshold=1200)
        self.effectiveness_override = None
    
    def get_all_user_preferences(self):
        return self.users
    
    def calculate_compromise_preference(self, user_ids):
        if self.effectiveness_override is not None:
            # Use override for testing
            return self.effectiveness_override
        
        if not user_ids:
            return CompromisePreference(0, 20.0, 24.0, 1000, 30.0, 60.0, 1.0)
        
        # Simple average for testing
        temp_mins = [self.users[uid].temp_min for uid in user_ids if uid in self.users]
        temp_maxs = [self.users[uid].temp_max for uid in user_ids if uid in self.users]
        co2_thresholds = [self.users[uid].co2_threshold for uid in user_ids if uid in self.users]
        
        if temp_mins and temp_maxs and co2_thresholds:
            avg_temp_min = sum(temp_mins) / len(temp_mins)
            avg_temp_max = sum(temp_maxs) / len(temp_maxs)
            avg_co2 = sum(co2_thresholds) / len(co2_thresholds)
            effectiveness = random.uniform(0.3, 0.9)  # Random for testing
            
            return CompromisePreference(len(user_ids), avg_temp_min, avg_temp_max, 
                                      int(avg_co2), 30.0, 60.0, effectiveness)
        
        return CompromisePreference(0, 20.0, 24.0, 1000, 30.0, 60.0, 1.0)

class MockOccupancyAnalyzer:
    """Mock occupancy analyzer for testing."""
    def get_expected_empty_duration(self, current_datetime):
        # Return different durations for testing
        hour = current_datetime.hour
        if 9 <= hour < 17:  # Work hours
            return timedelta(hours=4)  # Long absence
        elif 0 <= hour < 6:  # Night
            return timedelta(hours=6)  # Very long absence
        else:
            return timedelta(hours=1)  # Short absence
    
    def get_next_expected_return_time(self, current_datetime):
        # Return next return time based on current time
        hour = current_datetime.hour
        if 9 <= hour < 17:  # Work hours
            return current_datetime.replace(hour=17, minute=30)
        elif 0 <= hour < 6:  # Night
            return current_datetime.replace(hour=7, minute=0)
        else:
            return current_datetime + timedelta(hours=1)

def create_test_scenario(data_manager, scenario_name):
    """Create different test scenarios."""
    scenarios = {
        "high_co2_occupied": {
            "co2": 1500,
            "temperature": 22.0,
            "occupants": 2,
            "description": "High CO2 with people present"
        },
        "good_conditions": {
            "co2": 600,
            "temperature": 22.5,
            "occupants": 1,
            "description": "Good conditions, some experimentation allowed"
        },
        "cold_and_empty": {
            "co2": 700,
            "temperature": 18.0,
            "occupants": 0,
            "description": "Empty room, cold temperature"
        },
        "night_mode": {
            "co2": 800,
            "temperature": 21.0,
            "occupants": 2,
            "description": "Night time conditions"
        },
        "medium_co2_occupied": {
            "co2": 900,
            "temperature": 22.0,
            "occupants": 1,
            "description": "Medium CO2 with one person"
        }
    }
    
    if scenario_name in scenarios:
        scenario = scenarios[scenario_name]
        data_manager.set_conditions(
            co2=scenario["co2"],
            temperature=scenario["temperature"],
            occupants=scenario["occupants"]
        )
        return scenario["description"]
    else:
        return "Unknown scenario"

def test_markov_controller(model_dir=None, auto_test=False):
    """Test MarkovController with dynamic exploration rate."""
    # Create a temporary model directory if none provided
    if model_dir is None:
        temp_dir = tempfile.mkdtemp()
        model_dir = os.path.join(temp_dir, "test_markov_model")
        os.makedirs(model_dir, exist_ok=True)
        logger.info(f"Using temporary model directory: {model_dir}")
    
    # Initialize mock components
    data_manager = MockDataManager()
    pico_manager = MockPicoManager()
    preference_manager = MockPreferenceManager()
    occupancy_analyzer = MockOccupancyAnalyzer()
    
    # Initialize the controller
    try:
        controller = MarkovController(
            data_manager=data_manager,
            pico_manager=pico_manager,
            preference_manager=preference_manager,
            model_dir=model_dir,
            scan_interval=60,
            occupancy_analyzer=occupancy_analyzer
        )
        
        # Set up thresholds properly
        controller.co2_thresholds = {"low_max": 800, "medium_max": 1200}
        controller.temp_thresholds = {"low_max": 20, "medium_max": 24}
        
    except Exception as e:
        logger.error(f"Failed to initialize MarkovController: {e}")
        return False
    
    logger.info("Successfully initialized MarkovController")
    
    if auto_test:
        return run_automated_test(controller, data_manager, preference_manager)
    else:
        run_interactive_menu(controller, data_manager, preference_manager)
        return True

def run_automated_test(controller, data_manager, preference_manager):
    """Run an automated test suite for the MarkovController."""
    print("\n====== STARTING AUTOMATED TEST SEQUENCE ======")
    print("This will test all major functionality of MarkovController with dynamic exploration")
    print("Press Ctrl+C to abort at any time")
    
    test_results = []
    all_passed = True
    
    # Test 1: Basic initialization
    print("\nTest 1: Checking initialization...")
    try:
        status = controller.get_status()
        success = status["auto_mode"] and "base_exploration_rate" in status and "current_exploration_rate" in status
        test_results.append({
            "test": "Initialization",
            "success": success,
            "details": f"Auto mode: {status['auto_mode']}, Base rate: {status['base_exploration_rate']:.4f}"
        })
        if success:
            print(f"✅ Test 1 PASSED: Controller initialized properly")
        else:
            print("❌ Test 1 FAILED: Missing initialization parameters")
            all_passed = False
    except Exception as e:
        print(f"❌ Test 1 FAILED: {e}")
        all_passed = False
        test_results.append({"test": "Initialization", "success": False, "details": str(e)})
    
    # Test 2: Dynamic exploration rate calculation
    print("\nTest 2: Testing dynamic exploration rate...")
    try:
        # Test different scenarios
        scenarios = [
            ("high_co2_occupied", "should have low exploration rate"),
            ("good_conditions", "should have higher exploration rate"),
            ("cold_and_empty", "should have moderate exploration rate")
        ]
        
        exploration_rates = []
        for scenario_name, description in scenarios:
            create_test_scenario(data_manager, scenario_name)
            # Ensure thresholds are updated for the scenario
            occupants = 2 if scenario_name == "high_co2_occupied" else 1 if scenario_name == "good_conditions" else 0
            controller._update_thresholds_for_occupancy(occupants)
            controller._evaluate_state()
            rate = controller._calculate_dynamic_exploration_rate()
            exploration_rates.append((scenario_name, rate))
        
        # Check if rates are different for different scenarios
        rates_only = [r[1] for r in exploration_rates]
        success = len(set(f"{r:.4f}" for r in rates_only)) > 1  # Rates should be different
        
        details = ", ".join([f"{name}: {rate:.4f}" for name, rate in exploration_rates])
        test_results.append({
            "test": "Dynamic exploration rate",
            "success": success,
            "details": details
        })
        
        if success:
            print(f"✅ Test 2 PASSED: Exploration rates differ across scenarios")
            for name, rate in exploration_rates:
                print(f"   {name}: {rate:.4f}")
        else:
            print("❌ Test 2 FAILED: Exploration rates don't vary")
            all_passed = False
    except Exception as e:
        print(f"❌ Test 2 FAILED: {e}")
        all_passed = False
        test_results.append({"test": "Dynamic exploration rate", "success": False, "details": str(e)})
    
    # Test 3: Model confidence modifier
    print("\nTest 3: Testing model confidence modifier...")
    try:
        # Set up a scenario where confidence can be calculated
        data_manager.set_conditions(co2=1000, temperature=23, occupants=1)
        controller._evaluate_state()
        modifier = controller._calculate_model_confidence_modifier()
        
        success = 0.1 <= modifier <= 2.0  # Reasonable range
        test_results.append({
            "test": "Model confidence modifier",
            "success": success,
            "details": f"Confidence modifier: {modifier:.4f}"
        })
        
        if success:
            print(f"✅ Test 3 PASSED: Confidence modifier = {modifier:.4f}")
        else:
            print(f"❌ Test 3 FAILED: Invalid confidence modifier {modifier}")
            all_passed = False
    except Exception as e:
        print(f"❌ Test 3 FAILED: {e}")
        all_passed = False
        test_results.append({"test": "Model confidence modifier", "success": False, "details": str(e)})
    
    # Test 4: State value modifier
    print("\nTest 4: Testing state value modifier...")
    try:
        test_scenarios = [
            ({"co2": 500, "temperature": 22, "occupants": 2}, "good state"),
            ({"co2": 1600, "temperature": 22, "occupants": 2}, "bad state"),
            ({"co2": 800, "temperature": 22, "occupants": 0}, "empty home")
        ]
        
        modifiers = []
        for conditions, desc in test_scenarios:
            data_manager.set_conditions(**conditions)
            controller._evaluate_state()
            modifier = controller._calculate_current_state_value_modifier()
            modifiers.append((desc, modifier))
        
        # Check if modifiers vary appropriately
        success = all(0.1 <= m[1] <= 3.0 for m in modifiers)
        details = ", ".join([f"{desc}: {mod:.4f}" for desc, mod in modifiers])
        
        test_results.append({
            "test": "State value modifier",
            "success": success,
            "details": details
        })
        
        if success:
            print(f"✅ Test 4 PASSED: State value modifiers calculated")
            for desc, mod in modifiers:
                print(f"   {desc}: {mod:.4f}")
        else:
            print("❌ Test 4 FAILED: Invalid state value modifiers")
            all_passed = False
    except Exception as e:
        print(f"❌ Test 4 FAILED: {e}")
        all_passed = False
        test_results.append({"test": "State value modifier", "success": False, "details": str(e)})
    
    # Test 5: Preference effectiveness modifier
    print("\nTest 5: Testing preference effectiveness modifier...")
    try:
        # Test with different preference effectiveness values
        # Test high effectiveness (should reduce exploration)
        preference_manager.effectiveness_override = CompromisePreference(2, 20.0, 24.0, 1000, 30.0, 60.0, 0.9)
        modifier_high = controller._calculate_preference_effectiveness_modifier()
        
        # Test low effectiveness (should increase exploration)
        preference_manager.effectiveness_override = CompromisePreference(2, 20.0, 24.0, 1000, 30.0, 60.0, 0.4)
        modifier_low = controller._calculate_preference_effectiveness_modifier()
        
        # Reset
        preference_manager.effectiveness_override = None
        success = modifier_high < modifier_low  # High effectiveness should have lower modifier
        
        test_results.append({
            "test": "Preference effectiveness modifier",
            "success": success,
            "details": f"High effectiveness: {modifier_high:.4f}, Low effectiveness: {modifier_low:.4f}"
        })
        
        if success:
            print(f"✅ Test 5 PASSED: Preference modifiers work correctly")
            print(f"   High effectiveness: {modifier_high:.4f}")
            print(f"   Low effectiveness: {modifier_low:.4f}")
        else:
            print("❌ Test 5 FAILED: Preference modifiers don't work as expected")
            all_passed = False
    except Exception as e:
        print(f"❌ Test 5 FAILED: {e}")
        all_passed = False
        test_results.append({"test": "Preference effectiveness modifier", "success": False, "details": str(e)})
    
    # Test 6: Action selection with exploration
    print("\nTest 6: Testing action selection with exploration...")
    try:
        actions_chosen = {}
        
        # Use a scenario that should have moderate exploration rate
        create_test_scenario(data_manager, "medium_co2_occupied")
        
        # Run multiple decisions to see variety
        for i in range(100):  # Increased iterations for better statistics
            # Ensure state is properly evaluated
            state = controller._evaluate_state()
            
            if state:
                controller.current_state = state
                action = controller._decide_action()
                actions_chosen[action] = actions_chosen.get(action, 0) + 1
            else:
                # For debugging: let's see why state is None
                logger.debug(f"State evaluation failed at iteration {i}")
                continue
        
        # Check if we have any actions at all
        total_actions = sum(actions_chosen.values())
        
        if total_actions < 50:  # Less than half the iterations succeeded
            success = False
            details = f"Only {total_actions} actions out of 100 iterations - state evaluation issues"
        else:
            # Should have chosen multiple actions due to exploration
            success = len(actions_chosen) >= 2  # At least 2 different actions
            details = ", ".join([f"{action}: {count}" for action, count in actions_chosen.items()])
        
        test_results.append({
            "test": "Action selection with exploration",
            "success": success,
            "details": details
        })
        
        if success:
            print(f"✅ Test 6 PASSED: Multiple actions chosen")
            for action, count in actions_chosen.items():
                print(f"   {action}: {count} times")
            
            # Show exploration rate for this scenario
            exploration_rate = controller._calculate_dynamic_exploration_rate()
            print(f"   Exploration rate used: {exploration_rate:.4f}")
        else:
            print("❌ Test 6 FAILED: Not enough action variety")
            if total_actions == 0:
                print("   No actions were selected - state evaluation problem")
            else:
                print(f"   Only selected {len(actions_chosen)} different actions")
            all_passed = False
    except Exception as e:
        print(f"❌ Test 6 FAILED: {e}")
        all_passed = False
        test_results.append({"test": "Action selection", "success": False, "details": str(e)})
    
    # Test 7: Action selection with different exploration rates
    print("\nTest 7: Testing exploration with different rates...")
    try:
        # Test high exploration rate scenario
        data_manager.set_conditions(co2=750, temperature=22, occupants=0)  # Empty home with good conditions
        controller._update_thresholds_for_occupancy(0)
        state = controller._evaluate_state()
        
        if state:
            controller.current_state = state
            high_exploration_rate = controller._calculate_dynamic_exploration_rate()
            
            # Test low exploration rate scenario
            data_manager.set_conditions(co2=1600, temperature=22, occupants=2)  # High CO2 with people
            controller._update_thresholds_for_occupancy(2)
            state = controller._evaluate_state()
            
            if state:
                controller.current_state = state
                low_exploration_rate = controller._calculate_dynamic_exploration_rate()
                
                success = high_exploration_rate > low_exploration_rate
                details = f"Empty home rate: {high_exploration_rate:.4f}, High CO2 rate: {low_exploration_rate:.4f}"
                
                test_results.append({
                    "test": "Exploration rate scenarios",
                    "success": success,
                    "details": details
                })
                
                if success:
                    print(f"✅ Test 7 PASSED: Exploration rates vary correctly")
                    print(f"   Empty home rate: {high_exploration_rate:.4f}")
                    print(f"   High CO2 rate: {low_exploration_rate:.4f}")
                else:
                    print("❌ Test 7 FAILED: Exploration rates don't vary as expected")
                    all_passed = False
            else:
                print("❌ Test 7 FAILED: Could not evaluate second state")
                all_passed = False
        else:
            print("❌ Test 7 FAILED: Could not evaluate first state")
            all_passed = False
    except Exception as e:
        print(f"❌ Test 7 FAILED: {e}")
        all_passed = False
        test_results.append({"test": "Exploration rate scenarios", "success": False, "details": str(e)})
    
    # Display summary
    print("\n====== AUTOMATED TEST RESULTS ======")
    for result in test_results:
        status = "✅ PASSED" if result["success"] else "❌ FAILED"
        print(f"{result['test']}: {status} - {result['details']}")
    
    if all_passed:
        print("\n✅ ALL TESTS PASSED! MarkovController with dynamic exploration rate is working correctly.")
    else:
        print("\n❌ SOME TESTS FAILED. Check the log for details.")
    
    return all_passed

def run_interactive_menu(controller, data_manager, preference_manager):
    """Run an interactive menu for manual testing."""
    while True:
        # Display current status
        status = controller.get_status()
        print("\n====== MARKOV CONTROLLER TEST MENU ======")
        print(f"Current state: {status.get('current_state', 'None')}")
        print(f"Base exploration rate: {status['base_exploration_rate']:.4f}")
        print(f"Current exploration rate: {status['current_exploration_rate']:.4f}")
        print(f"Room conditions: CO2={data_manager.latest_data['scd41']['co2']} ppm, "
              f"Temp={data_manager.latest_data['scd41']['temperature']}°C, "
              f"Occupants={data_manager.latest_data['room']['occupants']}")
        
        print("\nOptions:")
        print("1. Show current status")
        print("2. Test different scenarios")
        print("3. Calculate exploration rate components")
        print("4. Make a decision")
        print("5. Modify room conditions")
        print("6. Run single step of control loop")
        print("7. Run automated test")
        print("8. Exit")
        
        # Get user input
        try:
            choice = input("\nEnter your choice (1-8): ")
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        
        # Process choice
        try:
            if choice == "1":
                print("\nDetailed Status:")
                for key, value in status.items():
                    if isinstance(value, dict):
                        print(f"{key}:")
                        for k, v in value.items():
                            print(f"  {k}: {v}")
                    else:
                        print(f"{key}: {value}")
            
            elif choice == "2":
                print("\nAvailable scenarios:")
                scenarios = ["high_co2_occupied", "good_conditions", "cold_and_empty", "night_mode", "medium_co2_occupied"]
                for i, scenario in enumerate(scenarios, 1):
                    print(f"{i}. {scenario}")
                
                try:
                    idx = int(input("Choose scenario (1-5): ")) - 1
                    if 0 <= idx < len(scenarios):
                        desc = create_test_scenario(data_manager, scenarios[idx])
                        print(f"Applied scenario: {desc}")
                        # Update state
                        controller._evaluate_state()
                        print("State updated")
                    else:
                        print("Invalid choice")
                except ValueError:
                    print("Please enter a valid number")
            
            elif choice == "3":
                print("\nExploration Rate Components:")
                base_rate = controller._base_exploration_rate
                model_conf = controller._calculate_model_confidence_modifier()
                state_val = controller._calculate_current_state_value_modifier()
                pref_eff = controller._calculate_preference_effectiveness_modifier()
                final_rate = controller._calculate_dynamic_exploration_rate()
                
                print(f"Base rate: {base_rate:.4f}")
                print(f"Model confidence modifier: {model_conf:.4f}")
                print(f"State value modifier: {state_val:.4f}")
                print(f"Preference effectiveness modifier: {pref_eff:.4f}")
                print(f"Final exploration rate: {final_rate:.4f}")
                print(f"Formula: {base_rate:.4f} × {model_conf:.4f} × {state_val:.4f} × {pref_eff:.4f} = {final_rate:.4f}")
            
            elif choice == "4":
                print("\nMaking decision...")
                controller._evaluate_state()
                action = controller._decide_action()
                print(f"Selected action: {action}")
                
                # Show what would happen
                success = controller._execute_action(action)
                print(f"Action execution: {'Success' if success else 'Failed'}")
            
            elif choice == "5":
                print("\nModify room conditions:")
                try:
                    co2 = input(f"CO2 ppm (current: {data_manager.latest_data['scd41']['co2']}): ")
                    temp = input(f"Temperature °C (current: {data_manager.latest_data['scd41']['temperature']}): ")
                    occupants = input(f"Occupants (current: {data_manager.latest_data['room']['occupants']}): ")
                    
                    conditions = {}
                    if co2: conditions['co2'] = float(co2)
                    if temp: conditions['temperature'] = float(temp)
                    if occupants: conditions['occupants'] = int(occupants)
                    
                    if conditions:
                        data_manager.set_conditions(**conditions)
                        controller._evaluate_state()
                        print("Conditions updated")
                    else:
                        print("No changes made")
                except ValueError:
                    print("Invalid input. Please enter numbers.")
            
            elif choice == "6":
                print("\nRunning one control loop iteration...")
                previous_state = controller.current_state
                controller._evaluate_state()
                print(f"Current state: {controller.current_state}")
                
                action = controller._decide_action()
                print(f"Decided action: {action}")
                
                success = controller._execute_action(action)
                print(f"Action executed: {'Success' if success else 'Failed'}")
                
                if previous_state and controller.current_state:
                    print("Updating model...")
                    controller._update_model(previous_state, action, controller.current_state)
                    print("Model updated")
            
            elif choice == "7":
                print("\nStarting automated test...")
                run_automated_test(controller, data_manager, preference_manager)
            
            elif choice == "8":
                print("Exiting...")
                break
            
            else:
                print("Invalid choice. Please enter a number between 1 and 8.")
                
        except Exception as e:
            logger.error(f"Error processing command: {e}")
            print(f"Error: {e}")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Test MarkovController with dynamic exploration rate")
    parser.add_argument("--model-dir", help="Path to model directory")
    parser.add_argument("--auto", action="store_true", help="Run automated test sequence")
    args = parser.parse_args()
    
    # Run test
    success = test_markov_controller(model_dir=args.model_dir, auto_test=args.auto)
    
    # Exit with appropriate status code
    sys.exit(0 if success else 1)