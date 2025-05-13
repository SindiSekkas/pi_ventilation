"""Test script for night mode functionality."""
import logging
import time
import os
import sys
from datetime import datetime

# parent directory to the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("night_mode_test")

def display_night_mode_status(controller):
    """Display current night mode status."""
    status = controller.get_status()
    night_mode = status.get("night_mode", {})
    
    print("\n===== NIGHT MODE STATUS =====")
    print(f"Night Mode Enabled: {night_mode.get('enabled', False)}")
    print(f"Start Hour: {night_mode.get('start_hour', 23)}:00")
    print(f"End Hour: {night_mode.get('end_hour', 7)}:00")
    print(f"Currently Active: {night_mode.get('currently_active', False)}")
    
    current_hour = datetime.now().hour
    print(f"Current Time: {current_hour}:00")
    print("============================")

def test_night_mode_activation():
    """Test night mode activation and deactivation."""
    try:
        # Import components
        from sensors.data_manager import DataManager
        from utils.pico_manager import PicoManager
        from control.markov_controller import MarkovController
        
        # Create test directory
        os.makedirs("test_data", exist_ok=True)
        os.makedirs("test_data/markov", exist_ok=True)
        
        # Initialize components
        data_manager = DataManager(csv_dir="test_data/csv")
        pico_manager = PicoManager(pico_ip="192.168.0.110")  # Adjust IP if needed
        
        # Check Pico connection
        if not pico_manager.find_pico_service():
            print("❌ Failed to connect to Pico W")
            return False
        
        print("✅ Connected to Pico W")
        
        # Create Markov controller with shorter interval for testing
        controller = MarkovController(
            data_manager=data_manager,
            pico_manager=pico_manager,
            model_dir="test_data/markov",
            scan_interval=5  # 5 seconds between checks for testing
        )
        
        # Display initial night mode status
        display_night_mode_status(controller)
        
        # Test 1: Enable night mode
        print("\nTest 1: Enabling night mode...")
        controller.set_night_mode(enabled=True, start_hour=22, end_hour=8)
        display_night_mode_status(controller)
        
        # Test 2: Test various hours
        print("\nTest 2: Testing different times...")
        test_hours = [6, 7, 8, 21, 22, 23, 0, 1]
        
        for hour in test_hours:
            # Manually set hour for testing (would normally be done by system time)
            original_hour = datetime.now().hour
            
            # Simulate different hours
            print(f"\nSimulating hour: {hour}:00")
            
            # Check if night mode would be active at this hour
            if controller.night_mode_start_hour > controller.night_mode_end_hour:
                # Night mode crosses midnight
                is_active = hour >= controller.night_mode_start_hour or hour < controller.night_mode_end_hour
            else:
                # Night mode doesn't cross midnight
                is_active = controller.night_mode_start_hour <= hour < controller.night_mode_end_hour
            
            print(f"Night mode would be {'ACTIVE' if is_active else 'INACTIVE'} at {hour}:00")
        
        # Test 3: Test with controller running
        print("\nTest 3: Starting controller with high CO2 during night hours...")
        
        # Set test data with high CO2
        data_manager.latest_data["scd41"]["co2"] = 1500
        data_manager.latest_data["scd41"]["temperature"] = 22.0
        data_manager.latest_data["scd41"]["humidity"] = 50.0
        data_manager.latest_data["room"]["occupants"] = 1
        
        # Start controller
        controller.start()
        
        # Let it run for a few cycles
        print("Monitoring for 30 seconds...")
        for i in range(6):
            time.sleep(5)
            status = pico_manager.get_ventilation_status()
            speed = pico_manager.get_ventilation_speed()
            night_active = controller._is_night_mode_active()
            
            print(f"Cycle {i+1}: Ventilation {'ON' if status else 'OFF'} ({speed}), "
                  f"Night mode {'ACTIVE' if night_active else 'INACTIVE'}")
        
        # Test 4: Disable night mode and see if controller reacts to high CO2
        print("\nTest 4: Disabling night mode...")
        controller.set_night_mode(enabled=False)
        
        # Monitor for another 30 seconds
        print("Monitoring for 30 seconds with night mode disabled...")
        for i in range(6):
            time.sleep(5)
            status = pico_manager.get_ventilation_status()
            speed = pico_manager.get_ventilation_speed()
            
            print(f"Cycle {i+1}: Ventilation {'ON' if status else 'OFF'} ({speed})")
        
        # Stop controller
        controller.stop()
        print("\n✅ Night mode test completed successfully!")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in night mode test: {e}", exc_info=True)
        print(f"\nError: {e}")
        return False

if __name__ == "__main__":
    print("\n===== NIGHT MODE FUNCTIONALITY TEST =====\n")
    successful = test_night_mode_activation()
    sys.exit(0 if successful else 1)

# tests/test_night_mode.py