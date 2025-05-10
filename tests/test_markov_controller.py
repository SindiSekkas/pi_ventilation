# tests/test_markov_controller.py

"""Test script for Markov controller."""
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
logger = logging.getLogger("markov_test")

def display_status(data_manager, pico_manager, controller):
    """Display current system status."""
    ventilation_status = pico_manager.get_ventilation_status()
    ventilation_speed = pico_manager.get_ventilation_speed()
    
    co2 = data_manager.latest_data["scd41"]["co2"]
    temp = data_manager.latest_data["scd41"]["temperature"]
    humidity = data_manager.latest_data["scd41"]["humidity"]
    occupants = data_manager.latest_data["room"]["occupants"]
    
    controller_status = controller.get_status()
    
    print("\n===== SYSTEM STATUS =====")
    print(f"CO2: {co2} ppm")
    print(f"Temperature: {temp}°C")
    print(f"Humidity: {humidity}%")
    print(f"Occupants: {occupants}")
    print(f"Ventilation: {'ON' if ventilation_status else 'OFF'}, Speed: {ventilation_speed}")
    print(f"Auto Mode: {'Enabled' if controller_status['auto_mode'] else 'Disabled'}")
    print(f"Current State: {controller_status['current_state']}")
    print(f"Last Action: {controller_status['last_action']}")
    print(f"Exploration Rate: {controller_status['exploration_rate']:.4f}")
    
    # Determine CO2 level category
    if co2 is not None:
        if co2 < controller.co2_thresholds["low_max"]:
            co2_category = "LOW"
        elif co2 < controller.co2_thresholds["medium_max"]:
            co2_category = "MEDIUM"
        else:
            co2_category = "HIGH"
        print(f"CO2 Level: {co2_category}")
    
    print("========================")

def test_markov_controller():
    """Test the Markov controller."""
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
            scan_interval=10  # 10 seconds between checks for testing
        )
        
        # Enter test data
        print("\nEnter test sensor values:")
        co2_input = input("CO2 level (ppm) [default=800]: ")
        co2 = int(co2_input) if co2_input.strip() else 800
        
        temp_input = input("Temperature (°C) [default=22]: ")
        temp = float(temp_input) if temp_input.strip() else 22.0
        
        humidity_input = input("Humidity (%) [default=50]: ")
        humidity = float(humidity_input) if humidity_input.strip() else 50.0
        
        occupants_input = input("Occupants [default=1]: ")
        occupants = int(occupants_input) if occupants_input.strip() else 1
        
        # Update data manager with test values
        data_manager.latest_data["scd41"]["co2"] = co2
        data_manager.latest_data["scd41"]["temperature"] = temp
        data_manager.latest_data["scd41"]["humidity"] = humidity
        data_manager.latest_data["room"]["occupants"] = occupants
        
        # Display initial status
        display_status(data_manager, pico_manager, controller)
        
        # Start controller
        controller.start()
        print("\n✅ Started Markov controller")
        print("The controller will check conditions every 10 seconds")
        print("Press Ctrl+C to exit")
        
        # Monitor status for a while
        try:
            for i in range(15):  # Run for 15 cycles
                time.sleep(10)
                display_status(data_manager, pico_manager, controller)
                
                # Ask to update CO2 after 7 cycles
                if i == 7:
                    print("\nUpdate test values:")
                    co2_input = input("New CO2 level (ppm) [keep current]: ")
                    if co2_input.strip():
                        data_manager.latest_data["scd41"]["co2"] = int(co2_input)
                    
                    occupants_input = input("New Occupants [keep current]: ")
                    if occupants_input.strip():
                        data_manager.latest_data["room"]["occupants"] = int(occupants_input)
        
        except KeyboardInterrupt:
            print("\nTest interrupted by user")
        
        # Stop controller
        controller.stop()
        print("\n✅ Stopped Markov controller")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in Markov controller test: {e}", exc_info=True)
        print(f"\nError: {e}")
        return False

if __name__ == "__main__":
    print("\n===== MARKOV CONTROLLER TEST =====\n")
    successful = test_markov_controller()
    sys.exit(0 if successful else 1)