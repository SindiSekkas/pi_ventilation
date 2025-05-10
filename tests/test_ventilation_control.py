"""Test script for Pico ventilation control.

For interactive menu:
python test_ventilation_control.py

For automated test:
python test_ventilation_control.py --auto

If Pico W has a different IP:
python test_ventilation_control.py --ip 192.168.1.100"""

# tests/test_ventilation_control.py


import time
import logging
import sys
import argparse
from datetime import datetime

# parent directory to the python path
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("ventilation_test")

# Import components
try:
    from utils.pico_manager import PicoManager
except ImportError:
    logger.error("Unable to import PicoManager. Please check your installation.")
    sys.exit(1)

def test_ventilation_control(pico_ip="192.168.0.110", auto_test=False):
    """Test ventilation control through Pico W."""
    # Initialize the Pico manager
    pico = PicoManager(pico_ip)
    
    # Check connection
    if not pico.find_pico_service():
        logger.error(f"Failed to connect to Pico W at {pico_ip}")
        return False
    
    logger.info(f"Successfully connected to Pico W at {pico_ip}")
    
    # Get initial status
    status = pico.get_ventilation_status()
    speed = pico.get_ventilation_speed()
    logger.info(f"Initial state: Ventilation {'ON' if status else 'OFF'}, Speed: {speed}")
    
    if auto_test:
        return run_automated_test(pico)
    else:
        run_interactive_menu(pico)
        return True

def run_automated_test(pico):
    """Run an automated test cycle through all ventilation speeds."""
    print("\n====== STARTING AUTOMATED TEST SEQUENCE ======")
    print("This will cycle through all ventilation speeds")
    print("Press Ctrl+C to abort at any time")
    
    # Define the test sequence
    test_sequence = [
        {"name": "OFF", "state": "off"},
        {"name": "LOW", "state": "on", "speed": "low"},
        {"name": "MEDIUM", "state": "on", "speed": "medium"},
        {"name": "MAX", "state": "on", "speed": "max"},
        {"name": "OFF", "state": "off"}  # Return to off state at the end
    ]
    
    test_results = []
    all_passed = True
    
    # Run each test in the sequence
    for i, test in enumerate(test_sequence):
        step_num = i + 1
        print(f"\nStep {step_num}/{len(test_sequence)}: Setting ventilation to {test['name']}")
        
        # Set the ventilation state
        success = False
        if test["state"] == "off":
            success = pico.control_ventilation("off")
        else:
            success = pico.control_ventilation("on", test["speed"])
        
        # Wait for the change to take effect
        time.sleep(2)
        
        # Verify the change
        status = pico.get_ventilation_status()
        speed = pico.get_ventilation_speed()
        
        expected_status = (test["state"] == "on")
        expected_speed = test["speed"] if test["state"] == "on" else "off"
        
        verification = (status == expected_status and speed == expected_speed)
        
        # Record results
        result = {
            "step": step_num,
            "command": test["name"],
            "command_success": success,
            "verification_success": verification,
            "actual_status": "ON" if status else "OFF",
            "actual_speed": speed
        }
        test_results.append(result)
        
        if not success or not verification:
            all_passed = False
        
        # Display result
        if success and verification:
            print(f"✅ Step {step_num} PASSED: Ventilation set to {test['name']} successfully")
        else:
            print(f"❌ Step {step_num} FAILED")
            if not success:
                print("   - Command failed to execute")
            if not verification:
                print(f"   - Verification failed: Expected {'ON' if expected_status else 'OFF'} and {expected_speed}, "
                      f"Got {'ON' if status else 'OFF'} and {speed}")
        
        # Pause between steps for better observation
        if i < len(test_sequence) - 1:
            print(f"Waiting 5 seconds before next step...")
            time.sleep(5)
    
    # Display summary
    print("\n====== AUTOMATED TEST RESULTS ======")
    for result in test_results:
        status = "✅ PASSED" if result["command_success"] and result["verification_success"] else "❌ FAILED"
        print(f"Step {result['step']} ({result['command']}): {status}")
    
    if all_passed:
        print("\n✅ ALL TESTS PASSED! Ventilation control is working correctly.")
    else:
        print("\n❌ SOME TESTS FAILED. Check the log for details.")
    
    return all_passed

def run_interactive_menu(pico):
    """Run an interactive menu for manual testing."""
    while True:
        # Get current status
        try:
            status = pico.get_ventilation_status()
            speed = pico.get_ventilation_speed()
        except Exception as e:
            logger.error(f"Error getting ventilation status: {e}")
            status = False
            speed = "unknown"
        
        # Display menu
        print("\n====== VENTILATION CONTROL TEST ======")
        print(f"Current State: Ventilation {'ON' if status else 'OFF'}, Speed: {speed}")
        print("\nOptions:")
        print("1. Turn Ventilation OFF")
        print("2. Set Ventilation to LOW")
        print("3. Set Ventilation to MEDIUM")
        print("4. Set Ventilation to MAX")
        print("5. Run Automated Test Sequence")
        print("6. Exit")
        
        # Get user input
        try:
            choice = input("\nEnter your choice (1-6): ")
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        
        # Process choice
        try:
            if choice == "1":
                result = pico.control_ventilation("off")
                print(f"Command {'succeeded' if result else 'failed'}")
            elif choice == "2":
                result = pico.control_ventilation("on", "low")
                print(f"Command {'succeeded' if result else 'failed'}")
            elif choice == "3":
                result = pico.control_ventilation("on", "medium")
                print(f"Command {'succeeded' if result else 'failed'}")
            elif choice == "4":
                result = pico.control_ventilation("on", "max")
                print(f"Command {'succeeded' if result else 'failed'}")
            elif choice == "5":
                run_automated_test(pico)
            elif choice == "6":
                print("Exiting...")
                break
            else:
                print("Invalid choice. Please enter a number between 1 and 6.")
        except Exception as e:
            logger.error(f"Error processing command: {e}")
            print(f"Error: {e}")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Test ventilation control through Pico W")
    parser.add_argument("--ip", default="192.168.0.110", help="IP address of the Pico W")
    parser.add_argument("--auto", action="store_true", help="Run automated test sequence")
    args = parser.parse_args()
    
    # Run test
    success = test_ventilation_control(pico_ip=args.ip, auto_test=args.auto)
    
    # Exit with appropriate status code
    sys.exit(0 if success else 1)