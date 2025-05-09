"""Script to integrate and test presence detection with the main system."""
import os
import sys
import time
import logging
import threading
from datetime import datetime

# parent directory to sys.path to allow importing modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("presence_integration.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("presence_integration")

# Import components
from sensors.data_manager import DataManager
from utils.pico_manager import PicoManager
from presence.device_manager import DeviceManager
from presence.presence_controller import PresenceController
from config.settings import PICO_IP

def notification_handler(action, **kwargs):
    """Example notification handler for device events."""
    if action == "new_device":
        logger.info(f"New device detected: {kwargs.get('device_name', 'Unknown device')}")
        
        if kwargs.get('device_type') == 'phone':
            confidence = kwargs.get('confidence', 0)
            if confidence > 0.7:
                logger.info(f"High confidence phone detected: {kwargs.get('device_name')} "
                           f"(MAC: {kwargs.get('device_mac')}, Vendor: {kwargs.get('vendor')})")
                # Here you could send a notification via Telegram, etc.
                print(f"\n🔔 NEW PHONE DETECTED: {kwargs.get('device_name')}")
                print(f"   Confidence score: {confidence:.2f}")
                print(f"   MAC: {kwargs.get('device_mac')}")
                print(f"   Vendor: {kwargs.get('vendor')}")
                print(f"   When adding Telegram bot, this would trigger a notification.")

def status_monitor(device_manager, data_manager, interval=10):
    """Monitor and display presence status periodically."""
    previous_count = -1
    
    while True:
        try:
            # Calculate people present
            people_count = device_manager.calculate_people_present()
            
            # If count changed, update data manager
            if people_count != previous_count:
                data_manager.update_room_data(occupants=people_count)
                previous_count = people_count
                
                # Print status update
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 👤 People count updated: {people_count}")
                
                # List active devices counted for presence
                active_phones = [d for d in device_manager.devices.values() 
                              if d.status == "active" and 
                              d.count_for_presence and 
                              d.device_type == "phone"]
                
                if active_phones:
                    print("Active phones:")
                    for phone in active_phones:
                        owner = f" ({phone.owner})" if phone.owner else ""
                        print(f"  - {phone.name}{owner}")
                else:
                    print("No active phones detected")
            
            # Sleep until next check
            time.sleep(interval)
            
        except Exception as e:
            logger.error(f"Error in status monitor: {e}")
            time.sleep(interval)

def main():
    """Main integration function."""
    try:
        print("\n===== PRESENCE DETECTION INTEGRATION =====\n")
        
        # Create data directories
        os.makedirs("data", exist_ok=True)
        os.makedirs("data/csv", exist_ok=True)
        os.makedirs("data/presence", exist_ok=True)
        
        # Initialize components
        data_manager = DataManager(csv_dir="data/csv")
        pico_manager = PicoManager(pico_ip=PICO_IP)
        
        # Check Pico connection
        if pico_manager.find_pico_service():
            print(f"✅ Connected to Pico W at {PICO_IP}")
            ventilation_status = pico_manager.get_ventilation_status()
            ventilation_speed = pico_manager.get_ventilation_speed()
            print(f"Current ventilation status: {'ON' if ventilation_status else 'OFF'}, Speed: {ventilation_speed}")
        else:
            print(f"⚠️ Could not connect to Pico W at {PICO_IP}")
        
        # Initialize device manager with notification callback
        device_manager = DeviceManager(
            data_dir="data/presence",
            notification_callback=notification_handler
        )
        
        # Create presence controller
        presence_controller = PresenceController(
            device_manager=device_manager,
            data_manager=data_manager,
            scan_interval=60  # Scan every 60 seconds
        )
        
        # Start presence controller
        if presence_controller.start():
            print("✅ Presence detection started successfully")
        else:
            print("❌ Failed to start presence detection")
            return 1
        
        # Start status monitor in a separate thread
        status_thread = threading.Thread(
            target=status_monitor,
            args=(device_manager, data_manager, 15),
            daemon=True
        )
        status_thread.start()
        
        print("\nPresence detection is running...")
        print("Press Ctrl+C to exit")
        
        # Keep main thread alive
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping presence detection...")
            presence_controller.stop()
            print("Presence detection stopped")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error in presence integration: {e}", exc_info=True)
        print(f"\nError: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())