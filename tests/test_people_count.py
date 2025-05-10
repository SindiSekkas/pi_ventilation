# tests/test_people_count.py

"""Test script to diagnose people presence detection."""
import time
import logging
import sys
import os
from datetime import datetime

# Add parent directory to the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("people_count_debug.log")
    ]
)
logger = logging.getLogger("people_count_test")

def test_people_presence():
    """Test and diagnose people presence detection."""
    try:
        # Import components
        from sensors.data_manager import DataManager
        from presence.device_manager import DeviceManager
        from presence.models import ConfirmationStatus
        from utils.network_scanner import scan_network
        
        print("\n=== People Presence Detection Diagnostics ===\n")
        
        # Initialize components
        data_manager = DataManager(csv_dir="test_data/csv")
        device_manager = DeviceManager(data_dir="test_data/presence")
        
        # Step 1: Check default occupants in data manager
        current_occupants = data_manager.latest_data["room"]["occupants"]
        print(f"Current occupancy in data manager: {current_occupants} people")
        print(f"Default occupancy setting: {current_occupants}")
        
        # Step 2: Check network scanning
        print("\n--- Network Scanning ---")
        devices = scan_network()
        print(f"Found {len(devices)} devices on network")
        
        phones = []
        for device in devices:
            mac, ip, vendor = device[:3] if len(device) >= 3 else (device[0], device[1], "Unknown")
            print(f"  - {vendor} (MAC: {mac}, IP: {ip})")
            
            # Check if it looks like a phone
            if any(pattern in vendor.lower() for pattern in ['apple', 'iphone', 'samsung', 'xiaomi', 'oneplus']):
                phones.append(device)
        
        print(f"\nFound {len(phones)} potential phones:")
        for phone in phones:
            mac, ip, vendor = phone[:3] if len(phone) >= 3 else (phone[0], phone[1], "Unknown")
            print(f"  - {vendor} (MAC: {mac})")
        
        # Step 3: Check device manager configuration
        print("\n--- Device Manager Configuration ---")
        print(f"Total devices in device manager: {len(device_manager.devices)}")
        
        confirmed_phones = 0
        for mac, device in device_manager.devices.items():
            print(f"\nDevice: {device.name}")
            print(f"  MAC: {device.mac}")
            print(f"  Type: {device.device_type}")
            print(f"  Owner: {device.owner}")
            print(f"  Count for presence: {device.count_for_presence}")
            print(f"  Confirmation status: {device.confirmation_status}")
            print(f"  Current status: {device.status}")
            
            if device.device_type == "phone" and device.count_for_presence:
                confirmed_phones += 1
        
        print(f"\nDevices configured to count for presence: {confirmed_phones}")
        
        # Step 4: Test actual presence calculation
        print("\n--- People Presence Calculation ---")
        
        # Update device status based on current scan
        for device in devices:
            mac = device[0].lower()
            device_manager.update_device_status(mac, True)
        
        # Calculate people present
        people_count = device_manager.calculate_people_present()
        print(f"Calculated people count: {people_count}")
        
        # Update data manager with this count
        data_manager.update_room_data(occupants=people_count)
        new_occupants = data_manager.latest_data["room"]["occupants"]
        print(f"Updated occupancy in data manager: {new_occupants} people")
        
        # Step 5: Interactive testing
        print("\n--- Interactive Testing ---")
        
        if confirmed_phones == 0:
            print("\n⚠️  No phones are configured to count for presence!")
            print("Would you like to add phones from the detected devices?")
            
            choice = input("Add detected phones as recognized devices? (y/n): ")
            if choice.lower() == 'y':
                for phone in phones:
                    mac, ip, vendor = phone[:3] if len(phone) >= 3 else (phone[0], phone[1], "Unknown")
                    name = input(f"\nSet name for {vendor} ({mac}) [default: {vendor}]: ").strip() or vendor
                    owner = input(f"Set owner name for {name} [optional]: ").strip() or None
                    
                    # Use the correct method signature from device_manager.py
                    device_manager.add_device(
                        mac=mac,
                        name=name,
                        owner=owner,
                        device_type="phone",
                        vendor=vendor,
                        count_for_presence=True,
                        confirmation_status=ConfirmationStatus.CONFIRMED.value
                    )
                    print(f"Added {name} to presence detection")
                
                # Recalculate with new devices  
                people_count = device_manager.calculate_people_present()
                print(f"\nRecalculated people count: {people_count}")
                data_manager.update_room_data(occupants=people_count)
        
        # Step 6: Manual override test
        manual_count = input("\nEnter manual people count to test [press enter to skip]: ")
        if manual_count.isdigit():
            count = int(manual_count)
            data_manager.update_room_data(occupants=count)
            print(f"Set occupancy to {count} people")
            print(f"Current occupancy in system: {data_manager.latest_data['room']['occupancy']}")
        
        # Step 7: Monitoring mode
        monitor = input("\nStart continuous monitoring? (y/n): ")
        if monitor.lower() == 'y':
            print("\nStarting continuous monitoring. Press Ctrl+C to stop...")
            try:
                while True:
                    # Scan and update
                    devices = scan_network()
                    for device in devices:
                        mac = device[0].lower()
                        device_manager.update_device_status(mac, True)
                    
                    # Mark missing devices as offline
                    scanned_macs = [d[0].lower() for d in devices]
                    for mac in device_manager.devices:
                        if mac not in scanned_macs:
                            device_manager.update_device_status(mac, False)
                    
                    # Calculate and display
                    people_count = device_manager.calculate_people_present()
                    data_manager.update_room_data(occupants=people_count)
                    
                    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] People present: {people_count}")
                    
                    # Show which devices are active
                    active_phones = [
                        d for d in device_manager.devices.values()
                        if d.device_type == "phone" and d.count_for_presence and d.status == "active"
                    ]
                    
                    if active_phones:
                        print("Active phones:")
                        for phone in active_phones:
                            owner_info = f" ({phone.owner})" if phone.owner else ""
                            print(f"  - {phone.name}{owner_info}")
                    
                    time.sleep(30)
            except KeyboardInterrupt:
                print("\nMonitoring stopped")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error in people presence test: {e}", exc_info=True)
        print(f"\nError: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(test_people_presence())