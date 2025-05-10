"""Test script for presence detection functionality."""
import time
import logging
import sys
import os
from datetime import datetime

# Add parent directory to module search path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("presence_test")

def test_presence_detection():
    """Test the presence detection system."""
    try:
        # Import required components
        from utils.network_scanner import scan_network, guess_device_type
        from presence.models import Device, DeviceType, ConfirmationStatus
        
        # Create test directory
        os.makedirs("test_data", exist_ok=True)
        
        # Run network scan to get devices
        print("\n=== Running Network Scan ===")
        devices = scan_network()
        print(f"Found {len(devices)} devices on the network")
        
        # Print discovered devices
        for i, device in enumerate(devices):
            if len(device) == 3:
                mac, ip, vendor = device
            else:
                mac, ip = device
                vendor = "Unknown"
                
            device_type = guess_device_type(mac, vendor)
            print(f"{i+1}. MAC: {mac}, IP: {ip}, Vendor: {vendor}, Type: {device_type}")
            
        # Process devices manually
        print("\n=== Processing Devices ===")
        processed_devices = []
        
        for device_info in devices:
            if len(device_info) == 3:
                mac, ip, vendor = device_info
            else:
                mac, ip = device_info
                vendor = "Unknown"
                
            device_type = guess_device_type(mac, vendor)
            
            # Create device object
            device = Device(
                mac=mac,
                name=vendor if vendor != "Unknown" else f"New-{mac[-5:]}",
                device_type=device_type,
                vendor=vendor,
                count_for_presence=(device_type == "phone")  # Count phones for presence
            )
            
            # Mark as active
            device.status = "active"
            device.last_seen = datetime.now().isoformat()
            
            processed_devices.append(device)
            print(f"Processed: {device.name} ({device.mac}, type: {device.device_type})")
            
        # Calculate people count
        phones = [d for d in processed_devices 
                 if d.device_type == "phone" and d.count_for_presence]
        unique_vendors = len(set(d.vendor for d in phones))
        
        print(f"\nFound {len(phones)} phones from {unique_vendors} unique vendors")
        print(f"Estimated people present: {len(phones)}")
        
        # Print all detected devices
        print("\n=== Detected Devices ===")
        
        # Sort with phones first, then by vendor
        processed_devices.sort(key=lambda d: (d.device_type != "phone", d.vendor))
        
        for i, device in enumerate(processed_devices):
            status_icon = "✅" if device.status == "active" else "❌"
            type_icon = "📱" if device.device_type == "phone" else "💻"
            
            print(f"{i+1}. {status_icon} {type_icon} {device.name}")
            print(f"   MAC: {device.mac}")
            print(f"   Type: {device.device_type}, Vendor: {device.vendor}")
            print(f"   Would count for presence: {device.count_for_presence}")
            print("")
        
        # Simulate confirming a device
        if phones:
            print(f"\n=== Simulating device confirmation ===")
            phone = phones[0]
            print(f"Would confirm device: {phone.name} ({phone.mac})")
            print(f"After confirmation, it would count as a person present")
        
        print("\nSimple presence detection test completed successfully.")
        return True
        
    except Exception as e:
        logger.error(f"Error in presence detection test: {e}", exc_info=True)
        print(f"\nError: {e}")
        return False

if __name__ == "__main__":
    print("\n===== PRESENCE DETECTION SYSTEM TEST =====\n")
    result = test_presence_detection()
    sys.exit(0 if result else 1)

# tests/test_presence.py