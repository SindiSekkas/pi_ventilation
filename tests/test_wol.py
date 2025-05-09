"""Test script for Wake-on-LAN functionality."""
import time
import logging
import sys
import os
from datetime import datetime

# parent directory to the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# Setup logging
logging.basicConfig(
    level=logging.DEBUG,  # Use DEBUG level to see all messages
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("wol_test")

def test_wol():
    """Test Wake-on-LAN functionality."""
    from utils.wol import wake_device, check_device_responds, wake_and_check
    
    print("\n=== Wake-on-LAN Test ===\n")
    
    # Get MAC and IP address from user input
    mac = input("Enter MAC address to wake (e.g., 44:da:30:bd:cb:88): ")
    ip = input("Enter IP address to check (e.g., 192.168.0.102): ")
    
    print(f"\nTesting Wake-on-LAN for device: {mac} ({ip})")
    
    # Check if device already responds
    print("\nChecking if device already responds...")
    if check_device_responds(ip):
        print("✅ Device is already responding to ping")
    else:
        print("❌ Device is not responding to ping")
    
    # Send WoL packet
    print("\nSending Wake-on-LAN magic packet...")
    success = wake_device(mac)
    if success:
        print("✅ Magic packet sent successfully")
    else:
        print("❌ Failed to send magic packet")
    
    # Wait and check
    print("\nWaiting for device to wake up...")
    for i in range(5):
        time.sleep(2)
        if check_device_responds(ip):
            print(f"✅ Device responded after {(i+1)*2} seconds!")
            break
        else:
            print(f"⏳ No response after {(i+1)*2} seconds...")
    else:
        print("❌ Device failed to respond after 10 seconds")
    
    # Try wake_and_check
    print("\nTrying combined wake_and_check function...")
    result = wake_and_check(mac, ip, max_attempts=2)
    if result:
        print("✅ wake_and_check succeeded")
    else:
        print("❌ wake_and_check failed")
    
    return 0

if __name__ == "__main__":
    sys.exit(test_wol())