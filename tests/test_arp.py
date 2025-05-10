"""Test script for WiFi-based presence detection."""
import time
import logging
import sys
import os
import subprocess
import json
from datetime import datetime, timedelta

# Add parent directory to the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("wifi_presence")

class WiFiPresenceDetector:
    """Detects device presence using WiFi and ARP scanning."""
    
    def __init__(self, target_devices=None, history_size=10):
        """
        Initialize the WiFi presence detector.
        
        Args:
            target_devices: Dictionary {name: mac_address} of devices to monitor
            history_size: Number of scans to keep in history
        """
        self.target_devices = target_devices or {}
        self.history_size = history_size
        self.device_history = {}  # {mac: [timestamps]}
        self.device_status = {}   # {mac: True/False}
        self.last_scan_results = []
        self.last_scan_time = None
        
        # Initialize history and status for all target devices
        for name, mac in self.target_devices.items():
            mac = self._normalize_mac(mac)
            self.device_history[mac] = []
            self.device_status[mac] = False
    
    def _normalize_mac(self, mac):
        """Normalize MAC address format."""
        mac = mac.lower().replace(':', '').replace('-', '')
        return ':'.join(mac[i:i+2] for i in range(0, 12, 2))
    
    def scan_network(self, timeout=5):
        """
        Scan local network for devices using arp-scan.
        
        Args:
            timeout: Timeout for scan in seconds
            
        Returns:
            list: List of tuples containing (mac, ip) of discovered devices
        """
        try:
            # Run arp-scan to find all devices on the network
            result = subprocess.run(
                ["sudo", "arp-scan", "--localnet"], 
                capture_output=True, text=True,
                timeout=timeout
            )
            
            # Extract MAC addresses and IPs
            devices = []
            for line in result.stdout.splitlines():
                # Look for lines with IP and MAC pattern
                if line.count(':') == 5:  # MAC address has 5 colons
                    parts = line.split()
                    if len(parts) >= 2:
                        ip = parts[0]
                        mac = self._normalize_mac(parts[1])
                        devices.append((mac, ip))
            
            self.last_scan_results = devices
            self.last_scan_time = datetime.now()
            
            logger.info(f"ARP scan found {len(devices)} devices")
            return devices
            
        except subprocess.TimeoutExpired:
            logger.error("Network scan timed out")
            return []
            
        except Exception as e:
            logger.error(f"Error scanning network: {e}")
            return []
    
    def check_arp_table(self, mac_address):
        """
        Check if a device is in the system's ARP table.
        
        Args:
            mac_address: MAC address to check
            
        Returns:
            bool: True if device is in ARP table, False otherwise
        """
        try:
            # Normalize the MAC address
            mac = self._normalize_mac(mac_address)
            
            # Check ARP table
            result = subprocess.run(
                ["arp", "-a"], 
                capture_output=True, text=True,
                timeout=2
            )
            
            # Look for the MAC address in the output
            return mac.lower() in result.stdout.lower()
            
        except Exception as e:
            logger.error(f"Error checking ARP table: {e}")
            return False
    
    def ping_device(self, ip_address, count=1, timeout=1):
        """
        Ping a device to check if it's responsive.
        
        Args:
            ip_address: IP address to ping
            count: Number of ping packets to send
            timeout: Timeout for each ping in seconds
            
        Returns:
            bool: True if device responds, False otherwise
        """
        try:
            result = subprocess.run(
                ["ping", "-c", str(count), "-W", str(timeout), ip_address],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout+1
            )
            return result.returncode == 0
            
        except Exception as e:
            logger.error(f"Error pinging device {ip_address}: {e}")
            return False
    
    def update_device_status(self):
        """
        Update status of all target devices using latest scan results.
        
        Returns:
            dict: Dictionary of {name: status} for all target devices
        """
        # Scan the network
        devices = self.scan_network()
        mac_list = [mac for mac, _ in devices]
        
        results = {}
        
        # Check each target device
        for name, mac in self.target_devices.items():
            mac = self._normalize_mac(mac)
            
            # Check if the device is in the scan results
            if mac in mac_list:
                device_present = True
                logger.info(f"Device {name} ({mac}) is present in ARP scan")
            else:
                # If not found in scan, try ARP table as backup
                device_present = self.check_arp_table(mac)
                if device_present:
                    logger.info(f"Device {name} ({mac}) found in ARP table")
                else:
                    logger.info(f"Device {name} ({mac}) not detected")
            
            # Update status
            self.device_status[mac] = device_present
            
            # Update history
            if device_present:
                self.device_history[mac].append(datetime.now())
                # Trim history if needed
                if len(self.device_history[mac]) > self.history_size:
                    self.device_history[mac] = self.device_history[mac][-self.history_size:]
            
            results[name] = device_present
        
        return results
    
    def is_probably_present(self, mac_address, max_absence_minutes=10):
        """
        Determine if a device is probably present even if not detected in latest scan.
        
        Args:
            mac_address: MAC address to check
            max_absence_minutes: Maximum minutes of absence to still consider present
            
        Returns:
            bool: True if device is probably present, False otherwise
        """
        mac = self._normalize_mac(mac_address)
        
        # If device is currently detected, it's present
        if self.device_status.get(mac, False):
            return True
        
        # Check recent history
        history = self.device_history.get(mac, [])
        if not history:
            return False
        
        # Check if device was seen recently
        last_seen = history[-1]
        time_since_last_seen = datetime.now() - last_seen
        
        return time_since_last_seen < timedelta(minutes=max_absence_minutes)

    def count_people_present(self, max_absence_minutes=10):
        """
        Estimate the number of people present based on device detection.
        
        Args:
            max_absence_minutes: Maximum minutes of absence to still consider present
            
        Returns:
            int: Estimated number of people present
        """
        count = 0
        detected_devices = []
        
        for name, mac in self.target_devices.items():
            if self.is_probably_present(mac, max_absence_minutes):
                count += 1
                detected_devices.append(name)
        
        logger.info(f"Estimated people count: {count} (Devices: {', '.join(detected_devices) if detected_devices else 'none'})")
        return count


def test_wifi_presence():
    """Test WiFi presence detection."""
    print("\n=== WiFi Presence Detection Test ===\n")
    
    # First, check if arp-scan is installed
    try:
        subprocess.run(["which", "arp-scan"], check=True, capture_output=True)
    except subprocess.CalledProcessError:
        print("❌ arp-scan not found. Please install with:")
        print("   sudo apt-get install arp-scan")
        return 1
    
    # Check if running as root (needed for arp-scan)
    if os.geteuid() != 0:
        print("\n⚠️  Warning: This script should be run as root to use arp-scan")
        print("   Try running with: sudo python tests/test_wifi_presence.py")
        return 1
    
    # Define target devices - ADD YOUR DEVICES HERE
    target_devices = {
        "iPhone": "44:da:30:bd:cb:88",  # Replace with your actual iPhone MAC
        "OnePlus": "30:bb:7d:c6:ea:45",  # Add your OnePlus MAC
        # Add more devices as needed
    }
    
    print("Target devices:")
    for name, mac in target_devices.items():
        print(f"- {name}: {mac}")
    
    # Create detector
    detector = WiFiPresenceDetector(target_devices)
    
    # Run initial scan
    print("\nRunning initial network scan...")
    devices = detector.scan_network()
    
    if not devices:
        print("❌ No devices found in ARP scan. Check your network setup.")
        return 1
    
    print(f"✅ Found {len(devices)} devices on the network:")
    for i, (mac, ip) in enumerate(devices):
        print(f"{i+1}. MAC: {mac}, IP: {ip}")
    
    # Check for target devices
    print("\nChecking for target devices...")
    status = detector.update_device_status()
    
    for name, present in status.items():
        if present:
            print(f"✅ {name} is PRESENT")
        else:
            print(f"❌ {name} is NOT DETECTED")
    
    # Continuous monitoring option
    try:
        choice = input("\nWould you like to continuously monitor presence? (y/n): ")
        if choice.lower() == 'y':
            print("\nStarting continuous monitoring. Press Ctrl+C to stop...")
            print("Scan interval is 60 seconds. Results will be logged to file.")
            
            # Setup file logging
            log_file = os.path.join(os.path.dirname(__file__), "wifi_presence.log")
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
            logger.addHandler(file_handler)
            
            scan_count = 1
            while True:
                print(f"\n--- Scan #{scan_count} ---")
                status = detector.update_device_status()
                count = detector.count_people_present()
                
                print(f"Detected people count: {count}")
                for name, present in status.items():
                    if present:
                        print(f"✅ {name} is PRESENT")
                    else:
                        print(f"❌ {name} is NOT DETECTED")
                
                # Log to file in JSON format for later analysis
                log_data = {
                    "timestamp": datetime.now().isoformat(),
                    "scan_number": scan_count,
                    "people_count": count,
                    "device_status": status
                }
                logger.info(f"PRESENCE_DATA: {json.dumps(log_data)}")
                
                # Wait for next scan
                print(f"Waiting 60 seconds for next scan...")
                time.sleep(60)
                scan_count += 1
    
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")
    
    return 0


if __name__ == "__main__":
    try:
        sys.exit(test_wifi_presence())
    except Exception as e:
        logger.error(f"Error running WiFi presence test: {e}", exc_info=True)
        print(f"\nError: {e}")
        sys.exit(1)

# tests/test_arp.py