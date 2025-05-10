# utils/network_scanner.py

"""Network scanner for presence detection."""
import subprocess
import re
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

def scan_network():
    """
    Scan local network for devices using BOTH arp-scan AND ARP table.
    Returns combined list of all found devices.
    """
    devices = {}  # Dictionary to prevent duplicates by MAC address
    
    # PART 1: ARP-SCAN
    try:
        logger.info("Running arp-scan to find devices")
        result = subprocess.run(
            ["sudo", "arp-scan", "--localnet"], 
            capture_output=True, text=True,
            timeout=30
        )
        
        for line in result.stdout.splitlines():
            match = re.search(r'(\d+\.\d+\.\d+\.\d+)\s+([0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2})\s+(.+?)(?:\s+\(DUP: \d+\))?$', line)
            if match:
                ip = match.group(1)
                mac = match.group(2).lower()
                vendor = match.group(3).strip()
                
                devices[mac] = (mac, ip, vendor)
                
        logger.info(f"ARP-SCAN found {len(devices)} devices")
    except Exception as e:
        logger.error(f"Error in arp-scan: {e}")
    
    # PART 2: READ /proc/net/arp FILE
    try:
        logger.info("Reading /proc/net/arp file")
        with open('/proc/net/arp', 'r') as f:
            lines = f.readlines()[1:]  # Skip header
            
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 4 and parts[3] != "00:00:00:00:00:00":
                ip = parts[0]
                mac = parts[3].lower()
                
                # Only add if not already found
                if mac not in devices:
                    logger.info(f"Found additional device in ARP table: {mac} ({ip})")
                    devices[mac] = (mac, ip, "Unknown")
    except Exception as e:
        logger.error(f"Error reading /proc/net/arp: {e}")
    
    # PART 3: RUN ARP COMMAND
    try:
        logger.info("Running 'arp -a' command")
        result = subprocess.run(
            ["arp", "-a"], 
            capture_output=True, text=True,
            timeout=5
        )
        
        # Parse output to find MAC addresses
        for line in result.stdout.splitlines():
            # Try to extract IP and MAC
            match = re.search(r'\((\d+\.\d+\.\d+\.\d+)\)\s+at\s+([0-9a-fA-F:]{17})', line)
            if match:
                ip = match.group(1)
                mac = match.group(2).lower()
                
                # Only add if not already found
                if mac not in devices:
                    logger.info(f"Found additional device via 'arp -a': {mac} ({ip})")
                    devices[mac] = (mac, ip, "Unknown")
    except Exception as e:
        logger.error(f"Error running arp command: {e}")
    
    # Convert dictionary to list for return
    device_list = list(devices.values())
    logger.info(f"TOTAL: Found {len(device_list)} unique devices via all methods")
    
    # Print all found devices for debugging
    for mac, ip, vendor in device_list:
        logger.debug(f"FOUND DEVICE: MAC={mac}, IP={ip}, Vendor={vendor}")
    
    return device_list

def fallback_scan():
    """Fallback method: read ARP table directly."""
    try:
        logger.info("Using fallback to ARP table")
        with open('/proc/net/arp', 'r') as f:
            lines = f.readlines()[1:]  # Skip header
            
        online_devices = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 4 and parts[3] != "00:00:00:00:00:00":
                ip = parts[0]
                mac = parts[3].lower()
                online_devices.append((mac, ip, "Unknown"))
                
        logger.info(f"Fallback method found {len(online_devices)} devices")
        return online_devices
    except Exception as e:
        logger.error(f"Fallback also failed: {e}")
        return []

def check_arp_table(mac_address):
    """
    Check if a device is in the ARP table.
    
    Args:
        mac_address: MAC address to check
        
    Returns:
        bool: True if device is in ARP table, False otherwise
    """
    try:
        # Normalize MAC format for comparison
        mac = mac_address.lower().replace('-', ':')
        
        # First check /proc/net/arp file
        try:
            with open('/proc/net/arp', 'r') as f:
                for line in f.readlines()[1:]:  # Skip header
                    parts = line.strip().split()
                    if len(parts) >= 4 and parts[3].lower() == mac:
                        logger.debug(f"Device {mac} found in /proc/net/arp")
                        return True
        except Exception:
            pass
        
        # Then try using the arp command
        result = subprocess.run(
            ["arp", "-a"], 
            capture_output=True, 
            text=True,
            timeout=2
        )
        
        # Look for the MAC in the output
        if mac in result.stdout.lower():
            logger.debug(f"Device {mac} found in arp -a output")
            return True
            
        return False
        
    except Exception as e:
        logger.error(f"Error checking ARP table for {mac_address}: {e}")
        return False

def ping_device(ip_address, count=1, timeout=1):
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
        success = result.returncode == 0
        if success:
            logger.debug(f"Ping to {ip_address} successful")
        return success
        
    except Exception as e:
        logger.error(f"Error pinging device {ip_address}: {e}")
        return False

def check_device_presence(mac_address, ip_address=None, methods=None):
    """
    Check if a specific device is present using multiple methods.
    
    Args:
        mac_address: MAC address to check
        ip_address: IP address if known, otherwise attempt to find it
        methods: List of methods to try ['arp_scan', 'arp_table', 'ping']
        
    Returns:
        tuple: (is_present, detection_method, additional_info)
    """
    methods = methods or ['arp_scan', 'arp_table', 'ping']
    mac_address = mac_address.lower()
    
    # Method 1: Check ARP table
    if 'arp_table' in methods:
        arp_present = check_arp_table(mac_address)
        if arp_present:
            return (True, 'arp_table', None)
    
    # Method 2: Check with directed ARP scan if IP is known
    if 'arp_scan' in methods and ip_address:
        try:
            result = subprocess.run(
                ["sudo", "arp-scan", ip_address], 
                capture_output=True, 
                text=True,
                timeout=2
            )
            if mac_address in result.stdout.lower():
                return (True, 'direct_arp_scan', None)
        except Exception:
            pass
    
    # Method 3: Try to find IP if not provided
    if not ip_address and 'arp_scan' in methods:
        try:
            # Do a quick network scan to find the device
            devices = scan_network()
            for device_mac, device_ip, _ in devices:
                if device_mac.lower() == mac_address.lower():
                    ip_address = device_ip
                    return (True, 'network_scan', {'ip': device_ip})
        except Exception:
            pass
    
    # Method 4: Try ping if IP is known
    if 'ping' in methods and ip_address:
        ping_result = ping_device(ip_address)
        if ping_result:
            return (True, 'ping', {'ip': ip_address})
    
    return (False, None, None)

def guess_device_type(mac, vendor):
    """
    Guess device type based on MAC address and vendor.
    
    Args:
        mac: MAC address
        vendor: Vendor name
    
    Returns:
        str: Device type (phone, laptop, tablet, tv, iot_device, unknown)
    """
    mac = mac.lower()
    vendor = vendor.lower()

    # PHONE detection pattern list
    phone_patterns = [
        'apple', 'iphone', 'ipad', 'samsung', 'xiaomi', 'huawei',
        'oneplus', 'google', 'motorola', 'nokia', 'sony mobile',
        'htc', 'oppo', 'vivo', 'realme', 'lg electronics'
    ]
    
    # LAPTOP detection pattern list
    laptop_patterns = [
        'intel', 'dell', 'lenovo', 'hp', 'compaq', 'asus', 'acer',
        'microsoft', 'toshiba', 'msi', 'alienware', 'samsung electronics',
        'panasonic computer', 'asustek', 'asrock'
    ]

    # SMART TV detection pattern list
    tv_patterns = [
        'samsung tv', 'lg electronics', 'sony', 'philips', 'vizio',
        'roku', 'hisense', 'tcl', 'panasonic', 'sharp'
    ]

    # IOT DEVICE pattern list
    iot_patterns = [
        'nest', 'ring', 'ecobee', 'tuya', 'sonos', 'amazon',
        'google home', 'philips hue', 'belkin', 'netatmo', 'arlo',
        'blink', 'sonoff', 'broadlink', 'tp-link', 'd-link', 'azurewave', 'compal',
        'raspberry pi', 'arduino', 'beaglebone'
    ]

    # Check phone patterns first (highest priority)
    if any(pattern in vendor for pattern in phone_patterns):
        return 'phone'

    # Check laptop patterns next
    if any(pattern in vendor for pattern in laptop_patterns):
        return 'laptop'

    # Check TV patterns
    if any(pattern in vendor for pattern in tv_patterns):
        return 'tv'

    # Check IoT device patterns
    if any(pattern in vendor for pattern in iot_patterns):
        return 'iot_device'

    # If no match, return unknown
    return 'unknown'

def get_vendor_confidence_score(vendor):
    """
    Calculate confidence score for a device being a personal device based on vendor.
    
    Args:
        vendor: Vendor string from network scan
    
    Returns:
        float: Confidence score between 0 and 1
    """
    vendor = vendor.lower()
    
    # High confidence phone manufacturers
    if any(name in vendor for name in ['apple', 'iphone', 'samsung', 'xiaomi', 'huawei', 'google pixel']):
        return 0.9
    
    # Medium confidence phone manufacturers
    if any(name in vendor for name in ['oneplus', 'oppo', 'vivo', 'motorola', 'nokia']):
        return 0.7
    
    # Probably not a phone
    if any(name in vendor for name in ['raspberry', 'arduino', 'printer', 'router', 'switch']):
        return 0.1
    
    # Default to low-medium confidence
    return 0.4