"""Network scanner for presence detection."""
import subprocess
import re
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

def scan_network():
    """
    Scan local network for devices using arp-scan.
    
    Returns:
        list: List of tuples containing (mac, ip, vendor) of discovered devices
    """
    try:
        # Primary method: sudo arp-scan
        logger.info("Starting network scan with sudo arp-scan")
        result = subprocess.run(
            ["sudo", "arp-scan", "--localnet"], 
            capture_output=True, text=True,
            timeout=30  # Set timeout to prevent hanging
        )
        
        # Extract MAC addresses, IPs, and vendors
        online_devices = []
        seen_macs = set()  # To handle duplicate entries
        
        for line in result.stdout.splitlines():
            # Match IP, MAC address, and vendor pattern
            match = re.search(r'(\d+\.\d+\.\d+\.\d+)\s+([0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2})\s+(.*?)(?:\s+\(DUP: \d+\))?$', line)
            if match:
                ip = match.group(1)
                mac = match.group(2).lower()
                vendor = match.group(3).strip()
                
                # Skip duplicates
                if mac in seen_macs:
                    continue
                    
                seen_macs.add(mac)
                online_devices.append((mac, ip, vendor))
        
        logger.info(f"Network scan found {len(online_devices)} devices")
        return online_devices
    
    except subprocess.TimeoutExpired:
        logger.error("Network scan timed out")
        return fallback_scan()
    except Exception as e:
        logger.error(f"Error scanning network: {e}")
        return fallback_scan()

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