"""
Network device discovery and presence detection utility.
Provides tools to scan local networks and verify device presence using various methods.
"""
import subprocess
import re
import logging
from datetime import datetime
from typing import Optional
import asyncio

logger = logging.getLogger(__name__)

def scan_network(target_ip: Optional[str] = None):
    """
    Scan local network for connected devices using arp-scan.
    
    Args:
        target_ip: Optional specific IP to scan, scans entire subnet if None
        
    Returns:
        list: Discovered devices as (mac, ip, vendor) tuples
    """
    devices = {}  # Using dictionary to deduplicate by MAC

    try:
        if target_ip:
            logger.info(f"Running targeted arp-scan for {target_ip}")
            result = subprocess.run(
                ["sudo", "arp-scan", target_ip], 
                capture_output=True, text=True,
                timeout=10
            )
        else:
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
    
    # Convert dictionary to list for return
    device_list = list(devices.values())
    logger.info(f"TOTAL: Found {len(device_list)} unique devices via all methods")
    
    # Print all found devices for debugging
    for mac, ip, vendor in device_list:
        logger.debug(f"FOUND DEVICE: MAC={mac}, IP={ip}, Vendor={vendor}")
    
    return device_list


async def scan_network_async(target_ip: Optional[str] = None):
    """Asynchronous network scan to prevent blocking."""
    devices = {}
    
    try:
        if target_ip:
            cmd = ["sudo", "arp-scan", target_ip]
        else:
            cmd = ["sudo", "arp-scan", "--localnet"]
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=30)
        output = stdout.decode()
        
        for line in output.splitlines():
            match = re.search(r'(\d+\.\d+\.\d+\.\d+)\s+([0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2})\s+(.+?)(?:\s+\(DUP: \d+\))?$', line)
            if match:
                ip = match.group(1)
                mac = match.group(2).lower()
                vendor = match.group(3).strip()
                devices[mac] = (mac, ip, vendor)
    except Exception as e:
        logger.error(f"Error in async scan: {e}")
    
    return list(devices.values())


def fallback_scan():
    """
    Read device information directly from system ARP table when arp-scan fails.
    
    Returns:
        list: Discovered devices from ARP table as (mac, ip, "Unknown") tuples
    """
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
                for line in f.readlines()[1:]:
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
    Test network connectivity to a device via ICMP ping.
    
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
    Determine if a specific device is present on the network using multiple detection methods.
    
    Args:
        mac_address: MAC address to check
        ip_address: IP address if known, otherwise will attempt to discover
        methods: List of methods to try ['arp_scan', 'arp_table', 'ping']
        
    Returns:
        tuple: (is_present, detection_method, additional_info)
    """
    methods = methods or ['arp_scan', 'arp_table', 'ping']
    mac_address = mac_address.lower()
    
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
    Classify device type based on manufacturer information.
    
    Args:
        mac: MAC address
        vendor: Vendor string from network scan
    
    Returns:
        str: Device classification (phone, laptop, tablet, tv, iot_device, unknown)
    """
    mac = mac.lower()
    vendor = vendor.lower()

    # PHONE detection pattern list
    phone_patterns = [
        'apple', 'iphone', 'ipad', 'samsung', 'huawei',
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
    Determine the likelihood that a device is a personal mobile device.
    
    Args:
        vendor: Vendor string from network scan
    
    Returns:
        float: Confidence score between 0 and 1
    """
    vendor = vendor.lower()
    
    # High confidence phone manufacturers
    if any(name in vendor for name in ['apple', 'iphone', 'samsung', 'huawei', 'google pixel']):
        return 0.9
    
    # Medium confidence phone manufacturers
    if any(name in vendor for name in ['oneplus', 'oppo', 'vivo', 'motorola', 'nokia']):
        return 0.7
    
    # Probably not a phone
    if any(name in vendor for name in ['raspberry', 'arduino', 'printer', 'router', 'switch']):
        return 0.1
    
    # Default to low-medium confidence
    return 0.4