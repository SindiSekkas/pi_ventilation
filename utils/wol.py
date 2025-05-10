"""Wake-on-LAN (WoL) functionality for waking network devices."""
# utils/wol.py
import logging
from wakeonlan import send_magic_packet
import socket
import subprocess
import time

logger = logging.getLogger(__name__)

def wake_device(mac_address):
    """
    Send a Wake-on-LAN magic packet to wake a device.
    
    Args:
        mac_address: MAC address of device to wake
        
    Returns:
        bool: Success status
    """
    try:
        # Format MAC address - ensure proper format for WoL
        mac = mac_address.replace(':', '').replace('-', '').lower()
        mac = ':'.join(mac[i:i+2] for i in range(0, 12, 2))
        
        logger.info(f"Sending WoL magic packet to {mac}")
        send_magic_packet(mac)
        return True
    except Exception as e:
        logger.error(f"Error sending WoL packet to {mac_address}: {e}")
        return False

def check_device_responds(ip_address, timeout=3):
    """
    Check if a device responds to ping after wake-up attempt.
    
    Args:
        ip_address: IP address to ping
        timeout: Timeout in seconds
        
    Returns:
        bool: Whether device responded
    """
    try:
        # Try ping
        result = subprocess.run(
            ["ping", "-c", "1", "-W", str(timeout), ip_address],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout+1
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        logger.debug(f"Ping timeout for {ip_address}")
        return False
    except Exception as e:
        logger.error(f"Error pinging {ip_address}: {e}")
        return False

def wake_and_check(mac_address, ip_address, max_attempts=2):
    """
    Wake a device and check if it responds.
    
    Args:
        mac_address: MAC address of device to wake
        ip_address: IP address to check after waking
        max_attempts: Maximum number of wake attempts
        
    Returns:
        bool: Whether device was successfully woken
    """
    # Check if already responding
    if check_device_responds(ip_address):
        logger.debug(f"Device {ip_address} already responding")
        return True
    
    # Attempt to wake
    for attempt in range(max_attempts):
        logger.debug(f"Wake attempt {attempt+1}/{max_attempts} for {mac_address}")
        
        # Send magic packet
        wake_device(mac_address)
        
        # Wait for device to wake up
        for i in range(3):  # Check a few times over 3 seconds
            time.sleep(1)
            if check_device_responds(ip_address):
                logger.info(f"Successfully woke device {mac_address} ({ip_address})")
                return True
    
    logger.debug(f"Failed to wake device {mac_address} after {max_attempts} attempts")
    return False