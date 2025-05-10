# presence/device_manager.py
"""Device manager for presence detection."""
import json
import os
import logging
from datetime import datetime, timedelta, time
import threading
from .models import Device, DeviceType, ConfirmationStatus
from utils.network_scanner import guess_device_type, get_vendor_confidence_score
from utils.network_scanner import check_device_presence, ping_device
from utils.wol import wake_and_check

logger = logging.getLogger(__name__)

class DeviceManager:
    """Manages devices for presence detection."""
    
    def __init__(self, data_dir="data/presence", notification_callback=None):
        """
        Initialize the device manager.
        
        Args:
            data_dir: Directory to store device data
            notification_callback: Function to call when new devices are found
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.devices_file = os.path.join(data_dir, "devices.json")
        self.devices = {}
        self.notification_callback = notification_callback
        self._load_devices()
        
        # Lock for thread safety
        self._lock = threading.Lock()
        
        # System parameters
        self.phone_offline_threshold = 5  # How many scans before marking phone offline
        self.sleep_hours = (23, 7)  # Between 11 PM and 7 AM
        
        # Store last scan results for reference
        self._last_scan_results = []
    
    def _load_devices(self):
        """Load devices from file."""
        if os.path.exists(self.devices_file):
            try:
                with open(self.devices_file, 'r') as f:
                    data = json.load(f)
                    for device_data in data.get("devices", []):
                        device = Device.from_dict(device_data)
                        if device:
                            self.devices[device.mac] = device
                logger.info(f"Loaded {len(self.devices)} devices from {self.devices_file}")
            except Exception as e:
                logger.error(f"Error loading devices: {e}")
                
    def _save_devices(self):
        """Save devices to file."""
        try:
            with self._lock:
                data = {
                    "devices": [device.to_dict() for device in self.devices.values()]
                }
                with open(self.devices_file, 'w') as f:
                    json.dump(data, f, indent=2)
                return True
        except Exception as e:
            logger.error(f"Error saving devices: {e}")
            return False

    def _update_active_hours(self, device, current_time, is_online):
        """
        Update the typical active hours for a device.
        
        Args:
            device: Device object
            current_time: Current time
            is_online: Whether device is currently online
        """
        # Only track active hours for devices that are online
        if not is_online:
            return
            
        # Get current hour (0-23)
        hour = current_time.hour
        
        # Check if this hour is already in typical active hours
        hour_exists = False
        for hour_range in device.typical_active_hours:
            start_hour, end_hour = hour_range
            if start_hour <= hour <= end_hour:
                hour_exists = True
                break
        
        # If not, add it
        if not hour_exists:
            # Start with just this hour
            new_range = [hour, hour]
            
            # Check if we can extend an existing range
            merged = False
            for i, hour_range in enumerate(device.typical_active_hours):
                start_hour, end_hour = hour_range
                
                # If this hour is adjacent to an existing range, extend the range
                if start_hour - 1 == hour:
                    device.typical_active_hours[i][0] = hour
                    merged = True
                    break
                elif end_hour + 1 == hour:
                    device.typical_active_hours[i][1] = hour
                    merged = True
                    break
            
            # If not merged with an existing range, add as new range
            if not merged:
                device.typical_active_hours.append(new_range)
        
        logger.debug(f"Updated typical active hours for {device.name}: {device.typical_active_hours}")

    def update_device_status(self, mac, is_online, current_time=None):
        """
        Update device status based on network scan with enhanced detection.
        
        Args:
            mac: MAC address
            is_online: Whether device is online
            current_time: Current time (default: now)
        
        Returns:
            bool: Whether status changed
        """
        with self._lock:
            mac = mac.lower()
            if mac not in self.devices:
                return False
                
            device = self.devices[mac]
            status_changed = False
            now = current_time or datetime.now()
            
            if is_online:
                # Find IP address for this MAC from recent scan
                for device_info in self._last_scan_results:
                    if len(device_info) >= 2 and device_info[0].lower() == mac.lower():
                        device.last_ip = device_info[1]  # Store IP address
                        break
                
                # Device is online - mark active and reset offline counter
                if device.status != "active":
                    status_changed = True
                    logger.info(f"Device {mac} ({device.name}) is now active")
                
                device.record_connection()
                device.offline_count = 0
                device.status = "active"
                
            else:
                # Try checking ARP table for all devices
                if device.last_ip:
                    # Try ARP table check first
                    is_present = self.check_arp_table(mac)
                    
                    if is_present:
                        logger.info(f"Device {mac} ({device.name}) found in ARP table")
                        # Mark as online and update timestamp
                        device.record_connection()
                        device.offline_count = 0
                        
                        if device.status != "active":
                            device.status = "active"
                            status_changed = True
                        
                        # Return early - device is actually present
                        if status_changed:
                            self._save_devices()
                        return status_changed
                
                # Additional checks only for important devices (phones)
                if (device.device_type == DeviceType.PHONE.value and 
                    device.count_for_presence and device.last_ip):
                    
                    # Try additional methods to detect the device (ping)
                    if ping_device(device.last_ip):
                        logger.info(f"Device {mac} ({device.name}) responded to ping")
                        # Mark as online and update timestamp
                        device.record_connection()
                        device.offline_count = 0
                        
                        if device.status != "active":
                            device.status = "active"
                            status_changed = True
                        
                        # Return early - device is actually present
                        if status_changed:
                            self._save_devices()
                        return status_changed
            
                # Device is offline - increment counter
                device.offline_count += 1
                
                # Try to wake phones that support WoL
                if (device.device_type == DeviceType.PHONE.value and 
                    device.count_for_presence and 
                    device.supports_wol and
                    device.offline_count >= 2):  # Only try after missing 2 scans
                    
                    logger.debug(f"Attempting to wake device {mac} ({device.name})")
                    wake_success = self.try_wake_device(mac)
                    
                    if wake_success:
                        # If wake successful, mark as online and reset offline counter
                        device.status = "active"
                        device.offline_count = 0
                        device.record_connection()
                        logger.info(f"Successfully woken device {mac} ({device.name})")
                        status_changed = True
            
            # Check if we should mark device as inactive
            offline_threshold = self._get_offline_threshold(device, now)
            
            if device.offline_count > offline_threshold and device.status == "active":
                # Mark device as inactive
                device.status = "inactive"
                device.record_disconnection()
                status_changed = True
                logger.info(f"Device {mac} ({device.name}) is now inactive after {device.offline_count} missed scans")
        
        # Save on status change
        if status_changed:
            self._save_devices()
            
            # Update typical active hours on status change
            if device.device_type == DeviceType.PHONE.value:
                self._update_active_hours(device, now, is_online)
        
        return status_changed

    def check_arp_table(self, mac):
        """
        Check if a MAC address is present in the system's ARP table.
        
        Args:
            mac: MAC address to check
        
        Returns:
            bool: Whether the device is found in the ARP table
        """
        try:
            # Use the check_device_presence function but only with arp_table method
            is_present, method, _ = check_device_presence(
                mac, 
                None,  # IP not needed for ARP table check
                methods=['arp_table']
            )
            return is_present
        except Exception as e:
            logger.error(f"Error checking ARP table for {mac}: {e}")
            return False

    def _get_offline_threshold(self, device, current_time):
        """
        Get the threshold for marking a device offline based on type and time.
        
        Args:
            device: Device object
            current_time: Current time
        
        Returns:
            int: Number of missed scans before marking inactive
        """
        # Only special handling for phones
        if device.device_type != DeviceType.PHONE.value:
            return 3  # Default threshold for non-phones
        
        # Check if current time is within sleep hours
        hour = current_time.hour
        sleep_start, sleep_end = self.sleep_hours
        
        is_sleep_time = False
        if sleep_start > sleep_end:  # Handles case where sleep hours cross midnight
            is_sleep_time = hour >= sleep_start or hour < sleep_end
        else:
            is_sleep_time = sleep_start <= hour < sleep_end
        
        # Higher threshold during sleep hours
        if is_sleep_time:
            return 10  # More tolerant during sleep hours (increased from 8)
        
        # Check if device is typically active at this hour
        is_typically_active = False
        for hour_range in device.typical_active_hours:
            start_hour, end_hour = hour_range
            if start_hour <= hour <= end_hour:
                is_typically_active = True
                break
        
        # More tolerant if device is typically active now
        if is_typically_active:
            return 6  # Increased from 4
            
        # Default threshold for phones during non-sleep hours
        return self.phone_offline_threshold
    
    def is_probably_present(self, device, current_time=None):
        """
        Determine if a device is probably present even if currently inactive.
        
        Args:
            device: Device object
            current_time: Current time (default: now)
        
        Returns:
            bool: Whether device is probably present
        """
        now = current_time or datetime.now()
        
        # Case 1: Device is currently active
        if device.status == "active":
            return True
            
        # Case 2: Device has no history
        if not device.last_seen:
            return False
        
        # Case 3: Special case for phones - more lenient with them
        if device.device_type == DeviceType.PHONE.value:
            try:
                last_seen = datetime.fromisoformat(device.last_seen)
                time_since_last_seen = (now - last_seen).total_seconds() / 60  # minutes
                
                # Check if current time is within typical active hours
                hour = now.hour
                is_typically_active = False
                for hour_range in device.typical_active_hours:
                    start_hour, end_hour = hour_range
                    if start_hour <= hour <= end_hour:
                        is_typically_active = True
                        break
                
                # More lenient threshold during active hours
                threshold = 30 if is_typically_active else 15
                
                # If we're within the time window, consider probably present
                if time_since_last_seen < threshold:
                    return True
                    
                # Additional method - check if there are multiple recent connections
                cutoff = now - timedelta(minutes=60)  # Past hour
                recent_connections = [
                    event for event in device.connection_history
                    if event.get('type') == 'connect' and
                    datetime.fromisoformat(event.get('timestamp', '')) > cutoff
                ]
                
                # If there are 3+ connections in the past hour, likely still present
                if len(recent_connections) >= 3:
                    return True
                    
            except (ValueError, TypeError):
                pass
                
        # For non-phones, use a simple time window
        else:
            try:
                last_seen = datetime.fromisoformat(device.last_seen)
                if (now - last_seen) < timedelta(minutes=10):
                    return True
            except (ValueError, TypeError):
                pass
        
        return False

    def add_device(self, mac, name=None, owner=None, device_type="unknown", vendor=None, 
                  count_for_presence=False, confirmation_status="unconfirmed"):
        """
        Add a new device to the device manager.
        
        Args:
            mac: MAC address of device
            name: Name of device
            owner: Owner of device
            device_type: Type of device (phone, laptop, etc.)
            vendor: Device manufacturer
            count_for_presence: Whether to count this device for presence detection
            confirmation_status: Whether device has been confirmed by a user
        
        Returns:
            bool: Success status
        """
        with self._lock:
            mac = mac.lower()
            
            # Check if device already exists
            if mac in self.devices:
                logger.warning(f"Device {mac} already exists, updating instead of adding")
                # Update existing device with new information
                device = self.devices[mac]
                if name:
                    device.name = name
                if owner:
                    device.owner = owner
                if device_type:
                    device.device_type = device_type
                if vendor:
                    device.vendor = vendor
                device.count_for_presence = count_for_presence
                device.confirmation_status = confirmation_status
            else:
                # Create new device
                device = Device(
                    mac=mac,
                    name=name,
                    owner=owner,
                    device_type=device_type,
                    vendor=vendor,
                    count_for_presence=count_for_presence,
                    confirmation_status=confirmation_status
                )
                self.devices[mac] = device
                logger.info(f"Added new device: {device.name} ({device.mac})")
        
        # Save changes to file
        self._save_devices()
        
        # Notify if callback is registered
        if self.notification_callback and mac not in self.devices:
            self.notification_callback("new_device", 
                                    device_name=device.name,
                                    device_mac=mac,
                                    device_type=device_type,
                                    vendor=vendor,
                                    confidence=device.confidence_score)
        
        return True

    def set_notification_callback(self, callback):
        """Set or update the notification callback function."""
        self.notification_callback = callback

    def calculate_people_present(self):
        """
        Calculate number of people present with enhanced reliability.
        
        Returns:
            int: Estimated number of people present
        """
        with self._lock:
            people_count = 0
            counted_owners = set()
            current_time = datetime.now()
            
            # First count phones with high confidence
            for device in self.devices.values():
                if ((device.status == "active" or self.is_probably_present(device, current_time)) and 
                    device.count_for_presence and 
                    device.device_type == DeviceType.PHONE.value and
                    device.owner and 
                    device.owner not in counted_owners):
                    
                    people_count += 1
                    counted_owners.add(device.owner)
                    logger.debug(f"Counting {device.owner} as present (phone: {device.name})")
            
            # For reliability, try extra detection for phones marked inactive
            inactive_phones = [
                d for d in self.devices.values()
                if d.device_type == DeviceType.PHONE.value and
                d.count_for_presence and
                d.owner and
                d.owner not in counted_owners and 
                d.status != "active" and 
                d.last_ip
            ]
            
            for device in inactive_phones:
                # Try direct ping as last resort
                if ping_device(device.last_ip):
                    people_count += 1
                    counted_owners.add(device.owner)
                    logger.info(f"Adding {device.owner} via direct ping to {device.last_ip}")
            
            # Count unknown phones (without owner) as separate people
            for device in self.devices.values():
                if ((device.status == "active" or self.is_probably_present(device, current_time)) and 
                    device.count_for_presence and 
                    device.device_type == DeviceType.PHONE.value and
                    not device.owner):
                    people_count += 1
                    logger.debug(f"Counting unknown phone as present (phone: {device.name})")
            
            logger.info(f"Calculated presence: {people_count} people present")
            return people_count