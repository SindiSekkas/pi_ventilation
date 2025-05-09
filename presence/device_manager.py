"""Device manager for presence detection."""
import json
import os
import logging
from datetime import datetime, timedelta, time
import threading
from .models import Device, DeviceType, ConfirmationStatus
from utils.network_scanner import guess_device_type, get_vendor_confidence_score
from ..utils.wol import wake_and_check

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
            
    def add_device(self, mac, name=None, owner=None, device_type=None, vendor=None, 
                  count_for_presence=False, confirmation_status=ConfirmationStatus.UNCONFIRMED.value):
        """
        Add a new device.
        
        Args:
            mac: MAC address
            name: Device name
            owner: Owner name
            device_type: Device type
            vendor: Device vendor
            count_for_presence: Whether to count for presence
            confirmation_status: Whether confirmed by user
        
        Returns:
            bool: Success status
        """
        with self._lock:
            mac = mac.lower()
            if mac in self.devices:
                return False
                
            # If device_type not provided, try to guess from vendor
            if device_type is None or device_type == 'unknown':
                device_type = guess_device_type(mac, vendor or "")
            
            # Default name if none provided
            if not name:
                if vendor and vendor != "Unknown":
                    name = f"{vendor}-{mac[-5:]}"
                else:
                    name = f"Device-{mac[-5:]}"
            
            # Determine if this device should count for presence
            # By default, confirmed phones count for presence
            auto_count = (device_type == DeviceType.PHONE.value and 
                         confirmation_status == ConfirmationStatus.CONFIRMED.value)
            
            count_for_presence = count_for_presence or auto_count
            
            # Create device
            self.devices[mac] = Device(
                mac=mac,
                name=name,
                owner=owner,
                device_type=device_type,
                vendor=vendor,
                count_for_presence=count_for_presence,
                confirmation_status=confirmation_status
            )
            
            # Set confidence score based on vendor
            if vendor:
                self.devices[mac].confidence_score = get_vendor_confidence_score(vendor)
            
            # Save devices
            self._save_devices()
            
            # Notify about new device if it's probably a phone
            if (self.notification_callback and 
                device_type == DeviceType.PHONE.value and
                confirmation_status == ConfirmationStatus.UNCONFIRMED.value):
                
                try:
                    self.notification_callback(
                        action="new_device", 
                        device_mac=mac,
                        device_name=name,
                        device_type=device_type,
                        vendor=vendor,
                        confidence=self.devices[mac].confidence_score
                    )
                except Exception as e:
                    logger.error(f"Error in notification callback: {e}")
            
            logger.info(f"Added new device: {mac}, {name}, {device_type}, {vendor}")
            return True
        
    def update_device(self, mac, **kwargs):
        """
        Update device properties.
        
        Args:
            mac: MAC address
            **kwargs: Properties to update
        
        Returns:
            bool: Success status
        """
        with self._lock:
            mac = mac.lower()
            if mac not in self.devices:
                logger.error(f"Cannot update device {mac}: not found")
                return False
            
            device = self.devices[mac]
            
            # Update properties
            if "name" in kwargs:
                device.name = kwargs["name"]
            if "owner" in kwargs:
                device.owner = kwargs["owner"]
            if "device_type" in kwargs:
                device.device_type = kwargs["device_type"]
            if "vendor" in kwargs:
                device.vendor = kwargs["vendor"]
            if "count_for_presence" in kwargs:
                device.count_for_presence = kwargs["count_for_presence"]
            if "confirmation_status" in kwargs:
                device.confirmation_status = kwargs["confirmation_status"]
                
                # If confirmed and it's a phone, automatically set count_for_presence
                if (kwargs["confirmation_status"] == ConfirmationStatus.CONFIRMED.value and
                    device.device_type == DeviceType.PHONE.value):
                    device.count_for_presence = True
            
            self._save_devices()
            logger.info(f"Updated device: {mac}")
            return True
        
    def remove_device(self, mac):
        """
        Remove a device.
        
        Args:
            mac: MAC address
        
        Returns:
            bool: Success status
        """
        with self._lock:
            mac = mac.lower()
            if mac not in self.devices:
                return False
            
            del self.devices[mac]
            self._save_devices()
            logger.info(f"Removed device: {mac}")
            return True
        
    def try_wake_device(self, mac):
        """
        Attempt to wake a device using Wake-on-LAN.
        
        Args:
            mac: MAC address of device to wake
            
        Returns:
            bool: Success status
        """
        with self._lock:
            mac = mac.lower()
            if mac not in self.devices:
                logger.error(f"Cannot wake device {mac}: not found")
                return False
                
            device = self.devices[mac]
            
            # Skip if device doesn't have an IP address
            if not device.last_ip:
                logger.debug(f"Cannot wake device {mac}: no known IP address")
                return False
                
            # Attempt to wake the device
            success = wake_and_check(mac, device.last_ip)
            
            # Update device statistics
            if success:
                device.wol_success_count += 1
                if device.wol_success_count > 2:
                    device.supports_wol = True
            else:
                device.wol_failure_count += 1
                if device.wol_failure_count > 5 and device.wol_success_count == 0:
                    device.supports_wol = False
            
            # Save changes
            self._save_devices()
            
            return success

    def update_device_status(self, mac, is_online, current_time=None):
        """
        Update device status based on network scan.
        
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
                for device_info in getattr(self, '_last_scan_results', []):
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
            return 8  # More tolerant during sleep hours
        
        # Check if device is typically active at this hour
        is_typically_active = False
        for hour_range in device.typical_active_hours:
            start_hour, end_hour = hour_range
            if start_hour <= hour <= end_hour:
                is_typically_active = True
                break
        
        # More tolerant if device is typically active now
        if is_typically_active:
            return 4
            
        # Default threshold for phones during non-sleep hours
        return self.phone_offline_threshold
    
    def _update_active_hours(self, device, current_time, is_active):
        """
        Update the typical active hours for a device.
        
        Args:
            device: Device object
            current_time: Current timestamp
            is_active: Whether device is active
        """
        hour = current_time.hour
        
        # Initialize typical active hours if empty
        if not device.typical_active_hours:
            device.typical_active_hours = []
        
        # If becoming active, record this hour
        if is_active:
            # Check if this hour is already in an existing range
            in_existing_range = False
            for i, (start_hour, end_hour) in enumerate(device.typical_active_hours):
                # If hour is at the start or end of a range, expand the range
                if hour == start_hour - 1:
                    device.typical_active_hours[i] = (hour, end_hour)
                    in_existing_range = True
                    break
                elif hour == end_hour + 1:
                    device.typical_active_hours[i] = (start_hour, hour)
                    in_existing_range = True
                    break
                # If hour is within the range, it's already recorded
                elif start_hour <= hour <= end_hour:
                    in_existing_range = True
                    break
            
            # If not in an existing range, add a new range
            if not in_existing_range:
                device.typical_active_hours.append((hour, hour))
                
            # Merge overlapping ranges
            self._merge_active_hour_ranges(device)
    
    def _merge_active_hour_ranges(self, device):
        """
        Merge overlapping active hour ranges.
        
        Args:
            device: Device object
        """
        # Sort ranges
        device.typical_active_hours.sort()
        
        # No ranges or only one range
        if len(device.typical_active_hours) <= 1:
            return
            
        # Merge overlapping ranges
        i = 0
        while i < len(device.typical_active_hours) - 1:
            current_start, current_end = device.typical_active_hours[i]
            next_start, next_end = device.typical_active_hours[i + 1]
            
            # If ranges overlap or are adjacent, merge them
            if current_end >= next_start - 1:
                merged_range = (current_start, max(current_end, next_end))
                device.typical_active_hours[i] = merged_range
                device.typical_active_hours.pop(i + 1)
            else:
                i += 1
        
    def calculate_people_present(self):
        """
        Calculate number of people present based on active devices.
        
        Returns:
            int: Estimated number of people present
        """
        with self._lock:
            people_count = 0
            counted_owners = set()
            current_time = datetime.now()
            
            # First count phones (highest priority)
            for device in self.devices.values():
                if ((device.status == "active" or device.is_probably_present(current_time)) and 
                    device.count_for_presence and 
                    device.device_type == DeviceType.PHONE.value and
                    device.owner and 
                    device.owner not in counted_owners):
                    people_count += 1
                    counted_owners.add(device.owner)
                    logger.debug(f"Counting {device.owner} as present (phone: {device.name})")
            
            # Count unknown phones (without owner) as separate people
            for device in self.devices.values():
                if ((device.status == "active" or device.is_probably_present(current_time)) and 
                    device.count_for_presence and 
                    device.device_type == DeviceType.PHONE.value and
                    not device.owner):
                    people_count += 1
                    logger.debug(f"Counting unknown phone as present (phone: {device.name})")
            
            logger.info(f"Calculated presence: {people_count} people present")
            return people_count
    
    def set_notification_callback(self, callback):
        """
        Set callback for device notifications.
        
        Args:
            callback: Function to call when new devices are found
        """
        self.notification_callback = callback
        
    def get_new_devices(self, only_phones=True, only_unconfirmed=True, min_confidence=0.6):
        """
        Get list of new devices that might require user confirmation.
        
        Args:
            only_phones: Whether to only include likely phones
            only_unconfirmed: Whether to only include unconfirmed devices
            min_confidence: Minimum confidence score to include
            
        Returns:
            list: List of device objects
        """
        with self._lock:
            new_devices = []
            
            for device in self.devices.values():
                # Filter according to parameters
                if (
                    (not only_phones or device.device_type == DeviceType.PHONE.value) and
                    (not only_unconfirmed or device.confirmation_status == ConfirmationStatus.UNCONFIRMED.value) and
                    device.confidence_score >= min_confidence
                ):
                    new_devices.append(device)
            
            return new_devices
    
    def confirm_device(self, mac, is_phone=True, owner=None, count_for_presence=True):
        """
        Confirm a device as a phone or other known device.
        
        Args:
            mac: MAC address
            is_phone: Whether the device is a phone
            owner: Owner name (optional)
            count_for_presence: Whether to count for presence
            
        Returns:
            bool: Success status
        """
        mac = mac.lower()
        if mac not in self.devices:
            logger.error(f"Cannot confirm device {mac}: not found")
            return False
        
        device = self.devices[mac]
        
        # Update device
        device.confirmation_status = ConfirmationStatus.CONFIRMED.value
        
        if is_phone:
            device.device_type = DeviceType.PHONE.value
        
        if owner:
            device.owner = owner
            
        device.count_for_presence = count_for_presence
        
        # Save changes
        self._save_devices()
        logger.info(f"Confirmed device {mac} as {'phone' if is_phone else 'other device'}, owner: {owner}")
        return True
    
    def ignore_device(self, mac):
        """
        Mark a device as ignored.
        
        Args:
            mac: MAC address
            
        Returns:
            bool: Success status
        """
        mac = mac.lower()
        if mac not in self.devices:
            logger.error(f"Cannot ignore device {mac}: not found")
            return False
        
        device = self.devices[mac]
        device.confirmation_status = ConfirmationStatus.IGNORED.value
        device.count_for_presence = False
        
        # Save changes
        self._save_devices()
        logger.info(f"Marked device {mac} as ignored")
        return True