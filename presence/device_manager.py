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
    """Manages network device discovery, tracking, and occupancy inference."""
    
    def __init__(self, data_dir="data/presence", notification_callback=None, telegram_ping_queue=None):
        """
        Initialize the device tracking system.
        
        Args:
            data_dir: Storage location for device persistence
            notification_callback: Hook for device discovery notifications
            telegram_ping_queue: Queue for interactive device verification
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.devices_file = os.path.join(data_dir, "devices.json")
        self.devices = {}
        self.notification_callback = notification_callback
        self.telegram_ping_queue = telegram_ping_queue
        self._load_devices()
        
        # Lock for thread safety
        self._lock = threading.Lock()
        
        # System parameters
        self.phone_offline_threshold = 5  # How many scans before marking phone offline
        self.sleep_hours = (23, 7)  # Between 11 PM and 7 AM
        self.TELEGRAM_PING_COOLDOWN_MINUTES = 10  # Minimum interval between Telegram pings
        
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
        Optimized to minimize lock time.
        """
        mac_lower = mac.lower()
        now = current_time or datetime.now()
        
        # First, get device reference with minimal lock time
        with self._lock:
            if mac_lower not in self.devices:
                return False
            device = self.devices[mac_lower]
        
        # Now work with device outside the lock
        status_changed = False
        
        if is_online:
            # Find IP address for this MAC from recent scan
            for device_info in self._last_scan_results:
                if len(device_info) >= 2 and device_info[0].lower() == mac_lower:
                    device.last_ip = device_info[1]  # Store IP address
                    break
            
            # Device is online - mark active and reset offline counter
            if device.status != "active":
                status_changed = True
                logger.info(f"Device {mac} ({device.name}) is now active")
            
            # These operations don't need locks
            device.record_connection()
            device.offline_count = 0
            device.status = "active"
            
        else:
            # For phones with Telegram-ping capability, prioritize Telegram-ping
            if (device.device_type == DeviceType.PHONE.value and 
                device.count_for_presence and 
                device.telegram_user_id is not None):
                
                # Check if cooldown has passed
                if device.last_telegram_ping_request_time is not None:
                    last_ping_time = datetime.fromisoformat(device.last_telegram_ping_request_time)
                    time_since_ping = (now - last_ping_time).total_seconds() / 60  # minutes
                    
                    if time_since_ping < self.TELEGRAM_PING_COOLDOWN_MINUTES:
                        logger.debug(f"Telegram ping cooldown not expired for {device.name} ({time_since_ping:.1f} < {self.TELEGRAM_PING_COOLDOWN_MINUTES} minutes)")
                    else:
                        # Queue Telegram ping and return without further checks
                        if self.telegram_ping_queue is not None:
                            ping_task = {
                                'mac': device.mac,
                                'telegram_user_id': device.telegram_user_id,
                                'ip_address': device.last_ip or None
                            }
                            try:
                                self.telegram_ping_queue.put(ping_task)
                                device.last_telegram_ping_request_time = now.isoformat()
                                device.is_pending_telegram_ping = True
                                # Save devices in background
                                threading.Thread(target=self._save_devices, daemon=True).start()
                                logger.info(f"Queued Telegram ping for {device.name} (MAC: {device.mac})")
                                # Return immediately to wait for ping result
                                return False
                            except Exception as e:
                                logger.error(f"Failed to queue Telegram ping: {e}")
                else:
                    # First time pinging - queue and return
                    if self.telegram_ping_queue is not None:
                        ping_task = {
                            'mac': device.mac,
                            'telegram_user_id': device.telegram_user_id,
                            'ip_address': device.last_ip or None
                        }
                        try:
                            self.telegram_ping_queue.put(ping_task)
                            device.last_telegram_ping_request_time = now.isoformat()
                            device.is_pending_telegram_ping = True
                            # Save devices in background
                            threading.Thread(target=self._save_devices, daemon=True).start()
                            logger.info(f"Queued first Telegram ping for {device.name} (MAC: {device.mac})")
                            # Return immediately to wait for ping result
                            return False
                        except Exception as e:
                            logger.error(f"Failed to queue Telegram ping: {e}")
            
            # Device is offline - increment counter
            device.offline_count += 1
        
        # Check if we should mark device as inactive
        offline_threshold = self._get_offline_threshold(device, now)
        
        if device.offline_count > offline_threshold and device.status == "active":
            # Mark device as inactive
            device.status = "inactive"
            device.record_disconnection()
            status_changed = True
            logger.info(f"Device {mac} ({device.name}) is now inactive after {device.offline_count} missed scans")
        
        # Save on status change in background
        if status_changed:
            threading.Thread(target=self._save_devices, daemon=True).start()
            
            # Update typical active hours on status change
            if device.device_type == DeviceType.PHONE.value:
                self._update_active_hours(device, now, is_online)
        
        return status_changed

    def process_telegram_ping_result(self, mac: str, detected_after_ping: bool):
        """
        Handle the outcome of an interactive device verification.
        
        When network scans are inconclusive, the system can request
        device owners to confirm their presence via Telegram.
        
        Args:
            mac: Device identifier
            detected_after_ping: Whether verification succeeded
            
        Returns:
            bool: Whether device status changed
        """
        with self._lock:
            mac = mac.lower()
            if mac not in self.devices:
                logger.warning(f"Device {mac} not found for Telegram ping result")
                return False
            
            device = self.devices[mac]
            device.is_pending_telegram_ping = False
            
            if detected_after_ping:
                # Device was found after ping - mark as active
                device.status = "active"
                device.offline_count = 0
                device.record_connection()
                logger.info(f"Device {device.name} detected after Telegram ping")
                self._save_devices()
                return True
            else:
                # Device was not found - increment offline count
                device.offline_count += 1
                logger.debug(f"Device {device.name} not detected after Telegram ping")
                
                # Check if we should mark as inactive
                now = datetime.now()
                offline_threshold = self._get_offline_threshold(device, now)
                
                if device.offline_count > offline_threshold and device.status == "active":
                    device.status = "inactive"
                    device.record_disconnection()
                    logger.info(f"Device {device.name} marked inactive after failed Telegram ping")
                    self._save_devices()
                    return True
        
        self._save_devices()
        return False

    def check_arp_table(self, mac):
        """
        Check if device appears in system ARP table as fallback detection.
        
        Args:
            mac: Device identifier
            
        Returns:
            bool: Whether device is in ARP table
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
        Determine how many missed scans to tolerate before marking device offline.
        
        Applies context-aware thresholds based on:
        - Device type (phones get special treatment)
        - Time of day (higher thresholds during sleep hours)
        - Usage patterns (higher thresholds during typical active hours)
        
        Args:
            device: Target device
            current_time: Reference timestamp
            
        Returns:
            int: Number of missed scans to tolerate
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
        Infer likely presence despite device being offline.
        
        Uses multiple heuristics:
        - For phones: Considers typical usage hours and recent connection history
        - For other devices: Uses simpler time-based threshold
        
        Args:
            device: Target device
            current_time: Reference timestamp
            
        Returns:
            bool: Likelihood of actual presence
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
        Register a new network device or update an existing one.
        
        Args:
            mac: Unique device identifier
            name: Human-readable device name
            owner: Person associated with the device
            device_type: Classification for presence inference
            vendor: Manufacturer information
            count_for_presence: Whether to include in occupancy calculation
            confirmation_status: User verification state
            
        Returns:
            bool: Success indicator
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

    def link_device_to_telegram_user(self, mac: str, telegram_user_id: int) -> bool:
        """
        Associate device with Telegram user for interactive verification.
        
        Args:
            mac: Device identifier
            telegram_user_id: Telegram user identifier
            
        Returns:
            bool: Success indicator
        """
        with self._lock:
            mac = mac.lower()
            if mac not in self.devices:
                return False
            
            self.devices[mac].telegram_user_id = telegram_user_id
            if not self.devices[mac].owner:
                self.devices[mac].owner = f"TelegramUser_{telegram_user_id}"
            
        # Save changes
        self._save_devices()
        logger.info(f"Linked device {mac} to Telegram user {telegram_user_id}")
        return True

    def unlink_device_from_telegram_user(self, mac: str) -> bool:
        """
        Remove Telegram association from device.
        
        Args:
            mac: Device identifier
            
        Returns:
            bool: Success indicator
        """
        with self._lock:
            mac = mac.lower()
            if mac not in self.devices:
                return False
            
            self.devices[mac].telegram_user_id = None
            self.devices[mac].last_telegram_ping_request_time = None
            self.devices[mac].is_pending_telegram_ping = False
            
        # Save changes
        self._save_devices()
        logger.info(f"Unlinked device {mac} from Telegram user")
        return True

    def set_notification_callback(self, callback):
        """Register function to be called when significant device events occur."""
        self.notification_callback = callback

    def calculate_people_present(self):
        """
        Estimate the number of people currently present in the monitored space.
        
        Uses a multi-stage algorithm:
        1. Count active/probable phones with owners (1 person per owner)
        2. Verify inactive phones with ping for improved reliability
        3. Count active/probable phones without owners (1 person per device)
        
        Returns:
            int: Estimated occupancy count
        """
        with self._lock:
            people_count = 0
            counted_owners = set()
            current_time = datetime.now()
            logger.info(f"Starting presence calculation. Current time: {current_time.isoformat()}")
            logger.debug(f"Initial people_count: {people_count}, counted_owners: {counted_owners}")

            # First count phones with high confidence
            logger.debug("Processing devices for primary count (active or probably present phones with owners):")
            for device in self.devices.values():
                logger.debug(f"Checking device: {device.name} (MAC: {device.mac}, Type: {device.device_type}, Status: {device.status}, Owner: '{device.owner}', Counts: {device.count_for_presence})")
                is_active = device.status == "active"
                is_probable = self.is_probably_present(device, current_time)
                logger.debug(f"Device {device.name}: is_active={is_active}, is_probable={is_probable}")
                if device.device_type == DeviceType.PHONE.value and device.count_for_presence:
                    logger.info(f"DEBUG PHONE {device.name}: status={device.status}, offline_count={device.offline_count}, last_seen={device.last_seen}")

                if ((is_active or is_probable) and
                    device.count_for_presence and 
                    device.device_type == DeviceType.PHONE.value and
                    device.owner and 
                    device.owner not in counted_owners):
                    
                    people_count += 1
                    counted_owners.add(device.owner)
                    logger.info(f"Incremented people_count to {people_count}. Added owner '{device.owner}' from device '{device.name}'. Reason: Active/Probable phone with owner.")
                elif not device.count_for_presence:
                    logger.debug(f"Skipping {device.name}: count_for_presence is False.")
                elif device.device_type != DeviceType.PHONE.value:
                    logger.debug(f"Skipping {device.name}: not a PHONE.")
                elif not device.owner:
                    logger.debug(f"Skipping {device.name} in this step: no owner assigned (will be checked later).")
                elif device.owner in counted_owners:
                    logger.debug(f"Skipping {device.name}: owner '{device.owner}' already counted.")
            
            logger.debug(f"After primary count: people_count={people_count}, counted_owners={counted_owners}")

            # For reliability, try extra detection for phones marked inactive
            logger.debug("Processing inactive phones with owners for potential ping check:")
            inactive_phones = [
                d for d in self.devices.values()
                if d.device_type == DeviceType.PHONE.value and
                d.count_for_presence and
                d.owner and
                d.owner not in counted_owners and 
                d.status != "active" and 
                d.last_ip
            ]
            logger.debug(f"Found {len(inactive_phones)} inactive phones with owners to check via ping: {[p.name for p in inactive_phones]}")
            
            for device in inactive_phones:
                logger.debug(f"Attempting ping for inactive phone: {device.name} (Owner: '{device.owner}', IP: {device.last_ip})")
                if ping_device(device.last_ip):
                    people_count += 1
                    counted_owners.add(device.owner)
                    logger.info(f"Incremented people_count to {people_count}. Added owner '{device.owner}' from device '{device.name}'. Reason: Inactive phone responded to ping.")
                else:
                    logger.debug(f"Ping failed for {device.name}.")
            
            logger.debug(f"After ping check for inactive owned phones: people_count={people_count}, counted_owners={counted_owners}")

            # Count unknown phones (without owner) as separate people
            logger.debug("Processing active or probably present phones without owners:")
            for device in self.devices.values():
                is_active = device.status == "active"
                is_probable = self.is_probably_present(device, current_time)
                # Log details for all phones without owners, regardless of active/probable status, for better debugging
                if device.device_type == DeviceType.PHONE.value and not device.owner and device.count_for_presence:
                    logger.debug(f"Checking unowned phone: {device.name} (MAC: {device.mac}, Status: {device.status}, Counts: {device.count_for_presence}, IsActive: {is_active}, IsProbable: {is_probable})")

                if ((is_active or is_probable) and 
                    device.count_for_presence and 
                    device.device_type == DeviceType.PHONE.value and
                    not device.owner): # Ensure no owner is present for this condition
                    people_count += 1
                    # We don't add to counted_owners here as there's no owner.
                    # Each unowned, active/probable phone increments the count.
                    logger.info(f"Incremented people_count to {people_count}. Added unowned phone '{device.name}'. Reason: Active/Probable phone without owner.")
                elif device.device_type == DeviceType.PHONE.value and not device.owner and device.count_for_presence:
                    if not (is_active or is_probable):
                         logger.debug(f"Skipping unowned phone {device.name}: Not active or probably present.")
                    elif not device.count_for_presence:
                         logger.debug(f"Skipping unowned phone {device.name}: count_for_presence is False.")


            logger.info(f"Final calculated presence: {people_count} people present")
            return people_count