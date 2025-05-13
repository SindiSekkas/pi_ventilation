# presence/presence_controller.py
"""Controller for presence detection system."""
import threading
import time
import logging
from datetime import datetime, timedelta
from utils.network_scanner import scan_network

logger = logging.getLogger(__name__)

class PresenceController:
    """Controls the presence detection system."""
    
    def __init__(self, device_manager, data_manager, occupancy_history_manager=None, scan_interval=300):
        """
        Initialize the presence controller.
        
        Args:
            device_manager: Registry of tracked network devices
            data_manager: System data storage interface
            occupancy_history_manager: Optional component for recording occupancy changes
            scan_interval: Seconds between network scans
        """
        self.device_manager = device_manager
        self.data_manager = data_manager
        self.occupancy_history_manager = occupancy_history_manager
        self.scan_interval = scan_interval
        self.running = False
        self.thread = None
        self.last_occupancy = 0
        self.last_occupancy_status = "EMPTY"
        
        # Register notification callback
        self.device_manager.set_notification_callback(self.handle_device_notification)
    
    def start(self):
        """Start presence detection in a separate thread."""
        if self.thread is not None and self.thread.is_alive():
            logger.warning("Presence detection already running")
            return False
            
        self.running = True
        self.thread = threading.Thread(target=self._presence_loop, daemon=True, name="PresenceController")
        self.thread.start()
        logger.info("Started presence detection")
        return True
        
    def stop(self):
        """Stop presence detection."""
        self.running = False
        logger.info("Stopped presence detection")
        
    def _presence_loop(self):
        """Main loop for presence detection."""
        while self.running:
            try:
                # Run scan in current thread - don't block other parts of the system
                logger.debug("Starting network scan...")
                online_devices = scan_network()
                logger.debug(f"Network scan completed, found {len(online_devices)} devices")
                
                # Process discovered devices - don't log every device to reduce spam
                self._process_discovered_devices(online_devices)
                
                # Update status of all devices - batch operations
                online_macs = [device[0].lower() for device in online_devices]
                devices_copy = dict(self.device_manager.devices)  # Create copy to avoid lock conflicts
                
                for mac, device in devices_copy.items():
                    is_online = mac in online_macs
                    self.device_manager.update_device_status(mac, is_online)
                
                # Calculate presence and update room data
                people_count = self.device_manager.calculate_people_present()
                
                # Only update if count changed
                if people_count != self.last_occupancy:
                    self.data_manager.update_room_data(occupants=people_count)
                    
                    # Record occupancy change if history manager is available
                    if self.occupancy_history_manager:
                        new_status = "EMPTY" if people_count == 0 else "OCCUPIED"
                        old_status = "EMPTY" if self.last_occupancy == 0 else "OCCUPIED"
                        
                        # Only record if status actually changed
                        if new_status != old_status:
                            self.occupancy_history_manager.record_occupancy_change(
                                new_status, 
                                people_count
                            )
                            logger.info(f"Occupancy status changed: {old_status} -> {new_status} ({people_count} people)")
                    
                    self.last_occupancy = people_count
                    logger.info(f"Updated occupancy: {people_count} people present")
                else:
                    logger.debug(f"Occupancy unchanged: {people_count} people")
                
            except Exception as e:
                logger.error(f"Error in presence detection: {e}")
                
            # Sleep until next scan - use smaller sleeps to be more responsive to stop
            elapsed = 0
            while elapsed < self.scan_interval and self.running:
                time.sleep(min(1, self.scan_interval - elapsed))
                elapsed += 1
            
    def _process_discovered_devices(self, devices):
        """
        Register and classify newly discovered network devices.
        
        Automatically categorizes devices and configures appropriate
        presence detection settings based on device characteristics.
        
        Args:
            devices: Device information from network scan [(mac, ip, vendor),...]
        """
        for device_info in devices:
            if len(device_info) == 3:
                mac, ip, vendor = device_info
            else:
                mac, ip = device_info
                vendor = "Unknown"
                
            if mac not in self.device_manager.devices:
                # New device discovered
                logger.info(f"New device discovered: {mac} ({ip}) - {vendor}")
                
                # Determine device type
                device_type = "unknown"
                if vendor:
                    from utils.network_scanner import guess_device_type
                    device_type = guess_device_type(mac, vendor)
                
                # Use vendor name for device name if available
                name = vendor if vendor != "Unknown" else f"New-{mac[-5:]}"
                
                # Count all phones for presence automatically
                count_for_presence = (device_type == "phone")
                
                self.device_manager.add_device(
                    mac=mac,
                    name=name,
                    vendor=vendor,
                    device_type=device_type,
                    count_for_presence=count_for_presence 
                )
    
    def handle_device_notification(self, action, **kwargs):
        """
        Process device-related events from the device manager.
        
        Enables integration with notification systems and
        provides special handling for high-confidence device detections.
        
        Args:
            action: Event type identifier
            **kwargs: Event-specific parameters
        """
        if action == "new_device":
            # Here you can add callbacks for various notification systems
            # like Telegram bot, web UI alerts, etc.
            logger.info(f"New device notification: {kwargs.get('device_name', 'Unknown device')}")
            
            # Log high confidence phones for easier review
            if (kwargs.get('device_type') == 'phone' and 
                kwargs.get('confidence', 0) > 0.7):
                logger.info(f"High confidence phone detected: {kwargs.get('device_name')} "
                           f"(MAC: {kwargs.get('device_mac')}, Vendor: {kwargs.get('vendor')})")