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
            device_manager: Device manager instance
            data_manager: Data manager instance
            occupancy_history_manager: OccupancyHistoryManager instance for logging changes
            scan_interval: Interval between scans in seconds
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
        self.thread = threading.Thread(target=self._presence_loop, daemon=True)
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
                # Scan network
                online_devices = scan_network()
                
                # Process discovered devices
                self._process_discovered_devices(online_devices)
                
                # Update status of all devices
                online_macs = [device[0] for device in online_devices]
                for mac in self.device_manager.devices:
                    is_online = mac in online_macs
                    self.device_manager.update_device_status(mac, is_online)
                    device = self.device_manager.devices[mac]
                    logger.info(f"Device {device.name} ({mac}): status={device.status}, offline_count={device.offline_count}")
                
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
                
            except Exception as e:
                logger.error(f"Error in presence detection: {e}")
                
            # Sleep until next scan
            time.sleep(self.scan_interval)
            
    def _process_discovered_devices(self, devices):
        """
        Process newly discovered devices.
        
        Args:
            devices: List of tuples (mac, ip, vendor) from network scan
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
        Handle device notifications from device manager.
        
        Args:
            action: Type of notification
            **kwargs: Details about the notification
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