# presence/models.py
"""Network device presence detection models for the ventilation system."""
from enum import Enum
from datetime import datetime, time

class DeviceType(Enum):
    """Classification of network-connected devices."""
    PHONE = "phone"
    LAPTOP = "laptop"
    TABLET = "tablet"
    TV = "tv"
    IOT_DEVICE = "iot_device"
    UNKNOWN = "unknown"

class ConfirmationStatus(Enum):
    """Status indicating whether a device has been verified by a user."""
    UNCONFIRMED = "unconfirmed"  # Newly discovered device
    CONFIRMED = "confirmed"      # User-verified device
    IGNORED = "ignored"          # Explicitly excluded device

class ConnectionEvent:
    """Network connection or disconnection event with timestamp."""
    def __init__(self, event_type, timestamp=None):
        """
        Create a new connection event.
        
        Args:
            event_type: Event classification ("connect" or "disconnect")
            timestamp: Event occurrence time (defaults to current time)
        """
        self.event_type = event_type
        self.timestamp = timestamp or datetime.now()
    
    def to_dict(self):
        """Convert event to serializable dictionary."""
        return {
            "type": self.event_type,
            "timestamp": self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data):
        """Create from dictionary."""
        if not data:
            return None
        
        try:
            timestamp = datetime.fromisoformat(data["timestamp"])
            return cls(data["type"], timestamp)
        except (KeyError, ValueError):
            return cls(data.get("type", "unknown"))

class Device:
    """Network device with presence detection capabilities."""
    
    def __init__(self, mac, name=None, owner=None, device_type=DeviceType.UNKNOWN.value, 
                vendor=None, count_for_presence=None, confirmation_status=ConfirmationStatus.UNCONFIRMED.value):
        """
        Create a new device.
        
        Args:
            mac: Device MAC address (unique identifier)
            name: User-friendly device name
            owner: Person associated with this device
            device_type: Category of device
            vendor: Manufacturer name
            count_for_presence: Whether device indicates occupancy
            confirmation_status: User verification status
        """
        self.mac = mac.lower()
        self.name = name or f"Device-{mac[-5:]}"
        self.owner = owner
        self.device_type = device_type
        self.vendor = vendor or "Unknown"
        
        # Phones are automatically used for presence detection
        if count_for_presence is None:
            count_for_presence = (device_type == DeviceType.PHONE.value)
        self.count_for_presence = count_for_presence
        
        self.confirmation_status = confirmation_status
        
        # Connection timing data
        self.last_seen = None
        self.first_seen = datetime.now().isoformat()
        
        # Network state tracking
        self.connection_history = []
        self.offline_count = 0
        self.status = "inactive"
        
        # Probabilistic presence features
        self.confidence_score = 0.5  # Confidence level (0.0-1.0)
        self.typical_active_hours = []  # Expected online periods [(start_hour, end_hour),...]
    
        # Remote management capabilities
        self.supports_wol = False
        self.last_ip = None
        self.wol_success_count = 0
        self.wol_failure_count = 0
        
        # Interactive verification
        self.telegram_user_id = None
        self.last_telegram_ping_request_time = None
        self.is_pending_telegram_ping = False
    
    def record_connection(self):
        """Log device connection with current timestamp."""
        now = datetime.now()
        self.last_seen = now.isoformat()
        self.connection_history.append(ConnectionEvent("connect").to_dict())
        
        # Maintain reasonable history size
        if len(self.connection_history) > 100:
            self.connection_history = self.connection_history[-100:]
    
    def record_disconnection(self):
        """Log device disconnection with current timestamp."""
        self.connection_history.append(ConnectionEvent("disconnect").to_dict())
        
        # Maintain reasonable history size
        if len(self.connection_history) > 100:
            self.connection_history = self.connection_history[-100:]
    
    def is_probably_present(self, current_time=None):
        """
        Estimate likelihood of user presence despite device being offline.
        
        Uses temporal patterns and recent connectivity history to make
        an informed prediction about occupancy.
        
        Args:
            current_time: Reference time (defaults to now)
            
        Returns:
            bool: True if user is likely present
        """
        if self.status == "active":
            return True
        
        # Consider recent phone connectivity as strong presence indicator
        if (self.device_type == DeviceType.PHONE.value and 
            self.last_seen and 
            self.count_for_presence):
            
            now = current_time or datetime.now()
            
            # Check if within typical usage hours
            current_hour = now.hour
            is_typical_active_hour = False
            
            for hour_range in self.typical_active_hours:
                start_hour, end_hour = hour_range
                if start_hour <= current_hour <= end_hour:
                    is_typical_active_hour = True
                    break
            
            # Recent activity during expected hours suggests presence
            try:
                last_seen_time = datetime.fromisoformat(self.last_seen)
                time_since_last_seen = (now - last_seen_time).total_seconds() / 60  # minutes
                
                if time_since_last_seen < 60 and is_typical_active_hour:
                    return True
            except (ValueError, TypeError):
                pass
        
        return False
    
    def to_dict(self):
        """Convert to serializable dictionary representation."""
        return {
            "mac": self.mac,
            "name": self.name,
            "owner": self.owner,
            "device_type": self.device_type,
            "vendor": self.vendor,
            "count_for_presence": self.count_for_presence,
            "confirmation_status": self.confirmation_status,
            "last_seen": self.last_seen,
            "first_seen": self.first_seen,
            "status": self.status,
            "connection_history": self.connection_history,
            "confidence_score": self.confidence_score,
            "typical_active_hours": self.typical_active_hours,
            "supports_wol": self.supports_wol,
            "last_ip": self.last_ip,
            "wol_success_count": self.wol_success_count,
            "wol_failure_count": self.wol_failure_count,
            "telegram_user_id": self.telegram_user_id,
            "last_telegram_ping_request_time": self.last_telegram_ping_request_time,
            "is_pending_telegram_ping": self.is_pending_telegram_ping
        }
        
    @classmethod
    def from_dict(cls, data):
        """Reconstruct device from dictionary representation."""
        if not data or "mac" not in data:
            return None
            
        device = cls(
            mac=data["mac"],
            name=data.get("name"),
            owner=data.get("owner"),
            device_type=data.get("device_type", DeviceType.UNKNOWN.value),
            vendor=data.get("vendor", "Unknown"),
            count_for_presence=data.get("count_for_presence", False),
            confirmation_status=data.get("confirmation_status", ConfirmationStatus.UNCONFIRMED.value)
        )
        
        # Load additional properties
        device.last_seen = data.get("last_seen")
        device.first_seen = data.get("first_seen", device.first_seen)
        device.status = data.get("status", "inactive")
        device.connection_history = data.get("connection_history", [])
        device.confidence_score = data.get("confidence_score", 0.5)
        device.typical_active_hours = data.get("typical_active_hours", [])
        
        # Load Wake-on-LAN properties
        device.supports_wol = data.get("supports_wol", False)
        device.last_ip = data.get("last_ip")
        device.wol_success_count = data.get("wol_success_count", 0)
        device.wol_failure_count = data.get("wol_failure_count", 0)
        
        # Load Telegram ping properties
        device.telegram_user_id = data.get("telegram_user_id")
        device.last_telegram_ping_request_time = data.get("last_telegram_ping_request_time")
        device.is_pending_telegram_ping = data.get("is_pending_telegram_ping", False)
        
        return device