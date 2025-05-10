# presence/models.py
"""Models for presence detection system."""
from enum import Enum
from datetime import datetime, time

class DeviceType(Enum):
    """Device type classification."""
    PHONE = "phone"
    LAPTOP = "laptop"
    TABLET = "tablet"
    TV = "tv"
    IOT_DEVICE = "iot_device"
    UNKNOWN = "unknown"

class ConfirmationStatus(Enum):
    """Device confirmation status."""
    UNCONFIRMED = "unconfirmed"  # Newly discovered, not yet confirmed
    CONFIRMED = "confirmed"      # Confirmed by user
    IGNORED = "ignored"          # User chose to ignore this device

class ConnectionEvent:
    """Represents a connection or disconnection event."""
    def __init__(self, event_type, timestamp=None):
        """
        Initialize a connection event.
        
        Args:
            event_type: Type of event ("connect" or "disconnect")
            timestamp: When the event occurred (default: now)
        """
        self.event_type = event_type
        self.timestamp = timestamp or datetime.now()
    
    def to_dict(self):
        """Convert to dictionary."""
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
    """Represents a network device for presence detection."""
    
    def __init__(self, mac, name=None, owner=None, device_type=DeviceType.UNKNOWN.value, 
                vendor=None, count_for_presence=None, confirmation_status=ConfirmationStatus.UNCONFIRMED.value):
        """
        Initialize a device.
        
        Args:
            mac: MAC address of device
            name: Name of device
            owner: Owner of device
            device_type: Type of device (phone, laptop, etc.)
            vendor: Device manufacturer
            count_for_presence: Whether to count this device for presence detection
            confirmation_status: Whether device has been confirmed by a user
        """
        self.mac = mac.lower()
        self.name = name or f"Device-{mac[-5:]}"
        self.owner = owner
        self.device_type = device_type
        self.vendor = vendor or "Unknown"
        
        # Automatically count phones for presence
        if count_for_presence is None:
            count_for_presence = (device_type == DeviceType.PHONE.value)
        self.count_for_presence = count_for_presence
        
        self.confirmation_status = confirmation_status
        
        # Timing information
        self.last_seen = None
        self.first_seen = datetime.now().isoformat()
        
        # Presence detection parameters
        self.connection_history = []
        self.offline_count = 0
        self.status = "inactive"
        
        # Advanced presence features
        self.confidence_score = 0.5  # Default: medium confidence
        self.typical_active_hours = []  # Store times when device is typically active
    
        # Wake-on-LAN support
        self.supports_wol = False  # Whether device supports Wake-on-LAN
        self.last_ip = None  # Last known IP address
        self.wol_success_count = 0  # Count of successful wake attempts
        self.wol_failure_count = 0  # Count of failed wake attempts
    
    def record_connection(self):
        """Record a connection event."""
        now = datetime.now()
        self.last_seen = now.isoformat()
        self.connection_history.append(ConnectionEvent("connect").to_dict())
        
        # Trim history if needed
        if len(self.connection_history) > 100:
            self.connection_history = self.connection_history[-100:]
    
    def record_disconnection(self):
        """Record a disconnection event."""
        self.connection_history.append(ConnectionEvent("disconnect").to_dict())
        
        # Trim history if needed
        if len(self.connection_history) > 100:
            self.connection_history = self.connection_history[-100:]
    
    def is_probably_present(self, current_time=None):
        """
        Determine if the device is probably present even if currently offline.
        Uses time-weighted probability based on connection history.
        
        Returns:
            bool: True if device is probably present
        """
        if self.status == "active":
            return True
        
        # If it's a phone and we've seen it recently
        if (self.device_type == DeviceType.PHONE.value and 
            self.last_seen and 
            self.count_for_presence):
            
            # Use current time or default to now
            now = current_time or datetime.now()
            
            # Check if the current time is within typical active hours
            current_hour = now.hour
            is_typical_active_hour = False
            
            for hour_range in self.typical_active_hours:
                start_hour, end_hour = hour_range
                if start_hour <= current_hour <= end_hour:
                    is_typical_active_hour = True
                    break
            
            # If last seen was within the past hour and we're in active hours
            try:
                last_seen_time = datetime.fromisoformat(self.last_seen)
                time_since_last_seen = (now - last_seen_time).total_seconds() / 60  # minutes
                
                if time_since_last_seen < 60 and is_typical_active_hour:
                    return True
            except (ValueError, TypeError):
                pass
        
        return False
    
    def to_dict(self):
        """Convert device to dictionary for serialization."""
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
            "wol_failure_count": self.wol_failure_count
        }
        
    @classmethod
    def from_dict(cls, data):
        """Create device from dictionary."""
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
        
        return device