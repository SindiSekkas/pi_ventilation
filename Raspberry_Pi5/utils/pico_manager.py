# utils/pico_manager.py
"""PicoWH ventilation control interface."""
import logging
import requests

logger = logging.getLogger(__name__)

class PicoManager:
    def __init__(self, pico_ip):
        """Initialize PicoWH manager with device IP."""
        self.pico_ip = pico_ip
        self.pico_url = f"http://{pico_ip}"
    
    def find_pico_service(self):
        """Try to connect to PicoWH service."""
        logger.info(f"Attempting to connect to PicoWH at {self.pico_ip}")
        
        try:
            response = requests.get(f"{self.pico_url}/status", timeout=5)
            if response.status_code == 200:
                logger.info(f"Successfully connected to PicoWH at {self.pico_ip}")
                return self.pico_url
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to connect to PicoWH at {self.pico_ip}: {str(e)}")
        
        return None
    
    def get_ventilation_status(self):
        """Get current ventilation status from PicoWH."""
        try:
            pico_url = self.find_pico_service()
            if not pico_url:
                logger.error("Could not find PicoWH for vent status check")
                return False
                
            response = requests.get(f"{pico_url}/status")
            if response.status_code == 200:
                data = response.json()
                return data.get("ventActive", False)
            else:
                logger.error(f"Error getting ventilation status: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Error checking ventilation status: {str(e)}")
            return False
    
    def control_ventilation(self, state, speed=None):
        """Control ventilation state and speed."""
        if state not in ['on', 'off']:
            logger.error(f"Invalid ventilation state: {state}")
            return False
        
        try:
            pico_url = self.find_pico_service()
            if not pico_url:
                logger.error("Could not find PicoWH for ventilation control")
                return False
                
            # For 'on' state, route to the appropriate speed endpoint
            if state == 'on':
                if speed not in ['low', 'medium', 'max']:
                    logger.error(f"Invalid ventilation speed: {speed}")
                    return False
                response = requests.get(f"{pico_url}/vent/{speed}")
            else:  # 'off' state
                response = requests.get(f"{pico_url}/vent/off")
                
            if response.status_code == 200:
                logger.info(f"Ventilation {state} ({speed if state == 'on' and speed else 'off'}) command sent successfully")
                return True
            else:
                logger.error(f"Error controlling ventilation: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Error in ventilation control: {str(e)}")
            return False
    
    def get_ventilation_speed(self):
        """Get current ventilation speed from PicoWH."""
        try:
            pico_url = self.find_pico_service()
            if not pico_url:
                logger.error("Could not find PicoWH for vent status check")
                return "off"
                
            response = requests.get(f"{pico_url}/status")
            if response.status_code == 200:
                data = response.json()
                return data.get("ventSpeed", "off")
            else:
                logger.error(f"Error getting ventilation speed: {response.status_code}")
                return "off"
        except Exception as e:
            logger.error(f"Error checking ventilation speed: {str(e)}")
            return "off"