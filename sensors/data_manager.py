# sensors/data_manager.py
"""Management of sensor data storage and retrieval."""
import os
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class DataManager:
    def __init__(self, csv_dir="data/csv"):
        """Initialize the data manager with default values."""
        self.latest_data = {
            "timestamp": None,
            "scd41": {"co2": None, "temperature": None, "humidity": None},
            "bmp280": {"temperature": None, "pressure": None},
            "room": {"occupants": 1, "ventilated": False, "ventilation_speed": "off"},
            "initialization": {
                "status": True,
                "current": 0,
                "total": 5,
                "time_remaining": 600
            }
        }
        self.csv_dir = csv_dir
        os.makedirs(csv_dir, exist_ok=True)
    
    def update_sensor_data(self, scd41_data, bmp280_data):
        """Update sensor data with new readings."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        self.latest_data.update({
            "timestamp": timestamp,
            "scd41": {
                "co2": round(scd41_data[0].co2, 1),
                "temperature": round(scd41_data[1].degrees_celsius, 1),
                "humidity": round(scd41_data[2].percent_rh, 1)
            },
            "bmp280": {
                "temperature": round(bmp280_data[0], 1),
                "pressure": round(bmp280_data[1], 1)
            }
        })
        
        logger.info(f"New sensor data: CO2={self.latest_data['scd41']['co2']} ppm, " 
                   f"Temp={self.latest_data['scd41']['temperature']}°C, "
                   f"Humidity={self.latest_data['scd41']['humidity']}%")
        return self.latest_data
    
    def update_room_data(self, occupants=None, ventilated=None, ventilation_speed=None):
        """Update room occupancy and ventilation status."""
        if occupants is not None:
            self.latest_data["room"]["occupants"] = occupants
        if ventilated is not None:
            self.latest_data["room"]["ventilated"] = ventilated
        if ventilation_speed is not None:
            self.latest_data["room"]["ventilation_speed"] = ventilation_speed
        return self.latest_data["room"]
    
    def update_init_status(self, start_time, completed_measurements):
        """Update initialization status."""
        current_time = datetime.now().timestamp()
        elapsed = int(current_time - start_time)
        time_remaining = max(600 - elapsed, 0)
        
        self.latest_data["initialization"].update({
            "status": completed_measurements < 5,
            "current": completed_measurements,
            "total": 5,
            "time_remaining": time_remaining
        })
        
        return self.latest_data["initialization"]
    
    def save_measurement_to_csv(self, ventilation_status, ventilation_speed="off"):
        """Save current measurement to CSV file."""
        today = datetime.now().strftime("%Y%m%d")
        filename = os.path.join(self.csv_dir, f"{today}.csv")
        file_exists = os.path.isfile(filename)
        
        try:
            with open(filename, "a") as f:
                if not file_exists:
                    f.write("Timestamp,CO2,SCD41_Temperature,Humidity,BMP280_Temperature,BMP280_Pressure,Occupants,ventilated,ventilation_speed\n")
                
                timestamp = self.latest_data.get("timestamp", "")
                scd41 = self.latest_data.get("scd41", {})
                bmp280 = self.latest_data.get("bmp280", {})
                room = self.latest_data.get("room", {})
                
                line = f'{timestamp},{scd41.get("co2", "")},{scd41.get("temperature", "")},'
                line += f'{scd41.get("humidity", "")},{bmp280.get("temperature", "")},{bmp280.get("pressure", "")},'
                line += f'{room.get("occupants", "")},{ventilation_status},{ventilation_speed}\n'
                
                f.write(line)
            return True
        except Exception as e:
            logger.error(f"Failed to save measurement to CSV: {e}")
            return False