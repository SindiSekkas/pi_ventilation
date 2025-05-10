"""Test script for verifying sensor functionality."""
import time
import logging
import sys
import os
from datetime import datetime

# parent directory to the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("sensor_test")

# Import components
try:
    from sensors.scd41_manager import SCD41Manager
    from sensors.bmp280 import BMP280
    from utils.pico_manager import PicoManager
except ImportError:
    logger.error("Unable to import required modules. Please check your installation.")
    sys.exit(1)

class SimpleDataLogger:
    """Simple logger for sensor data during testing."""
    def __init__(self):
        self.data = {
            "scd41": {"co2": None, "temperature": None, "humidity": None},
            "bmp280": {"temperature": None, "pressure": None}
        }
    
    def update(self, source, **kwargs):
        """Update data values."""
        for key, value in kwargs.items():
            if source in self.data and key in self.data[source]:
                self.data[source][key] = value
    
    def display(self):
        """Print current sensor values."""
        print("\n--- Sensor Readings ---")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\nSCD41 Sensor:")
        print(f"  CO2: {self.data['scd41']['co2']} ppm")
        print(f"  Temperature: {self.data['scd41']['temperature']} °C")
        print(f"  Humidity: {self.data['scd41']['humidity']} %")
        print("\nBMP280 Sensor:")
        print(f"  Temperature: {self.data['bmp280']['temperature']} °C")
        print(f"  Pressure: {self.data['bmp280']['pressure']} hPa")
        print("\n---------------------")

def test_pico():
    """Test connection to Pico W device."""
    logger.info("Testing PicoW connection...")
    pico_ip = "192.168.0.110"  # Use your actual PicoW IP
    pico = PicoManager(pico_ip)
    
    if pico.find_pico_service():
        logger.info("✅ Successfully connected to PicoW")
        status = pico.get_ventilation_status()
        speed = pico.get_ventilation_speed()
        logger.info(f"Current ventilation: {'ON' if status else 'OFF'}, Speed: {speed}")
        return True
    else:
        logger.error("❌ Failed to connect to PicoW")
        return False

def test_sensors():
    """Initialize and test both sensors."""
    logger.info("Testing sensors...")
    data_logger = SimpleDataLogger()
    sensors_ok = True
    
    # Test SCD41
    try:
        logger.info("Initializing SCD41 sensor...")
        scd41 = SCD41Manager()
        if scd41.initialize():
            logger.info("✅ SCD41 initialized successfully")
            
            # Wait for first measurement
            logger.info("Waiting for measurement (10 seconds)...")
            time.sleep(10)
            
            # Read data
            co2, temp, humidity = scd41.read_measurement()
            data_logger.update("scd41", 
                            co2=co2.co2, 
                            temperature=temp.degrees_celsius, 
                            humidity=humidity.percent_rh)
            logger.info(f"SCD41 readings: CO2={co2.co2} ppm, "
                      f"Temperature={temp.degrees_celsius} °C, "
                      f"Humidity={humidity.percent_rh} %")
        else:
            logger.error("❌ Failed to initialize SCD41 sensor")
            sensors_ok = False
    except Exception as e:
        logger.error(f"❌ Error testing SCD41: {e}")
        sensors_ok = False
    
    # Test BMP280
    try:
        logger.info("Initializing BMP280 sensor...")
        # Use actual bus number and address for your setup
        bmp280 = BMP280(bus_number=20, address=0x76)
        
        # Read data
        temperature = bmp280.read_temperature()
        pressure = bmp280.read_pressure()
        data_logger.update("bmp280", temperature=temperature, pressure=pressure)
        
        logger.info(f"BMP280 readings: Temperature={temperature} °C, Pressure={pressure} hPa")
        logger.info("✅ BMP280 working correctly")
    except Exception as e:
        logger.error(f"❌ Error testing BMP280: {e}")
        sensors_ok = False
    
    # Display readings
    data_logger.display()
    
    return sensors_ok

if __name__ == "__main__":
    print("\n===== VENTILATION SYSTEM SENSOR TEST =====\n")
    
    # Test PicoW connection
    pico_ok = test_pico()
    
    # Test sensors
    sensors_ok = test_sensors()
    
    # Report overall status
    print("\n===== TEST RESULTS =====")
    print(f"PicoW Connection: {'✅ OK' if pico_ok else '❌ FAILED'}")
    print(f"Sensors: {'✅ OK' if sensors_ok else '❌ ISSUES DETECTED'}")
    
    if not pico_ok or not sensors_ok:
        print("\n⚠️  Some tests failed. Please check the log for details.")
        sys.exit(1)
    else:
        print("\n✅ All tests passed! The system is working correctly.")
        sys.exit(0)

# tests/test_sensors.py