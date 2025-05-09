"""Configuration settings for the ventilation system."""
import os
import logging
from datetime import timedelta

# Configure logging
logging.basicConfig(
    filename='ventilation.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Create console handler and set level to debug
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

logger = logging.getLogger(__name__)

# Sensor configuration
MEASUREMENT_INTERVAL = 120  # 2 minutes between measurements
INIT_MEASUREMENTS = 5  # Number of initialization measurements

# PicoWH configuration
PICO_IP = os.environ.get("PICO_IP", "192.168.0.110")

# Room default settings
DEFAULT_OCCUPANTS = 1
DEFAULT_CO2_THRESHOLD = 1000  # ppm
DEFAULT_TEMP_MIN = 20.0  # °C
DEFAULT_TEMP_MAX = 24.0  # °C

# Ventilation settings
VENTILATION_SPEEDS = ["off", "low", "medium", "max"]
AUTO_VENTILATION = True  # Enable automatic ventilation control

# Directories
DATA_DIR = "data"
CSV_DIR = os.path.join(DATA_DIR, "csv")