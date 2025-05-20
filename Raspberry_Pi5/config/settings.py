# config/settings.py

"""Configuration settings for the ventilation system."""
import os
from datetime import timedelta

# Sensor configuration
MEASUREMENT_INTERVAL = 120  # 2 minutes between measurements
INIT_MEASUREMENTS = 5  # Number of initialization measurements

# PicoWH configuration
PICO_IP = os.environ.get("PICO_IP", "192.168.0.110")

# Room default settings
DEFAULT_OCCUPANTS = 2
DEFAULT_CO2_THRESHOLD = 1000  # ppm
DEFAULT_TEMP_MIN = 20.0  # °C
DEFAULT_TEMP_MAX = 24.0  # °C

# Ventilation settings
VENTILATION_SPEEDS = ["off", "low", "medium", "max"]
AUTO_VENTILATION = True  # Enable automatic ventilation control

# Night mode settings
NIGHT_MODE_ENABLED = True  # Enable night mode by default
NIGHT_MODE_START_HOUR = 23  # 11 PM
NIGHT_MODE_END_HOUR = 7     # 7 AM

# Markov Controller settings
MARKOV_ENABLE_EXPLORATION = True  # Enable random actions for exploration

# Directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
CSV_DIR = os.path.join(DATA_DIR, "csv")

# Occupancy history paths
OCCUPANCY_HISTORY_DIR = os.path.join(DATA_DIR, "occupancy_history") 
OCCUPANCY_HISTORY_FILE = os.path.join(OCCUPANCY_HISTORY_DIR, "occupancy_history.csv")
OCCUPANCY_PROBABILITIES_FILE = os.path.join(OCCUPANCY_HISTORY_DIR, "occupancy_probabilities.json")

# Skip initialisation measurements
SKIP_INITIALIZATION = True
INIT_MEASUREMENTS = 0 if SKIP_INITIALIZATION else 5

# Bot configuration
try:
    from dotenv import load_dotenv
    
    # Load bot environment variables
    bot_env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "bot", ".env")
    if os.path.exists(bot_env_path):
        load_dotenv(bot_env_path)
    
    # Bot settings
    BOT_TOKEN = os.environ.get("BOT_TOKEN")
    ADMIN_ID = int(os.environ.get("ADMIN_ID", 0))
    
except ImportError:
    BOT_TOKEN = None
    ADMIN_ID = None