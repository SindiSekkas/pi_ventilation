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

# Directories
DATA_DIR = "data"
CSV_DIR = os.path.join(DATA_DIR, "csv")

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
    else:
        logger.warning(f"Bot .env file not found at {bot_env_path}")
    
    # Bot settings
    BOT_TOKEN = os.environ.get("BOT_TOKEN")
    ADMIN_ID = int(os.environ.get("ADMIN_ID", 0))
    
    if not BOT_TOKEN:
        logger.warning("BOT_TOKEN not found in environment variables")
except ImportError:
    logger.warning("python-dotenv not installed, bot features will not work")
    BOT_TOKEN = None
    ADMIN_ID = None