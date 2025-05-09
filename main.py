"""Main entry point for the ventilation system."""
import os
import sys
import time
import logging
import threading
from datetime import datetime

# Setup logging before imports
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ventilation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import components
from config.settings import (
    MEASUREMENT_INTERVAL, INIT_MEASUREMENTS, 
    PICO_IP, DATA_DIR, CSV_DIR
)
from sensors.scd41_manager import SCD41Manager
from sensors.bmp280 import BMP280
from sensors.data_manager import DataManager
from sensors.reader import SensorReader
from utils.pico_manager import PicoManager
from control.ventilation_controller import VentilationController

def main():
    """Main application entry point."""
    try:
        logger.info("Starting ventilation system")
        
        # Create data directories
        os.makedirs(DATA_DIR, exist_ok=True)
        os.makedirs(CSV_DIR, exist_ok=True)
        
        # Initialize components
        data_manager = DataManager(csv_dir=CSV_DIR)
        pico_manager = PicoManager(pico_ip=PICO_IP)
        scd41_manager = SCD41Manager()
        
        # Initialize sensor reader
        sensor_reader = SensorReader(
            data_manager=data_manager,
            scd41_manager=scd41_manager,
            bmp280_manager=BMP280,
            pico_manager=pico_manager,
            measurement_interval=MEASUREMENT_INTERVAL
        )
        
        # Start sensor reader thread
        if not sensor_reader.start():
            logger.error("Failed to start sensor reader")
            return 1
            
        # Initialize ventilation controller
        ventilation_controller = VentilationController(
            data_manager=data_manager,
            pico_manager=pico_manager
        )
        
        # Start ventilation controller
        if ventilation_controller.start():
            logger.info("Ventilation controller started")
        else:
            logger.error("Failed to start ventilation controller")
        
        logger.info("Ventilation system started successfully")
        
        # Run indefinitely to keep main thread alive
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutting down ventilation system")
        
        return 0
        
    except Exception as e:
        logger.critical(f"Error starting ventilation system: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())