"""Main entry point for the ventilation system."""
# main.py
import os
import sys
import time
import logging
import threading
import subprocess
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
    PICO_IP, DATA_DIR, CSV_DIR, BOT_TOKEN
)
from sensors.scd41_manager import SCD41Manager
from sensors.bmp280 import BMP280
from sensors.data_manager import DataManager
from sensors.reader import SensorReader
from utils.pico_manager import PicoManager
from control.ventilation_controller import VentilationController
from control.markov_controller import MarkovController
from presence.device_manager import DeviceManager
from presence.presence_controller import PresenceController
from predictive.adaptive_sleep_analyzer import AdaptiveSleepAnalyzer

def run_bot(pico_manager, controller, data_manager, sleep_analyzer):
    """Run the Telegram bot in a separate process."""
    try:
        # Import bot main
        from bot.main import main as bot_main
        
        # Run bot with passed components
        bot_main(pico_manager, controller, data_manager, sleep_analyzer)
    except Exception as e:
        logger.error(f"Error in bot process: {e}", exc_info=True)
        # Don't exit, just log the error and continue

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

        # Initialize device manager
        device_manager = DeviceManager(data_dir=os.path.join(DATA_DIR, "presence"))   

        # Initialize presence controller
        presence_controller = PresenceController(
            device_manager=device_manager,
            data_manager=data_manager,
            scan_interval=300  # 5 minutes between scans
        )

        # Start presence controller
        if presence_controller.start():
            logger.info("Presence detection system started")
        else:
            logger.error("Failed to start presence detection system")

        # Initialize Markov controller
        markov_controller = MarkovController(
            data_manager=data_manager,
            pico_manager=pico_manager
        )
        
        # Start Markov controller
        if markov_controller.start():
            logger.info("Markov controller started")
        else:
            logger.error("Failed to start Markov controller")
            
        # Initialize adaptive sleep analyzer
        sleep_analyzer = AdaptiveSleepAnalyzer(
            data_manager=data_manager,
            controller=markov_controller
        )
        
        # Start adaptive sleep analyzer
        if sleep_analyzer.start():
            logger.info("Adaptive sleep analyzer started")
        else:
            logger.error("Failed to start adaptive sleep analyzer")
        
        # Start bot if token is configured
        bot_thread = None
        if BOT_TOKEN:
            logger.info("Starting Telegram bot")
            bot_thread = threading.Thread(
                target=run_bot, 
                args=(pico_manager, markov_controller, data_manager, sleep_analyzer),
                daemon=True
            )
            bot_thread.start()
            logger.info("Telegram bot started")
        else:
            logger.warning("BOT_TOKEN not configured, bot will not start")
        
        logger.info("Ventilation system started successfully")
        
        # Run indefinitely to keep main thread alive
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutting down ventilation system")
            
            # Stop controllers
            markov_controller.stop()
            presence_controller.stop()
            sleep_analyzer.stop()
            
            # Stop bot thread if it exists
            if bot_thread and bot_thread.is_alive():
                logger.info("Waiting for bot to stop...")
                # No direct way to stop bot gracefully, let daemon thread die
        
        return 0
        
    except Exception as e:
        logger.critical(f"Error starting ventilation system: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())