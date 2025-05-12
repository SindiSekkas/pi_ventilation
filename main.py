"""Main entry point for the ventilation system."""
import os
import sys
import time
import logging
import threading
import subprocess
import queue
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
    PICO_IP, DATA_DIR, CSV_DIR, BOT_TOKEN,
    MARKOV_ENABLE_EXPLORATION 
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
from preferences.preference_manager import PreferenceManager
from presence.occupancy_history_manager import OccupancyHistoryManager
from predictive.occupancy_pattern_analyzer import OccupancyPatternAnalyzer

def run_bot(pico_manager=None, controller=None, data_manager=None, sleep_analyzer=None, preference_manager=None, occupancy_analyzer=None, device_manager=None, telegram_ping_tasks_queue=None):
    """
    Run the Telegram bot in a separate process.
    
    Passes all system components to the bot to enable control and monitoring.
    """
    try:
        # Import bot main
        from bot.main import main as bot_main
        
        # Run bot with passed components
        bot_main(pico_manager, controller, data_manager, sleep_analyzer, preference_manager, occupancy_analyzer, device_manager, telegram_ping_tasks_queue)
    except Exception as e:
        logger.error(f"Error in bot process: {e}", exc_info=True)
        # Don't exit, just log the error and continue

def main():
    """
    Main application entry point.
    
    Initializes and orchestrates all system components, including:
    - Sensors and data collection
    - Presence detection
    - Ventilation control
    - Adaptive sleep pattern analysis
    - User preferences
    - Telegram bot interface
    """
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

        # Create telegram_ping_tasks_queue for device presence verification
        telegram_ping_tasks_queue = queue.Queue()

        # Initialize device manager with Telegram ping queue
        device_manager = DeviceManager(data_dir=os.path.join(DATA_DIR, "presence"), telegram_ping_queue=telegram_ping_tasks_queue)   

        # Initialize preference manager with updated data_dir parameter
        preference_manager = PreferenceManager(data_dir=DATA_DIR)

        # Initialize occupancy history manager
        occupancy_history_manager = OccupancyHistoryManager(data_dir=DATA_DIR)
        
        # Initialize occupancy pattern analyzer
        occupancy_history_file = os.path.join(DATA_DIR, "occupancy_history", "occupancy_history.csv")
        occupancy_pattern_analyzer = OccupancyPatternAnalyzer(occupancy_history_file)

        # Initialize presence controller with longer scan interval to prevent blocking
        presence_controller = PresenceController(
            device_manager=device_manager,
            data_manager=data_manager,
            occupancy_history_manager=occupancy_history_manager,
            scan_interval=600  # Changed from 300 to 600 (10 minutes) to prevent frequent scans
        )

        # Start presence controller
        if presence_controller.start():
            logger.info("Presence detection system started with 10-minute intervals")
        else:
            logger.error("Failed to start presence detection system")

        # Initialize Markov controller with preference_manager and occupancy_analyzer
        markov_controller = MarkovController(
            data_manager=data_manager,
            pico_manager=pico_manager,
            preference_manager=preference_manager,
            occupancy_analyzer=occupancy_pattern_analyzer,
            enable_exploration=MARKOV_ENABLE_EXPLORATION 
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
                args=(pico_manager, markov_controller, data_manager, sleep_analyzer, preference_manager, occupancy_pattern_analyzer, device_manager, telegram_ping_tasks_queue),
                daemon=True,
                name="TelegramBot"  # Thread name for easier debugging
            )
            bot_thread.start()
            logger.info("Telegram bot started in thread")
        else:
            logger.warning("BOT_TOKEN not configured, bot will not start")
        
        # Periodic task for pattern analysis (run every 6 hours)
        def periodic_pattern_update():
            while True:
                try:
                    # Process occupancy patterns periodically
                    occupancy_pattern_analyzer.update_patterns()
                    logger.info("Updated occupancy patterns")
                except Exception as e:
                    logger.error(f"Error updating occupancy patterns: {e}")
                # Wait 6 hours before next update
                time.sleep(21600)
        
        # Start periodic pattern update thread
        pattern_thread = threading.Thread(target=periodic_pattern_update, daemon=True, name="PatternUpdater")
        pattern_thread.start()
        logger.info("Started periodic pattern update task")
        
        # Monitor thread health
        def monitor_threads():
            while True:
                time.sleep(60)  # Check every minute
                if bot_thread and not bot_thread.is_alive():
                    logger.error("Bot thread has died!")
                logger.debug(f"Thread status - Bot: {'Alive' if bot_thread and bot_thread.is_alive() else 'Dead'}")
        
        # Start thread monitoring
        monitor_thread = threading.Thread(target=monitor_threads, daemon=True, name="ThreadMonitor")
        monitor_thread.start()
        
        logger.info("Ventilation system started successfully")
        
        # Run indefinitely to keep main thread alive
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutting down ventilation system")
            
            # Stop controllers gracefully
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