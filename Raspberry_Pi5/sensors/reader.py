# sensors/reader.py
"""Sensor reading thread functionality."""
import sys
import time
import threading
import logging
from datetime import datetime
from config.settings import SKIP_INITIALIZATION, INIT_MEASUREMENTS

logger = logging.getLogger(__name__)

class SensorReader:
    def __init__(self, data_manager, scd41_manager, bmp280_manager, pico_manager, measurement_interval=120):
        """Sets up sensor reading components and configuration."""
        self.data_manager = data_manager
        self.scd41_manager = scd41_manager
        self.bmp280_manager = bmp280_manager
        self.pico_manager = pico_manager
        self.measurement_interval = measurement_interval
        self.thread = None
        self.running = False
        self.start_time = None
        self.completed_measurements = 0
    
    def start(self):
        """Starts background sensor reading if not already running."""
        if self.thread is not None and self.thread.is_alive():
            logger.warning("Sensor reader thread already running")
            return False
        
        self.running = True
        self.thread = threading.Thread(target=self._reader_thread, daemon=True)
        self.thread.start()
        return True
    
    def _reader_thread(self):
        """Background thread for sensor initialization and continuous monitoring."""
        logger.info("Starting sensor initialization...")
        try:
            self.start_time = time.time()
            self.completed_measurements = 0
            
            # Initialize sensors
            if not self.scd41_manager.initialize():
                raise Exception("Failed to initialize SCD41 sensor")
            
            try:
                bmp280 = self.bmp280_manager(bus_number=20, address=0x76)
            except Exception as e:
                logger.error(f"Failed to initialize BMP280 sensor: {e}")
                raise
            
            if SKIP_INITIALIZATION:
                logger.info("Initialization phase skipped by configuration")
                self.completed_measurements = INIT_MEASUREMENTS if INIT_MEASUREMENTS > 0 else 5
                self.data_manager.update_init_status(self.start_time, self.completed_measurements)
                
                # Take one quick measurement to populate data
                try:
                    co2, temp_scd41, humidity = self.scd41_manager.read_measurement()
                    temp_bmp280 = bmp280.read_temperature()
                    pressure = bmp280.read_pressure()
                    
                    self.data_manager.update_sensor_data(
                        (co2, temp_scd41, humidity),
                        (temp_bmp280, pressure)
                    )
                    logger.info("Initial measurement taken")
                except Exception as e:
                    logger.error(f"Error during initial measurement: {e}")
                
            else:
                # Wait for first measurement
                logger.info("Waiting for the first measurement...")
                time.sleep(120)
                
                # Perform initialization measurements
                logger.info("Performing initialization measurements...")
                for i in range(INIT_MEASUREMENTS):
                    if not self.data_manager.latest_data["initialization"]["status"]:
                        logger.info("Initialization skipped by user")
                        break
                        
                    try:
                        co2, temp_scd41, humidity = self.scd41_manager.read_measurement()
                        temp_bmp280 = bmp280.read_temperature()
                        pressure = bmp280.read_pressure()
                        
                        self.completed_measurements += 1
                        self.data_manager.update_init_status(self.start_time, self.completed_measurements)
                        
                        if self.completed_measurements == INIT_MEASUREMENTS:
                            self.data_manager.update_sensor_data(
                                (co2, temp_scd41, humidity),
                                (temp_bmp280, pressure)
                            )
                            
                        logger.info(f"Initialization measurement {self.completed_measurements}/{INIT_MEASUREMENTS} complete")
                        time.sleep(120)
                    except Exception as e:
                        logger.error(f"Error during initialization measurement: {e}")
                        time.sleep(5)
                        continue
            
            # Start normal measurement loop
            logger.info("Starting normal measurement loop...")
            while self.running:
                try:
                    # Read sensors
                    co2, temp_scd41, humidity = self.scd41_manager.read_measurement()
                    temp_bmp280 = bmp280.read_temperature()
                    pressure = bmp280.read_pressure()
                    
                    # Update data
                    self.data_manager.update_sensor_data(
                        (co2, temp_scd41, humidity),
                        (temp_bmp280, pressure)
                    )
                    
                    # Check ventilation status and save to CSV
                    ventilation_status = self.pico_manager.get_ventilation_status()
                    ventilation_speed = self.pico_manager.get_ventilation_speed()
                    self.data_manager.save_measurement_to_csv(ventilation_status, ventilation_speed)
                    
                except Exception as e:
                    logger.error(f"Error reading sensors: {e}")
                    time.sleep(5)
                    continue
                    
                time.sleep(self.measurement_interval)
                
        except Exception as e:
            logger.critical(f"Critical error in sensor thread: {e}", exc_info=True)
            sys.exit(1)