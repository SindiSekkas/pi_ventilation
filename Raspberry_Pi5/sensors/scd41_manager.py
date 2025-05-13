# sensors/scd41_manager.py
"""SCD41 CO2, temperature and humidity sensor management."""
import time
import logging
from sensirion_i2c_driver import LinuxI2cTransceiver, I2cConnection
from sensirion_i2c_scd import Scd4xI2cDevice

logger = logging.getLogger(__name__)

class SCD41Manager:
    def __init__(self, i2c_device='/dev/i2c-1'):
        self.i2c_device = i2c_device
        self.scd41 = None
        
    def initialize(self):
        """Initialize the SCD41 sensor."""
        try:
            logger.info("Initializing SCD41 sensor...")
            i2c_transceiver = LinuxI2cTransceiver(self.i2c_device)
            self.scd41 = Scd4xI2cDevice(I2cConnection(i2c_transceiver))
            
            # Stop any previous measurements
            try:
                self.scd41.stop_periodic_measurement()
                time.sleep(1)
            except Exception as e:
                logger.warning(f"Warning while stopping measurements: {e}")
            
            # Reinitialize sensor
            self.scd41.reinit()
            time.sleep(1)
            
            # Start measurements
            self.scd41.start_periodic_measurement()
            logger.info("SCD41 initialization complete")
            return True
        except Exception as e:
            logger.error(f"Error initializing SCD41: {e}")
            return False
    
    def read_measurement(self):
        """Read a measurement from the SCD41 sensor."""
        if self.scd41 is None:
            raise Exception("SCD41 sensor not initialized")
        
        return self.scd41.read_measurement()