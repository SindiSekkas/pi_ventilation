"""Controller implementations for ventilation simulation."""
from typing import Dict, Any, Optional
import logging
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

logger = logging.getLogger(__name__)

class OnOffController:
    """
    Simple threshold-based ventilation controller.
    
    Uses fixed thresholds for CO2 and temperature to make ventilation decisions.
    """
    
    def __init__(self, co2_high_threshold: float = 1000.0,
                 co2_low_threshold: float = 800.0,
                 temp_high_threshold: float = 25.0):
        """
        Initialize the On/Off controller.
        
        Args:
            co2_high_threshold: CO2 level to trigger ventilation (ppm)
            co2_low_threshold: CO2 level to stop ventilation (ppm)
            temp_high_threshold: Temperature to trigger ventilation (°C)
        """
        self.co2_high_threshold = co2_high_threshold
        self.co2_low_threshold = co2_low_threshold
        self.temp_high_threshold = temp_high_threshold
        self.last_action = "off"
        self.min_ventilation_time = 5  # Minimum ventilation run time (minutes)
        self.min_off_time = 10  # Minimum off time (minutes)
        self.runtime = 0  # Runtime counter
        self.offtime = 0  # Off time counter
    
    def decide_action(self, data: Dict[str, Any]) -> str:
        """
        Determine ventilation action based on current conditions.
        
        Args:
            data: Dictionary containing current sensor data
            
        Returns:
            str: Ventilation action ("off", "low", "medium", "max")
        """
        co2 = data["scd41"]["co2"]
        temperature = data["scd41"]["temperature"]
        occupants = data["room"]["occupants"]
        
        # Get current ventilation status
        is_ventilated = data["room"]["ventilated"]
        
        # Update runtime/offtime counters
        if is_ventilated:
            self.runtime += 5  # Assuming 5-minute time steps
            self.offtime = 0
        else:
            self.runtime = 0
            self.offtime += 5
        
        # Nobody home, turn off ventilation
        if occupants == 0:
            return "off"
        
        # Enforce minimum runtime (if already running)
        if is_ventilated and self.runtime < self.min_ventilation_time:
            # Continue with current speed
            return data["room"]["ventilation_speed"]
        
        # Enforce minimum off time (if not running)
        if not is_ventilated and self.offtime < self.min_off_time:
            return "off"
        
        # On/Off control logic
        if co2 > self.co2_high_threshold:
            # CO2 high - turn on ventilation
            if co2 > 1200:
                return "max"
            elif co2 > 1100:
                return "medium"
            else:
                return "low"
        elif temperature > self.temp_high_threshold:
            # Temperature high - turn on ventilation
            return "low"
        elif co2 <= self.co2_low_threshold and temperature <= self.temp_high_threshold:
            # Both CO2 and temperature are below thresholds - turn off
            return "off"
        
        # Maintain current state if no decision made
        return data["room"]["ventilation_speed"]


class PIDController:
    """
    PID-based ventilation controller.
    
    Uses Proportional-Integral-Derivative control for CO2 and temperature.
    """
    
    def __init__(self, co2_setpoint: float = 800.0, 
                 temp_setpoint: float = 22.0,
                 kp_co2: float = 0.01, 
                 ki_co2: float = 0.001, 
                 kd_co2: float = 0.005,
                 kp_temp: float = 0.2, 
                 ki_temp: float = 0.05, 
                 kd_temp: float = 0.1):
        """
        Initialize the PID controller.
        
        Args:
            co2_setpoint: Target CO2 level (ppm)
            temp_setpoint: Target temperature (°C)
            kp_co2, ki_co2, kd_co2: PID coefficients for CO2 control
            kp_temp, ki_temp, kd_temp: PID coefficients for temperature control
        """
        # Setpoints
        self.co2_setpoint = co2_setpoint
        self.temp_setpoint = temp_setpoint
        
        # CO2 PID parameters
        self.kp_co2 = kp_co2
        self.ki_co2 = ki_co2
        self.kd_co2 = kd_co2
        
        # Temperature PID parameters
        self.kp_temp = kp_temp
        self.ki_temp = ki_temp
        self.kd_temp = kd_temp
        
        # Error tracking
        self.co2_errors = [0, 0, 0]  # Current, previous, previous-previous
        self.temp_errors = [0, 0, 0]  # Current, previous, previous-previous
        
        # Error integrals
        self.co2_integral = 0
        self.temp_integral = 0
        
        # Last values for derivative
        self.last_co2 = None
        self.last_temp = None
        
        # Simulation time step (minutes)
        self.time_step = 5
        
        # Minimum runtime
        self.min_runtime = 5  # minutes
        self.runtime = 0
    
    def decide_action(self, data: Dict[str, Any]) -> str:
        """
        Determine ventilation action based on PID control.
        
        Args:
            data: Dictionary containing current sensor data
            
        Returns:
            str: Ventilation action ("off", "low", "medium", "max")
        """
        co2 = data["scd41"]["co2"]
        temperature = data["scd41"]["temperature"]
        occupants = data["room"]["occupants"]
        is_ventilated = data["room"]["ventilated"]
        
        # Track runtime
        if is_ventilated:
            self.runtime += self.time_step
        else:
            self.runtime = 0
        
        # Nobody home, turn off ventilation
        if occupants == 0:
            self.co2_integral = 0  # Reset integral terms when unoccupied
            self.temp_integral = 0
            return "off"
        
        # Enforce minimum runtime (if already running)
        if is_ventilated and self.runtime < self.min_runtime:
            # Continue with current speed
            return data["room"]["ventilation_speed"]
        
        # Calculate CO2 error
        co2_error = co2 - self.co2_setpoint
        
        # Update error history
        self.co2_errors.pop()
        self.co2_errors.insert(0, co2_error)
        
        # Calculate integral with anti-windup
        self.co2_integral = max(-200, min(200, self.co2_integral + co2_error * self.time_step))
        
        # Calculate derivative (change per minute)
        if self.last_co2 is not None:
            co2_derivative = (co2 - self.last_co2) / self.time_step
        else:
            co2_derivative = 0
        self.last_co2 = co2
        
        # Calculate PID output for CO2
        co2_output = (
            self.kp_co2 * co2_error + 
            self.ki_co2 * self.co2_integral + 
            self.kd_co2 * co2_derivative
        )
        
        # Calculate temperature error
        temp_error = temperature - self.temp_setpoint
        
        # Update error history
        self.temp_errors.pop()
        self.temp_errors.insert(0, temp_error)
        
        # Calculate integral with anti-windup
        self.temp_integral = max(-10, min(10, self.temp_integral + temp_error * self.time_step))
        
        # Calculate derivative
        if self.last_temp is not None:
            temp_derivative = (temperature - self.last_temp) / self.time_step
        else:
            temp_derivative = 0
        self.last_temp = temperature
        
        # Calculate PID output for temperature
        temp_output = (
            self.kp_temp * temp_error + 
            self.ki_temp * self.temp_integral + 
            self.kd_temp * temp_derivative
        )
        
        # Combine outputs with weighting
        # CO2 is primary concern (70%), temperature second (30%)
        combined_output = (co2_output * 0.7) + (temp_output * 0.3)
        
        # Map combined output to ventilation action
        if combined_output <= 0:
            return "off"
        elif combined_output < 2:
            return "low"
        elif combined_output < 4:
            return "medium"
        else:
            return "max"


class FuzzyController:
    """
    Fuzzy Logic ventilation controller.
    
    Uses fuzzy membership functions and rules to determine optimal ventilation.
    """
    
    def __init__(self):
        """Initialize the Fuzzy Logic controller with membership functions and rules."""
        # Create fuzzy variables
        try:
            # Input variables
            self.co2 = ctrl.Antecedent(np.arange(400, 1500, 1), 'co2')
            self.temperature = ctrl.Antecedent(np.arange(18, 28, 0.1), 'temperature')
            self.occupancy = ctrl.Antecedent(np.arange(0, 5, 1), 'occupancy')
            
            # Output variable
            self.ventilation = ctrl.Consequent(np.arange(0, 4, 1), 'ventilation')
            
            # Define membership functions for CO2
            self.co2['low'] = fuzz.trapmf(self.co2.universe, [400, 400, 600, 800])
            self.co2['medium'] = fuzz.trapmf(self.co2.universe, [600, 800, 1000, 1200])
            self.co2['high'] = fuzz.trapmf(self.co2.universe, [1000, 1200, 1500, 1500])
            
            # Define membership functions for temperature
            self.temperature['cool'] = fuzz.trapmf(self.temperature.universe, [18, 18, 20, 22])
            self.temperature['comfortable'] = fuzz.trapmf(self.temperature.universe, [20, 22, 23, 25])
            self.temperature['warm'] = fuzz.trapmf(self.temperature.universe, [23, 25, 28, 28])
            
            # Define membership functions for occupancy
            self.occupancy['empty'] = fuzz.trimf(self.occupancy.universe, [0, 0, 1])
            self.occupancy['occupied'] = fuzz.trapmf(self.occupancy.universe, [1, 1, 5, 5])
            
            # Define membership functions for ventilation
            self.ventilation['off'] = fuzz.trimf(self.ventilation.universe, [0, 0, 1])
            self.ventilation['low'] = fuzz.trimf(self.ventilation.universe, [0, 1, 2])
            self.ventilation['medium'] = fuzz.trimf(self.ventilation.universe, [1, 2, 3])
            self.ventilation['high'] = fuzz.trimf(self.ventilation.universe, [2, 3, 3])
            
            # Define rules
            rules = [
                # Nobody home - ventilation off
                ctrl.Rule(self.occupancy['empty'], self.ventilation['off']),
                
                # If occupied:
                # Low CO2, comfortable/cool temperature - off
                ctrl.Rule(self.occupancy['occupied'] & self.co2['low'] & 
                          (self.temperature['cool'] | self.temperature['comfortable']), 
                          self.ventilation['off']),
                
                # Low CO2, warm temperature - low
                ctrl.Rule(self.occupancy['occupied'] & self.co2['low'] & 
                          self.temperature['warm'], 
                          self.ventilation['low']),
                
                # Medium CO2, cool temperature - low
                ctrl.Rule(self.occupancy['occupied'] & self.co2['medium'] & 
                          self.temperature['cool'], 
                          self.ventilation['low']),
                
                # Medium CO2, comfortable temperature - low
                ctrl.Rule(self.occupancy['occupied'] & self.co2['medium'] & 
                          self.temperature['comfortable'], 
                          self.ventilation['low']),
                
                # Medium CO2, warm temperature - medium
                ctrl.Rule(self.occupancy['occupied'] & self.co2['medium'] & 
                          self.temperature['warm'], 
                          self.ventilation['medium']),
                
                # High CO2, any temperature - medium to high
                ctrl.Rule(self.occupancy['occupied'] & self.co2['high'] & 
                          self.temperature['cool'], 
                          self.ventilation['medium']),
                
                ctrl.Rule(self.occupancy['occupied'] & self.co2['high'] & 
                          self.temperature['comfortable'], 
                          self.ventilation['medium']),
                
                ctrl.Rule(self.occupancy['occupied'] & self.co2['high'] & 
                          self.temperature['warm'], 
                          self.ventilation['high'])
            ]
            
            # Create control system
            self.ventilation_ctrl = ctrl.ControlSystem(rules)
            self.ventilation_simulation = ctrl.ControlSystemSimulation(self.ventilation_ctrl)
            
            # Minimum runtime
            self.min_runtime = 5  # minutes
            self.runtime = 0
            
            self.ready = True
            
        except Exception as e:
            logger.error(f"Error initializing Fuzzy Controller: {e}")
            self.ready = False
    
    def decide_action(self, data: Dict[str, Any]) -> str:
        """
        Determine ventilation action using fuzzy logic.
        
        Args:
            data: Dictionary containing current sensor data
            
        Returns:
            str: Ventilation action ("off", "low", "medium", "max")
        """
        if not self.ready:
            return "off"
        
        try:
            co2 = data["scd41"]["co2"]
            temperature = data["scd41"]["temperature"]
            occupants = data["room"]["occupants"]
            is_ventilated = data["room"]["ventilated"]
            
            # Track runtime
            if is_ventilated:
                self.runtime += 5  # Assuming 5-minute time steps
            else:
                self.runtime = 0
                
            # Enforce minimum runtime (if already running)
            if is_ventilated and self.runtime < self.min_runtime:
                # Continue with current speed
                return data["room"]["ventilation_speed"]
                
            # Fuzzy control
            self.ventilation_simulation.input['co2'] = co2
            self.ventilation_simulation.input['temperature'] = temperature
            self.ventilation_simulation.input['occupancy'] = occupants
            
            # Compute fuzzy result
            self.ventilation_simulation.compute()
            
            # Get defuzzified result
            result = self.ventilation_simulation.output['ventilation']
            
            # Map to ventilation action
            if result < 0.5:
                return "off"
            elif result < 1.5:
                return "low"
            elif result < 2.5:
                return "medium"
            else:
                return "max"
                
        except Exception as e:
            logger.error(f"Error in fuzzy decision making: {e}")
            return "off"


class MockMarkovController:
    """
    Mock implementation of the MarkovController for testing.
    
    This is a simplified version that uses the same interface as the real controller.
    In a real simulation, we'd use the actual MarkovController.
    """
    
    def __init__(self):
        """Initialize the mock Markov controller."""
        self.co2_thresholds = {
            "low_max": 800,
            "medium_max": 1000
        }
        self.temp_thresholds = {
            "low_max": 20.0,
            "medium_max": 24.0
        }
        self.auto_mode = True
        self.night_mode_enabled = False
        self.night_mode_start_hour = 23
        self.night_mode_end_hour = 7
        self.min_ventilation_time = 5  # minutes
        self.runtime = 0
    
    def decide_action(self, data: Dict[str, Any]) -> str:
        """
        Determine ventilation action.
        
        Args:
            data: Dictionary containing current sensor data
            
        Returns:
            str: Ventilation action ("off", "low", "medium", "max")
        """
        # In the main script, we'll use the real MarkovController class
        # This is just a placeholder for structural completeness
        return "off"
    
    def set_auto_mode(self, enabled):
        """Set auto mode state."""
        self.auto_mode = enabled