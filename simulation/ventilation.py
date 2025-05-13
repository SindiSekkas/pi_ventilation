# simulation/ventilation.py
"""
Ventilation system simulator for adaptive ventilation system.
Implements different control strategies for comparison.
"""
import logging
import random
from datetime import datetime, timedelta
from enum import Enum, auto

logger = logging.getLogger(__name__)

class VentilationMode(Enum):
    """Ventilation system modes."""
    OFF = "off"
    NATURAL = "natural"  # Window-based
    MECHANICAL = "mechanical"  # Fan-based

class VentilationSpeed(Enum):
    """Mechanical ventilation speed settings."""
    OFF = "off"
    LOW = "low"
    MEDIUM = "medium"
    MAX = "max"

class ControlStrategy(Enum):
    """Available ventilation control strategies."""
    MANUAL = auto()  # User-controlled
    CONSTANT = auto()  # Always at same speed
    THRESHOLD = auto()  # Based on CO2 threshold
    SCHEDULED = auto()  # Based on time of day
    MARKOV = auto()  # Reinforcement learning-based (your approach)
    PREDICTIVE = auto()  # Using occupancy prediction

class VentilationSystem:
    """
    Simulates ventilation hardware and control strategies.
    Allows comparison between different control approaches.
    """
    
    def __init__(self, environment_simulator, strategy=ControlStrategy.THRESHOLD):
        """
        Initialize the ventilation system simulator.
        
        Args:
            environment_simulator: Reference to environment simulator
            strategy: Control strategy to use
        """
        self.environment = environment_simulator
        
        # Initial state
        self.mode = VentilationMode.OFF
        self.speed = VentilationSpeed.OFF
        self.strategy = strategy
        
        # Energy consumption tracking (kWh)
        self.total_energy_consumption = 0.0
        self.hourly_consumption = {
            VentilationSpeed.OFF.value: 0.0,
            VentilationSpeed.LOW.value: 0.025,  # Fan 1: 25W = 0.025 kW
            VentilationSpeed.MEDIUM.value: 0.07,  # Fan 2: 70W = 0.07 kW
            VentilationSpeed.MAX.value: 0.095  # Both fans: 95W = 0.095 kW
        }
        
        # Strategy parameters
        self.parameters = {
            # Threshold strategy parameters
            'threshold_strategy': {
                'co2_low': 800,         # Turn off below this
                'co2_medium': 1000,     # Use medium speed above this
                'co2_high': 1200,       # Use max speed above this
                'night_mode_enabled': True,
                'night_mode_start_hour': 23,
                'night_mode_end_hour': 7
            },
            # Scheduled strategy parameters
            'scheduled_strategy': {
                'schedules': [
                    {'start_hour': 7, 'end_hour': 9, 'speed': VentilationSpeed.MEDIUM},  # Morning
                    {'start_hour': 17, 'end_hour': 21, 'speed': VentilationSpeed.MEDIUM},  # Evening
                    {'start_hour': 23, 'end_hour': 7, 'speed': VentilationSpeed.OFF}       # Night
                ]
            },
            # Constant strategy parameters
            'constant_strategy': {
                'speed': VentilationSpeed.LOW
            },
            # Markov strategy parameters - these will be overridden by your actual implementation
            'markov_strategy': {
                'q_values': {},
                'training_complete': False
            }
        }
        
        # Operational history
        self.operation_history = []
        
        # Record initial state
        self._record_state()
        
        logger.info(f"Initialized ventilation system with {strategy.name} strategy")
    
    def _record_state(self):
        """Record current operational state to history."""
        # Calculate energy usage
        energy_used = 0.0
        if self.mode == VentilationMode.MECHANICAL:
            energy_used = self.hourly_consumption[self.speed.value]
        
        # Record state
        self.operation_history.append({
            'timestamp': self.environment.current_time.isoformat(),
            'mode': self.mode.value,
            'speed': self.speed.value,
            'strategy': self.strategy.name,
            'energy_used_kw': energy_used
        })
        
        # Keep history at reasonable size
        if len(self.operation_history) > 10000:
            self.operation_history = self.operation_history[-5000:]
    
    def _update_energy_consumption(self, time_step_minutes):
        """
        Calculate and update energy consumption based on operation mode.
        
        Args:
            time_step_minutes: Time step in minutes
        """
        # Only mechanical ventilation consumes electricity
        if self.mode == VentilationMode.MECHANICAL:
            hours = time_step_minutes / 60.0
            kw_usage = self.hourly_consumption[self.speed.value]
            kwh_used = kw_usage * hours
            
            self.total_energy_consumption += kwh_used
    
    def _apply_manual_strategy(self, sensor_data, occupancy_data, time_step_minutes):
        """Simple pass-through for manual control."""
        # This strategy doesn't change ventilation settings automatically
        # It's controlled externally through set_mode() and set_speed()
        pass
    
    def _apply_constant_strategy(self, sensor_data, occupancy_data, time_step_minutes):
        """Apply constant ventilation at configured speed."""
        constant_speed_str = self.parameters['constant_strategy']['speed'].value \
            if isinstance(self.parameters['constant_strategy']['speed'], Enum) \
            else self.parameters['constant_strategy']['speed']
        constant_speed = VentilationSpeed[constant_speed_str.upper()]
        
        if self.mode != VentilationMode.MECHANICAL or self.speed != constant_speed:
            self.set_mode(VentilationMode.MECHANICAL)
            self.set_speed(constant_speed)
            logger.debug(f"Constant strategy: ventilation set to {constant_speed.value}")
    
    def _apply_threshold_strategy(self, sensor_data, occupancy_data, time_step_minutes):
        """
        Apply CO2 threshold-based ventilation strategy.
        
        This is a common basic strategy that responds to measured CO2 levels.
        """
        params = self.parameters['threshold_strategy']
        co2 = sensor_data['scd41']['co2']
        current_hour = self.environment.current_time.hour
        
        # Check night mode
        night_mode_active = False
        if params['night_mode_enabled']:
            start_hour = params['night_mode_start_hour']
            end_hour = params['night_mode_end_hour']
            
            if start_hour > end_hour:  # Crosses midnight
                night_mode_active = current_hour >= start_hour or current_hour < end_hour
            else:
                night_mode_active = start_hour <= current_hour < end_hour
        
        # If night mode is active, turn off ventilation
        if night_mode_active:
            if self.mode != VentilationMode.OFF:
                self.set_mode(VentilationMode.OFF)
                logger.debug("Threshold strategy: night mode active, ventilation OFF")
            return
        
        # Normal threshold logic
        if co2 >= params['co2_high']:
            if self.mode != VentilationMode.MECHANICAL or self.speed != VentilationSpeed.MAX:
                self.set_mode(VentilationMode.MECHANICAL)
                self.set_speed(VentilationSpeed.MAX)
                logger.debug(f"Threshold strategy: CO2={co2} > {params['co2_high']}, "
                           f"ventilation MAX")
        
        elif co2 >= params['co2_medium']:
            if self.mode != VentilationMode.MECHANICAL or self.speed != VentilationSpeed.MEDIUM:
                self.set_mode(VentilationMode.MECHANICAL)
                self.set_speed(VentilationSpeed.MEDIUM)
                logger.debug(f"Threshold strategy: CO2={co2} > {params['co2_medium']}, "
                           f"ventilation MEDIUM")
        
        elif co2 >= params['co2_low']:
            if self.mode != VentilationMode.MECHANICAL or self.speed != VentilationSpeed.LOW:
                self.set_mode(VentilationMode.MECHANICAL)
                self.set_speed(VentilationSpeed.LOW)
                logger.debug(f"Threshold strategy: CO2={co2} > {params['co2_low']}, "
                           f"ventilation LOW")
        
        else:
            if self.mode != VentilationMode.OFF:
                self.set_mode(VentilationMode.OFF)
                logger.debug(f"Threshold strategy: CO2={co2} < {params['co2_low']}, "
                           f"ventilation OFF")
    
    def _apply_scheduled_strategy(self, sensor_data, occupancy_data, time_step_minutes):
        """
        Apply time-based scheduled ventilation strategy.
        
        This strategy operates ventilation based on time of day, regardless of conditions.
        """
        current_hour = self.environment.current_time.hour
        schedules = self.parameters['scheduled_strategy']['schedules']
        
        # Find applicable schedule
        active_schedule = None
        for schedule in schedules:
            start_hour = schedule['start_hour']
            end_hour = schedule['end_hour']
            
            if start_hour > end_hour:  # Crosses midnight
                if current_hour >= start_hour or current_hour < end_hour:
                    active_schedule = schedule
                    break
            else:
                if start_hour <= current_hour < end_hour:
                    active_schedule = schedule
                    break
        
        # Apply schedule
        if active_schedule:
            target_speed_obj = active_schedule['speed']
            target_speed = target_speed_obj if isinstance(target_speed_obj, VentilationSpeed) else VentilationSpeed[target_speed_obj.upper()]

            if target_speed == VentilationSpeed.OFF:
                if self.mode != VentilationMode.OFF:
                    self.set_mode(VentilationMode.OFF)
                    logger.debug(f"Scheduled strategy: ventilation OFF "
                               f"({current_hour}:00 in {start_hour}:00-{end_hour}:00)")
            else:
                if self.mode != VentilationMode.MECHANICAL or self.speed != target_speed:
                    self.set_mode(VentilationMode.MECHANICAL)
                    self.set_speed(target_speed)
                    logger.debug(f"Scheduled strategy: ventilation {target_speed.value} "
                               f"({current_hour}:00 in {start_hour}:00-{end_hour}:00)")
        else:
            # Default to OFF if no schedule applies
            if self.mode != VentilationMode.OFF:
                self.set_mode(VentilationMode.OFF)
                logger.debug(f"Scheduled strategy: ventilation OFF (no applicable schedule)")
    
    def _apply_markov_strategy(self, sensor_data, occupancy_data, time_step_minutes, 
                              markov_controller=None):
        """
        Apply Markov-based reinforcement learning ventilation strategy.
        
        This is a simplified implementation - the actual implementation will come from
        your MarkovController when integrated in the Simulation class.
        
        Args:
            sensor_data: Environment sensor readings
            occupancy_data: Occupancy information
            time_step_minutes: Simulation time step
            markov_controller: Optional external Markov controller
        """
        # If an external Markov controller is provided, its make_step_decision
        # method will have already acted on the MockPicoManager.
        # This method in VentilationSystem primarily handles other strategies
        # or serves as a fallback/logging point if no external controller is used.
        if markov_controller:
            # The decision was already made by markov_controller.make_step_decision()
            # which updated self.mode and self.speed through MockPicoManager.
            # We just log the current state set by the Markov controller.
            logger.debug(f"Markov strategy (external controller): Mode={self.mode.value}, Speed={self.speed.value}")
            return # Do not apply simplified logic if external controller is used.
        
        # Simplified/Fallback Markov strategy implementation (if no external controller)
        # This part should ideally not be reached if the simulation setup is correct.
        logger.warning("Applying simplified internal Markov strategy - external controller not provided or not active.")
        co2 = sensor_data['scd41']['co2']
        occupants = occupancy_data['total_occupants']
        current_hour = self.environment.current_time.hour
        
        # Basic state categorization
        co2_level = "low"
        if co2 > 1200:
            co2_level = "high"
        elif co2 > 800:
            co2_level = "medium"
        
        time_of_day = "day"
        if 22 <= current_hour or current_hour < 6:
            time_of_day = "night"
        elif 6 <= current_hour < 9:
            time_of_day = "morning"
        elif 17 <= current_hour < 22:
            time_of_day = "evening"
        
        occupancy = "empty" if occupants == 0 else "occupied"
        
        # Simple state representation
        state_key = f"{co2_level}_{time_of_day}_{occupancy}"
        params = self.parameters['markov_strategy']
        
        # Default policy - during training this would be learned
        if not params['training_complete']:
            if state_key not in params['q_values']:
                params['q_values'][state_key] = {}
            
            if occupancy == "empty":
                # Conservative policy when empty - ventilate only if high CO2
                if co2_level == "high":
                    action = "mechanical_low"
                else:
                    action = "off"
            else:
                # Policy when occupied
                if co2_level == "high":
                    action = "mechanical_medium"
                elif co2_level == "medium":
                    action = "mechanical_low"
                else:
                    action = "off"
                
                # Override for nighttime
                if time_of_day == "night":
                    action = "off"
            
            # Record the default policy for this state
            params['q_values'][state_key][action] = 1.0
        else:
            # Use learned policy (would be populated during training)
            if state_key in params['q_values'] and params['q_values'][state_key]:
                # Find best action from Q-values
                best_action = max(params['q_values'][state_key].items(), 
                                 key=lambda x: x[1])[0]
                action = best_action
            else:
                # Fallback if state not seen
                action = "off"
        
        # Apply the selected action
        if action == "off":
            if self.mode != VentilationMode.OFF:
                self.set_mode(VentilationMode.OFF)
                logger.debug(f"Markov strategy (internal fallback): ventilation OFF (state: {state_key})")
        else:
            mode_str, speed_str = action.split('_')
            if mode_str == "mechanical":
                target_speed = VentilationSpeed[speed_str.upper()]
                if self.mode != VentilationMode.MECHANICAL or self.speed != target_speed:
                    self.set_mode(VentilationMode.MECHANICAL)
                    self.set_speed(target_speed)
                    logger.debug(f"Markov strategy (internal fallback): ventilation {speed_str} (state: {state_key})")
    
    def _apply_predictive_strategy(self, sensor_data, occupancy_data, time_step_minutes, 
                                  occupancy_analyzer=None):
        """
        Apply occupancy prediction-based ventilation strategy.
        
        Uses prediction of future occupancy to optimize ventilation timing.
        
        Args:
            sensor_data: Environment sensor readings
            occupancy_data: Current occupancy information
            time_step_minutes: Simulation time step
            occupancy_analyzer: Optional external occupancy analyzer
        """
        # This is a placeholder - in the full simulation this would use your
        # OccupancyPatternAnalyzer for predictions
        co2 = sensor_data['scd41']['co2']
        occupants = occupancy_data['total_occupants']
        current_hour = self.environment.current_time.hour
        current_time = self.environment.current_time
        
        # Simple predictive logic for demonstration
        if occupants == 0:
            # If empty, predict when people will return
            # For demonstration, assume return at 17:00 on weekdays if currently before that
            expected_return = None
            
            if occupancy_analyzer:
                # Use the real analyzer if provided
                expected_return = occupancy_analyzer.get_next_expected_return_time(current_time)
            else:
                # Simple prediction for demonstration
                if current_time.weekday() < 5 and current_hour < 17:
                    # Weekday, expecting return after work
                    expected_return = current_time.replace(hour=17, minute=0)
                    
                    # If it's already past 16:00, return is imminent
                    if current_hour >= 16:
                        expected_return = current_time + timedelta(minutes=60)
            
            if expected_return:
                # Calculate time until return
                time_until_return = (expected_return - current_time).total_seconds() / 3600  # hours
                
                # If return expected within 1 hour and CO2 is high, pre-ventilate
                if time_until_return <= 1 and co2 > 1000:
                    if self.mode != VentilationMode.MECHANICAL or self.speed != VentilationSpeed.MEDIUM:
                        self.set_mode(VentilationMode.MECHANICAL)
                        self.set_speed(VentilationSpeed.MEDIUM)
                        logger.debug(f"Predictive strategy: pre-ventilating for expected return in "
                                   f"{time_until_return:.1f} hours")
                    return
                
                # If nobody expected for a while, use minimal ventilation
                if time_until_return > 3:
                    if co2 > 1200:
                        # Only ventilate if CO2 is very high
                        if self.mode != VentilationMode.MECHANICAL or self.speed != VentilationSpeed.LOW:
                            self.set_mode(VentilationMode.MECHANICAL)
                            self.set_speed(VentilationSpeed.LOW)
                            logger.debug(f"Predictive strategy: minimal ventilation for "
                                       f"extended vacancy (CO2={co2})")
                    else:
                        # Otherwise turn off
                        if self.mode != VentilationMode.OFF:
                            self.set_mode(VentilationMode.OFF)
                            logger.debug(f"Predictive strategy: ventilation OFF for "
                                       f"extended vacancy (CO2={co2})")
                    return
            
            # Default behavior when empty with no prediction
            if co2 > 1100:
                if self.mode != VentilationMode.MECHANICAL or self.speed != VentilationSpeed.LOW:
                    self.set_mode(VentilationMode.MECHANICAL)
                    self.set_speed(VentilationSpeed.LOW)
            else:
                if self.mode != VentilationMode.OFF:
                    self.set_mode(VentilationMode.OFF)
        
        else:
            # When occupied, use CO2 thresholds
            if co2 > 1200:
                if self.mode != VentilationMode.MECHANICAL or self.speed != VentilationSpeed.MAX:
                    self.set_mode(VentilationMode.MECHANICAL)
                    self.set_speed(VentilationSpeed.MAX)
            elif co2 > 1000:
                if self.mode != VentilationMode.MECHANICAL or self.speed != VentilationSpeed.MEDIUM:
                    self.set_mode(VentilationMode.MECHANICAL)
                    self.set_speed(VentilationSpeed.MEDIUM)
            elif co2 > 800:
                if self.mode != VentilationMode.MECHANICAL or self.speed != VentilationSpeed.LOW:
                    self.set_mode(VentilationMode.MECHANICAL)
                    self.set_speed(VentilationSpeed.LOW)
            else:
                if self.mode != VentilationMode.OFF:
                    self.set_mode(VentilationMode.OFF)
    
    def update(self, sensor_data, occupancy_data, time_step_minutes, 
              markov_controller=None, occupancy_analyzer=None):
        """
        Update ventilation system based on current strategy and conditions.
        
        Args:
            sensor_data: Environment sensor readings
            occupancy_data: Occupancy information
            time_step_minutes: Simulation time step
            markov_controller: Optional external Markov controller
            occupancy_analyzer: Optional external occupancy analyzer
            
        Returns:
            dict: Current ventilation state
        """
        # Apply the selected strategy
        # Note: If MarkovController is used, it acts directly on self.mode/self.speed
        # via the MockPicoManager. So, _apply_markov_strategy here mainly logs or
        # handles the case where the external controller isn't active.
        if self.strategy == ControlStrategy.MANUAL:
            self._apply_manual_strategy(sensor_data, occupancy_data, time_step_minutes)
        
        elif self.strategy == ControlStrategy.CONSTANT:
            self._apply_constant_strategy(sensor_data, occupancy_data, time_step_minutes)
        
        elif self.strategy == ControlStrategy.THRESHOLD:
            self._apply_threshold_strategy(sensor_data, occupancy_data, time_step_minutes)
        
        elif self.strategy == ControlStrategy.SCHEDULED:
            self._apply_scheduled_strategy(sensor_data, occupancy_data, time_step_minutes)
        
        elif self.strategy == ControlStrategy.MARKOV:
            self._apply_markov_strategy(sensor_data, occupancy_data, time_step_minutes, 
                                       markov_controller)
        
        elif self.strategy == ControlStrategy.PREDICTIVE:
            self._apply_predictive_strategy(sensor_data, occupancy_data, time_step_minutes,
                                          occupancy_analyzer)
        
        # Update energy consumption
        self._update_energy_consumption(time_step_minutes)
        
        # Record current state
        self._record_state()
        
        return self.get_current_state()
    
    def set_mode(self, mode: VentilationMode):
        """
        Set ventilation mode.
        
        Args:
            mode: VentilationMode enum value
        """
        if not isinstance(mode, VentilationMode):
            try:
                mode = VentilationMode(mode)
            except ValueError:
                logger.error(f"Invalid ventilation mode value: {mode}")
                return

        old_mode = self.mode
        self.mode = mode
        
        # Automatically set speed to OFF when mode is OFF
        if mode == VentilationMode.OFF:
            self.speed = VentilationSpeed.OFF
        
        if old_mode != mode:
            logger.info(f"Ventilation mode changed: {old_mode.value} → {mode.value}")
    
    def set_speed(self, speed: VentilationSpeed):
        """
        Set ventilation speed.
        
        Args:
            speed: VentilationSpeed enum value
        """
        if not isinstance(speed, VentilationSpeed):
            try:
                speed = VentilationSpeed(speed)
            except ValueError:
                logger.error(f"Invalid ventilation speed value: {speed}")
                return
        
        old_speed = self.speed
        self.speed = speed
        
        if old_speed != speed:
            logger.info(f"Ventilation speed changed: {old_speed.value} → {speed.value}")
    
    def set_strategy(self, strategy):
        """
        Change the control strategy.
        
        Args:
            strategy: ControlStrategy enum value
        """
        if not isinstance(strategy, ControlStrategy):
            if isinstance(strategy, str):
                strategy = ControlStrategy[strategy.upper()]
            else:
                strategy = ControlStrategy(strategy)
        
        old_strategy = self.strategy
        self.strategy = strategy
        
        logger.info(f"Control strategy changed: {old_strategy.name} → {strategy.name}")
    
    def get_current_state(self):
        """
        Get current ventilation system state.
        
        Returns:
            dict: Current state
        """
        return {
            'timestamp': self.environment.current_time.isoformat(),
            'mode': self.mode.value,
            'speed': self.speed.value,
            'strategy': self.strategy.name,
            'total_energy_consumption': round(self.total_energy_consumption, 4)
        }
    
    def get_operation_history(self):
        """
        Get operational history.
        
        Returns:
            list: Historical operation data
        """
        return self.operation_history
    
    def reset_energy_consumption(self):
        """Reset energy consumption counter to zero."""
        old_consumption = self.total_energy_consumption
        self.total_energy_consumption = 0.0
        logger.info(f"Energy consumption reset from {old_consumption:.2f} kWh to 0.00 kWh")
    
    def reset(self):
        """Reset ventilation system to initial state."""
        self.mode = VentilationMode.OFF
        self.speed = VentilationSpeed.OFF
        self.total_energy_consumption = 0.0
        
        logger.info(f"Ventilation system reset to OFF state")