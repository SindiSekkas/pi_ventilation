# simulation/tests/test_simulation.py
"""
Unit tests for the ventilation simulation components.
"""
import unittest
import os
import tempfile
import json
from datetime import datetime, timedelta

# Import simulation components
from Simulation.simulation_environment import SimulationEnvironment
from Simulation.occupant_behavior_model import OccupantBehaviorModel
from Simulation.reward_function import RewardFunction
from Simulation.baseline_controllers import RuleBasedController, TimerBasedController
from Simulation.simulation_managers import SimulationDataManager, SimulationPicoManager
from Simulation.utils import (
    interpolate_missing_values,
    smooth_time_series,
    merge_config_files
)

class TestSimulationEnvironment(unittest.TestCase):
    """Test cases for SimulationEnvironment."""
    
    def setUp(self):
        """Set up test environment."""
        self.env = SimulationEnvironment()
    
    def test_initialization(self):
        """Test environment initialization."""
        self.assertIsNotNone(self.env)
        state = self.env.get_current_state()
        self.assertIn("co2_ppm", state)
        self.assertIn("temperature_c", state)
        self.assertIn("humidity_percent", state)
    
    def test_reset(self):
        """Test environment reset."""
        # Make a change
        self.env.co2_ppm = 2000
        self.env.temperature_c = 25.0
        
        # Reset
        state = self.env.reset()
        
        # Check reset values
        self.assertNotEqual(state["co2_ppm"], 2000)
        self.assertNotEqual(state["temperature_c"], 25.0)
    
    def test_step(self):
        """Test environment step function."""
        # Get initial state
        initial_state = self.env.get_current_state()
        
        # Take a step with no occupants and ventilation off
        new_state, energy = self.env.step("off", 0, 0, 5)
        
        # CO2 should decay toward external level
        self.assertLessEqual(new_state["co2_ppm"], initial_state["co2_ppm"])
        
        # Take a step with occupants
        new_state, energy = self.env.step("off", 2, 0, 5)
        
        # CO2 should increase
        self.assertGreater(new_state["co2_ppm"], initial_state["co2_ppm"])
        
        # Test ventilation impact
        self.env.co2_ppm = 1500  # Set high CO2
        high_co2_state = self.env.get_current_state()
        
        # Step with ventilation on
        new_state, energy = self.env.step("max", 0, 0, 5)
        
        # CO2 should decrease faster than natural decay
        decay_step1 = high_co2_state["co2_ppm"] - new_state["co2_ppm"]
        
        # Reset and try without ventilation
        self.env.co2_ppm = 1500
        new_state, energy = self.env.step("off", 0, 0, 5)
        decay_step2 = 1500 - new_state["co2_ppm"]
        
        # Decay with ventilation should be faster
        self.assertGreater(decay_step1, decay_step2)
    
    def test_energy_consumption(self):
        """Test energy consumption tracking."""
        # Test with ventilation off
        _, energy_off = self.env.step("off", 0, 0, 60)
        self.assertEqual(energy_off, 0)
        
        # Test with ventilation on
        _, energy_low = self.env.step("low", 0, 0, 60)
        self.assertGreater(energy_low, 0)
        
        # Test with higher speed
        _, energy_max = self.env.step("max", 0, 0, 60)
        self.assertGreater(energy_max, energy_low)

class TestOccupantBehaviorModel(unittest.TestCase):
    """Test cases for OccupantBehaviorModel."""
    
    def setUp(self):
        """Set up test environment."""
        # Use minimal config for faster tests
        self.config = {
            "SIMULATION_DAYS_TRAINING": 2,
            "SIMULATION_DAYS_TESTING": 1,
            "NUM_OCCUPANTS": 2
        }
        self.model = OccupantBehaviorModel(self.config)
    
    def test_initialization(self):
        """Test occupant model initialization."""
        self.assertIsNotNone(self.model)
        self.assertEqual(len(self.model.training_schedule) > 0, True)
        self.assertEqual(len(self.model.testing_schedule) > 0, True)
    
    def test_get_occupancy_status(self):
        """Test getting occupancy status at a specific time."""
        now = datetime.now()
        awake, sleeping = self.model.get_occupancy_status_at_time(now)
        self.assertIsInstance(awake, int)
        self.assertIsInstance(sleeping, int)
    
    def test_timestep_progression(self):
        """Test timestep progression."""
        # Reset to beginning of training
        self.model.reset(mode="training")
        
        # Get first timestep
        first_step = self.model.get_next_timestep(mode="training")
        
        # Should have timestamp and occupancy information
        self.assertIn("timestamp", first_step)
        self.assertIn("awake", first_step)
        self.assertIn("sleeping", first_step)
        
        # Get second timestep
        second_step = self.model.get_next_timestep(mode="training")
        
        # Second should be after first
        self.assertGreater(second_step["timestamp"], first_step["timestamp"])
    
    def test_peak_ahead(self):
        """Test peeking ahead in the schedule."""
        # Reset to beginning
        self.model.reset(mode="training")
        
        # Get current timestep
        current = self.model.get_next_timestep(mode="training")
        
        # Peek ahead 1 step
        peek = self.model.peek_ahead(steps=1, mode="training")
        
        # Peek should be after current
        self.assertGreater(peek["timestamp"], current["timestamp"])
        
        # Current index should still be the same
        next_step = self.model.get_next_timestep(mode="training")
        self.assertEqual(next_step["timestamp"], peek["timestamp"])

class TestRewardFunction(unittest.TestCase):
    """Test cases for RewardFunction."""
    
    def setUp(self):
        """Set up test environment."""
        self.reward_fn = RewardFunction()
    
    def test_co2_reward(self):
        """Test CO2 comfort rewards/penalties."""
        # Create test states
        comfortable_state = {"co2_ppm": 800, "temperature_c": 22.0, "humidity_percent": 40.0}
        uncomfortable_state = {"co2_ppm": 1500, "temperature_c": 22.0, "humidity_percent": 40.0}
        
        # Test preferences
        preferences = {"co2_threshold": 1000, "temp_min": 20.0, "temp_max": 24.0, 
                      "humidity_min": 30.0, "humidity_max": 60.0}
        
        # Calculate rewards
        reward1, components1 = self.reward_fn.calculate_reward(
            comfortable_state, preferences, "off", 0, "off"
        )
        
        reward2, components2 = self.reward_fn.calculate_reward(
            uncomfortable_state, preferences, "off", 0, "off"
        )
        
        # Comfortable state should have higher reward
        self.assertGreater(reward1, reward2)
        self.assertGreater(components1["co2"], components2["co2"])
    
    def test_energy_penalty(self):
        """Test energy consumption penalties."""
        state = {"co2_ppm": 800, "temperature_c": 22.0, "humidity_percent": 40.0}
        preferences = {"co2_threshold": 1000, "temp_min": 20.0, "temp_max": 24.0, 
                      "humidity_min": 30.0, "humidity_max": 60.0}
        
        # Calculate rewards with different energy consumption
        reward1, components1 = self.reward_fn.calculate_reward(
            state, preferences, "off", 0, "off"
        )
        
        reward2, components2 = self.reward_fn.calculate_reward(
            state, preferences, "max", 50, "off"
        )
        
        # Higher energy consumption should have lower reward
        self.assertLess(components2["energy"], components1["energy"])
        self.assertLess(reward2, reward1)
    
    def test_switching_penalty(self):
        """Test action switching penalties."""
        state = {"co2_ppm": 800, "temperature_c": 22.0, "humidity_percent": 40.0}
        preferences = {"co2_threshold": 1000, "temp_min": 20.0, "temp_max": 24.0, 
                      "humidity_min": 30.0, "humidity_max": 60.0}
        
        # Calculate rewards with and without switching
        reward1, components1 = self.reward_fn.calculate_reward(
            state, preferences, "off", 0, "off"  # No switch
        )
        
        reward2, components2 = self.reward_fn.calculate_reward(
            state, preferences, "low", 0, "off"  # Switch from off to low
        )
        
        # Switching should have lower reward
        self.assertLess(components2["switching"], components1["switching"])
        self.assertLess(reward2, reward1)

class TestBaselineControllers(unittest.TestCase):
    """Test cases for baseline controllers."""
    
    def setUp(self):
        """Set up test environment."""
        self.rule_based = RuleBasedController()
        self.timer_based = TimerBasedController()
    
    def test_rule_based_controller(self):
        """Test rule-based controller logic."""
        # Low CO2 level
        sensor_data_low = {"co2_ppm": 700, "temperature_c": 22.0, "occupants": 2}
        action_low = self.rule_based.decide_action(sensor_data_low)
        self.assertEqual(action_low, "off")
        
        # Medium CO2 level
        sensor_data_med = {"co2_ppm": 900, "temperature_c": 22.0, "occupants": 2}
        action_med = self.rule_based.decide_action(sensor_data_med)
        self.assertEqual(action_med, "low")
        
        # High CO2 level
        sensor_data_high = {"co2_ppm": 1300, "temperature_c": 22.0, "occupants": 2}
        action_high = self.rule_based.decide_action(sensor_data_high)
        self.assertEqual(action_high, "max")
    
    def test_timer_based_controller(self):
        """Test timer-based controller logic."""
        # Create sensor data (not used by timer controller)
        sensor_data = {"co2_ppm": 700, "temperature_c": 22.0, "occupants": 2}
        
        # Test with different times
        # Note: This is simplified and would need mocking datetime.now() for proper testing
        action = self.timer_based.decide_action(sensor_data)
        self.assertIn(action, ["off", "low", "medium", "max"])
        
        # Test scheduling structure
        day_schedule = self.timer_based.config["schedule"].get(0, [])  # Monday
        self.assertIsInstance(day_schedule, list)
        if day_schedule:
            # Each entry should have start hour, end hour, action
            self.assertEqual(len(day_schedule[0]), 3)

class TestSimulationDataManager(unittest.TestCase):
    """Test cases for SimulationDataManager."""
    
    def setUp(self):
        """Set up test environment."""
        self.data_manager = SimulationDataManager()
    
    def test_initialization(self):
        """Test data manager initialization."""
        self.assertIsNotNone(self.data_manager)
        self.assertIn("scd41", self.data_manager.latest_data)
        self.assertIn("room", self.data_manager.latest_data)
    
    def test_update_sensor_data(self):
        """Test updating sensor data."""
        env_state = {
            "co2_ppm": 900,
            "temperature_c": 23.5,
            "humidity_percent": 45.0
        }
        
        updated_data = self.data_manager.update_sensor_data(env_state)
        
        self.assertEqual(updated_data["scd41"]["co2"], 900)
        self.assertEqual(updated_data["scd41"]["temperature"], 23.5)
        self.assertEqual(updated_data["scd41"]["humidity"], 45.0)
    
    def test_update_room_data(self):
        """Test updating room data."""
        updated_data = self.data_manager.update_room_data(
            occupants=3,
            ventilated=True,
            ventilation_speed="medium"
        )
        
        self.assertEqual(updated_data["occupants"], 3)
        self.assertEqual(updated_data["ventilated"], True)
        self.assertEqual(updated_data["ventilation_speed"], "medium")

class TestSimulationPicoManager(unittest.TestCase):
    """Test cases for SimulationPicoManager."""
    
    def setUp(self):
        """Set up test environment."""
        self.pico_manager = SimulationPicoManager()
    
    def test_initialization(self):
        """Test pico manager initialization."""
        self.assertIsNotNone(self.pico_manager)
        self.assertEqual(self.pico_manager.ventilation_status, False)
        self.assertEqual(self.pico_manager.ventilation_speed, "off")
    
    def test_ventilation_control(self):
        """Test ventilation control."""
        # Turn on ventilation
        success = self.pico_manager.control_ventilation("on", "low")
        self.assertTrue(success)
        self.assertTrue(self.pico_manager.ventilation_status)
        self.assertEqual(self.pico_manager.ventilation_speed, "low")
        
        # Turn off ventilation
        success = self.pico_manager.control_ventilation("off")
        self.assertTrue(success)
        self.assertFalse(self.pico_manager.ventilation_status)
        self.assertEqual(self.pico_manager.ventilation_speed, "off")
        
        # Test invalid actions
        success = self.pico_manager.control_ventilation("invalid")
        self.assertFalse(success)
        
        success = self.pico_manager.control_ventilation("on", "invalid")
        self.assertFalse(success)
    
    def test_get_status_methods(self):
        """Test status retrieval methods."""
        # Set status
        self.pico_manager.control_ventilation("on", "medium")
        
        # Get status
        status = self.pico_manager.get_ventilation_status()
        speed = self.pico_manager.get_ventilation_speed()
        
        self.assertTrue(status)
        self.assertEqual(speed, "medium")

class TestUtils(unittest.TestCase):
    """Test cases for utility functions."""
    
    def test_interpolate_missing_values(self):
        """Test interpolation of missing values."""
        data = [1.0, 2.0, None, None, 5.0]
        result = interpolate_missing_values(data)
        
        # Check result length
        self.assertEqual(len(result), len(data))
        
        # Check missing values were filled
        self.assertIsNotNone(result[2])
        self.assertIsNotNone(result[3])
        
        # Check interpolation is reasonable
        self.assertTrue(2.0 < result[2] < 5.0)
        self.assertTrue(2.0 < result[3] < 5.0)
        
        # Check that large gaps aren't interpolated
        data = [1.0, None, None, None, None, 6.0]
        result = interpolate_missing_values(data, max_gap=3)
        
        # Check that the gap wasn't fully interpolated (> max_gap)
        self.assertTrue(np.isnan(result[3]) or np.isnan(result[4]))
    
    def test_smooth_time_series(self):
        """Test time series smoothing."""
        data = [1.0, 5.0, 3.0, 4.0, 2.0]
        result = smooth_time_series(data, window_size=3)
        
        # Check result length
        self.assertEqual(len(result), len(data))
        
        # Check smoothing effect - peaks should be reduced
        self.assertLess(result[1], data[1])
        
        # Check boundary handling
        self.assertEqual(result[0], data[0])
    
    def test_merge_config_files(self):
        """Test configuration merging."""
        base_config = {
            "param1": "value1",
            "param2": 123,
            "nested": {
                "inner1": True,
                "inner2": "original"
            }
        }
        
        override_config = {
            "param2": 456,
            "param3": "new_value",
            "nested": {
                "inner2": "modified"
            }
        }
        
        merged = merge_config_files(base_config, override_config)
        
        # Check merged values
        self.assertEqual(merged["param1"], "value1")  # Unchanged
        self.assertEqual(merged["param2"], 456)  # Overridden
        self.assertEqual(merged["param3"], "new_value")  # Added
        self.assertEqual(merged["nested"]["inner1"], True)  # Nested unchanged
        self.assertEqual(merged["nested"]["inner2"], "modified")  # Nested overridden

if __name__ == '__main__':
    unittest.main()