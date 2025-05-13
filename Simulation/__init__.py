"""
Simulation package for ventilation control system.
Provides a modular framework for training and evaluating ventilation controllers.
"""

# Import all simulation components for easier access
from Simulation.simulation_environment import SimulationEnvironment
from Simulation.occupant_behavior_model import OccupantBehaviorModel
from Simulation.reward_function import RewardFunction
from Simulation.baseline_controllers import (
    BaselineController, 
    RuleBasedController, 
    TimerBasedController, 
    OracleRuleBasedController
)
from Simulation.simulation_managers import SimulationDataManager, SimulationPicoManager
from Simulation.metrics_collector import MetricsCollector
from Simulation.simulation_runner import SimulationRunner
from Simulation.utils import (
    interpolate_missing_values,
    smooth_time_series,
    calculate_daily_statistics,
    extract_state_from_action_history,
    calculate_energy_efficiency,
    create_hourly_profile,
    merge_config_files
)

# Package metadata
__version__ = '1.0.0'
__author__ = 'Robertas Gaigalas'
__all__ = [
    'SimulationEnvironment',
    'OccupantBehaviorModel',
    'RewardFunction',
    'BaselineController',
    'RuleBasedController',
    'TimerBasedController',
    'OracleRuleBasedController',
    'SimulationDataManager',
    'SimulationPicoManager',
    'MetricsCollector',
    'SimulationRunner'
]