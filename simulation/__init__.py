# simulation/__init__.py
"""
Simulation package for adaptive ventilation system.
Provides components to evaluate various control strategies under realistic conditions.
"""
from simulation.environment import EnvironmentSimulator
from simulation.occupants import OccupantBehaviorModel
from simulation.ventilation import VentilationSystem, ControlStrategy
from simulation.simulation import Simulation, run_complete_simulation

__version__ = "1.0.0"

# Check availability of real system components
try:
    from control.markov_controller import MarkovController
    from predictive.occupancy_pattern_analyzer import OccupancyPatternAnalyzer
    from predictive.adaptive_sleep_analyzer import AdaptiveSleepAnalyzer
    REAL_COMPONENTS_AVAILABLE = True
except ImportError:
    REAL_COMPONENTS_AVAILABLE = False