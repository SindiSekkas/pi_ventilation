"""
Modified version of the ventilation simulation code with monkey patching
to fix the metrics initialization issue.
"""
import os
import sys
import json
import logging
import glob
import pandas as pd
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("simulation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ventilation_simulation")

# Import simulator and controllers
from simulation.simulator import VentilationSimulator, EnvironmentState
from simulation.controllers import OnOffController, PIDController, FuzzyController

# Import the user's Markov controller
from control.markov_controller import MarkovController

# =========== CRITICAL FIX: MONKEY PATCH VentilationSimulator =============
# Store the original reset method
original_reset = VentilationSimulator.reset

# Define a patched reset method that preserves metrics
def patched_reset(self, initial_state=None):
    # Save metrics before reset
    saved_metrics = {}
    if hasattr(self, 'metrics'):
        saved_metrics = self.metrics.copy()
    
    # Call original reset method
    result = original_reset(self, initial_state)
    
    # Restore saved metrics
    self.metrics = saved_metrics
    
    return result

# Apply the monkey patch
VentilationSimulator.reset = patched_reset
# ========================================================================

class SimulationRunner:
    """Manages the end-to-end ventilation simulation process."""
    
    def __init__(self, output_dir="simulation_results", real_data_dir="data/csv"):
        """Initialize the simulation runner."""
        self.output_dir = output_dir
        self.real_data_dir = real_data_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Simulation parameters
        self.sim_days = 90  # 3 months
        self.time_step_minutes = 2
        
        # Paths
        self.sim_data_path = os.path.join(output_dir, "simulated_data.csv")
        self.markov_model_dir = os.path.join(output_dir, "markov_model")
        os.makedirs(self.markov_model_dir, exist_ok=True)
    
    def combine_real_data(self):
        """
        Combine all CSV files from the real data directory into a single simulation-ready file.
        
        Returns:
            bool: Success indicator
        """
        try:
            logger.info(f"Searching for CSV files in {self.real_data_dir}")
            
            # Get all CSV files
            csv_files = glob.glob(os.path.join(self.real_data_dir, "*.csv"))
            if not csv_files:
                logger.error(f"No CSV files found in {self.real_data_dir}")
                return False
            
            logger.info(f"Found {len(csv_files)} CSV files")
            
            # Read and combine all files
            all_data = []
            for csv_file in csv_files:
                try:
                    # Handle different possible CSV formats
                    df = pd.read_csv(csv_file)
                    all_data.append(df)
                except Exception as e:
                    logger.warning(f"Error processing {csv_file}: {e}")
            
            if not all_data:
                logger.error("No valid data found in CSV files")
                return False
            
            # Combine all dataframes
            combined_df = pd.concat(all_data, ignore_index=True)
            
            # Ensure we have the required columns for simulation
            required_columns = [
                'Timestamp', 'CO2', 'SCD41_Temperature', 'Humidity', 
                'Occupants', 'ventilated', 'ventilation_speed'
            ]
            
            # Rename columns if necessary to match the expected format
            column_mapping = {
                'Timestamp': 'timestamp',
                'CO2': 'co2',
                'SCD41_Temperature': 'temperature',
                'Humidity': 'humidity',
                'Occupants': 'occupants',
                'ventilated': 'ventilated',
                'ventilation_speed': 'ventilation_action'  # Important: convert to action for the controller
            }
            
            # Check if columns exist (case-insensitive)
            columns_lower = {col.lower(): col for col in combined_df.columns}
            missing_columns = []
            for req_col in required_columns:
                if req_col.lower() not in columns_lower:
                    missing_columns.append(req_col)
            
            if missing_columns:
                logger.warning(f"Missing columns: {missing_columns}. Will attempt to proceed anyway.")
            
            # Rename columns to match simulation format
            for original, new_name in column_mapping.items():
                orig_col = columns_lower.get(original.lower())
                if orig_col and orig_col in combined_df.columns:
                    combined_df.rename(columns={orig_col: new_name}, inplace=True)
            
            # Handle missing required columns
            if 'timestamp' not in combined_df.columns:
                # Create timestamp column using date/time index
                combined_df['timestamp'] = pd.date_range(
                    start=datetime.now(), 
                    periods=len(combined_df), 
                    freq='5min'
                ).strftime('%Y-%m-%d %H:%M:%S')
            
            if 'co2' not in combined_df.columns:
                combined_df['co2'] = 800  # Default CO2 value
            
            if 'temperature' not in combined_df.columns:
                combined_df['temperature'] = 22.0  # Default temperature
            
            if 'humidity' not in combined_df.columns:
                combined_df['humidity'] = 50.0  # Default humidity
            
            if 'occupants' not in combined_df.columns:
                combined_df['occupants'] = 1  # Default occupancy
            
            # Map boolean string values if needed
            if 'ventilated' in combined_df.columns:
                # Convert to boolean or 0/1
                combined_df['ventilated'] = combined_df['ventilated'].apply(
                    lambda x: 1 if str(x).lower() in ['true', '1', 't', 'yes', 'y'] else 0
                )
            else:
                combined_df['ventilated'] = 0  # Default ventilation state
            
            # Convert ventilation_action to expected format (off/low/medium/max)
            if 'ventilation_action' in combined_df.columns:
                # Handle potential string representations of boolean
                combined_df['ventilation_action'] = combined_df['ventilation_action'].apply(
                    lambda x: 'off' if str(x).lower() in ['false', '0', 'f', 'no', 'n', 'off'] else 
                             (x if str(x).lower() in ['low', 'medium', 'max'] else 'off')
                )
            else:
                # Derive from ventilated state if available
                if 'ventilated' in combined_df.columns:
                    combined_df['ventilation_action'] = combined_df['ventilated'].apply(
                        lambda x: 'medium' if x == 1 else 'off'
                    )
                else:
                    combined_df['ventilation_action'] = 'off'  # Default ventilation action
            
            # Sort by timestamp
            if 'timestamp' in combined_df.columns:
                try:
                    combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'])
                    combined_df.sort_values('timestamp', inplace=True)
                    combined_df['timestamp'] = combined_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
                except Exception as e:
                    logger.warning(f"Error sorting by timestamp: {e}")
            
            # Save combined data to simulation data path
            combined_df.to_csv(self.sim_data_path, index=False)
            logger.info(f"Combined data saved to {self.sim_data_path} with {len(combined_df)} records")
            
            return True
        except Exception as e:
            logger.error(f"Error combining real data: {e}")
            return False
    
    def train_markov_model(self):
        """Train the Markov controller using simulated data."""
        logger.info("Training Markov model...")
        
        try:
            # Use the updated training script with Q-learning
            training_script_path = os.path.join(os.path.dirname(__file__), "q_train_markov_model.py")
            
            if not os.path.exists(training_script_path):
                logger.error(f"Training script not found at {training_script_path}")
                return False
            
            # Execute training script with appropriate arguments
            import subprocess
            subprocess.run([
                sys.executable, 
                training_script_path,
                "--csv", self.sim_data_path,
                "--model-dir", self.markov_model_dir,
                "--epochs", "15",
                "--learning-rate", "0.4",
                "--exploration-rate", "0.4",
                "--epsilon-decay", "0.95"
            ], check=True)
            
            # Verify model was created
            model_path = os.path.join(self.markov_model_dir, "markov_model.json")
            if not os.path.exists(model_path):
                raise Exception(f"Training failed - no model file generated at {model_path}")
            
            logger.info(f"Successfully trained Markov model in {self.markov_model_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error in Markov model training: {e}")
            return False
    
    def setup_real_markov_controller(self):
        """Create a real MarkovController instance with the trained model."""
        # Create mock components needed by MarkovController
        class MockDataManager:
            def __init__(self):
                self.latest_data = {
                    "scd41": {"co2": 600, "temperature": 21.0, "humidity": 45.0},
                    "room": {"occupants": 0, "ventilated": False, "ventilation_speed": "off"},
                    "timestamp": datetime.now().isoformat()
                }
        
        class MockPicoManager:
            def get_ventilation_status(self):
                return False
            def get_ventilation_speed(self):
                return "off"
            def control_ventilation(self, state, speed=None):
                return True
        
        # Initialize controller with mock components
        controller = MarkovController(
            data_manager=MockDataManager(),
            pico_manager=MockPicoManager(),
            model_dir=self.markov_model_dir,
            scan_interval=60
        )
        
        # Load the trained Q-values
        model_path = os.path.join(self.markov_model_dir, "markov_model.json")
        controller.load_q_values(model_path)
        
        # Wrap the controller with an adapter compatible with our simulation interface
        class MarkovControllerAdapter:
            def __init__(self, controller):
                self.controller = controller
                self.last_action = "off"
            
            def decide_action(self, data):
                # Update controller's data manager with simulation data
                self.controller.data_manager.latest_data = data
                
                # Evaluate the state based on the new data BEFORE deciding action
                self.controller.current_state = self.controller._evaluate_state()
                
                # Call the controller's decide action method
                action = self.controller._decide_action()
                
                # Record the action
                self.last_action = action
                
                return action
        
        return MarkovControllerAdapter(controller)
    
    def run_simulation(self):
        """Run the complete simulation with all controllers."""
        # 1. Create simulator
        simulator = VentilationSimulator(
            output_dir=self.output_dir,
            time_step_minutes=self.time_step_minutes
        )
        
        # 2. Load occupancy data
        if not os.path.exists(self.sim_data_path):
            logger.error(f"Simulation data file not found at {self.sim_data_path}")
            return False
            
        if not simulator.load_occupancy_pattern(self.sim_data_path):
            logger.error("Failed to load occupancy data. Exiting.")
            return False
        
        # 3. Create controllers
        controllers = {
            "OnOff": OnOffController(
                co2_high_threshold=1000, 
                co2_low_threshold=800,
                temp_high_threshold=25.0
            ),
            "PID": PIDController(
                co2_setpoint=800.0,
                temp_setpoint=22.0
            ),
            "Fuzzy": FuzzyController(),
            "Markov": self.setup_real_markov_controller()
        }
        
        # 4. Run simulation 
        logger.info("Starting simulation with all controllers...")
        
        # Initialize metrics for all controllers
        for controller_name in controllers:
            simulator.metrics[controller_name] = {
                "total_energy": 0.0,
                "avg_co2": 0.0,
                "time_co2_high": 0,
                "avg_temp": 0.0,
                "time_outside_comfort": 0,
                "comfort_score": 0.0,
                "high_co2_events": 0,
                "vent_cycles": 0,
                "vent_time": 0,
            }
        
        # Now run simulation
        metrics = simulator.run_simulation(controllers, days=self.sim_days)
        
        # 5. Generate visualizations
        self.visualize_results(metrics, simulator.controller_logs)
        
        return True
    
    def visualize_results(self, metrics, controller_logs):
        """Create visualizations comparing controller performance."""
        logger.info("Generating performance visualizations...")
        
        # 1. Create performance comparison bar chart
        try:
            # Convert metrics to DataFrame for easier plotting
            metrics_df = pd.DataFrame.from_dict(metrics, orient='index')
            
            # Create figure with subplots
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle('Ventilation Controller Performance Comparison', fontsize=16)
            
            # Energy consumption
            sns.barplot(x=metrics_df.index, y='total_energy', data=metrics_df, ax=axes[0, 0])
            axes[0, 0].set_title('Total Energy Consumption (Wh)')
            axes[0, 0].set_ylabel('Energy (Wh)')
            axes[0, 0].set_xlabel('')
            
            # Average CO2 levels
            sns.barplot(x=metrics_df.index, y='avg_co2', data=metrics_df, ax=axes[0, 1])
            axes[0, 1].set_title('Average CO2 Levels (ppm)')
            axes[0, 1].set_ylabel('CO2 (ppm)')
            axes[0, 1].set_xlabel('')
            
            # Time outside comfort zone
            sns.barplot(x=metrics_df.index, y='time_outside_comfort', data=metrics_df, ax=axes[1, 0])
            axes[1, 0].set_title('Time Outside Comfort Zone (minutes)')
            axes[1, 0].set_ylabel('Time (minutes)')
            axes[1, 0].set_xlabel('')
            
            # Comfort score (lower is better)
            sns.barplot(x=metrics_df.index, y='comfort_score', data=metrics_df, ax=axes[1, 1])
            axes[1, 1].set_title('Discomfort Score (lower is better)')
            axes[1, 1].set_ylabel('Score')
            axes[1, 1].set_xlabel('')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'performance_comparison.png'))
            
            # 2. Generate time series plots for sample period
            # Choose a representative 3-day period to visualize
            self._create_time_series_plots(controller_logs)
            
            logger.info("Visualizations saved to output directory")
            
        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")
    
    def _create_time_series_plots(self, controller_logs):
        """Create time series plots of controller behavior over a sample period."""
        try:
            # Convert logs to DataFrame
            import pandas as pd
            import matplotlib.pyplot as plt
            import numpy as np
            
            dfs = {}
            for controller_name, logs in controller_logs.items():
                if not logs:  # Skip if empty
                    logger.warning(f"No logs for controller: {controller_name}")
                    continue
                    
                df = pd.DataFrame(logs)
                
                # Ensure timestamp conversion is successful
                try:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df.set_index('timestamp', inplace=True)
                    dfs[controller_name] = df
                except Exception as e:
                    logger.error(f"Error processing logs for {controller_name}: {e}")
                    continue
            
            # Select a 3-day period for visualization (day 1-4 if data is less than 30 days)
            for controller_name, df in dfs.items():
                if len(df) == 0:
                    logger.warning(f"No data for controller: {controller_name}")
                    continue
                    
                # Find the timestamp for visualization
                if len(df) > 0:
                    try:
                        # If we have at least 30 days of data, use day 30-33
                        # Otherwise use day 1-4
                        days_of_data = (df.index.max() - df.index.min()).days
                        offset_days = min(days_of_data - 4, 30) if days_of_data > 4 else 0
                        
                        start_time = df.index.min() + timedelta(days=offset_days)
                        end_time = start_time + timedelta(days=3)
                        
                        # Select data in the range
                        mask = (df.index >= start_time) & (df.index <= end_time)
                        sample_df = df.loc[mask]
                        
                        # Skip if no data in the range
                        if len(sample_df) == 0:
                            logger.warning(f"No data in sample period for {controller_name}")
                            continue
                        
                        # Create figure with subplots
                        fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)
                        fig.suptitle(f'{controller_name} Controller: 3-Day Sample', fontsize=16)
                        
                        # CO2 plot
                        axes[0].plot(sample_df.index, sample_df['co2'], 'b-')
                        axes[0].axhline(y=800, color='g', linestyle='--', label='Target')
                        axes[0].axhline(y=1200, color='r', linestyle='--', label='High')
                        axes[0].set_ylabel('CO2 (ppm)')
                        axes[0].set_title('CO2 Levels')
                        axes[0].legend()
                        axes[0].grid(True)
                        
                        # Temperature plot
                        axes[1].plot(sample_df.index, sample_df['temperature'], 'r-')
                        axes[1].axhline(y=22, color='g', linestyle='--', label='Target')
                        axes[1].set_ylabel('Temp (°C)')
                        axes[1].set_title('Temperature')
                        axes[1].legend()
                        axes[1].grid(True)
                        
                        # Ventilation action plot
                        # Map actions to numeric values for plotting
                        action_map = {'off': 0, 'low': 1, 'medium': 2, 'max': 3}
                        
                        # Handle missing 'ventilation_speed' column
                        if 'ventilation_speed' in sample_df.columns:
                            numeric_actions = sample_df['ventilation_speed'].map(action_map)
                            
                            axes[2].step(sample_df.index, numeric_actions, 'g-', where='post')
                            axes[2].set_yticks([0, 1, 2, 3])
                            axes[2].set_yticklabels(['Off', 'Low', 'Medium', 'Max'])
                            axes[2].set_ylabel('Ventilation')
                            axes[2].set_title('Ventilation Action')
                            axes[2].grid(True)
                        else:
                            axes[2].text(0.5, 0.5, 'No ventilation data available', 
                                       horizontalalignment='center',
                                       transform=axes[2].transAxes)
                        
                        # Occupancy plot
                        if 'occupants' in sample_df.columns:
                            axes[3].step(sample_df.index, sample_df['occupants'], 'k-', where='post')
                            axes[3].set_ylabel('Occupants')
                            axes[3].set_title('Room Occupancy')
                            axes[3].grid(True)
                        else:
                            axes[3].text(0.5, 0.5, 'No occupancy data available', 
                                       horizontalalignment='center',
                                       transform=axes[3].transAxes)
                        
                        plt.tight_layout()
                        plt.savefig(os.path.join(self.output_dir, f'{controller_name}_timeseries.png'))
                        plt.close()
                        
                    except Exception as e:
                        logger.error(f"Error creating time series plot for {controller_name}: {e}")
                        continue
            
        except Exception as e:
            logger.error(f"Error creating time series plots: {e}")

def main():
    """Main entry point for simulation."""
    logger.info("Starting ventilation control simulation")
    
    # Create simulation runner
    runner = SimulationRunner()
    
    # Combine real data instead of generating simulation data
    if not runner.combine_real_data():
        logger.error("Failed to combine real data. Exiting.")
        return 1
    
    # Train Markov model using Q-learning
    if not runner.train_markov_model():
        logger.error("Failed to train Markov model. Exiting.")
        return 1
    
    # Run full simulation
    if not runner.run_simulation():
        logger.error("Simulation failed. Exiting.")
        return 1
    
    logger.info("Simulation completed successfully")
    return 0

if __name__ == "__main__":
    sys.exit(main())