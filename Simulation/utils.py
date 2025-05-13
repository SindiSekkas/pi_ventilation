# simulation/utils.py
"""
Utility functions for the ventilation simulation.
Provides helper methods for data processing and analysis.
"""
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

def interpolate_missing_values(data: List[float], max_gap: int = 3) -> List[float]:
    """
    Interpolate small gaps in time series data.
    
    Args:
        data: List of values with possible None/NaN values
        max_gap: Maximum gap size to interpolate
    
    Returns:
        list: Data with gaps filled
    """
    # Convert to numpy array
    arr = np.array(data, dtype=float)
    
    # Find indices of missing values
    missing_idx = np.isnan(arr)
    
    # Find contiguous missing regions
    regions = []
    region_start = None
    
    for i, missing in enumerate(missing_idx):
        if missing and region_start is None:
            region_start = i
        elif not missing and region_start is not None:
            regions.append((region_start, i-1))
            region_start = None
    
    # Add the last region if it extends to the end
    if region_start is not None:
        regions.append((region_start, len(arr)-1))
    
    # Interpolate small gaps
    for start, end in regions:
        gap_size = end - start + 1
        
        # Only interpolate small gaps
        if gap_size <= max_gap:
            # Find values before and after gap
            before_idx = start - 1
            after_idx = end + 1
            
            # Check if indices are valid
            if before_idx >= 0 and after_idx < len(arr):
                before_val = arr[before_idx]
                after_val = arr[after_idx]
                
                # Linear interpolation
                for i in range(start, end+1):
                    weight = (i - start + 1) / (gap_size + 1)
                    arr[i] = before_val * (1 - weight) + after_val * weight
    
    return arr.tolist()

def smooth_time_series(data: List[float], window_size: int = 3) -> List[float]:
    """
    Apply smoothing to time series data.
    
    Args:
        data: List of values to smooth
        window_size: Size of the smoothing window
    
    Returns:
        list: Smoothed data
    """
    if window_size <= 1 or len(data) <= window_size:
        return data
    
    # Convert to numpy array
    arr = np.array(data, dtype=float)
    
    # Apply moving average
    smoothed = np.convolve(arr, np.ones(window_size)/window_size, mode='valid')
    
    # Pad the beginning to maintain original length
    padding = np.full(window_size-1, np.nan)
    result = np.concatenate((padding, smoothed))
    
    # Fill in padding with original values
    for i in range(len(padding)):
        result[i] = arr[i]
    
    return result.tolist()

def calculate_daily_statistics(timestamps: List[datetime], values: List[float]) -> Dict[str, Dict[str, float]]:
    """
    Calculate daily statistics for time series data.
    
    Args:
        timestamps: List of timestamps
        values: List of corresponding values
    
    Returns:
        dict: Daily statistics
    """
    if len(timestamps) != len(values):
        logger.error(f"Length mismatch: timestamps={len(timestamps)}, values={len(values)}")
        return {}
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'value': values
    })
    
    # Extract date
    df['date'] = df['timestamp'].dt.date
    
    # Group by date and calculate statistics
    daily_stats = {}
    
    for date, group in df.groupby('date'):
        if group['value'].isna().all():
            continue
            
        stats = {
            'min': group['value'].min(),
            'max': group['value'].max(),
            'mean': group['value'].mean(),
            'median': group['value'].median(),
            'std': group['value'].std(),
            'count': len(group)
        }
        
        daily_stats[date.strftime('%Y-%m-%d')] = stats
    
    return daily_stats

def extract_state_from_action_history(actions: List[str]) -> Tuple[List[str], List[int]]:
    """
    Extract state transitions from action history.
    
    Args:
        actions: List of ventilation actions
    
    Returns:
        tuple: (unique_states, state_durations)
    """
    if not actions:
        return [], []
    
    unique_states = []
    state_durations = []
    
    current_state = actions[0]
    current_duration = 1
    
    for action in actions[1:]:
        if action == current_state:
            current_duration += 1
        else:
            unique_states.append(current_state)
            state_durations.append(current_duration)
            current_state = action
            current_duration = 1
    
    # Add the last state
    unique_states.append(current_state)
    state_durations.append(current_duration)
    
    return unique_states, state_durations

def calculate_energy_efficiency(energy_consumed: float, co2_reduction: float) -> float:
    """
    Calculate energy efficiency metric (energy per CO2 reduction).
    
    Args:
        energy_consumed: Total energy consumed (Wh)
        co2_reduction: Total CO2 reduction (ppm-hours)
    
    Returns:
        float: Energy efficiency (Wh per ppm-hour)
    """
    if co2_reduction <= 0:
        return 0.0
    return energy_consumed / co2_reduction

def create_hourly_profile(timestamps: List[datetime], values: List[float]) -> Dict[int, float]:
    """
    Create hourly profile from time series data.
    
    Args:
        timestamps: List of timestamps
        values: List of corresponding values
    
    Returns:
        dict: Average value by hour of day
    """
    if len(timestamps) != len(values):
        logger.error(f"Length mismatch: timestamps={len(timestamps)}, values={len(values)}")
        return {}
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'value': values
    })
    
    # Extract hour
    df['hour'] = df['timestamp'].dt.hour
    
    # Group by hour and calculate statistics
    hourly_profile = {}
    
    for hour, group in df.groupby('hour'):
        if group['value'].isna().all():
            continue
            
        hourly_profile[hour] = group['value'].mean()
    
    return hourly_profile

def merge_config_files(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge configuration dictionaries, with override values taking precedence.
    
    Args:
        base_config: Base configuration dictionary
        override_config: Override configuration dictionary
    
    Returns:
        dict: Merged configuration
    """
    merged = base_config.copy()
    
    def _merge_recursive(base, override):
        for key in override:
            if key in base and isinstance(base[key], dict) and isinstance(override[key], dict):
                _merge_recursive(base[key], override[key])
            else:
                base[key] = override[key]
    
    _merge_recursive(merged, override_config)
    return merged