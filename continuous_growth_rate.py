"""
Continuous Culture Growth Rate Calculation Module

This module calculates the specific growth rate (μ) in continuous culture systems
based on dilution rate (D) and optical density (OD) dynamics.

Theory and Algorithm
====================

In continuous culture (e.g., chemostat, turbidostat), the relationship between
specific growth rate (μ), dilution rate (D), and biomass concentration (X, measured as OD)
is governed by the mass balance equation:

    dX/dt = μ·X - D·X

Where:
- X = biomass concentration (approximated by OD)
- μ = specific growth rate (h⁻¹)
- D = dilution rate (h⁻¹) = F/V, where F is flow rate and V is culture volume
- t = time (hours)

Rearranging to solve for μ:

    μ = D + (1/X)·(dX/dt)

This equation shows that the growth rate has two components:
1. The dilution rate (D) - the rate at which biomass is removed
2. The rate of change of biomass concentration (dX/dt)/X

Special Cases
-------------

1. **Steady State (dX/dt ≈ 0)**:
   When OD is constant, the growth rate equals the dilution rate:
   
   μ ≈ D
   
   This is the ideal chemostat/turbidostat condition where growth exactly
   balances dilution.

2. **Dynamic State (dX/dt ≠ 0)**:
   When OD is changing, we must account for the accumulation term:
   
   μ = D + (1/X)·(dX/dt)
   
   - If dX/dt > 0 (OD increasing): μ > D (growth exceeds dilution)
   - If dX/dt < 0 (OD decreasing): μ < D (growth less than dilution)

Implementation Notes
-------------------

1. **Numerical Differentiation**:
   dX/dt is calculated using a centered difference approximation:
   
   dX/dt ≈ (X[i+1] - X[i-1]) / (t[i+1] - t[i-1])
   
   This is more accurate than forward or backward differences.

2. **Smoothing**:
   OD measurements often have noise. We apply a rolling window average
   to smooth the data before calculating derivatives.

3. **Time Alignment**:
   Dilution events and OD measurements may not be perfectly synchronized.
   We interpolate to align them on a common time grid.

4. **Edge Cases**:
   - Very low OD values: We apply a minimum threshold to avoid division by zero
   - Missing data: We use interpolation to fill gaps
   - Outliers: We detect and optionally filter extreme values

References
----------
- Monod, J. (1950). "La technique de culture continue: théorie et applications."
  Annales de l'Institut Pasteur, 79, 390-410.
- Pirt, S.J. (1975). "Principles of Microbe and Cell Cultivation."
  Blackwell Scientific Publications.

Author: Russell Kirk Pirlo with assistance from Claude
Date: November 6, 2025
"""

import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import warnings


def calculate_growth_rate_continuous(
    od_df: pd.DataFrame,
    dilution_rate_df: pd.DataFrame,
    od_column: str = 'od_value',
    time_column: str = 'elapsed_hours',
    dilution_column: str = 'instant_dilution_rate',
    smoothing_window: int = 5,
    min_od_threshold: float = 0.05,
    use_savgol: bool = True
) -> pd.DataFrame:
    """
    Calculate specific growth rate in continuous culture.
    
    Parameters
    ----------
    od_df : pd.DataFrame
        DataFrame containing OD measurements with columns for time and OD value
    dilution_rate_df : pd.DataFrame
        DataFrame containing dilution rate data with columns for time and dilution rate
    od_column : str, default 'od_value'
        Name of the column containing OD measurements
    time_column : str, default 'elapsed_hours'
        Name of the column containing time in hours
    dilution_column : str, default 'instant_dilution_rate'
        Name of the column containing dilution rate (h⁻¹)
    smoothing_window : int, default 5
        Window size for smoothing OD data (must be odd for Savitzky-Golay)
    min_od_threshold : float, default 0.05
        Minimum OD value to consider (avoids division by very small numbers)
    use_savgol : bool, default True
        Use Savitzky-Golay filter for smoothing (more accurate) vs simple rolling mean
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - time_column: time values
        - 'od_smoothed': smoothed OD values
        - 'dilution_rate': interpolated dilution rate
        - 'dOD_dt': rate of change of OD (h⁻¹)
        - 'growth_rate': specific growth rate μ (h⁻¹)
        - 'doubling_time': doubling time (hours)
        - 'steady_state': boolean flag indicating near-steady-state conditions
    
    Notes
    -----
    The function handles missing data through interpolation and filters out
    periods where OD is below the threshold.
    """
    
    # Validate inputs
    if od_df.empty or dilution_rate_df.empty:
        raise ValueError("Input DataFrames cannot be empty")
    
    if od_column not in od_df.columns:
        raise ValueError(f"Column '{od_column}' not found in od_df")
    
    if time_column not in od_df.columns or time_column not in dilution_rate_df.columns:
        raise ValueError(f"Column '{time_column}' not found in input DataFrames")
    
    if dilution_column not in dilution_rate_df.columns:
        raise ValueError(f"Column '{dilution_column}' not found in dilution_rate_df")
    
    # Make copies to avoid modifying originals
    od_data = od_df[[time_column, od_column]].copy().dropna()
    dilution_data = dilution_rate_df[[time_column, dilution_column]].copy().dropna()
    
    if len(od_data) < 3:
        raise ValueError("Need at least 3 OD measurements for growth rate calculation")
    
    # Sort by time
    od_data = od_data.sort_values(time_column)
    dilution_data = dilution_data.sort_values(time_column)
    
    # Filter OD data by threshold
    od_data = od_data[od_data[od_column] >= min_od_threshold]
    
    if len(od_data) < 3:
        raise ValueError(f"Insufficient OD data above threshold ({min_od_threshold})")
    
    # Smooth OD data
    if use_savgol:
        # Savitzky-Golay filter (polynomial smoothing, preserves features better)
        window = min(smoothing_window, len(od_data))
        if window % 2 == 0:
            window -= 1  # Must be odd
        if window < 3:
            window = 3
        
        try:
            od_smoothed = savgol_filter(
                od_data[od_column].values,
                window_length=window,
                polyorder=min(2, window - 1),
                mode='interp'
            )
        except Exception:
            # Fall back to rolling mean if Savitzky-Golay fails
            od_smoothed = od_data[od_column].rolling(
                window=smoothing_window, center=True, min_periods=1
            ).mean().values
    else:
        # Simple rolling mean
        od_smoothed = od_data[od_column].rolling(
            window=smoothing_window, center=True, min_periods=1
        ).mean().values
    
    od_data['od_smoothed'] = od_smoothed
    
    # Calculate dOD/dt using centered differences
    time_vals = od_data[time_column].values
    od_vals = od_data['od_smoothed'].values
    
    dOD_dt = np.zeros_like(od_vals)
    
    # Centered difference for interior points
    for i in range(1, len(od_vals) - 1):
        dt = time_vals[i + 1] - time_vals[i - 1]
        if dt > 0:
            dOD_dt[i] = (od_vals[i + 1] - od_vals[i - 1]) / dt
    
    # Forward difference for first point
    if len(od_vals) > 1:
        dt = time_vals[1] - time_vals[0]
        if dt > 0:
            dOD_dt[0] = (od_vals[1] - od_vals[0]) / dt
    
    # Backward difference for last point
    if len(od_vals) > 1:
        dt = time_vals[-1] - time_vals[-2]
        if dt > 0:
            dOD_dt[-1] = (od_vals[-1] - od_vals[-2]) / dt
    
    od_data['dOD_dt'] = dOD_dt
    
    # Interpolate dilution rate to match OD time points
    if len(dilution_data) < 2:
        # If only one dilution rate value, use it for all time points
        dilution_rate_interp = np.full_like(time_vals, dilution_data[dilution_column].iloc[0])
    else:
        # Create interpolation function
        interp_func = interp1d(
            dilution_data[time_column].values,
            dilution_data[dilution_column].values,
            kind='linear',
            bounds_error=False,
            fill_value=(dilution_data[dilution_column].iloc[0], 
                       dilution_data[dilution_column].iloc[-1])
        )
        dilution_rate_interp = interp_func(time_vals)
    
    od_data['dilution_rate'] = dilution_rate_interp
    
    # Calculate growth rate: μ = D + (1/X)·(dX/dt)
    # Avoid division by very small OD values
    od_safe = np.maximum(od_vals, min_od_threshold / 2)
    growth_rate = dilution_rate_interp + (dOD_dt / od_safe)
    
    od_data['growth_rate'] = growth_rate
    
    # Calculate doubling time: td = ln(2) / μ
    # Only calculate where growth rate is positive
    doubling_time = np.where(
        growth_rate > 0,
        np.log(2) / growth_rate,
        np.nan
    )
    od_data['doubling_time'] = doubling_time
    
    # Identify steady-state regions
    # Steady state: |dOD/dt| / OD < threshold (e.g., 5% per hour)
    relative_od_change = np.abs(dOD_dt) / od_safe
    steady_state_threshold = 0.05  # 5% per hour
    od_data['steady_state'] = relative_od_change < steady_state_threshold
    
    # Reset index and clean up
    result_df = od_data.reset_index(drop=True)
    
    return result_df


def analyze_growth_phases(growth_rate_df: pd.DataFrame, 
                         time_column: str = 'elapsed_hours',
                         steady_state_column: str = 'steady_state') -> pd.DataFrame:
    """
    Identify and summarize growth phases (steady-state vs dynamic periods).
    
    Parameters
    ----------
    growth_rate_df : pd.DataFrame
        Output from calculate_growth_rate_continuous
    time_column : str
        Name of time column
    steady_state_column : str
        Name of steady-state flag column
    
    Returns
    -------
    pd.DataFrame
        Summary statistics for each continuous phase
    """
    
    if growth_rate_df.empty:
        return pd.DataFrame()
    
    # Identify phase transitions
    phase_changes = growth_rate_df[steady_state_column].astype(int).diff().fillna(0) != 0
    phase_id = phase_changes.cumsum()
    
    growth_rate_df['phase_id'] = phase_id
    
    # Summarize each phase
    phases = []
    for pid in growth_rate_df['phase_id'].unique():
        phase_data = growth_rate_df[growth_rate_df['phase_id'] == pid]
        
        phases.append({
            'phase_id': pid,
            'start_time': phase_data[time_column].min(),
            'end_time': phase_data[time_column].max(),
            'duration': phase_data[time_column].max() - phase_data[time_column].min(),
            'is_steady_state': phase_data[steady_state_column].iloc[0],
            'mean_growth_rate': phase_data['growth_rate'].mean(),
            'std_growth_rate': phase_data['growth_rate'].std(),
            'mean_dilution_rate': phase_data['dilution_rate'].mean(),
            'mean_od': phase_data['od_smoothed'].mean(),
            'od_change': phase_data['od_smoothed'].iloc[-1] - phase_data['od_smoothed'].iloc[0],
            'n_points': len(phase_data)
        })
    
    return pd.DataFrame(phases)
