"""
Dilution rate analysis functions for continuous culture (chemostat/turbidostat).

This module provides functions for calculating dilution rates from dosing events
and analyzing continuous culture dynamics.
"""

import pandas as pd
import numpy as np
from typing import Optional

from pioreactor_analysis.core.data_models import DilutionRateData


def calculate_dilution_rate(
    dosing_events: pd.DataFrame,
    reactor_volume_ml: float = 14.0,
    moving_avg_window: int = 5,
    timestamp_column: str = 'timestamp',
    volume_column: str = 'volume_ml'
) -> pd.DataFrame:
    """
    Calculate dilution rate from dosing events.

    Theory: Dilution rate D = V_dose / (V_reactor · Δt)
    Where:
    - D = dilution rate (h⁻¹)
    - V_dose = volume of media added (mL)
    - V_reactor = reactor volume (mL)
    - Δt = time between doses (hours)

    Args:
        dosing_events: DataFrame with dilution event data
        reactor_volume_ml: Reactor volume in mL (default: 14.0 for Pioreactor)
        moving_avg_window: Window size for moving average smoothing (default: 5)
        timestamp_column: Name of timestamp column (default: 'timestamp')
        volume_column: Name of volume column (default: 'volume_ml')

    Returns:
        DataFrame with added columns:
        - instant_dilution_rate: Instantaneous dilution rate (h⁻¹)
        - moving_avg_dilution_rate: Smoothed dilution rate with moving average (h⁻¹)
        - time_diff_hours: Time between consecutive doses (hours)

    Raises:
        ValueError: If insufficient data or invalid parameters

    Example:
        from pioreactor_analysis import parse_pioreactor_csv, calculate_dilution_rate

        # Parse dosing events
        data = parse_pioreactor_csv("dosing_events.csv")
        df_dosing = data.to_dilution_dataframe()

        # Calculate dilution rates
        df_with_rates = calculate_dilution_rate(
            df_dosing,
            reactor_volume_ml=14.0,
            moving_avg_window=5
        )

        # Plot results
        import matplotlib.pyplot as plt
        plt.plot(df_with_rates['timestamp'], df_with_rates['moving_avg_dilution_rate'])
        plt.xlabel('Time')
        plt.ylabel('Dilution Rate (h⁻¹)')
        plt.show()
    """
    # Validate inputs
    if len(dosing_events) < 2:
        raise ValueError("Need at least 2 dosing events to calculate dilution rate")

    if reactor_volume_ml <= 0:
        raise ValueError("Reactor volume must be positive")

    if timestamp_column not in dosing_events.columns:
        raise ValueError(f"Timestamp column '{timestamp_column}' not found in DataFrame")

    if volume_column not in dosing_events.columns:
        raise ValueError(f"Volume column '{volume_column}' not found in DataFrame")

    # Make a copy to avoid modifying original
    df = dosing_events.copy()

    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_column]):
        df[timestamp_column] = pd.to_datetime(df[timestamp_column])

    # Sort by timestamp
    df = df.sort_values(timestamp_column).reset_index(drop=True)

    # Calculate time differences between consecutive doses
    df['next_timestamp'] = df[timestamp_column].shift(-1)
    df['time_diff_hours'] = (df['next_timestamp'] - df[timestamp_column]).dt.total_seconds() / 3600

    # Calculate instantaneous dilution rate: D = V / (V_reactor * Δt)
    # Use np.where to handle division by zero safely
    df['instant_dilution_rate'] = np.where(
        df['time_diff_hours'] > 0,
        df[volume_column] / reactor_volume_ml / df['time_diff_hours'],
        np.nan
    )

    # Replace infinite values with NaN
    df['instant_dilution_rate'] = df['instant_dilution_rate'].replace([np.inf, -np.inf], np.nan)

    # Calculate moving average of dilution rate
    df['moving_avg_dilution_rate'] = (
        df['instant_dilution_rate']
        .rolling(window=moving_avg_window, min_periods=1)
        .mean()
    )

    # Drop the temporary next_timestamp column
    df = df.drop(columns=['next_timestamp'])

    return df


def calculate_dilution_rate_from_model(
    dosing_events: pd.DataFrame,
    reactor_volume_ml: float = 14.0,
    moving_avg_window: int = 5,
    timestamp_column: str = 'timestamp',
    volume_column: str = 'volume_ml'
) -> DilutionRateData:
    """
    Calculate dilution rate and return as DilutionRateData model.

    This is a wrapper around calculate_dilution_rate that returns
    a Pydantic data model instead of a DataFrame.

    Args:
        dosing_events: DataFrame with dilution event data
        reactor_volume_ml: Reactor volume in mL (default: 14.0)
        moving_avg_window: Window size for moving average (default: 5)
        timestamp_column: Name of timestamp column (default: 'timestamp')
        volume_column: Name of volume column (default: 'volume_ml')

    Returns:
        DilutionRateData model with processed dilution rate data

    Example:
        dilution_data = calculate_dilution_rate_from_model(df_dosing)

        print(f"Mean dilution rate: {dilution_data.get_mean_dilution_rate():.4f} h⁻¹")

        steady_state_d = dilution_data.get_steady_state_dilution_rate()
        if steady_state_d:
            print(f"Steady-state D: {steady_state_d:.4f} h⁻¹")
    """
    # Calculate dilution rates
    df = calculate_dilution_rate(
        dosing_events,
        reactor_volume_ml=reactor_volume_ml,
        moving_avg_window=moving_avg_window,
        timestamp_column=timestamp_column,
        volume_column=volume_column
    )

    # Remove rows with NaN dilution rates (last event has no next event)
    df_valid = df.dropna(subset=['instant_dilution_rate'])

    # Extract unit and experiment info if available
    unit = df_valid['unit'].iloc[0] if 'unit' in df_valid.columns else None
    experiment = df_valid['experiment'].iloc[0] if 'experiment' in df_valid.columns else None

    return DilutionRateData(
        timestamps=df_valid[timestamp_column].tolist(),
        instant_dilution_rates=df_valid['instant_dilution_rate'].tolist(),
        moving_avg_dilution_rates=df_valid['moving_avg_dilution_rate'].tolist(),
        volumes_ml=df_valid[volume_column].tolist(),
        reactor_volume_ml=reactor_volume_ml,
        moving_avg_window=moving_avg_window,
        unit=unit,
        experiment=experiment
    )


def detect_steady_state(
    dilution_rate_data: pd.DataFrame,
    dilution_rate_column: str = 'moving_avg_dilution_rate',
    time_column: str = 'timestamp',
    cv_threshold: float = 0.05,
    min_window_hours: float = 1.0,
    min_points: int = 10
) -> Optional[pd.DataFrame]:
    """
    Detect steady-state periods in continuous culture.

    Steady state is defined as a period where the dilution rate has
    low variation (coefficient of variation < threshold).

    Args:
        dilution_rate_data: DataFrame with dilution rate data
        dilution_rate_column: Column name for dilution rate (default: 'moving_avg_dilution_rate')
        time_column: Column name for timestamps (default: 'timestamp')
        cv_threshold: Maximum coefficient of variation for steady state (default: 0.05 = 5%)
        min_window_hours: Minimum duration for steady state (default: 1.0 hour)
        min_points: Minimum number of points in steady-state window (default: 10)

    Returns:
        DataFrame containing only steady-state data, or None if no steady state found

    Example:
        df_rates = calculate_dilution_rate(df_dosing)
        df_steady = detect_steady_state(df_rates, cv_threshold=0.05)

        if df_steady is not None:
            mean_d = df_steady['moving_avg_dilution_rate'].mean()
            print(f"Steady-state dilution rate: {mean_d:.4f} h⁻¹")
        else:
            print("No steady state detected")
    """
    if len(dilution_rate_data) < min_points:
        return None

    if dilution_rate_column not in dilution_rate_data.columns:
        raise ValueError(f"Column '{dilution_rate_column}' not found")

    if time_column not in dilution_rate_data.columns:
        raise ValueError(f"Column '{time_column}' not found")

    df = dilution_rate_data.copy()

    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[time_column]):
        df[time_column] = pd.to_datetime(df[time_column])

    # Sort by time
    df = df.sort_values(time_column).reset_index(drop=True)

    # Remove NaN values
    df = df.dropna(subset=[dilution_rate_column])

    if len(df) < min_points:
        return None

    # Try different window sizes to find steady state
    best_window = None
    best_cv = float('inf')

    for window_size in range(min_points, len(df) + 1):
        for i in range(len(df) - window_size + 1):
            window = df.iloc[i:i+window_size]

            # Check duration
            duration_hours = (window[time_column].iloc[-1] - window[time_column].iloc[0]).total_seconds() / 3600
            if duration_hours < min_window_hours:
                continue

            # Calculate coefficient of variation
            rates = window[dilution_rate_column].values
            mean_rate = np.mean(rates)
            std_rate = np.std(rates)

            if mean_rate > 1e-9:  # Avoid division by zero
                cv = std_rate / mean_rate

                if cv < cv_threshold and cv < best_cv:
                    best_cv = cv
                    best_window = window.copy()

    return best_window


def calculate_dilution_rate_statistics(
    dilution_rate_data: pd.DataFrame,
    dilution_rate_column: str = 'moving_avg_dilution_rate'
) -> dict:
    """
    Calculate summary statistics for dilution rate data.

    Args:
        dilution_rate_data: DataFrame with dilution rate data
        dilution_rate_column: Column name for dilution rate (default: 'moving_avg_dilution_rate')

    Returns:
        Dictionary with statistics:
        - mean: Mean dilution rate (h⁻¹)
        - median: Median dilution rate (h⁻¹)
        - std: Standard deviation (h⁻¹)
        - cv: Coefficient of variation (dimensionless)
        - min: Minimum dilution rate (h⁻¹)
        - max: Maximum dilution rate (h⁻¹)
        - n_points: Number of data points

    Example:
        stats = calculate_dilution_rate_statistics(df_rates)
        print(f"Mean D: {stats['mean']:.4f} ± {stats['std']:.4f} h⁻¹")
        print(f"CV: {stats['cv']*100:.1f}%")
    """
    if dilution_rate_column not in dilution_rate_data.columns:
        raise ValueError(f"Column '{dilution_rate_column}' not found")

    # Remove NaN values
    rates = dilution_rate_data[dilution_rate_column].dropna().values

    if len(rates) == 0:
        raise ValueError("No valid dilution rate data")

    mean_rate = np.mean(rates)
    std_rate = np.std(rates)
    cv = std_rate / mean_rate if mean_rate > 1e-9 else np.nan

    return {
        'mean': mean_rate,
        'median': np.median(rates),
        'std': std_rate,
        'cv': cv,
        'min': np.min(rates),
        'max': np.max(rates),
        'n_points': len(rates)
    }


def align_od_with_dilution_rate(
    od_data: pd.DataFrame,
    dilution_rate_data: pd.DataFrame,
    od_time_column: str = 'timestamp',
    dilution_time_column: str = 'timestamp',
    method: str = 'nearest'
) -> pd.DataFrame:
    """
    Align OD readings with dilution rate data by timestamp.

    This is useful for plotting OD vs dilution rate or for calculating
    growth rate in continuous culture (μ = D + (1/X)·(dX/dt)).

    Args:
        od_data: DataFrame with OD readings
        dilution_rate_data: DataFrame with dilution rate data
        od_time_column: Timestamp column in OD data (default: 'timestamp')
        dilution_time_column: Timestamp column in dilution data (default: 'timestamp')
        method: Interpolation method - 'nearest', 'linear', or 'ffill' (default: 'nearest')

    Returns:
        DataFrame with OD data and interpolated dilution rates

    Example:
        df_aligned = align_od_with_dilution_rate(df_od, df_dilution)

        # Now you can plot OD vs dilution rate
        plt.scatter(df_aligned['moving_avg_dilution_rate'], df_aligned['od_smooth'])
        plt.xlabel('Dilution Rate (h⁻¹)')
        plt.ylabel('OD')
        plt.show()
    """
    # Make copies
    od_df = od_data.copy()
    dilution_df = dilution_rate_data.copy()

    # Ensure timestamps are datetime
    if not pd.api.types.is_datetime64_any_dtype(od_df[od_time_column]):
        od_df[od_time_column] = pd.to_datetime(od_df[od_time_column])

    if not pd.api.types.is_datetime64_any_dtype(dilution_df[dilution_time_column]):
        dilution_df[dilution_time_column] = pd.to_datetime(dilution_df[dilution_time_column])

    # Set timestamp as index for interpolation
    dilution_df = dilution_df.set_index(dilution_time_column)
    od_df = od_df.set_index(od_time_column)

    # Reindex dilution data to match OD timestamps
    if method == 'nearest':
        dilution_reindexed = dilution_df.reindex(od_df.index, method='nearest', limit=1)
    elif method == 'linear':
        dilution_reindexed = dilution_df.reindex(od_df.index).interpolate(method='linear')
    elif method == 'ffill':
        dilution_reindexed = dilution_df.reindex(od_df.index, method='ffill')
    else:
        raise ValueError(f"Unknown interpolation method: {method}")

    # Merge the dataframes
    result = od_df.join(
        dilution_reindexed[['instant_dilution_rate', 'moving_avg_dilution_rate']],
        how='left',
        rsuffix='_dilution'
    )

    # Reset index
    result = result.reset_index()

    # Rename the index column back
    result = result.rename(columns={'index': od_time_column})

    return result
