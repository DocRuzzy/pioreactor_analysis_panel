"""
Data preprocessing functions for Pioreactor analysis.

This module provides functions for cleaning, smoothing, and filtering
OD data before growth rate analysis.
"""

import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from typing import Optional, Literal


def calculate_elapsed_time(
    df: pd.DataFrame,
    time_column: str = 'timestamp',
    unit: Literal['hours', 'minutes', 'seconds'] = 'hours',
    group_by: Optional[list] = None
) -> pd.DataFrame:
    """
    Calculate elapsed time from the start of each experiment.

    Args:
        df: DataFrame with timestamp column
        time_column: Name of the timestamp column (default: 'timestamp')
        unit: Time unit for elapsed time - 'hours', 'minutes', or 'seconds' (default: 'hours')
        group_by: Optional list of columns to group by (e.g., ['experiment', 'unit'])
                 If None, treats all data as single experiment

    Returns:
        DataFrame with added 'elapsed_time' column (in specified units)

    Example:
        df = calculate_elapsed_time(df, group_by=['experiment', 'unit'])
        # Adds 'elapsed_hours' column with time since start for each unit
    """
    df = df.copy()

    # Ensure timestamp column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[time_column]):
        df[time_column] = pd.to_datetime(df[time_column])

    # Sort by time
    df = df.sort_values(time_column)

    # Calculate conversion factor
    divisors = {
        'hours': 3600,
        'minutes': 60,
        'seconds': 1
    }
    divisor = divisors.get(unit, 3600)
    col_name = f'elapsed_{unit}'

    if group_by:
        # Calculate elapsed time for each group using transform (faster and safer)
        start_times = df.groupby(group_by)[time_column].transform('min')
        df[col_name] = (df[time_column] - start_times).dt.total_seconds() / divisor
    else:
        # Calculate elapsed time for entire dataset
        start_time = df[time_column].min()
        df[col_name] = (df[time_column] - start_time).dt.total_seconds() / divisor

    return df


def smooth_od_data(
    df: pd.DataFrame,
    od_column: str = 'od_value',
    window_size: int = 5,
    method: Literal['rolling', 'savgol'] = 'rolling',
    savgol_polyorder: int = 2,
    output_column: str = 'od_smooth',
    group_by: Optional[list] = None
) -> pd.DataFrame:
    """
    Apply smoothing to OD readings to reduce noise.

    Args:
        df: DataFrame with OD readings
        od_column: Name of the OD column to smooth (default: 'od_value')
        window_size: Size of the smoothing window (default: 5 points)
        method: Smoothing method - 'rolling' (moving average) or 'savgol' (Savitzky-Golay)
        savgol_polyorder: Polynomial order for Savitzky-Golay filter (default: 2)
        output_column: Name for the smoothed output column (default: 'od_smooth')
        group_by: Optional list of columns to group by (e.g., ['experiment', 'unit'])
                 If provided, smoothing is applied separately to each group

    Returns:
        DataFrame with added smoothed OD column

    Example:
        # Rolling average smoothing
        df = smooth_od_data(df, window_size=5, method='rolling')

        # Smoothing per unit (important for multi-unit data)
        df = smooth_od_data(df, window_size=5, group_by=['experiment', 'unit'])

        # Savitzky-Golay smoothing (better preserves peaks)
        df = smooth_od_data(df, window_size=7, method='savgol', savgol_polyorder=3)
    """
    df = df.copy()

    if od_column not in df.columns:
        raise ValueError(f"Column '{od_column}' not found in DataFrame")

    if method == 'rolling':
        # Rolling average (centered window)
        if group_by:
            # Apply smoothing separately to each group
            df[output_column] = df.groupby(group_by)[od_column].transform(
                lambda x: x.rolling(window=window_size, min_periods=1, center=True).mean()
            )
        else:
            df[output_column] = df[od_column].rolling(
                window=window_size,
                min_periods=1,
                center=True
            ).mean()

    elif method == 'savgol':
        # Savitzky-Golay filter
        if window_size < savgol_polyorder + 2:
            raise ValueError(f"window_size ({window_size}) must be >= polyorder + 2 ({savgol_polyorder + 2})")

        # Make window size odd (required by savgol_filter)
        if window_size % 2 == 0:
            window_size += 1

        if group_by:
            # Apply savgol separately to each group
            def apply_savgol(series):
                if len(series) < window_size:
                    # Fallback to rolling average for small groups
                    return series.rolling(window=window_size, min_periods=1, center=True).mean()
                try:
                    return pd.Series(
                        savgol_filter(
                            series.values,
                            window_length=window_size,
                            polyorder=savgol_polyorder,
                            mode='nearest'
                        ),
                        index=series.index
                    )
                except ValueError as e:
                    print(f"Warning: Savitzky-Golay filter failed for group ({e}), using rolling average")
                    return series.rolling(window=window_size, min_periods=1, center=True).mean()

            df[output_column] = df.groupby(group_by)[od_column].transform(apply_savgol)
        else:
            try:
                df[output_column] = savgol_filter(
                    df[od_column].values,
                    window_length=window_size,
                    polyorder=savgol_polyorder,
                    mode='nearest'
                )
            except ValueError as e:
                # Fallback to rolling average if savgol fails
                print(f"Warning: Savitzky-Golay filter failed ({e}), using rolling average instead")
                df[output_column] = df[od_column].rolling(
                    window=window_size,
                    min_periods=1,
                    center=True
                ).mean()

    else:
        raise ValueError(f"Unknown smoothing method: {method}. Use 'rolling' or 'savgol'")

    return df


def filter_by_threshold(
    df: pd.DataFrame,
    od_column: str = 'od_value',
    min_od: float = 0.05,
    max_od: Optional[float] = None
) -> pd.DataFrame:
    """
    Filter data to keep only OD values within specified range.

    Useful for removing:
    - Low OD values (sensor noise, lag phase)
    - High OD values (stationary phase, sensor saturation)

    Args:
        df: DataFrame with OD readings
        od_column: Name of the OD column to filter (default: 'od_value')
        min_od: Minimum OD threshold (default: 0.05)
        max_od: Optional maximum OD threshold (default: None = no upper limit)

    Returns:
        Filtered DataFrame

    Example:
        # Keep only OD between 0.05 and 2.0
        df_filtered = filter_by_threshold(df, min_od=0.05, max_od=2.0)

        # Remove only low OD values
        df_filtered = filter_by_threshold(df, min_od=0.1)
    """
    df = df.copy()

    if od_column not in df.columns:
        raise ValueError(f"Column '{od_column}' not found in DataFrame")

    # Apply filters
    mask = df[od_column] >= min_od

    if max_od is not None:
        mask = mask & (df[od_column] <= max_od)

    n_before = len(df)
    df_filtered = df[mask].copy()
    n_after = len(df_filtered)

    if n_after == 0:
        raise ValueError(
            f"All data filtered out! (min_od={min_od}, max_od={max_od}). "
            f"OD range in data: {df[od_column].min():.4f} - {df[od_column].max():.4f}"
        )

    print(f"Filtered {n_before - n_after} points ({(n_before - n_after) / n_before * 100:.1f}%), "
          f"{n_after} points remaining")

    return df_filtered


def filter_by_time_range(
    df: pd.DataFrame,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    time_column: str = 'elapsed_hours'
) -> pd.DataFrame:
    """
    Filter data to keep only specified time range.

    Args:
        df: DataFrame with time column
        start_time: Optional start time (default: None = from beginning)
        end_time: Optional end time (default: None = to end)
        time_column: Name of the time column (default: 'elapsed_hours')

    Returns:
        Filtered DataFrame

    Example:
        # Keep only data between 2 and 8 hours
        df_filtered = filter_by_time_range(df, start_time=2.0, end_time=8.0)
    """
    df = df.copy()

    if time_column not in df.columns:
        raise ValueError(f"Column '{time_column}' not found in DataFrame")

    mask = pd.Series([True] * len(df), index=df.index)

    if start_time is not None:
        mask = mask & (df[time_column] >= start_time)

    if end_time is not None:
        mask = mask & (df[time_column] <= end_time)

    n_before = len(df)
    df_filtered = df[mask].copy()
    n_after = len(df_filtered)

    if n_after == 0:
        raise ValueError(
            f"All data filtered out! (start={start_time}, end={end_time}). "
            f"Time range in data: {df[time_column].min():.2f} - {df[time_column].max():.2f}"
        )

    print(f"Time range filter: kept {n_after}/{n_before} points "
          f"({n_after / n_before * 100:.1f}%)")

    return df_filtered


def calculate_ln_od(
    df: pd.DataFrame,
    od_column: str = 'od_smooth',
    output_column: str = 'ln_od',
    clip_min: float = 1e-6
) -> pd.DataFrame:
    """
    Calculate natural logarithm of OD for exponential growth analysis.

    Theory: During exponential growth, ln(OD) vs time is linear with slope = μ

    Args:
        df: DataFrame with OD values
        od_column: Name of the OD column (default: 'od_smooth')
        output_column: Name for ln(OD) output column (default: 'ln_od')
        clip_min: Minimum OD value to avoid log(0) errors (default: 1e-6)

    Returns:
        DataFrame with added ln(OD) column

    Example:
        df = calculate_ln_od(df, od_column='od_smooth')
        # Adds 'ln_od' column
    """
    df = df.copy()

    if od_column not in df.columns:
        raise ValueError(f"Column '{od_column}' not found in DataFrame")

    # Clip to avoid log(0) and handle negative values (sensor noise)
    od_clipped = df[od_column].clip(lower=clip_min)

    df[output_column] = np.log(od_clipped)

    return df


def preprocess_od_data(
    df: pd.DataFrame,
    smoothing_window: int = 5,
    min_od_threshold: float = 0.05,
    max_od_threshold: Optional[float] = None,
    smoothing_method: Literal['rolling', 'savgol'] = 'rolling',
    calculate_ln: bool = True,
    time_column: str = 'timestamp',
    od_column: str = 'od_value',
    group_by: Optional[list] = None
) -> pd.DataFrame:
    """
    Complete preprocessing pipeline for OD data.

    Performs:
    1. Calculate elapsed time
    2. Smooth OD data
    3. Filter by OD threshold
    4. Calculate ln(OD) for exponential analysis

    Args:
        df: Raw DataFrame with OD readings
        smoothing_window: Window size for smoothing (default: 5)
        min_od_threshold: Minimum OD to keep (default: 0.05)
        max_od_threshold: Optional maximum OD to keep (default: None)
        smoothing_method: 'rolling' or 'savgol' (default: 'rolling')
        calculate_ln: Whether to add ln(OD) column (default: True)
        time_column: Name of timestamp column (default: 'timestamp')
        od_column: Name of OD column (default: 'od_value')
        group_by: Optional columns to group by (default: None)

    Returns:
        Fully preprocessed DataFrame ready for growth rate analysis

    Example:
        from pioreactor_analysis import preprocess_od_data

        df_processed = preprocess_od_data(
            df,
            smoothing_window=7,
            min_od_threshold=0.1,
            max_od_threshold=3.0,
            group_by=['experiment', 'unit']
        )
    """
    # Step 1: Calculate elapsed time (only if not already present)
    if 'elapsed_hours' not in df.columns or df['elapsed_hours'].isna().all():
        df = calculate_elapsed_time(df, time_column=time_column, group_by=group_by)

    # Step 2: Smooth OD data (apply grouping if specified)
    df = smooth_od_data(
        df,
        od_column=od_column,
        window_size=smoothing_window,
        method=smoothing_method,
        group_by=group_by
    )

    # Step 3: Filter by OD threshold
    df = filter_by_threshold(
        df,
        od_column='od_smooth',  # Use smoothed OD for filtering
        min_od=min_od_threshold,
        max_od=max_od_threshold
    )

    # Step 4: Calculate ln(OD)
    if calculate_ln:
        df = calculate_ln_od(df, od_column='od_smooth')

    return df


def detect_outliers(
    df: pd.DataFrame,
    od_column: str = 'od_value',
    method: Literal['iqr', 'zscore'] = 'iqr',
    threshold: float = 1.5,
    mark_only: bool = True
) -> pd.DataFrame:
    """
    Detect outliers in OD data.

    Args:
        df: DataFrame with OD readings
        od_column: Name of the OD column (default: 'od_value')
        method: Detection method - 'iqr' (interquartile range) or 'zscore' (default: 'iqr')
        threshold: Threshold for outlier detection
                  - For IQR: multiplier for IQR (default: 1.5)
                  - For Z-score: number of standard deviations (default: 3)
        mark_only: If True, adds 'is_outlier' column. If False, removes outliers (default: True)

    Returns:
        DataFrame with outliers marked or removed

    Example:
        # Mark outliers
        df = detect_outliers(df, method='iqr', mark_only=True)
        # Access outliers: df[df['is_outlier']]

        # Remove outliers
        df_clean = detect_outliers(df, method='iqr', mark_only=False)
    """
    df = df.copy()

    if od_column not in df.columns:
        raise ValueError(f"Column '{od_column}' not found in DataFrame")

    if method == 'iqr':
        # Interquartile range method
        Q1 = df[od_column].quantile(0.25)
        Q3 = df[od_column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        is_outlier = (df[od_column] < lower_bound) | (df[od_column] > upper_bound)

    elif method == 'zscore':
        # Z-score method
        mean = df[od_column].mean()
        std = df[od_column].std()
        z_scores = np.abs((df[od_column] - mean) / std)
        is_outlier = z_scores > threshold

    else:
        raise ValueError(f"Unknown method: {method}. Use 'iqr' or 'zscore'")

    n_outliers = is_outlier.sum()
    print(f"Detected {n_outliers} outliers ({n_outliers / len(df) * 100:.1f}%)")

    if mark_only:
        df['is_outlier'] = is_outlier
        return df
    else:
        return df[~is_outlier].copy()
