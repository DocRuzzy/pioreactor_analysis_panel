"""
Batch growth rate analysis functions.

This module provides functions for analyzing exponential growth rates in batch culture,
including automatic detection of the exponential phase and yield calculations.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Tuple, Optional, List, Dict, Any

from pioreactor_analysis.core.data_models import GrowthRateResult, YieldResult


def calculate_batch_growth_rate(
    od_data: pd.DataFrame,
    start_time: float,
    end_time: float,
    od_column: str = 'od_smooth',
    time_column: str = 'elapsed_hours',
    min_points: int = 5
) -> GrowthRateResult:
    """
    Calculate exponential growth rate from ln(OD) vs time regression.

    Theory: During exponential growth, ln(OD) = μ·t + ln(OD₀)
    Where μ is the specific growth rate (h⁻¹)

    Args:
        od_data: DataFrame with preprocessed OD data (already smoothed and filtered)
        start_time: Start of analysis window (hours)
        end_time: End of analysis window (hours)
        od_column: Name of the OD column to use (default: 'od_smooth')
        time_column: Name of the time column (default: 'elapsed_hours')
        min_points: Minimum number of data points required (default: 5)

    Returns:
        GrowthRateResult with complete statistical analysis

    Raises:
        ValueError: If insufficient data points or invalid time range

    Example:
        from pioreactor_analysis import preprocess_od_data, calculate_batch_growth_rate

        # Preprocess data
        df_processed = preprocess_od_data(df, smoothing_window=5, min_od_threshold=0.05)

        # Calculate growth rate for exponential phase (2-8 hours)
        result = calculate_batch_growth_rate(df_processed, start_time=2.0, end_time=8.0)

        print(result)  # Human-readable summary
        print(f"μ = {result.growth_rate:.4f} h⁻¹")
        print(f"td = {result.doubling_time:.2f} h")
    """
    # Validate inputs
    if start_time >= end_time:
        raise ValueError(f"start_time ({start_time}) must be less than end_time ({end_time})")

    if time_column not in od_data.columns:
        raise ValueError(f"Time column '{time_column}' not found in DataFrame. Available: {list(od_data.columns)}")

    if od_column not in od_data.columns:
        raise ValueError(f"OD column '{od_column}' not found in DataFrame. Available: {list(od_data.columns)}")

    # Extract data for the specified time window
    mask = (od_data[time_column] >= start_time) & (od_data[time_column] <= end_time)
    window_data = od_data[mask].copy()

    if len(window_data) < min_points:
        raise ValueError(
            f"Insufficient data points in time window {start_time:.2f}-{end_time:.2f}h. "
            f"Found {len(window_data)} points, need at least {min_points}."
        )

    # Sort by time
    window_data = window_data.sort_values(time_column)

    # Calculate ln(OD) if not already present
    if 'ln_od' not in window_data.columns:
        # Clip to avoid log(0)
        od_clipped = window_data[od_column].clip(lower=1e-6)
        window_data['ln_od'] = np.log(od_clipped)

    # Filter out non-finite values
    mask_finite = np.isfinite(window_data['ln_od']) & np.isfinite(window_data[time_column])
    window_data = window_data[mask_finite]

    if len(window_data) < min_points:
        raise ValueError(
            f"Insufficient valid data points after filtering non-finite values. "
            f"Found {len(window_data)} points, need at least {min_points}."
        )

    # Check for sufficient variation in time values
    time_range = np.ptp(window_data[time_column])
    if time_range < 1e-9:
        raise ValueError("Time values in window are identical - cannot perform regression")

    # Perform linear regression: ln(OD) = slope * time + intercept
    try:
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            window_data[time_column],
            window_data['ln_od']
        )
    except ValueError as e:
        raise ValueError(f"Linear regression failed: {e}")

    # Growth rate is the slope
    growth_rate = slope  # h⁻¹

    # Calculate doubling time: td = ln(2) / μ
    if growth_rate > 1e-9:
        doubling_time = np.log(2) / growth_rate
    else:
        raise ValueError(f"Growth rate too small or negative ({growth_rate:.6f} h⁻¹) - not exponential growth")

    # Calculate 95% confidence interval
    n = len(window_data)
    t_val = stats.t.ppf(0.975, n - 2)  # 95% CI, two-tailed
    ci_width = t_val * std_err
    ci_lower = growth_rate - ci_width
    ci_upper = growth_rate + ci_width

    # Extract unit and experiment info if available
    unit = window_data['unit'].iloc[0] if 'unit' in window_data.columns else None
    experiment = window_data['experiment'].iloc[0] if 'experiment' in window_data.columns else None

    return GrowthRateResult(
        growth_rate=growth_rate,
        doubling_time=doubling_time,
        r_squared=r_value**2,
        std_error=std_err,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        p_value=p_value,
        n_points=n,
        start_time=start_time,
        end_time=end_time,
        intercept=intercept,
        unit=unit,
        experiment=experiment
    )


def auto_detect_exponential_phase(
    od_data: pd.DataFrame,
    od_column: str = 'od_smooth',
    time_column: str = 'elapsed_hours',
    min_r_squared: float = 0.90,
    min_window_size: Optional[int] = None,
    max_window_size: Optional[int] = None,
    growth_rate_weight: float = 0.60,
    r_squared_weight: float = 0.25,
    early_time_weight: float = 0.15
) -> Tuple[float, float, GrowthRateResult]:
    """
    Automatically detect the exponential growth phase using sliding window analysis.

    Algorithm:
    1. Try multiple window sizes (dynamic based on data length)
    2. For each window, perform linear regression on ln(OD) vs time
    3. Keep windows with positive growth rate and good fit (R² > min_r_squared)
    4. Score each window: prioritize max growth rate, good fit, and early timing
    5. Return the best scoring window

    Scoring:
    - Growth rate: 60% (prioritizes maximum specific growth rate)
    - R² quality: 25% (ensures good fit)
    - Early timing: 15% (exponential phase typically occurs early)

    Args:
        od_data: DataFrame with preprocessed OD data (smoothed, filtered, with ln_od)
        od_column: Name of the OD column (default: 'od_smooth')
        time_column: Name of the time column (default: 'elapsed_hours')
        min_r_squared: Minimum R² to consider (default: 0.90)
        min_window_size: Minimum window size in points (default: max(8, 10% of data))
        max_window_size: Maximum window size in points (default: min(25, 33% of data))
        growth_rate_weight: Weight for growth rate in scoring (default: 0.60)
        r_squared_weight: Weight for R² in scoring (default: 0.25)
        early_time_weight: Weight for early timing in scoring (default: 0.15)

    Returns:
        Tuple of (start_time, end_time, GrowthRateResult) for the best exponential phase

    Raises:
        ValueError: If no exponential regions found or insufficient data

    Example:
        from pioreactor_analysis import preprocess_od_data, auto_detect_exponential_phase

        # Preprocess data
        df_processed = preprocess_od_data(df, smoothing_window=5, min_od_threshold=0.05)

        # Auto-detect exponential phase
        start, end, result = auto_detect_exponential_phase(df_processed)

        print(f"Detected exponential phase: {start:.2f} - {end:.2f} hours")
        print(f"μmax = {result.growth_rate:.4f} h⁻¹")
        print(f"R² = {result.r_squared:.4f}")
    """
    # Validate inputs
    if time_column not in od_data.columns:
        raise ValueError(f"Time column '{time_column}' not found in DataFrame")

    if od_column not in od_data.columns:
        raise ValueError(f"OD column '{od_column}' not found in DataFrame")

    if len(od_data) < 10:
        raise ValueError(f"Insufficient data for auto-detection. Need at least 10 points, got {len(od_data)}")

    # Ensure data is sorted by time
    od_data = od_data.sort_values(time_column).reset_index(drop=True)

    # Calculate ln(OD) if not present
    if 'ln_od' not in od_data.columns:
        od_clipped = od_data[od_column].clip(lower=1e-6)
        od_data['ln_od'] = np.log(od_clipped)

    # Filter out non-finite values
    mask_finite = np.isfinite(od_data['ln_od']) & np.isfinite(od_data[time_column])
    od_data = od_data[mask_finite].reset_index(drop=True)

    if len(od_data) < 10:
        raise ValueError("Insufficient valid data points after filtering non-finite values")

    # Determine window sizes
    if min_window_size is None:
        min_window_size = max(8, len(od_data) // 10)  # At least 8 points, or 10% of data

    if max_window_size is None:
        max_window_size = min(25, len(od_data) // 3)   # Max 25 points, or 33% of data

    # Ensure valid window sizes
    if min_window_size >= len(od_data):
        min_window_size = len(od_data) // 2

    if max_window_size >= len(od_data):
        max_window_size = len(od_data) - 1

    if min_window_size > max_window_size:
        min_window_size = max_window_size

    # Slide window across data and evaluate each potential exponential phase
    all_results = []

    for window_size in range(min_window_size, max_window_size + 1, 2):  # Step by 2 for efficiency
        for i in range(len(od_data) - window_size + 1):
            window = od_data.iloc[i:i+window_size]

            # Check for sufficient variation in time
            if np.ptp(window[time_column]) < 1e-9:
                continue

            try:
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    window[time_column],
                    window['ln_od']
                )

                # Only consider positive growth rates with good fit
                if slope > 0 and r_value**2 >= min_r_squared:
                    midpoint_time = window[time_column].iloc[window_size//2]
                    start_time = window[time_column].iloc[0]
                    end_time = window[time_column].iloc[-1]

                    all_results.append({
                        'start_idx': i,
                        'window_size': window_size,
                        'start_time': start_time,
                        'end_time': end_time,
                        'midpoint_time': midpoint_time,
                        'growth_rate': slope,
                        'r_squared': r_value**2,
                        'std_err': std_err,
                        'p_value': p_value,
                        'intercept': intercept,
                        'n_points': window_size
                    })

            except (ValueError, np.linalg.LinAlgError):
                continue

    if not all_results:
        raise ValueError(
            f"No exponential regions found with R² > {min_r_squared}. "
            f"Try adjusting smoothing, threshold, or lowering min_r_squared."
        )

    # Convert to DataFrame for easier manipulation
    results_df = pd.DataFrame(all_results)

    # Calculate composite score
    # Normalize growth rates to 0-1 scale
    gr_min = results_df['growth_rate'].min()
    gr_max = results_df['growth_rate'].max()
    if gr_max - gr_min > 1e-9:
        gr_normalized = (results_df['growth_rate'] - gr_min) / (gr_max - gr_min)
    else:
        gr_normalized = pd.Series([1.0] * len(results_df))

    # Normalize R² (already 0-1, but emphasize high values)
    r2_normalized = results_df['r_squared']

    # Time penalty: prefer earlier times (exponential phase typically early)
    max_time = results_df['midpoint_time'].max()
    if max_time > 1e-9:
        time_normalized = 1 - (results_df['midpoint_time'] / max_time)
    else:
        time_normalized = pd.Series([1.0] * len(results_df))

    # Composite score
    results_df['score'] = (
        gr_normalized * growth_rate_weight +
        r2_normalized * r_squared_weight +
        time_normalized * early_time_weight
    )

    # Find the best window
    best_idx = results_df['score'].idxmax()
    best_result = results_df.loc[best_idx]

    start_time = float(best_result['start_time'])
    end_time = float(best_result['end_time'])

    # Extract unit and experiment info if available
    unit = od_data['unit'].iloc[0] if 'unit' in od_data.columns else None
    experiment = od_data['experiment'].iloc[0] if 'experiment' in od_data.columns else None

    # Create GrowthRateResult
    growth_rate = float(best_result['growth_rate'])
    doubling_time = np.log(2) / growth_rate

    # Calculate confidence interval
    n = int(best_result['n_points'])
    t_val = stats.t.ppf(0.975, n - 2)
    ci_width = t_val * best_result['std_err']
    ci_lower = growth_rate - ci_width
    ci_upper = growth_rate + ci_width

    result = GrowthRateResult(
        growth_rate=growth_rate,
        doubling_time=doubling_time,
        r_squared=float(best_result['r_squared']),
        std_error=float(best_result['std_err']),
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        p_value=float(best_result['p_value']),
        n_points=n,
        start_time=start_time,
        end_time=end_time,
        intercept=float(best_result['intercept']),
        unit=unit,
        experiment=experiment
    )

    return start_time, end_time, result


def calculate_apparent_yield(
    od_data: pd.DataFrame,
    initial_substrate_conc_gl: float,
    reactor_volume_ml: float = 14.0,
    od_column: str = 'od_smooth',
    time_column: str = 'elapsed_hours',
    smoothing_window: int = 5
) -> YieldResult:
    """
    Calculate apparent biomass yield (Yx/s) based on OD change and substrate consumption.

    Theory: Y = ΔX / ΔS
    Where X is biomass (approximated by OD) and S is substrate.

    This function assumes all substrate is consumed during growth and calculates
    apparent yield as: Y = (OD_max - OD_initial) / (substrate_mass)

    Args:
        od_data: DataFrame with OD readings (already smoothed, or will be smoothed)
        initial_substrate_conc_gl: Initial substrate concentration (g/L)
        reactor_volume_ml: Reactor volume in mL (default: 14.0 for Pioreactor)
        od_column: Name of the OD column to use (default: 'od_smooth')
        time_column: Name of the time column (default: 'elapsed_hours')
        smoothing_window: Window size for smoothing if not already smoothed (default: 5)

    Returns:
        YieldResult with yield coefficient and growth status

    Raises:
        ValueError: If insufficient data or invalid parameters

    Example:
        from pioreactor_analysis import preprocess_od_data, calculate_apparent_yield

        # Preprocess data
        df_processed = preprocess_od_data(df, smoothing_window=5)

        # Calculate yield (assuming 20 g/L glucose initial concentration)
        yield_result = calculate_apparent_yield(
            df_processed,
            initial_substrate_conc_gl=20.0,
            reactor_volume_ml=14.0
        )

        print(yield_result)
        print(f"Yield: {yield_result.yield_g_biomass_per_g_substrate:.3f} OD/(g substrate)")
    """
    # Validate inputs
    if len(od_data) < 5:
        raise ValueError("Insufficient data for yield calculation")

    if initial_substrate_conc_gl <= 0:
        raise ValueError("Initial substrate concentration must be positive")

    if reactor_volume_ml <= 0:
        raise ValueError("Reactor volume must be positive")

    if time_column not in od_data.columns:
        raise ValueError(f"Time column '{time_column}' not found")

    if od_column not in od_data.columns:
        raise ValueError(f"OD column '{od_column}' not found")

    # Sort by time
    od_data = od_data.sort_values(time_column)

    # Apply smoothing if needed (create temporary smoothed column)
    if od_column == 'od_smooth' and 'od_smooth' not in od_data.columns:
        # Assume raw OD is in 'od_value' or 'od_reading'
        raw_col = 'od_value' if 'od_value' in od_data.columns else 'od_reading'
        if raw_col not in od_data.columns:
            raise ValueError("Cannot find raw OD column for smoothing")
        od_smooth = od_data[raw_col].rolling(window=smoothing_window, min_periods=1, center=True).mean()
    else:
        od_smooth = od_data[od_column]

    # Get initial OD (use median of first few points to be robust)
    n_initial = min(5, len(od_data))
    initial_od = od_smooth.iloc[:n_initial].median()

    # Get maximum OD
    max_od = od_smooth.max()
    max_od_idx = od_smooth.idxmax()
    max_od_time = od_data.loc[max_od_idx, time_column]

    # Calculate OD change
    delta_od = max_od - initial_od

    if delta_od < 0:
        raise ValueError(f"OD decreased (initial={initial_od:.4f}, max={max_od:.4f}) - not valid growth")

    # Determine if growth has plateaued or is still ongoing
    # Check if there's substantial data after max OD
    data_after_max = od_data[od_data[time_column] > max_od_time]

    # Get the last 10% of data points
    last_points_count = max(5, len(od_data) // 10)
    last_points = od_data.tail(last_points_count)

    # Default status
    yield_status = 'At least'  # Still growing

    # Check if OD is declining or plateauing
    if len(data_after_max) >= 3 and len(last_points) >= 3:
        # Get indices that are valid for both od_smooth Series and od_data DataFrame
        last_indices = last_points.index
        # Use positional indexing for Series, label indexing for DataFrame
        last_od_values = od_smooth.loc[last_indices].values
        last_time_values = od_data.loc[last_indices, time_column].values

        # Check for sufficient time variation
        if np.ptp(last_time_values) > 1e-9:
            # Linear regression on last portion
            slope_last = np.polyfit(last_time_values, last_od_values, 1)[0]

            # Calculate relative slope compared to growth phase
            if max_od_time > 1e-9:
                avg_growth_slope = max_od / max_od_time
                relative_slope = abs(slope_last) / avg_growth_slope if avg_growth_slope > 0 else 0

                # If OD is decreasing or flat (< 1% of growth slope)
                if slope_last < 0 or relative_slope < 0.01:
                    yield_status = 'Complete'

    # Calculate substrate mass in reactor
    substrate_mass_g = initial_substrate_conc_gl * (reactor_volume_ml / 1000.0)

    # Calculate apparent yield: ΔOD / substrate consumed
    # Assuming all substrate is consumed
    apparent_yield = delta_od / substrate_mass_g

    # Extract unit and experiment info if available
    unit = od_data['unit'].iloc[0] if 'unit' in od_data.columns else None
    experiment = od_data['experiment'].iloc[0] if 'experiment' in od_data.columns else None

    return YieldResult(
        yield_g_biomass_per_g_substrate=apparent_yield,
        initial_od=initial_od,
        max_od=max_od,
        delta_od=delta_od,
        initial_substrate_gl=initial_substrate_conc_gl,
        substrate_consumed_gl=initial_substrate_conc_gl,  # Assuming all consumed
        unit=unit,
        experiment=experiment
    )


def batch_analyze_multiple_units(
    od_data: pd.DataFrame,
    start_time: float,
    end_time: float,
    unit_column: str = 'unit',
    od_column: str = 'od_smooth',
    time_column: str = 'elapsed_hours'
) -> List[GrowthRateResult]:
    """
    Analyze growth rates for multiple units/reactors in a single dataset.

    Args:
        od_data: DataFrame with data from multiple units
        start_time: Start of analysis window (hours)
        end_time: End of analysis window (hours)
        unit_column: Name of the column identifying different units (default: 'unit')
        od_column: Name of the OD column (default: 'od_smooth')
        time_column: Name of the time column (default: 'elapsed_hours')

    Returns:
        List of GrowthRateResult, one for each unit

    Example:
        results = batch_analyze_multiple_units(df_processed, start_time=2.0, end_time=8.0)

        for result in results:
            print(f"{result.unit}: μ = {result.growth_rate:.4f} h⁻¹")
    """
    if unit_column not in od_data.columns:
        raise ValueError(f"Unit column '{unit_column}' not found in DataFrame")

    units = od_data[unit_column].unique()
    results = []

    for unit in units:
        unit_data = od_data[od_data[unit_column] == unit]

        try:
            result = calculate_batch_growth_rate(
                unit_data,
                start_time=start_time,
                end_time=end_time,
                od_column=od_column,
                time_column=time_column
            )
            results.append(result)
        except ValueError as e:
            print(f"Warning: Could not analyze unit {unit}: {e}")
            continue

    return results
