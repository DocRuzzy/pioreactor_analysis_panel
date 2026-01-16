"""
Test script for analysis functions with real Pioreactor data.
"""

import sys
from pathlib import Path

# Add pioreactor_analysis to path
sys.path.insert(0, str(Path(__file__).parent))

from pioreactor_analysis.core.csv_parser import PioreactorCSVParser
from pioreactor_analysis.core.preprocessing import preprocess_od_data
from pioreactor_analysis.analysis.batch_growth import (
    calculate_batch_growth_rate,
    auto_detect_exponential_phase,
    calculate_apparent_yield,
    batch_analyze_multiple_units
)
from pioreactor_analysis.analysis.dilution_rate import (
    calculate_dilution_rate,
    calculate_dilution_rate_statistics,
    detect_steady_state,
    align_od_with_dilution_rate
)

def main():
    print("=" * 80)
    print("Testing Pioreactor Analysis Functions")
    print("=" * 80)
    print()

    parser = PioreactorCSVParser()

    # Parse data files
    print("Loading data files...")
    print("-" * 80)
    od_file = Path("data/export_20260109160338_p3/od_readings/od_readings-in_class_10.29-all_units-20260109110344.csv")
    dosing_file = Path("data/export_20260109160338_p3/dosing_automation_events/dosing_automation_events-in_class_10.29-all_units-20260109110344.csv")

    od_data = parser.parse(od_file)
    print(f"[OK] Loaded {len(od_data.od_readings)} OD readings")

    dosing_data = parser.parse(dosing_file)
    print(f"[OK] Loaded {len(dosing_data.dilution_events)} dilution events")
    print()

    # Test 1: Batch Growth Rate Analysis
    print("\nTest 1: Batch Growth Rate Analysis")
    print("-" * 80)

    try:
        # Preprocess OD data
        df_od = od_data.to_dataframe()
        df_processed = preprocess_od_data(
            df_od,
            smoothing_window=5,
            min_od_threshold=0.05,
            time_column='timestamp',
            od_column='od_value',
            group_by=['experiment', 'unit']
        )
        print(f"[OK] Preprocessed {len(df_processed)} OD points")

        # Auto-detect exponential phase
        start, end, result = auto_detect_exponential_phase(df_processed)
        print(f"[OK] Auto-detected exponential phase: {start:.2f} - {end:.2f} hours")
        print(f"     Growth rate (mu): {result.growth_rate:.4f} h^-1")
        print(f"     Doubling time: {result.doubling_time:.2f} hours ({result.doubling_time*60:.1f} min)")
        print(f"     R-squared: {result.r_squared:.4f}")
        print(f"     95% CI: [{result.ci_lower:.4f}, {result.ci_upper:.4f}]")
        print(f"     P-value: {result.p_value:.2e}")
        print(f"     Data points: {result.n_points}")
        print()

        # Calculate growth rate manually for specific window
        print("Testing manual growth rate calculation (6-12 hours)...")
        result_manual = calculate_batch_growth_rate(
            df_processed,
            start_time=6.0,
            end_time=12.0
        )
        print(f"[OK] Manual calculation:")
        print(f"     Growth rate: {result_manual.growth_rate:.4f} h^-1")
        print(f"     Doubling time: {result_manual.doubling_time:.2f} hours")
        print(f"     R-squared: {result_manual.r_squared:.4f}")
        print()

    except Exception as e:
        print(f"[FAIL] Batch growth rate analysis failed: {e}")
        import traceback
        traceback.print_exc()
        print()

    # Test 2: Yield Calculation
    print("\nTest 2: Apparent Yield Calculation")
    print("-" * 80)

    try:
        # Calculate yield (assuming 20 g/L glucose)
        yield_result = calculate_apparent_yield(
            df_processed,
            initial_substrate_conc_gl=20.0,
            reactor_volume_ml=14.0
        )
        print(f"[OK] Yield calculation:")
        print(f"     Apparent yield: {yield_result.yield_g_biomass_per_g_substrate:.3f} OD/(g substrate)")
        print(f"     Initial OD: {yield_result.initial_od:.4f}")
        print(f"     Max OD: {yield_result.max_od:.4f}")
        print(f"     Delta OD: {yield_result.delta_od:.4f}")
        print(f"     Substrate consumed: {yield_result.substrate_consumed_gl:.2f} g/L")
        print()

    except Exception as e:
        print(f"[FAIL] Yield calculation failed: {e}")
        import traceback
        traceback.print_exc()
        print()

    # Test 3: Dilution Rate Calculation
    print("\nTest 3: Dilution Rate Analysis")
    print("-" * 80)

    try:
        # Convert dosing events to DataFrame
        df_dosing = dosing_data.to_dilution_dataframe()

        # Calculate dilution rates
        df_dilution = calculate_dilution_rate(
            df_dosing,
            reactor_volume_ml=14.0,
            moving_avg_window=5
        )
        print(f"[OK] Calculated dilution rates for {len(df_dilution)} events")

        # Calculate statistics
        stats = calculate_dilution_rate_statistics(df_dilution)
        print(f"     Mean dilution rate: {stats['mean']:.4f} h^-1")
        print(f"     Std deviation: {stats['std']:.4f} h^-1")
        print(f"     Coefficient of variation: {stats['cv']*100:.1f}%")
        print(f"     Min/Max: {stats['min']:.4f} / {stats['max']:.4f} h^-1")
        print()

        # Detect steady state
        df_steady = detect_steady_state(df_dilution, cv_threshold=0.10)
        if df_steady is not None:
            steady_mean = df_steady['moving_avg_dilution_rate'].mean()
            steady_duration = (df_steady['timestamp'].iloc[-1] - df_steady['timestamp'].iloc[0]).total_seconds() / 3600
            print(f"[OK] Detected steady-state period:")
            print(f"     Duration: {steady_duration:.2f} hours")
            print(f"     Mean dilution rate: {steady_mean:.4f} h^-1")
            print(f"     Points: {len(df_steady)}")
        else:
            print(f"[INFO] No steady-state period detected (CV threshold = 10%)")
        print()

    except Exception as e:
        print(f"[FAIL] Dilution rate analysis failed: {e}")
        import traceback
        traceback.print_exc()
        print()

    # Test 4: OD and Dilution Rate Alignment
    print("\nTest 4: Aligning OD with Dilution Rate")
    print("-" * 80)

    try:
        # Align OD data with dilution rate
        df_aligned = align_od_with_dilution_rate(
            df_processed,
            df_dilution,
            od_time_column='timestamp',
            dilution_time_column='timestamp',
            method='nearest'
        )
        print(f"[OK] Aligned {len(df_aligned)} OD points with dilution rates")
        print(f"     Columns: {list(df_aligned.columns)}")
        print()
        print("     Sample of aligned data:")
        display_cols = ['timestamp', 'od_smooth', 'moving_avg_dilution_rate']
        display_cols = [c for c in display_cols if c in df_aligned.columns]
        print(df_aligned[display_cols].dropna().head(10).to_string())
        print()

    except Exception as e:
        print(f"[FAIL] Alignment failed: {e}")
        import traceback
        traceback.print_exc()
        print()

    # Test 5: Multiple Units Analysis (if available)
    print("\nTest 5: Multiple Units Batch Analysis")
    print("-" * 80)

    try:
        # Check if there are multiple units
        units = df_processed['unit'].unique() if 'unit' in df_processed.columns else []

        if len(units) > 1:
            results = batch_analyze_multiple_units(
                df_processed,
                start_time=start,
                end_time=end
            )
            print(f"[OK] Analyzed {len(results)} units:")
            for r in results:
                print(f"     {r.unit}: mu = {r.growth_rate:.4f} h^-1, R^2 = {r.r_squared:.4f}")
        else:
            print(f"[INFO] Only 1 unit in dataset, skipping multi-unit analysis")
        print()

    except Exception as e:
        print(f"[FAIL] Multiple units analysis failed: {e}")
        import traceback
        traceback.print_exc()
        print()

    # Test 6: Continuous Growth Rate (if available)
    print("\nTest 6: Continuous Culture Growth Rate")
    print("-" * 80)

    try:
        from pioreactor_analysis.analysis.continuous_growth import calculate_growth_rate_continuous

        # Need to prepare data with elapsed_hours
        df_od_continuous = df_processed.copy()
        df_dilution_continuous = df_dilution.copy()

        # Add elapsed_hours to dilution data if not present
        if 'elapsed_hours' not in df_dilution_continuous.columns:
            start_time = df_dilution_continuous['timestamp'].min()
            df_dilution_continuous['elapsed_hours'] = (
                df_dilution_continuous['timestamp'] - start_time
            ).dt.total_seconds() / 3600

        # Calculate continuous growth rate
        df_continuous = calculate_growth_rate_continuous(
            df_od_continuous,
            df_dilution_continuous,
            od_column='od_smooth',
            time_column='elapsed_hours',
            dilution_column='moving_avg_dilution_rate',
            smoothing_window=5,
            min_od_threshold=0.05
        )
        print(f"[OK] Calculated continuous growth rate for {len(df_continuous)} time points")

        # Find steady-state periods
        steady_points = df_continuous[df_continuous['steady_state'] == True]
        if len(steady_points) > 0:
            mean_growth = steady_points['growth_rate'].mean()
            mean_dilution = steady_points['dilution_rate'].mean()
            print(f"     Steady-state periods: {len(steady_points)} points")
            print(f"     Mean growth rate (mu): {mean_growth:.4f} h^-1")
            print(f"     Mean dilution rate (D): {mean_dilution:.4f} h^-1")
            print(f"     Ratio mu/D: {mean_growth/mean_dilution:.3f} (should be ~1.0 at steady state)")
        else:
            print(f"     No steady-state periods detected")
        print()

    except Exception as e:
        print(f"[FAIL] Continuous growth rate analysis failed: {e}")
        import traceback
        traceback.print_exc()
        print()

    print("=" * 80)
    print("Testing complete!")
    print("=" * 80)


if __name__ == "__main__":
    import pandas as pd
    main()
