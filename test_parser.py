"""
Test script for CSV parser with existing Pioreactor data.
"""

import sys
from pathlib import Path

# Add pioreactor_analysis to path
sys.path.insert(0, str(Path(__file__).parent))

from pioreactor_analysis.core.csv_parser import PioreactorCSVParser
from pioreactor_analysis.core.preprocessing import preprocess_od_data

def main():
    print("=" * 80)
    print("Testing Pioreactor CSV Parser")
    print("=" * 80)
    print()

    parser = PioreactorCSVParser()

    # Test 1: Parse OD readings file
    print("Test 1: Parsing OD readings file...")
    print("-" * 80)
    od_file = Path("data/export_20260109160338_p3/od_readings/od_readings-in_class_10.29-all_units-20260109110344.csv")

    try:
        od_data = parser.parse(od_file)
        print(f"[OK] Successfully parsed OD file!")
        print(f"  Format detected: {parser.detect_format(pd.read_csv(od_file))}")
        print(f"  Experiment ID: {od_data.experiment_id}")
        print(f"  Number of readings: {len(od_data.od_readings)}")
        print(f"  Units: {od_data.get_units()}")
        print(f"  Time range: {od_data.od_readings[0].timestamp} to {od_data.od_readings[-1].timestamp}")
        print(f"  OD range: {min(r.od_value for r in od_data.od_readings):.4f} to {max(r.od_value for r in od_data.od_readings):.4f}")
        print()

        # Convert to DataFrame
        df = od_data.to_dataframe()
        print(f"  DataFrame shape: {df.shape}")
        print(f"  DataFrame columns: {list(df.columns)}")
        print()
        print("  First few rows:")
        print(df.head().to_string())
        print()

    except Exception as e:
        print(f"[FAIL] Failed to parse OD file: {e}")
        import traceback
        traceback.print_exc()
        print()

    # Test 2: Parse dosing events file
    print("\nTest 2: Parsing dosing events file...")
    print("-" * 80)
    dosing_file = Path("data/export_20260109160338_p3/dosing_automation_events/dosing_automation_events-in_class_10.29-all_units-20260109110344.csv")

    try:
        dosing_data = parser.parse(dosing_file)
        print(f"[OK] Successfully parsed dosing file!")
        print(f"  Experiment ID: {dosing_data.experiment_id}")
        print(f"  Number of dilution events: {len(dosing_data.dilution_events)}")
        print(f"  Units: {dosing_data.get_units()}")
        print(f"  Volume range: {min(e.volume_ml for e in dosing_data.dilution_events):.4f} to {max(e.volume_ml for e in dosing_data.dilution_events):.4f} mL")
        print()

        # Convert to DataFrame
        df = dosing_data.to_dilution_dataframe()
        print(f"  DataFrame shape: {df.shape}")
        print(f"  DataFrame columns: {list(df.columns)}")
        print()
        print("  First few rows:")
        print(df.head().to_string())
        print()

    except Exception as e:
        print(f"[FAIL] Failed to parse dosing file: {e}")
        import traceback
        traceback.print_exc()
        print()

    # Test 3: Combine OD and dosing data
    print("\nTest 3: Combining OD and dosing data...")
    print("-" * 80)

    try:
        combined_data = parser.parse_combined_od_and_dosing(od_file, dosing_file)
        print(f"[OK] Successfully combined data!")
        print(f"  Experiment ID: {combined_data.experiment_id}")
        print(f"  Number of OD readings: {len(combined_data.od_readings)}")
        print(f"  Number of dilution events: {len(combined_data.dilution_events)}")
        print()

    except Exception as e:
        print(f"[FAIL] Failed to combine data: {e}")
        import traceback
        traceback.print_exc()
        print()

    # Test 4: Preprocessing
    print("\nTest 4: Testing preprocessing functions...")
    print("-" * 80)

    try:
        od_data = parser.parse(od_file)
        df = od_data.to_dataframe()

        # Rename columns to match preprocessing expectations
        df = df.rename(columns={'od_value': 'od_value'})

        # Preprocess
        df_processed = preprocess_od_data(
            df,
            smoothing_window=5,
            min_od_threshold=0.05,
            time_column='timestamp',
            od_column='od_value',
            group_by=['experiment', 'unit']
        )

        print(f"[OK] Successfully preprocessed data!")
        print(f"  Original points: {len(df)}")
        print(f"  After preprocessing: {len(df_processed)}")
        print(f"  New columns added: {[col for col in df_processed.columns if col not in df.columns]}")
        print()
        print("  Sample of processed data:")
        display_cols = [col for col in ['timestamp', 'elapsed_hours', 'od_value', 'od_smooth', 'ln_od'] if col in df_processed.columns]
        print(df_processed[display_cols].head(10).to_string())
        print()

    except Exception as e:
        print(f"[FAIL] Failed to preprocess data: {e}")
        import traceback
        traceback.print_exc()
        print()

    print("=" * 80)
    print("Testing complete!")
    print("=" * 80)


if __name__ == "__main__":
    import pandas as pd
    main()
