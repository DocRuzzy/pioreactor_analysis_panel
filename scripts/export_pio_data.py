import sqlite3
import pandas as pd
import os
from functools import reduce

# Database configurations
DATABASES = [
    {
        'path': r'data\Pioreactor 1\pioreactor.sqlite',
        'output_name': 'Pioreactor_1'
    },
    {
        'path': r'data\Pioreactor 8\storage\pioreactor.sqlite',
        'output_name': 'Pioreactor_8'
    }
]

OUTPUT_DIR = 'pioreactor_exports'


def sanitize_name(name):
    """Sanitize experiment name for use as folder name."""
    return "".join([c for c in name if c.isalpha() or c.isdigit() or c in (' ', '-', '_')]).strip().replace(' ', '_')


def get_experiments(conn):
    """Get list of all experiments from the database."""
    query = "SELECT DISTINCT experiment FROM od_readings"
    try:
        return pd.read_sql_query(query, conn)['experiment'].tolist()
    except Exception as e:
        print(f"  Could not list experiments: {e}")
        return []


def get_units_for_experiment(conn, experiment):
    """Get list of pioreactor units for an experiment."""
    query = "SELECT DISTINCT pioreactor_unit FROM od_readings WHERE experiment = ?"
    try:
        return pd.read_sql_query(query, conn, params=(experiment,))['pioreactor_unit'].tolist()
    except Exception:
        return ['unknown']


def create_rollup_format_data(conn, experiment, unit):
    """
    Create data in Pioreactor rollup format that the analysis tool expects.

    Output columns match pioreactor_unit_activity_data_rollup format:
    - experiment, pioreactor_unit, timestamp
    - avg_od_reading, avg_normalized_od_reading
    - avg_temperature_c, avg_growth_rate, avg_measured_rpm
    - sum_add_media_ml, sum_remove_waste_ml, sum_add_alt_media_ml
    - timestamp_localtime, hours_since_experiment_created

    Uses OUTER joins to include ALL timestamps from all data sources (OD, dosing, etc.)
    """
    # Get experiment start time
    start_query = "SELECT MIN(timestamp) as start_time FROM od_readings WHERE experiment = ?"
    start_result = pd.read_sql_query(start_query, conn, params=(experiment,))
    experiment_start = pd.to_datetime(start_result['start_time'].iloc[0])

    # Collect all dataframes to merge
    all_dfs = []

    # Raw OD readings
    od_query = """
        SELECT timestamp, od_reading
        FROM od_readings
        WHERE experiment = ? AND pioreactor_unit = ?
        ORDER BY timestamp
    """
    od_df = pd.read_sql_query(od_query, conn, params=(experiment, unit))
    if not od_df.empty:
        od_df['timestamp'] = pd.to_datetime(od_df['timestamp'])
        all_dfs.append(od_df)

    # Normalized OD readings
    try:
        norm_query = """
            SELECT timestamp, normalized_od_reading
            FROM od_readings_filtered
            WHERE experiment = ? AND pioreactor_unit = ?
        """
        norm_df = pd.read_sql_query(norm_query, conn, params=(experiment, unit))
        if not norm_df.empty:
            norm_df['timestamp'] = pd.to_datetime(norm_df['timestamp'])
            all_dfs.append(norm_df)
    except Exception:
        pass

    # Temperature readings
    try:
        temp_query = """
            SELECT timestamp, temperature_c
            FROM temperature_readings
            WHERE experiment = ? AND pioreactor_unit = ?
        """
        temp_df = pd.read_sql_query(temp_query, conn, params=(experiment, unit))
        if not temp_df.empty:
            temp_df['timestamp'] = pd.to_datetime(temp_df['timestamp'])
            all_dfs.append(temp_df)
    except Exception:
        pass

    # Growth rates
    try:
        growth_query = """
            SELECT timestamp, rate as growth_rate
            FROM growth_rates
            WHERE experiment = ? AND pioreactor_unit = ?
        """
        growth_df = pd.read_sql_query(growth_query, conn, params=(experiment, unit))
        if not growth_df.empty:
            growth_df['timestamp'] = pd.to_datetime(growth_df['timestamp'])
            all_dfs.append(growth_df)
    except Exception:
        pass

    # Stirring rates
    try:
        stir_query = """
            SELECT timestamp, measured_rpm
            FROM stirring_rates
            WHERE experiment = ? AND pioreactor_unit = ?
        """
        stir_df = pd.read_sql_query(stir_query, conn, params=(experiment, unit))
        if not stir_df.empty:
            stir_df['timestamp'] = pd.to_datetime(stir_df['timestamp'])
            all_dfs.append(stir_df)
    except Exception:
        pass

    # Dosing events - aggregate by timestamp and create separate columns
    try:
        dosing_query = """
            SELECT timestamp, event, volume_change_ml
            FROM dosing_events
            WHERE experiment = ? AND pioreactor_unit = ?
        """
        dosing_df = pd.read_sql_query(dosing_query, conn, params=(experiment, unit))
        if not dosing_df.empty:
            dosing_df['timestamp'] = pd.to_datetime(dosing_df['timestamp'])

            # Separate add_media and remove_waste events
            add_media = dosing_df[dosing_df['event'].str.contains('add_media', case=False, na=False)]
            remove_waste = dosing_df[dosing_df['event'].str.contains('remove_waste', case=False, na=False)]
            add_alt = dosing_df[dosing_df['event'].str.contains('add_alt', case=False, na=False)]

            if not add_media.empty:
                add_media_agg = add_media.groupby('timestamp')['volume_change_ml'].sum().reset_index()
                add_media_agg.columns = ['timestamp', 'sum_add_media_ml']
                all_dfs.append(add_media_agg)

            if not remove_waste.empty:
                remove_waste_agg = remove_waste.groupby('timestamp')['volume_change_ml'].sum().reset_index()
                remove_waste_agg.columns = ['timestamp', 'sum_remove_waste_ml']
                all_dfs.append(remove_waste_agg)

            if not add_alt.empty:
                add_alt_agg = add_alt.groupby('timestamp')['volume_change_ml'].sum().reset_index()
                add_alt_agg.columns = ['timestamp', 'sum_add_alt_media_ml']
                all_dfs.append(add_alt_agg)
    except Exception:
        pass

    if not all_dfs:
        return pd.DataFrame()

    # Merge all dataframes using OUTER join to include all timestamps
    result_df = all_dfs[0]
    for df in all_dfs[1:]:
        result_df = pd.merge(result_df, df, on='timestamp', how='outer')

    # Sort by timestamp
    result_df = result_df.sort_values('timestamp').reset_index(drop=True)

    # Add required columns for rollup format
    result_df['experiment'] = experiment
    result_df['pioreactor_unit'] = unit

    # Calculate hours_since_experiment_created
    result_df['hours_since_experiment_created'] = (result_df['timestamp'] - experiment_start).dt.total_seconds() / 3600

    # Add timestamp_localtime (same as timestamp for now - adjust timezone if needed)
    result_df['timestamp_localtime'] = result_df['timestamp']

    # Rename columns to match rollup format
    column_mapping = {
        'od_reading': 'avg_od_reading',
        'normalized_od_reading': 'avg_normalized_od_reading',
        'temperature_c': 'avg_temperature_c',
        'growth_rate': 'avg_growth_rate',
        'measured_rpm': 'avg_measured_rpm',
    }
    result_df = result_df.rename(columns=column_mapping)

    # Reorder columns to match expected format
    expected_cols = [
        'experiment', 'pioreactor_unit', 'timestamp',
        'avg_od_reading', 'avg_normalized_od_reading',
        'avg_temperature_c', 'avg_growth_rate', 'avg_measured_rpm',
        'sum_add_media_ml', 'sum_remove_waste_ml', 'sum_add_alt_media_ml',
        'timestamp_localtime', 'hours_since_experiment_created'
    ]

    # Only include columns that exist
    final_cols = [c for c in expected_cols if c in result_df.columns]
    result_df = result_df[final_cols]

    return result_df


def create_simplified_data(conn, experiment, unit):
    """Create simplified dataframe with key metrics: timestamp, elapsed_hours, raw_od, dilution events, logs."""
    # Get experiment start time
    start_query = "SELECT MIN(timestamp) as start_time FROM od_readings WHERE experiment = ?"
    start_result = pd.read_sql_query(start_query, conn, params=(experiment,))
    experiment_start = pd.to_datetime(start_result['start_time'].iloc[0])

    dfs = []

    # Raw OD readings
    try:
        od = pd.read_sql_query(
            "SELECT timestamp, od_reading as raw_od FROM od_readings WHERE experiment = ? AND pioreactor_unit = ?",
            conn, params=(experiment, unit)
        )
        if not od.empty:
            dfs.append(od)
    except Exception:
        pass

    # Dosing events (dilutions)
    try:
        dosing = pd.read_sql_query(
            "SELECT timestamp, event as dilution_event, volume_change_ml as dilution_volume_ml FROM dosing_events WHERE experiment = ? AND pioreactor_unit = ?",
            conn, params=(experiment, unit)
        )
        if not dosing.empty:
            dfs.append(dosing)
    except Exception:
        pass

    # Logs
    try:
        logs = pd.read_sql_query(
            "SELECT timestamp, message as log_message, level as log_level FROM logs WHERE experiment = ? AND pioreactor_unit = ?",
            conn, params=(experiment, unit)
        )
        if not logs.empty:
            dfs.append(logs)
    except Exception:
        pass

    if not dfs:
        return pd.DataFrame()

    # Merge all dataframes on timestamp using outer join
    simplified = reduce(lambda left, right: pd.merge(left, right, on='timestamp', how='outer'), dfs)

    # Convert timestamp and sort
    simplified['timestamp'] = pd.to_datetime(simplified['timestamp'])
    simplified = simplified.sort_values('timestamp').reset_index(drop=True)

    # Calculate elapsed hours from experiment start
    if not simplified.empty:
        simplified['elapsed_hours'] = (simplified['timestamp'] - experiment_start).dt.total_seconds() / 3600

        # Reorder columns
        cols = ['timestamp', 'elapsed_hours', 'raw_od', 'dilution_event', 'dilution_volume_ml', 'log_message', 'log_level']
        # Only include columns that exist
        cols = [c for c in cols if c in simplified.columns]
        simplified = simplified[cols]

    return simplified


def export_database(db_path, output_name):
    """Export all experiments from a single database."""
    if not os.path.exists(db_path):
        print(f"Error: Could not find database at {db_path}")
        return

    print(f"\n{'='*60}")
    print(f"Processing: {db_path}")
    print(f"{'='*60}")

    conn = sqlite3.connect(db_path)
    experiments = get_experiments(conn)

    if not experiments:
        print("No experiments found in database.")
        conn.close()
        return

    print(f"Found {len(experiments)} experiments")

    # Create output directory for this pioreactor
    pioreactor_dir = os.path.join(OUTPUT_DIR, output_name)
    os.makedirs(pioreactor_dir, exist_ok=True)

    for exp in experiments:
        safe_name = sanitize_name(exp)
        exp_dir = os.path.join(pioreactor_dir, safe_name)
        os.makedirs(exp_dir, exist_ok=True)

        print(f"\n  Exporting: {exp}")

        # Get all units for this experiment
        units = get_units_for_experiment(conn, exp)
        print(f"    Units: {units}")

        for unit in units:
            safe_unit = sanitize_name(unit)

            # Create and save rollup format data (compatible with analysis tool)
            rollup = create_rollup_format_data(conn, exp, unit)
            if not rollup.empty:
                # Use naming convention similar to official Pioreactor exports
                rollup_filename = f"pioreactor_unit_activity_data_rollup-{safe_name}-{safe_unit}.csv"
                rollup_path = os.path.join(exp_dir, rollup_filename)
                rollup.to_csv(rollup_path, index=False)
                print(f"    -> {rollup_filename} ({len(rollup)} rows)")
            else:
                print(f"    -> No data for {unit} rollup")

            # Create and save simplified data
            simplified = create_simplified_data(conn, exp, unit)
            if not simplified.empty:
                simplified_filename = f"simplified_data-{safe_name}-{safe_unit}.csv"
                simplified_path = os.path.join(exp_dir, simplified_filename)
                simplified.to_csv(simplified_path, index=False)
                print(f"    -> {simplified_filename} ({len(simplified)} rows)")
            else:
                print(f"    -> No data for {unit} simplified")

    conn.close()
    print(f"\nCompleted export for {output_name}")


def main():
    """Main entry point."""
    print("Pioreactor Data Export Tool")
    print("="*60)

    # Create main output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Process each database
    for db_config in DATABASES:
        export_database(db_config['path'], db_config['output_name'])

    print(f"\n{'='*60}")
    print(f"Export complete! Files saved to: {OUTPUT_DIR}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
