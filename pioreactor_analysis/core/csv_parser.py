"""
CSV parser for Pioreactor data with automatic format detection.

This module provides intelligent parsing of CSV files from Pioreactor experiments,
with auto-detection of different CSV formats and helpful error messages.
"""

import pandas as pd
import numpy as np
import json
import re
from enum import Enum
from pathlib import Path
from typing import Union, List, Tuple, Optional
from datetime import datetime

from pioreactor_analysis.core.data_models import (
    ODReading,
    DilutionEvent,
    BatchGrowthData,
    ContinuousGrowthData,
)


class CSVFormat(Enum):
    """Supported CSV formats."""
    PIOREACTOR_EXPORT_OD = "pioreactor_export_od"  # Official Pioreactor OD export
    PIOREACTOR_EXPORT_DOSING = "pioreactor_export_dosing"  # Official dosing events
    PIOREACTOR_ROLLUP = "pioreactor_rollup"  # Pioreactor unit activity data rollup
    CUSTOM_BATCH = "custom_batch"  # Simple timestamp + OD format
    CUSTOM_CONTINUOUS = "custom_continuous"  # Timestamp + OD + dilution
    UNKNOWN = "unknown"


class CSVFormatError(Exception):
    """Raised when CSV format cannot be determined or parsed."""
    pass


class PioreactorCSVParser:
    """
    Intelligent CSV parser with format auto-detection.

    Supports multiple CSV formats:
    1. Pioreactor Export Format (OD readings)
    2. Pioreactor Export Format (dosing events)
    3. Custom batch format (simple timestamp + OD)
    4. Custom continuous format (timestamp + OD + dilution)

    Example:
        parser = PioreactorCSVParser()
        data = parser.parse("experiment_data.csv")
        # Returns BatchGrowthData or ContinuousGrowthData depending on content
    """

    # Regex patterns for volume extraction from message fields
    VOLUME_PATTERNS = [
        r'cycled\s+([0-9.]+)\s*mL',  # "cycled 0.27 mL"
        r'(?:volume|added|removed|cycled)[:\s]+([0-9.]+)\s*m?L',  # "volume: 0.5 mL"
        r'([0-9.]+)\s*m?L',  # "0.5 mL"
    ]

    def __init__(self, default_reactor_volume_ml: float = 14.0):
        """
        Initialize parser.

        Args:
            default_reactor_volume_ml: Default reactor volume (default: 14 mL for Pioreactor)
        """
        self.default_reactor_volume_ml = default_reactor_volume_ml

    def _columns_with_data(self, df: pd.DataFrame, columns: List[str], min_valid_count: int = 1) -> List[str]:
        """
        Check which columns actually have non-NaN positive numeric data.

        Uses vectorized pandas operations for speed.

        Args:
            df: DataFrame to check
            columns: List of column names to check
            min_valid_count: Minimum number of valid values required (default 1)

        Returns:
            List of column names that have at least min_valid_count positive values
        """
        cols_with_data = []

        for col in columns:
            if col not in df.columns:
                continue

            # Use vectorized operations - much faster than sampling
            series = pd.to_numeric(df[col], errors='coerce')
            valid_count = (series > 0).sum()

            if valid_count >= min_valid_count:
                cols_with_data.append(col)

        return cols_with_data

    def parse(self, filepath: Union[str, Path]) -> Union[BatchGrowthData, ContinuousGrowthData]:
        """
        Parse CSV file and return appropriate data model.

        Args:
            filepath: Path to CSV file

        Returns:
            BatchGrowthData or ContinuousGrowthData depending on file content

        Raises:
            CSVFormatError: If format cannot be determined or parsing fails
            FileNotFoundError: If file doesn't exist
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"CSV file not found: {filepath}")

        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            raise CSVFormatError(f"Failed to read CSV file: {e}")

        if df.empty:
            raise CSVFormatError("CSV file is empty")

        # Detect format
        format_type = self.detect_format(df)

        if format_type == CSVFormat.UNKNOWN:
            raise CSVFormatError(self._generate_helpful_error(df))

        # Parse based on detected format
        if format_type == CSVFormat.PIOREACTOR_EXPORT_OD:
            return self._parse_pioreactor_od_export(df, filepath)
        elif format_type == CSVFormat.PIOREACTOR_EXPORT_DOSING:
            return self._parse_pioreactor_dosing_export(df, filepath)
        elif format_type == CSVFormat.PIOREACTOR_ROLLUP:
            return self._parse_pioreactor_rollup(df, filepath)
        elif format_type == CSVFormat.CUSTOM_BATCH:
            return self._parse_custom_batch(df, filepath)
        elif format_type == CSVFormat.CUSTOM_CONTINUOUS:
            return self._parse_custom_continuous(df, filepath)

        raise CSVFormatError(f"Unsupported format: {format_type}")

    def detect_format(self, df: pd.DataFrame) -> CSVFormat:
        """
        Auto-detect CSV format based on columns and content.

        Args:
            df: Pandas DataFrame loaded from CSV

        Returns:
            CSVFormat enum indicating detected format
        """
        columns = set(df.columns)

        # Check for Pioreactor OD export format
        if {'experiment', 'pioreactor_unit', 'timestamp', 'od_reading'}.issubset(columns):
            return CSVFormat.PIOREACTOR_EXPORT_OD

        # Check for Pioreactor dosing events export format
        if {'experiment', 'pioreactor_unit', 'timestamp', 'event_name'}.issubset(columns):
            # Verify it contains dilution events
            if 'event_name' in df.columns:
                has_dilution = df['event_name'].str.contains('dilution', case=False, na=False).any()
                if has_dilution:
                    return CSVFormat.PIOREACTOR_EXPORT_DOSING

        # Check for Pioreactor rollup format (unit activity data rollup)
        # Has avg_od_reading or avg_normalized_od_reading columns
        if {'experiment', 'pioreactor_unit', 'timestamp'}.issubset(columns):
            if 'avg_od_reading' in columns or 'avg_normalized_od_reading' in columns:
                return CSVFormat.PIOREACTOR_ROLLUP

        # Check for simple batch format (timestamp + OD columns)
        if self._is_batch_format(df):
            return CSVFormat.CUSTOM_BATCH

        # Check for continuous format (has dilution data)
        if self._is_continuous_format(df):
            return CSVFormat.CUSTOM_CONTINUOUS

        return CSVFormat.UNKNOWN

    def _is_batch_format(self, df: pd.DataFrame) -> bool:
        """Check if DataFrame matches simple batch format."""
        columns = set(df.columns)

        # Need timestamp and OD columns
        has_time = any(col for col in columns if 'time' in col.lower())
        has_od = any(col for col in columns if 'od' in col.lower() and 'method' not in col.lower())

        return has_time and has_od

    def _is_continuous_format(self, df: pd.DataFrame) -> bool:
        """Check if DataFrame matches continuous culture format."""
        columns = set(df.columns)

        # Need timestamp, OD, and dilution/volume columns
        has_time = any(col for col in columns if 'time' in col.lower())
        has_od = any(col for col in columns if 'od' in col.lower())
        has_dilution = any(col for col in columns
                          if 'dilution' in col.lower() or 'volume' in col.lower())

        return has_time and has_od and has_dilution

    def _parse_pioreactor_od_export(self, df: pd.DataFrame, filepath: Path) -> BatchGrowthData:
        """Parse official Pioreactor OD reading export format (FAST vectorized version)."""

        # FAST: Pre-filter rows with valid OD values
        od_values = pd.to_numeric(df['od_reading'], errors='coerce')
        valid_mask = od_values.notna()
        df_valid = df.loc[valid_mask].copy()
        od_vals = od_values.loc[valid_mask]

        if df_valid.empty:
            raise CSVFormatError("No valid OD readings found in file")

        # Handle zero or negative OD values (sensor noise) - clip to small positive
        od_vals = od_vals.clip(lower=1e-6)

        # Parse timestamps vectorized
        if 'timestamp_localtime' in df_valid.columns and df_valid['timestamp_localtime'].notna().any():
            timestamps = pd.to_datetime(df_valid['timestamp_localtime'], errors='coerce')
            fallback_mask = timestamps.isna()
            if fallback_mask.any():
                timestamps.loc[fallback_mask] = pd.to_datetime(df_valid.loc[fallback_mask, 'timestamp'], errors='coerce')
        else:
            timestamps = pd.to_datetime(df_valid['timestamp'], errors='coerce')

        # Prepare arrays for fast iteration
        units = df_valid['pioreactor_unit'].astype(str).values
        experiments = df_valid['experiment'].astype(str).values
        ts_values = timestamps.values
        od_values_arr = od_vals.values

        # Handle optional columns
        has_angle = 'angle' in df_valid.columns
        has_channel = 'channel' in df_valid.columns
        angles = df_valid['angle'].values if has_angle else None
        channels = df_valid['channel'].values if has_channel else None

        # Build ODReading objects with skip tracking
        od_readings = []
        skip_reasons = {'nan_timestamp': 0, 'nan_od': 0, 'exception': 0}

        for i in range(len(df_valid)):
            try:
                # Check for specific issues before creating ODReading
                if pd.isna(ts_values[i]):
                    skip_reasons['nan_timestamp'] += 1
                    continue
                if pd.isna(od_values_arr[i]) or od_values_arr[i] <= 0:
                    skip_reasons['nan_od'] += 1
                    continue

                reading = ODReading(
                    timestamp=pd.Timestamp(ts_values[i]).to_pydatetime(),
                    od_value=float(od_values_arr[i]),
                    unit=units[i],
                    experiment=experiments[i],
                    angle=float(angles[i]) if has_angle and pd.notna(angles[i]) else None,
                    channel=int(channels[i]) if has_channel and pd.notna(channels[i]) else None,
                )
                od_readings.append(reading)
            except Exception as e:
                skip_reasons['exception'] += 1
                # Log first few exceptions for debugging
                if skip_reasons['exception'] <= 3:
                    print(f"Warning: Skipped row {i} due to: {e}")
                continue

        if not od_readings:
            raise CSVFormatError("No valid OD readings found in file")

        experiment_id = od_readings[0].experiment

        # Report parsing statistics
        total_skipped = sum(skip_reasons.values())
        print(f"Parsed {len(od_readings)} OD readings from {len(df)} total rows")
        if total_skipped > 0:
            print(f"Skipped {total_skipped} rows: {skip_reasons}")

        return BatchGrowthData(
            od_readings=od_readings,
            experiment_id=experiment_id,
            reactor_volume_ml=self.default_reactor_volume_ml,
            metadata={'source_file': str(filepath)}
        )

    def _parse_pioreactor_dosing_export(self, df: pd.DataFrame, filepath: Path) -> ContinuousGrowthData:
        """
        Parse official Pioreactor dosing events export format.

        Note: This format typically doesn't include OD data, so we return a minimal
        ContinuousGrowthData with only dilution events. User should combine with OD data.
        """
        # Filter for dilution events
        dilution_df = df[df['event_name'].str.contains('dilution', case=False, na=False)].copy()

        if dilution_df.empty:
            raise CSVFormatError("No dilution events found in dosing events file")

        dilution_events = []
        od_readings = []  # Will be empty - user needs to provide OD data separately

        for _, row in dilution_df.iterrows():
            try:
                # Parse timestamp
                if 'timestamp_localtime' in df.columns and pd.notna(row['timestamp_localtime']):
                    timestamp = pd.to_datetime(row['timestamp_localtime'])
                else:
                    timestamp = pd.to_datetime(row['timestamp'])

                # Extract volume from JSON data field or message
                volume_ml = self._extract_volume_from_row(row)

                if volume_ml is None or volume_ml <= 0:
                    print(f"Warning: Could not extract volume from row, skipping")
                    continue

                event = DilutionEvent(
                    timestamp=timestamp,
                    volume_ml=volume_ml,
                    unit=str(row['pioreactor_unit']),
                    experiment=str(row['experiment']),
                    event_name=str(row['event_name']) if 'event_name' in row else None,
                    source=str(row.get('source_of_event', 'unknown')),
                )
                dilution_events.append(event)
            except Exception as e:
                print(f"Warning: Skipping dilution event due to error: {e}")
                continue

        if not dilution_events:
            raise CSVFormatError("No valid dilution events found in file")

        # Extract experiment ID
        experiment_id = dilution_events[0].experiment

        # Return with empty OD readings - user should combine with OD file
        return ContinuousGrowthData(
            od_readings=od_readings,
            dilution_events=dilution_events,
            experiment_id=experiment_id,
            reactor_volume_ml=self.default_reactor_volume_ml,
            metadata={
                'source_file': str(filepath),
                'note': 'OD readings not included - combine with OD data file'
            }
        )

    def _parse_pioreactor_rollup(self, df: pd.DataFrame, filepath: Path) -> Union[BatchGrowthData, ContinuousGrowthData]:
        """
        Parse Pioreactor unit activity data rollup format (FAST vectorized version).

        This format contains aggregated data with columns like:
        - experiment, pioreactor_unit, timestamp
        - avg_od_reading, avg_normalized_od_reading
        - avg_temperature_c, avg_growth_rate, avg_measured_rpm
        - sum_add_media_ml, sum_remove_waste_ml, etc.

        Many rows may have NaN for OD readings (other sensor data rows).
        We filter to only rows with valid OD data using vectorized operations.
        
        Returns:
            ContinuousGrowthData if dilution events are found, BatchGrowthData otherwise.
        """
        # Determine which OD column to use - check if they actually have data
        # IMPORTANT: Prefer raw OD over normalized OD for growth rate analysis
        od_candidates = ['avg_od_reading', 'avg_normalized_od_reading']
        cols_with_data = self._columns_with_data(df, od_candidates)

        if not cols_with_data:
            raise CSVFormatError("No OD column with valid data found in rollup file. "
                               f"Checked columns: {od_candidates}")

        # Use first column that has data (prefer raw OD for accurate growth rate calculations)
        od_col = cols_with_data[0]

        # FAST: Filter rows with valid OD values using vectorized operations
        od_values = pd.to_numeric(df[od_col], errors='coerce')
        valid_mask = od_values.notna() & (od_values > 0)
        df_od = df.loc[valid_mask].copy()

        if df_od.empty:
            raise CSVFormatError(f"No valid OD readings found in column '{od_col}'")

        # Parse timestamps vectorized
        if 'timestamp_localtime' in df_od.columns and df_od['timestamp_localtime'].notna().any():
            timestamps = pd.to_datetime(df_od['timestamp_localtime'], errors='coerce')
            # Fall back to timestamp for rows where localtime failed
            fallback_mask = timestamps.isna()
            if fallback_mask.any():
                timestamps.loc[fallback_mask] = pd.to_datetime(df_od.loc[fallback_mask, 'timestamp'], errors='coerce')
        else:
            timestamps = pd.to_datetime(df_od['timestamp'], errors='coerce')

        # Extract OD values (already validated)
        od_vals = od_values.loc[valid_mask]

        # Build ODReading objects vectorized (much faster than iterrows)
        od_readings = []
        units = df_od['pioreactor_unit'].astype(str).values
        experiments = df_od['experiment'].astype(str).values
        ts_values = timestamps.values
        od_values_arr = od_vals.values

        for i in range(len(df_od)):
            try:
                reading = ODReading(
                    timestamp=pd.Timestamp(ts_values[i]).to_pydatetime(),
                    od_value=float(od_values_arr[i]),
                    unit=units[i],
                    experiment=experiments[i],
                    angle=None,
                    channel=None,
                )
                od_readings.append(reading)
            except Exception:
                # Skip silently - already filtered for valid data
                continue

        if not od_readings:
            raise CSVFormatError("No valid OD readings found in rollup file after filtering")

        # Extract experiment ID from first reading
        experiment_id = od_readings[0].experiment

        # Extract dilution events from sum_add_media_ml and sum_remove_waste_ml columns
        dilution_events = self._extract_dilution_events_from_rollup(df, filepath)

        print(f"Parsed {len(od_readings)} OD readings from {len(df)} total rows (column: {od_col})")
        if dilution_events:
            print(f"Extracted {len(dilution_events)} dilution events from rollup data")

        metadata = {
            'source_file': str(filepath),
            'format': 'pioreactor_rollup',
            'od_column_used': od_col,
            'total_rows': len(df),
            'valid_od_rows': len(od_readings)
        }

        # Return ContinuousGrowthData if we have dilution events, otherwise BatchGrowthData
        if dilution_events:
            return ContinuousGrowthData(
                od_readings=od_readings,
                dilution_events=dilution_events,
                experiment_id=experiment_id,
                reactor_volume_ml=self.default_reactor_volume_ml,
                metadata=metadata
            )
        else:
            return BatchGrowthData(
                od_readings=od_readings,
                experiment_id=experiment_id,
                reactor_volume_ml=self.default_reactor_volume_ml,
                metadata=metadata
            )

    def _extract_dilution_events_from_rollup(self, df: pd.DataFrame, filepath: Path) -> List[DilutionEvent]:
        """
        Extract dilution events from rollup data columns like sum_add_media_ml, sum_remove_waste_ml.
        
        The rollup format aggregates dosing data at minute resolution. Each row with a non-zero
        sum_add_media_ml or sum_remove_waste_ml value represents one or more dilution events.
        We use sum_add_media_ml as the primary indicator of a dilution event.
        """
        dilution_events = []
        
        # Check which dilution columns are available
        add_media_col = 'sum_add_media_ml' if 'sum_add_media_ml' in df.columns else None
        add_alt_media_col = 'sum_add_alt_media_ml' if 'sum_add_alt_media_ml' in df.columns else None
        
        if not add_media_col and not add_alt_media_col:
            # No dilution columns found
            return dilution_events
        
        # Use the column with more data
        dilution_col = add_media_col
        if add_alt_media_col:
            alt_count = pd.to_numeric(df[add_alt_media_col], errors='coerce').notna().sum()
            main_count = pd.to_numeric(df[add_media_col], errors='coerce').notna().sum() if add_media_col else 0
            if alt_count > main_count:
                dilution_col = add_alt_media_col
        
        if not dilution_col:
            return dilution_events
        
        # Filter rows with valid dilution volumes
        volumes = pd.to_numeric(df[dilution_col], errors='coerce')
        valid_mask = volumes.notna() & (volumes > 0)
        df_dilution = df.loc[valid_mask].copy()
        
        if df_dilution.empty:
            return dilution_events
        
        # Parse timestamps
        if 'timestamp_localtime' in df_dilution.columns and df_dilution['timestamp_localtime'].notna().any():
            timestamps = pd.to_datetime(df_dilution['timestamp_localtime'], errors='coerce')
            fallback_mask = timestamps.isna()
            if fallback_mask.any():
                timestamps.loc[fallback_mask] = pd.to_datetime(df_dilution.loc[fallback_mask, 'timestamp'], errors='coerce')
        else:
            timestamps = pd.to_datetime(df_dilution['timestamp'], errors='coerce')
        
        # Build DilutionEvent objects
        units = df_dilution['pioreactor_unit'].astype(str).values
        experiments = df_dilution['experiment'].astype(str).values
        ts_values = timestamps.values
        volume_values = volumes.loc[valid_mask].values
        
        # Track unique events (same timestamp + unit = same event, use max volume)
        seen_events = {}  # key: (timestamp, unit) -> max volume
        
        for i in range(len(df_dilution)):
            try:
                ts = pd.Timestamp(ts_values[i]).to_pydatetime()
                vol = float(volume_values[i])
                unit = units[i]
                exp = experiments[i]
                key = (ts, unit)
                
                # Keep the maximum volume for duplicate timestamps (aggregation artifact)
                if key in seen_events:
                    if vol > seen_events[key][0]:
                        seen_events[key] = (vol, unit, exp)
                else:
                    seen_events[key] = (vol, unit, exp)
            except Exception:
                continue
        
        # Create DilutionEvent objects from deduplicated data
        for (ts, unit), (vol, unit_val, exp) in seen_events.items():
            try:
                event = DilutionEvent(
                    timestamp=ts,
                    volume_ml=vol,
                    unit=unit_val,
                    experiment=exp,
                    event_name='media_addition',
                    source='rollup_aggregate'
                )
                dilution_events.append(event)
            except Exception:
                continue
        
        # Sort by timestamp
        dilution_events.sort(key=lambda e: e.timestamp)
        
        return dilution_events

    def _extract_volume_from_row(self, row: pd.Series) -> Optional[float]:
        """
        Extract volume from a dosing event row.

        Tries multiple methods:
        1. Parse JSON from 'data' field
        2. Regex patterns on 'message' field
        3. Direct 'volume' or 'volume_ml' column

        Returns:
            Volume in mL, or None if extraction fails
        """
        # Method 1: Try JSON data field
        if 'data' in row and pd.notna(row['data']):
            try:
                data_dict = json.loads(row['data'])
                if 'exchange_volume_ml' in data_dict:
                    return float(data_dict['exchange_volume_ml'])
                if 'volume' in data_dict:
                    return float(data_dict['volume'])
                if 'volume_ml' in data_dict:
                    return float(data_dict['volume_ml'])
            except (json.JSONDecodeError, ValueError, KeyError):
                pass

        # Method 2: Try regex on message field
        if 'message' in row and pd.notna(row['message']):
            message = str(row['message'])
            for pattern in self.VOLUME_PATTERNS:
                match = re.search(pattern, message, re.IGNORECASE)
                if match:
                    try:
                        return float(match.group(1))
                    except (ValueError, IndexError):
                        continue

        # Method 3: Try direct volume columns
        for vol_col in ['volume_ml', 'volume', 'volume_change_ml']:
            if vol_col in row and pd.notna(row[vol_col]):
                try:
                    return float(row[vol_col])
                except ValueError:
                    continue

        return None

    def _parse_custom_batch(self, df: pd.DataFrame, filepath: Path) -> BatchGrowthData:
        """Parse custom batch format (simple timestamp + OD) - FAST vectorized version."""
        # Find timestamp and OD columns
        time_col = next((col for col in df.columns if 'time' in col.lower()), None)
        od_col = next((col for col in df.columns if 'od' in col.lower() and 'method' not in col.lower()), None)

        if time_col is None or od_col is None:
            raise CSVFormatError("Could not identify timestamp and OD columns in custom format")

        # Check if OD column actually has data
        cols_with_data = self._columns_with_data(df, [od_col])
        if not cols_with_data:
            raise CSVFormatError(f"OD column '{od_col}' has no valid data")

        # Find unit/experiment columns if they exist
        unit_col = next((col for col in df.columns if 'unit' in col.lower()), None)
        exp_col = next((col for col in df.columns if 'exp' in col.lower()), None)

        default_unit = "unit1"
        default_experiment = filepath.stem  # Use filename as experiment ID

        # FAST: Pre-filter rows with valid OD values
        od_values = pd.to_numeric(df[od_col], errors='coerce')
        valid_mask = od_values.notna() & (od_values > 0)
        df_valid = df.loc[valid_mask].copy()
        od_vals = od_values.loc[valid_mask]

        if df_valid.empty:
            raise CSVFormatError(f"No valid OD readings found in column '{od_col}'")

        # Handle zero values - clip to small positive
        od_vals = od_vals.clip(lower=1e-6)

        # Parse timestamps vectorized
        timestamps = pd.to_datetime(df_valid[time_col], errors='coerce')

        # Prepare arrays
        ts_values = timestamps.values
        od_values_arr = od_vals.values

        # Handle unit/experiment columns
        if unit_col and unit_col in df_valid.columns:
            units = df_valid[unit_col].fillna(default_unit).astype(str).values
        else:
            units = [default_unit] * len(df_valid)

        if exp_col and exp_col in df_valid.columns:
            experiments = df_valid[exp_col].fillna(default_experiment).astype(str).values
        else:
            experiments = [default_experiment] * len(df_valid)

        # Build ODReading objects
        od_readings = []
        for i in range(len(df_valid)):
            try:
                ts = pd.Timestamp(ts_values[i])
                if pd.isna(ts):
                    continue
                reading = ODReading(
                    timestamp=ts.to_pydatetime(),
                    od_value=float(od_values_arr[i]),
                    unit=units[i] if isinstance(units, list) else units[i],
                    experiment=experiments[i] if isinstance(experiments, list) else experiments[i],
                )
                od_readings.append(reading)
            except Exception:
                continue

        if not od_readings:
            raise CSVFormatError("No valid OD readings found in custom batch format")

        print(f"Parsed {len(od_readings)} OD readings from {len(df)} total rows (column: {od_col})")

        return BatchGrowthData(
            od_readings=od_readings,
            experiment_id=default_experiment,
            reactor_volume_ml=self.default_reactor_volume_ml,
            metadata={'source_file': str(filepath), 'format': 'custom_batch'}
        )

    def _parse_custom_continuous(self, df: pd.DataFrame, filepath: Path) -> ContinuousGrowthData:
        """Parse custom continuous culture format."""
        # This would need to be implemented based on specific custom format
        # For now, raise an error directing user to use standard format
        raise CSVFormatError(
            "Custom continuous culture format detected but not yet fully implemented.\n"
            "Please use Pioreactor export format (separate OD and dosing event files) "
            "or contact support for assistance with your custom format."
        )

    def _generate_helpful_error(self, df: pd.DataFrame) -> str:
        """
        Generate helpful error message when format cannot be determined.

        Args:
            df: The problematic DataFrame

        Returns:
            Detailed error message with suggestions
        """
        columns = list(df.columns)
        sample_data = df.head(3).to_string()

        msg = [
            "❌ Unable to parse CSV file - format not recognized.",
            "",
            f"Found columns: {columns}",
            "",
            "Expected one of these formats:",
            "",
            "1. Pioreactor OD Export:",
            "   Required columns: experiment, pioreactor_unit, timestamp, od_reading",
            "   Example: Exported from Pioreactor UI → Data → Export",
            "",
            "2. Pioreactor Dosing Events Export:",
            "   Required columns: experiment, pioreactor_unit, timestamp, event_name, data/message",
            "   Must contain 'dilution' events",
            "",
            "3. Custom Batch Format:",
            "   Required: Any column with 'time' + any column with 'od'",
            "   Example columns: timestamp, od_value OR time_hours, optical_density",
            "",
            "Sample of your data:",
            sample_data,
            "",
            "💡 Suggestions:",
        ]

        # Provide specific suggestions based on what's missing
        has_time = any('time' in col.lower() for col in columns)
        has_od = any('od' in col.lower() for col in columns)
        has_experiment = any('exp' in col.lower() for col in columns)

        if not has_time:
            msg.append("   - Add a timestamp column (e.g., 'timestamp', 'time', 'time_hours')")
        if not has_od:
            msg.append("   - Add an OD column (e.g., 'od_reading', 'od_value', 'optical_density')")
        if not has_experiment and len(columns) > 2:
            msg.append("   - For Pioreactor export format, ensure 'experiment' and 'pioreactor_unit' columns exist")

        msg.append("")
        msg.append("📖 For more help, see documentation or contact support")

        return "\n".join(msg)

    def parse_combined_od_and_dosing(
        self,
        od_filepath: Union[str, Path],
        dosing_filepath: Union[str, Path]
    ) -> ContinuousGrowthData:
        """
        Parse separate OD and dosing event files and combine them.

        This is the recommended approach for continuous culture analysis
        when using Pioreactor export format.

        Args:
            od_filepath: Path to OD readings CSV
            dosing_filepath: Path to dosing events CSV

        Returns:
            ContinuousGrowthData with both OD readings and dilution events

        Example:
            parser = PioreactorCSVParser()
            data = parser.parse_combined_od_and_dosing(
                "od_readings.csv",
                "dosing_events.csv"
            )
        """
        # Parse OD file
        od_data = self.parse(od_filepath)
        if not isinstance(od_data, BatchGrowthData):
            raise CSVFormatError(f"OD file {od_filepath} does not contain OD readings")

        # Parse dosing file
        dosing_data = self.parse(dosing_filepath)
        if not isinstance(dosing_data, ContinuousGrowthData):
            raise CSVFormatError(f"Dosing file {dosing_filepath} does not contain dilution events")

        # Combine into ContinuousGrowthData
        return ContinuousGrowthData(
            od_readings=od_data.od_readings,
            dilution_events=dosing_data.dilution_events,
            experiment_id=od_data.experiment_id,
            reactor_volume_ml=od_data.reactor_volume_ml,
            metadata={
                'od_source_file': str(od_filepath),
                'dosing_source_file': str(dosing_filepath)
            }
        )


def parse_pioreactor_csv(filepath: Union[str, Path]) -> Union[BatchGrowthData, ContinuousGrowthData]:
    """
    Convenience function to parse a Pioreactor CSV file.

    Args:
        filepath: Path to CSV file

    Returns:
        BatchGrowthData or ContinuousGrowthData

    Example:
        from pioreactor_analysis import parse_pioreactor_csv

        data = parse_pioreactor_csv("my_experiment.csv")
        df = data.to_dataframe()
    """
    parser = PioreactorCSVParser()
    return parser.parse(filepath)
