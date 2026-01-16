"""
Data models for Pioreactor analysis using Pydantic for validation.

These models provide type-safe data structures for working with Pioreactor
experimental data, including OD readings, dilution events, and analysis results.
"""

from pydantic import BaseModel, Field, field_validator, ConfigDict
from datetime import datetime
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np


class ODReading(BaseModel):
    """
    Single optical density (OD) measurement.

    Attributes:
        timestamp: When the measurement was taken
        od_value: Optical density reading (must be positive)
        unit: Piore

actor unit identifier (e.g., "pioreactor1")
        experiment: Experiment identifier
        angle: Optional - measurement angle (degrees)
        channel: Optional - measurement channel
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    timestamp: datetime
    od_value: float = Field(gt=0, description="OD must be positive")
    unit: str
    experiment: str
    angle: Optional[float] = None
    channel: Optional[int] = None

    @field_validator('od_value')
    @classmethod
    def validate_od(cls, v: float) -> float:
        if not np.isfinite(v):
            raise ValueError("OD value must be finite")
        return v


class DilutionEvent(BaseModel):
    """
    Single dilution event (media addition).

    Attributes:
        timestamp: When the dilution occurred
        volume_ml: Volume of media added (mL, must be positive)
        unit: Pioreactor unit identifier
        experiment: Experiment identifier
        event_name: Optional - name/type of the event
        source: Optional - source of event (e.g., "dosing_automation", "UI")
    """
    timestamp: datetime
    volume_ml: float = Field(gt=0, description="Volume must be positive")
    unit: str
    experiment: str
    event_name: Optional[str] = None
    source: Optional[str] = None

    @field_validator('volume_ml')
    @classmethod
    def validate_volume(cls, v: float) -> float:
        if not np.isfinite(v):
            raise ValueError("Volume must be finite")
        if v > 50:  # Sanity check - unlikely to add >50mL to a 14mL reactor
            raise ValueError(f"Volume {v} mL seems unusually large. Typical reactor volume is 14 mL.")
        return v


class BatchGrowthData(BaseModel):
    """
    Complete dataset for batch culture growth rate analysis.

    Attributes:
        od_readings: List of OD measurements
        experiment_id: Unique experiment identifier
        reactor_volume_ml: Volume of the reactor (default 14 mL for Pioreactor)
        initial_substrate_conc_gl: Optional - initial substrate concentration (g/L)
        metadata: Optional additional metadata
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    od_readings: List[ODReading]
    experiment_id: str
    reactor_volume_ml: float = 14.0
    initial_substrate_conc_gl: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator('od_readings')
    @classmethod
    def validate_od_readings(cls, v: List[ODReading]) -> List[ODReading]:
        if len(v) < 5:
            raise ValueError("Need at least 5 OD readings for meaningful analysis")
        return v

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert to pandas DataFrame for analysis.

        Returns:
            DataFrame with columns: timestamp, od_value, unit, experiment, etc.
        """
        data = []
        for reading in self.od_readings:
            data.append({
                'timestamp': reading.timestamp,
                'od_value': reading.od_value,
                'unit': reading.unit,
                'experiment': reading.experiment,
                'angle': reading.angle,
                'channel': reading.channel,
            })
        return pd.DataFrame(data)

    def get_units(self) -> List[str]:
        """Get list of unique pioreactor units in this dataset."""
        return list(set(reading.unit for reading in self.od_readings))


class ContinuousGrowthData(BaseModel):
    """
    Complete dataset for continuous culture (chemostat/turbidostat) analysis.

    Attributes:
        od_readings: List of OD measurements
        dilution_events: List of dilution events (media additions)
        experiment_id: Unique experiment identifier
        reactor_volume_ml: Volume of the reactor (default 14 mL)
        metadata: Optional additional metadata
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    od_readings: List[ODReading]
    dilution_events: List[DilutionEvent]
    experiment_id: str
    reactor_volume_ml: float = 14.0
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator('od_readings')
    @classmethod
    def validate_od_readings(cls, v: List[ODReading]) -> List[ODReading]:
        # Allow empty od_readings when parsing dosing-only files
        # User should combine with OD file using parse_combined_od_and_dosing()
        if len(v) > 0 and len(v) < 10:
            raise ValueError(
                f"Need at least 10 OD readings for continuous culture analysis, but got {len(v)}. "
                f"This may indicate: (1) OD values were filtered out (must be > 0), "
                f"(2) Timestamp parsing failed for some rows, or "
                f"(3) This is a dosing-only file (set od_readings to empty list). "
                f"Use the 'Diagnose CSV' button to inspect your file."
            )
        return v

    @field_validator('dilution_events')
    @classmethod
    def validate_dilution_events(cls, v: List[DilutionEvent]) -> List[DilutionEvent]:
        if len(v) < 3:
            raise ValueError("Need at least 3 dilution events for continuous culture analysis")
        return v

    def to_od_dataframe(self) -> pd.DataFrame:
        """Convert OD readings to DataFrame."""
        data = []
        for reading in self.od_readings:
            data.append({
                'timestamp': reading.timestamp,
                'od_value': reading.od_value,
                'unit': reading.unit,
                'experiment': reading.experiment,
                'angle': reading.angle,
                'channel': reading.channel,
            })
        return pd.DataFrame(data)

    def to_dilution_dataframe(self) -> pd.DataFrame:
        """Convert dilution events to DataFrame."""
        data = []
        for event in self.dilution_events:
            data.append({
                'timestamp': event.timestamp,
                'volume_ml': event.volume_ml,
                'unit': event.unit,
                'experiment': event.experiment,
                'event_name': event.event_name,
                'source': event.source,
            })
        return pd.DataFrame(data)

    def get_units(self) -> List[str]:
        """Get list of unique pioreactor units in this dataset."""
        od_units = set(reading.unit for reading in self.od_readings)
        dilution_units = set(event.unit for event in self.dilution_events)
        return list(od_units | dilution_units)


class GrowthRateResult(BaseModel):
    """
    Results from exponential growth rate calculation.

    Based on linear regression of ln(OD) vs time during exponential phase.
    Theory: ln(OD) = μ·t + ln(OD₀), where μ is the specific growth rate.

    Attributes:
        growth_rate: Specific growth rate μ (h⁻¹)
        doubling_time: Population doubling time td = ln(2)/μ (hours)
        r_squared: Coefficient of determination (goodness of fit)
        std_error: Standard error of the growth rate estimate
        ci_lower: Lower bound of 95% confidence interval
        ci_upper: Upper bound of 95% confidence interval
        p_value: P-value for slope significance test
        n_points: Number of data points used in regression
        start_time: Start of analysis window (hours)
        end_time: End of analysis window (hours)
        intercept: Y-intercept of the regression (ln(OD₀))
        unit: Pioreactor unit identifier
        experiment: Experiment identifier
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    growth_rate: float  # h^-1
    doubling_time: float  # hours
    r_squared: float = Field(ge=0, le=1)
    std_error: float = Field(ge=0)
    ci_lower: float  # 95% confidence interval
    ci_upper: float
    p_value: float = Field(ge=0, le=1)
    n_points: int = Field(gt=0)
    start_time: float
    end_time: float
    intercept: float
    unit: Optional[str] = None
    experiment: Optional[str] = None

    @field_validator('growth_rate')
    @classmethod
    def validate_growth_rate(cls, v: float) -> float:
        if not np.isfinite(v):
            raise ValueError("Growth rate must be finite")
        if v < 0:
            raise ValueError("Growth rate cannot be negative during exponential phase")
        if v > 10.0:  # Sanity check - extremely fast growth (td < 4 min is very unlikely)
            raise ValueError(f"Growth rate {v:.4f} h^-1 seems unusually high (td = {np.log(2)/v*60:.1f} min)")
        return v

    @field_validator('doubling_time')
    @classmethod
    def validate_doubling_time(cls, v: float) -> float:
        if not np.isfinite(v):
            raise ValueError("Doubling time must be finite")
        if v <= 0:
            raise ValueError("Doubling time must be positive")
        return v

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for easy export/display."""
        return {
            'growth_rate_h-1': self.growth_rate,
            'doubling_time_h': self.doubling_time,
            'doubling_time_min': self.doubling_time * 60,
            'r_squared': self.r_squared,
            'std_error': self.std_error,
            'ci_95_lower': self.ci_lower,
            'ci_95_upper': self.ci_upper,
            'p_value': self.p_value,
            'n_points': self.n_points,
            'start_time_h': self.start_time,
            'end_time_h': self.end_time,
            'intercept': self.intercept,
            'unit': self.unit,
            'experiment': self.experiment,
        }

    def __str__(self) -> str:
        """Human-readable string representation."""
        return (
            f"Growth Rate Analysis Results:\n"
            f"  μ = {self.growth_rate:.4f} h⁻¹ (95% CI: [{self.ci_lower:.4f}, {self.ci_upper:.4f}])\n"
            f"  Doubling time = {self.doubling_time:.2f} h ({self.doubling_time*60:.1f} min)\n"
            f"  R² = {self.r_squared:.4f}\n"
            f"  Time window: {self.start_time:.2f} - {self.end_time:.2f} h\n"
            f"  N points: {self.n_points}"
        )


class YieldResult(BaseModel):
    """
    Results from apparent biomass yield calculation.

    Yield (Y) represents the efficiency of substrate conversion to biomass.
    Y = ΔX / ΔS, where X is biomass and S is substrate.

    Attributes:
        yield_g_biomass_per_g_substrate: Apparent yield coefficient
        initial_od: Initial OD value
        max_od: Maximum OD reached
        delta_od: Change in OD (max - initial)
        initial_substrate_gl: Initial substrate concentration (g/L)
        substrate_consumed_gl: Estimated substrate consumed (g/L)
        unit: Pioreactor unit identifier
        experiment: Experiment identifier
    """
    yield_g_biomass_per_g_substrate: float = Field(ge=0)
    initial_od: float = Field(gt=0)
    max_od: float = Field(gt=0)
    delta_od: float = Field(gt=0)
    initial_substrate_gl: float = Field(gt=0)
    substrate_consumed_gl: float = Field(ge=0)
    unit: Optional[str] = None
    experiment: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for easy export/display."""
        return {
            'yield_g_biomass_per_g_substrate': self.yield_g_biomass_per_g_substrate,
            'initial_od': self.initial_od,
            'max_od': self.max_od,
            'delta_od': self.delta_od,
            'initial_substrate_g_L': self.initial_substrate_gl,
            'substrate_consumed_g_L': self.substrate_consumed_gl,
            'unit': self.unit,
            'experiment': self.experiment,
        }

    def __str__(self) -> str:
        """Human-readable string representation."""
        return (
            f"Apparent Yield Analysis:\n"
            f"  Y = {self.yield_g_biomass_per_g_substrate:.3f} g biomass / g substrate\n"
            f"  Initial OD: {self.initial_od:.3f} → Max OD: {self.max_od:.3f} (ΔOD = {self.delta_od:.3f})\n"
            f"  Substrate consumed: {self.substrate_consumed_gl:.2f} g/L"
        )


class DilutionRateData(BaseModel):
    """
    Processed dilution rate data for continuous culture analysis.

    Attributes:
        timestamps: List of timestamps
        instant_dilution_rates: Instantaneous dilution rates (h⁻¹)
        moving_avg_dilution_rates: Smoothed dilution rates (h⁻¹)
        volumes_ml: Volumes added at each event (mL)
        reactor_volume_ml: Reactor volume used in calculation
        moving_avg_window: Window size for moving average
        unit: Pioreactor unit identifier
        experiment: Experiment identifier
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    timestamps: List[datetime]
    instant_dilution_rates: List[float]
    moving_avg_dilution_rates: List[float]
    volumes_ml: List[float]
    reactor_volume_ml: float = 14.0
    moving_avg_window: int = 5
    unit: Optional[str] = None
    experiment: Optional[str] = None

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame for analysis."""
        return pd.DataFrame({
            'timestamp': self.timestamps,
            'instant_dilution_rate': self.instant_dilution_rates,
            'moving_avg_dilution_rate': self.moving_avg_dilution_rates,
            'volume_ml': self.volumes_ml,
        })

    def get_mean_dilution_rate(self) -> float:
        """Calculate mean dilution rate over the period."""
        return float(np.mean([r for r in self.moving_avg_dilution_rates if np.isfinite(r)]))

    def get_steady_state_dilution_rate(self, threshold: float = 0.05) -> Optional[float]:
        """
        Calculate steady-state dilution rate.

        Steady state is defined as period where dilution rate variation
        is below the threshold (default 5% relative standard deviation).

        Args:
            threshold: Relative standard deviation threshold for steady state

        Returns:
            Mean dilution rate during steady state, or None if no steady state found
        """
        rates = np.array([r for r in self.moving_avg_dilution_rates if np.isfinite(r)])
        if len(rates) < 10:
            return None

        # Rolling window to find steady state
        window = min(10, len(rates) // 3)
        for i in range(len(rates) - window):
            window_rates = rates[i:i+window]
            rel_std = np.std(window_rates) / np.mean(window_rates)
            if rel_std < threshold:
                return float(np.mean(window_rates))

        return None
