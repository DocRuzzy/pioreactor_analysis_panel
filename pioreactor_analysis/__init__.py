"""
Pioreactor Analysis Library

A comprehensive library for analyzing Pioreactor experimental data including:
- Batch culture growth rate analysis
- Continuous culture (chemostat/turbidostat) analysis
- Dilution rate calculations
- Publication-quality figure generation

Author: Pioreactor Analysis Team
Version: 2.0.0
"""

from pioreactor_analysis.core.data_models import (
    ODReading,
    DilutionEvent,
    BatchGrowthData,
    ContinuousGrowthData,
    GrowthRateResult,
    YieldResult,
)

from pioreactor_analysis.core.csv_parser import (
    PioreactorCSVParser,
    CSVFormat,
    CSVFormatError,
)

from pioreactor_analysis.core.preprocessing import (
    smooth_od_data,
    filter_by_threshold,
    calculate_elapsed_time,
    preprocess_od_data,
)

from pioreactor_analysis.analysis.batch_growth import (
    calculate_batch_growth_rate,
    auto_detect_exponential_phase,
    calculate_apparent_yield,
    batch_analyze_multiple_units,
)

from pioreactor_analysis.analysis.dilution_rate import (
    calculate_dilution_rate,
    calculate_dilution_rate_statistics,
    detect_steady_state,
    align_od_with_dilution_rate,
)

from pioreactor_analysis.analysis.continuous_growth import (
    calculate_growth_rate_continuous,
)

from pioreactor_analysis.plotting.themes import (
    PlotConfig,
    JournalTheme,
    get_color_palette,
)

from pioreactor_analysis.plotting.publication import (
    PublicationPlotter,
)

__version__ = "2.0.0"
__all__ = [
    # Data models
    "ODReading",
    "DilutionEvent",
    "BatchGrowthData",
    "ContinuousGrowthData",
    "GrowthRateResult",
    "YieldResult",
    # CSV parsing
    "PioreactorCSVParser",
    "CSVFormat",
    "CSVFormatError",
    # Preprocessing
    "smooth_od_data",
    "filter_by_threshold",
    "calculate_elapsed_time",
    "preprocess_od_data",
    # Batch growth analysis
    "calculate_batch_growth_rate",
    "auto_detect_exponential_phase",
    "calculate_apparent_yield",
    "batch_analyze_multiple_units",
    # Dilution rate analysis
    "calculate_dilution_rate",
    "calculate_dilution_rate_statistics",
    "detect_steady_state",
    "align_od_with_dilution_rate",
    # Continuous culture analysis
    "calculate_growth_rate_continuous",
    # Publication plotting
    "PlotConfig",
    "JournalTheme",
    "PublicationPlotter",
    "get_color_palette",
]
