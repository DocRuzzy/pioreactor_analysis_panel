# batch_growth_rate_analysis_refactored.py
"""
Interactive Growth Rate Analysis Tool (Refactored with Core Library)

This is the refactored version that uses the pioreactor_analysis core library
for all calculations, making it a thin UI wrapper around pure analysis functions.

New features:
- Better CSV parsing with helpful error messages
- Export to Script functionality for reproducibility
- Publication-quality figure export with journal templates
- Cleaner separation between UI and analysis logic

Author: Based on work by Russell Kirk Pirlo
Date: January 10, 2026
"""

import pandas as pd
import numpy as np
import hvplot.pandas
import holoviews as hv
import panel as pn
import param
from bokeh.models import ColumnDataSource, Range1d, Span, BoxSelectTool, LinearAxis
from bokeh.models.formatters import DatetimeTickFormatter
from bokeh.models.callbacks import CustomJS
import io
from scipy import stats
import traceback
import re
from pathlib import Path
from datetime import datetime
import sys
import os
import threading
import glob

# Add current directory to path for core library import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import core library functions
try:
    from pioreactor_analysis import (
        PioreactorCSVParser,
        preprocess_od_data,
        calculate_batch_growth_rate,
        auto_detect_exponential_phase,
        calculate_apparent_yield,
        batch_analyze_multiple_units,
        PlotConfig,
        JournalTheme,
        PublicationPlotter,
        get_color_palette,
        ContinuousGrowthData,
        calculate_dilution_rate,
    )
    CORE_LIBRARY_AVAILABLE = True
    print("[OK] Core library loaded successfully")
except ImportError as e:
    print(f"[FAIL] Core library not available: {e}")
    print("Running in fallback mode with embedded analysis logic")
    CORE_LIBRARY_AVAILABLE = False

# Configure Panel and HoloViews
pn.extension('plotly', 'tabulator')
hv.extension('bokeh')

# Check if Bokeh export backend is available (requires selenium + webdriver)
try:
    from bokeh.io.export import get_screenshot_as_png
    BOKEH_EXPORT_AVAILABLE = True
except ImportError:
    BOKEH_EXPORT_AVAILABLE = False
    print("[WARNING] Bokeh PNG export unavailable - install selenium and webdriver-manager for plot export")

# Style the panel with a nice theme
pn.config.sizing_mode = 'stretch_width'

class GrowthRateAnalysisRefactored(param.Parameterized):
    """
    Refactored growth rate analysis application using core library.

    This version delegates all analysis logic to the pioreactor_analysis library
    and focuses solely on UI interactions and visualization.
    """

    smoothing_window = param.Integer(5, bounds=(1, 50), step=1,
                                    doc="Number of consecutive data points to average for smoothing.")
    min_od_threshold = param.Number(0.05, bounds=(0.001, 1.0), step=0.01,
                                   doc="Minimum OD value to include in analysis.")
    semi_log_plot = param.Boolean(True, doc="Display OD on logarithmic scale.")
    reactor_volume = param.Number(14.0, bounds=(1.0, 100.0), step=0.1,
                                 doc="Total volume of culture in the reactor (mL).")
    initial_substrate_conc = param.Number(20.0, bounds=(0.0, 200.0), step=0.5,
                                         doc="Initial substrate concentration in g/L.")

    # Plot appearance parameters
    plot_width = param.Integer(800, bounds=(400, 1600), step=50,
                              doc="Width of plots in pixels.")
    plot_height = param.Integer(400, bounds=(200, 800), step=50,
                               doc="Height of plots in pixels.")
    plot_dpi = param.Integer(100, bounds=(72, 300), step=10,
                            doc="DPI for exported plot images.")
    font_size = param.Integer(12, bounds=(8, 20), step=1,
                             doc="Base font size for plot labels.")
    color_scheme = param.Selector(default='default', objects=['default', 'colorblind', 'grayscale'],
                                 doc="Color scheme for plots.")

    # Region overlay toggle
    show_region_overlay = param.Boolean(True, doc="Show selected region overlay on plots.")

    def __init__(self, **params):
        """Initialize the refactored growth rate analysis application."""
        super().__init__(**params)

        # Initialize data containers
        self.od_data_df = pd.DataFrame()
        self.auxiliary_events = pd.DataFrame() # Store additional events (e.g. dosing)
        self.dilution_rate_df = pd.DataFrame()  # Store calculated dilution rates
        self.preprocessed_data = None  # Store preprocessed data from core library
        self.selected_region = {'start': None, 'end': None}
        self.selected_units = []
        self.current_filename = None
        self.current_filepath = None  # Store full path for script export

        # CSV parser from core library
        if CORE_LIBRARY_AVAILABLE:
            self.csv_parser = PioreactorCSVParser()

        # Upload lock to prevent race conditions when switching files
        self._upload_lock = threading.Lock()
        self._upload_in_progress = False

        # DataFrame to store cumulative results
        self.cumulative_results_df = pd.DataFrame(columns=[
            'Filename', 'Unit', 'Region Start (h)', 'Region End (h)',
            'Duration (h)', 'Growth Rate (h⁻¹)', 'Doubling Time (h)',
            'R²', 'Std Error', 'CI Lower', 'CI Upper', 'P-value', 'Data Points',
            'Apparent Yield (OD/g)', 'Yield Status', 'Max OD', 'ΔOD'
        ])

        # Status messages
        self.status_message = pn.pane.Markdown("", styles={'color': 'blue'})
        self.error_message = pn.pane.Markdown("", styles={'color': 'red'})
        self.debug_message = pn.pane.Markdown("", styles={'color': 'green', 'font-family': 'monospace', 'font-size': '12px'})

        # Loading spinner
        self.loading_spinner = pn.indicators.LoadingSpinner(value=False, width=24, height=24)

        # Data preview pane
        self.data_preview = pn.pane.HTML("", styles={'overflow-x': 'auto'})

        # Set up UI components
        self._setup_ui()

    def _setup_ui(self):
        """Set up all UI components and layout."""

        # File upload
        self.file_input = pn.widgets.FileInput(accept='.csv', multiple=False)
        self.file_input.param.watch(self._upload_file_callback, 'value')

        # CSV diagnostic button
        self.diagnose_csv_button = pn.widgets.Button(
            name='Diagnose CSV',
            button_type='warning',
            width=120
        )
        self.diagnose_csv_button.on_click(self._diagnose_csv_callback)

        # File path input for large files (bypasses browser memory)
        self.file_path_input = pn.widgets.TextInput(
            name='Or load from file path (for large files)',
            placeholder='/path/to/large_file.csv',
            width=400
        )
        self.load_path_button = pn.widgets.Button(
            name='Load from Path',
            button_type='primary',
            width=120
        )
        self.load_path_button.on_click(self._load_from_path_callback)

        self.update_button = pn.widgets.Button(name='Update Plots', button_type='primary')
        self.update_button.on_click(self._update_plots_callback)

        # Unit selection
        self.unit_selector = pn.widgets.MultiSelect(name='Select Units', options=[])
        self.unit_selector.param.watch(self._units_changed, 'value')

        # Plot panes
        self.od_plot = pn.pane.HoloViews(sizing_mode='stretch_width', height=300)
        self.ln_od_plot = pn.pane.HoloViews(sizing_mode='stretch_width', height=300)

        # Region selection widgets
        self.region_start = pn.widgets.FloatSlider(name='Region Start (hours)', start=0, end=100, step=0.1, value=0)
        self.region_end = pn.widgets.FloatSlider(name='Region End (hours)', start=0, end=100, step=0.1, value=100)

        self.region_start_input = pn.widgets.FloatInput(name='Start Value', width=100)
        self.region_end_input = pn.widgets.FloatInput(name='End Value', width=100)

        # Set up callbacks
        self.region_start.param.watch(self._update_region_from_slider, 'value')
        self.region_end.param.watch(self._update_region_from_slider, 'value')
        self.region_start_input.param.watch(self._update_region_from_input, 'value')
        self.region_end_input.param.watch(self._update_region_from_input, 'value')

        # Auto-detect button
        self.auto_detect_button = pn.widgets.Button(name='Auto-detect Exponential Phase', button_type='success')
        self.auto_detect_button.on_click(self._auto_detect_callback)

        # Region overlay toggle
        self.region_overlay_toggle = pn.widgets.Toggle(
            name='Show Region Overlay',
            value=True,
            button_type='primary'
        )
        self.region_overlay_toggle.param.watch(self._toggle_overlay_callback, 'value')

        # Results output
        self.results_output = pn.pane.HTML("")

        # Action buttons
        self.add_analysis_button = pn.widgets.Button(name='Add Current Analysis to Table', button_type='primary')
        self.add_analysis_button.on_click(self._add_analysis_callback)

        self.export_csv_button = pn.widgets.Button(name='📥 Export Results to CSV', button_type='success')
        self.export_csv_button.on_click(self._export_results_callback)

        self.clear_results_button = pn.widgets.Button(name='🗑️ Clear All Results', button_type='danger')
        self.clear_results_button.on_click(self._clear_results_callback)

        # NEW: Export to Script button
        self.export_script_button = pn.widgets.Button(
            name='📜 Export to Python Script',
            button_type='primary',
            width=200
        )
        self.export_script_button.on_click(self._export_script_callback)

        # NEW: Publication plot export
        self.journal_selector = pn.widgets.Select(
            name='Journal Theme',
            options=['Nature', 'Science', 'ACS', 'Elsevier', 'PLOS'],
            value='Nature',
            width=150
        )

        self.export_publication_button = pn.widgets.Button(
            name='📊 Export Publication Figure',
            button_type='success',
            width=200
        )
        self.export_publication_button.on_click(self._export_publication_callback)

        # Plot export buttons (keep for quick exports)
        self.export_od_plot_button = pn.widgets.Button(name='📊 Export OD Plot', button_type='default', width=150)
        self.export_od_plot_button.on_click(self._export_od_plot_callback)

        self.export_ln_od_plot_button = pn.widgets.Button(name='📊 Export ln(OD) Plot', button_type='default', width=150)
        self.export_ln_od_plot_button.on_click(self._export_ln_od_plot_callback)

        # Controls grouped in Accordion
        controls_accordion = pn.Accordion(
            (
                "Upload & Session",
                pn.Column(
                    pn.pane.Markdown("### Upload Data"),
                    self.file_input,
                    pn.Row(self.loading_spinner, self.diagnose_csv_button),
                    pn.pane.Markdown("### Load Large Files from Path"),
                    pn.Row(self.file_path_input, self.load_path_button),
                    self.status_message,
                    self.error_message,
                    self.data_preview,
                ),
            ),
            (
                "Settings",
                pn.Column(
                    pn.Row(self.param.smoothing_window, self.param.min_od_threshold),
                    self.param.semi_log_plot,
                    self.update_button,
                ),
            ),
            (
                "Yield Calculation",
                pn.Column(
                    pn.Row(self.param.reactor_volume, self.param.initial_substrate_conc),
                    pn.pane.Markdown("*Required for apparent yield calculation*",
                                   styles={'font-size': '11px', 'color': '#666'}),
                ),
            ),
            (
                "Plot Appearance",
                pn.Column(
                    pn.Row(self.param.plot_width, self.param.plot_height),
                    pn.Row(self.param.font_size, self.param.color_scheme, self.param.plot_dpi),
                    pn.pane.Markdown("*DPI applies to exported images*",
                                   styles={'font-size': '11px', 'color': '#666'}),
                ),
            ),
            active=[],
            sizing_mode='stretch_width'
        )

        # Main layout
        self.main_layout = pn.Tabs(
            (
                'Analysis',
                pn.Column(
                    pn.pane.Markdown("# Growth Rate Analysis for Batch Culture (Refactored)"),
                    pn.pane.Markdown("*Using pioreactor_analysis core library*", styles={'color': '#666', 'font-size': '12px'}),
                    pn.Row(
                        pn.Column(
                            pn.pane.Markdown("### Unit Selection"),
                            self.unit_selector,
                            width=300
                        ),
                        pn.Column(
                            pn.pane.Markdown("### Region Selection"),
                            pn.Row(
                                pn.Column(self.region_start, width=300),
                                pn.Column(self.region_start_input, width=100)
                            ),
                            pn.Row(
                                pn.Column(self.region_end, width=300),
                                pn.Column(self.region_end_input, width=100)
                            ),
                            pn.Row(self.auto_detect_button, self.region_overlay_toggle),
                            width=450
                        )
                    ),
                    pn.Row(self.export_od_plot_button, self.export_ln_od_plot_button, align='end'),
                    self.od_plot,
                    self.ln_od_plot,
                    sizing_mode='stretch_width'
                )
            ),
            (
                'Controls',
                pn.Column(
                    pn.pane.Markdown("# Controls"),
                    controls_accordion,
                    sizing_mode='stretch_width'
                )
            ),
            (
                'Results',
                pn.Column(
                    pn.Row(
                        self.add_analysis_button,
                        self.export_csv_button,
                        self.clear_results_button,
                    ),
                    pn.pane.Markdown("### Export Options"),
                    pn.Row(
                        self.export_script_button,
                        self.journal_selector,
                        self.export_publication_button,
                    ),
                    self.results_output,
                    sizing_mode='stretch_width'
                )
            ),
            (
                'Debug',
                self.debug_message
            )
        )

        # Initial update
        self._update_results_display()
        self.param.watch(self._update_plots_callback, 'semi_log_plot')
        self.param.watch(self._update_plots_callback, ['plot_width', 'plot_height', 'font_size', 'color_scheme'])

    def show_success(self, message):
        """Display a success message."""
        # Remove checkmark emoji that causes encoding issues on Windows
        self.status_message.object = f"**{message}**"
        self.error_message.object = ""

    def show_error(self, message):
        """Display an error message."""
        # Remove X emoji that causes encoding issues on Windows
        self.error_message.object = f"**Error:** {message}"
        self.status_message.object = ""

    def show_debug(self, message):
        """Display a debug message."""
        self.debug_message.object = f"```\n{message}\n```"

    def _get_color_palette(self):
        """Get the current color palette."""
        if self.color_scheme == 'colorblind':
            return ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7', '#000000']
        elif self.color_scheme == 'grayscale':
            return ['#000000', '#404040', '#808080', '#C0C0C0', '#606060', '#A0A0A0', '#303030', '#909090']
        else:
            return None

    def _add_secondary_yaxis(self, plot, element):
        """
        Bokeh hook to add secondary y-axis for dilution rate.

        This is called after the plot is rendered but before display.
        It adds a right y-axis with proper scaling for dilution rate values.
        """
        from bokeh.models import LinearAxis, Range1d
        from bokeh.models.renderers import GlyphRenderer

        p = plot.state  # Get the Bokeh figure

        # Get dilution rate range (stored during plot creation)
        if not hasattr(self, '_dr_range') or self._dr_range is None:
            return

        dr_min, dr_max = self._dr_range

        # Add padding to range (ensure reasonable minimum range)
        dr_range = dr_max - dr_min if dr_max > dr_min else 0.1
        y_min_dr = max(0, dr_min - 0.1 * dr_range)
        y_max_dr = dr_max + 0.1 * dr_range

        # Ensure minimum range of 0.1 h⁻¹ for readability
        if y_max_dr - y_min_dr < 0.1:
            y_max_dr = y_min_dr + 0.1

        # Set up secondary y-range
        p.extra_y_ranges = {"dilution_rate": Range1d(start=y_min_dr, end=y_max_dr)}

        # Add right y-axis
        right_axis = LinearAxis(
            y_range_name="dilution_rate",
            axis_label="Dilution Rate (h⁻¹)",
            axis_label_text_color="orange",
            major_tick_line_color="orange",
            minor_tick_line_color="orange",
            axis_line_color="orange",
            major_label_text_color="orange"
        )
        p.add_layout(right_axis, 'right')

        # Find and update the dilution rate glyph to use secondary axis
        # Look for orange lines (dilution rate curve)
        for renderer in p.renderers:
            if isinstance(renderer, GlyphRenderer):
                glyph = renderer.glyph
                # Check multiple ways to identify the dilution rate line
                is_dilution_line = False
                if hasattr(glyph, 'line_color'):
                    color = glyph.line_color
                    # Handle both string and dict color specs
                    if isinstance(color, str) and color == 'orange':
                        is_dilution_line = True
                    elif isinstance(color, dict) and color.get('value') == 'orange':
                        is_dilution_line = True

                if is_dilution_line:
                    renderer.y_range_name = "dilution_rate"

    def _diagnose_csv_callback(self, event):
        """Diagnose CSV file without full processing - show column info and data quality."""
        if not self.file_input.value:
            self.show_error("No file uploaded. Upload a CSV file first.")
            return

        try:
            # Read raw CSV
            df = pd.read_csv(io.BytesIO(self.file_input.value))

            # Generate diagnostic report
            report = [
                f"## CSV Diagnostic Report",
                f"",
                f"**File:** {self.file_input.filename}",
                f"**Total rows:** {len(df)}",
                f"**Columns:** {', '.join(df.columns)}",
                f"",
            ]

            # Check for OD columns
            od_cols = [c for c in df.columns if 'od' in c.lower() and 'method' not in c.lower()]
            for col in od_cols:
                values = pd.to_numeric(df[col], errors='coerce')
                valid = values.notna() & (values > 0)
                report.append(f"### Column '{col}':")
                report.append(f"- Total values: {len(values)}")
                report.append(f"- Valid (numeric > 0): {valid.sum()}")
                report.append(f"- NaN/missing: {values.isna().sum()}")
                report.append(f"- Zero or negative: {((values <= 0) & values.notna()).sum()}")
                if valid.any():
                    report.append(f"- Range: {values[valid].min():.4f} - {values[valid].max():.4f}")
                report.append("")

            # Check timestamp columns
            ts_cols = [c for c in df.columns if 'time' in c.lower() or 'date' in c.lower()]
            for col in ts_cols:
                parsed = pd.to_datetime(df[col], errors='coerce')
                report.append(f"### Column '{col}':")
                report.append(f"- Parseable timestamps: {parsed.notna().sum()}/{len(df)}")
                if parsed.notna().any():
                    report.append(f"- Time range: {parsed.min()} to {parsed.max()}")
                report.append("")

            # Check unit/experiment columns
            if 'pioreactor_unit' in df.columns:
                units = df['pioreactor_unit'].unique()
                report.append(f"### Units found: {list(units)}")
                report.append("")

            if 'experiment' in df.columns:
                experiments = df['experiment'].unique()
                report.append(f"### Experiments found: {list(experiments)}")
                report.append("")

            self.show_debug("\n".join(report))
            self.show_success("CSV diagnosis complete - check Debug tab for report")

        except Exception as e:
            self.show_error(f"Diagnostic failed: {str(e)}")
            self.show_debug(f"Error details:\n{traceback.format_exc()}")

    def _clear_state_for_new_file(self):
        """Clear application state before loading a new file to prevent race conditions."""
        # Clear data containers
        self.od_data_df = pd.DataFrame()
        self.auxiliary_events = pd.DataFrame()
        self.dilution_rate_df = pd.DataFrame()
        self.preprocessed_data = None

        # Reset region selection
        self.selected_region = {'start': None, 'end': None}
        self.selected_units = []

        # Clear plots to prevent stale data display
        self.od_plot.object = hv.Text(0, 0, 'Loading...').opts(width=800, height=300)
        self.ln_od_plot.object = hv.Text(0, 0, 'Loading...').opts(width=800, height=300)

        # Clear unit selector
        self.unit_selector.options = []
        self.unit_selector.value = []

        # Clean up temp files from previous uploads
        self._cleanup_temp_files()

    def _cleanup_temp_files(self):
        """Remove temporary files from previous uploads."""
        for temp_file in glob.glob("temp_*.csv"):
            try:
                Path(temp_file).unlink()
            except Exception:
                pass  # Ignore cleanup errors

    def _load_from_path_callback(self, event):
        """Load file directly from filesystem path (bypasses browser memory for large files)."""
        filepath = self.file_path_input.value
        if not filepath:
            self.show_error("Please enter a file path.")
            return

        filepath = Path(filepath)
        if not filepath.exists():
            self.show_error(f"File not found: {filepath}")
            return

        if not filepath.suffix.lower() == '.csv':
            self.show_error("Please specify a CSV file.")
            return

        if not CORE_LIBRARY_AVAILABLE:
            self.show_error("Core library not available. Cannot process file.")
            return

        # Check if another upload is in progress
        if self._upload_in_progress:
            self.show_error("Please wait for current file to finish loading.")
            return

        # Acquire lock
        if not self._upload_lock.acquire(blocking=False):
            self.show_error("Another file operation is in progress. Please wait.")
            return

        self._upload_in_progress = True
        self.loading_spinner.value = True

        try:
            # Get file size info
            file_size = filepath.stat().st_size
            self.show_debug(f"Loading file from path: {filepath} ({file_size / 1024 / 1024:.1f} MB)")

            # Clear previous state
            self._clear_state_for_new_file()

            # Parse directly from filesystem (no browser memory usage)
            data = self.csv_parser.parse(filepath)
            self.show_debug(f"CSV Format detected: {data.__class__.__name__}")

            # Convert to DataFrame
            if hasattr(data, 'to_dataframe'):
                df = data.to_dataframe()
            elif hasattr(data, 'to_od_dataframe'):
                df = data.to_od_dataframe()
                if data.dilution_events:
                    dosing_df = data.to_dilution_dataframe()
                    if 'timestamp' in dosing_df.columns:
                        dosing_df['timestamp'] = pd.to_datetime(dosing_df['timestamp'])
                    self.auxiliary_events = dosing_df
            else:
                raise AttributeError(f"Parsed data has no 'to_dataframe' method")

            if df.empty:
                raise ValueError("Parsed dataset is empty.")

            # Store filename and path
            self.current_filename = filepath.name
            self.current_filepath = str(filepath.absolute())

            # Process the data
            self._process_data_with_core_library(df)

            self.show_success(f"Loaded {filepath.name} ({file_size / 1024 / 1024:.1f} MB) successfully!")
            self._update_data_preview()
            self._update_plots()

        except Exception as e:
            self.show_error(f"Error loading file: {str(e)}")
            self.show_debug(f"Error details:\n{traceback.format_exc()}")
            self.current_filename = None
            self.current_filepath = None
        finally:
            self._upload_in_progress = False
            self.loading_spinner.value = False
            self._upload_lock.release()

    def _upload_file_callback(self, event):
        """Handle file upload using core library CSV parser."""
        if not self.file_input.value or not self.file_input.filename.endswith('.csv'):
            self.show_error("Please upload a CSV file.")
            return

        if not CORE_LIBRARY_AVAILABLE:
            self.show_error("Core library not available. Cannot process file.")
            return

        # Check file size (max 50MB via browser upload)
        MAX_FILE_SIZE_MB = 50
        file_size = len(self.file_input.value)
        if file_size > MAX_FILE_SIZE_MB * 1024 * 1024:
            self.show_error(
                f"File too large ({file_size / 1024 / 1024:.1f} MB). "
                f"Maximum supported size via browser upload is {MAX_FILE_SIZE_MB} MB. "
                f"For larger files, use the 'Load from Path' option in Controls tab."
            )
            return

        # Check if another upload is in progress
        if self._upload_in_progress:
            self.show_error("Please wait for current file to finish loading.")
            return

        # Acquire lock
        if not self._upload_lock.acquire(blocking=False):
            self.show_error("Another file operation is in progress. Please wait.")
            return

        self._upload_in_progress = True
        self.loading_spinner.value = True
        try:
            filename = self.file_input.filename
            self.show_debug(f"Processing upload: {filename}")

            # Clear previous state before loading new file
            self._clear_state_for_new_file()

            # Save file temporarily for core library parser
            temp_path = Path(f"temp_{filename}")
            temp_path.write_bytes(self.file_input.value)

            # If main script context is set (file was valid OD data previously), don't overwrite blindly
            is_additional = not self.od_data_df.empty

            # Use core library CSV parser
            try:
                data = self.csv_parser.parse(temp_path)
                self.show_debug(f"CSV Format detected: {data.__class__.__name__}")

                # Handle Special Case: Dosing Events Only (ContinuousGrowthData with no OD)
                is_dosing_only = False
                if isinstance(data, ContinuousGrowthData) and not data.od_readings and data.dilution_events:
                    is_dosing_only = True
                
                if is_dosing_only:
                    # Convert dilution events to DataFrame and append to auxiliary
                    new_events = data.to_dilution_dataframe()
                    
                    # Ensure timestamp is datetime
                    if 'timestamp' in new_events.columns:
                        new_events['timestamp'] = pd.to_datetime(new_events['timestamp'])
                        
                    # Calculate elapsed time if we have OD data with reference time
                    if not self.od_data_df.empty and 'timestamp' in self.od_data_df.columns:
                        start_time = self.od_data_df['timestamp'].min()
                        new_events['elapsed_hours'] = (new_events['timestamp'] - start_time).dt.total_seconds() / 3600
                    
                    if self.auxiliary_events.empty:
                        self.auxiliary_events = new_events
                    else:
                        self.auxiliary_events = pd.concat([self.auxiliary_events, new_events], ignore_index=True)
                        
                    self.show_success(f"Loaded {len(data.dilution_events)} dosing events from {filename}.")
                    
                    if self.od_data_df.empty:
                         self.show_error(
                            "Dosing events loaded, but NO OD data is present. "
                            "Please upload the accompanying OD readings file (e.g. od_readings_...) to proceed with growth analysis."
                        )
                    else:
                        self._update_plots()
                    return

                # Normal OD Data Processing
                # Convert to DataFrame (support multiple data model types)
                if hasattr(data, 'to_dataframe'):
                    df = data.to_dataframe()
                elif hasattr(data, 'to_od_dataframe'):
                    # ContinuousGrowthData provides OD and dilution tables separately
                    df = data.to_od_dataframe()
                    # Also extract dilution events if present
                    if data.dilution_events:
                        dosing_df = data.to_dilution_dataframe()
                        if 'timestamp' in dosing_df.columns:
                            dosing_df['timestamp'] = pd.to_datetime(dosing_df['timestamp'])
                        if self.auxiliary_events.empty:
                            self.auxiliary_events = dosing_df
                        else:
                            self.auxiliary_events = pd.concat([self.auxiliary_events, dosing_df], ignore_index=True)
                else:
                    raise AttributeError(f"Parsed data object of type {type(data).__name__} has no 'to_dataframe' or 'to_od_dataframe' method")

                # Check if we have valid OD data
                if df.empty:
                    raise ValueError("Parsed dataset is empty. Check your CSV file format.")
                    
                # Store as current main file
                self.current_filename = filename
                self.current_filepath = str(temp_path.absolute())

                # Process the data (timestamps, elapsed time, exp_unit)
                self._process_data_with_core_library(df)
                
                # Re-calculate elapsed time for auxiliary events if they exist
                if not self.auxiliary_events.empty and 'timestamp' in self.auxiliary_events.columns:
                     if 'timestamp' in self.od_data_df.columns:
                        start_time = self.od_data_df['timestamp'].min()
                        # Align timestamps - ensuring timezone naive/aware consistency
                        try:
                            # Try direct subtraction
                            self.auxiliary_events['elapsed_hours'] = (self.auxiliary_events['timestamp'] - start_time).dt.total_seconds() / 3600
                        except TypeError:
                            # Handle timezone mismatch
                            ts_aux = pd.to_datetime(self.auxiliary_events['timestamp'], utc=True)
                            ts_started = pd.to_datetime(start_time, utc=True)
                            self.auxiliary_events['elapsed_hours'] = (ts_aux - ts_started).dt.total_seconds() / 3600

                self.show_success(f"OD Data from {filename} loaded successfully!")
                if not self.auxiliary_events.empty:
                    self.show_success(f"OD Data loaded. Also have {len(self.auxiliary_events)} auxiliary events.")

                # Update displays
                self._update_data_preview()
                self._update_plots()

            except Exception as parse_error:
                self.show_error(f"CSV parsing failed: {str(parse_error)}")
                self.show_debug(f"Error details:\n{traceback.format_exc()}")
                return

        except Exception as e:
            error_msg = str(e).lower()
            if 'websocket' in error_msg or 'connection' in error_msg:
                self.show_error(
                    "Connection error during file processing. "
                    "Try refreshing the browser or re-uploading the file."
                )
            else:
                self.show_error(f"Error processing file: {str(e)}")
            self.show_debug(f"Error details:\n{traceback.format_exc()}")
            self.current_filename = None
            self.current_filepath = None
        finally:
            self._upload_in_progress = False
            self.loading_spinner.value = False
            self._upload_lock.release()

    def _process_data_with_core_library(self, df):
        """Process data using core library preprocessing functions."""
        # Robust column detection and normalization
        cols_lower = {c.lower(): c for c in df.columns}

        # Timestamp column detection
        ts_col = None
        for cand in ['timestamp', 'time', 'date', 'datetime']:
            if cand in cols_lower:
                ts_col = cols_lower[cand]
                break
        if ts_col is None:
            # Try any column containing 'time' or 'date'
            for c in df.columns:
                if 'time' in c.lower() or 'date' in c.lower():
                    ts_col = c
                    break

        if ts_col is not None:
            df['timestamp'] = pd.to_datetime(df[ts_col])
        elif 'elapsed_hours' in df.columns:
            # Already has elapsed hours
            pass
        else:
            # No timestamp info - fail early
            raise ValueError('No timestamp or elapsed time column found in CSV data')

        # OD column detection (support multiple readouts/channels)
        # First check if 'od_value' already exists (from parser output)
        if 'od_value' in df.columns and df['od_value'].notna().any():
            # Already have od_value from parser, ensure it's numeric
            df['od_value'] = pd.to_numeric(df['od_value'], errors='coerce')
            od_candidates = ['od_value']  # Skip detection, use existing column
        else:
            od_candidates = []
            od_patterns = [r'\bod\b', r'od_reading', r'od_value', r'avg_od', r'absorbance', r'as7341', r'as73', r'as7']
            for c in df.columns:
                cn = c.lower()
                if 'method' in cn:
                    continue
                for pat in od_patterns:
                    if re.search(pat, cn):
                        od_candidates.append(c)
                        break

        # Also include columns like 'channel_1', 'ch1' if they explicitly contain od-like tokens
        od_candidates = list(dict.fromkeys(od_candidates))

        if len(od_candidates) == 1:
            df['od_value'] = pd.to_numeric(df[od_candidates[0]], errors='coerce')
        elif len(od_candidates) > 1:
            # Unpivot multiple OD/channel columns into long form
            id_vars = [c for c in df.columns if c not in od_candidates]
            melted = df.melt(id_vars=id_vars, value_vars=od_candidates, var_name='raw_channel', value_name='od_value')

            # Try to extract angle/channel info from the column name
            def extract_channel_info(name: str):
                name_l = name.lower()
                angle = None
                ch = None
                m = re.search(r'channel[_\- ]?(\d+)', name_l)
                if m:
                    ch = int(m.group(1))
                m2 = re.search(r'angle[_\- ]?(\d+)', name_l)
                if m2:
                    angle = float(m2.group(1))
                return angle, ch

            melted['od_value'] = pd.to_numeric(melted['od_value'], errors='coerce')
            angles = []
            channels = []
            for rc in melted['raw_channel'].astype(str):
                a, ch = extract_channel_info(rc)
                angles.append(a)
                channels.append(ch)
            melted['angle'] = angles
            melted['channel'] = channels

            df = melted
            cols_lower = {c.lower(): c for c in df.columns}
            # ensure timestamp/experiment/unit columns are still present after melt
            if 'timestamp' not in df.columns and ts_col is not None:
                df['timestamp'] = pd.to_datetime(df[ts_col])
        else:
            # No OD-like columns found; create placeholder and continue (preprocessing will filter)
            df['od_value'] = pd.NA

        # Unit column detection
        unit_col = None
        for cand in ['unit', 'pioreactor_unit', 'unit_id']:
            if cand in cols_lower:
                unit_col = cols_lower[cand]
                break
        if unit_col is None:
            for c in df.columns:
                if 'unit' in c.lower():
                    unit_col = c
                    break
        if unit_col is None:
            df['unit'] = 'unit1'
        else:
            df['unit'] = df[unit_col].astype(str)

        # Experiment column detection
        exp_col = None
        for cand in ['experiment', 'experiment_id', 'exp', 'experimentname']:
            if cand in cols_lower:
                exp_col = cols_lower[cand]
                break
        if exp_col is None:
            for c in df.columns:
                if 'exp' in c.lower() or 'experiment' in c.lower():
                    exp_col = c
                    break
        if exp_col is None:
            # Fall back to filename or a default name
            default_exp = Path(self.current_filename).stem if self.current_filename else 'experiment1'
            df['experiment'] = default_exp
        else:
            df['experiment'] = df[exp_col].astype(str)

        # Calculate elapsed time for each experiment/unit
        df_list = []
        group_cols = ['experiment', 'unit']
        for (exp, unit), group in df.groupby(group_cols):
            if 'elapsed_hours' not in group.columns or group['elapsed_hours'].isnull().all():
                start_time = group['timestamp'].min()
                group = group.copy()
                group['elapsed_hours'] = (group['timestamp'] - start_time).dt.total_seconds() / 3600
            group['exp_unit'] = f"{exp}_{unit}"
            df_list.append(group)

        if df_list:
            self.od_data_df = pd.concat(df_list, ignore_index=True)

            # Apply core library preprocessing
            # elapsed_hours already calculated above, preprocessing will skip recalculating
            self.preprocessed_data = preprocess_od_data(
                self.od_data_df,
                smoothing_window=self.smoothing_window,
                min_od_threshold=self.min_od_threshold,
                time_column='timestamp',
                od_column='od_value',
                group_by=['experiment', 'unit']
            )

            # Update unit selector
            units = sorted(self.od_data_df['exp_unit'].unique())
            self.unit_selector.options = units
            if units:
                self.unit_selector.value = [units[0]]
                self.selected_units = [units[0]]

            # Update region sliders
            min_time = self.od_data_df['elapsed_hours'].min()
            max_time = self.od_data_df['elapsed_hours'].max()

            self.region_start.start = min_time
            self.region_start.end = max_time
            self.region_start.value = min_time

            self.region_end.start = min_time
            self.region_end.end = max_time
            self.region_end.value = max_time

            self.region_start_input.value = min_time
            self.region_end_input.value = max_time

            self.selected_region = {'start': min_time, 'end': max_time}
        else:
            self.show_error("No valid data found in CSV file.")
            self.od_data_df = pd.DataFrame()

    def _units_changed(self, event):
        """Handle unit selection changes."""
        self.selected_units = self.unit_selector.value
        self._update_plots()

    def _update_region_from_slider(self, event):
        """Handle region slider updates."""
        if event.obj is self.region_start:
            self.region_start_input.value = event.new
        elif event.obj is self.region_end:
            self.region_end_input.value = event.new
        self._update_region()

    def _update_region_from_input(self, event):
        """Handle text input updates."""
        if event.obj is self.region_start_input:
            value = max(min(event.new, self.region_start.end), self.region_start.start)
            self.region_start.value = value
            if value != event.new:
                self.region_start_input.value = value
        elif event.obj is self.region_end_input:
            value = max(min(event.new, self.region_end.end), self.region_end.start)
            self.region_end.value = value
            if value != event.new:
                self.region_end_input.value = value
        self._update_region()

    def _update_region(self):
        """Update the selected region."""
        if self.region_start.value > self.region_end.value:
            self.region_end.value = self.region_start.value
            self.region_end_input.value = self.region_start.value

        self.selected_region = {
            'start': self.region_start.value,
            'end': self.region_end.value
        }
        self._update_plots()

    def _toggle_overlay_callback(self, event):
        """Handle region overlay toggle."""
        self.show_region_overlay = event.new
        self._update_plots()

    def _update_plots_callback(self, event):
        """Handle update button clicks."""
        self.loading_spinner.value = True
        try:
            self._update_plots()
            self.show_success("Plots updated with current settings.")
        finally:
            self.loading_spinner.value = False

    def _auto_detect_callback(self, event):
        """Auto-detect exponential phase using core library function."""
        if self.preprocessed_data is None or self.preprocessed_data.empty or len(self.selected_units) == 0:
            self.show_error("No data available for auto-detection.")
            return

        if not CORE_LIBRARY_AVAILABLE:
            self.show_error("Core library not available.")
            return

        self.loading_spinner.value = True
        try:
            unit = self.selected_units[0]

            # Filter data for selected unit
            unit_data = self.preprocessed_data[
                self.preprocessed_data['exp_unit'] == unit
            ].copy()

            if len(unit_data) < 10:
                self.show_error(f"Not enough data points for unit {unit} to auto-detect.")
                return

            # Use core library auto-detect function
            start_time, end_time, growth_result = auto_detect_exponential_phase(
                unit_data,
                od_column='od_smooth',
                time_column='elapsed_hours'
            )

            # Update region sliders
            self.region_start.value = start_time
            self.region_start_input.value = start_time
            self.region_end.value = end_time
            self.region_end_input.value = end_time

            # Show success message
            success_msg = (
                f"Auto-detected exponential phase for {unit}:\n"
                f"• Time range: {start_time:.2f}h to {end_time:.2f}h ({end_time-start_time:.2f}h duration)\n"
                f"• **Max Specific Growth Rate (μmax): {growth_result.growth_rate:.4f} h^-1**\n"
                f"• Doubling time: {growth_result.doubling_time:.2f} h\n"
                f"• R²: {growth_result.r_squared:.4f}"
            )
            self.show_success(success_msg)

        except Exception as e:
            self.show_error(f"Auto-detection failed: {str(e)}")
            self.show_debug(f"Error details:\n{traceback.format_exc()}")
        finally:
            self.loading_spinner.value = False

    def _update_plots(self):
        """Update all plots with current data and parameters."""
        if self.preprocessed_data is None or self.preprocessed_data.empty or len(self.selected_units) == 0:
            self.od_plot.object = hv.Text(0, 0, 'No valid OD data found').opts(width=800, height=300)
            self.ln_od_plot.object = hv.Text(0, 0, 'No valid ln(OD) data found').opts(width=800, height=300)
            return

        # Use preprocessed data from core library
        palette = self._get_color_palette()

        # Calculate dilution rate if we have auxiliary events
        self._calculate_dilution_rate_from_events()
        has_dilution = not self.dilution_rate_df.empty

        # Create OD vs Time plot using HoloViews
        od_curves = []
        for idx, unit in enumerate(self.selected_units):
            unit_data = self.preprocessed_data[self.preprocessed_data['exp_unit'] == unit].copy()
            unit_data = unit_data.sort_values('elapsed_hours')

            color = palette[idx % len(palette)] if palette else None
            curve_opts = {
                'line_width': 2,
                'tools': ['hover', 'tap']
            }
            if color:
                curve_opts['color'] = color

            curve = hv.Curve(
                unit_data, 'elapsed_hours', 'od_smooth',
                label=unit
            ).opts(**curve_opts)
            od_curves.append(curve)

        # Build the OD overlay
        od_overlay = hv.Overlay(od_curves)

        # Add dilution event markers if available
        has_dilution_events = hasattr(self, 'dilution_events_for_plot') and not self.dilution_events_for_plot.empty

        # Add dilution rate curve if available (with dual y-axis)
        title = 'OD vs Time'
        if has_dilution:
            title = 'OD & Dilution Rate vs Time'
            dr_data = self.dilution_rate_df.copy()
            dr_data = dr_data.sort_values('elapsed_hours')
            mean_dr = dr_data['moving_avg_dilution_rate'].mean()
            dr_min = dr_data['moving_avg_dilution_rate'].min()
            dr_max = dr_data['moving_avg_dilution_rate'].max()

            # Store dilution rate range for the hook to use
            self._dr_range = (dr_min, dr_max)

            dr_data_plot = pd.DataFrame({
                'elapsed_hours': dr_data['elapsed_hours'],
                'dilution_rate': dr_data['moving_avg_dilution_rate']
            })

            # Create dilution rate curve (will be mapped to secondary axis by hook)
            dr_curve = hv.Curve(
                dr_data_plot, 'elapsed_hours', 'dilution_rate',
                label=f'Dilution Rate (D={mean_dr:.3f} h⁻¹)'
            ).opts(
                color='orange',
                line_width=2,
                line_dash='dashed',
                alpha=0.8,
                tools=['hover']
            )
            od_overlay = od_overlay * dr_curve

            # Add dilution event markers as vertical lines (thin gray lines at each dilution)
            if has_dilution_events:
                # Group events by minute to reduce visual clutter
                events_df = self.dilution_events_for_plot.copy()
                events_df['minute_bin'] = (events_df['elapsed_hours'] * 60).round() / 60  # Round to nearest minute
                unique_event_times = events_df['minute_bin'].unique()

                # Add vertical spike markers for dilution events (subsample if too many)
                max_markers = 200  # Limit markers to avoid cluttering
                if len(unique_event_times) > max_markers:
                    # Subsample evenly
                    step = len(unique_event_times) // max_markers
                    unique_event_times = unique_event_times[::step]

                # Create spike markers at event times
                for event_time in unique_event_times:
                    spike = hv.VLine(event_time).opts(
                        color='gray',
                        line_width=0.5,
                        alpha=0.3
                    )
                    od_overlay = od_overlay * spike

            # Show dilution rate statistics in debug
            self.show_debug(
                f"Dilution rate: {dr_min:.4f} - {dr_max:.4f} h⁻¹ (mean: {mean_dr:.4f})\n"
                f"Reactor volume: {self.reactor_volume} mL\n"
                f"Time bins: {len(dr_data)}, Total events: {len(self.dilution_events_for_plot) if has_dilution_events else 0}"
            )
        else:
            self._dr_range = None

        # Apply Bokeh hook for dual y-axis if dilution data exists
        hooks = [self._add_secondary_yaxis] if has_dilution else []

        od_plot = od_overlay.opts(
            title=title,
            xlabel='Time (hours)',
            ylabel='OD (smoothed)',
            legend_position='top_right',
            width=self.plot_width,
            height=self.plot_height,
            fontsize={
                'title': self.font_size + 2,
                'labels': self.font_size,
                'xticks': self.font_size - 2,
                'yticks': self.font_size - 2,
                'legend': self.font_size - 1,
            },
            hooks=hooks
        )

        # Create ln(OD) vs Time plot using HoloViews
        ln_od_curves = []
        for idx, unit in enumerate(self.selected_units):
            unit_data = self.preprocessed_data[self.preprocessed_data['exp_unit'] == unit].copy()
            unit_data = unit_data.sort_values('elapsed_hours')
            # Filter out invalid ln values
            unit_data = unit_data[np.isfinite(unit_data['ln_od'])]

            color = palette[idx % len(palette)] if palette else None
            curve_opts = {
                'line_width': 2,
                'tools': ['hover', 'tap']
            }
            if color:
                curve_opts['color'] = color

            curve = hv.Curve(
                unit_data, 'elapsed_hours', 'ln_od',
                label=unit
            ).opts(**curve_opts)
            ln_od_curves.append(curve)

        ln_od_plot = hv.Overlay(ln_od_curves).opts(
            title='ln(OD) vs Time',
            xlabel='Time (hours)',
            ylabel='ln(OD)',
            legend_position='top_right',
            width=self.plot_width,
            height=self.plot_height,
            fontsize={
                'title': self.font_size + 2,
                'labels': self.font_size,
                'xticks': self.font_size - 2,
                'yticks': self.font_size - 2,
                'legend': self.font_size - 1,
            }
        )

        # Add selected region visualization to both plots (only if toggle enabled)
        if self.show_region_overlay and self.selected_region['start'] is not None and self.selected_region['end'] is not None:
            start_line = hv.VLine(self.selected_region['start']).opts(color='red', line_width=1.5)
            end_line = hv.VLine(self.selected_region['end']).opts(color='red', line_width=1.5)
            region_shade = hv.VSpan(
                self.selected_region['start'],
                self.selected_region['end']
            ).opts(alpha=0.2, color='red')

            od_plot = od_plot * start_line * end_line * region_shade
            ln_od_plot = ln_od_plot * start_line * end_line * region_shade

            # Add regression lines for the selected region
            for unit in self.selected_units:
                unit_data = self.preprocessed_data[self.preprocessed_data['exp_unit'] == unit].copy()
                region_data = unit_data[
                    (unit_data['elapsed_hours'] >= self.selected_region['start']) &
                    (unit_data['elapsed_hours'] <= self.selected_region['end']) &
                    np.isfinite(unit_data['ln_od'])
                ]

                if len(region_data) >= 5:
                    x = region_data['elapsed_hours'].values
                    y = region_data['ln_od'].values

                    if np.ptp(x) > 1e-9:
                        try:
                            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

                            # Create regression line
                            x_range = np.array([self.selected_region['start'], self.selected_region['end']])
                            y_range = slope * x_range + intercept

                            # Calculate 95% confidence interval
                            n = len(x)
                            t_val = stats.t.ppf(0.975, n - 2)
                            residuals = y - (slope * x + intercept)
                            s_residuals = np.sqrt(np.sum(residuals**2) / (n - 2))
                            x_mean = np.mean(x)
                            sxx = np.sum((x - x_mean)**2)

                            se_fit_start = s_residuals * np.sqrt(1/n + (self.selected_region['start'] - x_mean)**2 / sxx)
                            se_fit_end = s_residuals * np.sqrt(1/n + (self.selected_region['end'] - x_mean)**2 / sxx)

                            y_start = slope * self.selected_region['start'] + intercept
                            y_end = slope * self.selected_region['end'] + intercept

                            ci_lower_start = y_start - t_val * se_fit_start
                            ci_upper_start = y_start + t_val * se_fit_start
                            ci_lower_end = y_end - t_val * se_fit_end
                            ci_upper_end = y_end + t_val * se_fit_end

                            ci_x = [self.selected_region['start'], self.selected_region['end'],
                                   self.selected_region['end'], self.selected_region['start']]
                            ci_y = [ci_lower_start, ci_lower_end, ci_upper_end, ci_upper_start]

                            ci_band = hv.Polygons([np.column_stack([ci_x, ci_y])]).opts(
                                alpha=0.2,
                                color='green',
                                line_width=0
                            )

                            regression_line = hv.Curve(
                                np.column_stack([x_range, y_range]),
                                label=f"{unit} fit (μ={slope:.3f}h⁻¹)"
                            ).opts(
                                color='green',
                                line_width=2,
                                line_dash='dashed'
                            )

                            ln_od_plot = ln_od_plot * ci_band * regression_line
                        except ValueError as ve:
                            self.show_debug(f"Skipping regression for {unit}: {ve}")

        # Update the plot panes
        self.od_plot.object = od_plot
        self.ln_od_plot.object = ln_od_plot

    def _calculate_dilution_rate_from_events(self):
        """
        Calculate dilution rate from auxiliary events if available.

        Aggregates pump events into time bins (default: 1 minute) before calculating
        dilution rate to avoid artificially high rates from rapid pump strokes.

        Formula: D = V_total / (V_reactor * Δt)
        Where:
        - D = dilution rate (h⁻¹)
        - V_total = total volume added in time bin (mL)
        - V_reactor = reactor volume (mL) - uses self.reactor_volume
        - Δt = time bin duration (hours)
        """
        if not hasattr(self, 'auxiliary_events') or self.auxiliary_events.empty:
            self.dilution_rate_df = pd.DataFrame()
            self.dilution_events_for_plot = pd.DataFrame()
            return

        if 'volume_ml' not in self.auxiliary_events.columns:
            # No volume data, can't calculate dilution rate
            self.dilution_rate_df = pd.DataFrame()
            self.dilution_events_for_plot = pd.DataFrame()
            return

        try:
            # Filter to events with valid volume data
            events_with_volume = self.auxiliary_events[
                self.auxiliary_events['volume_ml'].notna() &
                (self.auxiliary_events['volume_ml'] > 0)
            ].copy()

            if len(events_with_volume) < 2:
                self.dilution_rate_df = pd.DataFrame()
                self.dilution_events_for_plot = pd.DataFrame()
                return

            # Store raw events for plotting markers
            self.dilution_events_for_plot = events_with_volume.copy()

            # Ensure timestamp is datetime
            if not pd.api.types.is_datetime64_any_dtype(events_with_volume['timestamp']):
                events_with_volume['timestamp'] = pd.to_datetime(events_with_volume['timestamp'])

            # Get experiment start time for elapsed hours calculation
            if not self.od_data_df.empty and 'timestamp' in self.od_data_df.columns:
                start_time = self.od_data_df['timestamp'].min()
            else:
                start_time = events_with_volume['timestamp'].min()

            # Aggregate into 1-minute bins
            # This is necessary because Pioreactor pumps in rapid small doses (every 0.5 sec)
            events_with_volume = events_with_volume.sort_values('timestamp')
            events_with_volume['time_bin'] = events_with_volume['timestamp'].dt.floor('1min')

            # Sum volumes per time bin
            binned = events_with_volume.groupby('time_bin').agg({
                'volume_ml': 'sum',
                'timestamp': 'first'  # Keep first timestamp in bin
            }).reset_index()
            binned = binned.rename(columns={'time_bin': 'bin_start', 'timestamp': 'first_event'})
            binned = binned.sort_values('bin_start')

            # Calculate time difference between bins (in hours)
            binned['time_diff_hours'] = binned['bin_start'].diff().dt.total_seconds() / 3600

            # Calculate dilution rate: D = V_total / (V_reactor * Δt)
            # Use the time from the PREVIOUS bin to the current bin
            binned['dilution_rate'] = np.where(
                binned['time_diff_hours'] > 0,
                binned['volume_ml'] / self.reactor_volume / binned['time_diff_hours'],
                np.nan
            )

            # Calculate moving average for smoothing (window of 5 bins = 5 minutes)
            binned['moving_avg_dilution_rate'] = (
                binned['dilution_rate']
                .rolling(window=5, min_periods=1)
                .mean()
            )

            # Add elapsed hours from experiment start
            binned['elapsed_hours'] = (binned['bin_start'] - start_time).dt.total_seconds() / 3600
            binned['timestamp'] = binned['bin_start']

            # Filter out invalid rates
            self.dilution_rate_df = binned[
                binned['moving_avg_dilution_rate'].notna() &
                np.isfinite(binned['moving_avg_dilution_rate'])
            ].copy()

            # Also add elapsed hours to dilution events for plotting
            if not self.dilution_events_for_plot.empty:
                if not pd.api.types.is_datetime64_any_dtype(self.dilution_events_for_plot['timestamp']):
                    self.dilution_events_for_plot['timestamp'] = pd.to_datetime(self.dilution_events_for_plot['timestamp'])
                self.dilution_events_for_plot['elapsed_hours'] = (
                    self.dilution_events_for_plot['timestamp'] - start_time
                ).dt.total_seconds() / 3600

            if not self.dilution_rate_df.empty:
                mean_rate = self.dilution_rate_df['moving_avg_dilution_rate'].mean()
                total_events = len(events_with_volume)
                total_bins = len(self.dilution_rate_df)
                print(f"Calculated dilution rate: mean={mean_rate:.4f} h⁻¹ from {total_events} events in {total_bins} time bins (reactor vol={self.reactor_volume} mL)")
        except Exception as e:
            print(f"Warning: Could not calculate dilution rate: {e}")
            import traceback
            traceback.print_exc()
            self.dilution_rate_df = pd.DataFrame()
            self.dilution_events_for_plot = pd.DataFrame()

    def _calculate_growth_rate_with_core_library(self, filename):
        """Calculate growth rate using core library function."""
        results_list = []

        if self.preprocessed_data is None or self.preprocessed_data.empty or len(self.selected_units) == 0:
            self.show_error("No data loaded.")
            return results_list

        if not CORE_LIBRARY_AVAILABLE:
            self.show_error("Core library not available.")
            return results_list

        start_time = self.selected_region['start']
        end_time = self.selected_region['end']

        if start_time is None or end_time is None or start_time >= end_time:
            self.show_error("Invalid region selected.")
            return results_list

        duration = end_time - start_time

        for unit in self.selected_units:
            try:
                # Filter data for unit and region
                unit_data = self.preprocessed_data[
                    (self.preprocessed_data['exp_unit'] == unit) &
                    (self.preprocessed_data['elapsed_hours'] >= start_time) &
                    (self.preprocessed_data['elapsed_hours'] <= end_time)
                ].copy()

                # Use core library function
                growth_result = calculate_batch_growth_rate(
                    unit_data,
                    start_time=start_time,
                    end_time=end_time,
                    od_column='od_smooth',
                    time_column='elapsed_hours'
                )

                # Calculate yield using core library
                all_unit_data = self.preprocessed_data[
                    self.preprocessed_data['exp_unit'] == unit
                ].copy()

                yield_result = calculate_apparent_yield(
                    all_unit_data,
                    initial_substrate_conc_g_per_l=self.initial_substrate_conc,
                    reactor_volume_ml=self.reactor_volume,
                    od_column='od_smooth',
                    time_column='elapsed_hours'
                )

                # Build result entry
                result_entry = {
                    'Filename': filename,
                    'Unit': unit,
                    'Region Start (h)': start_time,
                    'Region End (h)': end_time,
                    'Duration (h)': duration,
                    'Growth Rate (h⁻¹)': growth_result.growth_rate,
                    'Doubling Time (h)': growth_result.doubling_time,
                    'R²': growth_result.r_squared,
                    'Std Error': growth_result.std_error,
                    'CI Lower': growth_result.ci_lower,
                    'CI Upper': growth_result.ci_upper,
                    'P-value': growth_result.p_value,
                    'Data Points': growth_result.n_points,
                    'Apparent Yield (OD/g)': yield_result.yield_value,
                    'Yield Status': yield_result.status,
                    'Max OD': yield_result.max_od,
                    'ΔOD': yield_result.delta_od
                }

                results_list.append(result_entry)

            except Exception as e:
                self.show_debug(f"Calculation failed for {unit}: {str(e)}")
                # Add entry with NaNs
                result_entry = {
                    'Filename': filename,
                    'Unit': unit,
                    'Region Start (h)': start_time,
                    'Region End (h)': end_time,
                    'Duration (h)': duration,
                    'Growth Rate (h⁻¹)': np.nan,
                    'Doubling Time (h)': np.nan,
                    'R²': np.nan,
                    'Std Error': np.nan,
                    'CI Lower': np.nan,
                    'CI Upper': np.nan,
                    'P-value': np.nan,
                    'Data Points': 0,
                    'Apparent Yield (OD/g)': np.nan,
                    'Yield Status': 'N/A',
                    'Max OD': np.nan,
                    'ΔOD': np.nan
                }
                results_list.append(result_entry)

        return results_list

    def _add_analysis_callback(self, event):
        """Add current analysis to results table."""
        if self.current_filename is None:
            self.show_error("Please upload a CSV file first.")
            return

        if not self.selected_units:
            self.show_error("Please select at least one unit.")
            return

        if self.selected_region['start'] is None or self.selected_region['end'] is None:
            self.show_error("Invalid region selected.")
            return

        # Calculate using core library
        current_results_list = self._calculate_growth_rate_with_core_library(self.current_filename)

        if not current_results_list:
            self.show_error("Calculation failed.")
            return

        # Add to cumulative results
        new_results_df = pd.DataFrame(current_results_list)
        self.cumulative_results_df = pd.concat([self.cumulative_results_df, new_results_df], ignore_index=True)

        self._update_results_display()
        self.show_success(f"Added {len(new_results_df)} analysis row(s) to results table.")

    def _export_results_callback(self, event):
        """Export cumulative results to CSV."""
        if self.cumulative_results_df.empty:
            self.show_error("No results to export.")
            return

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"growth_rate_analysis_{timestamp}.csv"
            self.cumulative_results_df.to_csv(filename, index=False)
            self.show_success(f"Results exported to {filename}")
        except Exception as e:
            self.show_error(f"Failed to export: {str(e)}")

    def _clear_results_callback(self, event):
        """Clear all results."""
        if self.cumulative_results_df.empty:
            return

        self.cumulative_results_df = self.cumulative_results_df[0:0]
        self._update_results_display()
        self.show_success("All results cleared.")

    def _export_script_callback(self, event):
        """Export current analysis as standalone Python script."""
        if self.current_filepath is None or self.preprocessed_data is None:
            self.show_error("No data loaded. Upload a file first.")
            return

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            script_filename = f"analysis_script_{timestamp}.py"

            script_content = f'''#!/usr/bin/env python3
"""
Generated analysis script for {self.current_filename}
Created: {datetime.now().isoformat()}

This script was exported from the interactive UI and reproduces
the current analysis settings exactly.
"""

from pathlib import Path
from pioreactor_analysis import (
    PioreactorCSVParser,
    preprocess_od_data,
    calculate_batch_growth_rate,
    auto_detect_exponential_phase,
    calculate_apparent_yield,
)

# Parse CSV file
parser = PioreactorCSVParser()
data = parser.parse("{self.current_filepath}")

# Convert to DataFrame (support multiple data model types)
if hasattr(data, 'to_dataframe'):
    df = data.to_dataframe()
elif hasattr(data, 'to_od_dataframe'):
    df = data.to_od_dataframe()
else:
    raise AttributeError(f"Parsed data object of type {type(data).__name__} has no 'to_dataframe' or 'to_od_dataframe' method")

# Add elapsed time calculation
import pandas as pd
df['timestamp'] = pd.to_datetime(df['timestamp'])
df_list = []
for (exp, unit), group in df.groupby(['experiment', 'unit']):
    start_time = group['timestamp'].min()
    group['elapsed_hours'] = (group['timestamp'] - start_time).dt.total_seconds() / 3600
    group['exp_unit'] = f"{{exp}}_{{unit}}"
    df_list.append(group)
df = pd.concat(df_list)

# Preprocess data
preprocessed = preprocess_od_data(
    df,
    smoothing_window={self.smoothing_window},
    min_od_threshold={self.min_od_threshold},
    time_column='elapsed_hours',
    od_column='od_value',
    group_by=['experiment', 'unit']
)

# Filter for selected unit(s)
selected_units = {self.selected_units}
unit_data = preprocessed[preprocessed['exp_unit'].isin(selected_units)].copy()

# Calculate growth rate for selected region
for unit in selected_units:
    unit_filtered = unit_data[unit_data['exp_unit'] == unit].copy()

    growth_result = calculate_batch_growth_rate(
        unit_filtered,
        start_time={self.selected_region['start']},
        end_time={self.selected_region['end']},
        od_column='od_smooth',
        time_column='elapsed_hours'
    )

    # Calculate apparent yield
    yield_result = calculate_apparent_yield(
        unit_filtered,
        initial_substrate_conc_g_per_l={self.initial_substrate_conc},
        reactor_volume_ml={self.reactor_volume},
        od_column='od_smooth',
        time_column='elapsed_hours'
    )

    # Print results
    print(f"\\nResults for {{unit}}:")
    print(f"  Growth rate: {{growth_result.growth_rate:.4f}} h^-1")
    print(f"  Doubling time: {{growth_result.doubling_time:.2f}} hours")
    print(f"  R²: {{growth_result.r_squared:.4f}}")
    print(f"  95% CI: [{{growth_result.ci_lower:.4f}}, {{growth_result.ci_upper:.4f}}]")
    print(f"  P-value: {{growth_result.p_value:.2e}}")
    print(f"  Data points: {{growth_result.n_points}}")
    print(f"  Apparent yield: {{yield_result.yield_value:.4f}} OD/g ({{yield_result.status}})")
    print(f"  Max OD: {{yield_result.max_od:.4f}}")
    print(f"  ΔOD: {{yield_result.delta_od:.4f}}")
'''

            with open(script_filename, 'w') as f:
                f.write(script_content)

            self.show_success(f"Analysis script exported to {script_filename}")

        except Exception as e:
            self.show_error(f"Failed to export script: {str(e)}")
            self.show_debug(f"Error details:\n{traceback.format_exc()}")

    def _export_publication_callback(self, event):
        """Export publication-quality figure using core library."""
        if self.preprocessed_data is None or self.preprocessed_data.empty:
            self.show_error("No data to export. Upload a file first.")
            return

        if not CORE_LIBRARY_AVAILABLE:
            self.show_error("Core library not available.")
            return

        try:
            # Get journal theme
            journal_map = {
                'Nature': JournalTheme.NATURE,
                'Science': JournalTheme.SCIENCE,
                'ACS': JournalTheme.ACS,
                'Elsevier': JournalTheme.ELSEVIER,
                'PLOS': JournalTheme.PLOS
            }
            journal_theme = journal_map[self.journal_selector.value]

            # Create plot config
            config = PlotConfig.from_journal(journal_theme)
            plotter = PublicationPlotter(config)

            # Calculate growth result for plot
            unit = self.selected_units[0] if self.selected_units else None
            if unit is None:
                self.show_error("Please select a unit first.")
                return

            unit_data = self.preprocessed_data[
                (self.preprocessed_data['exp_unit'] == unit) &
                (self.preprocessed_data['elapsed_hours'] >= self.selected_region['start']) &
                (self.preprocessed_data['elapsed_hours'] <= self.selected_region['end'])
            ].copy()

            growth_result = calculate_batch_growth_rate(
                unit_data,
                start_time=self.selected_region['start'],
                end_time=self.selected_region['end'],
                od_column='od_smooth',
                time_column='elapsed_hours'
            )

            # Generate publication figure
            all_unit_data = self.preprocessed_data[
                self.preprocessed_data['exp_unit'] == unit
            ].copy()

            fig, axes = plotter.plot_growth_curve(
                all_unit_data,
                growth_result,
                title=f"Batch Culture Growth Analysis - {unit}",
                add_panel_labels=True
            )

            # Save figure
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = Path(f"publication_figure_{timestamp}")
            plotter.save(fig, output_path, formats=['png', 'pdf'])
            plotter.close(fig)

            self.show_success(f"Publication figure exported: {output_path}.png and .pdf")

        except PermissionError as e:
            self.show_error(f"Permission denied writing file: {str(e)}")
            self.show_debug(f"Error details:\n{traceback.format_exc()}")
        except OSError as e:
            self.show_error(f"File system error: {str(e)}")
            self.show_debug(f"Error details:\n{traceback.format_exc()}")
        except Exception as e:
            self.show_error(f"Failed to export publication figure: {str(e)}")
            self.show_debug(f"Error details:\n{traceback.format_exc()}")

    def _export_od_plot_callback(self, event):
        """Export OD plot to PNG (quick export)."""
        if self.od_plot.object is None or str(self.od_plot.object) == 'Text':
            self.show_error("No OD plot to export.")
            return

        if not BOKEH_EXPORT_AVAILABLE:
            self.show_error(
                "PNG export unavailable. Install dependencies: pip install selenium webdriver-manager\n"
                "Then restart the application. Alternatively, use 'Export Publication Figure' which uses matplotlib."
            )
            return

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"od_plot_{timestamp}.png"
            hv.save(self.od_plot.object, filename, fmt='png', dpi=self.plot_dpi)
            self.show_success(f"OD plot exported to {filename}")
        except Exception as e:
            error_msg = str(e).lower()
            if 'selenium' in error_msg or 'webdriver' in error_msg or 'geckodriver' in error_msg:
                self.show_error(
                    "Export failed: WebDriver not found. Install via:\n"
                    "pip install selenium webdriver-manager\n"
                    "Or use 'Export Publication Figure' instead."
                )
            else:
                self.show_error(f"Failed to export plot: {str(e)}")
            self.show_debug(f"Export error details:\n{traceback.format_exc()}")

    def _export_ln_od_plot_callback(self, event):
        """Export ln(OD) plot to PNG (quick export)."""
        if self.ln_od_plot.object is None or str(self.ln_od_plot.object) == 'Text':
            self.show_error("No ln(OD) plot to export.")
            return

        if not BOKEH_EXPORT_AVAILABLE:
            self.show_error(
                "PNG export unavailable. Install dependencies: pip install selenium webdriver-manager\n"
                "Then restart the application. Alternatively, use 'Export Publication Figure' which uses matplotlib."
            )
            return

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ln_od_plot_{timestamp}.png"
            hv.save(self.ln_od_plot.object, filename, fmt='png', dpi=self.plot_dpi)
            self.show_success(f"ln(OD) plot exported to {filename}")
        except Exception as e:
            error_msg = str(e).lower()
            if 'selenium' in error_msg or 'webdriver' in error_msg or 'geckodriver' in error_msg:
                self.show_error(
                    "Export failed: WebDriver not found. Install via:\n"
                    "pip install selenium webdriver-manager\n"
                    "Or use 'Export Publication Figure' instead."
                )
            else:
                self.show_error(f"Failed to export plot: {str(e)}")
            self.show_debug(f"Export error details:\n{traceback.format_exc()}")

    def _update_results_display(self):
        """Update the results table display."""
        if self.cumulative_results_df.empty:
            self.results_output.object = "<p>No analysis results added yet. Select units/region and click 'Add Current Analysis to Table'.</p>"
            return

        # Create HTML table header
        html = "<h3>Cumulative Growth Rate Analysis Results</h3>"
        html += "<p><em>Calculations performed by pioreactor_analysis core library v2.0.0</em></p>"

        # Add max growth rate summary
        max_growth_rate = self.cumulative_results_df['Growth Rate (h⁻¹)'].max()
        max_growth_idx = self.cumulative_results_df['Growth Rate (h⁻¹)'].idxmax()
        max_growth_row = self.cumulative_results_df.loc[max_growth_idx]
        max_doubling_time = np.log(2) / max_growth_rate if max_growth_rate > 0 else np.inf

        html += f"""
        <div style="margin-bottom: 20px; padding: 15px; background-color: #e8f4f8; border-left: 4px solid #2196F3; border-radius: 4px;">
          <h4 style="margin-top: 0; color: #1976D2;">Maximum Specific Growth Rate (μmax)</h4>
          <div style="font-size: 24px; font-weight: bold; color: #1976D2; margin: 10px 0;">
            μmax = {max_growth_rate:.4f} h⁻¹
          </div>
          <div style="font-size: 14px; color: #555;">
            <strong>Doubling Time:</strong> {max_doubling_time:.2f} hours<br>
            <strong>Source:</strong> {max_growth_row['Filename']} - {max_growth_row['Unit']}<br>
            <strong>Time Range:</strong> {max_growth_row['Region Start (h)']:.2f}h to {max_growth_row['Region End (h)']:.2f}h<br>
            <strong>R²:</strong> {max_growth_row['R²']:.4f}
          </div>
        </div>
        """

        html += """
        <table style="width:100%; border-collapse:collapse; margin-top:10px;">
          <thead>
            <tr style="background-color:#f2f2f2;">
              <th style="padding:8px; border:1px solid #ddd; text-align:left;">Filename</th>
              <th style="padding:8px; border:1px solid #ddd; text-align:left;">Unit</th>
              <th style="padding:8px; border:1px solid #ddd; text-align:right;">Start (h)</th>
              <th style="padding:8px; border:1px solid #ddd; text-align:right;">End (h)</th>
              <th style="padding:8px; border:1px solid #ddd; text-align:right;">Growth Rate (h⁻¹)</th>
              <th style="padding:8px; border:1px solid #ddd; text-align:right;">Doubling Time (h)</th>
              <th style="padding:8px; border:1px solid #ddd; text-align:right;">R²</th>
              <th style="padding:8px; border:1px solid #ddd; text-align:right;">Data Points</th>
              <th style="padding:8px; border:1px solid #ddd; text-align:right;">Yield (OD/g)</th>
            </tr>
          </thead>
          <tbody>
        """

        # Add table rows
        for idx, row in self.cumulative_results_df.iterrows():
            row_style = "background-color: #fff9c4;" if idx == max_growth_idx else ""

            html += f"""
            <tr style="{row_style}">
              <td style="padding:8px; border:1px solid #ddd;">{row['Filename']}</td>
              <td style="padding:8px; border:1px solid #ddd;">{row['Unit']}</td>
              <td style="padding:8px; border:1px solid #ddd; text-align:right;">{row['Region Start (h)']:.2f}</td>
              <td style="padding:8px; border:1px solid #ddd; text-align:right;">{row['Region End (h)']:.2f}</td>
              <td style="padding:8px; border:1px solid #ddd; text-align:right;">{row['Growth Rate (h⁻¹)']:.4f}</td>
              <td style="padding:8px; border:1px solid #ddd; text-align:right;">{row['Doubling Time (h)']:.2f}</td>
              <td style="padding:8px; border:1px solid #ddd; text-align:right;">{row['R²']:.4f}</td>
              <td style="padding:8px; border:1px solid #ddd; text-align:right;">{int(row['Data Points'])}</td>
              <td style="padding:8px; border:1px solid #ddd; text-align:right;">{row['Apparent Yield (OD/g)']:.4f if pd.notna(row['Apparent Yield (OD/g)']) else 'N/A'}</td>
            </tr>
            """

        html += """
          </tbody>
        </table>
        """

        self.results_output.object = html

    def _update_data_preview(self):
        """Update data preview pane."""
        if self.od_data_df.empty:
            self.data_preview.object = ""
            return

        total_rows = len(self.od_data_df)
        experiments = self.od_data_df['experiment'].nunique()
        units = self.od_data_df['unit'].nunique() if 'unit' in self.od_data_df.columns else 0

        # Time range
        if 'elapsed_hours' in self.od_data_df.columns:
            time_min = self.od_data_df['elapsed_hours'].min()
            time_max = self.od_data_df['elapsed_hours'].max()
            time_range = f"{time_min:.2f}h - {time_max:.2f}h ({time_max - time_min:.2f}h duration)"
        else:
            time_range = "N/A"

        # OD range
        od_col = 'od_value' if 'od_value' in self.od_data_df.columns else 'od_reading'
        if od_col in self.od_data_df.columns:
            od_min = self.od_data_df[od_col].min()
            od_max = self.od_data_df[od_col].max()
            od_mean = self.od_data_df[od_col].mean()
            od_range = f"{od_min:.4f} - {od_max:.4f} (mean: {od_mean:.4f})"
        else:
            od_range = "N/A"

        html = f"""
        <div style="border: 1px solid #ddd; border-radius: 5px; padding: 15px; margin: 10px 0; background-color: #f9f9f9;">
            <h3 style="margin-top:0; color: #333;">Data Preview</h3>
            <div style="margin-bottom: 15px;">
                <strong>Summary:</strong>
                <ul style="margin: 5px 0;">
                    <li>Total Rows: {total_rows}</li>
                    <li>Experiments: {experiments}</li>
                    <li>Units: {units}</li>
                    <li>Time Range: {time_range}</li>
                    <li>OD Range: {od_range}</li>
                </ul>
            </div>
            <strong>First 10 Rows:</strong>
            <div style="overflow-x: auto; margin-top: 10px;">
                {self.od_data_df.head(10).to_html(index=False, border=1, classes='dataframe',
                                                   float_format=lambda x: f'{x:.4f}' if abs(x) < 1000 else f'{x:.2f}')}
            </div>
        </div>
        """

        self.data_preview.object = html

    def view(self):
        """Return the main layout."""
        return self.main_layout


# Create the application
if CORE_LIBRARY_AVAILABLE:
    growth_analysis = GrowthRateAnalysisRefactored()
    app = pn.panel(growth_analysis.view())
else:
    # Fallback message if core library not available
    app = pn.pane.Markdown("""
    # Error: Core Library Not Available

    The `pioreactor_analysis` core library is required but could not be imported.

    Please ensure the library is installed:
    ```bash
    pip install -e .
    ```

    Or check that the package is in your Python path.
    """)

# Server
if __name__ == '__main__':
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # Allow overriding the port via environment variable PIO_PORT
    try:
        port = int(os.environ.get('PIO_PORT', '5007'))
    except Exception:
        port = 5007
    app.show(port=port)
