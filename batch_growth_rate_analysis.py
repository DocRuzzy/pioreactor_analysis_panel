# batch_growth_rate_analysis.py
"""
Interactive Growth Rate Analysis Tool with Panel/HoloViz

This application provides an interactive dashboard for analyzing batch culture growth rates
from Pioreactor OD measurements. It allows users to upload CSV data files, visualize
growth curves, and calculate growth rates for selected regions.

Features:
- Interactive time series visualization of OD readings
- Log-transformed OD view to identify exponential growth phase
- Region selection for targeted growth rate analysis
- Statistical analysis including growth rate, doubling time, and R²
- Support for multi-unit experiments

Author: Based on work by Russell Kirk Pirlo
Date: April 25, 2025
"""

import pandas as pd
import numpy as np
import hvplot.pandas
import holoviews as hv
import panel as pn
import param
from bokeh.models import ColumnDataSource, Range1d, Span, BoxSelectTool
from bokeh.models.formatters import DatetimeTickFormatter
from bokeh.models.callbacks import CustomJS
import io
from scipy import stats
import traceback # Added for detailed error logging

# Configure Panel and HoloViews
pn.extension('plotly', 'tabulator')
hv.extension('bokeh')

# Style the panel with a nice theme
pn.config.sizing_mode = 'stretch_width'

class GrowthRateAnalysis(param.Parameterized):
    """
    Main application class for batch culture growth rate analysis.
    
    This class manages data processing, visualization, and the user interface
    for analyzing batch culture OD data.
    
    Parameters
    ----------
    smoothing_window : int
        Number of readings to include in moving average calculations (default: 5)
    min_od_threshold : float
        Minimum OD value to consider for growth rate calculation (default: 0.05)
    semi_log_plot : boolean
        Use semi-logarithmic scale for OD plot (default: True)
    """
    
    smoothing_window = param.Integer(5, bounds=(1, 50), step=1, 
                                    doc="Number of consecutive data points to average for smoothing. Higher values = smoother curves but may hide real fluctuations.")
    min_od_threshold = param.Number(0.05, bounds=(0.001, 1.0), step=0.01, 
                                   doc="Minimum OD value to include in analysis. Data below this threshold will be filtered out to exclude lag phase.")
    semi_log_plot = param.Boolean(True, doc="Display OD on logarithmic scale. Recommended: keep enabled to better visualize exponential growth.")
    reactor_volume = param.Number(14.0, bounds=(1.0, 100.0), step=0.1,
                                 doc="Total volume of culture in the reactor (mL). Used for yield calculations.")
    initial_substrate_conc = param.Number(20.0, bounds=(0.0, 200.0), step=0.5,
                                         doc="Initial substrate concentration in g/L (e.g., glucose). Required for accurate yield calculations.")
    
    # Plot appearance parameters
    plot_width = param.Integer(800, bounds=(400, 1600), step=50,
                              doc="Width of plots in pixels.")
    plot_height = param.Integer(400, bounds=(200, 800), step=50,
                               doc="Height of plots in pixels.")
    plot_dpi = param.Integer(100, bounds=(72, 300), step=10,
                            doc="DPI (dots per inch) for exported plot images. Higher = better quality but larger files.")
    font_size = param.Integer(12, bounds=(8, 20), step=1,
                             doc="Base font size for plot labels and legends.")
    color_scheme = param.Selector(default='default', objects=['default', 'colorblind', 'grayscale'],
                                 doc="Color scheme for plots. Colorblind-friendly uses distinct hues; grayscale for printing.")
    
    def __init__(self, **params):
        """
        Initialize the growth rate analysis application.
        
        Sets up the UI components, data containers, and layout structure.
        """
        super().__init__(**params)
        
        # Initialize empty data containers
        self.od_data_df = pd.DataFrame()
        self.selected_region = {'start': None, 'end': None}
        self.selected_units = []
        self.current_filename = None # Store the name of the current file
        
        # DataFrame to store cumulative results
        self.cumulative_results_df = pd.DataFrame(columns=[
            'Filename', 'Unit', 'Region Start (h)', 'Region End (h)', 
            'Duration (h)', 'Growth Rate (h⁻¹)', 'Doubling Time (h)', 
            'R²', 'Std Error', 'CI Lower', 'CI Upper', 'P-value', 'Data Points',
            'Apparent Yield (OD/g)', 'Yield Status', 'Max OD', 'ΔOD'
        ])
        
    # Status message for notifications
        self.status_message = pn.pane.Markdown("", styles={'color': 'blue'})
        self.error_message = pn.pane.Markdown("", styles={'color': 'red'})
        self.debug_message = pn.pane.Markdown("", styles={'color': 'green', 'font-family': 'monospace', 'font-size': '12px'})
        
        # Loading spinner (inactive by default)
        self.loading_spinner = pn.indicators.LoadingSpinner(value=False, width=24, height=24)
        
        # Data preview pane
        self.data_preview = pn.pane.HTML("", styles={'overflow-x': 'auto'})
        
        # Set up the interface
        self.file_input = pn.widgets.FileInput(accept='.csv', multiple=False)
        self.file_input.param.watch(self._upload_file_callback, 'value')
        
        self.update_button = pn.widgets.Button(name='Update Plots', button_type='primary')
        self.update_button.on_click(self._update_plots_callback)
        
        # Unit selection for multi-unit experiments
        self.unit_selector = pn.widgets.MultiSelect(name='Select Units', options=[])
        self.unit_selector.param.watch(self._units_changed, 'value')
        
        # Create plot panes
        self.od_plot = pn.pane.HoloViews(sizing_mode='stretch_width', height=300)
        self.ln_od_plot = pn.pane.HoloViews(sizing_mode='stretch_width', height=300)
        
        # Selected region widgets - sliders
        self.region_start = pn.widgets.FloatSlider(name='Region Start (hours)', start=0, end=100, step=0.1, value=0)
        self.region_end = pn.widgets.FloatSlider(name='Region End (hours)', start=0, end=100, step=0.1, value=100)
        
        # Add manual input text fields for precise control
        self.region_start_input = pn.widgets.FloatInput(name='Start Value', width=100)
        self.region_end_input = pn.widgets.FloatInput(name='End Value', width=100)
        
        # Set up callbacks for all region control widgets
        self.region_start.param.watch(self._update_region_from_slider, 'value')
        self.region_end.param.watch(self._update_region_from_slider, 'value')
        self.region_start_input.param.watch(self._update_region_from_input, 'value')
        self.region_end_input.param.watch(self._update_region_from_input, 'value')
        
        # Auto-detect button
        self.auto_detect_button = pn.widgets.Button(name='Auto-detect Exponential Phase', button_type='success')
        self.auto_detect_button.on_click(self._auto_detect_callback)
        
        # Results output (will display the cumulative table)
        self.results_output = pn.pane.HTML("")
        
        # Button to add current analysis to the results table
        self.add_analysis_button = pn.widgets.Button(name='Add Current Analysis to Table', button_type='primary')
        self.add_analysis_button.on_click(self._add_analysis_callback)
        
        # Button to export results to CSV
        self.export_csv_button = pn.widgets.Button(name='📥 Export Results to CSV', button_type='success')
        self.export_csv_button.on_click(self._export_results_callback)
        
        # Button to clear cumulative results
        self.clear_results_button = pn.widgets.Button(name='🗑️ Clear All Results', button_type='danger')
        self.clear_results_button.on_click(self._clear_results_callback)
        
        # Batch export all button
        self.export_all_button = pn.widgets.Button(name='📦 Export All (Plots + Data)', button_type='success', width=200)
        self.export_all_button.on_click(self._export_all_callback)
        
        # Plot export buttons
        self.export_od_plot_button = pn.widgets.Button(name='📊 Export OD Plot', button_type='default', width=150)
        self.export_od_plot_button.on_click(self._export_od_plot_callback)
        
        self.export_ln_od_plot_button = pn.widgets.Button(name='📊 Export ln(OD) Plot', button_type='default', width=150)
        self.export_ln_od_plot_button.on_click(self._export_ln_od_plot_callback)
        
        # Session save/load buttons
        self.save_session_button = pn.widgets.Button(name='💾 Save Session', button_type='primary', width=150)
        self.save_session_button.on_click(self.save_session_callback)
        
        self.load_session_input = pn.widgets.FileInput(accept='.json', multiple=False, name='📂 Load Session')
        self.load_session_input.param.watch(self.load_session_callback, 'value')
        
        self.status_text = pn.pane.Markdown("", styles={'font-size': '12px', 'margin-top': '10px'})
        
        # Loading indicator (reuse existing spinner)
        self.loading_indicator = self.loading_spinner
        
        # Controls grouped in a compact Accordion for separate Controls tab
        controls_accordion = pn.Accordion(
            (
                "Upload & Session",
                pn.Column(
                    pn.pane.Markdown("### Upload Data"),
                    self.file_input,
                    pn.Row(self.loading_spinner),
                    self.status_message,
                    self.error_message,
                    self.data_preview,
                    pn.pane.Markdown("---"),
                    pn.pane.Markdown("### Session Management"),
                    pn.Row(self.save_session_button, self.load_session_input),
                    self.status_text,
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
            active=[],  # start collapsed
            sizing_mode='stretch_width'
        )
        
        # Main layout with plots and analysis tools first, controls in separate tab
        self.main_layout = pn.Tabs(
            (
                'Analysis',
                pn.Column(
                    pn.pane.Markdown("# Growth Rate Analysis for Batch Culture"),
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
                            self.auto_detect_button,
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
                        self.export_all_button
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
        # Initial update of the (empty) results display
        self._update_results_display()
        self.param.watch(self._update_plots_callback, 'semi_log_plot')
        
        # Watch plot aesthetic parameters
        self.param.watch(self._update_plots_callback, ['plot_width', 'plot_height', 'font_size', 'color_scheme'])

    def show_success(self, message):
        """Display a success message to the user."""
        self.status_message.object = f"✅ **{message}**"
        self.error_message.object = ""
    
    def show_error(self, message):
        """Display an error message to the user."""
        self.error_message.object = f"❌ **Error:** {message}"
        self.status_message.object = ""
    
    def show_debug(self, message):
        """Display a debug message on the debug tab."""
        self.debug_message.object = f"```\n{message}\n```"
    
    def _get_color_palette(self):
        """Get the current color palette based on the selected scheme."""
        if self.color_scheme == 'colorblind':
            # Wong's colorblind-friendly palette
            return ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7', '#000000']
        elif self.color_scheme == 'grayscale':
            # Grayscale palette
            return ['#000000', '#404040', '#808080', '#C0C0C0', '#606060', '#A0A0A0', '#303030', '#909090']
        else:
            # Default HoloViews palette
            return None  # Use HoloViews default
    
    def _get_plot_options(self):
        """Get plot options dictionary for consistent styling."""
        opts_dict = {
            'width': self.plot_width,
            'height': self.plot_height,
            'fontsize': {
                'title': self.font_size + 2,
                'labels': self.font_size,
                'xticks': self.font_size - 2,
                'yticks': self.font_size - 2,
                'legend': self.font_size - 1,
            }
        }
        
        # Add color palette if not default
        palette = self._get_color_palette()
        if palette:
            opts_dict['color'] = hv.Cycle(palette)
        
        return opts_dict

    def _upload_file_callback(self, event):
        """
        Handle file upload events from the UI.
        
        Reads the uploaded CSV file, processes the data, and updates the visualization.
        Resets the cumulative results if desired (currently keeps accumulating).
        """
        if self.file_input.value is not None and self.file_input.filename.endswith('.csv'):
            # Show loading spinner during file processing
            self.loading_spinner.value = True
            try:
                self.current_filename = self.file_input.filename # Store filename
                
                # Decode the file contents
                decoded = io.BytesIO(self.file_input.value)
                
                # Read the CSV file
                df = pd.read_csv(decoded)
                
                # Process the data
                self._process_data(df)
                
                # Debug information
                cols_info = ", ".join(df.columns)
                debug_info = f"CSV loaded: {self.current_filename}, {len(df)} rows, {len(df.columns)} columns\n"
                debug_info += f"Columns: {cols_info}\n"
                
                # Sample data rows for debugging
                debug_info += "\nSample rows:\n"
                for i, row in df.head(3).iterrows():
                    debug_info += f"Row {i}:\n"
                    for col in df.columns:
                        debug_info += f"  {col}: {row.get(col, 'N/A')}\n"
                
                self.show_debug(debug_info)
                
                # Validate data quality and show warnings
                self._validate_data_quality()
                
                # Display data preview
                self._update_data_preview()
                
                # Update plots
                self._update_plots()
                
                # Show success message
                self.show_success(f"File {self.current_filename} uploaded successfully!")
                
                # Optional: Reset cumulative results when a new file is loaded
                # self.cumulative_results_df = self.cumulative_results_df[0:0] # Clear DataFrame
                # self._update_results_display() 
                
            except Exception as e:
                tb = traceback.format_exc()
                self.show_error(f"Error processing file: {str(e)}")
                self.show_debug(f"Error details:\n{tb}")
                self.current_filename = None # Reset filename on error
            finally:
                # Hide loading spinner
                self.loading_spinner.value = False
        else:
            self.show_error("Please upload a CSV file.")
            self.current_filename = None
            self.loading_spinner.value = False

    def _process_data(self, df):
        """
        Process the uploaded CSV data for analysis.
        
        Converts timestamps, calculates elapsed time, and organizes data by experiment and unit.
        """
        # Convert timestamps to datetime objects
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        if 'timestamp_localtime' in df.columns:
            df['timestamp_localtime'] = pd.to_datetime(df['timestamp_localtime'])
            # Use timestamp_localtime for elapsed time calculation
            df['timestamp_to_use'] = df['timestamp_localtime']
        else:
            # Fallback to timestamp if timestamp_localtime doesn't exist
            df['timestamp_to_use'] = df['timestamp']
        
        # Calculate elapsed time in hours for each experiment/unit
        df_list = []
        for (exp, unit), group in df.groupby(['experiment', 'pioreactor_unit']):
            start_time = group['timestamp_to_use'].min()
            group['elapsed_hours'] = (group['timestamp_to_use'] - start_time).dt.total_seconds() / 3600
            group['exp_unit'] = f"{exp}_{unit}"
            df_list.append(group)
        
        # Combine all processed dataframes
        if df_list:
            self.od_data_df = pd.concat(df_list)
            
            # Update unit selector options
            units = sorted(self.od_data_df['exp_unit'].unique())
            self.unit_selector.options = units
            if units:
                self.unit_selector.value = [units[0]]  # Select first unit by default
                self.selected_units = [units[0]]
            
            # Update region sliders based on data range
            min_time = self.od_data_df['elapsed_hours'].min()
            max_time = self.od_data_df['elapsed_hours'].max()
            
            # Update slider ranges and values
            self.region_start.start = min_time
            self.region_start.end = max_time
            self.region_start.value = min_time
            
            self.region_end.start = min_time
            self.region_end.end = max_time
            self.region_end.value = max_time
            
            # Update the text input values to match sliders
            self.region_start_input.value = min_time
            self.region_end_input.value = max_time
            
            self.selected_region = {'start': min_time, 'end': max_time}
        else:
            self.show_error("No valid data found in the CSV file.")
            self.od_data_df = pd.DataFrame() # Ensure it's empty

    def _units_changed(self, event):
        """Handle unit selection changes."""
        self.selected_units = self.unit_selector.value
        self._update_plots() # Only update plots, calculation is separate

    def _update_region_from_slider(self, event):
        """Handle region slider updates and synchronize with text inputs."""
        # Update the corresponding text input
        if event.obj is self.region_start:
            self.region_start_input.value = event.new
        elif event.obj is self.region_end:
            self.region_end_input.value = event.new
        
        self._update_region()
    
    def _update_region_from_input(self, event):
        """Handle text input updates and synchronize with sliders."""
        # Update the corresponding slider
        if event.obj is self.region_start_input:
            # Make sure the value is within slider bounds
            value = max(min(event.new, self.region_start.end), self.region_start.start)
            self.region_start.value = value
            # Ensure the input shows the constrained value
            if value != event.new:
                self.region_start_input.value = value
        elif event.obj is self.region_end_input:
            # Make sure the value is within slider bounds
            value = max(min(event.new, self.region_end.end), self.region_end.start)
            self.region_end.value = value
            # Ensure the input shows the constrained value
            if value != event.new:
                self.region_end_input.value = value
        
        self._update_region()
    
    def _update_region(self):
        """Update the selected region values and validate them."""
        # Ensure start <= end
        if self.region_start.value > self.region_end.value:
            # Adjust end to match start
            self.region_end.value = self.region_start.value
            self.region_end_input.value = self.region_start.value
        
        self.selected_region = {
            'start': self.region_start.value,
            'end': self.region_end.value
        }
        
        # Update plots with selected region
        self._update_plots()

    def _update_plots_callback(self, event):
        """Handle update button click events or semi_log_plot changes."""
        self.loading_spinner.value = True
        try:
            self._update_plots()
            self.show_success("Plots updated with current settings.")
        finally:
            self.loading_spinner.value = False
        # Add debug info
        self.show_debug(f"Plot update triggered. Semi-log mode: {self.semi_log_plot}")

    def _auto_detect_callback(self, event):
        """
        Automatically detect the exponential growth phase with focus on maximum specific growth rate.
        Prioritizes early exponential phase with highest growth rate and good linear fit.
        """
        if self.od_data_df.empty or len(self.selected_units) == 0:
            self.show_error("No data available for auto-detection.")
            return
        
        self.loading_spinner.value = True
        # Get the first selected unit (auto-detection works on one unit at a time)
        unit = self.selected_units[0]
        
        # Get data for the selected unit
        unit_data = self.od_data_df[self.od_data_df['exp_unit'] == unit].copy()
        
        # Sort by time
        unit_data = unit_data.sort_values('elapsed_hours')
        
        # Apply smoothing
        unit_data['od_smooth'] = unit_data['od_reading'].rolling(
            window=self.smoothing_window, min_periods=1, center=True).mean()
        
        # Filter out OD values below threshold
        unit_data = unit_data[unit_data['od_smooth'] >= self.min_od_threshold]
        
        if len(unit_data) < 10:  # Need enough points for detection
            self.show_error(f"Not enough valid data points for unit {unit} to auto-detect.")
            self.loading_spinner.value = False
            return
        
        # Calculate ln(OD)
        unit_data['ln_od'] = np.log(unit_data['od_smooth'])
        
        # Dynamic window size: smaller for better resolution, but need enough points
        min_window = max(8, len(unit_data) // 10)  # At least 8 points, or 10% of data
        max_window = min(25, len(unit_data) // 3)   # Max 25 points, or 33% of data
        
        # Try multiple window sizes and keep the best results
        all_results = []
        
        for window_size in range(min_window, max_window + 1, 2):  # Step by 2 for efficiency
            for i in range(len(unit_data) - window_size):
                window = unit_data.iloc[i:i+window_size]
                
                try:
                    # Check for sufficient variation in x values
                    if np.ptp(window['elapsed_hours']) < 1e-9: 
                        continue
                    
                    slope, intercept, r_value, p_value, std_err = stats.linregress(
                        window['elapsed_hours'], window['ln_od']
                    )
                    
                    # Only consider positive growth rates with good fit
                    if slope > 0 and r_value**2 > 0.90:  # Require R² > 0.90
                        midpoint_time = window['elapsed_hours'].iloc[window_size//2]
                        start_time = window['elapsed_hours'].iloc[0]
                        end_time = window['elapsed_hours'].iloc[-1]
                        
                        all_results.append({
                            'start_idx': i,
                            'window_size': window_size,
                            'start_time': start_time,
                            'end_time': end_time,
                            'midpoint_time': midpoint_time,
                            'growth_rate': slope,
                            'r_squared': r_value**2,
                            'std_err': std_err
                        })
                        
                except ValueError:
                    continue
        
        if not all_results:
            self.show_error("Could not find any exponential regions with good fit (R² > 0.90). Try adjusting smoothing or threshold.")
            self.loading_spinner.value = False
            return
        
        # Convert to DataFrame for easier manipulation
        results_df = pd.DataFrame(all_results)
        
        # Calculate composite score prioritizing:
        # 1. Maximum growth rate (most important for max μ)
        # 2. Early timing (exponential phase is typically early)
        # 3. Good R² fit (ensure quality)
        
        # Normalize growth rates to 0-1 scale
        gr_normalized = (results_df['growth_rate'] - results_df['growth_rate'].min()) / \
                       (results_df['growth_rate'].max() - results_df['growth_rate'].min() + 1e-9)
        
        # Normalize R² to 0-1 scale (already between 0-1, but emphasize high values)
        r2_normalized = results_df['r_squared']
        
        # Time penalty: prefer earlier times (exponential phase near start)
        # Use midpoint time, normalize and invert so early = high score
        max_time = results_df['midpoint_time'].max()
        time_normalized = 1 - (results_df['midpoint_time'] / (max_time + 1e-9))
        
        # Composite score: heavily weight growth rate, moderate weight to early timing and R²
        # Growth rate: 60%, R²: 25%, Early timing: 15%
        results_df['score'] = (gr_normalized * 0.60) + (r2_normalized * 0.25) + (time_normalized * 0.15)
        
        # Find the best window
        best_result = results_df.loc[results_df['score'].idxmax()]
        
        start_time = float(best_result['start_time'])
        end_time = float(best_result['end_time'])
        max_growth_rate = float(best_result['growth_rate'])
        best_r_squared = float(best_result['r_squared'])
        
        # Ensure start/end times are within the slider bounds
        start_time = max(self.region_start.start, start_time)
        end_time = min(self.region_end.end, end_time)

        # Update the region selection sliders and text inputs
        self.region_start.value = start_time
        self.region_start_input.value = start_time
        self.region_end.value = end_time
        self.region_end_input.value = end_time
        
        # Calculate doubling time for context
        doubling_time = np.log(2) / max_growth_rate if max_growth_rate > 0 else np.inf
        
        # Show detailed success message with max specific growth rate
        success_msg = (f"Auto-detected exponential phase for {unit}:\n"
                      f"• Time range: {start_time:.2f}h to {end_time:.2f}h ({end_time-start_time:.2f}h duration)\n"
                      f"• **Max Specific Growth Rate (μmax): {max_growth_rate:.4f} h⁻¹**\n"
                      f"• Doubling time: {doubling_time:.2f} h\n"
                      f"• R²: {best_r_squared:.4f}")
        
        self.show_success(success_msg)
        
        # Show additional debug info
        debug_msg = (f"Auto-detection details:\n"
                    f"• Evaluated {len(all_results)} potential windows\n"
                    f"• Window sizes tested: {min_window} to {max_window} points\n"
                    f"• Growth rate range found: {results_df['growth_rate'].min():.4f} to {results_df['growth_rate'].max():.4f} h⁻¹\n"
                    f"• Top 5 candidates by score:\n")
        
        top_5 = results_df.nlargest(5, 'score')[['start_time', 'end_time', 'growth_rate', 'r_squared', 'score']]
        for idx, row in top_5.iterrows():
            debug_msg += f"  - {row['start_time']:.2f}h-{row['end_time']:.2f}h: μ={row['growth_rate']:.4f}, R²={row['r_squared']:.4f}, score={row['score']:.3f}\n"
        
        self.show_debug(debug_msg)
        self.loading_spinner.value = False

    def _update_plots(self):
        """Update all plots with current data and parameters."""
        # Check if we have data
        if self.od_data_df.empty or len(self.selected_units) == 0:
            # Display empty placeholders
            self.od_plot.object = hv.Text(0, 0, 'No valid OD data found').opts(width=800, height=300)
            self.ln_od_plot.object = hv.Text(0, 0, 'No valid ln(OD) data found').opts(width=800, height=300)
            return
        
        # Create a shared x-axis range for linked plots
        shared_x_range = Range1d()
        
        # Filter data for selected units
        filtered_df = self.od_data_df[self.od_data_df['exp_unit'].isin(self.selected_units)]
        
        # Create processed data for each unit
        processed_data = {}
        for unit in self.selected_units:
            unit_data = filtered_df[filtered_df['exp_unit'] == unit].copy()
            unit_data = unit_data.sort_values('elapsed_hours')
            unit_data['od_smooth'] = unit_data['od_reading'].rolling(
                window=self.smoothing_window, min_periods=1, center=True).mean()
            unit_data['ln_od'] = np.log(unit_data['od_smooth'].clip(lower=1e-6))  # Avoid log(0)
            processed_data[unit] = unit_data
        
        # Create OD vs Time plot
        od_curves = []
        palette = self._get_color_palette()
        color_cycle = hv.Cycle(palette) if palette else hv.Cycle()
        
        for idx, unit in enumerate(self.selected_units):
            unit_data = processed_data[unit]
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
        
        # Set up plot options with customizable aesthetics
        plot_opts = {
            'title': 'OD vs Time (Semi-log)',
            'xlabel': 'Time (hours)',
            'ylabel': 'OD (smoothed, log scale)',
            'legend_position': 'top_right',
            'width': self.plot_width,
            'height': self.plot_height,
            'fontsize': {
                'title': self.font_size + 2,
                'labels': self.font_size,
                'xticks': self.font_size - 2,
                'yticks': self.font_size - 2,
                'legend': self.font_size - 1,
            }
        }
        
        # Always use log scale in backend options
        backend_opts = {
            'x_range': shared_x_range,
            'y_axis_type': 'log'  # Always use log scale
        }
        
        # Apply all options
        od_plot = hv.Overlay(od_curves).opts(**plot_opts, backend_opts=backend_opts)
        
        # Create ln(OD) vs Time plot
        ln_od_curves = []
        for idx, unit in enumerate(self.selected_units):
            unit_data = processed_data[unit]
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
        
        # Combine the curves with customizable aesthetics
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
            },
            backend_opts={'x_range': shared_x_range}
        )
        
        # Add selected region visualization
        if self.selected_region['start'] is not None and self.selected_region['end'] is not None:
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
                unit_data = processed_data[unit]
                # Filter for the selected region
                region_data = unit_data[
                    (unit_data['elapsed_hours'] >= self.selected_region['start']) &
                    (unit_data['elapsed_hours'] <= self.selected_region['end']) &
                    np.isfinite(unit_data['ln_od']) # Ensure ln_od is valid
                ]
                
                if len(region_data) >= 5:  # Need enough points for regression
                    # Perform linear regression on ln(OD)
                    x = region_data['elapsed_hours'].values
                    y = region_data['ln_od'].values
                    
                    # Check for sufficient variation in x values within the region
                    if np.ptp(x) < 1e-9:
                        self.show_debug(f"Skipping regression line for {unit}: Time values in region are identical.")
                        continue

                    try:
                        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                        
                        # Create regression line
                        x_range = np.array([self.selected_region['start'], self.selected_region['end']])
                        y_range = slope * x_range + intercept
                        
                        # Calculate 95% confidence interval
                        n = len(x)
                        t_val = stats.t.ppf(0.975, n - 2)  # 95% CI, two-tailed
                        
                        # Calculate standard error of the regression
                        residuals = y - (slope * x + intercept)
                        s_residuals = np.sqrt(np.sum(residuals**2) / (n - 2))
                        
                        # Standard error of the mean prediction
                        x_mean = np.mean(x)
                        sxx = np.sum((x - x_mean)**2)
                        
                        # Calculate CI for the regression line at the boundaries
                        se_fit_start = s_residuals * np.sqrt(1/n + (self.selected_region['start'] - x_mean)**2 / sxx)
                        se_fit_end = s_residuals * np.sqrt(1/n + (self.selected_region['end'] - x_mean)**2 / sxx)
                        
                        # Upper and lower bounds
                        y_start = slope * self.selected_region['start'] + intercept
                        y_end = slope * self.selected_region['end'] + intercept
                        
                        ci_lower_start = y_start - t_val * se_fit_start
                        ci_upper_start = y_start + t_val * se_fit_start
                        ci_lower_end = y_end - t_val * se_fit_end
                        ci_upper_end = y_end + t_val * se_fit_end
                        
                        # Create confidence band as a polygon
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
                        
                        # Add to ln(OD) plot
                        ln_od_plot = ln_od_plot * ci_band * regression_line
                    except ValueError as ve:
                        self.show_debug(f"Skipping regression line for {unit}: Linregress error - {ve}")
        
        # Add debug info 
        self.show_debug(f"Plot update complete. Semi-log mode: {self.semi_log_plot}, backend_opts: {backend_opts}")
        
        # Update the plot panes
        self.od_plot.object = od_plot
        self.ln_od_plot.object = ln_od_plot

    def _calculate_growth_rate(self, filename):
        """
        Calculate growth rate for the selected region and units.
        Returns a list of dictionaries containing results for each unit.
        """
        results_list = [] # Changed from results = []

        if self.od_data_df.empty or len(self.selected_units) == 0:
            self.show_error("No data loaded or no units selected for calculation.")
            return results_list # Return empty list
        
        # Get region boundaries
        start_time = self.selected_region['start']
        end_time = self.selected_region['end']
        
        if start_time is None or end_time is None or start_time >= end_time:
            self.show_error("Invalid region selected for analysis (start must be before end).")
            return results_list # Return empty list
            
        duration = end_time - start_time

        for unit in self.selected_units:
            # Get data for the selected unit and region
            unit_data = self.od_data_df[
                (self.od_data_df['exp_unit'] == unit) &
                (self.od_data_df['elapsed_hours'] >= start_time) &
                (self.od_data_df['elapsed_hours'] <= end_time)
            ].copy()
            
            # Default result structure
            result_entry = {
                'Filename': filename,
                'Unit': unit,
                'Region Start (h)': start_time,
                'Region End (h)': end_time,
                'Duration (h)': duration,
                'Growth Rate (h⁻¹)': np.nan, # Use NaN for easier averaging
                'Doubling Time (h)': np.nan,
                'R²': np.nan,
                'Std Error': np.nan,
                'CI Lower': np.nan,
                'CI Upper': np.nan,
                'P-value': np.nan,
                'Data Points': len(unit_data), # Store initial count
                'Apparent Yield (OD/g)': np.nan,
                'Yield Status': 'N/A',
                'Max OD': np.nan,
                'ΔOD': np.nan
            }

            if len(unit_data) < 5:  # Need at least 5 points for a reasonable fit
                results_list.append(result_entry)
                continue
            
            # Sort by time
            unit_data = unit_data.sort_values('elapsed_hours')
            
            # Apply smoothing
            unit_data['od_smooth'] = unit_data['od_reading'].rolling(
                window=self.smoothing_window, min_periods=1, center=True).mean()
            
            # Filter out OD values below threshold
            unit_data = unit_data[unit_data['od_smooth'] >= self.min_od_threshold]
            
            # Update data points count after filtering
            result_entry['Data Points'] = len(unit_data) 
            
            if len(unit_data) < 5:  # Check again after filtering
                results_list.append(result_entry)
                continue
            
            # Calculate ln(OD)
            unit_data['ln_od'] = np.log(unit_data['od_smooth'])
            
            # Filter out non-finite ln(OD) values
            unit_data = unit_data[np.isfinite(unit_data['ln_od'])]

            # Update data points count again
            result_entry['Data Points'] = len(unit_data) 

            if len(unit_data) < 5: # Check again after ln(OD) filtering
                results_list.append(result_entry)
                continue

            # Check for sufficient variation in time values
            if np.ptp(unit_data['elapsed_hours']) < 1e-9:
                self.show_debug(f"Skipping calculation for {unit}: Time values in region are identical.")
                results_list.append(result_entry)
                continue

            try:
                # Linear regression on ln(OD) vs time
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    unit_data['elapsed_hours'], unit_data['ln_od']
                )
                
                # Calculate growth rate (slope of ln(OD) vs time)
                growth_rate = slope  # in h^-1
                
                # Calculate doubling time
                doubling_time = np.log(2) / growth_rate if growth_rate > 1e-9 else np.nan # Avoid division by zero/small numbers
                
                # Calculate 95% confidence interval for growth rate
                n = len(unit_data)
                t_val = stats.t.ppf(0.975, n - 2)  # 95% CI, two-tailed
                ci_width = t_val * std_err
                ci_lower = growth_rate - ci_width
                ci_upper = growth_rate + ci_width
                
                # Calculate apparent yield
                yield_data = self._calculate_apparent_yield(unit)
                
                # Store results
                result_entry.update({
                    'Growth Rate (h⁻¹)': growth_rate,
                    'Doubling Time (h)': doubling_time,
                    'R²': r_value**2,
                    'Std Error': std_err,
                    'CI Lower': ci_lower,
                    'CI Upper': ci_upper,
                    'P-value': p_value,
                    'Apparent Yield (OD/g)': yield_data['yield'],
                    'Yield Status': yield_data['status'],
                    'Max OD': yield_data['max_od'],
                    'ΔOD': yield_data['delta_od']
                })
                results_list.append(result_entry)

            except ValueError as ve:
                self.show_debug(f"Linregress failed for {unit}: {ve}")
                results_list.append(result_entry) # Append with NaNs

        return results_list # Return the list of dictionaries

    def _calculate_apparent_yield(self, unit):
        """
        Calculate apparent yield (Yx/s) based on OD change and substrate consumption.
        
        Parameters
        ----------
        unit : str
            The experiment unit identifier
            
        Returns
        -------
        dict
            Dictionary containing:
            - yield: apparent yield in OD/g substrate
            - status: 'Complete' if OD plateaus/rolls off, 'At least' if still growing
            - max_od: maximum OD reached
            - delta_od: OD change from start to max
        """
        # Get all data for the unit
        unit_data = self.od_data_df[self.od_data_df['exp_unit'] == unit].copy()
        
        if unit_data.empty:
            return {
                'yield': np.nan,
                'status': 'N/A',
                'max_od': np.nan,
                'delta_od': np.nan
            }
        
        # Sort by time
        unit_data = unit_data.sort_values('elapsed_hours')
        
        # Apply smoothing
        unit_data['od_smooth'] = unit_data['od_reading'].rolling(
            window=self.smoothing_window, min_periods=1, center=True).mean()
        
        # Get initial OD (use median of first few points to be robust)
        initial_od = unit_data['od_smooth'].iloc[:5].median()
        
        # Get maximum OD
        max_od = unit_data['od_smooth'].max()
        max_od_idx = unit_data['od_smooth'].idxmax()
        max_od_time = unit_data.loc[max_od_idx, 'elapsed_hours']
        
        # Calculate OD change
        delta_od = max_od - initial_od
        
        # Determine if growth has plateaued or rolled off
        # Check if there's data after max OD
        data_after_max = unit_data[unit_data['elapsed_hours'] > max_od_time]
        
        # Get the last 10% of data points
        last_points_count = max(5, len(unit_data) // 10)
        last_points = unit_data.tail(last_points_count)
        
        # Check if OD is declining or plateauing
        if not data_after_max.empty and len(data_after_max) >= 3:
            # Calculate slope of last portion
            last_od_values = last_points['od_smooth'].values
            last_time_values = last_points['elapsed_hours'].values
            
            # Simple linear regression on last points
            if len(last_od_values) >= 3 and np.ptp(last_time_values) > 0:
                slope_last = np.polyfit(last_time_values, last_od_values, 1)[0]
                
                # Check if OD is decreasing or flat (slope near zero or negative)
                # Threshold: less than 1% of max growth rate
                relative_slope = abs(slope_last) / (max_od / max_od_time) if max_od_time > 0 else 0
                
                if slope_last < 0 or relative_slope < 0.01:
                    yield_status = 'Complete'
                else:
                    yield_status = 'At least'
            else:
                # Not enough data to determine
                yield_status = 'At least'
        else:
            # Max OD is at the end - still growing
            yield_status = 'At least'
        
        # Calculate apparent yield: ΔOD / substrate consumed
        # Apparent yield = (OD_max - OD_initial) / (S_initial * Volume)
        # Units: OD / g substrate
        
        if self.initial_substrate_conc > 0 and self.reactor_volume > 0:
            # Convert substrate concentration from g/L to g in reactor
            substrate_mass = self.initial_substrate_conc * (self.reactor_volume / 1000.0)  # g
            
            # Calculate yield
            apparent_yield = delta_od / substrate_mass  # OD per g substrate
        else:
            apparent_yield = np.nan
        
        return {
            'yield': apparent_yield,
            'status': yield_status,
            'max_od': max_od,
            'delta_od': delta_od
        }


    def _add_analysis_callback(self, event):
        """Calculates growth rate for current settings and adds to cumulative table."""
        if self.current_filename is None:
            self.show_error("Please upload a CSV file first.")
            return
            
        if not self.selected_units:
            self.show_error("Please select at least one unit.")
            return

        if self.selected_region['start'] is None or self.selected_region['end'] is None or self.selected_region['start'] >= self.selected_region['end']:
             self.show_error("Invalid region selected for analysis.")
             return

        # Calculate results for the current settings
        current_results_list = self._calculate_growth_rate(filename=self.current_filename)
        
        if not current_results_list:
            self.show_error("Calculation failed or produced no results for the current selection.")
            return

        # Convert list of dicts to DataFrame
        new_results_df = pd.DataFrame(current_results_list)
        
        # Append to the cumulative DataFrame
        self.cumulative_results_df = pd.concat([self.cumulative_results_df, new_results_df], ignore_index=True)
        
        # Update the display
        self._update_results_display()
        self.show_success(f"Added {len(new_results_df)} analysis row(s) to the results table.")

    def _export_results_callback(self, event):
        """Export cumulative results to CSV file."""
        if self.cumulative_results_df.empty:
            self.show_error("No results to export. Please add analysis results first.")
            return
        
        try:
            # Generate filename with timestamp
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"growth_rate_analysis_{timestamp}.csv"
            
            # Save to CSV
            self.cumulative_results_df.to_csv(filename, index=False)
            
            self.show_success(f"Results exported to {filename}")
        except Exception as e:
            self.show_error(f"Failed to export results: {str(e)}")

    def _clear_results_callback(self, event):
        """Clear all cumulative results after confirmation."""
        if self.cumulative_results_df.empty:
            self.show_warning("Results table is already empty.")
            return
        
        # Clear the dataframe
        self.cumulative_results_df = self.cumulative_results_df[0:0]  # Empty the dataframe
        
        # Update display
        self._update_results_display()
        self.show_success("All results cleared successfully.")

    def _export_od_plot_callback(self, event):
        """Export OD plot to PNG file."""
        if self.od_plot.object is None or str(self.od_plot.object) == 'Text':
            self.show_error("No OD plot to export. Please upload data first.")
            return
        
        try:
            from datetime import datetime
            import holoviews as hv
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"od_plot_{timestamp}.png"
            
            # Save plot using HoloViews save function with custom DPI
            hv.save(self.od_plot.object, filename, fmt='png', dpi=self.plot_dpi)
            
            self.show_success(f"OD plot exported to {filename} (DPI: {self.plot_dpi})")
        except Exception as e:
            self.show_error(f"Failed to export plot: {str(e)}")
            self.show_debug(f"Export error: {str(e)}")

    def _export_ln_od_plot_callback(self, event):
        """Export ln(OD) plot to PNG file."""
        if self.ln_od_plot.object is None or str(self.ln_od_plot.object) == 'Text':
            self.show_error("No ln(OD) plot to export. Please upload data first.")
            return
        
        try:
            from datetime import datetime
            import holoviews as hv
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ln_od_plot_{timestamp}.png"
            
            # Save plot using HoloViews save function with custom DPI
            hv.save(self.ln_od_plot.object, filename, fmt='png', dpi=self.plot_dpi)
            
            self.show_success(f"ln(OD) plot exported to {filename} (DPI: {self.plot_dpi})")
        except Exception as e:
            self.show_error(f"Failed to export plot: {str(e)}")
            self.show_debug(f"Export error: {str(e)}")

    def _export_all_callback(self, event):
        """Export all plots and data to a timestamped folder."""
        import os
        from datetime import datetime
        
        try:
            # Create timestamped export directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_dir = f"batch_analysis_export_{timestamp}"
            os.makedirs(export_dir, exist_ok=True)
            
            export_count = 0
            
            # Export OD plot if available
            if self.od_plot.object is not None and str(self.od_plot.object) != 'Text':
                try:
                    import holoviews as hv
                    od_filename = os.path.join(export_dir, "od_plot.png")
                    hv.save(self.od_plot.object, od_filename, fmt='png', dpi=self.plot_dpi)
                    export_count += 1
                except Exception as e:
                    self.show_debug(f"OD plot export failed: {str(e)}")
            
            # Export ln(OD) plot if available
            if self.ln_od_plot.object is not None and str(self.ln_od_plot.object) != 'Text':
                try:
                    import holoviews as hv
                    ln_od_filename = os.path.join(export_dir, "ln_od_plot.png")
                    hv.save(self.ln_od_plot.object, ln_od_filename, fmt='png', dpi=self.plot_dpi)
                    export_count += 1
                except Exception as e:
                    self.show_debug(f"ln(OD) plot export failed: {str(e)}")
            
            # Export results CSV if available
            if not self.cumulative_results_df.empty:
                csv_filename = os.path.join(export_dir, "growth_rate_results.csv")
                self.cumulative_results_df.to_csv(csv_filename, index=False)
                export_count += 1
            
            # Export raw data if available
            if self.data is not None and not self.data.empty:
                raw_data_filename = os.path.join(export_dir, "raw_data.csv")
                self.data.to_csv(raw_data_filename, index=False)
                export_count += 1
            
            if export_count > 0:
                self.show_success(f"Exported {export_count} items to folder: {export_dir}")
            else:
                self.show_error("No data available to export. Please upload data first.")
                
        except Exception as e:
            self.show_error(f"Failed to export all: {str(e)}")
            self.show_debug(f"Export all error: {str(e)}")

    def _update_results_display(self):
        """Generates and updates the HTML for the cumulative results table."""
        
        if self.cumulative_results_df.empty:
            self.results_output.object = "<p>No analysis results added yet. Select units/region and click 'Add Current Analysis to Table'.</p>"
            return

        # Create HTML table header
        html = "<h3>Cumulative Growth Rate Analysis Results</h3>"
        
        # Add max growth rate summary at the top
        max_growth_rate = self.cumulative_results_df['Growth Rate (h⁻¹)'].max()
        max_growth_idx = self.cumulative_results_df['Growth Rate (h⁻¹)'].idxmax()
        max_growth_row = self.cumulative_results_df.loc[max_growth_idx]
        max_doubling_time = np.log(2) / max_growth_rate if max_growth_rate > 0 else np.inf
        
        html += f"""
        <div style="margin-bottom: 20px; padding: 15px; background-color: #e8f4f8; border-left: 4px solid #2196F3; border-radius: 4px;">
          <h4 style="margin-top: 0; color: #1976D2;">📈 Maximum Specific Growth Rate (μmax)</h4>
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
              <th style="padding:8px; border:1px solid #ddd; text-align:right;">Region Start (h)</th>
              <th style="padding:8px; border:1px solid #ddd; text-align:right;">Region End (h)</th>
              <th style="padding:8px; border:1px solid #ddd; text-align:right;">Duration (h)</th>
              <th style="padding:8px; border:1px solid #ddd; text-align:right;">Growth Rate (h⁻¹)</th>
              <th style="padding:8px; border:1px solid #ddd; text-align:right;">95% CI</th>
              <th style="padding:8px; border:1px solid #ddd; text-align:right;">Doubling Time (h)</th>
              <th style="padding:8px; border:1px solid #ddd; text-align:right;">R²</th>
              <th style="padding:8px; border:1px solid #ddd; text-align:right;">P-value</th>
              <th style="padding:8px; border:1px solid #ddd; text-align:right;">Data Points</th>
              <th style="padding:8px; border:1px solid #ddd; text-align:right;">Yield (OD/g)</th>
              <th style="padding:8px; border:1px solid #ddd; text-align:left;">Yield Status</th>
              <th style="padding:8px; border:1px solid #ddd; text-align:right;">Max OD</th>
              <th style="padding:8px; border:1px solid #ddd; text-align:right;">ΔOD</th>
            </tr>
          </thead>
          <tbody>
        """
        
        # Add table rows from the DataFrame
        for idx, row in self.cumulative_results_df.iterrows():
            # Highlight the max growth rate row
            row_style = "background-color: #fff9c4;" if idx == max_growth_idx else ""
            
            # Format yield display based on status
            if pd.notna(row['Apparent Yield (OD/g)']):
                yield_display = f"{row['Apparent Yield (OD/g)']:.4f}"
            else:
                yield_display = "N/A"
            
            # Color code yield status
            yield_status = row['Yield Status']
            if yield_status == 'Complete':
                status_color = '#4CAF50'  # Green
                status_display = '✓ Complete'
            elif yield_status == 'At least':
                status_color = '#FF9800'  # Orange
                status_display = '≥ At least'
            else:
                status_color = '#999'
                status_display = 'N/A'
            
            # Format CI display
            if pd.notna(row['CI Lower']) and pd.notna(row['CI Upper']):
                ci_display = f"±{(row['CI Upper'] - row['CI Lower'])/2:.4f}"
            else:
                ci_display = "N/A"
            
            # Format P-value
            if pd.notna(row['P-value']):
                p_display = f"{row['P-value']:.2e}" if row['P-value'] < 0.001 else f"{row['P-value']:.4f}"
            else:
                p_display = "N/A"
            
            html += f"""
            <tr style="{row_style}">
              <td style="padding:8px; border:1px solid #ddd;">{row['Filename']}</td>
              <td style="padding:8px; border:1px solid #ddd;">{row['Unit']}</td>
              <td style="padding:8px; border:1px solid #ddd; text-align:right;">{row['Region Start (h)']:.2f}</td>
              <td style="padding:8px; border:1px solid #ddd; text-align:right;">{row['Region End (h)']:.2f}</td>
              <td style="padding:8px; border:1px solid #ddd; text-align:right;">{row['Duration (h)']:.2f}</td>
              <td style="padding:8px; border:1px solid #ddd; text-align:right;">{row['Growth Rate (h⁻¹)']:.4f}</td>
              <td style="padding:8px; border:1px solid #ddd; text-align:right;">{ci_display}</td>
              <td style="padding:8px; border:1px solid #ddd; text-align:right;">{row['Doubling Time (h)']:.2f}</td>
              <td style="padding:8px; border:1px solid #ddd; text-align:right;">{row['R²']:.4f}</td>
              <td style="padding:8px; border:1px solid #ddd; text-align:right;">{p_display}</td>
              <td style="padding:8px; border:1px solid #ddd; text-align:right;">{int(row['Data Points'])}</td>
              <td style="padding:8px; border:1px solid #ddd; text-align:right;">{yield_display}</td>
              <td style="padding:8px; border:1px solid #ddd; color:{status_color}; font-weight:bold;">{status_display}</td>
              <td style="padding:8px; border:1px solid #ddd; text-align:right;">{row['Max OD']:.4f if pd.notna(row['Max OD']) else 'N/A'}</td>
              <td style="padding:8px; border:1px solid #ddd; text-align:right;">{row['ΔOD']:.4f if pd.notna(row['ΔOD']) else 'N/A'}</td>
            </tr>
            """
            
        # Calculate Averages (ignoring NaNs)
        numeric_cols = ['Growth Rate (h⁻¹)', 'Doubling Time (h)', 'R²', 'Data Points', 
                       'Apparent Yield (OD/g)', 'Max OD', 'ΔOD']
        averages = self.cumulative_results_df[numeric_cols].mean(skipna=True)

        # Add Average Row
        html += f"""
            <tr style="background-color:#f2f2f2; font-weight:bold;">
              <td style="padding:8px; border:1px solid #ddd;" colspan="5">Average</td>
              <td style="padding:8px; border:1px solid #ddd; text-align:right;">{averages['Growth Rate (h⁻¹)']:.4f}</td>
              <td style="padding:8px; border:1px solid #ddd; text-align:right;">N/A</td>
              <td style="padding:8px; border:1px solid #ddd; text-align:right;">{averages['Doubling Time (h)']:.2f}</td>
              <td style="padding:8px; border:1px solid #ddd; text-align:right;">{averages['R²']:.4f}</td>
              <td style="padding:8px; border:1px solid #ddd; text-align:right;">N/A</td>
              <td style="padding:8px; border:1px solid #ddd; text-align:right;">{averages['Data Points']:.1f}</td>
              <td style="padding:8px; border:1px solid #ddd; text-align:right;">{averages['Apparent Yield (OD/g)']:.4f}</td>
              <td style="padding:8px; border:1px solid #ddd;" colspan="3">—</td>
            </tr>
        """

        html += """
          </tbody>
        </table>
        """
        
        # Add explanation of results
        html += """
        <div style="margin-top: 20px; padding: 10px; background-color: #f9f9f9; border-left: 4px solid #5A7B9C;">
          <h4>Explanation of Results:</h4>
          <ul>
            <li><strong>Growth Rate (μ):</strong> The exponential growth rate constant in h⁻¹. The maximum value represents μmax.</li>
            <li><strong>Doubling Time:</strong> Time required for the cell population to double (ln(2)/μ)</li>
            <li><strong>R²:</strong> Coefficient of determination, measures goodness of fit (1.0 = perfect fit)</li>
            <li><strong>Std Error:</strong> Standard error of the growth rate estimate</li>
            <li><strong>Data Points:</strong> Number of data points used in the regression after filtering</li>
            <li><strong>Apparent Yield (Yx/s):</strong> Biomass produced per gram of substrate consumed, measured as ΔOD/g substrate</li>
            <li><strong>Yield Status:</strong> 
              <ul>
                <li><span style="color:#4CAF50;">✓ Complete:</span> OD plateaued or declined - reliable yield estimate (substrate depleted)</li>
                <li><span style="color:#FF9800;">≥ At least:</span> OD still increasing at end - minimum yield estimate (substrate not fully consumed)</li>
              </ul>
            </li>
            <li><strong>Max OD:</strong> Maximum OD value reached during the batch</li>
            <li><strong>ΔOD:</strong> Change in OD from start to maximum (Max OD - Initial OD)</li>
          </ul>
          <p><strong>Note on Units:</strong></p>
          <ul>
            <li>Growth rate (μ) is in h⁻¹ based on OD measurements</li>
            <li>Apparent yield is in OD per gram of substrate (OD/g)</li>
            <li>To convert to cell-specific values, you need:</li>
            <ul>
              <li>OD to cell concentration conversion (cells/mL per OD unit)</li>
              <li>OD to dry cell weight (DCW) conversion (g DCW/L per OD unit)</li>
            </ul>
            <li>Example conversion: If 1 OD = 0.4 g DCW/L, then Yx/s in g DCW/g substrate = Apparent Yield × 0.4</li>
          </ul>
        </div>
        """
        
        
        self.results_output.object = html

    def _validate_data_quality(self):
        """
        Validate data quality and show warnings for potential issues.
        """
        if self.od_data_df.empty:
            return
        
        warnings = []
        
        # Check for negative OD values
        if 'od_reading' in self.od_data_df.columns:
            negative_od_count = (self.od_data_df['od_reading'] < 0).sum()
            if negative_od_count > 0:
                warnings.append(f"⚠️ Found {negative_od_count} negative OD values (will be filtered)")
        
            # Check for unrealistic high OD values
            very_high_od = (self.od_data_df['od_reading'] > 10).sum()
            if very_high_od > 0:
                warnings.append(f"⚠️ Found {very_high_od} OD values > 10 (check if realistic)")
            
            # Check data frequency
            for unit in self.od_data_df['exp_unit'].unique():
                unit_data = self.od_data_df[self.od_data_df['exp_unit'] == unit].sort_values('elapsed_hours')
                if len(unit_data) > 1:
                    time_diffs = unit_data['elapsed_hours'].diff().dropna()
                    median_interval = time_diffs.median()
                    max_gap = time_diffs.max()
                    
                    # Warn if there are large gaps in data
                    if max_gap > median_interval * 5:
                        warnings.append(f"⚠️ Unit {unit}: Large time gap detected ({max_gap:.1f}h vs median {median_interval:.1f}h)")
                    
                    # Warn if very few data points
                    if len(unit_data) < 10:
                        warnings.append(f"⚠️ Unit {unit}: Only {len(unit_data)} data points (may affect analysis quality)")
        
        # Display warnings if any
        if warnings:
            warning_msg = "**Data Quality Warnings:**\n" + "\n".join(warnings)
            self.show_warning(warning_msg)
    
    def _update_data_preview(self):
        """
        Create and display a preview of the uploaded data.
        """
        if self.od_data_df.empty:
            self.data_preview.object = ""
            return
        
        # Generate data summary
        total_rows = len(self.od_data_df)
        experiments = self.od_data_df['experiment'].nunique()
        units = self.od_data_df['pioreactor_unit'].nunique()
        
        # Time range
        if 'elapsed_hours' in self.od_data_df.columns:
            time_min = self.od_data_df['elapsed_hours'].min()
            time_max = self.od_data_df['elapsed_hours'].max()
            time_range = f"{time_min:.2f}h - {time_max:.2f}h ({time_max - time_min:.2f}h duration)"
        else:
            time_range = "N/A"
        
        # OD range
        if 'od_reading' in self.od_data_df.columns:
            od_min = self.od_data_df['od_reading'].min()
            od_max = self.od_data_df['od_reading'].max()
            od_mean = self.od_data_df['od_reading'].mean()
            od_range = f"{od_min:.4f} - {od_max:.4f} (mean: {od_mean:.4f})"
        else:
            od_range = "N/A"
        
        # Column check
        required_cols = ['experiment', 'pioreactor_unit', 'od_reading']
        missing_cols = [col for col in required_cols if col not in self.od_data_df.columns]
        optional_cols = ['timestamp', 'timestamp_localtime']
        present_optional = [col for col in optional_cols if col in self.od_data_df.columns]
        
        # Build HTML preview
        html = f"""
        <div style="border: 1px solid #ddd; border-radius: 5px; padding: 15px; margin: 10px 0; background-color: #f9f9f9;">
            <h3 style="margin-top:0; color: #333;">📊 Data Preview</h3>
            
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
            
            <div style="margin-bottom: 15px;">
                <strong>Columns:</strong>
                <ul style="margin: 5px 0;">
                    <li>Required: {', '.join(required_cols)} {'✅' if not missing_cols else f'❌ Missing: {", ".join(missing_cols)}'}</li>
                    <li>Optional Present: {', '.join(present_optional) if present_optional else 'None'}</li>
                    <li>All Columns ({len(self.od_data_df.columns)}): {', '.join(self.od_data_df.columns)}</li>
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
    
    def save_session_callback(self, event=None):
        """Save the current session state to a JSON file."""
        import json
        from datetime import datetime
        
        try:
            # Create session state dictionary
            session_state = {
                'timestamp': datetime.now().isoformat(),
                'parameters': {
                    'smoothing_window': self.smoothing_window,
                    'min_od_threshold': self.min_od_threshold,
                    'semi_log_plot': self.semi_log_plot,
                    'reactor_volume': self.reactor_volume,
                    'initial_substrate_conc': self.initial_substrate_conc,
                },
                'data': {
                    'od_data': self.od_data_df.to_json(orient='split', date_format='iso') if not self.od_data_df.empty else None,
                    'results': self.cumulative_results_df.to_json(orient='split', date_format='iso') if not self.cumulative_results_df.empty else None,
                },
                'ui_state': {
                    'selected_units': self.selected_units,
                    'uploaded_files': self.uploaded_files,
                }
            }
            
            # Save to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'session_{timestamp}.json'
            
            with open(filename, 'w') as f:
                json.dump(session_state, f, indent=2)
            
            self.status_text.object = f"✅ **Session saved successfully to {filename}**"
            
        except Exception as e:
            self.status_text.object = f"❌ **Error saving session:** {str(e)}"
            print(f"Save error: {traceback.format_exc()}")
    
    def load_session_callback(self, event):
        """Load session state from a JSON file."""
        import json
        
        if not event.new:
            return
        
        self.loading_indicator.value = True
        
        try:
            # Read the uploaded file
            file_obj = event.new[0] if isinstance(event.new, list) else event.new
            content = file_obj.decode('utf-8') if isinstance(file_obj, bytes) else file_obj
            
            session_state = json.loads(content)
            
            # Restore parameters
            params = session_state.get('parameters', {})
            self.smoothing_window = params.get('smoothing_window', 5)
            self.min_od_threshold = params.get('min_od_threshold', 0.05)
            self.semi_log_plot = params.get('semi_log_plot', True)
            self.reactor_volume = params.get('reactor_volume', 14.0)
            self.initial_substrate_conc = params.get('initial_substrate_conc', 20.0)
            
            # Restore data
            data = session_state.get('data', {})
            if data.get('od_data'):
                self.od_data_df = pd.read_json(io.StringIO(data['od_data']), orient='split')
            else:
                self.od_data_df = pd.DataFrame()
            
            if data.get('results'):
                self.cumulative_results_df = pd.read_json(io.StringIO(data['results']), orient='split')
            else:
                self.cumulative_results_df = pd.DataFrame(columns=[
                    'Filename', 'Unit', 'Region Start (h)', 'Region End (h)', 'Duration (h)',
                    'Growth Rate (h⁻¹)', 'Std Error', 'CI Lower', 'CI Upper', 'P-value', 'Data Points',
                    'Doubling Time (h)', 'R²', 'Apparent Yield (OD/g)', 'Yield Status', 'Max OD', 'ΔOD'
                ])
            
            # Restore UI state
            ui_state = session_state.get('ui_state', {})
            self.selected_units = ui_state.get('selected_units', [])
            self.uploaded_files = ui_state.get('uploaded_files', {})
            
            # Update UI components
            if not self.od_data_df.empty:
                units = sorted(self.od_data_df['exp_unit'].unique())
                self.unit_selector.options = units
                if self.selected_units:
                    self.unit_selector.value = [u for u in self.selected_units if u in units]
                
                # Update time range
                min_time = self.od_data_df['elapsed_hours'].min()
                max_time = self.od_data_df['elapsed_hours'].max()
                self.region_start_slider.start = float(min_time)
                self.region_start_slider.end = float(max_time)
                self.region_end_slider.start = float(min_time)
                self.region_end_slider.end = float(max_time)
                
                # Update plots
                self._update_plots()
                
                # Update preview
                self._update_data_preview()
                
            # Update results display
            if not self.cumulative_results_df.empty:
                self._update_results_display()
            
            timestamp = session_state.get('timestamp', 'unknown')
            self.status_text.object = f"✅ **Session loaded successfully (saved: {timestamp})**"
            
        except Exception as e:
            self.status_text.object = f"❌ **Error loading session:** {str(e)}"
            print(f"Load error: {traceback.format_exc()}")
        
        finally:
            self.loading_indicator.value = False
    
    def view(self):
        """Return the main layout for display."""
        return self.main_layout# Create the application
growth_analysis = GrowthRateAnalysis()

# Create a Panel server
app = pn.panel(growth_analysis.view())

# Server
if __name__ == '__main__':
    # Ensure the server uses the __file__ directory if run as script
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__))) 
    app.show(port=5006)