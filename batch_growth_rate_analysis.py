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
    """
    
    smoothing_window = param.Integer(5, bounds=(1, 50), step=1, 
                                    doc="Smoothing window size (# of readings)")
    min_od_threshold = param.Number(0.05, bounds=(0.001, 1.0), step=0.01, 
                                   doc="Minimum OD threshold for analysis")
    
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
            'R²', 'Std Error', 'Data Points'
        ])
        
        # Status message for notifications
        self.status_message = pn.pane.Markdown("", styles={'color': 'blue'})
        self.error_message = pn.pane.Markdown("", styles={'color': 'red'})
        self.debug_message = pn.pane.Markdown("", styles={'color': 'green', 'font-family': 'monospace', 'font-size': '12px'})
        
        # Set up the interface
        self.file_input = pn.widgets.FileInput(accept='.csv', multiple=False)
        self.file_input.param.watch(self._upload_file_callback, 'value')
        
        self.update_button = pn.widgets.Button(name='Update Plots', button_type='primary')
        self.update_button.on_click(self._update_plots_callback) # Changed callback
        
        # Unit selection for multi-unit experiments
        self.unit_selector = pn.widgets.MultiSelect(name='Select Units', options=[])
        self.unit_selector.param.watch(self._units_changed, 'value')
        
        # Create plot panes
        self.od_plot = pn.pane.HoloViews(sizing_mode='stretch_width', height=300)
        self.ln_od_plot = pn.pane.HoloViews(sizing_mode='stretch_width', height=300)
        
        # Selected region widgets
        self.region_start = pn.widgets.FloatSlider(name='Region Start (hours)', start=0, end=100, step=0.1, value=0)
        self.region_end = pn.widgets.FloatSlider(name='Region End (hours)', start=0, end=100, step=0.1, value=100)
        self.region_start.param.watch(self._update_region, 'value')
        self.region_end.param.watch(self._update_region, 'value')
        
        # Auto-detect button
        self.auto_detect_button = pn.widgets.Button(name='Auto-detect Exponential Phase', button_type='success')
        self.auto_detect_button.on_click(self._auto_detect_callback)
        
        # Results output (will display the cumulative table)
        self.results_output = pn.pane.HTML("")
        
        # Button to add current analysis to the results table
        self.add_analysis_button = pn.widgets.Button(name='Add Current Analysis to Table', button_type='primary')
        self.add_analysis_button.on_click(self._add_analysis_callback)
        
        # Main layout
        self.main_layout = pn.Column(
            pn.pane.Markdown("# Growth Rate Analysis for Batch Culture"),
            pn.Row(
                pn.Column(
                    pn.pane.Markdown("### Settings"),
                    self.param.smoothing_window,
                    self.param.min_od_threshold,
                    self.update_button, # Renamed from 'Update' to 'Update Plots'
                    width=400
                ),
                pn.Column(
                    pn.pane.Markdown("### Upload Data"),
                    self.file_input,
                    self.status_message,
                    self.error_message,
                    width=400
                )
            ),
            pn.Row(
                pn.Column(
                    pn.pane.Markdown("### Unit Selection"),
                    self.unit_selector,
                    width=300
                ),
                pn.Column(
                    pn.pane.Markdown("### Region Selection"),
                    self.region_start,
                    self.region_end,
                    self.auto_detect_button,
                    width=400
                )
            ),
            pn.Tabs(
                ('OD vs Time', pn.Column(self.od_plot)),
                ('ln(OD) vs Time', pn.Column(self.ln_od_plot)),
                ('Results', pn.Column(
                    self.add_analysis_button, # Add button here
                    self.results_output
                )),
                ('Debug', self.debug_message)
            )
        )
        # Initial update of the (empty) results display
        self._update_results_display()

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

    def _upload_file_callback(self, event):
        """
        Handle file upload events from the UI.
        
        Reads the uploaded CSV file, processes the data, and updates the visualization.
        Resets the cumulative results if desired (currently keeps accumulating).
        """
        if self.file_input.value is not None and self.file_input.filename.endswith('.csv'):
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
        else:
            self.show_error("Please upload a CSV file.")
            self.current_filename = None

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
            
            self.region_start.start = min_time
            self.region_start.end = max_time
            self.region_start.value = min_time
            
            self.region_end.start = min_time
            self.region_end.end = max_time
            self.region_end.value = max_time
            
            self.selected_region = {'start': min_time, 'end': max_time}
        else:
            self.show_error("No valid data found in the CSV file.")
            self.od_data_df = pd.DataFrame() # Ensure it's empty

    def _units_changed(self, event):
        """Handle unit selection changes."""
        self.selected_units = self.unit_selector.value
        self._update_plots() # Only update plots, calculation is separate

    def _update_region(self, event):
        """Handle region slider updates."""
        # Ensure start <= end
        if self.region_start.value > self.region_end.value:
            if event.name == 'value' and event.obj is self.region_start:
                self.region_start.value = self.region_end.value
            else:
                self.region_end.value = self.region_start.value
        
        self.selected_region = {
            'start': self.region_start.value,
            'end': self.region_end.value
        }
        
        # Update plots with selected region
        self._update_plots() # Only update plots, calculation is separate

    def _update_plots_callback(self, event):
        """Handle update button click events - only updates plots."""
        self._update_plots()
        self.show_success("Plots updated with current settings.")

    def _auto_detect_callback(self, event):
        """Automatically detect the exponential growth phase."""
        if self.od_data_df.empty or len(self.selected_units) == 0:
            self.show_error("No data available for auto-detection.")
            return
        
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
            return
        
        # Calculate ln(OD)
        unit_data['ln_od'] = np.log(unit_data['od_smooth'])
        
        # Calculate local growth rate using rolling window
        window_size = min(20, len(unit_data) // 4)  # Dynamic window size based on data points
        growth_rates = []
        r_squared_values = []
        midpoints = []
        
        for i in range(len(unit_data) - window_size):
            window = unit_data.iloc[i:i+window_size]
            try:
                # Check for sufficient variation in x values
                if np.ptp(window['elapsed_hours']) < 1e-9: 
                    continue # Skip if time values are identical
                
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    window['elapsed_hours'], window['ln_od']
                )
                growth_rates.append(slope)
                r_squared_values.append(r_value**2)
                midpoints.append(window['elapsed_hours'].iloc[window_size//2])
            except ValueError as ve:
                self.show_debug(f"Linregress error in window {i}: {ve}")
                continue # Skip this window if linregress fails
        
        if not growth_rates:
            self.show_error("Could not calculate local growth rates (check data variability).")
            return
        
        # Convert to numpy arrays for easier processing
        growth_rates = np.array(growth_rates)
        r_squared_values = np.array(r_squared_values)
        midpoints = np.array(midpoints)
        
        # Filter out potential NaNs or Infs from calculations
        valid_indices = np.isfinite(growth_rates) & np.isfinite(r_squared_values)
        if not np.any(valid_indices):
            self.show_error("No valid growth rates found after filtering.")
            return
            
        growth_rates = growth_rates[valid_indices]
        r_squared_values = r_squared_values[valid_indices]
        midpoints = midpoints[valid_indices]

        # Find regions with high growth rate and good linear fit
        # Penalize very low R^2 values more heavily
        score = growth_rates * (r_squared_values ** 2) # Emphasize fit quality
        
        # Find the region with the highest score
        best_idx = np.argmax(score)
        
        # Define a region around this best point
        # Use the start/end times of the window that produced the best score
        best_window_start_index = np.where(midpoints == midpoints[best_idx])[0][0] # Find original index
        best_window_data = unit_data.iloc[best_window_start_index : best_window_start_index + window_size]
        
        start_time = best_window_data['elapsed_hours'].iloc[0]
        end_time = best_window_data['elapsed_hours'].iloc[-1]

        # Ensure start/end times are within the slider bounds
        start_time = max(self.region_start.start, start_time)
        end_time = min(self.region_end.end, end_time)

        # Update the region selection sliders (triggers _update_region -> _update_plots)
        self.region_start.value = start_time
        self.region_end.value = end_time
        
        # Show success message
        self.show_success(f"Auto-detected exponential phase for {unit} between {start_time:.2f}h and {end_time:.2f}h. Plots updated.")

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
        for unit in self.selected_units:
            unit_data = processed_data[unit]
            curve = hv.Curve(
                unit_data, 'elapsed_hours', 'od_smooth',
                label=unit
            ).opts(
                line_width=2,
                tools=['hover', 'tap']
            )
            od_curves.append(curve)
        
        # Combine the curves
        od_plot = hv.Overlay(od_curves).opts(
            title='OD vs Time',
            xlabel='Time (hours)',
            ylabel='OD (smoothed)',
            legend_position='top_right',
            width=800,
            height=300,
            backend_opts={'x_range': shared_x_range}
        )
        
        # Create ln(OD) vs Time plot
        ln_od_curves = []
        for unit in self.selected_units:
            unit_data = processed_data[unit]
            # Filter out invalid ln values
            unit_data = unit_data[np.isfinite(unit_data['ln_od'])]
            curve = hv.Curve(
                unit_data, 'elapsed_hours', 'ln_od',
                label=unit
            ).opts(
                line_width=2,
                tools=['hover', 'tap']
            )
            ln_od_curves.append(curve)
        
        # Combine the curves
        ln_od_plot = hv.Overlay(ln_od_curves).opts(
            title='ln(OD) vs Time',
            xlabel='Time (hours)',
            ylabel='ln(OD)',
            legend_position='top_right',
            width=800,
            height=300,
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
                        
                        regression_line = hv.Curve(
                            np.column_stack([x_range, y_range]),
                            label=f"{unit} fit (μ={slope:.3f}h⁻¹)"
                        ).opts(
                            color='green',
                            line_width=2,
                            line_dash='dashed'
                        )
                        
                        # Add to ln(OD) plot
                        ln_od_plot = ln_od_plot * regression_line
                    except ValueError as ve:
                         self.show_debug(f"Skipping regression line for {unit}: Linregress error - {ve}")

        
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
                'Data Points': len(unit_data) # Store initial count
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
                
                # Store results
                result_entry.update({
                    'Growth Rate (h⁻¹)': growth_rate,
                    'Doubling Time (h)': doubling_time,
                    'R²': r_value**2,
                    'Std Error': std_err,
                })
                results_list.append(result_entry)

            except ValueError as ve:
                self.show_debug(f"Linregress failed for {unit}: {ve}")
                results_list.append(result_entry) # Append with NaNs

        return results_list # Return the list of dictionaries

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

    def _update_results_display(self):
        """Generates and updates the HTML for the cumulative results table."""
        
        if self.cumulative_results_df.empty:
            self.results_output.object = "<p>No analysis results added yet. Select units/region and click 'Add Current Analysis to Table'.</p>"
            return

        # Create HTML table header
        html = "<h3>Cumulative Growth Rate Analysis Results</h3>"
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
              <th style="padding:8px; border:1px solid #ddd; text-align:right;">Doubling Time (h)</th>
              <th style="padding:8px; border:1px solid #ddd; text-align:right;">R²</th>
              <th style="padding:8px; border:1px solid #ddd; text-align:right;">Std Error</th>
              <th style="padding:8px; border:1px solid #ddd; text-align:right;">Data Points</th>
            </tr>
          </thead>
          <tbody>
        """
        
        # Add table rows from the DataFrame
        for _, row in self.cumulative_results_df.iterrows():
            html += f"""
            <tr>
              <td style="padding:8px; border:1px solid #ddd;">{row['Filename']}</td>
              <td style="padding:8px; border:1px solid #ddd;">{row['Unit']}</td>
              <td style="padding:8px; border:1px solid #ddd; text-align:right;">{row['Region Start (h)']:.2f}</td>
              <td style="padding:8px; border:1px solid #ddd; text-align:right;">{row['Region End (h)']:.2f}</td>
              <td style="padding:8px; border:1px solid #ddd; text-align:right;">{row['Duration (h)']:.2f}</td>
              <td style="padding:8px; border:1px solid #ddd; text-align:right;">{row['Growth Rate (h⁻¹)']:.4f}</td>
              <td style="padding:8px; border:1px solid #ddd; text-align:right;">{row['Doubling Time (h)']:.2f}</td>
              <td style="padding:8px; border:1px solid #ddd; text-align:right;">{row['R²']:.4f}</td>
              <td style="padding:8px; border:1px solid #ddd; text-align:right;">{row['Std Error']:.4f}</td>
              <td style="padding:8px; border:1px solid #ddd; text-align:right;">{int(row['Data Points'])}</td>
            </tr>
            """
            
        # Calculate Averages (ignoring NaNs)
        numeric_cols = ['Growth Rate (h⁻¹)', 'Doubling Time (h)', 'R²', 'Std Error', 'Data Points']
        averages = self.cumulative_results_df[numeric_cols].mean(skipna=True)

        # Add Average Row
        html += f"""
            <tr style="background-color:#f2f2f2; font-weight:bold;">
              <td style="padding:8px; border:1px solid #ddd;" colspan="5">Average</td>
              <td style="padding:8px; border:1px solid #ddd; text-align:right;">{averages['Growth Rate (h⁻¹)']:.4f}</td>
              <td style="padding:8px; border:1px solid #ddd; text-align:right;">{averages['Doubling Time (h)']:.2f}</td>
              <td style="padding:8px; border:1px solid #ddd; text-align:right;">{averages['R²']:.4f}</td>
              <td style="padding:8px; border:1px solid #ddd; text-align:right;">{averages['Std Error']:.4f}</td>
              <td style="padding:8px; border:1px solid #ddd; text-align:right;">{averages['Data Points']:.1f}</td>
            </tr>
        """

        html += """
          </tbody>
        </table>
        """
        
        # Add explanation of results (same as before)
        html += """
        <div style="margin-top: 20px; padding: 10px; background-color: #f9f9f9; border-left: 4px solid #5A7B9C;">
          <h4>Explanation of Results:</h4>
          <ul>
            <li><strong>Growth Rate (μ):</strong> The exponential growth rate constant in h⁻¹</li>
            <li><strong>Doubling Time:</strong> Time required for the cell population to double (ln(2)/μ)</li>
            <li><strong>R²:</strong> Coefficient of determination, measures goodness of fit (1.0 = perfect fit)</li>
            <li><strong>Std Error:</strong> Standard error of the growth rate estimate</li>
            <li><strong>Data Points:</strong> Number of data points used in the regression after filtering</li>
          </ul>
        </div>
        """
        
        self.results_output.object = html

    def view(self):
        """Return the main layout for display."""
        return self.main_layout

# Create the application
growth_analysis = GrowthRateAnalysis()

# Create a Panel server
app = pn.panel(growth_analysis.view())

# Server
if __name__ == '__main__':
    # Ensure the server uses the __file__ directory if run as script
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__))) 
    app.show(port=5006)