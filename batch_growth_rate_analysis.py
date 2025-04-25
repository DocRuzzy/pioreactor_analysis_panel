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
        self.current_results = {}
        
        # Status message for notifications
        self.status_message = pn.pane.Markdown("", styles={'color': 'blue'})
        self.error_message = pn.pane.Markdown("", styles={'color': 'red'})
        self.debug_message = pn.pane.Markdown("", styles={'color': 'green', 'font-family': 'monospace', 'font-size': '12px'})
        
        # Set up the interface
        self.file_input = pn.widgets.FileInput(accept='.csv', multiple=False)
        self.file_input.param.watch(self._upload_file_callback, 'value')
        
        self.update_button = pn.widgets.Button(name='Update', button_type='primary')
        self.update_button.on_click(self._update_callback)
        
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
        
        # Results output
        self.results_output = pn.pane.HTML("")
        
        # Main layout
        self.main_layout = pn.Column(
            pn.pane.Markdown("# Growth Rate Analysis for Batch Culture"),
            pn.Row(
                pn.Column(
                    pn.pane.Markdown("### Settings"),
                    self.param.smoothing_window,
                    self.param.min_od_threshold,
                    self.update_button,
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
                ('Results', self.results_output),
                ('Debug', self.debug_message)
            )
        )
    
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
        """
        if self.file_input.value is not None and self.file_input.filename.endswith('.csv'):
            try:
                # Decode the file contents
                decoded = io.BytesIO(self.file_input.value)
                
                # Read the CSV file
                df = pd.read_csv(decoded)
                
                # Process the data
                self._process_data(df)
                
                # Debug information
                cols_info = ", ".join(df.columns)
                debug_info = f"CSV loaded: {len(df)} rows, {len(df.columns)} columns\n"
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
                self.show_success(f"File {self.file_input.filename} uploaded successfully!")
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                self.show_error(f"Error processing file: {str(e)}")
                self.show_debug(f"Error details:\n{tb}")
        else:
            self.show_error("Please upload a CSV file.")
    
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
    
    def _units_changed(self, event):
        """Handle unit selection changes."""
        self.selected_units = self.unit_selector.value
        self._update_plots()
        self._calculate_growth_rate()
    
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
        self._update_plots()
        
        # Calculate growth rate for the selected region
        self._calculate_growth_rate()
    
    def _update_callback(self, event):
        """Handle update button click events."""
        self._update_plots()
        self._calculate_growth_rate()
    
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
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                window['elapsed_hours'], window['ln_od']
            )
            growth_rates.append(slope)
            r_squared_values.append(r_value**2)
            midpoints.append(window['elapsed_hours'].iloc[window_size//2])
        
        if not growth_rates:
            self.show_error("Could not calculate local growth rates.")
            return
        
        # Convert to numpy arrays for easier processing
        growth_rates = np.array(growth_rates)
        r_squared_values = np.array(r_squared_values)
        midpoints = np.array(midpoints)
        
        # Find regions with high growth rate and good linear fit
        score = growth_rates * r_squared_values  # Combine growth rate and fit quality
        
        # Find the region with the highest score
        best_idx = np.argmax(score)
        
        # Define a region around this best point
        best_time = midpoints[best_idx]
        window_width = (unit_data['elapsed_hours'].max() - unit_data['elapsed_hours'].min()) / 10
        start_time = max(unit_data['elapsed_hours'].min(), best_time - window_width)
        end_time = min(unit_data['elapsed_hours'].max(), best_time + window_width)
        
        # Update the region selection
        self.region_start.value = start_time
        self.region_end.value = end_time
        
        # Show success message
        self.show_success(f"Auto-detected exponential phase for {unit} between {start_time:.2f}h and {end_time:.2f}h")
    
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
                    (unit_data['elapsed_hours'] <= self.selected_region['end'])
                ]
                
                if len(region_data) >= 5:  # Need enough points for regression
                    # Perform linear regression on ln(OD)
                    x = region_data['elapsed_hours'].values
                    y = region_data['ln_od'].values
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
        
        # Update the plot panes
        self.od_plot.object = od_plot
        self.ln_od_plot.object = ln_od_plot
    
    def _calculate_growth_rate(self):
        """Calculate growth rate for the selected region."""
        if self.od_data_df.empty or len(self.selected_units) == 0:
            self.results_output.object = "<p>No data available for growth rate calculation.</p>"
            return
        
        # Get region boundaries
        start_time = self.selected_region['start']
        end_time = self.selected_region['end']
        
        if start_time is None or end_time is None:
            self.results_output.object = "<p>Please select a valid region for analysis.</p>"
            return
        
        results = []
        
        for unit in self.selected_units:
            # Get data for the selected unit and region
            unit_data = self.od_data_df[
                (self.od_data_df['exp_unit'] == unit) &
                (self.od_data_df['elapsed_hours'] >= start_time) &
                (self.od_data_df['elapsed_hours'] <= end_time)
            ].copy()
            
            if len(unit_data) < 5:  # Need at least 5 points for a reasonable fit
                results.append({
                    'unit': unit,
                    'growth_rate': None,
                    'doubling_time': None,
                    'r_squared': None,
                    'data_points': len(unit_data)
                })
                continue
            
            # Sort by time
            unit_data = unit_data.sort_values('elapsed_hours')
            
            # Apply smoothing
            unit_data['od_smooth'] = unit_data['od_reading'].rolling(
                window=self.smoothing_window, min_periods=1, center=True).mean()
            
            # Filter out OD values below threshold
            unit_data = unit_data[unit_data['od_smooth'] >= self.min_od_threshold]
            
            if len(unit_data) < 5:  # Check again after filtering
                results.append({
                    'unit': unit,
                    'growth_rate': None,
                    'doubling_time': None,
                    'r_squared': None,
                    'data_points': len(unit_data)
                })
                continue
            
            # Calculate ln(OD)
            unit_data['ln_od'] = np.log(unit_data['od_smooth'])
            
            # Linear regression on ln(OD) vs time
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                unit_data['elapsed_hours'], unit_data['ln_od']
            )
            
            # Calculate growth rate (slope of ln(OD) vs time)
            growth_rate = slope  # in h^-1
            
            # Calculate doubling time
            doubling_time = np.log(2) / growth_rate if growth_rate > 0 else None
            
            # Store results
            results.append({
                'unit': unit,
                'growth_rate': growth_rate,
                'doubling_time': doubling_time,
                'r_squared': r_value**2,
                'data_points': len(unit_data),
                'std_error': std_err
            })
        
        # Store current results
        self.current_results = {r['unit']: r for r in results}
        
        # Create HTML table for results
        html = "<h3>Growth Rate Analysis Results</h3>"
        html += "<p>Analysis for time period: "
        html += f"{start_time:.2f}h to {end_time:.2f}h ({end_time-start_time:.2f}h duration)</p>"
        html += """
        <table style="width:100%; border-collapse:collapse; margin-top:10px;">
          <thead>
            <tr style="background-color:#f2f2f2;">
              <th style="padding:8px; border:1px solid #ddd; text-align:left;">Unit</th>
              <th style="padding:8px; border:1px solid #ddd; text-align:left;">Growth Rate (h⁻¹)</th>
              <th style="padding:8px; border:1px solid #ddd; text-align:left;">Doubling Time (h)</th>
              <th style="padding:8px; border:1px solid #ddd; text-align:left;">R²</th>
              <th style="padding:8px; border:1px solid #ddd; text-align:left;">Std Error</th>
              <th style="padding:8px; border:1px solid #ddd; text-align:left;">Data Points</th>
            </tr>
          </thead>
          <tbody>
        """
        
        for result in results:
            growth_rate = f"{result['growth_rate']:.4f}" if result['growth_rate'] is not None else "N/A"
            doubling_time = f"{result['doubling_time']:.2f}" if result['doubling_time'] is not None else "N/A"
            r_squared = f"{result['r_squared']:.4f}" if result['r_squared'] is not None else "N/A"
            std_err = f"{result['std_error']:.4f}" if result['std_error'] is not None else "N/A"
            
            html += f"""
            <tr>
              <td style="padding:8px; border:1px solid #ddd;">{result['unit']}</td>
              <td style="padding:8px; border:1px solid #ddd;">{growth_rate}</td>
              <td style="padding:8px; border:1px solid #ddd;">{doubling_time}</td>
              <td style="padding:8px; border:1px solid #ddd;">{r_squared}</td>
              <td style="padding:8px; border:1px solid #ddd;">{std_err}</td>
              <td style="padding:8px; border:1px solid #ddd;">{result['data_points']}</td>
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
            <li><strong>Growth Rate (μ):</strong> The exponential growth rate constant in h⁻¹</li>
            <li><strong>Doubling Time:</strong> Time required for the cell population to double (ln(2)/μ)</li>
            <li><strong>R²:</strong> Coefficient of determination, measures goodness of fit (1.0 = perfect fit)</li>
            <li><strong>Std Error:</strong> Standard error of the growth rate estimate</li>
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
    app.show(port=5006)