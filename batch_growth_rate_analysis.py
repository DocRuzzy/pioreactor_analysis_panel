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
import time
from bokeh.models import ColumnDataSource, Range1d, Span, BoxSelectTool
from bokeh.models.formatters import DatetimeTickFormatter
from bokeh.models.callbacks import CustomJS
import io
from scipy import stats
from functools import partial


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
        Initialize the growth rate analysis application with performance improvements.
        """
        super().__init__(**params)
        
        # Initialize empty data containers
        self.od_data_df = pd.DataFrame()
        self.selected_region = {'start': None, 'end': None}
        self.selected_units = []
        self.current_results = {}
        
        # Add debounce timing variables for improved slider performance
        self.last_update_time = 0
        self.debounce_timeout = 250  # milliseconds
        
        # Status message for notifications
        self.status_message = pn.pane.Markdown("", styles={'color': 'blue'})
        self.error_message = pn.pane.Markdown("", styles={'color': 'red'})
        self.debug_message = pn.pane.Markdown("", styles={'color': 'green', 'font-family': 'monospace', 'font-size': '12px'})
        
        # Add visual feedback for sliders
        self.region_feedback = pn.pane.Markdown("", styles={'font-weight': 'bold', 'color': '#007bff'})
        
        # Set up the interface
        self.file_input = pn.widgets.FileInput(accept='.csv', multiple=False)
        self.file_input.param.watch(self._upload_file_callback, 'value')
        
        self.update_button = pn.widgets.Button(name='Update Plots', button_type='primary')
        self.update_button.on_click(self._update_callback)
        
        # Unit selection for multi-unit experiments
        self.unit_selector = pn.widgets.MultiSelect(name='Select Units', options=[])
        self.unit_selector.param.watch(self._units_changed, 'value')
        
        # Create plot panes
        self.od_plot = pn.pane.HoloViews(sizing_mode='stretch_width', height=300)
        self.ln_od_plot = pn.pane.HoloViews(sizing_mode='stretch_width', height=300)
        
        # Selected region widgets with debounced updates
        self.region_start = pn.widgets.FloatSlider(name='Region Start (hours)', start=0, end=100, step=0.1, value=0)
        self.region_end = pn.widgets.FloatSlider(name='Region End (hours)', start=0, end=100, step=0.1, value=100)
        # Remove the watch here, we'll use value_throttled
        # self.region_start.param.watch(self._debounced_update_region, 'value')
        # self.region_end.param.watch(self._debounced_update_region, 'value')
        
        # Use value_throttled for smoother updates after dragging stops
        self.region_start.param.watch(self._throttled_region_update, 'value_throttled')
        self.region_end.param.watch(self._throttled_region_update, 'value_throttled')
        
        # Add a watch on 'value' just for immediate feedback text update
        self.region_start.param.watch(self._update_feedback_text, 'value')
        self.region_end.param.watch(self._update_feedback_text, 'value')
        
        # Add a button to set the region through clicking
        self.region_reset_button = pn.widgets.Button(name='Reset Region Selection', button_type='default')
        self.region_reset_button.on_click(self._reset_region)
        
        # Set up event listener for region boundary changes from JavaScript
        self._setup_boundary_event_listener()
        
        # Auto-detect button
        self.auto_detect_button = pn.widgets.Button(name='Auto-detect Exponential Phase', button_type='success')
        self.auto_detect_button.on_click(self._auto_detect_callback)
        
        # Add a separate calculate button for growth rate (instead of doing it as sliders move)
        self.calculate_button = pn.widgets.Button(
            name='Calculate Growth Rate',
            button_type='primary',
            icon='calculator'
        )
        self.calculate_button.on_click(self._calculate_growth_rate_callback)
        
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
                    self.region_feedback,  # Add feedback display
                    pn.Row(
                        self.auto_detect_button, 
                        self.region_reset_button, 
                        self.calculate_button  # Add separate calculate button
                    ),
                    pn.pane.Markdown("*Click on the plot to adjust region boundaries*", 
                                      styles={'font-style': 'italic', 'color': '#666'}),
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
    
    def _update_feedback_text(self, event):
        """Update the feedback text immediately as sliders move."""
        start = self.region_start.value
        end = self.region_end.value
        # Ensure start <= end for feedback text consistency
        if start > end:
            if event.obj is self.region_start:
                end = start
            else:
                start = end
        feedback_text = f"Selected region: {start:.2f}h - {end:.2f}h" 
        self.region_feedback.object = feedback_text

    def _throttled_region_update(self, event):
        """Update plots after slider movement stops (using value_throttled)."""
        start = self.region_start.value
        end = self.region_end.value

        # Ensure start <= end and update the other slider if needed
        if start > end:
            if event.obj is self.region_start:
                # If start slider moved past end, update end slider's value
                self.region_end.value = start 
                end = start # Use the adjusted value
            else:
                # If end slider moved before start, update start slider's value
                self.region_start.value = end
                start = end # Use the adjusted value

        # Update the internal selected_region dictionary
        self.selected_region['start'] = start
        self.selected_region['end'] = end
        
        # Update the plots visually
        self._update_plots()
        
        # Update the feedback text one last time after adjustment
        feedback_text = f"Selected region: {start:.2f}h - {end:.2f}h" 
        self.region_feedback.object = feedback_text

    def _calculate_growth_rate_callback(self, event):
        """Handle calculate button click events."""
        try:
            self._calculate_growth_rate()
            self.show_success("Growth rate calculated successfully!")
        except Exception as e:
            import traceback
            self.show_error(f"Error calculating growth rate: {str(e)}")
            self.show_debug(traceback.format_exc())

    def _setup_boundary_event_listener(self):
        """Set up event listener for region boundary changes."""
        # This method will now be implemented using Bokeh events
        # The actual functionality is in the _add_draggable_bounds method
        pass

    def _reset_region(self, event):
        """Reset the region selection to the full data range."""
        if not self.od_data_df.empty:
            min_time = self.od_data_df['elapsed_hours'].min()
            max_time = self.od_data_df['elapsed_hours'].max()
            
            # Update both slider values and selected_region directly
            self.region_start.value = min_time
            self.region_end.value = max_time
            self.selected_region = {'start': min_time, 'end': max_time}
            
            # Force plot update
            self._update_plots()
            
            # Calculate growth rate with the new region
            try:
                self._calculate_growth_rate()
                self.show_success("Region reset to full data range")
            except Exception as e:
                import traceback
                self.show_error(f"Error calculating growth rate: {str(e)}")
                self.show_debug(traceback.format_exc())

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
        # Skip if this is a programmatic update with no event
        if event is None:
            return
        
        # Update the selected region directly from slider values
        self.selected_region['start'] = self.region_start.value
        self.selected_region['end'] = self.region_end.value
        
        # Ensure start <= end
        if self.selected_region['start'] > self.selected_region['end']:
            if event.obj is self.region_start:
                # User moved start beyond end - move end to match
                self.selected_region['end'] = self.selected_region['start']
                self.region_end.value = self.selected_region['end']
            else:
                # User moved end below start - move start to match
                self.selected_region['start'] = self.selected_region['end']
                self.region_start.value = self.selected_region['start']
        
        # Update plots with selected region
        self._update_plots()
        
        # Calculate growth rate for the selected region
        try:
            self._calculate_growth_rate()
        except Exception as e:
            import traceback
            self.show_error(f"Error calculating growth rate: {str(e)}")
            self.show_debug(traceback.format_exc())

    def _update_callback(self, event):
        """Handle update button click events."""
        self._update_plots()
        self._calculate_growth_rate()
    
    def _auto_detect_callback(self, event):
        """Improved algorithm to detect exponential growth phase."""
        if self.od_data_df.empty or len(self.selected_units) == 0:
            self.show_error("No data available for auto-detection.")
            return
        
        # Get the first selected unit
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
        
        # Calculate the first derivative (growth rate)
        unit_data['growth_rate'] = unit_data['ln_od'].diff() / unit_data['elapsed_hours'].diff()
        
        # Find most linear segment (lowest variance in growth rate)
        # Use sliding windows of increasing size to find best R²
        min_window_size = 5  # At least 5 points
        max_window_size = min(30, len(unit_data) // 2)  # Up to half the data
        
        best_score = 0
        best_start_idx = 0
        best_end_idx = 0
        best_window_size = 0
        best_slope = 0
        
        # Try different window sizes
        for window_size in range(min_window_size, max_window_size + 1):
            # Slide the window
            for start_idx in range(len(unit_data) - window_size):
                end_idx = start_idx + window_size - 1
                window = unit_data.iloc[start_idx:end_idx+1]
                
                # Linear regression on the window
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    window['elapsed_hours'], window['ln_od']
                )
                
                r2 = r_value**2
                
                # Calculate biological reasonableness score
                # Higher score for windows with:
                # 1. Higher r² (linearity)
                # 2. Positive slope (growth, not death)
                # 3. Reasonable growth rate (not noise or instrumental error)
                # 4. Covers multiple doublings (longer is better, up to a point)
                
                doublings = (window['ln_od'].max() - window['ln_od'].min()) / np.log(2)
                min_reasonable_slope = 0.05  # Minimum biologically reasonable growth rate
                max_reasonable_slope = 3.0   # Maximum biologically reasonable growth rate
                
                # Skip unreasonable growth rates
                if slope < min_reasonable_slope or slope > max_reasonable_slope:
                    continue
                    
                # Biological score factors
                linearity_score = r2
                growth_factor = 1.0 if slope > 0 else 0.0
                doubling_factor = min(1.0, doublings / 3.0)  # Max score at 3+ doublings
                
                # Combined score - prioritize linearity and biological relevance
                bio_score = linearity_score * growth_factor * doubling_factor
                
                if bio_score > best_score:
                    best_score = bio_score
                    best_start_idx = start_idx
                    best_end_idx = end_idx
                    best_window_size = window_size
                    best_slope = slope
        
        if best_window_size == 0:
            self.show_error("Could not identify a clear exponential phase.")
            return
        
        # Extend region in both directions while R² remains high
        extended_start_idx = best_start_idx
        extended_end_idx = best_end_idx
        
        # Try extending the start backwards
        for i in range(best_start_idx - 1, -1, -1):
            test_window = unit_data.iloc[i:best_end_idx+1]
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                test_window['elapsed_hours'], test_window['ln_od']
            )
            if r_value**2 > 0.98 * best_score:  # Allow small decrease in R²
                extended_start_idx = i
            else:
                break
        
        # Try extending the end forwards
        for i in range(best_end_idx + 1, len(unit_data)):
            test_window = unit_data.iloc[extended_start_idx:i+1]
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                test_window['elapsed_hours'], test_window['ln_od']
            )
            if r_value**2 > 0.98 * best_score:  # Allow small decrease in R²
                extended_end_idx = i
            else:
                break
        
        # Get the start and end times
        start_time = unit_data.iloc[extended_start_idx]['elapsed_hours']
        end_time = unit_data.iloc[extended_end_idx]['elapsed_hours']
        
        # Create a preview visualization of the detected region before applying
        preview_data = unit_data.iloc[extended_start_idx:extended_end_idx+1]
        
        # Calculate the exponential parameters for reporting
        final_window = unit_data.iloc[extended_start_idx:extended_end_idx+1]
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            final_window['elapsed_hours'], final_window['ln_od']
        )
        
        # Calculate confidence intervals for the growth rate (95%)
        t_value = stats.t.ppf(0.975, len(final_window)-2)  # 95% CI
        ci_lower = slope - t_value * std_err
        ci_upper = slope + t_value * std_err
        
        # Calculate number of doublings in this window
        doublings = (final_window['ln_od'].max() - final_window['ln_od'].min()) / np.log(2)
        
        # Create preview plot
        preview_curve = hv.Curve(
            preview_data, 'elapsed_hours', 'ln_od', 
            label=f'Selected region'
        ).opts(
            color='red',
            line_width=2
        )
        
        # Generate x values for the regression line
        x_range = np.array([start_time, end_time])
        y_range = slope * x_range + intercept
        
        # Create regression line
        reg_line = hv.Curve(
            np.column_stack([x_range, y_range]),
            label=f"Linear fit (R²={r_value**2:.4f})"
        ).opts(
            color='green',
            line_width=2,
            line_dash='dashed'
        )
        
        # Combine the preview plot elements
        preview_plot = (preview_curve * reg_line).opts(
            title='Detected Exponential Phase',
            xlabel='Time (hours)',
            ylabel='ln(OD)',
            width=400,
            height=300,
            legend_position='top_right'
        )
        
        # Create the preview modal
        preview_modal = pn.Column(
            pn.pane.Markdown("### Detected Exponential Growth Phase"),
            pn.pane.HoloViews(preview_plot),
            pn.pane.Markdown(f"""
            **Growth rate (μ):** {slope:.4f}h⁻¹ (95% CI: {ci_lower:.4f} - {ci_upper:.4f})  
            **R²:** {r_value**2:.4f}  
            **Time range:** {start_time:.2f}h - {end_time:.2f}h ({end_time-start_time:.2f}h duration)  
            **Doublings:** {doublings:.1f}
            """),
            pn.Row(
                pn.widgets.Button(name='Apply', button_type='success', 
                                on_click=lambda e: self._apply_detected_region(start_time, end_time)),
                pn.widgets.Button(name='Cancel', button_type='danger',
                                on_click=lambda e: self._cancel_detection())
            )
        )
        
        # Store the preview modal
        self.preview_modal = preview_modal
        
        # Show the modal
        try:
            if hasattr(pn, 'state') and hasattr(pn.state, 'modal'):
                # For Panel >= 0.13.0
                self.modal_window = pn.state.modal(self.preview_modal, title="Exponential Phase Preview", size='large')
                self.modal_window.show()
            else:
                # Fallback for older Panel versions or when state/modal is not available
                self._apply_detected_region(start_time, end_time)
                
                # And show success message
                self.show_success(
                    f"Auto-detected exponential phase from {start_time:.2f}h to {end_time:.2f}h "
                    f"(μ={slope:.4f}h⁻¹, R²={r_value**2:.4f}, {doublings:.1f} doublings)"
                )
        except Exception as e:
            # If anything goes wrong, use the direct approach
            import traceback
            self.show_debug(f"Modal error: {str(e)}\n{traceback.format_exc()}")
            self._apply_detected_region(start_time, end_time)

    def _apply_detected_region(self, start_time, end_time):
        """Apply the detected region from the preview"""
        # Update both the region selection sliders and internal dictionary
        self.selected_region = {'start': start_time, 'end': end_time}
        self.region_start.value = start_time
        self.region_end.value = end_time
        
        # Close the modal if it exists
        if hasattr(self, 'modal_window'):
            try:
                self.modal_window.close()
            except Exception as e:
                self.show_debug(f"Warning: Could not close modal window: {str(e)}")
        
        # Update plots
        self._update_plots()
        
        # Calculate growth rate with the new region
        try:
            self._calculate_growth_rate()
            # Show success message
            self.show_success(f"Applied exponential phase detection: {start_time:.2f}h - {end_time:.2f}h")
        except Exception as e:
            import traceback
            self.show_error(f"Error calculating growth rate: {str(e)}")
            self.show_debug(traceback.format_exc())

    def _cancel_detection(self):
        """Cancel the detection preview"""
        # Just close the modal
        if hasattr(self, 'modal_window'):
            self.modal_window.close()
        
        self.show_success("Exponential phase detection cancelled")
    
    def _update_plots(self):
        """Update all plots with current data and parameters, optimized for performance."""
        # Check if we have data and if plots are already created
        if self.od_data_df.empty or len(self.selected_units) == 0:
            # Display empty placeholders
            self.od_plot.object = hv.Text(0, 0, 'No valid OD data found').opts(width=800, height=300)
            self.ln_od_plot.object = hv.Text(0, 0, 'No valid ln(OD) data found').opts(width=800, height=300)
            return
        
        # Check if we need to create new plots or just update existing ones
        create_new_plots = (
            not hasattr(self, '_base_od_plot') or 
            not hasattr(self, '_base_ln_od_plot') or
            getattr(self, '_current_units', []) != self.selected_units
        )
        
        if create_new_plots:
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
            
            # Combine the curves into a base plot
            od_overlay = hv.Overlay(od_curves)
            self._base_od_plot = od_overlay.opts(
                title='OD vs Time',
                xlabel='Time (hours)',
                ylabel='OD (smoothed)',
                legend_position='top_right',
                width=800,
                height=300
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
            
            # Combine the curves into a base plot
            ln_od_overlay = hv.Overlay(ln_od_curves)
            self._base_ln_od_plot = ln_od_overlay.opts(
                title='ln(OD) vs Time',
                xlabel='Time (hours)',
                ylabel='ln(OD)',
                legend_position='top_right',
                width=800,
                height=300
            )
            
            # Store current units to detect changes
            self._current_units = list(self.selected_units)
            
            # Store processed data for later use
            self._processed_data = processed_data
        
        # Now, overlay the region selection - this is much faster than redrawing everything
        if self.selected_region['start'] is not None and self.selected_region['end'] is not None:
            # Create a shaded region between start and end
            region_box = hv.VSpan(
                self.selected_region['start'], 
                self.selected_region['end']
            ).opts(alpha=0.2, color='red')
            
            # Add region to base plots
            od_plot = self._base_od_plot * region_box
            ln_od_plot = self._base_ln_od_plot * region_box
            
            # Add vertical lines to clearly mark boundaries
            start_line = hv.VLine(self.selected_region['start']).opts(
                color='red', line_width=1.5, line_dash='dashed')
            end_line = hv.VLine(self.selected_region['end']).opts(
                color='red', line_width=1.5, line_dash='dashed')
                
            od_plot = od_plot * start_line * end_line
            ln_od_plot = ln_od_plot * start_line * end_line
        else:
            # If no region is selected, just use the base plots
            od_plot = self._base_od_plot
            ln_od_plot = self._base_ln_od_plot
        
        # Add draggable bounds using the existing method
        od_plot = od_plot.opts(tools=['tap'])
        ln_od_plot = ln_od_plot.opts(tools=['tap'])
        
        # Update plot panes (much faster than full redraw)
        self.od_plot.object = od_plot
        self.ln_od_plot.object = ln_od_plot

    def _add_draggable_bounds(self, bokeh_plot):
        """Add draggable bounds to a Bokeh plot."""
        from bokeh.models import Span, CustomJS, TapTool
        
        # Create spans for start and end boundaries
        start_span = Span(
            location=self.selected_region['start'],
            dimension='height',
            line_color='red',
            line_width=3
        )
        
        end_span = Span(
            location=self.selected_region['end'],
            dimension='height',
            line_color='red',
            line_width=3
        )
        
        # Add spans to the plot
        bokeh_plot.add_layout(start_span)
        bokeh_plot.add_layout(end_span)
        
        # Create a ColumnDataSource to communicate between JS and Python
        from bokeh.models import ColumnDataSource
        callback_source = ColumnDataSource(data=dict(x=[0], is_start=[True]))
        
        # Add the source to the plot
        bokeh_plot.add_tools(TapTool())
        
        # Create the JS callback
        js_callback = CustomJS(args=dict(source=callback_source), code="""
        const x = cb_obj.x;
        // Figure out if we're closer to the start or end span
        const start_loc = """ + str(self.selected_region['start']) + """;
        const end_loc = """ + str(self.selected_region['end']) + """;
        
        // Decide which boundary to move based on which is closer
        let is_start = Math.abs(x - start_loc) < Math.abs(x - end_loc);
        
        // Update the source data - this will trigger the Python callback
        source.data = {
            'x': [x],
            'is_start': [is_start]
        };
        
        source.change.emit();
        """)
        
        # Connect JavaScript to the tap event
        bokeh_plot.js_on_event('tap', js_callback)
        
        # Add a Python callback for the source changes
        def update_boundary(attr, old, new):
            if len(new['x']) > 0:
                x = new['x'][0]
                is_start = bool(new['is_start'][0])
                if is_start:
                    self._update_region_start(x)
                else:
                    self._update_region_end(x)
        
        callback_source.on_change('data', update_boundary)
        
        return bokeh_plot

    def _update_region_start(self, value):
        """Update the start of the region."""
        # Ensure it doesn't go beyond the end
        if value <= self.region_end.value:
            # Update both the slider and the region dictionary
            self.region_start.value = value
            self.selected_region['start'] = value
            
            # Manually call updates
            self._update_plots()
            try:
                self._calculate_growth_rate()
            except Exception as e:
                import traceback
                self.show_error(f"Error calculating growth rate: {str(e)}")
                self.show_debug(traceback.format_exc())

    def _update_region_end(self, value):
        """Update the end of the region."""
        # Ensure it doesn't go below the start
        if value >= self.region_start.value:
            # Update both the slider and the region dictionary
            self.region_end.value = value
            self.selected_region['end'] = value
            
            # Manually call updates
            self._update_plots()
            try:
                self._calculate_growth_rate()
            except Exception as e:
                import traceback
                self.show_error(f"Error calculating growth rate: {str(e)}")
                self.show_debug(traceback.format_exc())
        
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
                    'std_error': None,  # Add this key to prevent KeyError
                    'data_points': len(unit_data)
                })
                continue
            
            # Apply smoothing first
            unit_data['od_smooth'] = unit_data['od_reading'].rolling(
                window=self.smoothing_window, min_periods=1, center=True).mean()
            
            # Calculate ln(OD)
            unit_data['ln_od'] = np.log(unit_data['od_smooth'].clip(lower=1e-6))  # Avoid log(0)
            
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
            
            # Fix this line to safely handle missing std_error key
            std_err = f"{result['std_error']:.4f}" if result.get('std_error') is not None else "N/A"
            
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