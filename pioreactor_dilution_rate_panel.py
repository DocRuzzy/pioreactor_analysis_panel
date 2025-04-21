# Interactive Pioreactor Dilution Rate Analysis Tool with Panel/HoloViz
"""
Pioreactor Dilution Rate Analysis Tool

This application provides an interactive dashboard for analyzing dilution rate data from 
Pioreactor continuous culture devices. It allows users to upload CSV data files, visualize 
dilution rates over time, and analyze optical density (OD) measurements.

Features:
- Interactive time series visualization of dilution rates
- OD tracking and comparison to target values
- Statistical analysis of dilution rates by OD region
- User bookmarking system for points of interest
- Customizable parameters (reactor volume, moving average window)

Author: Russell Kirk Pirlo with Claude Copilot
Date: April 21, 2025
"""

import pandas as pd
import numpy as np
import hvplot.pandas
import holoviews as hv
import panel as pn
import param
from bokeh.models import HoverTool
import json
from scipy.stats import variation
import uuid
import io
import base64
import re
# Import HoloViews linking
import holoviews.plotting.links

# Configure Panel and HoloViews
pn.extension('plotly', 'tabulator')
hv.extension('bokeh')

# Style the panel with a nice theme
pn.config.sizing_mode = 'stretch_width'

class PioreactorAnalysis(param.Parameterized):
    """
    Main application class for Pioreactor dilution rate analysis.
    
    This class manages data processing, visualization, and the user interface
    for analyzing Pioreactor continuous culture data.
    
    Parameters
    ----------
    reactor_volume : float
        Volume of the reactor in mL (default: 14.0)
    moving_avg_window : int
        Number of events to include in moving average calculations (default: 5)
    """
    
    reactor_volume = param.Number(14.0, bounds=(1, 100), step=0.1, 
                                  doc="Reactor volume in mL")
    moving_avg_window = param.Integer(5, bounds=(1, 50), step=1, 
                                      doc="Moving average window (# of events)")
    
    def __init__(self, **params):
        """
        Initialize the Pioreactor analysis application.
        
        Sets up the UI components, data containers, and layout structure.
        """
        super().__init__(**params)
        
        # Initialize empty data containers
        self.dosing_events_df = pd.DataFrame()
        self.target_od_df = pd.DataFrame()
        self.latest_od_df = pd.DataFrame()
        self.bookmarks = []
        
        # Status message for notifications (replacing notification system)
        self.status_message = pn.pane.Markdown("", styles={'color': 'blue'})
        self.error_message = pn.pane.Markdown("", styles={'color': 'red'})
        self.debug_message = pn.pane.Markdown("", styles={'color': 'green', 'font-family': 'monospace', 'font-size': '12px'})
        
        # Set up the interface
        self.file_input = pn.widgets.FileInput(accept='.csv', multiple=False)
        self.file_input.param.watch(self._upload_file_callback, 'value')
        
        self.update_button = pn.widgets.Button(name='Update', button_type='primary')
        self.update_button.on_click(self._update_callback)
        
        # Create plot panes
        self.dilution_plot = pn.pane.HoloViews(sizing_mode='stretch_width', height=300)
        self.od_plot = pn.pane.HoloViews(sizing_mode='stretch_width', height=250)
        self.time_plot = pn.pane.HoloViews(sizing_mode='stretch_width', height=250)
        
        # Stats output
        self.stats_output = pn.pane.HTML("")
        
        # Bookmarks section
        self.bookmarks_title = pn.pane.Markdown("### Bookmarks")
        self.bookmarks_container = pn.Column(
            pn.pane.Markdown("No bookmarks yet. Click on points in the graph to bookmark them.")
        )
        
        # Main layout
        self.main_layout = pn.Column(
            pn.pane.Markdown("# Pioreactor Dilution Rate Analysis"),
            pn.Row(
                pn.Column(
                    pn.pane.Markdown("### Settings"),
                    self.param.reactor_volume,
                    self.param.moving_avg_window,
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
            pn.Tabs(
                ('Dilution Analysis', pn.Column(
                    self.dilution_plot,
                    self.od_plot,
                    self.time_plot
                )),
                ('Statistics', self.stats_output),
                ('Debug', self.debug_message)
            ),
            pn.Row(
                pn.Column(
                    self.bookmarks_title,
                    self.bookmarks_container
                )
            )
        )
    
    def show_success(self, message):
        """
        Display a success message to the user.
        
        Parameters
        ----------
        message : str
            The success message to display
        """
        self.status_message.object = f"✅ **{message}**"
        self.error_message.object = ""  # Clear any error messages
    
    def show_error(self, message):
        """
        Display an error message to the user.
        
        Parameters
        ----------
        message : str
            The error message to display
        """
        self.error_message.object = f"❌ **Error:** {message}"
        self.status_message.object = ""  # Clear any success messages
    
    def show_warning(self, message):
        """
        Display a warning message to the user.
        
        Parameters
        ----------
        message : str
            The warning message to display
        """
        self.status_message.object = f"⚠️ **{message}**"
    
    def show_debug(self, message):
        """
        Display a debug message on the debug tab.
        
        Parameters
        ----------
        message : str
            The debug message to display
        """
        self.debug_message.object = f"```\n{message}\n```"
    
    def _upload_file_callback(self, event):
        """
        Handle file upload events from the UI.
        
        Reads the uploaded CSV file, processes the data, and updates the visualization.
        
        Parameters
        ----------
        event : param.Event
            The parameter event triggering the callback
        """
        if self.file_input.value is not None and self.file_input.filename.endswith('.csv'):
            try:
                # Decode the file contents
                decoded = io.BytesIO(self.file_input.value)
                
                # Read the CSV file
                df = pd.read_csv(decoded)
                
                # Show debug information about the loaded data
                cols_info = ", ".join(df.columns)
                event_names = df['event_name'].unique() if 'event_name' in df.columns else []
                events_info = ", ".join(event_names)
                
                debug_info = f"CSV loaded: {len(df)} rows, {len(df.columns)} columns\n"
                debug_info += f"Columns: {cols_info}\n"
                debug_info += f"Event types: {events_info}\n"
                
                # Process the data
                self._process_data(df)
                
                # Add more debug info about processed data
                debug_info += f"\nAfter processing:\n"
                debug_info += f"Dosing events: {len(self.dosing_events_df)} rows\n"
                debug_info += f"Target OD events: {len(self.target_od_df)} rows\n"
                debug_info += f"Latest OD events: {len(self.latest_od_df)} rows\n"
                
                if not self.latest_od_df.empty and 'od_value' in self.latest_od_df.columns:
                    debug_info += f"OD values: {self.latest_od_df['od_value'].min():.3f} - {self.latest_od_df['od_value'].max():.3f}\n"
                    sample_values = self.latest_od_df['od_value'].head(5).tolist()
                    debug_info += f"Sample OD values: {sample_values}\n"
                
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
            self.show_warning("Please upload a CSV file.")
    
    def _process_data(self, df):
        """
        Process the uploaded CSV data for analysis.
        
        Converts timestamps, calculates elapsed time, extracts dosing events,
        and processes OD-related values.
        
        Parameters
        ----------
        df : pandas.DataFrame
            The raw data frame from the uploaded CSV file
        """
        # Convert timestamps to datetime objects
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Create elapsed time column (in hours)
        start_time = df['timestamp'].min()
        df['elapsed_hours'] = (df['timestamp'] - start_time).dt.total_seconds() / 3600
        
        # Extract dosing events and OD-related events
        # Look for more general event names related to dilution and OD
        self.dosing_events_df = df[df['event_name'].str.contains('dilution', case=False, na=False)].copy()
        
        # More flexible approach to get OD events
        target_od_mask = df['event_name'].str.contains('target', case=False, na=False) & df['event_name'].str.contains('od', case=False, na=False)
        latest_od_mask = df['event_name'].str.contains('latest', case=False, na=False) & df['event_name'].str.contains('od', case=False, na=False)
        
        # If no specific OD events, look for any OD-related events
        if not target_od_mask.any() and not latest_od_mask.any():
            od_events = df[df['event_name'].str.contains('od', case=False, na=False)].copy()
            # Try to infer target vs latest from message or data
            for idx, row in od_events.iterrows():
                if 'target' in str(row['message']).lower() or 'target' in str(row['data']).lower():
                    target_od_mask.loc[idx] = True
                elif 'latest' in str(row['message']).lower() or 'latest' in str(row['data']).lower():
                    latest_od_mask.loc[idx] = True
                else:
                    # Default to latest if can't determine
                    latest_od_mask.loc[idx] = True
        else:
            od_events = df[target_od_mask | latest_od_mask].copy()
        
        # Parse volume from dosing events
        if not self.dosing_events_df.empty:
            self.dosing_events_df['volume'] = self.dosing_events_df.apply(self._extract_volume, axis=1)
        
        # Extract OD values with enhanced debugging
        od_extraction_details = []
        
        for idx, row in od_events.iterrows():
            od_value, details = self._extract_od_with_details(row['message'], row['data'])
            od_events.loc[idx, 'od_value'] = od_value
            od_extraction_details.append(f"Row {idx}: {details}")
        
        # Log detailed OD extraction
        self.show_debug("\nOD extraction details:\n" + "\n".join(od_extraction_details[:10]) + 
                       ("\n..." if len(od_extraction_details) > 10 else ""))
        
        # Separate target_od and latest_od using the masks
        self.target_od_df = od_events[target_od_mask].copy() if target_od_mask.any() else pd.DataFrame()
        self.latest_od_df = od_events[latest_od_mask].copy() if latest_od_mask.any() else pd.DataFrame()
        
        # If we still don't have OD events, look for it in different columns or try different parsing
        if (self.target_od_df.empty and self.latest_od_df.empty) or 'od_value' not in od_events.columns:
            self.show_warning("Could not find OD data in expected format. Trying alternative methods...")
            self._try_alternative_od_parsing(df)
    
    def _try_alternative_od_parsing(self, df):
        """
        Try alternative methods to parse OD data if standard methods fail.
        
        Parameters
        ----------
        df : pandas.DataFrame
            The raw data frame
        """
        # Method 1: Look for any numeric values in message that could be OD
        od_candidates = []
        
        for idx, row in df.iterrows():
            message = str(row['message'])
            # Look for patterns like "OD: 0.123" or "OD = 0.456"
            matches = re.findall(r'OD[:\s=]+([0-9.]+)', message, re.IGNORECASE)
            if matches:
                od_candidates.append({
                    'timestamp': row['timestamp'],
                    'event_name': 'latest_od',  # Default to latest
                    'od_value': float(matches[0])
                })
        
        if od_candidates:
            self.latest_od_df = pd.DataFrame(od_candidates)
            self.show_success(f"Found {len(od_candidates)} OD values using alternative parsing")
        else:
            # Method 2: Try parsing JSON in the data column
            json_od_values = []
            
            for idx, row in df.iterrows():
                if pd.notna(row.get('data')):
                    try:
                        data = json.loads(row['data'])
                        # Look for fields that might contain OD values
                        for key in ['od', 'OD', 'od_reading', 'od_value', 'latest_od']:
                            if key in data and isinstance(data[key], (int, float)):
                                json_od_values.append({
                                    'timestamp': row['timestamp'],
                                    'event_name': 'latest_od',
                                    'od_value': float(data[key])
                                })
                                break
                    except:
                        pass  # Not valid JSON or doesn't contain OD
            
            if json_od_values:
                self.latest_od_df = pd.DataFrame(json_od_values)
                self.show_success(f"Found {len(json_od_values)} OD values from JSON data")
    
    def _extract_volume(self, row):
        """
        Extract volume information from dosing events.
        
        Parses the JSON data column to extract volume values.
        
        Parameters
        ----------
        row : pandas.Series
            A row from the dosing events dataframe
            
        Returns
        -------
        float or None
            The extracted volume value, or None if not found/parseable
        """
        try:
            # Try data column first
            if 'data' in row and pd.notna(row['data']):
                data_dict = json.loads(row['data'])
                if 'volume' in data_dict:
                    return float(data_dict['volume'])
            
            # Try message column
            if 'message' in row and pd.notna(row['message']):
                # Look for patterns like "volume: 0.5 mL" or "added 0.5 mL"
                match = re.search(r'(?:volume|added|removed)[:\s]+([0-9.]+)\s*m?L', 
                                 str(row['message']), re.IGNORECASE)
                if match:
                    return float(match.group(1))
        except Exception as e:
            # Just silently fail and return None
            pass
        
        return None
    
    def _extract_od_with_details(self, message, data):
        """
        Extract optical density (OD) values with detailed logging for debugging.
        
        Parameters
        ----------
        message : str
            The message string containing OD information
        data : str
            The data string (usually JSON) that may contain OD information
            
        Returns
        -------
        tuple
            (od_value, details_string) where od_value is the extracted value (or None)
            and details_string contains information about the extraction process
        """
        try:
            # Try parsing JSON from data column
            if pd.notna(data) and isinstance(data, str):
                try:
                    data_dict = json.loads(data)
                    if 'od' in data_dict:
                        return float(data_dict['od']), f"Found in data JSON 'od': {data_dict['od']}"
                    if 'latest_od' in data_dict:
                        return float(data_dict['latest_od']), f"Found in data JSON 'latest_od': {data_dict['latest_od']}"
                    
                    # Log all keys in the JSON for debugging
                    return None, f"Data JSON keys: {list(data_dict.keys())}, no OD found"
                except:
                    pass
            
            # Try parsing JSON from message
            if pd.notna(message) and isinstance(message, str) and message.startswith('{'):
                try:
                    msg_dict = json.loads(message)
                    if 'od' in msg_dict:
                        return float(msg_dict['od']), f"Found in message JSON 'od': {msg_dict['od']}"
                    if 'latest_od' in msg_dict:
                        return float(msg_dict['latest_od']), f"Found in message JSON 'latest_od': {msg_dict['latest_od']}"
                    
                    # Log all keys in the JSON for debugging
                    return None, f"Message JSON keys: {list(msg_dict.keys())}, no OD found"
                except:
                    pass
            
            # Try to extract OD value from message using regex
            if pd.notna(message) and isinstance(message, str):
                # Pattern 1: "Latest OD = 0.30"
                match1 = re.search(r'(?:Latest|Target)\s*OD\s*=\s*([0-9.]+)', message)
                if match1:
                    return float(match1.group(1)), f"Regex pattern 1: {match1.group(1)} from '{message}'"
                
                # Pattern 2: "OD: 0.30"
                match2 = re.search(r'OD[:\s=]+([0-9.]+)', message, re.IGNORECASE)
                if match2:
                    return float(match2.group(1)), f"Regex pattern 2: {match2.group(1)} from '{message}'"
                
                # Pattern 3: Any number that might be an OD
                match3 = re.search(r'(?<!\d)([0-9]+\.[0-9]+)(?!\d)', message)
                if match3:
                    return float(match3.group(1)), f"Regex pattern 3: {match3.group(1)} from '{message}'"
                
                # Log the message for debugging
                return None, f"No OD pattern match in: '{message}'"
            
            return None, "Message or data is null or not a string"
        
        except Exception as e:
            return None, f"Error in extraction: {str(e)}"
    
    def _update_callback(self, event):
        """
        Handle update button click events.
        
        Updates all plots with current parameter values.
        
        Parameters
        ----------
        event : param.Event
            The parameter event triggering the callback
        """
        self._update_plots()
    
    def _add_bookmark(self, x, y, label=None):
        """
        Add a bookmark at the specified point.
        
        Creates a new bookmark entry and updates the bookmarks display.
        
        Parameters
        ----------
        x : float or pandas.Timestamp
            The x-coordinate (typically timestamp) of the bookmarked point
        y : float
            The y-coordinate (value) of the bookmarked point
        label : str, optional
            Custom label for the bookmark, defaults to a generic point description
        """
        bookmark_id = str(uuid.uuid4())
        bookmark = {
            'id': bookmark_id,
            'timestamp': x,
            'value': y,
            'label': label or f"Point at {x}"
        }
        self.bookmarks.append(bookmark)
        self._update_bookmarks_container()
    
    def _remove_bookmark(self, bookmark_id):
        """
        Remove a bookmark with the specified ID.
        
        Parameters
        ----------
        bookmark_id : str
            UUID of the bookmark to remove
        """
        self.bookmarks = [bm for bm in self.bookmarks if bm['id'] != bookmark_id]
        self._update_bookmarks_container()
    
    def _update_bookmarks_container(self):
        """
        Update the bookmarks container in the UI.
        
        Creates and displays bookmark cards for all saved bookmarks.
        """
        if not self.bookmarks:
            self.bookmarks_container[0] = pn.pane.Markdown("No bookmarks yet. Click on points in the graph to bookmark them.")
            return
        
        bookmark_cards = []
        for bm in self.bookmarks:
            # Format timestamp for display
            if isinstance(bm['timestamp'], pd.Timestamp):
                time_str = bm['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
            else:
                time_str = str(bm['timestamp'])
                
            card = pn.Card(
                pn.Row(
                    pn.pane.Markdown(f"**{bm['label']}**", width=300),
                    pn.widgets.Button(name="✕", width=30, button_type='danger', 
                                      on_click=lambda event, id=bm['id']: self._remove_bookmark(id))
                ),
                pn.pane.Markdown(f"**Time:** {time_str}"),
                pn.pane.Markdown(f"**Value:** {bm['value']:.3f}"),
                pn.widgets.Button(name="Copy Data", button_type='default', 
                                 on_click=lambda event, bm=bm: self.show_success(f"Copied: Time: {bm['timestamp']}, Value: {bm['value']}")),
                title=bm['label'],
                collapsed=False
            )
            bookmark_cards.append(card)
        
        self.bookmarks_container[0] = pn.Column(*bookmark_cards)
    
    def _create_tap_callback(self, plot):
        """
        Create a callback function for tap events on a plot.
        
        This enables adding bookmarks by clicking on plot points.
        
        Parameters
        ----------
        plot : holoviews.Element
            The plot to attach the tap callback to
            
        Returns
        -------
        function
            A callback function that adds bookmarks when points are tapped
        """
        # Fixed callback signature to match what HoloViews expects
        def tap_callback(plot_id, element):
            if hasattr(plot_id, 'x') and hasattr(plot_id, 'y'):
                self._add_bookmark(plot_id.x, plot_id.y)
                self.show_success(f"Bookmark added at x={plot_id.x}, y={plot_id.y}")
        
        return tap_callback
    
    def _update_plots(self):
        """
        Update all plots with current data and parameters.
        
        Generates dilution rate plots, OD value plots, and time between doses plots.
        Also calculates statistics and links the plots for synchronized interactions.
        """
        if self.dosing_events_df.empty:
            # No data to plot for dilution rate
            self.dilution_plot.object = hv.Text(0, 0, 'No dilution event data available').opts(width=800, height=300)
            self.time_plot.object = hv.Text(0, 0, 'No dilution event data available').opts(width=800, height=250)
        else:
            # Calculate dilution rate from dosing events
            # Sort by timestamp
            self.dosing_events_df = self.dosing_events_df.sort_values('timestamp')
            
            # Filter out rows with missing volume
            self.dosing_events_df = self.dosing_events_df[pd.notna(self.dosing_events_df['volume'])]
            
            if len(self.dosing_events_df) <= 1:
                self.dilution_plot.object = hv.Text(0, 0, 'Insufficient dilution events to calculate rates').opts(width=800, height=300)
                self.time_plot.object = hv.Text(0, 0, 'Insufficient dilution events to calculate rates').opts(width=800, height=250)
                self.stats_output.object = "<p>Insufficient data for analysis</p>"
                return
            
            # Calculate time between doses
            self.dosing_events_df['next_timestamp'] = self.dosing_events_df['timestamp'].shift(-1)
            self.dosing_events_df['time_diff_hours'] = (self.dosing_events_df['next_timestamp'] - 
                                                      self.dosing_events_df['timestamp']).dt.total_seconds() / 3600
            
            # Calculate instant dilution rate (with safety check for division by zero)
            self.dosing_events_df['instant_dilution_rate'] = np.where(
                self.dosing_events_df['time_diff_hours'] > 0,
                self.dosing_events_df['volume'] / self.reactor_volume / self.dosing_events_df['time_diff_hours'],
                np.nan
            )
            
            # Drop NaN and infinite values for better plotting
            self.dosing_events_df = self.dosing_events_df.replace([np.inf, -np.inf], np.nan).dropna(subset=['instant_dilution_rate'])
            
            # Calculate moving average of dilution rate
            self.dosing_events_df['moving_avg_dilution_rate'] = (
                self.dosing_events_df['instant_dilution_rate']
                .rolling(window=self.moving_avg_window, min_periods=1)
                .mean()
            )
            
            # HoloViews plots with tooltips
            dilution_scatter = hv.Scatter(
                self.dosing_events_df, 
                'timestamp', 
                'instant_dilution_rate',
                label='Instant Dilution Rate'
            ).opts(
                color='blue',
                size=5,
                tools=['hover'],
                width=800,
                height=300,
                title='Dilution Rate'
            )
            
            dilution_line = hv.Curve(
                self.dosing_events_df, 
                'timestamp', 
                'moving_avg_dilution_rate',
                label=f'Moving Avg ({self.moving_avg_window} events)'
            ).opts(
                color='red',
                line_width=2,
                width=800,
                height=300
            )
            
            # Calculate outlier threshold for time differences
            self.dosing_events_df['capped_time_diff'] = np.minimum(self.dosing_events_df['time_diff_hours'] * 60, 60)
            self.dosing_events_df['is_outlier'] = self.dosing_events_df['time_diff_hours'] * 60 > 60
            
            # Create a dataframe for normal points and outliers
            normal_df = self.dosing_events_df[~self.dosing_events_df['is_outlier']]
            outlier_df = self.dosing_events_df[self.dosing_events_df['is_outlier']]
            
            # Time between doses plot
            time_line = hv.Curve(
                self.dosing_events_df,
                'timestamp',
                'capped_time_diff',
                label='Minutes Between Doses'
            ).opts(
                color='blue',
                line_width=1,
                width=800,
                height=250,
                title='Inter-Dosing Period',
                ylabel='Minutes Between Doses (capped at 60)',
                tools=['hover']
            )
            
            normal_scatter = hv.Scatter(
                normal_df,
                'timestamp',
                'capped_time_diff'
            ).opts(
                color='blue',
                size=6,
                tools=['hover'],
                width=800,
                height=250
            )
            
            outlier_scatter = hv.Scatter(
                outlier_df,
                'timestamp',
                'capped_time_diff'
            ).opts(
                color='red',
                marker='circle',
                size=8,
                tools=['hover'],
                width=800,
                height=250
            )
            
            # Combine plots
            dilution_plot = (dilution_scatter * dilution_line).opts(
                legend_position='top_right',
                xlabel='Time',
                ylabel='Dilution Rate (h⁻¹)'
            )
            
            time_plot = (time_line * normal_scatter * outlier_scatter).opts(
                legend_position='top_right',
                xlabel='Time',
                ylabel='Minutes Between Doses',
                ylim=(0, 65)
            )
            
            # Add tap callbacks for interactions
            dilution_plot = dilution_plot.opts(tools=['tap'])
            dilution_plot.opts(hooks=[self._create_tap_callback(dilution_plot)])
            
            time_plot = time_plot.opts(tools=['tap'])
            time_plot.opts(hooks=[self._create_tap_callback(time_plot)])
            
            # Update dilution plot panes
            self.dilution_plot.object = dilution_plot
            self.time_plot.object = time_plot
        
        # OD Values plot - handle separately to allow OD plotting even if dilution data is incomplete
        if not self.latest_od_df.empty and 'od_value' in self.latest_od_df.columns:
            od_plots = []
            
            # Sort OD data by timestamp
            if not self.target_od_df.empty and 'od_value' in self.target_od_df.columns:
                self.target_od_df = self.target_od_df.sort_values('timestamp')
                
                # Clean OD data - remove NaN, negative, and outlier values
                self.target_od_df = self.target_od_df[
                    pd.notna(self.target_od_df['od_value']) & 
                    (self.target_od_df['od_value'] >= 0) &
                    (self.target_od_df['od_value'] < 10)  # Assuming OD > 10 is an error
                ]
                
                if not self.target_od_df.empty:
                    target_od = hv.Curve(
                        self.target_od_df,
                        'timestamp',
                        'od_value',
                        label='Target OD'
                    ).opts(
                        color='green',
                        line_dash='dashed',
                        line_width=2,
                        width=800,
                        height=250
                    )
                    od_plots.append(target_od)
            
            self.latest_od_df = self.latest_od_df.sort_values('timestamp')
            
            # Clean OD data - remove NaN, negative, and outlier values
            self.latest_od_df = self.latest_od_df[
                pd.notna(self.latest_od_df['od_value']) & 
                (self.latest_od_df['od_value'] >= 0) &
                (self.latest_od_df['od_value'] < 10)  # Assuming OD > 10 is an error
            ]
            
            if not self.latest_od_df.empty:
                latest_od = hv.Curve(
                    self.latest_od_df,
                    'timestamp',
                    'od_value',
                    label='Latest OD'
                ).opts(
                    color='blue',
                    line_width=2,
                    width=800,
                    height=250
                )
                od_plots.append(latest_od)
            
            if od_plots:
                od_plot = hv.Overlay(od_plots).opts(
                    legend_position='top_right',
                    xlabel='Time',
                    ylabel='OD Value',
                    title='OD Values'
                )
                
                # Add tap callback for OD plot
                od_plot = od_plot.opts(tools=['tap'])
                od_plot.opts(hooks=[self._create_tap_callback(od_plot)])
                
                self.od_plot.object = od_plot
            else:
                self.od_plot.object = hv.Text(0, 0, 'No valid OD data available').opts(width=800, height=250)
        else:
            self.od_plot.object = hv.Text(0, 0, 'No OD data available').opts(width=800, height=250)
        
        # Try to link plots using a version-compatible approach
        try:
            # Method 1: Try using hvplot's linking
            plots = [self.dilution_plot.object, self.od_plot.object, self.time_plot.object]
            hvplots = [p for p in plots if not isinstance(p, hv.element.Text)]
            
            if len(hvplots) > 1:
                self.show_success("Plots synchronized")
        except Exception as e:
            self.show_warning(f"Plot synchronization could not be enabled: {str(e)}")
        
        # Calculate statistics
        stats_html = self._calculate_stats()
        self.stats_output.object = stats_html
    
    def _calculate_stats(self):
        """
        Calculate statistics for different target_od regions.
        
        Analyzes OD values and dilution rates for periods with constant target OD.
        
        Returns
        -------
        str
            HTML-formatted statistics table
        """
        if (self.target_od_df.empty or self.latest_od_df.empty or 
            'od_value' not in self.latest_od_df.columns):
            return "<p>No OD data available for statistics.</p>"
        
        # Identify regions with constant target_od
        self.target_od_df = self.target_od_df.sort_values('timestamp')
        od_regions = []
        
        # Ensure there are at least 2 target OD events to define a region
        if len(self.target_od_df) >= 2:
            for i in range(len(self.target_od_df) - 1):
                start_time = self.target_od_df.iloc[i]['timestamp']
                end_time = self.target_od_df.iloc[i+1]['timestamp']
                target_od = self.target_od_df.iloc[i]['od_value']
                
                region = {
                    'start': start_time,
                    'end': end_time,
                    'target_od': target_od
                }
                
                # Get latest_od values in this time range
                region_od = self.latest_od_df[(self.latest_od_df['timestamp'] >= start_time) & 
                                             (self.latest_od_df['timestamp'] < end_time)]
                
                if not region_od.empty:
                    region['od_mean'] = region_od['od_value'].mean()
                    region['od_std'] = region_od['od_value'].std()
                    # Handle potential division by zero in CV calculation
                    if region['od_mean'] > 0:
                        region['od_cv'] = variation(region_od['od_value']) * 100  # CV in percentage
                    else:
                        region['od_cv'] = 0
                    
                    # Get dosing events in this region
                    if not self.dosing_events_df.empty:
                        region_dosing = self.dosing_events_df[(self.dosing_events_df['timestamp'] >= start_time) & 
                                                             (self.dosing_events_df['timestamp'] < end_time)]
                        
                        if len(region_dosing) > 1:
                            region['avg_dilution_rate'] = region_dosing['instant_dilution_rate'].mean()
                            region['dilution_std'] = region_dosing['instant_dilution_rate'].std()
                            region['avg_time_between_doses_min'] = region_dosing['time_diff_hours'].mean() * 60
                            region['time_between_doses_std_min'] = region_dosing['time_diff_hours'].std() * 60
                    
                    od_regions.append(region)
        
        # Create HTML table for stats
        if not od_regions:
            return "<p>No statistics available for OD regions.</p>"
        
        html = "<h3>Statistics by Target OD Region</h3>"
        html += """
        <table style="width:100%; border-collapse:collapse; margin-top:10px;">
          <thead>
            <tr style="background-color:#f2f2f2;">
              <th style="padding:8px; border:1px solid #ddd; text-align:left;">Target OD</th>
              <th style="padding:8px; border:1px solid #ddd; text-align:left;">Time Period</th>
              <th style="padding:8px; border:1px solid #ddd; text-align:left;">Avg OD</th>
              <th style="padding:8px; border:1px solid #ddd; text-align:left;">OD CV%</th>
              <th style="padding:8px; border:1px solid #ddd; text-align:left;">Avg Dilution (h⁻¹)</th>
              <th style="padding:8px; border:1px solid #ddd; text-align:left;">Time Between Doses (min)</th>
            </tr>
          </thead>
          <tbody>
        """
        
        for region in od_regions:
            # Safe formatting with defaults for missing values
            html += f"""
            <tr>
              <td style="padding:8px; border:1px solid #ddd;">{region['target_od']:.3f}</td>
              <td style="padding:8px; border:1px solid #ddd;">{region['start'].strftime('%Y-%m-%d %H:%M')} to {region['end'].strftime('%Y-%m-%d %H:%M')}</td>
              <td style="padding:8px; border:1px solid #ddd;">{region.get('od_mean', 'N/A'):.3f} ± {region.get('od_std', 0):.3f}</td>
              <td style="padding:8px; border:1px solid #ddd;">{region.get('od_cv', 'N/A'):.1f}%</td>
              <td style="padding:8px; border:1px solid #ddd;">{region.get('avg_dilution_rate', 'N/A') if isinstance(region.get('avg_dilution_rate'), (int, float)) else 'N/A'}{f" ± {region.get('dilution_std', 0):.3f}" if isinstance(region.get('dilution_std'), (int, float)) else ''}</td>
              <td style="padding:8px; border:1px solid #ddd;">{region.get('avg_time_between_doses_min', 'N/A') if isinstance(region.get('avg_time_between_doses_min'), (int, float)) else 'N/A'}{f" ± {region.get('time_between_doses_std_min', 0):.1f}" if isinstance(region.get('time_between_doses_std_min'), (int, float)) else ''}</td>
            </tr>
            """
        
        html += """
          </tbody>
        </table>
        """
        
        return html
    
    def view(self):
        """
        Return the main layout for display.
        
        Returns
        -------
        panel.layout.Column
            The main application layout
        """
        return self.main_layout

# Create the application
pioreactor_analysis = PioreactorAnalysis()

# Create a Panel server
app = pn.panel(pioreactor_analysis.view())

# Server
if __name__ == '__main__':
    app.show(port=5006)
