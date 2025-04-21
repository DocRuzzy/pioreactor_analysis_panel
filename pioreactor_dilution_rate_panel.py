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

Author: Russell Kirk Pirlo Improved by Claude
Date: April 21, 2025
"""

import pandas as pd
import numpy as np
import hvplot.pandas
import holoviews as hv
import panel as pn
import param
from bokeh.models import CustomJSTickFormatter, ColumnDataSource, Range1d, HoverTool
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
        Number of events to include in moving average calculations (default: 30)
    """
    
    reactor_volume = param.Number(14.0, bounds=(1, 100), step=0.1, 
                                  doc="Reactor volume in mL")
    moving_avg_window = param.Integer(30, bounds=(1, 50), step=1, 
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
                
                # Sample data rows for debugging
                debug_info += "\nSample rows:\n"
                for i, row in df.head(3).iterrows():
                    debug_info += f"Row {i}:\n"
                    debug_info += f"  event_name: {row.get('event_name', 'N/A')}\n"
                    debug_info += f"  message: {row.get('message', 'N/A')}\n"
                    debug_info += f"  data: {row.get('data', 'N/A')}\n"
                
                # Process the data
                self._process_data(df)
                
                # Add more debug info about processed data
                debug_info += f"\nAfter processing:\n"
                debug_info += f"Dosing events: {len(self.dosing_events_df)} rows\n"
                debug_info += f"Target OD events: {len(self.target_od_df)} rows\n"
                debug_info += f"Latest OD events: {len(self.latest_od_df)} rows\n"
                
                if not self.latest_od_df.empty and 'od_value' in self.latest_od_df.columns:
                    debug_info += f"Latest OD values: {self.latest_od_df['od_value'].min():.3f} - {self.latest_od_df['od_value'].max():.3f}\n"
                    sample_values = self.latest_od_df['od_value'].head(5).tolist()
                    debug_info += f"Sample Latest OD values: {sample_values}\n"
                
                if not self.target_od_df.empty and 'od_value' in self.target_od_df.columns:
                    debug_info += f"Target OD values: {self.target_od_df['od_value'].min():.3f} - {self.target_od_df['od_value'].max():.3f}\n"
                    sample_values = self.target_od_df['od_value'].head(5).tolist()
                    debug_info += f"Sample Target OD values: {sample_values}\n"
                
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
        # Convert timestamps to datetime objects (and normalize timezone)
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
        
        # Also convert the timestamp_localtime column if it exists
        if 'timestamp_localtime' in df.columns:
            df['timestamp_localtime'] = pd.to_datetime(df['timestamp_localtime']).dt.tz_localize(None)
            # Use timestamp_localtime for elapsed time calculation
            start_time = df['timestamp_localtime'].min()
            df['elapsed_hours'] = (df['timestamp_localtime'] - start_time).dt.total_seconds() / 3600
        else:
            # Fallback to timestamp if timestamp_localtime doesn't exist
            start_time = df['timestamp'].min()
            df['elapsed_hours'] = (df['timestamp'] - start_time).dt.total_seconds() / 3600
        
        # Extract dosing events (DilutionEvent)
        self.dosing_events_df = df[df['event_name'].str.contains('dilution', case=False, na=False)].copy()
        
        if self.dosing_events_df.empty:
            print("Warning: No dilution events found in the data.")
            return
        
        # Extract OD data using the specialized approach
        self._extract_od_data_specialized(df)
        
        # Parse volume from dosing events with debugging
        print(f"\nFound {len(self.dosing_events_df)} dilution events")
        
        # Sample events before extraction to check format
        print("\nSample events before volume extraction:")
        sample_data = self.dosing_events_df[['timestamp', 'data', 'message']].head(2)
        for _, row in sample_data.iterrows():
            print(f"Timestamp: {row['timestamp']}")
            print(f"Data: {row['data']}")
            print(f"Message: {row['message']}")
            print("---")
        
        # Apply volume extraction
        self.dosing_events_df['volume'] = self.dosing_events_df.apply(self._extract_volume, axis=1)
        self.dosing_events_df['volume'] = pd.to_numeric(self.dosing_events_df['volume'], errors='coerce')
        
        # Show statistics about extracted volumes
        extracted_count = self.dosing_events_df['volume'].notna().sum()
        print(f"\nSuccessfully extracted volumes for {extracted_count}/{len(self.dosing_events_df)} events ({extracted_count/len(self.dosing_events_df)*100:.1f}%)")
        
        unique_volumes = sorted(self.dosing_events_df['volume'].dropna().unique())
        print(f"Unique volume values found: {unique_volumes}")
        
        # Check for volume changes
        self.dosing_events_df.sort_values('timestamp', inplace=True)
        self.dosing_events_df['volume_changed'] = self.dosing_events_df['volume'] != self.dosing_events_df['volume'].shift(1)
        
        # Sample of first few events with volumes
        volume_sample = self.dosing_events_df.head(5)[['timestamp', 'volume']]
        print("\nSample of first few events with extracted volumes:")
        print(volume_sample)
        
        # Show where volume changes occur
        volume_changes = self.dosing_events_df[self.dosing_events_df['volume_changed'] == True].copy()
        if not volume_changes.empty:
            print(f"\nDetected {len(volume_changes)} volume changes. First few changes:")
            changes_sample = volume_changes[['timestamp', 'volume']].head(5)
            print(changes_sample)
        else:
            print("\nNo volume changes detected in the data.")
        
    def _extract_od_data_specialized(self, df):
        """
        Specialized extraction method based on the format of the provided data.
        
        This method assumes both latest_od and target_od values might be in the same row,
        either in the data JSON field or in the message field.
        
        Parameters
        ----------
        df : pandas.DataFrame
            The raw data frame
        """
        # Initialize empty dataframes for target and latest OD
        target_od_data = []
        latest_od_data = []
        
        # Debug collection
        debug_info = []
        
        # Count of rows processed for each method
        data_json_count = 0
        message_regex_count = 0
        
        # Process each row that may contain OD information
        for idx, row in df.iterrows():
            # Use timestamp_localtime if available, otherwise use timestamp
            timestamp_to_use = row['timestamp_localtime'] if 'timestamp_localtime' in row and pd.notna(row['timestamp_localtime']) else row['timestamp']
            
            # Try to extract from data JSON field (most reliable)
            if pd.notna(row.get('data')) and isinstance(row['data'], str):
                try:
                    data_dict = json.loads(row['data'])
                    
                    # Extract target_od if available
                    if 'target_od' in data_dict and isinstance(data_dict['target_od'], (int, float)):
                        target_od_value = float(data_dict['target_od'])
                        target_od_data.append({
                            'timestamp': timestamp_to_use,
                            'od_value': target_od_value
                        })
                        data_json_count += 1
                        debug_info.append(f"Row {idx}: Found target_od = {target_od_value} in data JSON")
                    
                    # Extract latest_od if available
                    if 'latest_od' in data_dict and isinstance(data_dict['latest_od'], (int, float)):
                        latest_od_value = float(data_dict['latest_od'])
                        latest_od_data.append({
                            'timestamp': timestamp_to_use,
                            'od_value': latest_od_value
                        })
                        data_json_count += 1
                        debug_info.append(f"Row {idx}: Found latest_od = {latest_od_value} in data JSON")
                except Exception as e:
                    debug_info.append(f"Row {idx}: Error parsing data JSON: {str(e)}")
            
            # As a fallback, try extracting from message field using regex
            if pd.notna(row.get('message')) and isinstance(row['message'], str):
                # Try to extract latest OD from message
                latest_match = re.search(r'Latest\s*OD\s*=\s*([0-9.]+)', row['message'])
                if latest_match:
                    latest_od_value = float(latest_match.group(1))
                    latest_od_data.append({
                        'timestamp': timestamp_to_use,
                        'od_value': latest_od_value
                    })
                    message_regex_count += 1
                    debug_info.append(f"Row {idx}: Found latest_od = {latest_od_value} from message regex")
                
                # Try to extract target OD from message
                target_match = re.search(r'Target\s*OD\s*=\s*([0-9.]+)', row['message'])
                if target_match:
                    target_od_value = float(target_match.group(1))
                    target_od_data.append({
                        'timestamp': timestamp_to_use,
                        'od_value': target_od_value
                    })
                    message_regex_count += 1
                    debug_info.append(f"Row {idx}: Found target_od = {target_od_value} from message regex")
        
        # Remove duplicates (same timestamp and value)
        if target_od_data:
            target_od_df = pd.DataFrame(target_od_data)
            target_od_df = target_od_df.drop_duplicates(subset=['timestamp', 'od_value'])
            self.target_od_df = target_od_df
        else:
            self.target_od_df = pd.DataFrame()
        
        if latest_od_data:
            latest_od_df = pd.DataFrame(latest_od_data)
            latest_od_df = latest_od_df.drop_duplicates(subset=['timestamp', 'od_value'])
            self.latest_od_df = latest_od_df
        else:
            self.latest_od_df = pd.DataFrame()
        
        # Add summary to debug info
        debug_info.insert(0, f"OD Data Extraction Summary:")
        debug_info.insert(1, f"- Extracted {len(target_od_data)} target OD values and {len(latest_od_data)} latest OD values")
        debug_info.insert(2, f"- JSON data extraction: {data_json_count} values")
        debug_info.insert(3, f"- Message regex extraction: {message_regex_count} values")
        debug_info.insert(4, f"- After deduplication: {len(self.target_od_df)} target OD and {len(self.latest_od_df)} latest OD values")
        
        # Show detailed OD extraction info in debug tab
        self.show_debug("OD Data Extraction:\n" + "\n".join(debug_info[:50]) + 
                    ("\n..." if len(debug_info) > 50 else ""))
        

    def _extract_volume(self, row):
        """
        Extract volume information from dosing events.
        
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
            # Try data column first (JSON format)
            if 'data' in row and pd.notna(row['data']):
                try:
                    data_dict = json.loads(row['data'])
                    if 'volume' in data_dict:
                        volume_value = float(data_dict['volume'])
                        return volume_value
                except json.JSONDecodeError as e:
                    print(f"JSON decode error for data: {row['data'][:50]}... Error: {e}")
                except Exception as e:
                    print(f"Error extracting volume from data: {e}")
            
            # Try message column as fallback
            if 'message' in row and pd.notna(row['message']):
                # Look for patterns like "cycled 0.50 mL"
                patterns = [
                    r'cycled\s+([0-9.]+)\s*mL',                              # "cycled X mL" pattern
                    r'(?:volume|added|removed|cycled)[:\s]+([0-9.]+)\s*m?L',  # General pattern
                    r'([0-9.]+)\s*m?L',                                      # Just a number followed by mL
                ]
                
                for pattern in patterns:
                    match = re.search(pattern, str(row['message']), re.IGNORECASE)
                    if match:
                        volume_value = float(match.group(1))
                        print(f"Extracted volume {volume_value} from message text using pattern: {pattern}")
                        return volume_value
        except Exception as e:
            print(f"Unexpected error extracting volume: {e}")
        
        return None
        
    def _calculate_volume_changes(self):
        """Detect points where cycled volumes change"""
        # Make sure volumes are numeric
        if 'volume' not in self.dosing_events_df.columns:
            return pd.DataFrame()
        
        # Detect points where volume changes compared to previous point
        self.dosing_events_df['volume_change'] = self.dosing_events_df['volume'] != self.dosing_events_df['volume'].shift(1)
        
        # Get rows where volume changes occur
        volume_change_df = self.dosing_events_df[self.dosing_events_df['volume_change'] == True].copy()
        
        # Debug output
        if not volume_change_df.empty:
            print("\nDEBUG - VOLUME CHANGES DETECTED:")
            print(volume_change_df[['timestamp', 'volume', 'capped_time_diff']].head(10))
        
        return volume_change_df
        
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
            # Use time_display if available, otherwise format the timestamp
            if 'time_display' in bm:
                time_str = bm['time_display']
            elif isinstance(bm['timestamp'], pd.Timestamp):
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
                                 on_click=lambda event, bm=bm: self.show_success(f"Copied: Time: {time_str}, Value: {bm['value']}")),
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
        # Fixed callback signature for HoloViews 1.20.2
        def tap_callback(plot_id, element):
            if hasattr(plot_id, 'x') and hasattr(plot_id, 'y'):
                self._add_bookmark(plot_id.x, plot_id.y)
                self.show_success(f"Bookmark added at x={plot_id.x}, y={plot_id.y}")
        
        return tap_callback
        
    def _format_time_ticks(self, reference_time=None):
        # reference_time: a pd.Timestamp
        if reference_time is None:
            if self.dosing_events_df.empty:
                min_time = pd.Timestamp.now()
            else:
                min_time = self.dosing_events_df['timestamp'].min()
        else:
            min_time = reference_time

        if hasattr(min_time, 'tz') and min_time.tz is not None:
            min_time = min_time.tz_localize(None)
        min_time_ms = int(min_time.timestamp() * 1000)

        source = ColumnDataSource(data={'start_time_ms': [min_time_ms]})

        code = """
            // tick is elapsed_hours (float)
            const start_time_ms = source.data.start_time_ms[0];
            const tick_ms = start_time_ms + tick * 3600 * 1000;
            const date = new Date(tick_ms);

            // Format date as MM/DD HH:MM
            const monthDay = (date.getMonth() + 1).toString().padStart(2, '0') + '/' +
                            date.getDate().toString().padStart(2, '0');
            const timeStr = date.getHours().toString().padStart(2, '0') + ':' +
                            date.getMinutes().toString().padStart(2, '0');

            // Format elapsed hours
            const elapsed = tick.toFixed(1);

            return `${monthDay} ${timeStr}\\n(${elapsed}h)`;
        """

        formatter = CustomJSTickFormatter(code=code, args={'source': source})
        return formatter

    def _update_plots(self):
        """
        Update all plots with current data and parameters using a direct Bokeh range sharing approach.
        """
        # Check if we have data
        if self.dosing_events_df.empty or not all(col in self.dosing_events_df.columns for col in ['timestamp', 'volume']):
            # Display empty placeholders
            self.dilution_plot.object = hv.Text(0, 0, 'No valid dosing data found').opts(width=800, height=300)
            self.od_plot.object = hv.Text(0, 0, 'No valid OD data found').opts(width=800, height=250)
            self.time_plot.object = hv.Text(0, 0, 'No valid time data found').opts(width=800, height=250)
            self.stats_output.object = "<p>No data available for plotting or statistics.</p>"
            return
        
        # Create a shared x-axis range using Bokeh Range1d
        shared_x_range = Range1d()
        
        # Get a consistent reference time for time formatting
        if 'timestamp_localtime' in self.dosing_events_df.columns:
            start_time = self.dosing_events_df['timestamp_localtime'].min()
        else:
            start_time = self.dosing_events_df['timestamp'].min()
            
        # Create time formatter with consistent reference time
        time_formatter = self._format_time_ticks(reference_time=start_time)
        
        # Helper function to apply time formatter to plots
        def apply_time_formatter(plot, element):
            if 'xaxis' in plot.handles:
                plot.handles['xaxis'].formatter = time_formatter
        
        # Generate individual plots
        dilution_plot = self._update_dilution_plots()
        od_plot = self._update_od_plots() 
        time_plot = self._update_time_plots()
        
        # Apply shared x-range and formatter to each plot if they exist
        if dilution_plot is not None:
            dilution_plot = dilution_plot.opts(
                hooks=[apply_time_formatter],
                backend_opts={'x_range': shared_x_range}
            )
            self.dilution_plot.object = dilution_plot
        
        if od_plot is not None:
            od_plot = od_plot.opts(
                hooks=[apply_time_formatter],
                backend_opts={'x_range': shared_x_range}
            )
            self.od_plot.object = od_plot
        
        if time_plot is not None:
            time_plot = time_plot.opts(
                hooks=[apply_time_formatter],
                backend_opts={'x_range': shared_x_range}
            )
            self.time_plot.object = time_plot
        
        # Update statistics
        stats_html = self._calculate_stats()
        self.stats_output.object = stats_html

    def _update_dilution_plots(self):
        """Update dilution rate and timing plots"""
        if self.dosing_events_df.empty:
            # No data to plot for dilution rate
            self.dilution_plot.object = hv.Text(0, 0, 'No dilution event data available').opts(width=800, height=300)
            self.time_plot.object = hv.Text(0, 0, 'No dilution event data available').opts(width=800, height=250)
            return
        
        # Sort by timestamp
        if 'timestamp_localtime' in self.dosing_events_df.columns:
            self.dosing_events_df = self.dosing_events_df.sort_values('timestamp_localtime')
            # Store start time for axis formatting
            start_time = self.dosing_events_df['timestamp_localtime'].min()
        else:
            self.dosing_events_df = self.dosing_events_df.sort_values('timestamp')
            # Store start time for axis formatting
            start_time = self.dosing_events_df['timestamp'].min()
        
        # Filter out rows with missing volume
        self.dosing_events_df = self.dosing_events_df[pd.notna(self.dosing_events_df['volume'])]
        
        if len(self.dosing_events_df) <= 1:
            self.dilution_plot.object = hv.Text(0, 0, 'Insufficient dilution events to calculate rates').opts(width=800, height=300)
            self.time_plot.object = hv.Text(0, 0, 'Insufficient dilution events to calculate rates').opts(width=800, height=250)
            return
        
        # Ensure elapsed_hours column exists 
        if 'elapsed_hours' not in self.dosing_events_df.columns:
            if 'timestamp_localtime' in self.dosing_events_df.columns:
                self.dosing_events_df['elapsed_hours'] = (self.dosing_events_df['timestamp_localtime'] - start_time).dt.total_seconds() / 3600
            else:
                self.dosing_events_df['elapsed_hours'] = (self.dosing_events_df['timestamp'] - start_time).dt.total_seconds() / 3600
        
        # Calculate time between doses
        if 'timestamp_localtime' in self.dosing_events_df.columns:
            self.dosing_events_df['next_timestamp'] = self.dosing_events_df['timestamp_localtime'].shift(-1)
            self.dosing_events_df['time_diff_hours'] = (self.dosing_events_df['next_timestamp'] - 
                                                    self.dosing_events_df['timestamp_localtime']).dt.total_seconds() / 3600
        else:
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
        
        # HoloViews plots with tooltips - using elapsed_hours for consistency
        dilution_scatter = hv.Scatter(
            self.dosing_events_df, 
            'elapsed_hours', 
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
            'elapsed_hours', 
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
        
        # Time between doses plot - using elapsed_hours for x-axis
        time_line = hv.Curve(
            self.dosing_events_df,
            'elapsed_hours',
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
            'elapsed_hours',
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
            'elapsed_hours',
            'capped_time_diff'
        ).opts(
            color='red',
            marker='circle',
            size=8,
            tools=['hover'],
            width=800,
            height=250
        )
        
        # Use the new formatting method with the start_time
        formatter = self._format_time_ticks(reference_time=start_time)
        
        # Combine plots with the custom formatter
        dilution_plot = (dilution_scatter * dilution_line).opts(
            legend_position='top_right',
            xlabel='Time (elapsed hours / date)',
            ylabel='Dilution Rate (h⁻¹)'
        )
        
        time_plot = (time_line * normal_scatter * outlier_scatter).opts(
            legend_position='top_right',
            xlabel='Time (elapsed hours / date)',
            ylabel='Minutes Between Doses',
            ylim=(0, 65)
        )
        
        # Add tap callbacks for interactions
        dilution_plot = dilution_plot.opts(tools=['tap'])
        dilution_plot.opts(hooks=[self._create_tap_callback(dilution_plot)])
        
        time_plot = time_plot.opts(tools=['tap'])
        time_plot.opts(hooks=[self._create_tap_callback(time_plot)])
        
        # Define a function to set the x-axis formatter
        def set_xaxis_formatter(plot, element):
            plot.handles['xaxis'].formatter = formatter  # Apply only the formatter
        
        # Update plot panes with custom axis formatting
        dilution_plot = dilution_plot.opts(
            hooks=[set_xaxis_formatter]
        )
        
        time_plot = time_plot.opts(
            hooks=[set_xaxis_formatter]
        )
        
        # Update plot panes
        self.dilution_plot.object = dilution_plot
        return dilution_plot
    
    
    def _update_od_plots(self):
        """Update OD plots with both target and latest OD values"""
        od_plots = []
        
        # Check if we have data for either target or latest OD
        has_target_od = not self.target_od_df.empty and 'od_value' in self.target_od_df.columns
        has_latest_od = not self.latest_od_df.empty and 'od_value' in self.latest_od_df.columns
        
        # Determine start time - use consistent start time across all plots
        if not self.dosing_events_df.empty:
            # Prioritize timestamp_localtime if available
            if 'timestamp_localtime' in self.dosing_events_df.columns:
                start_time = self.dosing_events_df['timestamp_localtime'].min()
            else:
                start_time = self.dosing_events_df['timestamp'].min()
        elif has_target_od:
            # Prioritize timestamp_localtime if available
            if 'timestamp_localtime' in self.target_od_df.columns:
                start_time = self.target_od_df['timestamp_localtime'].min()
            else:
                start_time = self.target_od_df['timestamp'].min()
        elif has_latest_od:
            # Prioritize timestamp_localtime if available
            if 'timestamp_localtime' in self.latest_od_df.columns:
                start_time = self.latest_od_df['timestamp_localtime'].min()
            else:
                start_time = self.latest_od_df['timestamp'].min()
        else:
            # No data available
            self.od_plot.object = hv.Text(0, 0, 'No OD data available').opts(width=800, height=250)
            return
        
        # Use the new formatting method with the start_time
        formatter = self._format_time_ticks(reference_time=start_time)
        
        if has_target_od or has_latest_od:
            # Process target OD data
            if has_target_od:
                # Add elapsed_hours column based on timestamp_localtime if available
                if 'timestamp_localtime' in self.target_od_df.columns:
                    self.target_od_df['elapsed_hours'] = (self.target_od_df['timestamp_localtime'] - start_time).dt.total_seconds() / 3600
                    # Sort target OD data
                    self.target_od_df = self.target_od_df.sort_values('timestamp_localtime')
                else:
                    self.target_od_df['elapsed_hours'] = (self.target_od_df['timestamp'] - start_time).dt.total_seconds() / 3600
                    # Sort target OD data
                    self.target_od_df = self.target_od_df.sort_values('timestamp')
                
                # Clean target OD data
                self.target_od_df = self.target_od_df[
                    pd.notna(self.target_od_df['od_value']) & 
                    (self.target_od_df['od_value'] >= 0) &
                    (self.target_od_df['od_value'] < 10)  # Assuming OD > 10 is an error
                ]
                
                # Create target OD plot using elapsed_hours
                if not self.target_od_df.empty:
                    target_od = hv.Curve(
                        self.target_od_df,
                        'elapsed_hours',  # Use elapsed_hours
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
            
            # Process latest OD data
            if has_latest_od:
                # Add elapsed_hours column based on timestamp_localtime if available
                if 'timestamp_localtime' in self.latest_od_df.columns:
                    self.latest_od_df['elapsed_hours'] = (self.latest_od_df['timestamp_localtime'] - start_time).dt.total_seconds() / 3600
                    # Sort latest OD data
                    self.latest_od_df = self.latest_od_df.sort_values('timestamp_localtime')
                else:
                    self.latest_od_df['elapsed_hours'] = (self.latest_od_df['timestamp'] - start_time).dt.total_seconds() / 3600
                    # Sort latest OD data
                    self.latest_od_df = self.latest_od_df.sort_values('timestamp')
                
                # Clean latest OD data
                self.latest_od_df = self.latest_od_df[
                    pd.notna(self.latest_od_df['od_value']) & 
                    (self.latest_od_df['od_value'] >= 0) &
                    (self.latest_od_df['od_value'] < 10)  # Assuming OD > 10 is an error
                ]
                
                # Create latest OD plot using elapsed_hours
                if not self.latest_od_df.empty:
                    latest_od = hv.Curve(
                        self.latest_od_df,
                        'elapsed_hours',  # Use elapsed_hours
                        'od_value',
                        label='Latest OD'
                    ).opts(
                        color='blue',
                        line_width=2,
                        width=800,
                        height=250
                    )
                    od_plots.append(latest_od)
            
            # Create combined OD plot
            if od_plots:
                od_plot = hv.Overlay(od_plots).opts(
                    legend_position='top_right',
                    xlabel='Time (elapsed hours / date)',
                    ylabel='OD Value',
                    title='OD Values'
                )
                
                # Add tap callback for OD plot
                od_plot = od_plot.opts(tools=['tap'])
                od_plot.opts(hooks=[self._create_tap_callback(od_plot)])
                
                # Define a function to set the x-axis formatter
                def set_xaxis_formatter(plot, element):
                    plot.handles['xaxis'].formatter = formatter  # Apply only the formatter
                
                # Add custom formatter to x-axis
                od_plot = od_plot.opts(
                    hooks=[set_xaxis_formatter]
                )
                
                # Update plot pane
                self.od_plot.object = od_plot
                return od_plot
            else:
                self.od_plot.object = hv.Text(0, 0, 'No OD data available').opts(width=800, height=250)
        else:
            self.od_plot.object = hv.Text(0, 0, 'No OD data available').opts(width=800, height=250)

    def _update_time_plots(self):
        """Update the inter-dosing period plot with volume change markers"""
        # Check if data exists
        if self.dosing_events_df.empty or 'capped_time_diff' not in self.dosing_events_df.columns:
            self.time_plot.object = hv.Text(0, 0, 'No Inter-Dosing Time data').opts(width=800, height=250)
            return None
        
        # Get dataframes for normal points and outliers
        plot_df_time = self.dosing_events_df.dropna(subset=['elapsed_hours', 'capped_time_diff'])
        
        # Split into normal and outlier points
        if 'is_outlier' in self.dosing_events_df.columns:
            normal_df = plot_df_time[~plot_df_time['is_outlier']]
            outlier_df = plot_df_time[plot_df_time['is_outlier']]
        else:
            normal_df = plot_df_time
            outlier_df = pd.DataFrame()  # Empty DataFrame if no outliers marked
        
        # Create plot components
        time_plots_overlay = []
        
        # Add main time curve
        if not plot_df_time.empty:
            time_line = hv.Curve(
                plot_df_time, 'elapsed_hours', 'capped_time_diff', label='Minutes Between Doses'
            ).opts(color='blue', line_width=1, tools=['hover'])
            time_plots_overlay.append(time_line)
        
        # Add normal points
        if not normal_df.empty:
            normal_scatter = hv.Scatter(
                normal_df, 'elapsed_hours', 'capped_time_diff'
            ).opts(color='blue', size=6, tools=['hover'])
            time_plots_overlay.append(normal_scatter)
        
        # Add outlier points
        if not outlier_df.empty:
            outlier_scatter = hv.Scatter(
                outlier_df, 'elapsed_hours', 'capped_time_diff'
            ).opts(color='red', marker='circle', size=8, tools=['hover'])
            time_plots_overlay.append(outlier_scatter)
        
        # Add volume change markers
        volume_change_df = self.dosing_events_df[
            (self.dosing_events_df['volume_changed'] == True) & 
            ~self.dosing_events_df['capped_time_diff'].isna()
        ].copy()
        
        if not volume_change_df.empty:
            # Add markers at volume change points
            volume_markers = hv.Scatter(
                volume_change_df,
                'elapsed_hours',  # Changed from 'timestamp'
                'capped_time_diff'
            ).opts(
                color='green',
                size=12,
                marker='triangle',
                line_color='black',
                line_width=1,
                tools=['hover']
            )
            time_plots_overlay.append(volume_markers)
            
            # Add text labels showing the new volume
            for idx, row in volume_change_df.iterrows():
                if pd.notna(row['volume']):
                    label = hv.Text(
                        row['elapsed_hours'],  # Changed from 'timestamp'
                        min(row['capped_time_diff'] + 5, 60),
                        f"Volume: {row['volume']} mL"
                    ).opts(text_color='green', text_font_size='10pt', text_font_style='bold')
                    time_plots_overlay.append(label)
        
        # Combine all plot elements
        time_plot_combined = hv.Overlay(time_plots_overlay).opts(
            legend_position='top_right',
            xlabel='Time (elapsed hours)',
            ylabel='Minutes Between Doses (capped at 60)',
            title='Inter-Dosing Period',
            width=800,
            height=250,
            ylim=(0, 65),
            tools=['pan', 'wheel_zoom', 'box_zoom', 'reset', 'tap']
        ) if time_plots_overlay else hv.Text(0, 0, 'No Inter-Dosing Time data').opts(width=800, height=250)
        
        # Add tap callback
        if time_plots_overlay:
            time_plot_combined = time_plot_combined.opts(hooks=[self._create_tap_callback(time_plot_combined)])
        
        return time_plot_combined

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
            'od_value' not in self.latest_od_df.columns or
            'od_value' not in self.target_od_df.columns):
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
