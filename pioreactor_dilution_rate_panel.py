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

Author: Russell Kirk Pirlo wiht aid from Claude
Date: April 21, 2025
"""

import pandas as pd
import numpy as np
import hvplot.pandas
import holoviews as hv
import panel as pn
import param
from bokeh.models import CustomJSTickFormatter, ColumnDataSource, HoverTool
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
                                  doc="Reactor volume in mL. This is used to calculate dilution rates from media volumes. Typical Pioreactor volume is 14 mL.")
    moving_avg_window = param.Integer(30, bounds=(1, 50), step=1, 
                                      doc="Moving average window (# of events). Larger windows provide smoother curves but may obscure rapid changes. Start with 20-30 for typical experiments.")
    time_axis_mode = param.Selector(default="Elapsed Time (hours)", 
                                    objects=["Elapsed Time (hours)", "Actual Time (datetime)", "Both (dual axis)"],
                                    doc="Time axis display mode. 'Elapsed Time' shows hours from experiment start. 'Actual Time' shows real timestamps. 'Both' displays elapsed time with actual time on top axis.")
    
    # Plot aesthetic parameters
    plot_width = param.Integer(1200, bounds=(400, 3000), step=50, 
                              doc="Width of plots in pixels. Larger widths show more detail but require more screen space.")
    plot_height = param.Integer(400, bounds=(200, 1000), step=50, 
                               doc="Height of plots in pixels. Taller plots provide better vertical resolution.")
    font_size = param.Integer(12, bounds=(8, 24), step=1, 
                             doc="Base font size for plot labels and titles in points.")
    color_scheme = param.Selector(default='default', 
                                 objects=['default', 'wong', 'grayscale'],
                                 doc="Color scheme for plots. 'wong' is colorblind-friendly, 'grayscale' for printing.")
    plot_dpi = param.Integer(150, bounds=(50, 600), step=50, 
                            doc="Resolution (DPI) for exported plot images. Higher values produce sharper images but larger files.")
    
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
        
        # Loading spinner (inactive by default)
        self.loading_spinner = pn.indicators.LoadingSpinner(value=False, width=24, height=24)
        
        # Data preview pane
        self.data_preview = pn.pane.HTML("", styles={'overflow-x': 'auto'})
        
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
        
        # Export statistics button
        self.export_stats_button = pn.widgets.Button(name='📥 Export Statistics to CSV', button_type='success')
        self.export_stats_button.on_click(self._export_statistics_callback)
        
        # Batch export all button
        self.export_all_button = pn.widgets.Button(name='📦 Export All (Plots + Data)', button_type='success', width=250)
        self.export_all_button.on_click(self._export_all_callback)
        
        # Export plot buttons
        self.export_dilution_plot_button = pn.widgets.Button(name='📊 Export Dilution Rate Plot', button_type='default', width=200)
        self.export_dilution_plot_button.on_click(self._export_dilution_plot_callback)
        
        self.export_od_plot_button = pn.widgets.Button(name='📊 Export OD Plot', button_type='default', width=200)
        self.export_od_plot_button.on_click(self._export_od_plot_callback)
        
        self.export_time_plot_button = pn.widgets.Button(name='📊 Export Time Plot', button_type='default', width=200)
        self.export_time_plot_button.on_click(self._export_time_plot_callback)
        
        # Session save/load buttons
        self.save_session_button = pn.widgets.Button(name='💾 Save Session', button_type='primary', width=150)
        self.save_session_button.on_click(self.save_session_callback)
        
        self.load_session_input = pn.widgets.FileInput(accept='.json', multiple=False, name='📂 Load Session')
        self.load_session_input.param.watch(self.load_session_callback, 'value')
        
        self.session_status_text = pn.pane.Markdown("", styles={'font-size': '12px', 'margin-top': '10px'})
        
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
                    self.param.time_axis_mode,
                    pn.pane.Markdown("---"),
                    pn.pane.Markdown("### Plot Appearance"),
                    self.param.plot_width,
                    self.param.plot_height,
                    self.param.font_size,
                    self.param.color_scheme,
                    self.param.plot_dpi,
                    self.update_button,
                    width=400
                ),
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
                    self.session_status_text,
                    width=400
                )
            ),
            pn.Tabs(
                ('Dilution Analysis', pn.Column(
                    pn.Row(self.export_dilution_plot_button, align='end'),
                    self.dilution_plot,
                    pn.Row(self.export_od_plot_button, align='end'),
                    self.od_plot,
                    pn.Row(self.export_time_plot_button, align='end'),
                    self.time_plot
                )),
                ('Statistics', pn.Column(
                    pn.Row(
                        self.export_stats_button,
                        self.export_all_button
                    ),
                    self.stats_output
                )),
                ('Debug', self.debug_message)
            ),
            pn.Row(
                pn.Column(
                    self.bookmarks_title,
                    self.bookmarks_container
                )
            )
        )
        
        # Add watchers for plot aesthetic parameters to auto-update
        self.param.watch(self._on_aesthetic_change, ['plot_width', 'plot_height', 'font_size', 'color_scheme'])
    
    def _on_aesthetic_change(self, event):
        """Callback when plot aesthetic parameters change - auto-update plots"""
        # Only update if we have data
        if not self.dosing_events_df.empty or not self.target_od_df.empty or not self.latest_od_df.empty:
            self._update_plots()
    
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
            # Show loading spinner during file processing
            self.loading_spinner.value = True
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
                
                # Display data preview
                self._update_data_preview(df)
                
                # Update plots
                self._update_plots()
                
                # Show success message
                self.show_success(f"File {self.file_input.filename} uploaded successfully!")
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                self.show_error(f"Error processing file: {str(e)}")
                self.show_debug(f"Error details:\n{tb}")
            finally:
                # Hide loading spinner
                self.loading_spinner.value = False
        else:
            self.show_warning("Please upload a CSV file.")
            self.loading_spinner.value = False
    

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
        # Show spinner while updating plots
        self.loading_spinner.value = True
        try:
            self._update_plots()
        finally:
            self.loading_spinner.value = False
    
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
        Update all plots with current data and parameters using range linking.
        """
        # Check if we have data
        if self.dosing_events_df.empty or not all(col in self.dosing_events_df.columns for col in ['timestamp', 'volume']):
            # Display empty placeholders
            self.dilution_plot.object = hv.Text(0, 0, 'No valid dosing data found').opts(width=800, height=300)
            self.od_plot.object = hv.Text(0, 0, 'No valid OD data found').opts(width=800, height=250)
            self.time_plot.object = hv.Text(0, 0, 'No valid time data found').opts(width=800, height=250)
            self.stats_output.object = "<p>No data available for plotting or statistics.</p>"
            return
        
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
        
        # Store plots for range linking
        plots_to_link = []
        
        # Apply formatter to each plot if they exist and collect for linking
        if dilution_plot is not None:
            dilution_plot = dilution_plot.opts(hooks=[apply_time_formatter])
            plots_to_link.append(dilution_plot)
            self.dilution_plot.object = dilution_plot
        
        if od_plot is not None:
            od_plot = od_plot.opts(hooks=[apply_time_formatter])
            plots_to_link.append(od_plot)
            self.od_plot.object = od_plot
        
        if time_plot is not None:
            time_plot = time_plot.opts(hooks=[apply_time_formatter])
            plots_to_link.append(time_plot)
            self.time_plot.object = time_plot
        
        # Link the x-axis ranges of all plots
        if len(plots_to_link) >= 2:
            # Create RangeToolLink to sync x-axis ranges
            for i in range(len(plots_to_link) - 1):
                hv.plotting.links.RangeToolLink(plots_to_link[i], plots_to_link[i + 1], axes=['x'])
        
        # Update statistics
        stats_html = self._calculate_stats()
        self.stats_output.object = stats_html
    
    def _get_color_palette(self):
        """Return color palette based on selected color scheme"""
        if self.color_scheme == 'wong':
            # Wong colorblind-friendly palette
            return ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7', '#000000']
        elif self.color_scheme == 'grayscale':
            return ['#000000', '#404040', '#808080', '#A0A0A0', '#C0C0C0', '#E0E0E0']
        else:  # default
            return ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    
    def _get_plot_options(self):
        """Return common plot options based on aesthetic parameters"""
        return {
            'width': self.plot_width,
            'height': self.plot_height,
            'fontsize': {'title': self.font_size + 2, 'labels': self.font_size, 'xticks': self.font_size - 1, 'yticks': self.font_size - 1}
        }

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
        
        # Determine x-axis column and label based on time_axis_mode
        if self.time_axis_mode == "Actual Time (datetime)":
            x_col = 'timestamp_localtime' if 'timestamp_localtime' in self.dosing_events_df.columns else 'timestamp'
            xlabel = 'Time (date/time)'
        else:  # "Elapsed Time (hours)" or "Both (dual axis)"
            x_col = 'elapsed_hours'
            if self.time_axis_mode == "Both (dual axis)":
                xlabel = 'Elapsed Time (hours) - Actual time shown on hover'
            else:
                xlabel = 'Elapsed Time (hours)'
        
        # Get plot options and color palette
        plot_opts = self._get_plot_options()
        colors = self._get_color_palette()
        
        # HoloViews plots with tooltips - using selected x-axis
        dilution_scatter = hv.Scatter(
            self.dosing_events_df, 
            x_col, 
            'instant_dilution_rate',
            label='Instant Dilution Rate'
        ).opts(
            color=colors[0],
            size=5,
            tools=['hover'],
            width=plot_opts['width'],
            height=plot_opts['height'],
            title='Dilution Rate',
            fontsize=plot_opts['fontsize']
        )
        
        dilution_line = hv.Curve(
            self.dosing_events_df, 
            x_col, 
            'moving_avg_dilution_rate',
            label=f'Moving Avg ({self.moving_avg_window} events)'
        ).opts(
            color=colors[1],
            line_width=2,
            width=plot_opts['width'],
            height=plot_opts['height'],
            fontsize=plot_opts['fontsize']
        )
        
        # Calculate outlier threshold for time differences
        self.dosing_events_df['capped_time_diff'] = np.minimum(self.dosing_events_df['time_diff_hours'] * 60, 60)
        self.dosing_events_df['is_outlier'] = self.dosing_events_df['time_diff_hours'] * 60 > 60
        
        # Create a dataframe for normal points and outliers
        normal_df = self.dosing_events_df[~self.dosing_events_df['is_outlier']]
        outlier_df = self.dosing_events_df[self.dosing_events_df['is_outlier']]
        
        # Time between doses plot - using selected x-axis column
        time_line = hv.Curve(
            self.dosing_events_df,
            x_col,
            'capped_time_diff',
            label='Minutes Between Doses'
        ).opts(
            color=colors[0],
            line_width=1,
            width=plot_opts['width'],
            height=int(plot_opts['height'] * 0.625),  # Keep relative height
            title='Inter-Dosing Period',
            ylabel='Minutes Between Doses (capped at 60)',
            tools=['hover'],
            fontsize=plot_opts['fontsize']
        )
        
        normal_scatter = hv.Scatter(
            normal_df,
            x_col,
            'capped_time_diff'
        ).opts(
            color=colors[0],
            size=6,
            tools=['hover'],
            width=plot_opts['width'],
            height=int(plot_opts['height'] * 0.625),
            fontsize=plot_opts['fontsize']
        )
        
        outlier_scatter = hv.Scatter(
            outlier_df,
            x_col,
            'capped_time_diff'
        ).opts(
            color=colors[3],  # Use 4th color for outliers
            marker='circle',
            size=8,
            tools=['hover'],
            width=plot_opts['width'],
            height=int(plot_opts['height'] * 0.625),
            fontsize=plot_opts['fontsize']
        )
        
        # Use the new formatting method with the start_time
        formatter = self._format_time_ticks(reference_time=start_time)
        
        # Combine plots with the custom formatter and appropriate xlabel
        dilution_plot = (dilution_scatter * dilution_line).opts(
            legend_position='top_right',
            xlabel=xlabel,
            ylabel='Dilution Rate (h⁻¹)'
        )
        
        time_plot = (time_line * normal_scatter * outlier_scatter).opts(
            legend_position='top_right',
            xlabel=xlabel,
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
                
                # Determine x-axis column and label based on time_axis_mode
                if self.time_axis_mode == "Actual Time (datetime)":
                    x_col = 'timestamp_localtime' if 'timestamp_localtime' in self.target_od_df.columns else 'timestamp'
                    xlabel = 'Time (date/time)'
                else:  # "Elapsed Time (hours)" or "Both (dual axis)"
                    x_col = 'elapsed_hours'
                    if self.time_axis_mode == "Both (dual axis)":
                        xlabel = 'Elapsed Time (hours) - Actual time shown on hover'
                    else:
                        xlabel = 'Elapsed Time (hours)'
                
                # Create target OD plot using selected x-axis
                if not self.target_od_df.empty:
                    target_od = hv.Curve(
                        self.target_od_df,
                        x_col,
                        'od_value',
                        label='Target OD'
                    ).opts(
                        color=self._get_color_palette()[2],  # Use 3rd color
                        line_dash='dashed',
                        line_width=2,
                        width=self._get_plot_options()['width'],
                        height=int(self._get_plot_options()['height'] * 0.625),
                        fontsize=self._get_plot_options()['fontsize']
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
                
                # Determine x-axis column for latest OD (use same logic)
                if self.time_axis_mode == "Actual Time (datetime)":
                    x_col_latest = 'timestamp_localtime' if 'timestamp_localtime' in self.latest_od_df.columns else 'timestamp'
                else:
                    x_col_latest = 'elapsed_hours'
                
                # Create latest OD plot using selected x-axis
                if not self.latest_od_df.empty:
                    latest_od = hv.Curve(
                        self.latest_od_df,
                        x_col_latest,
                        'od_value',
                        label='Latest OD'
                    ).opts(
                        color=self._get_color_palette()[0],  # Use 1st color
                        line_width=2,
                        width=self._get_plot_options()['width'],
                        height=int(self._get_plot_options()['height'] * 0.625),
                        fontsize=self._get_plot_options()['fontsize']
                    )
                    od_plots.append(latest_od)
            
            # Create combined OD plot
            if od_plots:
                plot_opts = self._get_plot_options()
                od_plot = hv.Overlay(od_plots).opts(
                    legend_position='top_right',
                    xlabel=xlabel,
                    ylabel='OD Value',
                    title='OD Values',
                    width=plot_opts['width'],
                    height=int(plot_opts['height'] * 0.625),
                    fontsize=plot_opts['fontsize']
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
              <th style="padding:8px; border:1px solid #ddd; text-align:left;">95% CI (OD)</th>
              <th style="padding:8px; border:1px solid #ddd; text-align:left;">OD CV%</th>
              <th style="padding:8px; border:1px solid #ddd; text-align:left;">Avg Dilution (h⁻¹)</th>
              <th style="padding:8px; border:1px solid #ddd; text-align:left;">95% CI (Dilution)</th>
              <th style="padding:8px; border:1px solid #ddd; text-align:left;">Time Between Doses (min)</th>
            </tr>
          </thead>
          <tbody>
        """
        
        for region in od_regions:
            # Format CI for OD
            if pd.notna(region.get('od_ci_lower')) and pd.notna(region.get('od_ci_upper')):
                od_ci_display = f"±{(region['od_ci_upper'] - region['od_ci_lower'])/2:.4f}"
            else:
                od_ci_display = "N/A"
            
            # Format CI for dilution rate
            if pd.notna(region.get('dilution_ci_lower')) and pd.notna(region.get('dilution_ci_upper')):
                dilution_ci_display = f"±{(region['dilution_ci_upper'] - region['dilution_ci_lower'])/2:.4f}"
            else:
                dilution_ci_display = "N/A"
            
            # Safe formatting with defaults for missing values
            html += f"""
            <tr>
              <td style="padding:8px; border:1px solid #ddd;">{region['target_od']:.3f}</td>
              <td style="padding:8px; border:1px solid #ddd;">{region['start'].strftime('%Y-%m-%d %H:%M')} to {region['end'].strftime('%Y-%m-%d %H:%M')}</td>
              <td style="padding:8px; border:1px solid #ddd;">{region.get('od_mean', 'N/A'):.3f} ± {region.get('od_std', 0):.3f}</td>
              <td style="padding:8px; border:1px solid #ddd;">{od_ci_display}</td>
              <td style="padding:8px; border:1px solid #ddd;">{region.get('od_cv', 'N/A'):.1f}%</td>
              <td style="padding:8px; border:1px solid #ddd;">{region.get('avg_dilution_rate', 'N/A') if isinstance(region.get('avg_dilution_rate'), (int, float)) else 'N/A'}{f" ± {region.get('dilution_std', 0):.3f}" if isinstance(region.get('dilution_std'), (int, float)) else ''}</td>
              <td style="padding:8px; border:1px solid #ddd;">{dilution_ci_display}</td>
              <td style="padding:8px; border:1px solid #ddd;">{region.get('avg_time_between_doses_min', 'N/A') if isinstance(region.get('avg_time_between_doses_min'), (int, float)) else 'N/A'}{f" ± {region.get('time_between_doses_std_min', 0):.1f}" if isinstance(region.get('time_between_doses_std_min'), (int, float)) else ''}</td>
            </tr>
            """
        
        html += """
          </tbody>
        </table>
        """
        
        return html
    
    def _calculate_statistics_data(self):
        """
        Calculate statistics for different target_od regions and return as data list.
        
        Returns
        -------
        list
            List of dictionaries containing statistics for each region
        """
        if (self.target_od_df.empty or self.latest_od_df.empty or 
            'od_value' not in self.latest_od_df.columns or
            'od_value' not in self.target_od_df.columns):
            return []
        
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
        
        return od_regions
    
    def _export_statistics_callback(self, event):
        """Export statistics to CSV file."""
        if (self.target_od_df.empty or self.latest_od_df.empty or 
            'od_value' not in self.latest_od_df.columns or
            'od_value' not in self.target_od_df.columns):
            self.show_error("No statistics available to export. Please upload data first.")
            return
        
        # Calculate statistics (reuse existing method)
        # Identify regions with constant target_od
        self.target_od_df = self.target_od_df.sort_values('timestamp')
        od_regions = []
        
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
                    region['od_cv'] = variation(region_od['od_value']) * 100
                    
                    # Calculate 95% CI for OD mean
                    from scipy import stats
                    n_od = len(region_od)
                    if n_od > 1:
                        t_val = stats.t.ppf(0.975, n_od - 1)
                        od_se = region['od_std'] / np.sqrt(n_od)
                        region['od_ci_lower'] = region['od_mean'] - t_val * od_se
                        region['od_ci_upper'] = region['od_mean'] + t_val * od_se
                    else:
                        region['od_ci_lower'] = np.nan
                        region['od_ci_upper'] = np.nan if region['od_mean'] > 0 else 0
                    
                    # Get dosing events in this region
                    if not self.dosing_events_df.empty:
                        region_dosing = self.dosing_events_df[(self.dosing_events_df['timestamp'] >= start_time) & 
                                                             (self.dosing_events_df['timestamp'] < end_time)]
                        
                        if len(region_dosing) > 1:
                            region['avg_dilution_rate'] = region_dosing['instant_dilution_rate'].mean()
                            region['dilution_std'] = region_dosing['instant_dilution_rate'].std()
                            region['avg_time_between_doses_min'] = region_dosing['time_diff_hours'].mean() * 60
                            region['time_between_doses_std_min'] = region_dosing['time_diff_hours'].std() * 60
                            
                            # Calculate 95% CI for dilution rate
                            from scipy import stats
                            n_dilution = len(region_dosing)
                            if n_dilution > 1:
                                t_val = stats.t.ppf(0.975, n_dilution - 1)
                                dilution_se = region['dilution_std'] / np.sqrt(n_dilution)
                                region['dilution_ci_lower'] = region['avg_dilution_rate'] - t_val * dilution_se
                                region['dilution_ci_upper'] = region['avg_dilution_rate'] + t_val * dilution_se
                            else:
                                region['dilution_ci_lower'] = np.nan
                                region['dilution_ci_upper'] = np.nan
                    
                    od_regions.append(region)
        
        if not od_regions:
            self.show_error("No statistics regions found to export.")
            return
        
        try:
            # Convert to DataFrame
            stats_df = pd.DataFrame(od_regions)
            
            # Format timestamps
            stats_df['start'] = pd.to_datetime(stats_df['start']).dt.strftime('%Y-%m-%d %H:%M:%S')
            stats_df['end'] = pd.to_datetime(stats_df['end']).dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # Rename columns for clarity
            stats_df.columns = ['Start Time', 'End Time', 'Target OD', 'Avg OD', 'OD Std Dev', 
                               'OD CV%', 'Avg Dilution Rate (h⁻¹)', 'Dilution Std Dev',
                               'Avg Time Between Doses (min)', 'Time Between Doses Std Dev (min)']
            
            # Generate filename with timestamp
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"dilution_rate_statistics_{timestamp}.csv"
            
            # Save to CSV
            stats_df.to_csv(filename, index=False)
            
            self.show_success(f"Statistics exported to {filename}")
        except Exception as e:
            self.show_error(f"Failed to export statistics: {str(e)}")
    
    def _export_dilution_plot_callback(self, event):
        """Export dilution rate plot to PNG file."""
        if self.dilution_plot.object is None or str(self.dilution_plot.object) == 'Text':
            self.show_error("No dilution rate plot to export. Please upload data first.")
            return
        
        try:
            from datetime import datetime
            import holoviews as hv
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"dilution_rate_plot_{timestamp}.png"
            
            # Save plot using HoloViews save function with DPI
            hv.save(self.dilution_plot.object, filename, fmt='png', dpi=self.plot_dpi)
            
            self.show_success(f"Dilution rate plot exported to {filename} at {self.plot_dpi} DPI")
        except Exception as e:
            self.show_error(f"Failed to export plot: {str(e)}")
    
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
            
            # Save plot using HoloViews save function with DPI
            hv.save(self.od_plot.object, filename, fmt='png', dpi=self.plot_dpi)
            
            self.show_success(f"OD plot exported to {filename} at {self.plot_dpi} DPI")
        except Exception as e:
            self.show_error(f"Failed to export plot: {str(e)}")
    
    def _export_time_plot_callback(self, event):
        """Export time between doses plot to PNG file."""
        if self.time_plot.object is None or str(self.time_plot.object) == 'Text':
            self.show_error("No time plot to export. Please upload data first.")
            return
        
        try:
            from datetime import datetime
            import holoviews as hv
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"time_between_doses_plot_{timestamp}.png"
            
            # Save plot using HoloViews save function with DPI
            hv.save(self.time_plot.object, filename, fmt='png', dpi=self.plot_dpi)
            
            self.show_success(f"Time between doses plot exported to {filename} at {self.plot_dpi} DPI")
        except Exception as e:
            self.show_error(f"Failed to export plot: {str(e)}")
    
    def _export_all_callback(self, event):
        """Export all plots and data to a timestamped folder."""
        import os
        from datetime import datetime
        
        try:
            # Create timestamped export directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_dir = f"dilution_analysis_export_{timestamp}"
            os.makedirs(export_dir, exist_ok=True)
            
            export_count = 0
            
            # Export dilution rate plot if available
            if self.dilution_plot.object is not None and str(self.dilution_plot.object) != 'Text':
                try:
                    import holoviews as hv
                    dilution_filename = os.path.join(export_dir, "dilution_rate_plot.png")
                    hv.save(self.dilution_plot.object, dilution_filename, fmt='png', dpi=self.plot_dpi)
                    export_count += 1
                except Exception as e:
                    self.show_debug(f"Dilution plot export failed: {str(e)}")
            
            # Export OD plot if available
            if self.od_plot.object is not None and str(self.od_plot.object) != 'Text':
                try:
                    import holoviews as hv
                    od_filename = os.path.join(export_dir, "od_plot.png")
                    hv.save(self.od_plot.object, od_filename, fmt='png', dpi=self.plot_dpi)
                    export_count += 1
                except Exception as e:
                    self.show_debug(f"OD plot export failed: {str(e)}")
            
            # Export time plot if available
            if self.time_plot.object is not None and str(self.time_plot.object) != 'Text':
                try:
                    import holoviews as hv
                    time_filename = os.path.join(export_dir, "time_between_doses_plot.png")
                    hv.save(self.time_plot.object, time_filename, fmt='png', dpi=self.plot_dpi)
                    export_count += 1
                except Exception as e:
                    self.show_debug(f"Time plot export failed: {str(e)}")
            
            # Export statistics if available
            if (not self.target_od_df.empty or not self.latest_od_df.empty) and not self.dosing_events_df.empty:
                try:
                    # Reuse statistics calculation logic
                    stats_data = self._calculate_statistics_data()
                    if stats_data:
                        stats_df = pd.DataFrame(stats_data)
                        stats_filename = os.path.join(export_dir, "statistics.csv")
                        stats_df.to_csv(stats_filename, index=False)
                        export_count += 1
                except Exception as e:
                    self.show_debug(f"Statistics export failed: {str(e)}")
            
            # Export raw dosing events data if available
            if not self.dosing_events_df.empty:
                dosing_filename = os.path.join(export_dir, "dosing_events.csv")
                self.dosing_events_df.to_csv(dosing_filename, index=False)
                export_count += 1
            
            # Export OD data if available
            if not self.target_od_df.empty:
                target_od_filename = os.path.join(export_dir, "target_od.csv")
                self.target_od_df.to_csv(target_od_filename, index=False)
                export_count += 1
            
            if not self.latest_od_df.empty:
                latest_od_filename = os.path.join(export_dir, "latest_od.csv")
                self.latest_od_df.to_csv(latest_od_filename, index=False)
                export_count += 1
            
            if export_count > 0:
                self.show_success(f"Exported {export_count} items to folder: {export_dir}")
            else:
                self.show_error("No data available to export. Please upload data first.")
                
        except Exception as e:
            self.show_error(f"Failed to export all: {str(e)}")
            self.show_debug(f"Export all error: {str(e)}")
    
    def _update_data_preview(self, raw_df: pd.DataFrame | None = None):
        """Create and display a preview of the uploaded data.

        Parameters
        ----------
        raw_df : pd.DataFrame | None
            If provided, use this raw dataframe for the first-10-rows table; otherwise derive from
            the internal dataframes (dosing_events_df/target_od_df/latest_od_df).
        """
        try:
            preview_df = None
            if raw_df is not None and isinstance(raw_df, pd.DataFrame) and not raw_df.empty:
                preview_df = raw_df.copy()
            elif hasattr(self, 'dosing_events_df') and isinstance(self.dosing_events_df, pd.DataFrame) and not self.dosing_events_df.empty:
                preview_df = self.dosing_events_df.copy()
            elif hasattr(self, 'latest_od_df') and isinstance(self.latest_od_df, pd.DataFrame) and not self.latest_od_df.empty:
                preview_df = self.latest_od_df.copy()
            elif hasattr(self, 'target_od_df') and isinstance(self.target_od_df, pd.DataFrame) and not self.target_od_df.empty:
                preview_df = self.target_od_df.copy()

            if preview_df is None or preview_df.empty:
                self.data_preview.object = ""
                return

            total_rows = len(preview_df)
            cols = list(preview_df.columns)

            # Time / OD stats when possible
            time_range = "N/A"
            if 'elapsed_hours' in preview_df.columns:
                tmin = pd.to_numeric(preview_df['elapsed_hours'], errors='coerce').min()
                tmax = pd.to_numeric(preview_df['elapsed_hours'], errors='coerce').max()
                if pd.notna(tmin) and pd.notna(tmax):
                    time_range = f"{tmin:.2f}h - {tmax:.2f}h ({(tmax - tmin):.2f}h)"

            od_range = "N/A"
            od_col = next((c for c in ['od_reading', 'od_value', 'od'] if c in preview_df.columns), None)
            if od_col is not None:
                series = pd.to_numeric(preview_df[od_col], errors='coerce')
                if not series.empty:
                    omin = series.min()
                    omax = series.max()
                    omean = series.mean()
                    if pd.notna(omin) and pd.notna(omax) and pd.notna(omean):
                        od_range = f"{omin:.4f} - {omax:.4f} (mean: {omean:.4f})"

            # Build HTML
            def _float_fmt(x):
                try:
                    if isinstance(x, (int, float)):
                        return f"{x:.4f}" if abs(float(x)) < 1000 else f"{x:.2f}"
                    return f"{x}"
                except Exception:
                    return f"{x}"

            html = f"""
            <div style=\"border: 1px solid #ddd; border-radius: 5px; padding: 15px; margin: 10px 0; background-color: #f9f9f9;\">
                <h3 style=\"margin-top:0; color: #333;\">📊 Data Preview</h3>
                <div style=\"margin-bottom: 15px;\">
                    <strong>Summary:</strong>
                    <ul style=\"margin: 5px 0;\">
                        <li>Total Rows: {total_rows}</li>
                        <li>Columns ({len(cols)}): {', '.join(cols)}</li>
                        <li>Time Range: {time_range}</li>
                        <li>OD Range: {od_range}</li>
                    </ul>
                </div>
                <strong>First 10 Rows:</strong>
                <div style=\"overflow-x: auto; margin-top: 10px;\">
                    {preview_df.head(10).to_html(index=False, border=1, classes='dataframe',
                        float_format=_float_fmt)}
                </div>
            </div>
            """

            self.data_preview.object = html
        except Exception as e:
            self.data_preview.object = f"<div style='color:#b00;'>Preview error: {str(e)}</div>"
    
    def save_session_callback(self, event=None):
        """Save the current session state to a JSON file."""
        import json
        from datetime import datetime
        
        try:
            # Create session state dictionary
            session_state = {
                'timestamp': datetime.now().isoformat(),
                'parameters': {
                    'reactor_volume': self.reactor_volume,
                    'moving_avg_window': self.moving_avg_window,
                    'time_axis_mode': self.time_axis_mode,
                    'plot_width': self.plot_width,
                    'plot_height': self.plot_height,
                    'font_size': self.font_size,
                    'color_scheme': self.color_scheme,
                    'plot_dpi': self.plot_dpi,
                },
                'data': {
                    'dosing_events': self.dosing_events_df.to_json(orient='split', date_format='iso') if not self.dosing_events_df.empty else None,
                    'target_od': self.target_od_df.to_json(orient='split', date_format='iso') if not self.target_od_df.empty else None,
                    'latest_od': self.latest_od_df.to_json(orient='split', date_format='iso') if not self.latest_od_df.empty else None,
                },
                'bookmarks': self.bookmarks,
            }
            
            # Save to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'dilution_session_{timestamp}.json'
            
            with open(filename, 'w') as f:
                json.dump(session_state, f, indent=2)
            
            self.show_success(f"Session saved successfully to {filename}")
            
        except Exception as e:
            self.show_error(f"Error saving session: {str(e)}")
            import traceback
            self.show_debug(f"Save error: {traceback.format_exc()}")
    
    def load_session_callback(self, event):
        """Load session state from a JSON file."""
        import json
        
        if not event.new:
            return
        
        self.loading_spinner.value = True
        
        try:
            # Read the uploaded file
            file_obj = event.new[0] if isinstance(event.new, list) else event.new
            content = file_obj.decode('utf-8') if isinstance(file_obj, bytes) else file_obj
            
            session_state = json.loads(content)
            
            # Restore parameters
            params = session_state.get('parameters', {})
            self.reactor_volume = params.get('reactor_volume', 14.0)
            self.moving_avg_window = params.get('moving_avg_window', 30)
            self.time_axis_mode = params.get('time_axis_mode', 'Elapsed Time (hours)')
            self.plot_width = params.get('plot_width', 1200)
            self.plot_height = params.get('plot_height', 400)
            self.font_size = params.get('font_size', 12)
            self.color_scheme = params.get('color_scheme', 'default')
            self.plot_dpi = params.get('plot_dpi', 150)
            
            # Restore data
            data = session_state.get('data', {})
            if data.get('dosing_events'):
                self.dosing_events_df = pd.read_json(io.StringIO(data['dosing_events']), orient='split')
            else:
                self.dosing_events_df = pd.DataFrame()
            
            if data.get('target_od'):
                self.target_od_df = pd.read_json(io.StringIO(data['target_od']), orient='split')
            else:
                self.target_od_df = pd.DataFrame()
            
            if data.get('latest_od'):
                self.latest_od_df = pd.read_json(io.StringIO(data['latest_od']), orient='split')
            else:
                self.latest_od_df = pd.DataFrame()
            
            # Restore bookmarks
            self.bookmarks = session_state.get('bookmarks', [])
            self._update_bookmarks_container()
            
            # Update plots
            self._update_plots()
            
            timestamp = session_state.get('timestamp', 'unknown')
            self.show_success(f"Session loaded successfully (saved: {timestamp})")
            
        except Exception as e:
            self.show_error(f"Error loading session: {str(e)}")
            import traceback
            self.show_debug(f"Load error: {traceback.format_exc()}")
        
        finally:
            self.loading_spinner.value = False

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
