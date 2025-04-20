# Interactive Pioreactor Dilution Rate Analysis Tool with Panel/HoloViz
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

# Configure Panel and HoloViews
pn.extension('plotly', 'tabulator')
hv.extension('bokeh')

# Style the panel with a nice theme
pn.config.sizing_mode = 'stretch_width'

class PioreactorAnalysis(param.Parameterized):
    reactor_volume = param.Number(14.0, bounds=(1, 100), step=0.1, doc="Reactor volume in mL")
    moving_avg_window = param.Integer(5, bounds=(1, 50), step=1, doc="Moving average window (# of events)")
    
    def __init__(self, **params):
        super().__init__(**params)
        
        # Initialize empty data containers
        self.dosing_events_df = pd.DataFrame()
        self.target_od_df = pd.DataFrame()
        self.latest_od_df = pd.DataFrame()
        self.bookmarks = []
        
        # Set up the interface
        self.file_input = pn.widgets.FileInput(accept='.csv', multiple=False)
        self.file_input.param.watch(self._upload_file_callback, 'value')
        
        self.update_button = pn.widgets.Button(name='Update', button_type='primary')
        self.update_button.on_click(self._update_callback)
        
        # Create plot panes
        self.dilution_plot = pn.pane.HoloViews(sizing_mode='stretch_width', height=300)
        self.od_plot = pn.pane.HoloViews(sizing_mode='stretch_width', height=250)
        self.time_plot = pn.pane.HoloViews(sizing_mode='stretch_width', height=250)
        
        # Status pane for notifications fallback
        self.status_pane = pn.pane.Markdown("")
        
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
            self.status_pane,  # Add status pane for notifications
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
                    width=400
                )
            ),
            pn.Tabs(
                ('Dilution Analysis', pn.Column(
                    self.dilution_plot,
                    self.od_plot,
                    self.time_plot
                )),
                ('Statistics', self.stats_output)
            ),
            pn.Row(
                pn.Column(
                    self.bookmarks_title,
                    self.bookmarks_container
                )
            )
        )
    
    def show_notification(self, message, level="info"):
        """Safe way to show notifications with fallback to status pane"""
        # Print to console always (for debugging)
        print(f"{level.upper()}: {message}")
        
        # Try Panel notifications API if available
        try:
            if hasattr(pn.state, 'notifications') and pn.state.notifications is not None:
                notification_method = getattr(pn.state.notifications, level, None)
                if notification_method:
                    notification_method(message)
                    return
        except Exception:
            pass
            
        # Fallback to status pane
        color_map = {
            "info": "blue",
            "success": "green",
            "warning": "orange",
            "error": "red"
        }
        color = color_map.get(level, "black")
        self.status_pane.object = f"<div style='color: {color}; padding: 10px; border-left: 4px solid {color};'>{message}</div>"

    def _upload_file_callback(self, event):
        """Handle file upload event"""
        if self.file_input.value is not None and self.file_input.filename.endswith('.csv'):
            try:
                # Decode the file contents
                decoded = io.BytesIO(self.file_input.value)
                
                # Read the CSV file
                df = pd.read_csv(decoded)
                
                # Process the data
                self._process_data(df)
                
                # Update plots
                self._update_plots()
                
                # Show success message
                self.show_notification(f"File {self.file_input.filename} uploaded successfully!", "success")
            except Exception as e:
                self.show_notification(f"Error processing file: {str(e)}", "error")
        else:
            self.show_notification("Please upload a CSV file.", "warning")
    
    def _process_data(self, df):
        """Process the uploaded CSV data"""
        # Convert timestamps to datetime objects
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Create elapsed time column (in hours)
        start_time = df['timestamp'].min()
        df['elapsed_hours'] = (df['timestamp'] - start_time).dt.total_seconds() / 3600
        
        # Extract dosing events
        self.dosing_events_df = df[df['event_name'].str.contains('DilutionEvent', na=False)].copy()
        
        if self.dosing_events_df.empty:
            self.show_notification("No DilutionEvent rows found in CSV", "warning")
            return
        
        # Parse volume and OD values from dosing events
        self.dosing_events_df['volume'] = self.dosing_events_df.apply(self._extract_volume, axis=1)
        self.dosing_events_df['latest_od'] = self.dosing_events_df.apply(self._extract_latest_od, axis=1)
        self.dosing_events_df['target_od'] = self.dosing_events_df.apply(self._extract_target_od, axis=1)
        
        # Create target_od and latest_od dataframes from the extracted values
        # These will be used for plotting
        self.target_od_df = self.dosing_events_df[['timestamp', 'target_od']].rename(columns={'target_od': 'od_value'})
        self.latest_od_df = self.dosing_events_df[['timestamp', 'latest_od']].rename(columns={'latest_od': 'od_value'})
    
    def _extract_volume(self, row):
        """Extract volume from dosing events"""
        try:
            if 'data' in row and row['data']:
                # Parse JSON data column
                data_dict = json.loads(row['data'])
                if 'volume' in data_dict:
                    return float(data_dict['volume'])
        except Exception:
            pass
        return None
    
    def _extract_latest_od(self, row):
        """Extract latest_od value from data JSON"""
        try:
            if 'data' in row and row['data']:
                data_dict = json.loads(row['data'])
                if 'latest_od' in data_dict:
                    return float(data_dict['latest_od'])
        except Exception:
            pass
        return None

    def _extract_target_od(self, row):
        """Extract target_od value from data JSON"""
        try:
            if 'data' in row and row['data']:
                data_dict = json.loads(row['data'])
                if 'target_od' in data_dict:
                    return float(data_dict['target_od'])
        except Exception:
            pass
        return None

    def _extract_od(self, message):
        """Extract OD value from message string or JSON."""
        try:
            # Try parsing JSON if available
            if isinstance(message, str) and message.startswith('{'):
                data = json.loads(message)
                if 'od' in data:
                    return float(data['od'])
                if 'latest_od' in data:
                    return float(data['latest_od'])
            # Try to extract OD value from string like "Latest OD = 0.30 ≥ Target OD = 0.30; cycled 0.50 mL"
            if isinstance(message, str):
                match = re.search(r'Latest OD\s*=\s*([0-9.]+)', message)
                if match:
                    return float(match.group(1))
        except Exception:
            pass
        return None
    
    def _update_callback(self, event):
        """Update plots when the update button is clicked"""
        self._update_plots()
    
    def _add_bookmark(self, x, y, label=None):
        """Add a bookmark at the specified point"""
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
        """Remove a bookmark with the specified ID"""
        self.bookmarks = [bm for bm in self.bookmarks if bm['id'] != bookmark_id]
        self._update_bookmarks_container()
    
    def _update_bookmarks_container(self):
        """Update the bookmarks container"""
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
                                 on_click=lambda event, bm=bm: pn.state.notifications.info(f"Copied: Time: {bm['timestamp']}, Value: {bm['value']}")),
                title=bm['label'],
                collapsed=False
            )
            bookmark_cards.append(card)
        
        self.bookmarks_container[0] = pn.Column(*bookmark_cards)
    
    def _create_tap_callback(self, plot):
        """Create a callback function for tap events on a plot"""
        def tap_callback(plot, element):
            if not hasattr(plot, 'current_event'):
                return
            event = plot.current_event
            # Check if it's a tap event and has coordinates
            if hasattr(event, 'x') and hasattr(event, 'y'):
                self._add_bookmark(event.x, event.y)
                self.show_notification(f"Bookmark added at x={event.x}, y={event.y}", "info")
        
        return tap_callback
    
    def _update_plots(self):
        """Update all plots with current data and parameters"""
        if self.dosing_events_df.empty:
            # No data to plot
            self.dilution_plot.object = hv.Text(0, 0, 'No data uploaded').opts(width=800, height=300)
            self.od_plot.object = hv.Text(0, 0, 'No data uploaded').opts(width=800, height=250)
            self.time_plot.object = hv.Text(0, 0, 'No data uploaded').opts(width=800, height=250)
            self.stats_output.object = "<p>No data uploaded or error in data parsing</p>"
            return
        
        # Calculate dilution rate from dosing events
        # Sort by timestamp
        self.dosing_events_df = self.dosing_events_df.sort_values('timestamp')
        
        # Calculate time between doses
        self.dosing_events_df['next_timestamp'] = self.dosing_events_df['timestamp'].shift(-1)
        self.dosing_events_df['time_diff_hours'] = (self.dosing_events_df['next_timestamp'] - 
                                                   self.dosing_events_df['timestamp']).dt.total_seconds() / 3600
        
        # Calculate instant dilution rate
        self.dosing_events_df['instant_dilution_rate'] = (self.dosing_events_df['volume'] / 
                                                         self.reactor_volume / 
                                                         self.dosing_events_df['time_diff_hours'])
        
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
        
        # OD Values plot
        od_plots = []
        
        if not self.target_od_df.empty and 'od_value' in self.target_od_df.columns:
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
        
        if not self.latest_od_df.empty and 'od_value' in self.latest_od_df.columns:
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
        
        # Calculate statistics for different target_od regions
        stats_html = self._calculate_stats()
        
        # Combine plots
        dilution_plot = (dilution_scatter * dilution_line).opts(
            legend_position='top_right',
            xlabel='Time',
            ylabel='Dilution Rate (h⁻¹)'
        )
        
        od_plot = hv.Overlay(od_plots).opts(
            legend_position='top_right',
            xlabel='Time',
            ylabel='OD Value',
            title='OD Values'
        ) if od_plots else hv.Text(0, 0, 'No OD data available').opts(width=800, height=250)
        
        time_plot = (time_line * normal_scatter * outlier_scatter).opts(
            legend_position='top_right',
            xlabel='Time',
            ylabel='Minutes Between Doses',
            ylim=(0, 65)
        )
        
        # Add tap callbacks for interactions
        dilution_plot = dilution_plot.opts(tools=['tap'])
        od_plot = od_plot.opts(tools=['tap'])
        time_plot = time_plot.opts(tools=['tap'])

        # Use hover for displaying info but don't attach tap hooks yet
        # since they're causing issues - we'll revisit this later
        
        # Create a shared range for all plots
        shared_x_range = None

        # Link the x-ranges for synchronized zooming
        dilution_plot = dilution_plot.opts(shared_axes=True)
        od_plot = od_plot.opts(shared_axes=True)
        time_plot = time_plot.opts(shared_axes=True)

        # Now assign plots to their respective panes
        self.dilution_plot.object = dilution_plot
        self.od_plot.object = od_plot
        self.time_plot.object = time_plot
        
        # Update stats output
        self.stats_output.object = stats_html
    
    def _calculate_stats(self):
        """Calculate statistics for different target_od regions"""
        if (self.target_od_df.empty or self.latest_od_df.empty or 
            'od_value' not in self.latest_od_df.columns):
            return "<p>No OD data available for statistics.</p>"
        
        # Identify regions with constant target_od
        df = self.dosing_events_df.copy()
        if df.empty or 'target_od' not in df.columns:
            return "<p>No target OD data available for statistics.</p>"
        
        # Create a new column to identify changes in target_od
        df['target_od_group'] = (df['target_od'] != df['target_od'].shift(1)).cumsum()
        od_regions = []
        
        # Group by target_od_group to find regions with constant target_od
        for group_id, group_df in df.groupby('target_od_group'):
            if len(group_df) < 3:  # Skip groups with too few points
                continue
                
            # Create a region dictionary to store statistics
            region = {
                'target_od': group_df['target_od'].iloc[0],
                'start': group_df['timestamp'].min(),
                'end': group_df['timestamp'].max(),
                'count': len(group_df),
                'od_mean': group_df['latest_od'].mean(),
                'od_std': group_df['latest_od'].std()
            }
            
            # Calculate coefficient of variation for OD (std/mean * 100%)
            if pd.notna(region['od_mean']) and region['od_mean'] != 0 and pd.notna(region['od_std']):
                region['od_cv'] = (region['od_std'] / region['od_mean']) * 100
            else:
                region['od_cv'] = np.nan  # Assign NaN if mean is zero or std/mean is NaN

            # Calculate dilution stats (using instant rate calculated earlier)
            region['avg_dilution_rate'] = group_df['instant_dilution_rate'].mean()
            region['dilution_std'] = group_df['instant_dilution_rate'].std()

            # Calculate time between doses stats
            valid_time_diffs = group_df['time_diff_hours'].dropna()
            if not valid_time_diffs.empty:
                region['avg_time_between_doses_min'] = valid_time_diffs.mean() * 60
                region['time_between_doses_std_min'] = valid_time_diffs.std() * 60
            else:
                region['avg_time_between_doses_min'] = np.nan
                region['time_between_doses_std_min'] = np.nan

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
        
        # Helper function to safely format values
        def format_value(value, precision=3):
            if value == 'N/A' or pd.isna(value):
                return 'N/A'
            try:
                return f"{float(value):.{precision}f}"
            except (ValueError, TypeError):
                return str(value)
        
        for region in od_regions:
            # Format the time strings outside the f-string to avoid potential issues
            start_time = region['start'].strftime('%Y-%m-%d %H:%M') if hasattr(region['start'], 'strftime') else str(region['start'])
            end_time = region['end'].strftime('%Y-%m-%d %H:%M') if hasattr(region['end'], 'strftime') else str(region['end'])
            
            html += f"""
            <tr>
              <td style="padding:8px; border:1px solid #ddd;">{format_value(region['target_od'])}</td>
              <td style="padding:8px; border:1px solid #ddd;">{start_time} to {end_time}</td>
              <td style="padding:8px; border:1px solid #ddd;">{format_value(region.get('od_mean', 'N/A'))} ± {format_value(region.get('od_std', 'N/A'))}</td>
              <td style="padding:8px; border:1px solid #ddd;">{format_value(region.get('od_cv', 'N/A'), 1)}%</td>
              <td style="padding:8px; border:1px solid #ddd;">{format_value(region.get('avg_dilution_rate', 'N/A'))} ± {format_value(region.get('dilution_std', 'N/A'))}</td>
              <td style="padding:8px; border:1px solid #ddd;">{format_value(region.get('avg_time_between_doses_min', 'N/A'), 1)} ± {format_value(region.get('time_between_doses_std_min', 'N/A'), 1)}</td>
            </tr>
            """
        
        html += """
          </tbody>
        </table>
        """
        
        return html
    
    def view(self):
        """Return the main layout for display"""
        return self.main_layout

# Create the application
pioreactor_analysis = PioreactorAnalysis()

# Create a Panel server
app = pn.panel(pioreactor_analysis.view())

# Server
if __name__ == '__main__':
    app.show(port=5006)
