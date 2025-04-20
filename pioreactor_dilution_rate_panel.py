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
                pn.state.notifications.success(f"File {self.file_input.filename} uploaded successfully!")
            except Exception as e:
                pn.state.notifications.error(f"Error processing file: {str(e)}")
        else:
            pn.state.notifications.warning("Please upload a CSV file.")
    
    def _process_data(self, df):
        """Process the uploaded CSV data"""
        # Convert timestamps to datetime objects
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Create elapsed time column (in hours)
        start_time = df['timestamp'].min()
        df['elapsed_hours'] = (df['timestamp'] - start_time).dt.total_seconds() / 3600
        
        # Extract dosing events (add, remove) and OD-related events
        self.dosing_events_df = df[df['event_name'].str.contains('DilutionEvent', na=False)].copy()
        od_events = df[df['event_name'].str.contains('target_od|latest_od', na=False)].copy()
        
        # Parse volume from dosing events
        self.dosing_events_df['volume'] = self.dosing_events_df.apply(self._extract_volume, axis=1)
        
        # Extract OD values
        od_events['od_value'] = od_events['message'].apply(self._extract_od)
        
        # Separate target_od and latest_od
        self.target_od_df = od_events[od_events['event_name'] == 'target_od'].copy()
        self.latest_od_df = od_events[od_events['event_name'] == 'latest_od'].copy()
    
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
        def tap_callback(plot, element, event):
            # Check if it's a tap event and has coordinates
            if event.kind == 'tap' and hasattr(event, 'x') and hasattr(event, 'y'):
                self._add_bookmark(event.x, event.y)
                pn.state.notifications.info(f"Bookmark added at x={event.x}, y={event.y}")
        
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
        dilution_plot.opts(hooks=[self._create_tap_callback(dilution_plot)])
        
        od_plot = od_plot.opts(tools=['tap'])
        od_plot.opts(hooks=[self._create_tap_callback(od_plot)])
        
        time_plot = time_plot.opts(tools=['tap'])
        time_plot.opts(hooks=[self._create_tap_callback(time_plot)])
        
        # Update plot panes
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
        self.target_od_df = self.target_od_df.sort_values('timestamp')
        od_regions = []
        
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
                region['od_cv'] = variation(region_od['od_value']) * 100  # CV in percentage
                
                # Get dosing events in this region
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
            html += f"""
            <tr>
              <td style="padding:8px; border:1px solid #ddd;">{region['target_od']:.3f}</td>
              <td style="padding:8px; border:1px solid #ddd;">{region['start'].strftime('%Y-%m-%d %H:%M')} to {region['end'].strftime('%Y-%m-%d %H:%M')}</td>
              <td style="padding:8px; border:1px solid #ddd;">{region.get('od_mean', 'N/A'):.3f} ± {region.get('od_std', 'N/A'):.3f}</td>
              <td style="padding:8px; border:1px solid #ddd;">{region.get('od_cv', 'N/A'):.1f}%</td>
              <td style="padding:8px; border:1px solid #ddd;">{region.get('avg_dilution_rate', 'N/A'):.3f} ± {region.get('dilution_std', 'N/A'):.3f}</td>
              <td style="padding:8px; border:1px solid #ddd;">{region.get('avg_time_between_doses_min', 'N/A'):.1f} ± {region.get('time_between_doses_std_min', 'N/A'):.1f}</td>
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
