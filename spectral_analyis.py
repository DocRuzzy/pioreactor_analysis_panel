import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import panel as pn
import param
import io
import traceback

# Configure Panel
pn.extension('plotly')
pn.config.sizing_mode = 'stretch_width'

# Load the data
def load_and_process_data(csv_file, file_type='auto'):
    """
    Load and process CSV data for analysis.
    Handles files with optional comment lines beginning with '//' or '#'.

    Args:
        csv_file: CSV file to load, can be a file path or a BytesIO object.
        file_type: Type of file - 'spectrum', 'od', 'temperature', or 'auto' to detect

    Returns:
        A DataFrame containing the processed data.
    """
    try:
        # Read CSV - try to handle comment lines
        if isinstance(csv_file, io.BytesIO):
            # First, peek at the file content to look for comment lines
            content = csv_file.read().decode('utf-8')
            lines = content.split('\n')
            
            # Count comment lines
            comment_count = 0
            for line in lines:
                if line.strip().startswith('//') or line.strip().startswith('#'):
                    comment_count += 1
                else:
                    break
            
            # Reset BytesIO position
            csv_file.seek(0)
            
            # Skip comment lines if any were detected
            if comment_count > 0:
                df = pd.read_csv(csv_file, skiprows=comment_count, delimiter=',')
            else:
                # Try with comma delimiter first
                try:
                    df = pd.read_csv(csv_file, delimiter=',')
                except:
                    # If that fails, try with space delimiter
                    csv_file.seek(0)
                    df = pd.read_csv(csv_file, delimiter=' ')
        else:
            # For file paths, try comma delimiter first
            try:
                df = pd.read_csv(csv_file, delimiter=',')
            except:
                df = pd.read_csv(csv_file, delimiter=' ')
    
        # Display column information for debugging
        print(f"Loaded CSV with columns: {df.columns.tolist()}")
        
        # Auto-detect file type if not specified
        if file_type == 'auto':
            if all(col in df.columns for col in ['band', 'reading']):
                file_type = 'spectrum'
            elif 'od_reading' in df.columns:
                file_type = 'od'
            elif 'temperature_c' in df.columns:
                file_type = 'temperature'
            else:
                file_type = 'unknown'
            
            print(f"Auto-detected file type: {file_type}")
        
        # Convert timestamps to datetime for all file types
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        if 'timestamp_localtime' in df.columns:
            df['timestamp_localtime'] = pd.to_datetime(df['timestamp_localtime'])
        
        # Process specific columns based on file type
        if file_type == 'spectrum':
            # Expected columns for spectrum data
            expected_columns = ['experiment', 'pioreactor_unit', 'timestamp', 
                               'reading', 'band', 'timestamp_localtime']
            
            # Convert reading and band to numeric
            if 'reading' in df.columns:
                df['reading'] = pd.to_numeric(df['reading'], errors='coerce')
            
            if 'band' in df.columns:
                df['band'] = pd.to_numeric(df['band'], errors='coerce')
                
        elif file_type == 'od':
            # Expected columns for OD data
            expected_columns = ['experiment', 'pioreactor_unit', 'timestamp', 
                               'od_reading', 'angle', 'channel', 'timestamp_localtime']
            
            # Convert OD reading to numeric
            if 'od_reading' in df.columns:
                df['od_reading'] = pd.to_numeric(df['od_reading'], errors='coerce')
                
        elif file_type == 'temperature':
            # Expected columns for temperature data
            expected_columns = ['experiment', 'pioreactor_unit', 'timestamp', 
                               'temperature_c', 'timestamp_localtime']
            
            # Convert temperature to numeric
            if 'temperature_c' in df.columns:
                df['temperature_c'] = pd.to_numeric(df['temperature_c'], errors='coerce')
        
        # Check if expected columns exist (based on file type)
        missing_columns = [col for col in expected_columns if col not in df.columns]
        if missing_columns:
            print(f"Warning: Missing expected columns for {file_type} file: {missing_columns}")
        
        # Simplify timestamp to seconds for grouping
        if 'timestamp' in df.columns:
            df['simplified_time'] = df['timestamp'].dt.floor('s')
        elif 'timestamp_localtime' in df.columns:
            df['simplified_time'] = df['timestamp_localtime'].dt.floor('s')
        else:
            print("Warning: No timestamp columns found. Cannot create simplified_time.")
        
        return df
        
    except Exception as e:
        print(f"Error loading CSV file: {str(e)}")
        import traceback
        print(traceback.format_exc())
        # Return empty DataFrame rather than raising exception
        return pd.DataFrame()

def create_spectral_visualization(df, output_file=None, show_plot=True):
    # Get unique bands
    unique_bands = sorted(df['band'].unique())
    
    # Create a colormap that approximates the visible spectrum
    # Map wavelengths to approximate RGB colors
    def wavelength_to_rgb(wavelength):
        # This is a simplified approximation of visible spectrum
        gamma = 0.8
        intensity_max = 1.0
        
        if wavelength < 440:
            r = -(wavelength - 440) / (440 - 380)
            g = 0.0
            b = 1.0
        elif wavelength < 490:
            r = 0.0
            g = (wavelength - 440) / (490 - 440)
            b = 1.0
        elif wavelength < 510:
            r = 0.0
            g = 1.0
            b = -(wavelength - 510) / (510 - 490)
        elif wavelength < 580:
            r = (wavelength - 510) / (580 - 510)
            g = 1.0
            b = 0.0
        elif wavelength < 645:
            r = 1.0
            g = -(wavelength - 645) / (645 - 580)
            b = 0.0
        else:
            r = 1.0
            g = 0.0
            b = 0.0
            
        # Adjust intensity
        if wavelength < 420:
            factor = 0.3 + 0.7 * (wavelength - 380) / (420 - 380)
        elif wavelength > 700:
            factor = 0.3 + 0.7 * (780 - wavelength) / (780 - 700)
        else:
            factor = 1.0
            
        # Gamma adjust and intensity scale
        r = (intensity_max * factor * r ** gamma)
        g = (intensity_max * factor * g ** gamma)
        b = (intensity_max * factor * b ** gamma)
        
        return (r, g, b)
    
    # Create figure and axis
    fig = plt.figure(figsize=(12, 8))
    ax = plt.subplot(111)
    
    # Dictionary to store line objects for legend
    lines = {}
    
    # Plot each band
    for band in unique_bands:
        # Filter data for this band
        band_data = df[df['band'] == band]
        
        # Group by simplified time and calculate mean reading
        time_series = band_data.groupby('simplified_time')['reading'].mean()
        
        # Get color for this band
        color = wavelength_to_rgb(band)
        
        # Plot this band
        line, = ax.plot(time_series.index, time_series.values, 
                         marker='o', markersize=4, linestyle='-', 
                         linewidth=2, color=color, label=f'{band} nm')
        
        lines[band] = line
    
    # Set title and labels
    plt.title('Spectral Bands Over Time', fontsize=16)
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('Reading', fontsize=14)
    
    # Format x-axis to show times nicely
    plt.gcf().autofmt_xdate()
    
    # Add legend
    plt.legend(title='Wavelength (nm)')
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Tight layout
    plt.tight_layout()
    
    # Save if output file is specified
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    # Show plot if requested
    if show_plot:
        plt.show()
    
    return fig

def create_heatmap_visualization(df, output_file=None, show_plot=True):
    """Create a heatmap to show all bands over time"""
    # Pivot the data to create a matrix with time as rows and bands as columns
    pivot = df.pivot_table(index='simplified_time', columns='band', values='reading', aggfunc='mean')
    
    # Create figure
    fig = plt.figure(figsize=(12, 8))
    
    # Create heatmap
    im = plt.imshow(pivot.values, aspect='auto', cmap='viridis')
    
    # Set y-axis (time) ticks and labels
    plt.yticks(range(len(pivot.index)), 
               [t.strftime('%H:%M:%S') for t in pivot.index], 
               fontsize=10)
    
    # Set x-axis (band) ticks and labels
    plt.xticks(range(len(pivot.columns)), 
               [f"{b} nm" for b in pivot.columns], 
               rotation=45, fontsize=10)
    
    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label('Reading Value', rotation=270, labelpad=20)
    
    # Set title and labels
    plt.title('Spectral Bands Heatmap', fontsize=16)
    plt.xlabel('Wavelength (nm)', fontsize=14)
    plt.ylabel('Time', fontsize=14)
    
    # Tight layout
    plt.tight_layout()
    
    # Save if output file is specified
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    # Show plot if requested
    if show_plot:
        plt.show()
        
    return fig

class SpectralAnalysis(param.Parameterized):
    """
    Main application class for spectral data analysis.
    """
    
    # Parameter controls
    show_heatmap = param.Boolean(False, doc="Show heatmap visualization instead of spectrum plot")
    
    def __init__(self, **params):
        """
        Initialize the spectral analysis application.
        """
        super().__init__(**params)
        
        # Initialize empty data containers
        self.spectrum_df = pd.DataFrame()
        self.od_df = pd.DataFrame()
        self.temp_df = pd.DataFrame()
        self.current_filename = None
        
        # Status message for notifications
        self.status_message = pn.pane.Markdown("", styles={'color': 'blue'})
        self.error_message = pn.pane.Markdown("", styles={'color': 'red'})
        self.debug_message = pn.pane.Markdown("", styles={'color': 'green', 'font-family': 'monospace', 'font-size': '12px'})
        
        # Set up the interface
        self.spectrum_file_input = pn.widgets.FileInput(accept='.csv', multiple=False, name="Spectrum Data CSV")
        self.spectrum_file_input.param.watch(self._upload_spectrum_file_callback, 'value')
        
        self.od_file_input = pn.widgets.FileInput(accept='.csv', multiple=False, name="OD Readings CSV")
        self.od_file_input.param.watch(self._upload_od_file_callback, 'value')
        
        self.temp_file_input = pn.widgets.FileInput(accept='.csv', multiple=False, name="Temperature CSV")
        self.temp_file_input.param.watch(self._upload_temp_file_callback, 'value')
        
        # Add status indicators for each file
        self.spectrum_status = pn.pane.Markdown("Not loaded", styles={'color': 'gray', 'font-size': '12px'})
        self.od_status = pn.pane.Markdown("Not loaded", styles={'color': 'gray', 'font-size': '12px'})
        self.temp_status = pn.pane.Markdown("Not loaded", styles={'color': 'gray', 'font-size': '12px'})
        
        self.update_button = pn.widgets.Button(name='Update Plots', button_type='primary')
        self.update_button.on_click(self._update_plots_callback)
        
        # Create plot panes
        self.spectrum_plot = pn.pane.Matplotlib(sizing_mode='stretch_width', height=500)
        self.heatmap_plot = pn.pane.Matplotlib(sizing_mode='stretch_width', height=500)
        self.od_plot = pn.pane.Matplotlib(sizing_mode='stretch_width', height=300)
        self.temp_plot = pn.pane.Matplotlib(sizing_mode='stretch_width', height=300)
        
        # Save button
        self.save_button = pn.widgets.Button(name='Save Current Plot', button_type='success')
        self.save_button.on_click(self._save_plot_callback)
        self.save_filename = pn.widgets.TextInput(name='Filename', value='spectral_plot.png')
        
        # Main layout
        self.main_layout = pn.Column(
            pn.pane.Markdown("# Spectral Data Analysis"),
            pn.Row(
                pn.Column(
                    pn.pane.Markdown("### Settings"),
                    self.param.show_heatmap,
                    self.update_button,
                    width=300
                ),
                pn.Column(
                    pn.pane.Markdown("### Upload Data"),
                    pn.pane.Markdown("Upload each data file type:"),
                    pn.Row(self.spectrum_file_input, self.spectrum_status),
                    pn.Row(self.od_file_input, self.od_status),
                    pn.Row(self.temp_file_input, self.temp_status),
                    self.status_message,
                    self.error_message,
                    width=400
                )
            ),
            pn.Tabs(
                ('Visualization', pn.Column(
                    pn.pane.Markdown("### Spectral Analysis"),
                    pn.Row(
                        self.save_filename,
                        self.save_button
                    ),
                    # Use the dynamic panel for spectral/heatmap
                    pn.panel(self._current_plot_pane, loading_indicator=True),
                    
                    # Add OD plot to this tab
                    pn.pane.Markdown("### OD Readings"),
                    self.od_plot,
                    
                    # Add temperature plot to this tab
                    pn.pane.Markdown("### Temperature"),
                    self.temp_plot
                )),
                ('Debug', self.debug_message)
            )
        )
    
    @param.depends('show_heatmap')
    def _current_plot_pane(self):
        """Return the appropriate plot based on the show_heatmap parameter."""
        if self.show_heatmap:
            return self.heatmap_plot
        else:
            return self.spectrum_plot
    
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
    
    def _upload_spectrum_file_callback(self, event):
        """Handle spectrum data file upload"""
        if self.spectrum_file_input.value is not None and self.spectrum_file_input.filename.endswith('.csv'):
            try:
                # Create BytesIO object from uploaded file
                file_data = io.BytesIO(self.spectrum_file_input.value)
                
                # Process the file as spectrum data
                self.spectrum_df = load_and_process_data(file_data, file_type='spectrum')
                
                if self.spectrum_df.empty:
                    self.show_error("Failed to load spectrum data or file is empty.")
                    self.spectrum_status.object = "⚠️ Error loading file"
                    return
                    
                self.current_filename = self.spectrum_file_input.filename
                self.spectrum_status.object = f"✓ Loaded: {self.current_filename}"
                
                # Debug information
                debug_info = f"Loaded spectrum file: {self.current_filename}\n"
                debug_info += f"Shape: {self.spectrum_df.shape}\n"
                debug_info += f"Columns: {', '.join(self.spectrum_df.columns)}\n"
                if 'band' in self.spectrum_df.columns:
                    debug_info += f"Unique bands: {sorted(self.spectrum_df['band'].unique())}\n"
                debug_info += f"Time range: {self.spectrum_df['simplified_time'].min()} to {self.spectrum_df['simplified_time'].max()}"
                self.show_debug(debug_info)
                
                # Update the plots
                self._update_plots()
                
                self.show_success(f"Spectrum file {self.current_filename} loaded successfully!")
            except Exception as e:
                self.show_error(f"Error processing spectrum file: {str(e)}")
                self.spectrum_status.object = "⚠️ Error loading file"
                # Show detailed error in debug
                self.show_debug(traceback.format_exc())
        else:
            self.show_error("Please upload a CSV file.")

    def _upload_od_file_callback(self, event):
        """Handle OD readings file upload"""
        if self.od_file_input.value is not None and self.od_file_input.filename.endswith('.csv'):
            try:
                # Create BytesIO object from uploaded file
                file_data = io.BytesIO(self.od_file_input.value)
                
                # Load the CSV file as OD data
                self.od_df = load_and_process_data(file_data, file_type='od')
                self.od_status.object = f"✓ Loaded: {self.od_file_input.filename}"
                
                # Update OD plot
                self._update_od_plot()
                
                self.show_success(f"OD file loaded successfully!")
            except Exception as e:
                self.show_error(f"Error processing OD file: {str(e)}")
                self.od_status.object = "⚠️ Error loading file"
                self.show_debug(traceback.format_exc())
        else:
            self.show_error("Please upload a CSV OD file.")

    def _upload_temp_file_callback(self, event):
        """Handle temperature readings file upload"""
        if self.temp_file_input.value is not None and self.temp_file_input.filename.endswith('.csv'):
            try:
                # Create BytesIO object from uploaded file
                file_data = io.BytesIO(self.temp_file_input.value)
                
                # Load the CSV file as temperature data
                self.temp_df = load_and_process_data(file_data, file_type='temperature')
                self.temp_status.object = f"✓ Loaded: {self.temp_file_input.filename}"
                
                # Update temperature plot
                self._update_temp_plot()
                
                self.show_success(f"Temperature file loaded successfully!")
            except Exception as e:
                self.show_error(f"Error processing temperature file: {str(e)}")
                self.temp_status.object = "⚠️ Error loading file"
                self.show_debug(traceback.format_exc())
        else:
            self.show_error("Please upload a CSV temperature file.")
    
    def _update_plots_callback(self, event):
        """Handle update button click events."""
        self._update_plots()
        self.show_success("Plots updated with current settings.")

    def _update_plots(self):
        """Update all plots with current data."""
        if not self.spectrum_df.empty:
            # Update main spectral plot
            if self.show_heatmap:
                fig = create_heatmap_visualization(self.spectrum_df, show_plot=False)
                self.heatmap_plot.object = fig
            else:
                fig = create_spectral_visualization(self.spectrum_df, show_plot=False)
                self.spectrum_plot.object = fig
        
        # Update OD plot if data is available
        self._update_od_plot()
        
        # Update temperature plot if data is available
        self._update_temp_plot()

    def _update_od_plot(self):
        """Update the OD plot with current data."""
        if not hasattr(self, 'od_df') or self.od_df.empty:
            # Create empty figure with message
            fig = plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, 'No OD data available', ha='center', va='center')
            plt.close()
            self.od_plot.object = fig
            return
        
        # Create figure for OD plot
        fig = plt.figure(figsize=(10, 6))
        
        # Plot OD readings
        has_data = False
        if 'od_reading' in self.od_df.columns:
            plt.plot(self.od_df['simplified_time'], self.od_df['od_reading'], 
                     'o-', color='blue', label='OD Reading')
            has_data = True
            
        # Set labels and title
        plt.title('Optical Density Readings')
        plt.xlabel('Time')
        plt.ylabel('OD Reading')
        plt.grid(True, alpha=0.3)
        
        # Only add legend if we have data with labels
        if has_data:
            plt.legend()
        
        # Format x-axis for dates
        plt.gcf().autofmt_xdate()
        plt.tight_layout()
        
        # Update plot
        self.od_plot.object = fig

    def _update_temp_plot(self):
        """Update the temperature plot with current data."""
        if not hasattr(self, 'temp_df') or self.temp_df.empty:
            # Create empty figure with message
            fig = plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, 'No temperature data available', ha='center', va='center')
            plt.close()
            self.temp_plot.object = fig
            return
        
        # Create figure for temperature plot
        fig = plt.figure(figsize=(10, 6))
        
        # Plot temperature readings
        has_data = False
        if 'temperature_c' in self.temp_df.columns:
            plt.plot(self.temp_df['simplified_time'], self.temp_df['temperature_c'], 
                     'o-', color='red', label='Temperature (°C)')
            has_data = True
            
        # Set labels and title
        plt.title('Temperature Readings')
        plt.xlabel('Time')
        plt.ylabel('Temperature (°C)')
        plt.grid(True, alpha=0.3)
        
        # Only add legend if we have data with labels
        if has_data:
            plt.legend()
        
        # Format x-axis for dates
        plt.gcf().autofmt_xdate()
        plt.tight_layout()
        
        # Update plot
        self.temp_plot.object = fig
    
    def _save_plot_callback(self, event):
        """Save the current plot to a file."""
        if self.spectrum_df.empty:
            self.show_error("No data available to save.")
            return
        
        try:
            filename = self.save_filename.value
            if not filename.endswith(('.png', '.jpg', '.pdf', '.svg')):
                filename += '.png'
            
            if self.show_heatmap:
                fig = create_heatmap_visualization(self.spectrum_df, show_plot=False)
            else:
                fig = create_spectral_visualization(self.spectrum_df, show_plot=False)
            
            fig.savefig(filename, dpi=300, bbox_inches='tight')
            self.show_success(f"Plot saved to {filename}")
        except Exception as e:
            self.show_error(f"Error saving plot: {str(e)}")
            self.show_debug(traceback.format_exc())
    
    def view(self):
        """Return the main layout for display."""
        return self.main_layout

# Create the application
spectral_analysis = SpectralAnalysis()

# Create a Panel server
app = pn.panel(spectral_analysis.view())

# Server
if __name__ == "__main__":
    # Ensure the server uses the __file__ directory if run as script
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__))) 
    app.show(port=5007)