# app.py - Entry point for Hugging Face Spaces
import panel as pn
import os
import sys
from pioreactor_dilution_rate_panel import PioreactorAnalysis  # Using the correct file name

# Configure Panel for cloud environment
pn.extension(template="fast", sizing_mode="stretch_width", notifications=True)

# Configure logging to help debug initialization problems
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('pioreactor-app')

# Initialize the application
logger.info("Initializing Pioreactor Analysis application")
pioreactor_analysis = PioreactorAnalysis()
logger.info("Analysis object created successfully")

# Create the Panel template
app = pn.template.FastListTemplate(
    site="Pioreactor Dilution Rate Analysis",
    title="Pioreactor Dilution Rate Analysis",
    sidebar=[
        pn.pane.Markdown("## Controls"),
        pn.pane.Markdown("### Settings"),
        pioreactor_analysis.param.reactor_volume,
        pioreactor_analysis.param.moving_avg_window,
        pioreactor_analysis.update_button,
        pn.pane.Markdown("### Upload Data"),
        pioreactor_analysis.file_input,
    ],
    main=[
        pn.Row(
            pn.Column(
                pn.pane.Markdown("## Pioreactor Dilution Rate Analysis"),
                pn.pane.Markdown("""
                This tool analyzes dilution rates from Pioreactor CSV data exports.
                Upload your Pioreactor events CSV file using the sidebar controls.
                """),
                pioreactor_analysis.dilution_plot,
                pioreactor_analysis.od_plot, 
                pioreactor_analysis.time_plot,
                pioreactor_analysis.stats_output,
                pioreactor_analysis.bookmarks_title,
                pioreactor_analysis.bookmarks_container
            )
        )
    ],
    accent_base_color="#5A7B9C",
    header_background="#393939",
)

# Make the template servable
app.servable()

# For Hugging Face Spaces, we need to explicitly configure the server
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))  # Hugging Face expects port 7860
    address = "0.0.0.0"  # Listen on all network interfaces
    logger.info(f"Starting Panel server on {address}:{port}")
    pn.serve(app, port=port, address=address, allow_websocket_origin=["*"], show=False)