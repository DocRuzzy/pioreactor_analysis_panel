# app.py - Entry point for Hugging Face Spaces
import panel as pn
import os
import sys
import param
import logging

# Import application classes
from batch_growth_rate_analysis import GrowthRateAnalysis
from pioreactor_dilution_rate_panel import PioreactorAnalysis

# Configure Panel and logging
pn.extension(template="fast", sizing_mode="stretch_width", notifications=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('pioreactor-app')

"""Restructured with collapsible left sidebar and analysis type selector."""

# Initialize both analysis instances
batch_analysis = GrowthRateAnalysis()
dilution_analysis = PioreactorAnalysis()

# Analysis selector widget
analysis_selector = pn.widgets.RadioButtonGroup(
    name='Analysis Type',
    options=['Batch Growth Rate', 'Dilution Rate'],
    value='Batch Growth Rate',
    button_type='primary'
)

# Dynamic content based on selector
@pn.depends(analysis_selector.param.value)
def get_analysis_panel(analysis_type):
    if analysis_type == 'Batch Growth Rate':
        return batch_analysis.view()
    else:
        return dilution_analysis.view()

# Build collapsible sidebar with general info and instructions
sidebar_controls = [
    pn.pane.Markdown("## Pioreactor Analysis"),
    pn.pane.Markdown("---"),
    pn.pane.Markdown("### Select Analysis Type"),
    analysis_selector,
    pn.pane.Markdown("---"),
    pn.pane.Markdown("""
    ### Quick Start
    
    1. **Select Analysis Type** using the buttons above
    
    2. **Upload Data** in the Controls tab
       
       **Batch Growth Rate Requirements:**
       - CSV export with **absolute OD** (not normalized)
       - Timestamp and OD reading columns
       
       **Dilution Rate Requirements:**
       - CSV export with **automation dilution events**
       - Must include OD readings for growth rate calculation
       - Dilution event data (volume, timestamp)
    
    3. **View Plots** at the top of the analysis panel
    
    4. **Adjust Settings** in the Controls tab
    """),
    pn.pane.Markdown("---"),
    pn.pane.Markdown("""
    ### About
    
    - **Batch Growth Rate**: Calculate growth rates, doubling times, and yields from OD measurements in batch culture
    
    - **Dilution Rate**: Analyze dilution rates, OD tracking, dosing intervals, and **calculate growth rates** in continuous culture
      - *Growth rate calculation*: When OD is steady, μ ≈ D (dilution rate). When OD changes, μ is calculated from both dilution and OD dynamics.
    """),
]

# Create the Panel template with a collapsible sidebar
app = pn.template.FastListTemplate(
    site="Pioreactor Analysis Tools",
    title="Pioreactor Analysis Tools",
    sidebar=sidebar_controls,
    main=[get_analysis_panel],
    accent_base_color="#5A7B9C",
    header_background="#393939",
    sidebar_width=280,
)

# Make the template servable
app.servable()

def main():
    """Main entry point for command-line execution"""
    port = int(os.environ.get("PORT", 7860))
    address = os.environ.get("ADDRESS", "0.0.0.0")
    show_browser = os.environ.get("SHOW_BROWSER", "true").lower() == "true"
    
    logger.info(f"Starting Panel server on {address}:{port}")
    logger.info(f"Open your browser to: http://localhost:{port}")
    
    pn.serve(
        app, 
        port=port, 
        address=address, 
        allow_websocket_origin=["*"], 
        show=show_browser,
        title="Pioreactor Analysis Panel"
    )

# For Hugging Face Spaces and direct execution
if __name__ == "__main__":
    main()