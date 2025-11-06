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

# Create an app selector with lazy loading
class AppSelector(param.Parameterized):
    current_app = param.Selector(default="Batch Growth Rate", 
                               objects=["Batch Growth Rate", "Dilution Rate"])
    
    def __init__(self, **params):
        super().__init__(**params)
        # Create app instance cache but don't initialize anything yet
        self._app_instances = {}
        
    def _get_app_instance(self, app_name):
        """Lazy initializer for applications"""
        if app_name not in self._app_instances:
            logger.info(f"Initializing application: {app_name}")
            
            if app_name == "Batch Growth Rate":
                self._app_instances[app_name] = GrowthRateAnalysis().view()
            elif app_name == "Dilution Rate":
                self._app_instances[app_name] = PioreactorAnalysis().view()
            
            logger.info(f"{app_name} application initialized successfully")
        
        return self._app_instances[app_name]
    
    @param.depends('current_app')
    def view(self):
        return self._get_app_instance(self.current_app)

app_selector = AppSelector()

# Create the Panel template
app = pn.template.FastListTemplate(
    site="Pioreactor Analysis Tools",
    title="Pioreactor Analysis Tools",
    sidebar=[
        pn.pane.Markdown("## Navigation"),
        app_selector.param.current_app,
        pn.pane.Markdown("### About"),
        pn.pane.Markdown("""
        This application provides analysis tools for Pioreactor data:
        - Batch Growth Rate Analysis: Analyze growth rates from batch culture experiments
        - Dilution Rate Analysis: Analyze dilution rates in continuous culture
        """)
    ],
    main=[
        app_selector.view
    ],
    accent_base_color="#5A7B9C",
    header_background="#393939",
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