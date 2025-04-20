# app.py - Entry point for Hugging Face Spaces
import panel as pn
from pioreactor_panel import PioreactorAnalysis

# Initialize the application
pioreactor_analysis = PioreactorAnalysis()

# Set up the Panel template
pn.extension(template="fast", sizing_mode="stretch_width", notifications=True)

# Make the app servable for Hugging Face Spaces
pn.template.FastListTemplate(
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
).servable()
