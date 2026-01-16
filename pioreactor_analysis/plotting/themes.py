"""
Journal-specific themes and plot configurations for publication-quality figures.

This module provides pre-configured settings for major scientific journals
including Nature, Science, ACS, and custom configurations.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Dict
import matplotlib.pyplot as plt
import matplotlib as mpl


class JournalTheme(Enum):
    """Pre-configured themes for major scientific journals."""
    NATURE = "nature"
    SCIENCE = "science"
    ACS = "acs"
    ELSEVIER = "elsevier"
    PLOS = "plos"
    CUSTOM = "custom"


# Colorblind-friendly color palettes
WONG_PALETTE = [
    '#E69F00',  # Orange
    '#56B4E9',  # Sky Blue
    '#009E73',  # Bluish Green
    '#F0E442',  # Yellow
    '#0072B2',  # Blue
    '#D55E00',  # Vermillion
    '#CC79A7',  # Reddish Purple
    '#000000',  # Black
]

TABLEAU_COLORBLIND = [
    '#1170aa', '#fc7d0b', '#a3acb9', '#57606c',
    '#5fa2ce', '#c85200', '#7b848f', '#a3cce9',
]

SCIENTIFIC_PALETTE = [
    '#0173B2',  # Blue
    '#DE8F05',  # Orange
    '#029E73',  # Green
    '#CC78BC',  # Pink
    '#CA9161',  # Tan
    '#949494',  # Gray
    '#ECE133',  # Yellow
    '#56B4E9',  # Light Blue
]


@dataclass
class PlotConfig:
    """
    Configuration for publication-quality plots.

    Attributes:
        width_inches: Figure width in inches
        height_inches: Figure height in inches
        dpi: Dots per inch for resolution
        font_family: Font family (e.g., 'Arial', 'Helvetica', 'Times New Roman')
        font_size_base: Base font size in points
        font_size_title: Title font size
        font_size_label: Axis label font size
        font_size_tick: Tick label font size
        font_size_legend: Legend font size
        line_width: Default line width
        marker_size: Default marker size
        color_palette: List of colors for multi-line plots
        grid_alpha: Transparency of grid lines (0-1)
        spine_width: Width of plot spines/borders
        legend_frameon: Whether legend has a frame
        tight_layout: Whether to use tight_layout
    """
    width_inches: float = 3.5
    height_inches: float = 2.5
    dpi: int = 300
    font_family: str = "Arial"
    font_size_base: int = 8
    font_size_title: int = 10
    font_size_label: int = 8
    font_size_tick: int = 7
    font_size_legend: int = 7
    line_width: float = 1.0
    marker_size: float = 3.0
    color_palette: List[str] = field(default_factory=lambda: WONG_PALETTE.copy())
    grid_alpha: float = 0.3
    spine_width: float = 0.8
    legend_frameon: bool = False
    tight_layout: bool = True

    @classmethod
    def from_journal(cls, journal: JournalTheme) -> 'PlotConfig':
        """
        Get pre-configured settings for a specific journal.

        Args:
            journal: JournalTheme enum specifying the journal

        Returns:
            PlotConfig with journal-specific settings

        Example:
            config = PlotConfig.from_journal(JournalTheme.NATURE)
            plotter = PublicationPlotter(config)
        """
        if journal == JournalTheme.NATURE:
            return cls(
                width_inches=3.5,  # 89 mm single column
                height_inches=2.5,
                dpi=300,
                font_family="Helvetica",
                font_size_base=7,
                font_size_title=8,
                font_size_label=7,
                font_size_tick=6,
                font_size_legend=6,
                line_width=0.8,
                marker_size=2.5,
                color_palette=WONG_PALETTE.copy(),
                spine_width=0.75,
            )

        elif journal == JournalTheme.SCIENCE:
            return cls(
                width_inches=3.3,  # ~84 mm single column
                height_inches=2.5,
                dpi=300,
                font_family="Helvetica",
                font_size_base=6,
                font_size_title=7,
                font_size_label=6,
                font_size_tick=5,
                font_size_legend=5,
                line_width=0.75,
                marker_size=2.0,
                color_palette=SCIENTIFIC_PALETTE.copy(),
                spine_width=0.7,
            )

        elif journal == JournalTheme.ACS:
            return cls(
                width_inches=3.25,  # ACS single column
                height_inches=2.5,
                dpi=600,  # ACS prefers higher DPI
                font_family="Arial",
                font_size_base=8,
                font_size_title=10,
                font_size_label=8,
                font_size_tick=7,
                font_size_legend=7,
                line_width=1.0,
                marker_size=3.0,
                color_palette=TABLEAU_COLORBLIND.copy(),
                spine_width=0.8,
            )

        elif journal == JournalTheme.ELSEVIER:
            return cls(
                width_inches=3.5,  # 90 mm single column
                height_inches=2.6,
                dpi=300,
                font_family="Arial",
                font_size_base=8,
                font_size_title=9,
                font_size_label=8,
                font_size_tick=7,
                font_size_legend=7,
                line_width=1.0,
                marker_size=3.0,
                color_palette=SCIENTIFIC_PALETTE.copy(),
                spine_width=0.8,
            )

        elif journal == JournalTheme.PLOS:
            return cls(
                width_inches=3.27,  # PLOS ONE single column
                height_inches=2.5,
                dpi=300,
                font_family="Arial",
                font_size_base=8,
                font_size_title=10,
                font_size_label=8,
                font_size_tick=7,
                font_size_legend=7,
                line_width=1.0,
                marker_size=3.0,
                color_palette=WONG_PALETTE.copy(),
                legend_frameon=True,  # PLOS prefers framed legends
                spine_width=0.8,
            )

        else:  # CUSTOM or default
            return cls()

    @classmethod
    def two_column(cls, journal: JournalTheme = JournalTheme.CUSTOM) -> 'PlotConfig':
        """
        Get settings for two-column (full-width) figures.

        Args:
            journal: Base journal theme to use

        Returns:
            PlotConfig with two-column width
        """
        config = cls.from_journal(journal)

        # Adjust widths for two-column format
        width_map = {
            JournalTheme.NATURE: 7.0,  # 183 mm
            JournalTheme.SCIENCE: 6.9,  # ~175 mm
            JournalTheme.ACS: 7.0,
            JournalTheme.ELSEVIER: 7.5,  # 190 mm
            JournalTheme.PLOS: 6.83,  # 173.5 mm
        }

        config.width_inches = width_map.get(journal, 7.0)
        config.height_inches = config.height_inches * 1.2  # Slightly taller for two-column

        return config

    def apply_to_matplotlib(self):
        """
        Apply this configuration to matplotlib's global rcParams.

        This affects all plots created after calling this method.

        Example:
            config = PlotConfig.from_journal(JournalTheme.NATURE)
            config.apply_to_matplotlib()

            # Now all plots will use Nature settings
            fig, ax = plt.subplots()
            ...
        """
        mpl.rcParams['figure.dpi'] = self.dpi
        mpl.rcParams['savefig.dpi'] = self.dpi
        mpl.rcParams['font.family'] = 'sans-serif'
        mpl.rcParams['font.sans-serif'] = [self.font_family]
        mpl.rcParams['font.size'] = self.font_size_base
        mpl.rcParams['axes.titlesize'] = self.font_size_title
        mpl.rcParams['axes.labelsize'] = self.font_size_label
        mpl.rcParams['xtick.labelsize'] = self.font_size_tick
        mpl.rcParams['ytick.labelsize'] = self.font_size_tick
        mpl.rcParams['legend.fontsize'] = self.font_size_legend
        mpl.rcParams['axes.linewidth'] = self.spine_width
        mpl.rcParams['xtick.major.width'] = self.spine_width
        mpl.rcParams['ytick.major.width'] = self.spine_width
        mpl.rcParams['xtick.minor.width'] = self.spine_width * 0.75
        mpl.rcParams['ytick.minor.width'] = self.spine_width * 0.75
        mpl.rcParams['lines.linewidth'] = self.line_width
        mpl.rcParams['lines.markersize'] = self.marker_size
        mpl.rcParams['grid.alpha'] = self.grid_alpha
        mpl.rcParams['legend.frameon'] = self.legend_frameon
        mpl.rcParams['figure.autolayout'] = self.tight_layout

        # Set color cycle
        mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=self.color_palette)

    def create_figure(
        self,
        nrows: int = 1,
        ncols: int = 1,
        **subplot_kw
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create a figure with this configuration.

        Args:
            nrows: Number of subplot rows
            ncols: Number of subplot columns
            **subplot_kw: Additional arguments passed to plt.subplots

        Returns:
            Tuple of (figure, axes)

        Example:
            config = PlotConfig.from_journal(JournalTheme.NATURE)
            fig, (ax1, ax2) = config.create_figure(nrows=2, ncols=1)
        """
        # Apply settings
        self.apply_to_matplotlib()

        # Calculate total height for subplots
        total_height = self.height_inches * nrows

        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(self.width_inches, total_height),
            **subplot_kw
        )

        return fig, axes


def get_color_palette(palette_name: str = 'wong') -> List[str]:
    """
    Get a colorblind-friendly color palette.

    Args:
        palette_name: Name of palette ('wong', 'tableau', 'scientific')

    Returns:
        List of hex color codes

    Example:
        colors = get_color_palette('wong')
        plt.plot(x, y, color=colors[0])
    """
    palettes = {
        'wong': WONG_PALETTE,
        'tableau': TABLEAU_COLORBLIND,
        'scientific': SCIENTIFIC_PALETTE,
    }

    return palettes.get(palette_name.lower(), WONG_PALETTE).copy()


def format_axis_scientific(
    ax: plt.Axes,
    axis: str = 'both',
    scilimits: Tuple[int, int] = (-3, 3)
):
    """
    Format axis to use scientific notation for large/small numbers.

    Args:
        ax: Matplotlib axes object
        axis: Which axis to format ('x', 'y', or 'both')
        scilimits: Powers of 10 outside which to use scientific notation

    Example:
        fig, ax = plt.subplots()
        ax.plot(x, y)
        format_axis_scientific(ax, axis='y')
    """
    if axis in ('x', 'both'):
        ax.ticklabel_format(style='scientific', axis='x', scilimits=scilimits)
    if axis in ('y', 'both'):
        ax.ticklabel_format(style='scientific', axis='y', scilimits=scilimits)


def add_panel_label(
    ax: plt.Axes,
    label: str,
    x: float = -0.15,
    y: float = 1.05,
    fontsize: int = 12,
    fontweight: str = 'bold'
):
    """
    Add a panel label (A, B, C, etc.) to a subplot.

    Args:
        ax: Matplotlib axes object
        label: Label text (e.g., 'A', 'B', 'C')
        x: X position in axes coordinates
        y: Y position in axes coordinates
        fontsize: Font size for label
        fontweight: Font weight ('bold', 'normal', etc.)

    Example:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        add_panel_label(ax1, 'A')
        add_panel_label(ax2, 'B')
    """
    ax.text(
        x, y, label,
        transform=ax.transAxes,
        fontsize=fontsize,
        fontweight=fontweight,
        va='top',
        ha='right'
    )
