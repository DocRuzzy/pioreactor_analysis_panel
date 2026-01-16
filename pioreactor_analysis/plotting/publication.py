"""
Publication-quality matplotlib plotting for Pioreactor analysis.

This module provides high-level plotting functions that create publication-ready
figures with proper formatting, labels, and styling for scientific journals.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Optional, List, Tuple, Union
from datetime import datetime

from pioreactor_analysis.plotting.themes import PlotConfig, JournalTheme, add_panel_label
from pioreactor_analysis.core.data_models import GrowthRateResult, YieldResult


class PublicationPlotter:
    """
    Create publication-quality matplotlib figures for Pioreactor analysis.

    This class provides methods for creating properly formatted figures
    suitable for submission to scientific journals.

    Attributes:
        config: PlotConfig with formatting settings

    Example:
        from pioreactor_analysis.plotting import PublicationPlotter, JournalTheme, PlotConfig

        # Create plotter with Nature formatting
        config = PlotConfig.from_journal(JournalTheme.NATURE)
        plotter = PublicationPlotter(config)

        # Plot growth curve
        fig, axes = plotter.plot_growth_curve(df_od, growth_result)

        # Save in multiple formats
        plotter.save(fig, Path("figure1"), formats=['png', 'pdf', 'svg'])
    """

    def __init__(self, config: Optional[PlotConfig] = None):
        """
        Initialize plotter with configuration.

        Args:
            config: PlotConfig with formatting settings.
                   If None, uses default configuration.
        """
        self.config = config or PlotConfig()
        self.config.apply_to_matplotlib()

    def plot_growth_curve(
        self,
        od_data: pd.DataFrame,
        growth_result: Optional[GrowthRateResult] = None,
        od_column: str = 'od_smooth',
        time_column: str = 'elapsed_hours',
        show_regression: bool = True,
        show_confidence_band: bool = True,
        title: Optional[str] = None,
        add_panel_labels: bool = False
    ) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
        """
        Create a 2-panel growth curve figure.

        Panels:
        - Top: OD vs time (semi-log scale)
        - Bottom: ln(OD) vs time with regression line

        Args:
            od_data: DataFrame with OD and time data
            growth_result: Optional GrowthRateResult to show regression
            od_column: Name of OD column (default: 'od_smooth')
            time_column: Name of time column (default: 'elapsed_hours')
            show_regression: Whether to show regression line (default: True)
            show_confidence_band: Whether to show 95% CI band (default: True)
            title: Optional figure title
            add_panel_labels: Whether to add 'A', 'B' panel labels (default: False)

        Returns:
            Tuple of (figure, (ax_od, ax_ln))

        Example:
            fig, (ax1, ax2) = plotter.plot_growth_curve(
                df_processed,
                growth_result,
                title="E. coli Growth in LB Medium"
            )
            plotter.save(fig, Path("growth_curve"))
        """
        # Create figure with 2 subplots
        fig, (ax_od, ax_ln) = self.config.create_figure(
            nrows=2,
            ncols=1,
            sharex=True,
            gridspec_kw={'hspace': 0.15}
        )

        # Get colors from palette
        data_color = self.config.color_palette[0]
        fit_color = self.config.color_palette[1]
        ci_color = self.config.color_palette[2]

        # Panel A: OD vs time (semi-log)
        ax_od.semilogy(
            od_data[time_column],
            od_data[od_column],
            'o',
            markersize=self.config.marker_size,
            color=data_color,
            alpha=0.6,
            label='Data'
        )

        ax_od.set_ylabel('OD', fontsize=self.config.font_size_label)
        ax_od.grid(True, alpha=self.config.grid_alpha, which='both')
        ax_od.legend(frameon=self.config.legend_frameon, fontsize=self.config.font_size_legend)

        if add_panel_labels:
            add_panel_label(ax_od, 'A', fontsize=self.config.font_size_title)

        # Panel B: ln(OD) vs time
        if 'ln_od' not in od_data.columns:
            ln_od = np.log(od_data[od_column].clip(lower=1e-6))
        else:
            ln_od = od_data['ln_od']

        ax_ln.plot(
            od_data[time_column],
            ln_od,
            'o',
            markersize=self.config.marker_size,
            color=data_color,
            alpha=0.6,
            label='Data'
        )

        # Add regression line if provided
        if growth_result and show_regression:
            # Extract regression window
            time_fit = np.array([growth_result.start_time, growth_result.end_time])
            ln_od_fit = growth_result.growth_rate * time_fit + growth_result.intercept

            ax_ln.plot(
                time_fit,
                ln_od_fit,
                '-',
                linewidth=self.config.line_width * 1.5,
                color=fit_color,
                label=f'μ = {growth_result.growth_rate:.3f} h⁻¹\n' +
                      f'R² = {growth_result.r_squared:.3f}'
            )

            # Add confidence band
            if show_confidence_band:
                # Calculate confidence interval for regression line
                n = growth_result.n_points
                t_val = 1.96  # Approximate for large n

                # CI width increases with distance from center
                time_center = (growth_result.start_time + growth_result.end_time) / 2
                ci_width = growth_result.std_error * t_val

                ln_od_lower = ln_od_fit - ci_width
                ln_od_upper = ln_od_fit + ci_width

                ax_ln.fill_between(
                    time_fit,
                    ln_od_lower,
                    ln_od_upper,
                    color=ci_color,
                    alpha=0.3,
                    label='95% CI'
                )

        ax_ln.set_xlabel('Time (hours)', fontsize=self.config.font_size_label)
        ax_ln.set_ylabel('ln(OD)', fontsize=self.config.font_size_label)
        ax_ln.grid(True, alpha=self.config.grid_alpha)
        ax_ln.legend(frameon=self.config.legend_frameon, fontsize=self.config.font_size_legend)

        if add_panel_labels:
            add_panel_label(ax_ln, 'B', fontsize=self.config.font_size_title)

        # Add title if provided
        if title:
            fig.suptitle(title, fontsize=self.config.font_size_title, y=0.98)

        if self.config.tight_layout:
            fig.tight_layout()

        return fig, (ax_od, ax_ln)

    def plot_dilution_rate(
        self,
        dilution_data: pd.DataFrame,
        time_column: str = 'timestamp',
        dilution_column: str = 'moving_avg_dilution_rate',
        show_instant: bool = True,
        title: Optional[str] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot dilution rate vs time for continuous culture.

        Args:
            dilution_data: DataFrame with dilution rate data
            time_column: Name of time column (default: 'timestamp')
            dilution_column: Name of dilution rate column (default: 'moving_avg_dilution_rate')
            show_instant: Whether to show instantaneous dilution rate (default: True)
            title: Optional figure title

        Returns:
            Tuple of (figure, axes)

        Example:
            fig, ax = plotter.plot_dilution_rate(
                df_dilution,
                title="Turbidostat Dilution Rate"
            )
        """
        fig, ax = self.config.create_figure()

        # Get colors
        instant_color = self.config.color_palette[0]
        smooth_color = self.config.color_palette[1]

        # Convert timestamp if needed
        if pd.api.types.is_datetime64_any_dtype(dilution_data[time_column]):
            # Calculate elapsed hours from start
            start_time = dilution_data[time_column].min()
            time_vals = (dilution_data[time_column] - start_time).dt.total_seconds() / 3600
            xlabel = 'Time (hours)'
        else:
            time_vals = dilution_data[time_column]
            xlabel = 'Time (hours)' if 'hour' in time_column.lower() else 'Time'

        # Plot instantaneous rate if requested
        if show_instant and 'instant_dilution_rate' in dilution_data.columns:
            ax.scatter(
                time_vals,
                dilution_data['instant_dilution_rate'],
                s=self.config.marker_size**2,
                color=instant_color,
                alpha=0.3,
                label='Instantaneous'
            )

        # Plot smoothed rate
        ax.plot(
            time_vals,
            dilution_data[dilution_column],
            '-',
            linewidth=self.config.line_width * 1.5,
            color=smooth_color,
            label='Moving Average'
        )

        ax.set_xlabel(xlabel, fontsize=self.config.font_size_label)
        ax.set_ylabel('Dilution Rate (h⁻¹)', fontsize=self.config.font_size_label)
        ax.grid(True, alpha=self.config.grid_alpha)
        ax.legend(frameon=self.config.legend_frameon, fontsize=self.config.font_size_legend)

        if title:
            ax.set_title(title, fontsize=self.config.font_size_title)

        if self.config.tight_layout:
            fig.tight_layout()

        return fig, ax

    def plot_continuous_culture(
        self,
        od_data: pd.DataFrame,
        dilution_data: pd.DataFrame,
        growth_data: Optional[pd.DataFrame] = None,
        time_column: str = 'elapsed_hours',
        title: Optional[str] = None,
        add_panel_labels: bool = True
    ) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes, Optional[plt.Axes]]]:
        """
        Create a multi-panel continuous culture figure.

        Panels:
        - Panel A: OD vs time
        - Panel B: Dilution rate vs time
        - Panel C (optional): Growth rate vs time

        Args:
            od_data: DataFrame with OD data
            dilution_data: DataFrame with dilution rate data
            growth_data: Optional DataFrame with growth rate data
            time_column: Name of time column (default: 'elapsed_hours')
            title: Optional figure title
            add_panel_labels: Whether to add 'A', 'B', 'C' labels (default: True)

        Returns:
            Tuple of (figure, (ax_od, ax_dilution, ax_growth))

        Example:
            fig, axes = plotter.plot_continuous_culture(
                df_od,
                df_dilution,
                df_growth,
                title="Chemostat at D = 0.5 h⁻¹"
            )
        """
        n_panels = 3 if growth_data is not None else 2

        fig, axes = self.config.create_figure(
            nrows=n_panels,
            ncols=1,
            sharex=True,
            gridspec_kw={'hspace': 0.2}
        )

        if n_panels == 2:
            ax_od, ax_dilution = axes
            ax_growth = None
        else:
            ax_od, ax_dilution, ax_growth = axes

        # Colors
        color_od = self.config.color_palette[0]
        color_dilution = self.config.color_palette[1]
        color_growth = self.config.color_palette[2]

        # Panel A: OD vs time
        if 'od_smooth' in od_data.columns:
            od_col = 'od_smooth'
        elif 'od_value' in od_data.columns:
            od_col = 'od_value'
        else:
            od_col = od_data.select_dtypes(include=[np.number]).columns[1]  # First numeric column after time

        ax_od.plot(
            od_data[time_column],
            od_data[od_col],
            '-',
            linewidth=self.config.line_width,
            color=color_od
        )
        ax_od.set_ylabel('OD', fontsize=self.config.font_size_label)
        ax_od.grid(True, alpha=self.config.grid_alpha)

        if add_panel_labels:
            add_panel_label(ax_od, 'A', fontsize=self.config.font_size_title)

        # Panel B: Dilution rate vs time
        if 'moving_avg_dilution_rate' in dilution_data.columns:
            dilution_col = 'moving_avg_dilution_rate'
        else:
            dilution_col = 'instant_dilution_rate'

        # Align time axes
        if time_column in dilution_data.columns:
            time_dilution = dilution_data[time_column]
        else:
            # Convert timestamp to elapsed hours
            start_time = dilution_data['timestamp'].min()
            time_dilution = (dilution_data['timestamp'] - start_time).dt.total_seconds() / 3600

        ax_dilution.plot(
            time_dilution,
            dilution_data[dilution_col],
            '-',
            linewidth=self.config.line_width,
            color=color_dilution
        )
        ax_dilution.set_ylabel('Dilution Rate (h⁻¹)', fontsize=self.config.font_size_label)
        ax_dilution.grid(True, alpha=self.config.grid_alpha)

        if add_panel_labels:
            add_panel_label(ax_dilution, 'B', fontsize=self.config.font_size_title)

        # Panel C: Growth rate vs time (optional)
        if growth_data is not None and ax_growth is not None:
            # Plot growth rate
            ax_growth.plot(
                growth_data[time_column],
                growth_data['growth_rate'],
                '-',
                linewidth=self.config.line_width,
                color=color_growth,
                label='μ'
            )

            # Also plot dilution rate for comparison
            ax_growth.plot(
                growth_data[time_column],
                growth_data['dilution_rate'],
                '--',
                linewidth=self.config.line_width * 0.8,
                color=color_dilution,
                alpha=0.7,
                label='D'
            )

            # Highlight steady-state regions if available
            if 'steady_state' in growth_data.columns:
                steady_mask = growth_data['steady_state']
                if steady_mask.any():
                    ax_growth.fill_between(
                        growth_data[time_column],
                        ax_growth.get_ylim()[0],
                        ax_growth.get_ylim()[1],
                        where=steady_mask,
                        alpha=0.2,
                        color='gray',
                        label='Steady State'
                    )

            ax_growth.set_ylabel('Growth Rate (h⁻¹)', fontsize=self.config.font_size_label)
            ax_growth.set_xlabel('Time (hours)', fontsize=self.config.font_size_label)
            ax_growth.grid(True, alpha=self.config.grid_alpha)
            ax_growth.legend(frameon=self.config.legend_frameon, fontsize=self.config.font_size_legend)

            if add_panel_labels:
                add_panel_label(ax_growth, 'C', fontsize=self.config.font_size_title)
        else:
            ax_dilution.set_xlabel('Time (hours)', fontsize=self.config.font_size_label)

        # Add title if provided
        if title:
            fig.suptitle(title, fontsize=self.config.font_size_title, y=0.98)

        if self.config.tight_layout:
            fig.tight_layout()

        return fig, (ax_od, ax_dilution, ax_growth)

    def plot_growth_rate_comparison(
        self,
        results: List[GrowthRateResult],
        labels: Optional[List[str]] = None,
        title: Optional[str] = None,
        ylabel: str = 'Growth Rate (h⁻¹)',
        show_error_bars: bool = True,
        show_individual_points: bool = True
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create a bar chart comparing growth rates from multiple experiments.

        Args:
            results: List of GrowthRateResult objects
            labels: Optional list of labels for each result
            title: Optional figure title
            ylabel: Y-axis label (default: 'Growth Rate (h⁻¹)')
            show_error_bars: Whether to show 95% CI error bars (default: True)
            show_individual_points: Whether to show individual data points (default: True)

        Returns:
            Tuple of (figure, axes)

        Example:
            results = [result1, result2, result3]
            labels = ['Control', 'Treatment A', 'Treatment B']
            fig, ax = plotter.plot_growth_rate_comparison(
                results,
                labels,
                title="Effect of Temperature on Growth Rate"
            )
        """
        fig, ax = self.config.create_figure()

        n_results = len(results)

        # Use unit names as labels if not provided
        if labels is None:
            labels = [r.unit or f'Exp {i+1}' for i, r in enumerate(results)]

        # Extract growth rates and errors
        growth_rates = [r.growth_rate for r in results]
        if show_error_bars:
            errors = [(r.ci_upper - r.ci_lower) / 2 for r in results]
        else:
            errors = None

        # Create bar chart
        x_pos = np.arange(n_results)
        bars = ax.bar(
            x_pos,
            growth_rates,
            yerr=errors,
            capsize=3,
            color=self.config.color_palette[0],
            alpha=0.7,
            edgecolor='black',
            linewidth=self.config.spine_width
        )

        # Add individual points if requested
        if show_individual_points:
            ax.scatter(
                x_pos,
                growth_rates,
                s=self.config.marker_size**2 * 3,
                color='black',
                zorder=3
            )

        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel(ylabel, fontsize=self.config.font_size_label)
        ax.grid(True, alpha=self.config.grid_alpha, axis='y')

        if title:
            ax.set_title(title, fontsize=self.config.font_size_title)

        if self.config.tight_layout:
            fig.tight_layout()

        return fig, ax

    def plot_od_and_dilution(
        self,
        od_data: pd.DataFrame,
        dilution_data: pd.DataFrame,
        od_column: str = 'od_smooth',
        dilution_column: str = 'moving_avg_dilution_rate',
        time_column: str = 'elapsed_hours',
        use_ln_od: bool = False,
        max_growth_rate: Optional[float] = None,
        title: Optional[str] = None
    ) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
        """
        Create a dual-axis plot with OD and Dilution Rate vs Time.

        Args:
            od_data: DataFrame with OD data
            dilution_data: DataFrame with dilution rate data
            od_column: Name of OD column (default: 'od_smooth')
            dilution_column: Name of dilution rate column (default: 'moving_avg_dilution_rate')
            time_column: Name of time column (default: 'elapsed_hours')
            use_ln_od: Whether to plot ln(OD) instead of OD (default: False)
            max_growth_rate: Optional maximum specific growth rate to plot as reference (default: None)
            title: Optional figure title

        Returns:
            Tuple of (figure, (ax_od, ax_dilution))
        """
        fig, ax_od = self.config.create_figure()
        ax_dilution = ax_od.twinx()

        # Colors
        color_od = self.config.color_palette[0]
        color_dilution = self.config.color_palette[1]
        color_max_growth = self.config.color_palette[2]

        # Prepare OD data
        od_vals = od_data[od_column]
        if use_ln_od:
            if 'ln_od' in od_data.columns:
                 od_vals = od_data['ln_od']
            else:
                 od_vals = np.log(od_data[od_column].clip(lower=1e-6))
            od_label = 'ln(OD)'
        else:
            od_label = 'OD'

        # Plot OD
        line_od = ax_od.plot(
            od_data[time_column],
            od_vals,
            '-',
            linewidth=self.config.line_width,
            color=color_od,
            label=od_label
        )
        
        ax_od.set_xlabel('Time (hours)', fontsize=self.config.font_size_label)
        ax_od.set_ylabel(od_label, fontsize=self.config.font_size_label, color=color_od)
        ax_od.tick_params(axis='y', labelcolor=color_od)
        ax_od.grid(True, alpha=self.config.grid_alpha)

        # Prepare Dilution Data
        if time_column in dilution_data.columns:
            time_dilution = dilution_data[time_column]
        else:
             # Convert timestamp to elapsed hours if needed, aligning with od_data logic if possible
             # For now, best effort if time_column missing implies 'timestamp' exists and needs converting relative to start
             # Assuming dilution_data has timestamps
             start_time = dilution_data['timestamp'].min()
             time_dilution = (dilution_data['timestamp'] - start_time).dt.total_seconds() / 3600

        # Plot Dilution Rate
        line_dilution = ax_dilution.plot(
            time_dilution,
            dilution_data[dilution_column],
            '-',
            linewidth=self.config.line_width,
            color=color_dilution,
            label='Dilution Rate'
        )
        
        ax_dilution.set_ylabel('Dilution Rate (h⁻¹)', fontsize=self.config.font_size_label, color=color_dilution)
        ax_dilution.tick_params(axis='y', labelcolor=color_dilution)

        # Plot Max Growth Rate if provided
        lines = line_od + line_dilution
        if max_growth_rate is not None:
             line_max = ax_dilution.axhline(
                 y=max_growth_rate,
                 color=color_max_growth,
                 linestyle='--',
                 linewidth=self.config.line_width,
                 label=f'Max Growth Rate ({max_growth_rate:.3f} h⁻¹)'
             )
             lines.append(line_max)

        # Legend
        labels = [l.get_label() for l in lines]
        ax_od.legend(lines, labels, loc='best', frameon=self.config.legend_frameon, fontsize=self.config.font_size_legend)

        if title:
            ax_od.set_title(title, fontsize=self.config.font_size_title)

        if self.config.tight_layout:
             fig.tight_layout()

        return fig, (ax_od, ax_dilution)

    def save(
        self,
        fig: plt.Figure,
        filepath: Union[str, Path],
        formats: List[str] = ['png', 'pdf']
    ):
        """
        Save figure in multiple formats.

        Args:
            fig: Matplotlib figure to save
            filepath: Base path for saving (without extension)
            formats: List of format extensions (default: ['png', 'pdf'])

        Example:
            plotter.save(fig, Path("figures/figure1"), formats=['png', 'pdf', 'svg'])

            # Creates:
            # figures/figure1.png
            # figures/figure1.pdf
            # figures/figure1.svg
        """
        filepath = Path(filepath)

        # Create parent directory if it doesn't exist
        filepath.parent.mkdir(parents=True, exist_ok=True)

        for fmt in formats:
            output_path = filepath.with_suffix(f'.{fmt}')
            fig.savefig(
                output_path,
                dpi=self.config.dpi,
                bbox_inches='tight',
                format=fmt
            )
            print(f"Saved: {output_path}")

    def close(self, fig: Optional[plt.Figure] = None):
        """
        Close figure to free memory.

        Args:
            fig: Figure to close. If None, closes all figures.

        Example:
            plotter.close(fig)  # Close specific figure
            plotter.close()     # Close all figures
        """
        if fig is None:
            plt.close('all')
        else:
            plt.close(fig)
