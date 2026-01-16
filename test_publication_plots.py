"""
Test script for publication-quality plotting with real Pioreactor data.
"""

import sys
from pathlib import Path

# Add pioreactor_analysis to path
sys.path.insert(0, str(Path(__file__).parent))

from pioreactor_analysis import (
    PioreactorCSVParser,
    preprocess_od_data,
    auto_detect_exponential_phase,
    calculate_dilution_rate,
    calculate_growth_rate_continuous
)

from pioreactor_analysis.plotting.themes import PlotConfig, JournalTheme
from pioreactor_analysis.plotting.publication import PublicationPlotter

def main():
    print("=" * 80)
    print("Testing Publication-Quality Plotting")
    print("=" * 80)
    print()

    # Create output directory
    output_dir = Path("test_figures")
    output_dir.mkdir(exist_ok=True)
    print(f"Output directory: {output_dir.absolute()}")
    print()

    parser = PioreactorCSVParser()

    # Load data
    print("Loading data...")
    print("-" * 80)
    od_file = Path("data/export_20260109160338_p3/od_readings/od_readings-in_class_10.29-all_units-20260109110344.csv")
    dosing_file = Path("data/export_20260109160338_p3/dosing_automation_events/dosing_automation_events-in_class_10.29-all_units-20260109110344.csv")

    od_data = parser.parse(od_file)
    dosing_data = parser.parse(dosing_file)
    print(f"[OK] Loaded {len(od_data.od_readings)} OD readings")
    print(f"[OK] Loaded {len(dosing_data.dilution_events)} dilution events")
    print()

    # Preprocess OD data
    df_od = od_data.to_dataframe()
    df_processed = preprocess_od_data(
        df_od,
        smoothing_window=5,
        min_od_threshold=0.05,
        time_column='timestamp',
        od_column='od_value',
        group_by=['experiment', 'unit']
    )

    # Auto-detect exponential phase
    start, end, growth_result = auto_detect_exponential_phase(df_processed)
    print(f"Auto-detected exponential phase: {start:.2f} - {end:.2f} hours")
    print(f"Growth rate: {growth_result.growth_rate:.4f} h^-1")
    print()

    # Calculate dilution rates
    df_dosing = dosing_data.to_dilution_dataframe()
    df_dilution = calculate_dilution_rate(df_dosing, reactor_volume_ml=14.0)

    # Test 1: Nature-style Growth Curve
    print("\nTest 1: Nature-style Growth Curve")
    print("-" * 80)

    try:
        config = PlotConfig.from_journal(JournalTheme.NATURE)
        plotter = PublicationPlotter(config)

        fig, axes = plotter.plot_growth_curve(
            df_processed,
            growth_result,
            title="Batch Culture Growth Analysis",
            add_panel_labels=True
        )

        plotter.save(fig, output_dir / "nature_growth_curve", formats=['png', 'pdf'])
        plotter.close(fig)
        print("[OK] Created Nature-style growth curve")

    except Exception as e:
        print(f"[FAIL] Nature growth curve failed: {e}")
        import traceback
        traceback.print_exc()

    # Test 2: Science-style Growth Curve
    print("\nTest 2: Science-style Growth Curve")
    print("-" * 80)

    try:
        config = PlotConfig.from_journal(JournalTheme.SCIENCE)
        plotter = PublicationPlotter(config)

        fig, axes = plotter.plot_growth_curve(
            df_processed,
            growth_result,
            title="E. coli Growth in Minimal Medium",
            add_panel_labels=True
        )

        plotter.save(fig, output_dir / "science_growth_curve", formats=['png'])
        plotter.close(fig)
        print("[OK] Created Science-style growth curve")

    except Exception as e:
        print(f"[FAIL] Science growth curve failed: {e}")
        import traceback
        traceback.print_exc()

    # Test 3: ACS-style (High DPI)
    print("\nTest 3: ACS-style Growth Curve (High DPI)")
    print("-" * 80)

    try:
        config = PlotConfig.from_journal(JournalTheme.ACS)
        plotter = PublicationPlotter(config)

        fig, axes = plotter.plot_growth_curve(
            df_processed,
            growth_result,
            show_confidence_band=True,
            add_panel_labels=True
        )

        plotter.save(fig, output_dir / "acs_growth_curve", formats=['png', 'pdf'])
        plotter.close(fig)
        print("[OK] Created ACS-style growth curve (600 DPI)")

    except Exception as e:
        print(f"[FAIL] ACS growth curve failed: {e}")
        import traceback
        traceback.print_exc()

    # Test 4: Dilution Rate Plot
    print("\nTest 4: Dilution Rate Plot")
    print("-" * 80)

    try:
        config = PlotConfig.from_journal(JournalTheme.NATURE)
        plotter = PublicationPlotter(config)

        fig, ax = plotter.plot_dilution_rate(
            df_dilution,
            title="Turbidostat Operation",
            show_instant=True
        )

        plotter.save(fig, output_dir / "dilution_rate", formats=['png', 'pdf'])
        plotter.close(fig)
        print("[OK] Created dilution rate plot")

    except Exception as e:
        print(f"[FAIL] Dilution rate plot failed: {e}")
        import traceback
        traceback.print_exc()

    # Test 5: Continuous Culture Multi-Panel
    print("\nTest 5: Continuous Culture Multi-Panel Figure")
    print("-" * 80)

    try:
        # Calculate continuous growth rate
        df_dilution_with_hours = df_dilution.copy()
        start_time = df_dilution_with_hours['timestamp'].min()
        df_dilution_with_hours['elapsed_hours'] = (
            df_dilution_with_hours['timestamp'] - start_time
        ).dt.total_seconds() / 3600

        df_growth = calculate_growth_rate_continuous(
            df_processed,
            df_dilution_with_hours,
            od_column='od_smooth',
            time_column='elapsed_hours',
            dilution_column='moving_avg_dilution_rate'
        )

        config = PlotConfig.from_journal(JournalTheme.NATURE)
        config.height_inches = 4.5  # Taller for 3 panels
        plotter = PublicationPlotter(config)

        fig, axes = plotter.plot_continuous_culture(
            df_processed,
            df_dilution_with_hours,
            df_growth,
            title="Continuous Culture Dynamics",
            add_panel_labels=True
        )

        plotter.save(fig, output_dir / "continuous_culture", formats=['png', 'pdf'])
        plotter.close(fig)
        print("[OK] Created continuous culture multi-panel figure")

    except Exception as e:
        print(f"[FAIL] Continuous culture plot failed: {e}")
        import traceback
        traceback.print_exc()

    # Test 6: Growth Rate Comparison (simulated multiple experiments)
    print("\nTest 6: Growth Rate Comparison Plot")
    print("-" * 80)

    try:
        # Simulate multiple experiments with slight variations
        import copy
        results = [growth_result]

        # Create variant results
        for i in range(3):
            variant = copy.deepcopy(growth_result)
            variant.growth_rate *= (0.9 + i * 0.1)
            variant.doubling_time = 0.693 / variant.growth_rate
            variant.unit = f"Reactor {i+2}"
            results.append(variant)

        labels = ['Control', 'Treatment A', 'Treatment B', 'Treatment C']

        config = PlotConfig.from_journal(JournalTheme.NATURE)
        plotter = PublicationPlotter(config)

        fig, ax = plotter.plot_growth_rate_comparison(
            results,
            labels,
            title="Effect of Medium Composition on Growth Rate",
            show_error_bars=True
        )

        plotter.save(fig, output_dir / "growth_comparison", formats=['png', 'pdf'])
        plotter.close(fig)
        print("[OK] Created growth rate comparison plot")

    except Exception as e:
        print(f"[FAIL] Growth comparison plot failed: {e}")
        import traceback
        traceback.print_exc()

    # Test 7: Two-Column Figure
    print("\nTest 7: Two-Column Figure (Full Width)")
    print("-" * 80)

    try:
        config = PlotConfig.two_column(JournalTheme.NATURE)
        plotter = PublicationPlotter(config)

        fig, axes = plotter.plot_growth_curve(
            df_processed,
            growth_result,
            title="Full-Width Figure Example",
            add_panel_labels=True
        )

        plotter.save(fig, output_dir / "two_column_figure", formats=['png'])
        plotter.close(fig)
        print(f"[OK] Created two-column figure ({config.width_inches}\" wide)")

    except Exception as e:
        print(f"[FAIL] Two-column figure failed: {e}")
        import traceback
        traceback.print_exc()

    # Test 8: Dual Axis OD and Dilution Rate Plot
    print("\nTest 8: Dual Axis OD and Dilution Rate Plot")
    print("-" * 80)

    try:
        config = PlotConfig.from_journal(JournalTheme.NATURE)
        plotter = PublicationPlotter(config)

        # Using ln(OD) and showing max growth rate
        fig, axes = plotter.plot_od_and_dilution(
            df_processed,
            df_dilution,
            title="Chemostat Operation: OD and Dilution Rate",
            use_ln_od=True,
            max_growth_rate=growth_result.growth_rate
        )

        plotter.save(fig, output_dir / "dual_axis_od_dilution", formats=['png', 'pdf'])
        plotter.close(fig)
        print("[OK] Created dual axis OD and dilution rate plot")

    except Exception as e:
        print(f"[FAIL] Dual axis plot failed: {e}")
        import traceback
        traceback.print_exc()

    print()
    print("=" * 80)
    print("Testing complete!")
    print(f"Figures saved to: {output_dir.absolute()}")
    print("=" * 80)


if __name__ == "__main__":
    import pandas as pd
    main()
