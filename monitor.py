"""
Real-time monitoring dashboard for Honors-Thesis-Project metrics.
Reads config from JSON and tails metrics JSONL; displays one plot per stat.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import dearpygui.dearpygui as dpg

from config_loader import ConfigError, load_config
from metrics_reader import (
    MetricsReader,
    TRANEE_WIN_RATE_VS_GAPMAXIMIZER_KEY,
    is_training_only_stat,
)

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "stats_dashboard.json"

MAIN_WINDOW_TAG = "main_window"
SERIES_TAG_PREFIX = "series_"
POLL_INTERVAL = 1.5  # seconds


def _parse_args():
    parser = argparse.ArgumentParser(description="Real-time metrics dashboard for Honors-Thesis-Project")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to stats_dashboard.json",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=None,
        help="Path to metrics JSONL file or directory containing *.jsonl",
    )
    return parser.parse_args()


def _build_plot_for_stat(stat_key: str, label: str, graph_type: str) -> str:
    """Create one plot with X/Y axes and the appropriate series type. Returns series tag."""
    if stat_key == TRANEE_WIN_RATE_VS_GAPMAXIMIZER_KEY:
        x_label = "Round"
    elif is_training_only_stat(stat_key):
        x_label = "Training episode"
    else:
        x_label = "Episode"
    with dpg.child_window(width=480, height=220, border=True):
        dpg.add_text(f"{label} ({stat_key})", color=(100, 200, 255))
        with dpg.plot(label=label, height=180, width=460):
            dpg.add_plot_axis(dpg.mvXAxis, label=x_label)
            y_axis = dpg.add_plot_axis(dpg.mvYAxis, label=label)
            tag = f"{SERIES_TAG_PREFIX}{stat_key}"
            if graph_type == "line":
                dpg.add_line_series([], [], label=label, parent=y_axis, tag=tag)
            elif graph_type == "scatter":
                dpg.add_scatter_series([], [], label=label, parent=y_axis, tag=tag)
            elif graph_type == "bar":
                dpg.add_bar_series([], [], label=label, parent=y_axis, tag=tag)
            elif graph_type == "histogram":
                dpg.add_histogram_series([], label=label, parent=y_axis, tag=tag)
            else:
                dpg.add_line_series([], [], label=label, parent=y_axis, tag=tag)
            dpg.add_plot_legend()
    return tag


def _build_ui(config, metrics_reader: MetricsReader | None) -> dict[str, str]:
    """Build main window with one plot per stat. Returns dict of stat_key -> series tag."""
    series_tags: dict[str, str] = {}
    with dpg.window(label="Honors-Thesis Metrics Monitor", tag=MAIN_WINDOW_TAG):
        dpg.add_text("Metrics (real-time)", color=(150, 200, 255))
        if not metrics_reader and not config.metrics_path:
            dpg.add_text("No metrics path set. Use --metrics path/to/metrics_logs or set metrics_path in config.", color=(255, 180, 100))
        with dpg.child_window(border=True):
            for s in config.stats:
                tag = _build_plot_for_stat(s.key, s.label, s.graph_type)
                series_tags[s.key] = tag
    return series_tags


def _run_poll_callback(reader: MetricsReader, series_tags: dict[str, str], stat_keys: list[str], graph_types: dict[str, str]):
    """One tick: poll reader and update all series."""
    reader.poll()
    for key in stat_keys:
        tag = series_tags.get(key)
        if tag is None:
            continue
        try:
            x, y = reader.get_series(key)
            # Histogram series typically takes (values,) only
            if graph_types.get(key) == "histogram":
                dpg.set_value(tag, [list(y)])
            else:
                # Plot series expect [x_data, y_data] as a list (not tuple)
                dpg.set_value(tag, [list(x), list(y)])
        except Exception:
            pass


def main() -> None:
    args = _parse_args()
    try:
        config = load_config(args.config)
    except ConfigError as e:
        print(f"Config error: {e}")
        return

    metrics_path = args.metrics or (config.metrics_path if config.metrics_path else None)
    reader: MetricsReader | None = None
    if metrics_path:
        reader = MetricsReader(stat_keys=config.stat_keys())
        reader.add_path(Path(metrics_path))

    dpg.create_context()
    series_tags = _build_ui(config, reader)

    # Initial data if we have a reader
    if reader:
        reader.poll()
        for s in config.stats:
            tag = series_tags.get(s.key)
            if tag:
                try:
                    x, y = reader.get_series(s.key)
                    if s.graph_type == "histogram":
                        dpg.set_value(tag, [list(y)])
                    else:
                        dpg.set_value(tag, [list(x), list(y)])
                except Exception:
                    pass

    dpg.create_viewport(title="Honors-Thesis Metrics Monitor", width=900, height=640)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window(MAIN_WINDOW_TAG, True)

    graph_types = {s.key: s.graph_type for s in config.stats}
    # Polling: register a timer to update series periodically
    if reader and series_tags:

        def _tick():
            _run_poll_callback(reader, series_tags, config.stat_keys(), graph_types)

        dpg.set_frame_callback(60, _tick)  # run after a short delay
        # Use a repeating timer via schedule callback pattern: re-register each time
        def _schedule_next():
            _tick()
            dpg.set_frame_callback(90, _schedule_next)  # ~1.5s at 60fps ≈ 90 frames
        # Start repeating after first tick
        dpg.set_frame_callback(90, _schedule_next)

    dpg.start_dearpygui()
    dpg.destroy_context()


if __name__ == "__main__":
    main()
