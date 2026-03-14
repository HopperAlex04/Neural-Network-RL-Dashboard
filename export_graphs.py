"""
Generate stylized graph images from training metrics (post-training).
Uses the same config and data handling as the live dashboard; reads all
metrics once and writes one image per stat for reports/papers.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no GUI
import matplotlib.pyplot as plt

from config_loader import ConfigError, load_config
from metrics_reader import (
    MetricsReader,
    TRANEE_WIN_RATE_VS_GAPMAXIMIZER_KEY,
    is_training_only_stat,
)

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "stats_dashboard.json"

# No limit when exporting (we want full history)
EXPORT_MAX_POINTS = 10_000_000

# Styling: readable fonts, grid, consistent colors
FIGURE_DPI = 150
FIGURE_SIZE = (8, 4)
FONT_SIZE_TITLE = 12
FONT_SIZE_LABELS = 11
FONT_SIZE_TICKS = 10
LINE_WIDTH = 1.8
GRID_ALPHA = 0.4
PRIMARY_COLOR = "#2563eb"   # blue
SECONDARY_COLOR = "#059669" # teal (for scatter/bar accent)
# Rolling average for noisy metrics (same-graph overlay)
ROLLING_WINDOW = 20
RAW_ALPHA = 0.35  # transparency for raw series when rolling avg is shown
ROLLING_LINE_WIDTH = 2.0   # slightly thicker so trend stands out
STD_FILL_ALPHA = 0.25  # transparency for ±1 std shaded region
# Colors for multiple runs (one per metrics folder)
RUN_COLORS = [
    "#2563eb", "#059669", "#dc2626", "#7c3aed", "#ea580c",
    "#0891b2", "#ca8a04", "#db2777", "#4f46e5", "#0d9488",
]


def _rolling_average(x: list[float], y: list[float], window: int) -> tuple[list[float], list[float]]:
    """Compute rolling mean of y (same length); x is returned as-is for alignment."""
    if not y or window < 1:
        return (list(x), list(y))
    n = len(y)
    out_y: list[float] = []
    for i in range(n):
        start = max(0, i - window + 1)
        chunk = y[start : i + 1]
        out_y.append(sum(chunk) / len(chunk))
    return (list(x), out_y)


def _rolling_std(y: list[float], window: int) -> list[float]:
    """Compute rolling standard deviation of y (same length as y)."""
    if not y or window < 1:
        return list(y) if y else []
    n = len(y)
    out: list[float] = []
    for i in range(n):
        start = max(0, i - window + 1)
        chunk = y[start : i + 1]
        m = sum(chunk) / len(chunk)
        variance = sum((v - m) ** 2 for v in chunk) / len(chunk)
        out.append(variance ** 0.5)
    return out


def _x_label(stat_key: str) -> str:
    if stat_key == TRANEE_WIN_RATE_VS_GAPMAXIMIZER_KEY:
        return "Round"
    if is_training_only_stat(stat_key):
        return "Training episode"
    return "Episode"


def _run_label(metrics_path: Path) -> str:
    """Short label for a metrics path (folder name or file stem)."""
    p = Path(metrics_path).resolve()
    if p.is_file():
        return p.stem
    return p.name or str(p)

def _plot_series(
    ax,
    stat_key: str,
    label: str,
    graph_type: str,
    x: list[float],
    y: list[float],
    *,
    alpha: float = 1.0,
    color: str | None = None,
) -> None:
    if not x or not y:
        return
    c = color or PRIMARY_COLOR
    if graph_type == "line":
        ax.plot(x, y, color=c, linewidth=LINE_WIDTH, alpha=alpha, label=label)
    elif graph_type == "scatter":
        ax.scatter(x, y, color=c, s=8, alpha=min(alpha, 0.7), label=label)
    elif graph_type == "bar":
        ax.bar(x, y, color=c, alpha=0.8, width=0.8, label=label)
    elif graph_type == "histogram":
        ax.hist(y, bins=min(50, max(10, len(y) // 5)), color=c, alpha=0.8, edgecolor="white", linewidth=0.3, label=label)
    else:
        ax.plot(x, y, color=c, linewidth=LINE_WIDTH, alpha=alpha, label=label)


def _save_figure(
    out_dir: Path,
    stat_key: str,
    label: str,
    graph_type: str,
    series_list: list[tuple[str, list[float], list[float]]],
    formats: list[str],
    dpi: int,
) -> list[Path]:
    """Build one figure per stat and save in requested formats. series_list = [(run_label, x, y), ...]."""
    written: list[Path] = []
    # Drop runs with no data
    series_list = [(run_label, x, y) for run_label, x, y in series_list if x and y]
    if not series_list:
        return written

    # Safe filename from stat key
    base_name = stat_key.replace("/", "_").replace("\\", "_").strip() or "unnamed"

    fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=100)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fafafa")

    if graph_type in ("line", "scatter"):
        single_run = len(series_list) == 1
        for idx, (run_label, x, y) in enumerate(series_list):
            color = PRIMARY_COLOR if single_run else RUN_COLORS[idx % len(RUN_COLORS)]
            raw_label = "Raw" if single_run else None
            _plot_series(ax, stat_key, raw_label, graph_type, x, y, alpha=RAW_ALPHA, color=color)
            x_roll, y_roll = _rolling_average(x, y, ROLLING_WINDOW)
            std_roll = _rolling_std(y, ROLLING_WINDOW)
            if single_run:
                ax.fill_between(
                    x_roll,
                    [a - s for a, s in zip(y_roll, std_roll)],
                    [a + s for a, s in zip(y_roll, std_roll)],
                    color=color,
                    alpha=STD_FILL_ALPHA,
                    label=f"±1 std ({ROLLING_WINDOW})",
                )
            else:
                ax.fill_between(
                    x_roll,
                    [a - s for a, s in zip(y_roll, std_roll)],
                    [a + s for a, s in zip(y_roll, std_roll)],
                    color=color,
                    alpha=STD_FILL_ALPHA,
                )
            line_label = f"Rolling avg ({ROLLING_WINDOW})" if single_run else run_label
            ax.plot(
                x_roll,
                y_roll,
                color=color,
                linewidth=ROLLING_LINE_WIDTH,
                label=line_label,
            )
    else:
        for idx, (run_label, x, y) in enumerate(series_list):
            color = RUN_COLORS[idx % len(RUN_COLORS)]
            _plot_series(ax, stat_key, run_label, graph_type, x, y, color=color)

    if graph_type == "histogram":
        ax.set_xlabel(label, fontsize=FONT_SIZE_LABELS)
        ax.set_ylabel("Count", fontsize=FONT_SIZE_LABELS)
    else:
        ax.set_xlabel(_x_label(stat_key), fontsize=FONT_SIZE_LABELS)
        ax.set_ylabel(label, fontsize=FONT_SIZE_LABELS)
    ax.set_title(f"{label}", fontsize=FONT_SIZE_TITLE, fontweight="medium")
    ax.tick_params(axis="both", labelsize=FONT_SIZE_TICKS)
    ax.grid(True, alpha=GRID_ALPHA, linestyle="-")
    ax.legend(loc="best", fontsize=FONT_SIZE_TICKS)

    plt.tight_layout()

    for fmt in formats:
        ext = fmt.lower()
        if ext not in ("png", "svg", "pdf"):
            continue
        path = out_dir / f"{base_name}.{ext}"
        fig.savefig(path, dpi=dpi if ext == "png" else None, bbox_inches="tight", format=ext)
        written.append(path)

    plt.close(fig)
    return written


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export training metrics as stylized graph images (PNG/SVG)."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to stats_dashboard.json",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        nargs="+",
        required=True,
        metavar="PATH",
        help="One or more paths to metrics JSONL file(s) or directory(ies) containing *.jsonl. Each path is one run; all runs are overlaid on the same graph per stat.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("exported_graphs"),
        help="Output directory for image files",
    )
    parser.add_argument(
        "--format",
        choices=["png", "svg", "pdf", "png,svg", "png,pdf", "svg,pdf", "all"],
        default="png",
        help="Output format(s). 'all' = png, svg, pdf",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=FIGURE_DPI,
        help="DPI for PNG output (default %(default)s)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        config = load_config(args.config)
    except ConfigError as e:
        print(f"Config error: {e}")
        return

    metrics_paths = [Path(p) for p in args.metrics]
    for p in metrics_paths:
        if not p.exists():
            print(f"Metrics path does not exist: {p}")
            return

    if args.format == "all":
        formats = ["png", "svg", "pdf"]
    else:
        formats = [s.strip() for s in args.format.split(",")]

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load each metrics path into its own reader; aggregate by stat as (run_label, x, y) per stat
    stat_keys = config.stat_keys()
    series_by_stat: dict[str, list[tuple[str, list[float], list[float]]]] = {
        k: [] for k in stat_keys
    }
    for metrics_path in metrics_paths:
        reader = MetricsReader(stat_keys=stat_keys, max_points=EXPORT_MAX_POINTS)
        reader.add_path(metrics_path)
        reader.poll()
        run_label = _run_label(metrics_path)
        for key in stat_keys:
            x, y = reader.get_series(key)
            series_by_stat[key].append((run_label, x, y))

    graph_types = {s.key: s.graph_type for s in config.stats}
    written: list[Path] = []
    for s in config.stats:
        series_list = series_by_stat.get(s.key, [])
        paths = _save_figure(
            out_dir,
            s.key,
            s.label,
            graph_types.get(s.key, "line"),
            series_list,
            formats,
            args.dpi,
        )
        written.extend(paths)

    print(f"Exported {len(written)} file(s) to {out_dir.resolve()}")
    for p in sorted(written):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
