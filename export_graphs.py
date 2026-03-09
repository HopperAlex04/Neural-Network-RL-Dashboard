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


def _x_label(stat_key: str) -> str:
    if stat_key == TRANEE_WIN_RATE_VS_GAPMAXIMIZER_KEY:
        return "Round"
    if is_training_only_stat(stat_key):
        return "Training episode"
    return "Episode"


def _plot_series(ax, stat_key: str, label: str, graph_type: str, x: list[float], y: list[float]) -> None:
    if not x or not y:
        return
    if graph_type == "line":
        ax.plot(x, y, color=PRIMARY_COLOR, linewidth=LINE_WIDTH, label=label)
    elif graph_type == "scatter":
        ax.scatter(x, y, color=PRIMARY_COLOR, s=8, alpha=0.7, label=label)
    elif graph_type == "bar":
        ax.bar(x, y, color=PRIMARY_COLOR, alpha=0.8, width=0.8, label=label)
    elif graph_type == "histogram":
        ax.hist(y, bins=min(50, max(10, len(y) // 5)), color=PRIMARY_COLOR, alpha=0.8, edgecolor="white", linewidth=0.3, label=label)
    else:
        ax.plot(x, y, color=PRIMARY_COLOR, linewidth=LINE_WIDTH, label=label)


def _save_figure(
    out_dir: Path,
    stat_key: str,
    label: str,
    graph_type: str,
    x: list[float],
    y: list[float],
    formats: list[str],
    dpi: int,
) -> list[Path]:
    """Build one figure per stat and save in requested formats. Returns paths written."""
    written: list[Path] = []
    if not x and not y:
        return written

    # Safe filename from stat key
    base_name = stat_key.replace("/", "_").replace("\\", "_").strip() or "unnamed"

    fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=100)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fafafa")

    _plot_series(ax, stat_key, label, graph_type, x, y)

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
        required=True,
        help="Path to metrics JSONL file or directory containing *.jsonl",
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

    metrics_path = Path(args.metrics)
    if not metrics_path.exists():
        print(f"Metrics path does not exist: {metrics_path}")
        return

    if args.format == "all":
        formats = ["png", "svg", "pdf"]
    else:
        formats = [s.strip() for s in args.format.split(",")]

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    reader = MetricsReader(stat_keys=config.stat_keys(), max_points=EXPORT_MAX_POINTS)
    reader.add_path(metrics_path)
    reader.poll()

    graph_types = {s.key: s.graph_type for s in config.stats}
    written: list[Path] = []
    for s in config.stats:
        x, y = reader.get_series(s.key)
        paths = _save_figure(
            out_dir, s.key, s.label, graph_types.get(s.key, "line"), x, y, formats, args.dpi
        )
        written.extend(paths)

    print(f"Exported {len(written)} file(s) to {out_dir.resolve()}")
    for p in sorted(written):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
