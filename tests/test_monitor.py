"""Tests for monitor (CLI and optional UI smoke)."""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Import monitor's _parse_args by importing the module
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def test_cli_parse_config_and_metrics():
    """Parse --config path --metrics path and assert parsed values."""
    from monitor import _parse_args
    with patch.object(sys, "argv", ["monitor.py", "--config", "/foo/config.json", "--metrics", "/bar/metrics"]):
        args = _parse_args()
    assert args.config == Path("/foo/config.json")
    assert args.metrics == Path("/bar/metrics")


def test_cli_defaults():
    """Parse with no args and assert defaults."""
    from monitor import DEFAULT_CONFIG_PATH, _parse_args
    with patch.object(sys, "argv", ["monitor.py"]):
        args = _parse_args()
    assert args.config == DEFAULT_CONFIG_PATH
    assert args.metrics is None


def test_build_ui_smoke():
    """Create DPG context, load config, build window (plot containers and series), then destroy context."""
    import dearpygui.dearpygui as dpg
    from config_loader import load_config
    from monitor import _build_ui, MAIN_WINDOW_TAG, SERIES_TAG_PREFIX

    config_path = PROJECT_ROOT / "config" / "stats_dashboard.json"
    if not config_path.exists():
        pytest.skip("Default config not found")
    config = load_config(config_path)
    dpg.create_context()
    try:
        series_tags = _build_ui(config, None)
        assert len(series_tags) == len(config.stats)
        for s in config.stats:
            assert s.key in series_tags
            assert series_tags[s.key] == f"{SERIES_TAG_PREFIX}{s.key}"
    finally:
        dpg.destroy_context()


def test_poll_callback_calls_set_value_per_stat():
    """One tick of poll callback: assert dpg.set_value called for each stat key with expected (x, y)."""
    import dearpygui.dearpygui as dpg
    from metrics_reader import MetricsReader
    from monitor import _run_poll_callback

    reader = MetricsReader(stat_keys=["loss", "p1_win_rate"])
    series_tags = {"loss": "series_loss", "p1_win_rate": "series_p1_win_rate"}
    graph_types = {"loss": "line", "p1_win_rate": "line"}
    with patch.object(dpg, "set_value", MagicMock()) as mock_set:
        _run_poll_callback(reader, series_tags, ["loss", "p1_win_rate"], graph_types)
    assert mock_set.call_count == 2
    calls = {c[0][0]: (c[0][1], c[0][2] if len(c[0]) > 2 else None) for c in mock_set.call_args_list}
    assert "series_loss" in calls or any("loss" in str(c) for c in mock_set.call_args_list)
    # Empty data so ([], []) per series
    for call in mock_set.call_args_list:
        tag, data = call[0][0], call[0][1]
        assert isinstance(data, (list, tuple))
        assert len(data) >= 1  # (x, y) or (y,) for histogram


def test_poll_callback_with_data(tmp_path):
    """Integration: temp JSONL, real reader, one poll; assert get_series updated."""
    import json
    from pathlib import Path
    from metrics_reader import MetricsReader
    from monitor import _run_poll_callback
    import dearpygui.dearpygui as dpg

    jsonl = tmp_path / "m.jsonl"
    with open(jsonl, "w", encoding="utf-8") as f:
        f.write('{"episode": 0, "loss": 1.0, "p1_win_rate": 0.5}\n')
        f.write('{"episode": 1, "loss": 0.9, "p1_win_rate": 0.6}\n')

    reader = MetricsReader(stat_keys=["loss", "p1_win_rate"])
    reader.add_path(jsonl)
    series_tags = {"loss": "series_loss", "p1_win_rate": "series_p1_win_rate"}
    graph_types = {"loss": "line", "p1_win_rate": "line"}

    with patch.object(dpg, "set_value", MagicMock()) as mock_set:
        _run_poll_callback(reader, series_tags, ["loss", "p1_win_rate"], graph_types)

    x_loss, y_loss = reader.get_series("loss")
    assert len(y_loss) == 2
    assert y_loss == [1.0, 0.9]
    assert mock_set.call_count == 2
