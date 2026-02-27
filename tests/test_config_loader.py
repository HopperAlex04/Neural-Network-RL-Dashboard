"""Tests for config_loader."""
from pathlib import Path

import pytest

from config_loader import (
    ConfigError,
    DashboardConfig,
    StatEntry,
    load_config,
)

# Project root (parent of tests/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "stats_dashboard.json"


def test_load_valid_config_from_repo():
    """Load default config and assert stats length and first entry has key and graph_type."""
    config = load_config(DEFAULT_CONFIG_PATH)
    assert isinstance(config, DashboardConfig)
    assert len(config.stats) >= 5
    first = config.stats[0]
    assert hasattr(first, "key")
    assert hasattr(first, "graph_type")
    assert first.key in ("loss", "trainee_win_rate_vs_gapmaximizer")  # default config may list either first
    assert first.graph_type == "line"


def test_load_from_temp_valid_json(tmp_path):
    """Load from a temp valid JSON and assert returned structure."""
    config_file = tmp_path / "config.json"
    config_file.write_text(
        '{"metrics_path": "/some/path", "stats": [{"key": "x", "label": "X", "graph_type": "scatter"}]}'
    )
    config = load_config(config_file)
    assert config.metrics_path == "/some/path"
    assert len(config.stats) == 1
    assert config.stats[0].key == "x"
    assert config.stats[0].label == "X"
    assert config.stats[0].graph_type == "scatter"
    assert config.stat_keys() == ["x"]


def test_invalid_config_missing_key(tmp_path):
    """Entry missing 'key' raises with clear message."""
    config_file = tmp_path / "config.json"
    config_file.write_text('{"metrics_path": "", "stats": [{"label": "L", "graph_type": "line"}]}')
    with pytest.raises(ConfigError) as exc_info:
        load_config(config_file)
    assert "key" in str(exc_info.value).lower()


def test_invalid_config_bad_graph_type(tmp_path):
    """Invalid graph_type raises with clear message."""
    config_file = tmp_path / "config.json"
    config_file.write_text('{"metrics_path": "", "stats": [{"key": "x", "graph_type": "pie"}]}')
    with pytest.raises(ConfigError) as exc_info:
        load_config(config_file)
    assert "graph_type" in str(exc_info.value).lower() or "line" in str(exc_info.value)


def test_invalid_config_stats_not_list(tmp_path):
    """stats not a list raises with clear message."""
    config_file = tmp_path / "config.json"
    config_file.write_text('{"metrics_path": "", "stats": "not a list"}')
    with pytest.raises(ConfigError) as exc_info:
        load_config(config_file)
    assert "list" in str(exc_info.value).lower()


def test_default_config_exists_and_loadable():
    """Default config path exists and load_config succeeds (guard against breaking default)."""
    assert DEFAULT_CONFIG_PATH.exists(), f"Default config missing: {DEFAULT_CONFIG_PATH}"
    config = load_config(DEFAULT_CONFIG_PATH)
    assert config is not None
    assert len(config.stats) > 0


def test_missing_file_raises():
    """Missing config file raises ConfigError."""
    with pytest.raises(ConfigError) as exc_info:
        load_config(Path("/nonexistent/path/config.json"))
    assert "not found" in str(exc_info.value).lower() or "nonexistent" in str(exc_info.value)
