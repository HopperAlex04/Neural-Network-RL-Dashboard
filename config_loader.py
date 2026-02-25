"""
Load and validate stats dashboard JSON config.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

VALID_GRAPH_TYPES = frozenset({"line", "scatter", "bar", "histogram"})


@dataclass
class StatEntry:
    key: str
    label: str
    graph_type: str


@dataclass
class DashboardConfig:
    metrics_path: str
    stats: list[StatEntry]

    def stat_keys(self) -> list[str]:
        return [s.key for s in self.stats]


class ConfigError(Exception):
    """Raised when config is invalid or file is missing."""


def load_config(config_path: str | Path) -> DashboardConfig:
    """
    Load and validate dashboard config from a JSON file.

    Args:
        config_path: Path to the JSON config file.

    Returns:
        Validated DashboardConfig.

    Raises:
        ConfigError: If file is missing, JSON is invalid, or config fails validation.
    """
    path = Path(config_path)
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")

    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ConfigError(f"Invalid JSON in config file: {e}") from e

    return _parse_config(data, path)


def _parse_config(data: Any, path: Path) -> DashboardConfig:
    if not isinstance(data, dict):
        raise ConfigError(f"Config must be a JSON object; got {type(data).__name__}")

    metrics_path = data.get("metrics_path", "")
    if not isinstance(metrics_path, str):
        raise ConfigError("metrics_path must be a string")

    stats_raw = data.get("stats")
    if not isinstance(stats_raw, list):
        raise ConfigError("stats must be a list")

    stats: list[StatEntry] = []
    for i, item in enumerate(stats_raw):
        if not isinstance(item, dict):
            raise ConfigError(f"stats[{i}] must be an object; got {type(item).__name__}")

        key = item.get("key")
        if key is None:
            raise ConfigError(f"stats[{i}] missing required field 'key'")
        if not isinstance(key, str):
            raise ConfigError(f"stats[{i}].key must be a string; got {type(key).__name__}")

        label = item.get("label", key)
        if not isinstance(label, str):
            raise ConfigError(f"stats[{i}].label must be a string; got {type(label).__name__}")

        graph_type = item.get("graph_type", "line")
        if not isinstance(graph_type, str):
            raise ConfigError(f"stats[{i}].graph_type must be a string; got {type(graph_type).__name__}")
        if graph_type not in VALID_GRAPH_TYPES:
            raise ConfigError(
                f"stats[{i}].graph_type must be one of {sorted(VALID_GRAPH_TYPES)}; got {graph_type!r}"
            )

        stats.append(StatEntry(key=key, label=label, graph_type=graph_type))

    return DashboardConfig(metrics_path=metrics_path, stats=stats)
