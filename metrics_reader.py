"""
Tail metrics JSONL files and maintain per-stat series (x, y) for plotting.
Rounds are one training run: files are ordered by round number and episodes
are treated as sequential (round 0: 0..N-1, round 1: N..2N-1, etc.).
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

DEFAULT_MAX_POINTS = 10_000

# Derived stat: trainee win rate vs GapMaximizer per round (x = round, y = average of both seats' final win rate)
TRANEE_WIN_RATE_VS_GAPMAXIMIZER_KEY = "trainee_win_rate_vs_gapmaximizer"

# Stats that only make sense for training episodes (exclude validation); x-axis = training episode index
_TRAINING_ONLY_STAT_PREFIXES = ("ppo_", "dqn_")
_TRAINING_ONLY_STAT_KEYS = frozenset(
    {"loss", "p1_epsilon", "p2_epsilon", "p1_memory_size", "p2_memory_size"}
)


def is_training_only_stat(key: str) -> bool:
    """True if this stat should only use training episodes (not validation)."""
    if key in _TRAINING_ONLY_STAT_KEYS:
        return True
    return key.startswith(_TRAINING_ONLY_STAT_PREFIXES)

# Filename pattern for thesis metrics: metrics_round_N_... or metrics_round_N
_ROUND_PATTERN = re.compile(r"round_(\d+)", re.IGNORECASE)


def _is_gapmaximizer_path(path: Path) -> bool:
    """True if path stem indicates vs GapMaximizer metrics (e.g. round_N_vs_gapmaximizer_trainee_first)."""
    return "gapmaximizer" in path.stem.lower()


def _trainee_is_p1(path: Path) -> bool:
    """True if trainee_first (trainee is P1), False if trainee_second (trainee is P2)."""
    return "trainee_first" in path.stem.lower()


def _is_numeric(v: Any) -> bool:
    return v is not None and isinstance(v, (int, float))


def _round_sort_key(file_path: Path) -> tuple[int, str]:
    """Sort key: (round_number, name) so round_0 < round_1 < ... < round_10 < ..."""
    match = _ROUND_PATTERN.search(file_path.stem)
    if match:
        return (int(match.group(1)), file_path.name)
    return (0, file_path.name)


def _resolve_paths(path: str | Path) -> list[Path]:
    """Resolve path to a list of JSONL file paths (single file or directory).
    For a directory, files are sorted by round number so rounds are sequential.
    """
    p = Path(path).resolve()
    if not p.exists():
        return []
    if p.is_file():
        return [p] if p.suffix == ".jsonl" else []
    # Directory: *.jsonl sorted by round number (round_0, round_1, ..., round_10, ...)
    files = list(p.glob("*.jsonl"))
    return sorted(files, key=_round_sort_key)


class MetricsReader:
    """
    Tails one or more JSONL metrics files and maintains (episode, value) series per stat key.
    When multiple files are present (e.g. round_0, round_1, ...), they are treated as one
    sequential training run: x-axis is global episode index (0..N-1 from file 1, N.. from file 2, ...).
    """

    def __init__(self, stat_keys: list[str], max_points: int = DEFAULT_MAX_POINTS):
        self.stat_keys = list(stat_keys)
        self.max_points = max_points
        self._series: dict[str, tuple[list[float], list[float]]] = {
            k: ([], []) for k in self.stat_keys
        }
        self._paths: list[Path] = []
        self._positions: dict[Path, int] = {}
        # Records read so far per path (for global episode offset across rounds)
        self._records_read: dict[Path, int] = {}
        # Single source of truth for global episode index (strictly increasing, one per record)
        self._global_episode: int = 0
        # Derived: trainee win rate vs GapMaximizer per round; one wr per file (trainee_first / trainee_second), averaged per round
        self._trainee_wr_by_file: dict[Path, tuple[int, float]] = {}  # path -> (round_num, wr)
        # Training-only stats: x-axis = training episode count (excludes validation episodes)
        self._training_episode_index: int = 0

    def add_path(self, path: str | Path) -> None:
        """Add a file or directory to read from. Does not clear existing series."""
        resolved = _resolve_paths(path)
        for r in resolved:
            if r not in self._positions:
                self._paths.append(r)
                self._positions[r] = 0
                self._records_read[r] = 0

    def poll(self) -> None:
        """Read new lines from all added paths and update per-stat series.
        Files are processed in round order; episode numbers are made sequential
        (round 0: 0..n0-1, round 1: n0..n0+n1-1, ...).
        """
        for path in self._paths:
            if not path.exists():
                continue
            # Global offset = total records read from all files before this one
            path_index = self._paths.index(path)
            offset = sum(self._records_read.get(self._paths[i], 0) for i in range(path_index))
            try:
                # Per-round tracking for trainee win rate vs GapMaximizer (one point per round)
                round_wins: int | None = None
                round_games: int | None = None
                round_num = _round_sort_key(path)[0] if _is_gapmaximizer_path(path) else 0

                with open(path, encoding="utf-8") as f:
                    f.seek(self._positions[path])
                    new_count = 0
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        # Global episode index: one per record, strictly increasing (single source of truth)
                        ep_global = float(self._global_episode)
                        # Only training episodes (no validation) for training-only stats
                        validating = record.get("validating", False) is True

                        # Derived stat: trainee win rate vs GapMaximizer per round (one point per round file)
                        if (
                            TRANEE_WIN_RATE_VS_GAPMAXIMIZER_KEY in self.stat_keys
                            and _is_gapmaximizer_path(path)
                        ):
                            trainee_first = _trainee_is_p1(path)
                            p1_wins = record.get("p1_wins")
                            p2_wins = record.get("p2_wins")
                            episode = record.get("episode")
                            if (
                                _is_numeric(p1_wins)
                                and _is_numeric(p2_wins)
                                and _is_numeric(episode)
                            ):
                                trainee_wins = int(float(p1_wins)) if trainee_first else int(float(p2_wins))
                                games = int(float(episode)) + 1
                                round_wins = trainee_wins
                                round_games = games

                        # Normal stats (raw keys from record)
                        for key in self.stat_keys:
                            if key == TRANEE_WIN_RATE_VS_GAPMAXIMIZER_KEY:
                                continue
                            val = record.get(key)
                            if not _is_numeric(val):
                                continue
                            # Training-only stats: only training episodes, x = training episode index
                            if is_training_only_stat(key):
                                if validating:
                                    continue
                                x_val = float(self._training_episode_index)
                                self._training_episode_index += 1
                            else:
                                x_val = ep_global
                            x_list, y_list = self._series[key]
                            x_list.append(x_val)
                            y_list.append(float(val))
                            if len(y_list) > self.max_points:
                                x_list.pop(0)
                                y_list.pop(0)

                        # Advance global episode exactly once per record
                        new_count += 1
                        self._global_episode += 1

                    # Store this file's final win rate (we'll average by round after processing all paths)
                    if (
                        TRANEE_WIN_RATE_VS_GAPMAXIMIZER_KEY in self.stat_keys
                        and _is_gapmaximizer_path(path)
                        and round_games is not None
                        and round_games > 0
                    ):
                        wr = round_wins / round_games
                        self._trainee_wr_by_file[path] = (round_num, wr)

                    self._positions[path] = f.tell()
                    self._records_read[path] = self._records_read.get(path, 0) + new_count
            except OSError:
                pass

        # Build trainee win rate series: average both seats per round, then sort by round
        if TRANEE_WIN_RATE_VS_GAPMAXIMIZER_KEY in self.stat_keys and self._trainee_wr_by_file:
            by_round: dict[int, list[float]] = defaultdict(list)
            for (_path, (r, w)) in self._trainee_wr_by_file.items():
                by_round[r].append(w)
            x_list: list[float] = []
            y_list: list[float] = []
            for r in sorted(by_round.keys()):
                x_list.append(float(r))
                y_list.append(sum(by_round[r]) / len(by_round[r]))
            if len(y_list) > self.max_points:
                x_list = x_list[-self.max_points:]
                y_list = y_list[-self.max_points:]
            self._series[TRANEE_WIN_RATE_VS_GAPMAXIMIZER_KEY] = (x_list, y_list)

    def get_series(self, stat_key: str) -> tuple[list[float], list[float]]:
        """Return (x_list, y_list) for the given stat key. Copies to avoid mutation."""
        if stat_key not in self._series:
            return ([], [])
        x, y = self._series[stat_key]
        return (list(x), list(y))
