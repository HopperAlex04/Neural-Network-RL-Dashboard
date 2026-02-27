"""Tests for metrics_reader."""
import json
from pathlib import Path

import pytest

from metrics_reader import (
    MetricsReader,
    TRANEE_WIN_RATE_VS_GAPMAXIMIZER_KEY,
)


def test_poll_reads_jsonl_and_get_series_returns_correct_data(tmp_path):
    """Create temp JSONL with episode, loss, p1_win_rate; poll(); assert get_series returns correct (x, y)."""
    jsonl = tmp_path / "metrics.jsonl"
    lines = [
        {"episode": 0, "loss": 1.0, "p1_win_rate": 0.0},
        {"episode": 1, "loss": 0.9, "p1_win_rate": 0.5},
        {"episode": 2, "loss": 0.8, "p1_win_rate": 0.66},
    ]
    with open(jsonl, "w", encoding="utf-8") as f:
        for rec in lines:
            f.write(json.dumps(rec) + "\n")

    reader = MetricsReader(stat_keys=["loss", "p1_win_rate"])
    reader.add_path(jsonl)
    reader.poll()

    x_loss, y_loss = reader.get_series("loss")
    assert len(x_loss) == 3 and len(y_loss) == 3
    assert x_loss == [0.0, 1.0, 2.0]
    assert y_loss == [1.0, 0.9, 0.8]

    x_wr, y_wr = reader.get_series("p1_win_rate")
    assert len(x_wr) == 3 and len(y_wr) == 3
    assert y_wr == [0.0, 0.5, 0.66]


def test_record_with_missing_key_skips_that_stat_others_updated(tmp_path):
    """Record with missing key for a stat: that stat unchanged, others updated."""
    jsonl = tmp_path / "m.jsonl"
    with open(jsonl, "w", encoding="utf-8") as f:
        f.write('{"episode": 0, "loss": 1.0}\n')  # no p1_win_rate
        f.write('{"episode": 1, "loss": 0.5, "p1_win_rate": 0.5}\n')

    reader = MetricsReader(stat_keys=["loss", "p1_win_rate"])
    reader.add_path(jsonl)
    reader.poll()

    x_loss, y_loss = reader.get_series("loss")
    assert len(y_loss) == 2
    x_wr, y_wr = reader.get_series("p1_win_rate")
    assert len(y_wr) == 1
    assert y_wr[0] == 0.5


def test_non_numeric_value_skipped_for_that_stat(tmp_path):
    """Record with non-numeric value for a stat: that point skipped for that stat."""
    jsonl = tmp_path / "m.jsonl"
    with open(jsonl, "w", encoding="utf-8") as f:
        f.write('{"episode": 0, "loss": 1.0, "tag": "string"}\n')
        f.write('{"episode": 1, "loss": 0.5, "tag": 42}\n')

    reader = MetricsReader(stat_keys=["loss", "tag"])
    reader.add_path(jsonl)
    reader.poll()

    _, y_loss = reader.get_series("loss")
    assert len(y_loss) == 2
    _, y_tag = reader.get_series("tag")
    assert len(y_tag) == 1
    assert y_tag[0] == 42.0


def test_directory_path_resolves_to_sorted_jsonl(tmp_path):
    """Directory path: resolve to sorted *.jsonl and read."""
    (tmp_path / "b.jsonl").write_text('{"episode": 0, "x": 10}\n', encoding="utf-8")
    (tmp_path / "a.jsonl").write_text('{"episode": 0, "x": 20}\n', encoding="utf-8")

    reader = MetricsReader(stat_keys=["x"])
    reader.add_path(tmp_path)
    reader.poll()

    _, y = reader.get_series("x")
    # Sorted by name: a.jsonl then b.jsonl
    assert y == [20.0, 10.0]


def test_rounds_sequential_global_episode(tmp_path):
    """Multiple round files: episodes are sequential (0..n-1 from file1, n.. from file2)."""
    (tmp_path / "metrics_round_0_selfplay.jsonl").write_text(
        '{"episode": 0, "v": 10}\n{"episode": 1, "v": 11}\n', encoding="utf-8"
    )
    (tmp_path / "metrics_round_1_selfplay.jsonl").write_text(
        '{"episode": 0, "v": 20}\n{"episode": 1, "v": 21}\n', encoding="utf-8"
    )
    reader = MetricsReader(stat_keys=["v"])
    reader.add_path(tmp_path)
    reader.poll()
    x, y = reader.get_series("v")
    assert x == [0.0, 1.0, 2.0, 3.0], "Global episode should be 0,1,2,3 across rounds"
    assert y == [10.0, 11.0, 20.0, 21.0], "Values should follow round_0 then round_1"


def test_trainee_win_rate_vs_gapmaximizer_cumulative(tmp_path):
    """Cumulative trainee win rate: only GapMaximizer files, trainee position from filename, p1_wins/p2_wins."""
    # trainee_first -> P1 is trainee -> use p1_wins; 2 games, 1 win -> 1/2
    (tmp_path / "metrics_round_0_vs_gapmaximizer_trainee_first.jsonl").write_text(
        '{"episode": 0, "p1_wins": 1, "p2_wins": 0}\n'
        '{"episode": 1, "p1_wins": 1, "p2_wins": 1}\n',
        encoding="utf-8",
    )
    # trainee_second -> P2 is trainee -> use p2_wins; 2 games, 2 wins -> cumulative 1+2 / 2+2 = 3/4
    (tmp_path / "metrics_round_1_vs_gapmaximizer_trainee_second.jsonl").write_text(
        '{"episode": 0, "p1_wins": 0, "p2_wins": 1}\n'
        '{"episode": 1, "p1_wins": 0, "p2_wins": 2}\n',
        encoding="utf-8",
    )
    reader = MetricsReader(stat_keys=[TRANEE_WIN_RATE_VS_GAPMAXIMIZER_KEY])
    reader.add_path(tmp_path)
    reader.poll()
    x, y = reader.get_series(TRANEE_WIN_RATE_VS_GAPMAXIMIZER_KEY)
    assert x == [0.0, 1.0, 2.0, 3.0], "x should be game index vs GapMaximizer (0, 1, 2, 3)"
    assert len(y) == 4
    assert y[0] == 1.0  # 1 win / 1 game
    assert y[1] == 0.5  # 1 win / 2 games
    assert y[2] == 2.0 / 3.0  # (1+1) / 3 games
    assert y[3] == 3.0 / 4.0  # (1+2) / 4 games


def test_training_only_stats_skip_validation_episodes(tmp_path):
    """Training-only stats (loss, ppo_*, dqn_*) only count training episodes; x = training episode index."""
    (tmp_path / "metrics.jsonl").write_text(
        '{"episode": 0, "validating": false, "loss": 0.5, "p1_win_rate": 0.0}\n'
        '{"episode": 1, "validating": true, "loss": null, "p1_win_rate": 0.5}\n'
        '{"episode": 2, "validating": false, "loss": 0.3, "p1_win_rate": 0.66}\n',
        encoding="utf-8",
    )
    reader = MetricsReader(stat_keys=["loss", "p1_win_rate"])
    reader.add_path(tmp_path / "metrics.jsonl")
    reader.poll()
    x_loss, y_loss = reader.get_series("loss")
    x_wr, y_wr = reader.get_series("p1_win_rate")
    # loss: only episodes 0 and 2 (training), so x = 0, 1 (training episode index)
    assert x_loss == [0.0, 1.0]
    assert y_loss == [0.5, 0.3]
    # p1_win_rate: all episodes, x = global 0, 1, 2
    assert x_wr == [0.0, 1.0, 2.0]
    assert y_wr == [0.0, 0.5, 0.66]


def test_trainee_win_rate_ignores_selfplay(tmp_path):
    """trainee_win_rate_vs_gapmaximizer gets no data from selfplay files."""
    (tmp_path / "metrics_round_0_selfplay.jsonl").write_text(
        '{"episode": 0, "p1_wins": 10, "p2_wins": 10}\n', encoding="utf-8"
    )
    (tmp_path / "metrics_round_0_vs_gapmaximizer_trainee_first.jsonl").write_text(
        '{"episode": 0, "p1_wins": 1, "p2_wins": 0}\n', encoding="utf-8"
    )
    reader = MetricsReader(stat_keys=[TRANEE_WIN_RATE_VS_GAPMAXIMIZER_KEY])
    reader.add_path(tmp_path)
    reader.poll()
    x, y = reader.get_series(TRANEE_WIN_RATE_VS_GAPMAXIMIZER_KEY)
    assert len(y) == 1
    assert y[0] == 1.0  # only the gapmaximizer record


def test_max_points_cap_truncates_oldest(tmp_path):
    """max_points cap truncates oldest so len(y) <= max_points."""
    jsonl = tmp_path / "m.jsonl"
    with open(jsonl, "w", encoding="utf-8") as f:
        for i in range(5):
            f.write(json.dumps({"episode": i, "v": i * 10}) + "\n")

    reader = MetricsReader(stat_keys=["v"], max_points=3)
    reader.add_path(jsonl)
    reader.poll()

    x, y = reader.get_series("v")
    assert len(y) == 3
    assert x == [2.0, 3.0, 4.0]
    assert y == [20.0, 30.0, 40.0]
