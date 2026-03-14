"""
Generate sample metrics JSONL for testing the dashboard and export_graphs.
Produces noisy, training-like curves so rolling averages are clearly visible.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SAMPLE_DATA_DIR = PROJECT_ROOT / "sample_data"
NUM_EPISODES = 250
NOISE_SCALE = 0.08  # relative noise so trends remain visible


def _trend(ep: int, n: int, start: float, end: float) -> float:
    """Linear trend from start to end over n episodes."""
    t = ep / max(1, n - 1)
    return start + t * (end - start)


def _noisy(value: float, scale: float = NOISE_SCALE) -> float:
    """Add relative noise; keep value in [0, 1] or non-negative where appropriate."""
    noise = (random.random() - 0.5) * 2 * scale * (value + 1e-6)
    return max(0.0, value + noise)


def main() -> None:
    SAMPLE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(42)

    out_path = SAMPLE_DATA_DIR / "metrics_round_0.jsonl"
    n = NUM_EPISODES

    with open(out_path, "w", encoding="utf-8") as f:
        for ep in range(n):
            # Loss: decreasing with noise
            loss = _noisy(_trend(ep, n, 1.2, 0.15), 0.12)
            # Win rates: increasing with noise
            p1_wr = _noisy(_trend(ep, n, 0.35, 0.72))
            p2_wr = _noisy(_trend(ep, n, 0.38, 0.70))
            # Draw rate: slight decrease
            draw_rate = _noisy(_trend(ep, n, 0.25, 0.08))
            # Turns per episode: slight decrease (faster wins)
            episode_turns = _noisy(_trend(ep, n, 45, 22), 0.15)
            episode_turns = max(5, episode_turns)

            # PPO-like stats (training-only; no "validating" so all count)
            ppo_policy_loss = _noisy(_trend(ep, n, 0.5, 0.08), 0.2)
            ppo_value_loss = _noisy(_trend(ep, n, 2.0, 0.3), 0.15)
            ppo_entropy = _noisy(_trend(ep, n, 0.8, 0.25))
            ppo_ratio_mean = _noisy(1.0, 0.1)
            ppo_ratio_min = _noisy(0.6, 0.15)
            ppo_ratio_max = _noisy(1.4, 0.15)
            ppo_clip_fraction = _noisy(_trend(ep, n, 0.25, 0.05))
            ppo_advantage_mean = _noisy(0.0, 0.3)
            ppo_advantage_std = _noisy(_trend(ep, n, 0.6, 0.2), 0.2)
            # DQN-like
            dqn_q_mean = _noisy(_trend(ep, n, 2.0, 8.0), 0.12)
            dqn_td_abs_mean = _noisy(_trend(ep, n, 1.5, 0.2), 0.25)
            dqn_grad_norm = _noisy(_trend(ep, n, 3.0, 0.5), 0.2)

            record = {
                "episode": ep,
                "loss": round(loss, 6),
                "p1_win_rate": round(p1_wr, 6),
                "p2_win_rate": round(p2_wr, 6),
                "draw_rate": round(draw_rate, 6),
                "episode_turns": round(episode_turns, 4),
                "ppo_policy_loss": round(ppo_policy_loss, 6),
                "ppo_value_loss": round(ppo_value_loss, 6),
                "ppo_entropy": round(ppo_entropy, 6),
                "ppo_ratio_mean": round(ppo_ratio_mean, 6),
                "ppo_ratio_min": round(ppo_ratio_min, 6),
                "ppo_ratio_max": round(ppo_ratio_max, 6),
                "ppo_clip_fraction": round(ppo_clip_fraction, 6),
                "ppo_advantage_mean": round(ppo_advantage_mean, 6),
                "ppo_advantage_std": round(ppo_advantage_std, 6),
                "dqn_q_mean": round(dqn_q_mean, 6),
                "dqn_td_abs_mean": round(dqn_td_abs_mean, 6),
                "dqn_grad_norm": round(dqn_grad_norm, 6),
            }
            f.write(json.dumps(record) + "\n")

    print(f"Wrote {NUM_EPISODES} sample metric records to {out_path}")
    print("Run: python export_graphs.py --metrics sample_data --output exported_graphs")
    print("Then open the PNGs in exported_graphs/ to see raw + rolling average curves.")


if __name__ == "__main__":
    main()
