# Neural-Network-RL-Dashboard

A small dashboard project with a **DearPyGUI**-based GUI that demonstrates the library’s basic features.

## Setup

```bash
pip install -r requirements.txt
```

## Run the GUI demo

```bash
python gui_demo.py
```

## What the demo shows

- **Menu bar** — File (Clear log, Exit), View (Dark/Light theme), Help (About)
- **Controls** — Button, text, input field, float slider, checkbox
- **Plot** — Line series (demo reward curve) with axes and legend; supports pan/zoom
- **Table** — Episode summary with headers and borders
- **Log** — Read-only multiline text updated by button and menu actions
- **Themes** — Dark and light themes applied from the View menu
- **Callbacks** — Button and slider callbacks that update the log and click counter

All behavior is for demonstration only (no real training or data).

---

## Real-time metrics monitor (Honors-Thesis-Project)

The monitor shows live metrics from **Honors-Thesis-Project** training runs. It reads per-episode JSONL logs and displays one plot per stat (including debug stats like PPO ratio, clip fraction, DQN Q/TD).

### Run the monitor

```bash
python monitor.py [--config config/stats_dashboard.json] [--metrics path/to/metrics_logs]
```

- **`--config`** — Path to the dashboard config JSON (default: `config/stats_dashboard.json`).
- **`--metrics`** — Path to a single `.jsonl` file or a directory containing `*.jsonl` (e.g. a run’s `metrics_logs` folder, or your thesis project’s `experiments/<name>/runs/<run_id>/metrics_logs`). You can also set `metrics_path` in the config file.

Start training in Honors-Thesis-Project, then run the monitor with `--metrics` pointing at the same run’s `metrics_logs` to see live updates.

### Config: which stats and graph types

Edit `config/stats_dashboard.json` to:

- **Add or remove stats** — Each entry has `key` (metric name from the JSONL), optional `label`, and `graph_type`.
- **Change graph type** — Use `"line"`, `"scatter"`, `"bar"`, or `"histogram"` per stat.

The default config includes main metrics (loss, win rates, draw rate, episode turns) and debug stats (PPO policy/value loss, entropy, ratio, clip fraction, advantage; DQN Q mean, TD, grad norm). Stats missing from the current log (e.g. PPO keys in a DQN run) simply show no data.
