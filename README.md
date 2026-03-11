# autoresearch — time series anomaly detection

Autonomous AI-agent experimentation loop for time series anomaly detection. Adapted from [karpathy/autoresearch](https://github.com/karpathy/autoresearch).

The idea: give an AI agent a baseline anomaly detection model and let it experiment autonomously overnight. It modifies the code, trains for 5 minutes, checks if the F1 score improved, keeps or discards, and repeats. You wake up to a log of experiments and (hopefully) a better model.

## How it works

Three files that matter:

- **`prepare.py`** — fixed constants, data generation (synthetic multivariate time series with injected anomalies), normalization, windowing, and evaluation (`evaluate_f1`). Not modified by the agent.
- **`train.py`** — the single file the agent edits. Contains the model (baseline: LSTM-Autoencoder), optimizer, and training loop. Architecture, hyperparameters, loss functions, anomaly scoring — everything is fair game.
- **`program.md`** — instructions for the agent. Point your agent here and let it go.

Training runs for a **fixed 5-minute time budget**. The metric is **val_f1** (validation F1 score with point-adjustment) — higher is better.

## Quick start

```bash
# 1. Install uv (if needed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# 3. Generate data (one-time, ~5 seconds)
uv run prepare.py

# 4. Run a single training experiment (~5 min)
uv run train.py
```

## Platform support

**Linux + NVIDIA GPU (recommended for DL models):**
```bash
# For faster torch install, uncomment the CUDA source in pyproject.toml, then:
uv sync
```

**macOS / Apple Silicon:**
```bash
# Works out of the box — torch installs with MPS (Metal) support from PyPI
uv sync
```

- **sklearn models** (Isolation Forest, LOF): work everywhere, no GPU needed
- **LSTM-AE / DL models**: run on CUDA, MPS (Apple Silicon), or CPU — MPS is auto-detected
- Unlike the original autoresearch (which required Flash Attention 3 / Hopper GPU), this version has **no CUDA-only dependencies**

## Using custom data

To use your own time series data instead of synthetic:

```bash
uv run prepare.py --data-dir ./mydata
```

Your `./mydata/` directory should contain:
- `train.csv` — training data (normal only), columns are features
- `test.csv` — test data (with anomalies), same feature columns
- `test_labels.csv` — single `label` column (0=normal, 1=anomaly)

## Running the agent

Spin up Claude Code (or any coding agent) in this repo:

```
Hi, have a look at program.md and let's kick off a new experiment! Let's do the setup first.
```

## Data

The synthetic dataset generates multivariate time series with:
- **25 features** with realistic patterns (sinusoidal components, trends, noise, inter-feature correlations)
- **5 anomaly types**: spikes, level shifts, variance changes, trend changes, contextual anomalies
- **~5% anomaly ratio** in test/validation sets
- Training data is entirely normal (for reconstruction-based methods)

## Model interface

The agent's model must implement:

```python
def compute_anomaly_scores(self, windows):
    """
    Args: windows (batch, window_size, n_features)
    Returns: scores (batch,) — higher = more anomalous
    """
```

This is called by `evaluate_f1` in `prepare.py` which searches thresholds for best point-adjusted F1.

## Project structure

```
prepare.py      — constants, data prep + evaluation (do not modify)
train.py        — model, optimizer, training loop (agent modifies this)
program.md      — agent instructions
pyproject.toml  — dependencies
analysis.ipynb  — experiment results analysis
```

## License

MIT
