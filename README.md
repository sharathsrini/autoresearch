# fera-ae-autoresearch

Forked from [karpathy/autoresearch](https://github.com/karpathy/autoresearch)
and rewritten for unsupervised anomaly detection on energy forward curves.
The autoresearch loop discipline (single metric, ratchet, one branch per
experiment, append-only `results.tsv`) is preserved. Everything LLM-specific
is gone.

## What this is

A small autoencoder (~2k params) that learns the *shape* of a 36-tenor
forward curve and flags days where the shape deviates from learned normal.
This is the non-linear extension of the existing F11 PCA reconstruction
detector in FERA. The agent's job is to lower `val_metric` while staying
differentiated from F11 — see `program.md`.

## Setup

1. Place the five CSVs in a `data/` folder at the repo root (or leave them
   at the repo root — `prepare.py` looks in both places):

   ```
   data/ml_wide.csv
   data/ml_long.csv
   data/curve_features.csv
   data/tenor_features.csv
   data/cross-market-features.csv
   ```

2. Install Python deps. CPU-only build of PyTorch:

   ```
   pip install -r requirements.txt
   # If pip resolves a CUDA wheel of torch on your platform, force CPU:
   # pip install --index-url https://download.pytorch.org/whl/cpu torch
   ```

3. Run the baseline:

   ```
   python train.py --tag baseline
   ```

   This trains a `CurveShapeAE(bottleneck=6)` on TTF, evaluates on the
   2024-H2 walk-forward validation window, and appends one row to
   `results.tsv` with the four metrics plus a `kept` flag.

## Autoresearch loop

Point Claude Code (or any agent) at `program.md` and let it loop. Each
iteration is a single git branch off `master` that edits `train.py` only,
runs once, and merges back only if all four kept-guards pass. See
`program.md` for the full protocol and the ordered search directions.

## Files

| File              | Role |
|-------------------|------|
| `prepare.py`      | Fixed. CSV loading, walk-forward split, training filter, row-standardize, `compute_metric`, kept-guards, `append_result`. **Do not modify.** |
| `train.py`        | The model and training loop. **The only file the agent edits.** |
| `program.md`      | Agent instructions: the metric, the guards, the git protocol, the search directions. |
| `results.tsv`     | Append-only log: `timestamp tag val_metric train_mse latent_std_min corr_with_F11 kept`. |
| `requirements.txt`| Pinned deps. CPU-only. |
| `runs/<tag>/`     | Per-run artifacts: `model.pt`, `config.json`. |

## The metric

```
val_metric = mean over val rows of  sum_i (x_i - x_hat_i)^2
```

Lower is better. Kept-guards: `train_mse > 1e-4`, `latent_std_min > 0.05`,
`corr_with_F11 < 0.92`.

## Hardware

CPU. No CUDA. The model is tiny (~2k params); a baseline run finishes well
under the 3-minute budget on a laptop.
