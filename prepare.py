"""
Fixed data prep + runtime utilities for the FERA curve-shape autoencoder.
Do not modify after this rewrite. The agent edits train.py only.

Loads 5 CSVs from data/ (or repo root as a fallback so the human does not
have to move files around), builds the curve-shape dataset for a single
market, applies the no-leakage walk-forward split, filters the training
window with the documented anomaly rules, row-standardizes the 36-tenor
vectors, and exposes the validation metric used by autoresearch.
"""

import os
import math
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Constants (fixed; do not modify)
# ---------------------------------------------------------------------------

TENORS = [f"M{i}" for i in range(1, 37)]
N_TENORS = len(TENORS)

TRAIN_END  = pd.Timestamp("2024-06-30")
VAL_START  = pd.Timestamp("2024-07-01")
VAL_END    = pd.Timestamp("2024-12-31")
TEST_START = pd.Timestamp("2025-01-01")

F11_FILTER_Z = 3.0   # |F11_PCA_RECONSTRUCTION_ERROR_Z|
F12_FILTER_Z = 3.0   # F12_CURVE_VARIANCE_Z (signed; spec)
F02_FILTER_Z = 4.0   # max over tenors of |F02_LOG_RETURN_MAD_Z|

EPS = 1e-8

# ---------------------------------------------------------------------------
# CSV location: data/ first (per spec), then repo root as a fallback.
# ---------------------------------------------------------------------------

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_CSV_DIRS = [os.path.join(REPO_ROOT, "data"), REPO_ROOT]

def _csv_path(name):
    for d in _CSV_DIRS:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"{name} not found in {_CSV_DIRS}")

# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _parse_date(s):
    # ml_wide/long use "YYYY-MM-DD"; features files use "YYYY-MM-DD HH:MM:SS.fff".
    return pd.to_datetime(s, errors="coerce").dt.normalize()

def load_panels():
    """Read the 5 CSVs and return a dict of DataFrames with normalized dates."""
    ml_wide = pd.read_csv(_csv_path("ml_wide.csv"))
    ml_long = pd.read_csv(_csv_path("ml_long.csv"))
    curve   = pd.read_csv(_csv_path("curve_features.csv"))
    tenor   = pd.read_csv(_csv_path("tenor_features.csv"))
    cross   = pd.read_csv(_csv_path("cross-market-features.csv"))

    for df in (ml_wide, ml_long, curve, tenor, cross):
        if "TRADE_DATE" in df.columns:
            df["TRADE_DATE"] = _parse_date(df["TRADE_DATE"])
    return {
        "ml_wide": ml_wide,
        "ml_long": ml_long,
        "curve_features": curve,
        "tenor_features": tenor,
        "cross_market_features": cross,
    }

def build_curve_dataset(market, panels=None):
    """Join ml_wide (M1..M36) with per-(market,date) filter signals.

    Returns a DataFrame indexed by TRADE_DATE with columns:
        M1..M36, F11_Z, F12_Z, MAX_ABS_F02_Z
    """
    if panels is None:
        panels = load_panels()

    wide = panels["ml_wide"]
    wide = wide[wide["MARKET"] == market].copy()
    wide = wide.dropna(subset=["TRADE_DATE"]).sort_values("TRADE_DATE")
    wide = wide[["TRADE_DATE"] + TENORS]

    cf = panels["curve_features"]
    cf = cf[cf["MARKET"] == market][
        ["TRADE_DATE", "F11_PCA_RECONSTRUCTION_ERROR_Z", "F12_CURVE_VARIANCE_Z"]
    ].copy()
    cf.columns = ["TRADE_DATE", "F11_Z", "F12_Z"]

    tf = panels["tenor_features"]
    tf = tf[tf["MARKET"] == market][["TRADE_DATE", "F02_LOG_RETURN_MAD_Z"]].copy()
    max_f02 = (
        tf.assign(absz=tf["F02_LOG_RETURN_MAD_Z"].abs())
          .groupby("TRADE_DATE", as_index=False)["absz"].max()
          .rename(columns={"absz": "MAX_ABS_F02_Z"})
    )

    df = wide.merge(cf, on="TRADE_DATE", how="left").merge(max_f02, on="TRADE_DATE", how="left")
    df = df.set_index("TRADE_DATE").sort_index()
    return df

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def row_standardize(X):
    """Per-row standardize: (x - row_mean) / row_std. Returns (X_std, mu, sd)."""
    X = np.asarray(X, dtype=np.float64)
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    sd_safe = np.where(sd < EPS, 1.0, sd)
    return ((X - mu) / sd_safe).astype(np.float32), mu.squeeze(1), sd.squeeze(1)

def walk_forward_split(df):
    """Date-cutoff split. Train ≤ 2024-06-30, val 2024-07..12, test ≥ 2025-01."""
    train_df = df.loc[df.index <= TRAIN_END]
    val_df   = df.loc[(df.index >= VAL_START) & (df.index <= VAL_END)]
    test_df  = df.loc[df.index >= TEST_START]
    return train_df, val_df, test_df

def filter_training_rows(train_df):
    """Drop training rows that violate any of the three anomaly filters.

    Missing signals (NaN) keep the row — the spec lists three explicit
    conditions; a missing measurement is not a violation.
    """
    f11 = train_df["F11_Z"].abs() > F11_FILTER_Z
    f12 = train_df["F12_Z"] > F12_FILTER_Z
    f02 = train_df["MAX_ABS_F02_Z"] > F02_FILTER_Z
    bad = f11.fillna(False) | f12.fillna(False) | f02.fillna(False)
    return train_df.loc[~bad].copy()

def make_loaders(train_X, val_X, batch_size):
    """CPU DataLoaders. Autoencoder is unsupervised; target = input."""
    train_t = torch.from_numpy(np.asarray(train_X, dtype=np.float32))
    val_t   = torch.from_numpy(np.asarray(val_X,   dtype=np.float32))
    train_loader = DataLoader(TensorDataset(train_t), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(val_t),   batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------

def _pearson(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return float("nan")
    a, b = a[mask], b[mask]
    a = a - a.mean(); b = b - b.mean()
    denom = math.sqrt((a * a).sum() * (b * b).sum())
    if denom < EPS:
        return float("nan")
    return float((a * b).sum() / denom)

def compute_metric(model, val_X, F11_Z_val, train_mse):
    """Compute the four numbers logged in results.tsv.

    val_metric = mean over val rows of sum_i (x_i - x_hat_i)^2
    """
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(np.asarray(val_X, dtype=np.float32))
        x_hat, z = model(x)
        resid = (x - x_hat).numpy()
        z_np = z.numpy()

    per_row_score = (resid ** 2).sum(axis=1)
    val_metric = float(per_row_score.mean())
    latent_std_min = float(z_np.std(axis=0).min()) if z_np.shape[1] > 0 else float("nan")
    corr = _pearson(per_row_score, F11_Z_val)

    return {
        "val_metric":     val_metric,
        "train_mse":      float(train_mse),
        "latent_std_min": latent_std_min,
        "corr_with_F11":  corr,
    }

# ---------------------------------------------------------------------------
# Results log
# ---------------------------------------------------------------------------

RESULTS_HEADER = "timestamp\ttag\tval_metric\ttrain_mse\tlatent_std_min\tcorr_with_F11\tkept\n"

def append_result(results_tsv_path, run_tag, metrics_dict, kept):
    """Atomic append: read existing, write tmp, os.replace into place."""
    ts = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    row = "\t".join([
        ts, run_tag,
        f"{metrics_dict['val_metric']:.6f}",
        f"{metrics_dict['train_mse']:.6f}",
        f"{metrics_dict['latent_std_min']:.6f}",
        f"{metrics_dict['corr_with_F11']:.6f}",
        "1" if kept else "0",
    ]) + "\n"

    existing = ""
    if os.path.exists(results_tsv_path):
        with open(results_tsv_path, "r", encoding="utf-8") as f:
            existing = f.read()
    if not existing:
        existing = RESULTS_HEADER

    tmp = results_tsv_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(existing)
        f.write(row)
    os.replace(tmp, results_tsv_path)

# ---------------------------------------------------------------------------
# Kept-guards
# ---------------------------------------------------------------------------

GUARD_TRAIN_MSE_MIN  = 1e-4
GUARD_LATENT_STD_MIN = 0.05
GUARD_CORR_MAX       = 0.92

def evaluate_guards(metrics, best_val_metric):
    """Returns (kept_bool, reasons_list).

    A run is kept iff val_metric strictly improves AND all three sanity
    guards (memorization, collapse, differentiation-from-F11) pass.
    """
    reasons = []
    if best_val_metric is not None and not (metrics["val_metric"] < best_val_metric):
        reasons.append(f"val_metric {metrics['val_metric']:.6f} not < best {best_val_metric:.6f}")
    if not (metrics["train_mse"] > GUARD_TRAIN_MSE_MIN):
        reasons.append(f"train_mse {metrics['train_mse']:.6f} <= {GUARD_TRAIN_MSE_MIN}")
    if not (metrics["latent_std_min"] > GUARD_LATENT_STD_MIN):
        reasons.append(f"latent_std_min {metrics['latent_std_min']:.6f} <= {GUARD_LATENT_STD_MIN}")
    if not (metrics["corr_with_F11"] < GUARD_CORR_MAX):
        reasons.append(f"corr_with_F11 {metrics['corr_with_F11']:.6f} >= {GUARD_CORR_MAX}")
    return (len(reasons) == 0), reasons
