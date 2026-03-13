"""
Data preparation and feature engineering for tenor-level time series anomaly detection.

Loads the raw CSV export (trade_date × market × M1..M36 tenor curve), applies
best-practice data engineering (cleaning, imputation, normalization), and builds
a rich per-point feature matrix suitable for Isolation Forest anomaly scoring.

Usage:
    from tenor_prepare import load_tenor_data, build_features

    df_raw = load_tenor_data("2026-03-11T11-09_export.csv")
    feature_df, meta = build_features(df_raw)
"""

import os
from typing import Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TENOR_COLS = [f"M{i}" for i in range(1, 37)]  # M1 .. M36
DATE_COL = "trade_date"
MARKET_COL = "market"

# Rolling window sizes for temporal feature engineering
ROLLING_WINDOWS = [5, 10, 21]  # ~1 week, ~2 weeks, ~1 month (business days)


# ---------------------------------------------------------------------------
# Data loading & cleaning
# ---------------------------------------------------------------------------

def load_tenor_data(csv_path: str) -> pd.DataFrame:
    """Load the raw CSV export and apply basic cleaning.

    Returns a DataFrame sorted by (market, trade_date) with:
      - parsed datetime index
      - forward-filled NaNs within each market (stale price carry-forward)
      - remaining NaNs (leading) back-filled
    """
    df = pd.read_csv(csv_path)

    # Drop the unnamed index column if present
    if df.columns[0] == "" or df.columns[0].startswith("Unnamed"):
        df = df.drop(columns=[df.columns[0]])

    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values([MARKET_COL, DATE_COL]).reset_index(drop=True)

    # Per-market forward-fill then back-fill for NaN tenors
    for col in TENOR_COLS:
        df[col] = df.groupby(MARKET_COL)[col].transform(
            lambda s: s.ffill().bfill()
        )

    return df


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def _tenor_curve_shape_features(row_values: np.ndarray) -> dict:
    """Extract shape descriptors from a single tenor curve (1×36 vector).

    Returns dict of scalar features capturing curve level, slope, curvature,
    and higher-order shape characteristics.
    """
    n = len(row_values)
    x = np.arange(n, dtype=np.float64)

    level = np.mean(row_values)
    spread = row_values[-1] - row_values[0]

    # Fit quadratic for slope & curvature
    coeffs = np.polyfit(x, row_values, 2)
    curvature = coeffs[0]
    slope = coeffs[1]

    # Contango/backwardation: fraction of positive month-over-month diffs
    diffs = np.diff(row_values)
    contango_ratio = np.mean(diffs > 0)

    # Max drawdown along the curve
    cummax = np.maximum.accumulate(row_values)
    drawdown = np.min(row_values - cummax)

    # Kurtosis and skewness of cross-tenor distribution
    std = np.std(row_values, ddof=1) if n > 1 else 1e-8
    if std < 1e-8:
        std = 1e-8
    centered = (row_values - level) / std
    skewness = np.mean(centered ** 3)
    kurtosis = np.mean(centered ** 4) - 3.0  # excess kurtosis

    return {
        "curve_level": level,
        "curve_spread": spread,
        "curve_slope": slope,
        "curve_curvature": curvature,
        "curve_std": std,
        "curve_skew": skewness,
        "curve_kurtosis": kurtosis,
        "contango_ratio": contango_ratio,
        "curve_drawdown": drawdown,
    }


def _build_per_tenor_features(group: pd.DataFrame) -> pd.DataFrame:
    """Build time-series features for a single market group.

    For each tenor and each date, computes:
      - Daily absolute and percentage change
      - Rolling mean, std, min, max for several windows
      - Z-score relative to rolling window
      - Deviation from cross-tenor mean (tenor richness/cheapness)

    Returns a DataFrame with one row per (date, tenor) observation.
    """
    records = []
    dates = group[DATE_COL].values
    tenor_matrix = group[TENOR_COLS].values  # (n_dates, 36)
    market = group[MARKET_COL].iloc[0]

    n_dates, n_tenors = tenor_matrix.shape

    for t_idx in range(n_tenors):
        tenor_name = TENOR_COLS[t_idx]
        series = tenor_matrix[:, t_idx].astype(np.float64)

        # Pre-compute rolling stats
        rolling_stats = {}
        for w in ROLLING_WINDOWS:
            if n_dates < w:
                continue
            rm = pd.Series(series).rolling(w, min_periods=1)
            rolling_stats[w] = {
                "mean": rm.mean().values,
                "std": rm.std(ddof=1).fillna(0).values,
                "min": rm.min().values,
                "max": rm.max().values,
            }

        for i in range(n_dates):
            rec = {
                "trade_date": dates[i],
                "market": market,
                "tenor": tenor_name,
                "tenor_idx": t_idx,
                "value": series[i],
            }

            # Daily change
            if i > 0:
                rec["abs_change"] = series[i] - series[i - 1]
                prev = series[i - 1]
                rec["pct_change"] = (series[i] - prev) / abs(prev) if abs(prev) > 1e-8 else 0.0
            else:
                rec["abs_change"] = 0.0
                rec["pct_change"] = 0.0

            # Rolling features
            for w in ROLLING_WINDOWS:
                if w not in rolling_stats:
                    rec[f"roll_mean_{w}"] = series[i]
                    rec[f"roll_std_{w}"] = 0.0
                    rec[f"roll_zscore_{w}"] = 0.0
                    rec[f"roll_range_{w}"] = 0.0
                    continue
                rs = rolling_stats[w]
                rec[f"roll_mean_{w}"] = rs["mean"][i]
                rec[f"roll_std_{w}"] = rs["std"][i]
                std_val = rs["std"][i] if rs["std"][i] > 1e-8 else 1e-8
                rec[f"roll_zscore_{w}"] = (series[i] - rs["mean"][i]) / std_val
                rec[f"roll_range_{w}"] = rs["max"][i] - rs["min"][i]

            # Cross-tenor context: deviation from the curve mean on this date
            curve_mean = np.mean(tenor_matrix[i, :])
            curve_std = np.std(tenor_matrix[i, :])
            if curve_std < 1e-8:
                curve_std = 1e-8
            rec["cross_tenor_zscore"] = (series[i] - curve_mean) / curve_std

            # Neighbor-tenor spread (local structure)
            if 0 < t_idx < n_tenors - 1:
                rec["neighbor_spread"] = (
                    tenor_matrix[i, t_idx]
                    - 0.5 * (tenor_matrix[i, t_idx - 1] + tenor_matrix[i, t_idx + 1])
                )
            elif t_idx == 0:
                rec["neighbor_spread"] = tenor_matrix[i, 0] - tenor_matrix[i, 1]
            else:
                rec["neighbor_spread"] = tenor_matrix[i, -1] - tenor_matrix[i, -2]

            records.append(rec)

    return pd.DataFrame(records)


def _build_curve_level_features(group: pd.DataFrame) -> pd.DataFrame:
    """Build whole-curve shape features per (market, date)."""
    records = []
    dates = group[DATE_COL].values
    tenor_matrix = group[TENOR_COLS].values
    market = group[MARKET_COL].iloc[0]

    for i in range(len(dates)):
        feats = _tenor_curve_shape_features(tenor_matrix[i])
        feats["trade_date"] = dates[i]
        feats["market"] = market
        records.append(feats)

    return pd.DataFrame(records)


def build_features(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, dict]:
    """Full feature engineering pipeline.

    Args:
        df: cleaned DataFrame from load_tenor_data()

    Returns:
        feature_df: DataFrame at (date, market, tenor) granularity with all features
        meta: dict of metadata (feature names, markets, etc.)
    """
    markets = df[MARKET_COL].unique()

    # Per-tenor time-series features (parallelizable per market)
    tenor_parts = []
    curve_parts = []
    for mkt in markets:
        grp = df[df[MARKET_COL] == mkt].copy()
        tenor_parts.append(_build_per_tenor_features(grp))
        curve_parts.append(_build_curve_level_features(grp))

    tenor_df = pd.concat(tenor_parts, ignore_index=True)
    curve_df = pd.concat(curve_parts, ignore_index=True)

    # Merge curve-level features onto tenor-level rows
    feature_df = tenor_df.merge(curve_df, on=["trade_date", "market"], how="left")

    # Identify numeric feature columns (everything except identifiers)
    id_cols = {"trade_date", "market", "tenor", "tenor_idx", "value"}
    feature_cols = [c for c in feature_df.columns if c not in id_cols]

    # Replace any remaining NaN/inf with 0
    feature_df[feature_cols] = (
        feature_df[feature_cols]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    meta = {
        "feature_cols": feature_cols,
        "id_cols": list(id_cols),
        "markets": list(markets),
        "n_tenors": len(TENOR_COLS),
        "n_dates": df.groupby(MARKET_COL).size().to_dict(),
        "n_features": len(feature_cols),
        "n_rows": len(feature_df),
    }

    return feature_df, meta


# ---------------------------------------------------------------------------
# Standalone execution — preview the feature matrix
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    csv_path = sys.argv[1] if len(sys.argv) > 1 else "2026-03-11T11-09_export.csv"
    print(f"Loading data from {csv_path} ...")
    df_raw = load_tenor_data(csv_path)
    print(f"  Raw shape: {df_raw.shape}")
    print(f"  Markets: {df_raw[MARKET_COL].unique()}")
    print(f"  Date range: {df_raw[DATE_COL].min()} → {df_raw[DATE_COL].max()}")
    print()

    print("Building features ...")
    feature_df, meta = build_features(df_raw)
    print(f"  Feature matrix shape: {feature_df.shape}")
    print(f"  Number of engineered features: {meta['n_features']}")
    print(f"  Feature columns: {meta['feature_cols']}")
    print()
    print(feature_df.head(10).to_string(index=False))
