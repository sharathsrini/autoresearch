"""
Auto-tuning Isolation Forest for tenor-level pointwise anomaly detection.

Reads the raw CSV, builds engineered features via tenor_prepare, then
automatically tunes Isolation Forest hyperparameters using a synthetic-
anomaly injection strategy (since real labels are unavailable).

The detector operates at the (date, market, tenor) granularity — each point
in the tenor curve on each trading day is independently scored.

Usage:
    uv run tenor_detect.py                              # full run
    uv run tenor_detect.py --csv path/to/export.csv     # custom CSV
    uv run tenor_detect.py --skip-tune                  # use defaults, no tuning
"""

import argparse
import itertools
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

from tenor_prepare import (
    TENOR_COLS,
    DATE_COL,
    MARKET_COL,
    load_tenor_data,
    build_features,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_CSV = "2026-03-11T11-09_export.csv"
OUTPUT_DIR = "results"
SEED = 42

# Synthetic anomaly injection rate for auto-tuning validation
SYNTH_ANOMALY_RATIO = 0.03


# ---------------------------------------------------------------------------
# Auto-tuning parameter grid
# ---------------------------------------------------------------------------

@dataclass
class TuningGrid:
    """Hyperparameter search space for Isolation Forest."""
    n_estimators: List[int] = field(default_factory=lambda: [100, 200, 400])
    max_samples: List = field(default_factory=lambda: [256, 512, "auto"])
    max_features: List[float] = field(default_factory=lambda: [0.5, 0.75, 1.0])
    contamination: List = field(default_factory=lambda: [0.01, 0.03, 0.05, "auto"])

    def combinations(self):
        """Yield all (n_estimators, max_samples, max_features, contamination) tuples."""
        return list(itertools.product(
            self.n_estimators,
            self.max_samples,
            self.max_features,
            self.contamination,
        ))


# ---------------------------------------------------------------------------
# Synthetic anomaly injection (for unsupervised tuning)
# ---------------------------------------------------------------------------

def inject_synthetic_anomalies(
    X: np.ndarray,
    ratio: float = SYNTH_ANOMALY_RATIO,
    seed: int = SEED,
) -> Tuple[np.ndarray, np.ndarray]:
    """Inject synthetic anomalies into a feature matrix for validation.

    Strategy:
      - Randomly select `ratio` fraction of rows
      - For each selected row, perturb 1-3 features by 4-8 standard deviations
      - Return (X_contaminated, labels) where labels[i]=1 iff row i was perturbed

    This lets us evaluate detection quality without real ground-truth labels.
    """
    rng = np.random.RandomState(seed)
    n, d = X.shape
    n_anomalies = max(1, int(n * ratio))

    labels = np.zeros(n, dtype=np.int32)
    X_out = X.copy()

    anom_indices = rng.choice(n, n_anomalies, replace=False)
    labels[anom_indices] = 1

    col_stds = np.std(X, axis=0)
    col_stds[col_stds < 1e-8] = 1.0

    for idx in anom_indices:
        n_perturb = rng.randint(1, min(4, d + 1))
        cols = rng.choice(d, n_perturb, replace=False)
        for c in cols:
            direction = rng.choice([-1, 1])
            magnitude = rng.uniform(4, 8)
            X_out[idx, c] += direction * magnitude * col_stds[c]

    return X_out, labels


# ---------------------------------------------------------------------------
# Auto-tuning engine
# ---------------------------------------------------------------------------

def auto_tune(
    X_train: np.ndarray,
    grid: Optional[TuningGrid] = None,
    n_validation_rounds: int = 3,
    verbose: bool = True,
) -> Tuple[dict, pd.DataFrame]:
    """Find the best Isolation Forest hyperparameters via synthetic anomaly injection.

    For each hyperparameter combination:
      1. Fit IF on clean training data
      2. Inject synthetic anomalies into a held-out copy
      3. Score and threshold-search for best F1
      4. Repeat `n_validation_rounds` times with different seeds, average F1

    Returns:
        best_params: dict of best hyperparameters
        results_df: DataFrame of all trials
    """
    if grid is None:
        grid = TuningGrid()

    combos = grid.combinations()
    n_combos = len(combos)
    if verbose:
        print(f"Auto-tuning: {n_combos} hyperparameter combinations × "
              f"{n_validation_rounds} validation rounds")
        print()

    trial_records = []
    best_f1 = -1.0
    best_params = {}

    for i, (n_est, max_samp, max_feat, contam) in enumerate(combos):
        params = {
            "n_estimators": n_est,
            "max_samples": max_samp,
            "max_features": max_feat,
            "contamination": contam,
        }

        round_f1s = []
        t0 = time.time()

        for r in range(n_validation_rounds):
            # Inject synthetic anomalies with a different seed each round
            X_val, y_val = inject_synthetic_anomalies(
                X_train, ratio=SYNTH_ANOMALY_RATIO, seed=SEED + r * 7
            )

            clf = IsolationForest(
                n_estimators=n_est,
                max_samples=max_samp,
                max_features=max_feat,
                contamination=contam,
                random_state=SEED,
                n_jobs=-1,
            )
            clf.fit(X_train)

            # Score the contaminated validation set
            scores = -clf.score_samples(X_val)  # higher = more anomalous

            # Threshold search for best F1
            f1 = _threshold_search_f1(scores, y_val)
            round_f1s.append(f1)

        mean_f1 = np.mean(round_f1s)
        dt = time.time() - t0

        trial_records.append({**params, "mean_f1": mean_f1, "time_s": dt})

        if verbose and (i + 1) % max(1, n_combos // 20) == 0:
            print(f"  [{i + 1}/{n_combos}] F1={mean_f1:.4f}  "
                  f"n_est={n_est} max_samp={max_samp} "
                  f"max_feat={max_feat} contam={contam}  ({dt:.1f}s)")

        if mean_f1 > best_f1:
            best_f1 = mean_f1
            best_params = params.copy()

    results_df = pd.DataFrame(trial_records).sort_values("mean_f1", ascending=False)

    if verbose:
        print()
        print(f"Best F1: {best_f1:.4f}")
        print(f"Best params: {best_params}")

    return best_params, results_df


def _threshold_search_f1(scores: np.ndarray, labels: np.ndarray) -> float:
    """Search over percentile thresholds for best F1 score."""
    percentiles = np.arange(90, 100, 0.5)
    thresholds = np.percentile(scores, percentiles)
    extra = np.linspace(np.percentile(scores, 85), scores.max(), 30)
    thresholds = np.unique(np.concatenate([thresholds, extra]))

    best_f1 = 0.0
    for thresh in thresholds:
        preds = (scores >= thresh).astype(np.int32)
        if preds.sum() == 0:
            continue
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1

    return best_f1


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class TenorAnomalyDetector:
    """Auto-tuning Isolation Forest for tenor-level pointwise anomaly detection.

    Fits one model per market (each market has its own price dynamics).
    """

    def __init__(self, params: Optional[dict] = None):
        self.params = params or {
            "n_estimators": 200,
            "max_samples": "auto",
            "max_features": 1.0,
            "contamination": "auto",
        }
        self.models = {}       # market -> fitted IsolationForest
        self.scalers = {}      # market -> fitted StandardScaler
        self.feature_cols = []

    def fit(
        self,
        feature_df: pd.DataFrame,
        feature_cols: List[str],
        auto_tune_enabled: bool = True,
        verbose: bool = True,
    ):
        """Fit per-market Isolation Forest models.

        If auto_tune_enabled, tunes hyperparameters on each market independently.
        """
        self.feature_cols = feature_cols
        markets = feature_df[MARKET_COL].unique()

        for mkt in markets:
            if verbose:
                print(f"\n{'='*60}")
                print(f"Market: {mkt}")
                print(f"{'='*60}")

            mkt_df = feature_df[feature_df[MARKET_COL] == mkt]
            X_raw = mkt_df[feature_cols].values.astype(np.float64)

            # Standardize features
            scaler = StandardScaler()
            X = scaler.fit_transform(X_raw)
            self.scalers[mkt] = scaler

            # Auto-tune if enabled
            if auto_tune_enabled:
                best_params, tuning_results = auto_tune(X, verbose=verbose)
                params = best_params
            else:
                params = self.params

            # Fit final model on all data (unsupervised — no labels)
            clf = IsolationForest(
                n_estimators=params["n_estimators"],
                max_samples=params["max_samples"],
                max_features=params["max_features"],
                contamination=params["contamination"],
                random_state=SEED,
                n_jobs=-1,
            )
            clf.fit(X)
            self.models[mkt] = clf

            if verbose:
                preds = clf.predict(X)
                n_anom = (preds == -1).sum()
                print(f"  Fitted with params: {params}")
                print(f"  Anomalies detected: {n_anom}/{len(X)} "
                      f"({100*n_anom/len(X):.2f}%)")

    def score(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """Score every (date, market, tenor) point.

        Returns the input DataFrame with added columns:
          - anomaly_score: continuous score (higher = more anomalous)
          - is_anomaly: binary flag from the model's decision function
        """
        result_parts = []

        for mkt, mkt_df in feature_df.groupby(MARKET_COL):
            if mkt not in self.models:
                raise ValueError(f"No fitted model for market '{mkt}'")

            X_raw = mkt_df[self.feature_cols].values.astype(np.float64)
            X = self.scalers[mkt].transform(X_raw)

            clf = self.models[mkt]
            scores = -clf.score_samples(X)  # higher = more anomalous
            preds = clf.predict(X)           # -1 = anomaly, 1 = normal

            out = mkt_df.copy()
            out["anomaly_score"] = scores
            out["is_anomaly"] = (preds == -1).astype(int)
            result_parts.append(out)

        return pd.concat(result_parts, ignore_index=True)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_summary(scored_df: pd.DataFrame):
    """Print a human-readable anomaly summary."""
    print("\n" + "=" * 70)
    print("ANOMALY DETECTION SUMMARY")
    print("=" * 70)

    for mkt in sorted(scored_df[MARKET_COL].unique()):
        mkt_df = scored_df[scored_df[MARKET_COL] == mkt]
        n_total = len(mkt_df)
        n_anom = mkt_df["is_anomaly"].sum()
        print(f"\n  Market: {mkt}")
        print(f"    Total points:     {n_total}")
        print(f"    Anomalies:        {n_anom} ({100*n_anom/n_total:.2f}%)")

        if n_anom > 0:
            anom = mkt_df[mkt_df["is_anomaly"] == 1]
            # Top anomalous dates
            top_dates = (
                anom.groupby(DATE_COL)["anomaly_score"]
                .mean()
                .sort_values(ascending=False)
                .head(5)
            )
            print(f"    Top anomalous dates:")
            for dt, sc in top_dates.items():
                n_tenors_hit = anom[anom[DATE_COL] == dt]["tenor"].nunique()
                print(f"      {str(dt)[:10]}  score={sc:.4f}  tenors_affected={n_tenors_hit}")

            # Most affected tenors
            top_tenors = (
                anom.groupby("tenor")["anomaly_score"]
                .mean()
                .sort_values(ascending=False)
                .head(5)
            )
            print(f"    Most affected tenors:")
            for tn, sc in top_tenors.items():
                print(f"      {tn}  avg_score={sc:.4f}")

    print("\n" + "=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Auto-tuning Isolation Forest for tenor-level anomaly detection"
    )
    parser.add_argument(
        "--csv", type=str, default=DEFAULT_CSV,
        help="Path to the source CSV export"
    )
    parser.add_argument(
        "--skip-tune", action="store_true",
        help="Skip auto-tuning, use default hyperparameters"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to save scored output CSV (default: results/anomalies.csv)"
    )
    args = parser.parse_args()

    t_start = time.time()

    # 1. Load & clean data
    print(f"Loading data from {args.csv} ...")
    df_raw = load_tenor_data(args.csv)
    print(f"  Raw shape: {df_raw.shape}")
    print(f"  Markets: {list(df_raw[MARKET_COL].unique())}")
    print(f"  Date range: {df_raw[DATE_COL].min()} → {df_raw[DATE_COL].max()}")

    # 2. Feature engineering
    print("\nBuilding features ...")
    feature_df, meta = build_features(df_raw)
    print(f"  Feature matrix: {feature_df.shape[0]} rows × "
          f"{meta['n_features']} engineered features")

    # 3. Fit detector (with or without auto-tuning)
    print("\nFitting anomaly detector ...")
    detector = TenorAnomalyDetector()
    detector.fit(
        feature_df,
        feature_cols=meta["feature_cols"],
        auto_tune_enabled=not args.skip_tune,
    )

    # 4. Score all points
    print("\nScoring all points ...")
    scored_df = detector.score(feature_df)

    # 5. Summary
    print_summary(scored_df)

    # 6. Save results
    out_path = args.output
    if out_path is None:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out_path = os.path.join(OUTPUT_DIR, "anomalies.csv")

    # Save a clean output with identifiers + scores
    output_cols = [
        DATE_COL, MARKET_COL, "tenor", "tenor_idx", "value",
        "anomaly_score", "is_anomaly",
    ]
    scored_df[output_cols].to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")

    t_total = time.time() - t_start

    # Final metrics
    print("\n---")
    print(f"total_points:      {len(scored_df)}")
    print(f"total_anomalies:   {scored_df['is_anomaly'].sum()}")
    print(f"anomaly_rate:      {scored_df['is_anomaly'].mean():.4f}")
    print(f"markets:           {list(scored_df[MARKET_COL].unique())}")
    print(f"n_features:        {meta['n_features']}")
    print(f"total_seconds:     {t_total:.1f}")
    print(f"auto_tuned:        {not args.skip_tune}")


if __name__ == "__main__":
    main()
