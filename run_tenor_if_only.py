from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


@dataclass
class Config:
    input_csv: str = "/mnt/data/Untitled 15_2026-03-29-0307.csv"
    output_dir: str = "./fera_tenor_if_outputs"

    base_features: Tuple[str, ...] = (
        "F02_LOG_RETURN_MAD_Z",
        "F03_VOLATILITY_SURPRISE",
        "F04_PERCENTILE_RANK_Z",
        "F06_CROSS_TENOR_RESIDUAL_MAD_Z",
        "F07_BUTTERFLY_CURVATURE_MAD_Z",
    )
    multi_window_features: Tuple[str, ...] = (
        "F02_Z_21D", "F02_Z_63D", "F02_Z_126D", "F02_Z_252D",
        "F03_VOL_HL5", "F03_VOL_HL10", "F03_VOL_HL20", "F03_VOL_HL60",
        "F06_Z_21D", "F06_Z_63D", "F06_Z_126D", "F06_Z_252D",
        "F07_Z_21D", "F07_Z_63D", "F07_Z_126D", "F07_Z_252D",
    )
    metadata_cols: Tuple[str, ...] = (
        "TRADE_DATE", "MARKET", "TENOR", "TENOR_NUM", "PRICE_NATIVE", "LOG_PRICE_NATIVE",
    )

    use_multi_window: bool = False
    completeness_threshold: float = 0.60
    impute_window_rows: int = 63

    split_date: str = "2024-07-01"
    auto_adjust_split: bool = True
    auto_test_fraction: float = 0.30
    min_train_rows_per_market: int = 500
    min_test_rows_per_market: int = 100

    apply_scaling: bool = False
    n_estimators: int = 300
    max_samples: str | int | float = "auto"
    contamination: str | float = "auto"
    max_features: float | int = 1.0
    random_state: int = 42
    n_jobs: int = -1

    anomaly_threshold: float = 0.95
    precision_proxy_cutoff: float = 3.0

    known_events: Dict[str, str] = field(default_factory=lambda: {
        "2022-02-24": "Ukraine invasion",
        "2022-08-26": "TTF all-time high",
        "2022-09-26": "Nord Stream sabotage",
    })

    @property
    def feature_cols(self) -> List[str]:
        cols = list(self.base_features)
        if self.use_multi_window:
            cols.extend(self.multi_window_features)
        return cols


# ---------------------------------------------------------------------------
# Loading and validation
# ---------------------------------------------------------------------------

def _normalize_param(value: str):
    value = str(value).strip()
    if value.lower() == "auto":
        return "auto"
    try:
        ivalue = int(value)
        if str(ivalue) == value:
            return ivalue
    except Exception:
        pass
    try:
        return float(value)
    except Exception:
        return value


def load_feature_csv(cfg: Config) -> pd.DataFrame:
    path = Path(cfg.input_csv)
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")

    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed:")].copy()
    df.columns = df.columns.astype(str).str.upper()

    required = list(cfg.metadata_cols) + list(cfg.base_features)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["TRADE_DATE"] = pd.to_datetime(df["TRADE_DATE"], errors="coerce")
    df["TENOR_NUM"] = pd.to_numeric(df["TENOR_NUM"], errors="coerce")
    if df["TRADE_DATE"].isna().any():
        raise ValueError("TRADE_DATE contains unparsable values")
    if df["TENOR_NUM"].isna().any():
        raise ValueError("TENOR_NUM contains non-numeric values")

    missing_active = [c for c in cfg.feature_cols if c not in df.columns]
    if missing_active:
        raise ValueError(f"Missing active feature columns: {missing_active}")

    df = df.sort_values(["MARKET", "TENOR_NUM", "TRADE_DATE"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Split logic
# ---------------------------------------------------------------------------

def resolve_split_date(df: pd.DataFrame, cfg: Config) -> pd.Timestamp:
    unique_dates = np.array(sorted(pd.Series(df["TRADE_DATE"].dropna().unique()).astype("datetime64[ns]")))
    if len(unique_dates) < 2:
        raise ValueError("Need at least two unique trade dates to form a train/test split")

    min_date = pd.Timestamp(unique_dates[0])
    max_date = pd.Timestamp(unique_dates[-1])
    requested = pd.Timestamp(cfg.split_date)

    if min_date < requested <= max_date:
        return requested

    if not cfg.auto_adjust_split:
        raise ValueError(
            f"split_date {requested.date()} is outside the data range {min_date.date()} -> {max_date.date()}"
        )

    n_dates = len(unique_dates)
    cutoff_idx = max(1, int(np.floor(n_dates * (1 - cfg.auto_test_fraction))))
    cutoff_idx = min(cutoff_idx, n_dates - 1)
    return pd.Timestamp(unique_dates[cutoff_idx])


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def apply_completeness_filter(df: pd.DataFrame, feature_cols: List[str], threshold: float) -> pd.DataFrame:
    out = df.copy()
    completeness = out[feature_cols].notna().sum(axis=1) / max(len(feature_cols), 1)
    out["FEATURE_COMPLETENESS"] = completeness
    out = out.loc[out["FEATURE_COMPLETENESS"] >= threshold].reset_index(drop=True)
    return out


def _safe_ratio_max_min(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if pd.notna(v)]
    if not vals:
        return 0.0
    min_v = min(vals)
    max_v = max(vals)
    if min_v <= 0:
        return np.inf if max_v > 0 else 0.0
    return max_v / min_v


def causal_impute_leakage_safe(
    df: pd.DataFrame,
    feature_cols: List[str],
    split_dt: pd.Timestamp,
    window: int,
) -> pd.DataFrame:
    """Causally impute active features without using future information.

    Per (market, tenor, feature):
      1) trailing median from past rows only: shift(1).rolling(window).median()
      2) training-period group median fallback
      3) training-period market median fallback
      4) training-period global median fallback
      5) 0.0 only if the feature is entirely missing in train
    """
    out = df.copy().sort_values(["MARKET", "TENOR_NUM", "TRADE_DATE"]).reset_index(drop=True)
    train_mask = out["TRADE_DATE"] < split_dt

    global_train_medians = {
        col: float(out.loc[train_mask, col].median()) if out.loc[train_mask, col].notna().any() else 0.0
        for col in feature_cols
    }
    market_train_medians = out.loc[train_mask].groupby("MARKET")[feature_cols].median()
    group_train_medians = out.loc[train_mask].groupby(["MARKET", "TENOR"])[feature_cols].median()

    filled_groups: List[pd.DataFrame] = []
    for (market, tenor), grp in out.groupby(["MARKET", "TENOR"], sort=False):
        g = grp.copy().sort_values("TRADE_DATE")
        for col in feature_cols:
            hist_med = g[col].shift(1).rolling(window=window, min_periods=1).median()
            series = g[col].where(g[col].notna(), hist_med)

            group_fallback = np.nan
            if (market, tenor) in group_train_medians.index:
                group_fallback = group_train_medians.loc[(market, tenor), col]
            market_fallback = np.nan
            if market in market_train_medians.index:
                market_fallback = market_train_medians.loc[market, col]
            global_fallback = global_train_medians[col]

            if pd.notna(group_fallback):
                series = series.fillna(float(group_fallback))
            if pd.notna(market_fallback):
                series = series.fillna(float(market_fallback))
            series = series.fillna(float(global_fallback))
            g[col] = series
        filled_groups.append(g)

    out = pd.concat(filled_groups, ignore_index=True)
    out = out.sort_values(["MARKET", "TENOR_NUM", "TRADE_DATE"]).reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# Modeling
# ---------------------------------------------------------------------------

def select_market_features(train_df: pd.DataFrame, feature_cols: List[str]) -> List[str]:
    usable = []
    for col in feature_cols:
        series = train_df[col]
        if series.notna().sum() == 0:
            continue
        nunique = series.nunique(dropna=True)
        if nunique <= 1:
            continue
        usable.append(col)
    if not usable:
        raise ValueError("No usable non-constant features remain for this market")
    return usable


def build_market_model(train_df: pd.DataFrame, feature_cols: List[str], cfg: Config):
    model_features = select_market_features(train_df, feature_cols)
    X_train = train_df[model_features].to_numpy(dtype=float)

    scaler = None
    if cfg.apply_scaling:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)

    model = IsolationForest(
        n_estimators=cfg.n_estimators,
        max_samples=cfg.max_samples,
        contamination=cfg.contamination,
        max_features=cfg.max_features,
        random_state=cfg.random_state,
        n_jobs=cfg.n_jobs,
    )
    model.fit(X_train)
    train_raw = model.decision_function(X_train)
    return model, scaler, train_raw, model_features


def calibrate_percentile(train_raw: np.ndarray, raw_scores: np.ndarray) -> np.ndarray:
    """Map IF raw scores to anomaly probabilities using the lower train tail.

    IsolationForest decision_function is lower for more anomalous points.
    We convert that lower-tail extremeness into [0,1], where higher means more anomalous.
    """
    sorted_train = np.sort(np.asarray(train_raw, dtype=float))
    raw_scores = np.asarray(raw_scores, dtype=float)
    ranks = np.searchsorted(sorted_train, raw_scores, side="right") / max(len(sorted_train), 1)
    probs = 1.0 - ranks
    return np.clip(probs, 0.0, 1.0)


def score_market(
    df: pd.DataFrame,
    raw_feature_cols: List[str],
    model_feature_cols: List[str],
    model,
    scaler,
    train_raw: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    X = df[model_feature_cols].to_numpy(dtype=float)
    if scaler is not None:
        X = scaler.transform(X)

    raw = model.decision_function(X)
    probs = calibrate_percentile(train_raw=train_raw, raw_scores=raw)

    out = df.copy()
    out["IF_RAW_SCORE"] = raw
    out["IF_ANOMALY_PROB"] = probs
    out["IS_ANOMALY"] = (probs >= threshold).astype(int)

    abs_vals = out[raw_feature_cols].abs()
    out["DOMINANT_FEATURE"] = abs_vals.idxmax(axis=1)
    out["DOMINANT_VALUE"] = abs_vals.max(axis=1)
    out["ANOMALY_RANK_GLOBAL"] = out["IF_ANOMALY_PROB"].rank(ascending=False, method="min").astype(int)
    return out


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------

def _subset_metrics(df: pd.DataFrame, precision_cutoff: float) -> Dict[str, float | int]:
    if df.empty:
        return {
            "rows": 0,
            "anomaly_rate": 0.0,
            "precision_proxy": 0.0,
        }
    anomaly_mask = df["IS_ANOMALY"] == 1
    anomaly_count = int(anomaly_mask.sum())
    if anomaly_count == 0:
        precision_proxy = 0.0
    else:
        precision_proxy = float((df.loc[anomaly_mask, "DOMINANT_VALUE"] > precision_cutoff).mean())
    return {
        "rows": int(len(df)),
        "anomaly_rate": float(df["IS_ANOMALY"].mean()),
        "precision_proxy": precision_proxy,
        "anomaly_count": anomaly_count,
    }


def _bucketize_tenor(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["TENOR_BUCKET"] = pd.cut(
        out["TENOR_NUM"],
        bins=[0, 4, 12, 36],
        labels=["prompt_M1-M4", "mid_M5-M12", "far_M13-M36"],
    )
    return out


def build_summary(scored: pd.DataFrame, cfg: Config, split_dt: pd.Timestamp, market_feature_map: Dict[str, List[str]]) -> Dict[str, object]:
    scored = scored.copy()
    if "SPLIT" not in scored.columns:
        scored["SPLIT"] = np.where(scored["TRADE_DATE"] < split_dt, "train", "test")

    overall = _subset_metrics(scored, cfg.precision_proxy_cutoff)
    test_df = scored.loc[scored["SPLIT"] == "test"].copy()
    test_metrics = _subset_metrics(test_df, cfg.precision_proxy_cutoff)
    train_df = scored.loc[scored["SPLIT"] == "train"].copy()
    train_metrics = _subset_metrics(train_df, cfg.precision_proxy_cutoff)

    market_test = {
        market: _subset_metrics(grp, cfg.precision_proxy_cutoff)
        for market, grp in test_df.groupby("MARKET")
    }
    market_all = {
        market: _subset_metrics(grp, cfg.precision_proxy_cutoff)
        for market, grp in scored.groupby("MARKET")
    }

    test_buckets = _bucketize_tenor(test_df)
    bucket_test = {
        str(bucket): _subset_metrics(grp, cfg.precision_proxy_cutoff)
        for bucket, grp in test_buckets.groupby("TENOR_BUCKET", observed=True)
    }

    all_buckets = _bucketize_tenor(scored)
    bucket_all = {
        str(bucket): _subset_metrics(grp, cfg.precision_proxy_cutoff)
        for bucket, grp in all_buckets.groupby("TENOR_BUCKET", observed=True)
    }

    # Historical sanity checks on known market stress windows.
    event_summary = {}
    event_dates = pd.to_datetime(list(cfg.known_events.keys()))
    for event_dt, label in zip(event_dates, cfg.known_events.values()):
        mask = (scored["TRADE_DATE"] >= event_dt - pd.Timedelta(days=3)) & (scored["TRADE_DATE"] <= event_dt + pd.Timedelta(days=3))
        event_df = scored.loc[mask]
        event_summary[str(event_dt.date())] = {
            "label": label,
            **_subset_metrics(event_df, cfg.precision_proxy_cutoff),
        }

    summary: Dict[str, object] = {
        "resolved_split_date": str(split_dt.date()),
        "feature_cols_requested": cfg.feature_cols,
        "feature_cols_used_by_market": market_feature_map,
        "overall": overall,
        "train": train_metrics,
        "test": test_metrics,
        "by_market_test": market_test,
        "by_market_all": market_all,
        "by_tenor_bucket_test": bucket_test,
        "by_tenor_bucket_all": bucket_all,
        "market_rate_spread_test": _safe_ratio_max_min(v["anomaly_rate"] for v in market_test.values()),
        "tenor_bucket_spread_test": _safe_ratio_max_min(v["anomaly_rate"] for v in bucket_test.values()),
        "known_event_windows_all_data": event_summary,
    }
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Leakage-safe pointwise tenor-level Isolation Forest for FERA feature CSVs.")
    parser.add_argument("--input", default="/mnt/data/Untitled 15_2026-03-29-0307.csv")
    parser.add_argument("--output-dir", default="./fera_tenor_if_outputs")
    parser.add_argument("--split-date", default="2024-07-01")
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--scaling", action="store_true")
    parser.add_argument("--use-multi-window", action="store_true")
    parser.add_argument("--completeness-threshold", type=float, default=0.60)
    parser.add_argument("--impute-window", type=int, default=63)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-samples", default="auto")
    parser.add_argument("--contamination", default="auto")
    parser.add_argument("--max-features", default="1.0")
    args = parser.parse_args()

    cfg = Config(
        input_csv=args.input,
        output_dir=args.output_dir,
        split_date=args.split_date,
        anomaly_threshold=args.threshold,
        apply_scaling=args.scaling,
        use_multi_window=args.use_multi_window,
        completeness_threshold=args.completeness_threshold,
        impute_window_rows=args.impute_window,
        n_estimators=args.n_estimators,
        max_samples=_normalize_param(args.max_samples),
        contamination=_normalize_param(args.contamination),
        max_features=_normalize_param(args.max_features),
    )

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_feature_csv(cfg)
    df = apply_completeness_filter(df, cfg.feature_cols, cfg.completeness_threshold)
    split_dt = resolve_split_date(df, cfg)
    df = causal_impute_leakage_safe(df, cfg.feature_cols, split_dt, cfg.impute_window_rows)

    # Final post-imputation validation.
    if df[cfg.feature_cols].isna().any().any():
        na_cols = df[cfg.feature_cols].columns[df[cfg.feature_cols].isna().any()].tolist()
        raise ValueError(f"NaNs remain after imputation in active features: {na_cols}")

    train_df = df.loc[df["TRADE_DATE"] < split_dt].copy()
    test_df = df.loc[df["TRADE_DATE"] >= split_dt].copy()
    if train_df.empty:
        raise ValueError("Training split is empty")
    if test_df.empty:
        raise ValueError("Test split is empty")

    per_market_counts = (
        df.assign(SPLIT=np.where(df["TRADE_DATE"] < split_dt, "train", "test"))
          .groupby(["MARKET", "SPLIT"]).size().unstack(fill_value=0)
    )
    for market, row in per_market_counts.iterrows():
        if row.get("train", 0) < cfg.min_train_rows_per_market:
            raise ValueError(f"Market {market} has only {row.get('train', 0)} train rows")
        if row.get("test", 0) < cfg.min_test_rows_per_market:
            raise ValueError(f"Market {market} has only {row.get('test', 0)} test rows")

    scored_parts = []
    market_feature_map: Dict[str, List[str]] = {}
    model_bundle = {
        "config": asdict(cfg),
        "resolved_split_date": str(split_dt.date()),
        "markets": {},
    }

    for market in sorted(df["MARKET"].unique()):
        m_train = train_df.loc[train_df["MARKET"] == market].copy()
        m_all = df.loc[df["MARKET"] == market].copy()
        model, scaler, train_raw, model_features = build_market_model(m_train, cfg.feature_cols, cfg)
        market_feature_map[market] = model_features

        scored = score_market(
            df=m_all,
            raw_feature_cols=cfg.feature_cols,
            model_feature_cols=model_features,
            model=model,
            scaler=scaler,
            train_raw=train_raw,
            threshold=cfg.anomaly_threshold,
        )
        scored["SPLIT"] = np.where(scored["TRADE_DATE"] < split_dt, "train", "test")
        scored_parts.append(scored)

        model_bundle["markets"][market] = {
            "model": model,
            "scaler": scaler,
            "feature_cols_used": model_features,
            "train_rows": int(len(m_train)),
            "train_score_min": float(np.min(train_raw)),
            "train_score_max": float(np.max(train_raw)),
            "train_score_p01": float(np.quantile(train_raw, 0.01)),
            "train_score_p05": float(np.quantile(train_raw, 0.05)),
            "train_score_p50": float(np.quantile(train_raw, 0.50)),
        }

    scored_df = pd.concat(scored_parts, ignore_index=True).sort_values(
        ["MARKET", "TRADE_DATE", "TENOR_NUM"]
    ).reset_index(drop=True)

    # Test-only anomaly rank is more aligned with the optimization objective.
    scored_df["ANOMALY_RANK_TEST"] = np.nan
    test_mask = scored_df["SPLIT"] == "test"
    scored_df.loc[test_mask, "ANOMALY_RANK_TEST"] = (
        scored_df.loc[test_mask, "IF_ANOMALY_PROB"].rank(ascending=False, method="min")
    )

    scored_path = output_dir / "fera_tenor_if_scored.csv"
    summary_path = output_dir / "fera_tenor_if_summary.json"
    config_path = output_dir / "fera_tenor_if_config.json"
    model_path = output_dir / "fera_tenor_if_model.joblib"

    output_cols = list(cfg.metadata_cols) + ["FEATURE_COMPLETENESS"] + cfg.feature_cols + [
        "IF_RAW_SCORE",
        "IF_ANOMALY_PROB",
        "IS_ANOMALY",
        "DOMINANT_FEATURE",
        "DOMINANT_VALUE",
        "ANOMALY_RANK_GLOBAL",
        "ANOMALY_RANK_TEST",
        "SPLIT",
    ]
    scored_df[output_cols].to_csv(scored_path, index=False)

    summary = build_summary(scored_df, cfg, split_dt, market_feature_map)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    with open(config_path, "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    joblib.dump(model_bundle, model_path)

    # Backward-compatible stdout block; anomaly_rate is now test-split rate.
    print("---")
    print(f"input_csv:          {cfg.input_csv}")
    print(f"resolved_split:     {split_dt.date()}")
    print(f"rows_scored:        {len(scored_df)}")
    print(f"anomaly_rate:       {summary['test']['anomaly_rate']:.6f}")
    print(f"overall_rate:       {summary['overall']['anomaly_rate']:.6f}")
    print(f"precision_proxy:    {summary['test']['precision_proxy']:.6f}")
    print(f"markets:            {sorted(scored_df['MARKET'].unique().tolist())}")
    print(f"n_features:         {len(cfg.feature_cols)}")
    print(f"scored_csv:         {scored_path}")
    print(f"summary_json:       {summary_path}")
    print(f"model_joblib:       {model_path}")


if __name__ == "__main__":
    main()
