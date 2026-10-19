"""
One-time data preparation + fixed evaluation for Shape AE autoresearch.

Builds, per market, everything a forward-curve shape autoencoder needs:
  - the raw price curve M1..M36 and its transform T (LOG or ASINH(p/10))
  - curve channels: level-free shape, aligned day change of the shape, scaled curve
  - 14 scalar curve features (slopes, curvature, winter-summer, basis, RV, corr,
    return z, PCA factors 1-4 of the daily shape change)
  - the fold split (FIT calm days / early-stop / calibration / test, 35 d embargo)
  - frozen robust scalers (median / MAD) fitted on FIT rows only
  - an evaluation set: clean test days plus copies of test days with an injected
    shape anomaly (kink, broken spread, front twist, seasonal hump, back-end bend)

Usage:
    python prepare.py                         # synthetic 7-market curves (no data needed)
    python prepare.py --curves curves.parquet # real curves, long or wide format

Long format columns: TRADE_DATE, MARKET (or ENTITY_ID), MONTH_OFFSET (1..36), VALUE_NATIVE
Wide format columns: TRADE_DATE, MARKET, M1..M36

Data is stored in ~/.cache/autoresearch/shape_ae/.
"""

import os
import json
import argparse

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

N_TENORS = 36            # monthly tenors M1..M36
TIME_BUDGET = 120        # total training time budget in seconds (shared by all markets)
MIN_TRAIN_CURVES = 250   # below this many calm FIT rows, fall back to all FIT rows
STRESS_WINDOW = 20       # trailing window for the calm filter (vol of dT(M1))
Z_WINDOW, Z_MIN = 252, 63  # trailing window for return z / basis level z
MAX_GAP_DAYS = 5         # a day change across a larger gap is not computed
FEATURE_CLIP = 12.0      # scaled features are clipped to +-FEATURE_CLIP
EV_FLOOR = 0.80          # a market scores 0 if clean-test shape EV is below this
BAR_QUANTILE = 0.99      # anomaly bar = this quantile of calibration errors
INJECT_PER_TYPE = 60     # injected anomaly copies per anomaly type per market
INJECT_SEED = 1234

# Fold dates (the live fold of the deployed Shape AE)
TRAIN_END = pd.Timestamp("2023-12-31")
EARLY_STOP = (pd.Timestamp("2024-02-05"), pd.Timestamp("2024-09-30"))
CALIBRATION = (pd.Timestamp("2024-11-05"), pd.Timestamp("2025-06-30"))
TEST_START = pd.Timestamp("2025-08-05")

# Regions used for regional residual scores (0-based tenor slices)
REGIONS = {"R1_PROMPT": slice(0, 3), "R2_SHORTNEAR": slice(3, 6),
           "R3_NEARMID": slice(6, 12), "R4_MIDFAR": slice(12, 36)}

# Scalar features fed to the AE (order is the feature-vector order)
FEATURES = ["return_z", "rv_21d", "corr_20d", "front_slope_d", "slope_d", "back_slope_d",
            "curvature_d", "winter_summer_d", "basis_d", "basis_level_z",
            "pca_1", "pca_2", "pca_3", "pca_4"]
FEATURE_LABELS = {
    "return_z": "Return z", "rv_21d": "RV 21d", "corr_20d": "Corr 20d",
    "front_slope_d": "Front-slope Δ", "slope_d": "Slope Δ", "back_slope_d": "Back-slope Δ",
    "curvature_d": "Curvature Δ", "winter_summer_d": "Winter–summer Δ",
    "basis_d": "Basis Δ", "basis_level_z": "Basis level z",
    "pca_1": "PCA factor 1", "pca_2": "PCA factor 2", "pca_3": "PCA factor 3", "pca_4": "PCA factor 4",
}
N_FEATURES = len(FEATURES)
N_PCA = 4

ANOMALY_TYPES = ["kink", "broken_spread", "front_twist", "seasonal_hump", "back_bend"]

# Per-market config. basis_ref = market whose M1 is used for the basis features.
MARKETS = {
    "TTF":         dict(group="GAS",       transform="LOG",   basis_ref="THE",         stress_mult=2.0),
    "THE":         dict(group="GAS",       transform="LOG",   basis_ref="TTF",         stress_mult=2.0),
    "JKM":         dict(group="LNG",       transform="LOG",   basis_ref="TTF",         stress_mult=2.0),
    "POWER_DE_BL": dict(group="POWER",     transform="ASINH", basis_ref="NORDIC",      stress_mult=2.0),
    "NORDIC":      dict(group="POWER",     transform="ASINH", basis_ref="POWER_DE_BL", stress_mult=3.0),
    "EUA":         dict(group="EUA",       transform="LOG",   basis_ref=None,          stress_mult=2.0),
    "CSS_DE_BL":   dict(group="CSS_DE_BL", transform="ASINH", basis_ref="POWER_DE_BL", stress_mult=2.0),
}

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch", "shape_ae")
DATA_DIR = os.path.join(CACHE_DIR, "markets")
OUTPUT_DIR = os.path.join(CACHE_DIR, "outputs")
RAW_CURVES = os.path.join(CACHE_DIR, "curves_long.csv.gz")

# ---------------------------------------------------------------------------
# Price transform
# ---------------------------------------------------------------------------

def transform(p, kind):
    return np.log(p) if kind == "LOG" else np.arcsinh(p / 10.0)

def inverse_transform(t, kind):
    return np.exp(t) if kind == "LOG" else 10.0 * np.sinh(t)

def tenor_months(dates):
    """Calendar month (1..12) of each tenor: M1 = next calendar month. Shape (n, 36)."""
    m = np.asarray(pd.DatetimeIndex(dates).month)[:, None]
    return (m - 1 + np.arange(1, N_TENORS + 1)[None, :]) % 12 + 1

# ---------------------------------------------------------------------------
# Synthetic curves (used when no real data is given)
# ---------------------------------------------------------------------------

def synthesize_curves(seed=0, start="2015-01-02", end="2026-10-16"):
    """Factor-model forward curves for the 7 FERA markets, in long format."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    n = len(dates)
    k = np.arange(N_TENORS)
    tau = k / (N_TENORS - 1)
    season = np.cos(2 * np.pi * (tenor_months(dates) - 1) / 12.0)  # +1 Jan, -1 Jul

    vol_mult = np.ones(n)
    for a, b, m in [("2020-03-01", "2020-05-31", 2.0), ("2021-08-01", "2022-12-31", 3.0),
                    ("2023-01-01", "2023-03-31", 1.6)]:
        vol_mult[(dates >= a) & (dates <= b)] = m
    for _ in range(12):  # short random stress bursts
        i = rng.integers(0, n - 30)
        vol_mult[i:i + rng.integers(5, 25)] *= rng.uniform(1.5, 2.5)
    crisis = ((dates >= "2021-08-01") & (dates <= "2022-12-31")).astype(float)

    common = {g: rng.standard_normal(n) for g in ["GAS", "POWER", "EUA"]}
    specs = {  # p0, transform, group shock, rho, level vol, slope0, seasonal amp, crisis slope shift
        "TTF":         (20.0, "LOG", "GAS", 0.95, 0.022, 0.05, 0.06, -0.35),
        "THE":         (21.0, "LOG", "GAS", 0.93, 0.022, 0.05, 0.06, -0.35),
        "JKM":         (9.0,  "LOG", "GAS", 0.70, 0.024, 0.02, 0.05, -0.30),
        "POWER_DE_BL": (40.0, "ASINH", "POWER", 0.85, 0.020, 0.03, 0.08, -0.25),
        "NORDIC":      (30.0, "ASINH", "POWER", 0.55, 0.030, 0.06, 0.15, -0.10),
        "EUA":         (10.0, "LOG", "EUA", 1.00, 0.020, 0.03, 0.00, 0.00),
        "CSS_DE_BL":   (4.0,  "ASINH", "POWER", 0.60, 0.060, 0.02, 0.05, -0.10),
    }

    def ou(mean, theta, vol):
        x = np.empty(n)
        x[0] = mean[0]
        e = rng.standard_normal(n) * vol * vol_mult
        for t in range(1, n):
            x[t] = x[t - 1] + theta * (mean[t] - x[t - 1]) + e[t]
        return x

    rows = []
    for mkt, (p0, kind, grp, rho, lvol, slope0, amp0, crisis_slope) in specs.items():
        eps = rho * common[grp] + np.sqrt(1 - rho ** 2) * rng.standard_normal(n)
        y0 = transform(np.array(p0), kind)
        drift = 0.0006 if mkt == "EUA" else 0.0
        level = np.empty(n)
        level[0] = y0
        for t in range(1, n):
            level[t] = level[t - 1] + drift + 0.002 * (y0 + drift * t - level[t - 1]) + lvol * vol_mult[t] * eps[t]
        slope = ou(slope0 + crisis_slope * crisis, 0.02, 0.008)
        curv = ou(np.zeros(n), 0.03, 0.005)
        bend = ou(np.zeros(n), 0.02, 0.003)
        amp = ou(np.full(n, amp0), 0.01, 0.0015 if amp0 > 0 else 0.0)
        y = (level[:, None] + slope[:, None] * tau[None, :]
             + curv[:, None] * np.exp(-k / 3.0)[None, :]
             + bend[:, None] * (tau ** 2)[None, :]
             + amp[:, None] * season
             + 0.002 * rng.standard_normal((n, N_TENORS)))
        price = inverse_transform(y, kind)
        keep = np.ones(n, bool) if mkt not in ("THE", "CSS_DE_BL") else (dates >= "2019-01-02")
        df = pd.DataFrame(price[keep], index=dates[keep], columns=np.arange(1, N_TENORS + 1))
        df = df.stack().rename("VALUE_NATIVE").reset_index()
        df.columns = ["TRADE_DATE", "MONTH_OFFSET", "VALUE_NATIVE"]
        df["MARKET"] = mkt
        rows.append(df)
    return pd.concat(rows, ignore_index=True)

# ---------------------------------------------------------------------------
# Loading raw curves
# ---------------------------------------------------------------------------

def read_curves(path):
    """Read long (VW_ML_LONG style) or wide (M1..M36) curves -> {market: wide DataFrame}."""
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    df.columns = [c.upper() for c in df.columns]
    if "MARKET" not in df.columns and "ENTITY_ID" in df.columns:
        df = df.rename(columns={"ENTITY_ID": "MARKET"})
    df["TRADE_DATE"] = pd.to_datetime(df["TRADE_DATE"])
    df["MARKET"] = df["MARKET"].astype(str)
    out = {}
    if "MONTH_OFFSET" in df.columns:
        df = df[df["MONTH_OFFSET"].between(1, N_TENORS)]
        for mkt, g in df.groupby("MARKET"):
            w = g.pivot_table(index="TRADE_DATE", columns="MONTH_OFFSET", values="VALUE_NATIVE", aggfunc="last")
            out[mkt] = w.reindex(columns=range(1, N_TENORS + 1)).sort_index()
    else:
        cols = [f"M{i}" for i in range(1, N_TENORS + 1)]
        for mkt, g in df.groupby("MARKET"):
            w = g.set_index("TRADE_DATE")[cols].sort_index()
            w.columns = range(1, N_TENORS + 1)
            out[mkt] = w
    return out

def market_config(mkt, wide):
    if mkt in MARKETS:
        return dict(MARKETS[mkt])
    kind = "LOG" if (wide.values[np.isfinite(wide.values)] > 0).all() else "ASINH"
    return dict(group=mkt, transform=kind, basis_ref=None, stress_mult=2.0)

# ---------------------------------------------------------------------------
# Feature construction (causal; one row per trade date)
# ---------------------------------------------------------------------------

def aligned_previous(T, dates):
    """Yesterday's curve on today's tenors. On a roll day (new month) yesterday's
    M(k+1) is today's M(k) (ALIGN mode; M36 is carried flat). NaN after a gap."""
    dates = pd.DatetimeIndex(dates)
    prev = np.full_like(T, np.nan)
    prev[1:] = T[:-1]
    roll = np.zeros(len(T), bool)
    roll[1:] = dates.month[1:] != dates.month[:-1]
    prev[roll, :-1] = prev[roll, 1:]
    gap = np.zeros(len(T), bool)
    gap[1:] = (dates[1:] - dates[:-1]).days > MAX_GAP_DAYS
    gap[0] = True
    prev[gap] = np.nan
    return prev

def _trailing_z(x, window=Z_WINDOW, min_periods=Z_MIN):
    s = pd.Series(x)
    mu = s.rolling(window, min_periods=min_periods).mean().shift(1)
    sd = s.rolling(window, min_periods=min_periods).std().shift(1)
    return ((s - mu) / sd).to_numpy()

def raw_features(T, dates, T_ref=None):
    """Scalar curve features (all but PCA) + level-free shape and its aligned day change.

    T, T_ref: (n, 36) transformed curves on the same dates (T_ref may be None / NaN rows).
    Returns dict with 'scalar' (n, 10) in FEATURES order (without pca_*), 'shape', 'dshape', 'dM1'.
    """
    prev = aligned_previous(T, dates)
    shape = T - T.mean(axis=1, keepdims=True)
    dshape = shape - (prev - prev.mean(axis=1, keepdims=True))
    dM1 = T[:, 0] - prev[:, 0]
    dback = T[:, 12:24].mean(1) - prev[:, 12:24].mean(1)

    s_dm1 = pd.Series(dM1)
    sd_prior = s_dm1.rolling(Z_WINDOW, min_periods=Z_MIN).std().shift(1).to_numpy()
    return_z = dM1 / sd_prior
    rv_21d = s_dm1.rolling(21, min_periods=15).std().to_numpy() * np.sqrt(252)
    corr_20d = s_dm1.rolling(20, min_periods=15).corr(pd.Series(dback)).to_numpy()

    def spread_d(f):
        return f(T) - f(prev)
    front_slope_d = spread_d(lambda c: c[:, 2] - c[:, 0])
    slope_d = spread_d(lambda c: c[:, 11] - c[:, 0])
    back_slope_d = spread_d(lambda c: c[:, 24:36].mean(1) - c[:, 12:24].mean(1))
    curvature_d = spread_d(lambda c: c[:, 0] - 2 * c[:, 5] + c[:, 11])

    months = tenor_months(dates)[:, :12]
    winter = np.isin(months, [10, 11, 12, 1, 2, 3])
    ws = lambda c: (np.where(winter, c[:, :12], 0).sum(1) / winter.sum(1)
                    - np.where(~winter, c[:, :12], 0).sum(1) / (~winter).sum(1))
    winter_summer_d = ws(T) - ws(prev)

    if T_ref is not None and np.isfinite(T_ref[:, 0]).any():
        prev_ref = aligned_previous(T_ref, dates)
        basis = T[:, 0] - T_ref[:, 0]
        basis_d = basis - (prev[:, 0] - prev_ref[:, 0])
        basis_level_z = _trailing_z(basis)
    else:
        basis_d = np.zeros(len(T))
        basis_level_z = np.zeros(len(T))

    scalar = np.stack([return_z, rv_21d, corr_20d, front_slope_d, slope_d, back_slope_d,
                       curvature_d, winter_summer_d, basis_d, basis_level_z], axis=1)
    return dict(scalar=scalar, shape=shape, dshape=dshape, dM1=dM1)

# ---------------------------------------------------------------------------
# Per-market processing
# ---------------------------------------------------------------------------

def _robust(x):
    med = np.nanmedian(x, axis=0)
    mad = np.nanmedian(np.abs(x - med), axis=0) * 1.4826
    mad = np.where(mad <= 1e-6, 1.0, mad)
    return med.astype(np.float64), mad.astype(np.float64)

def _pca(dshape_fit):
    mu = dshape_fit.mean(0)
    _, _, vt = np.linalg.svd(dshape_fit - mu, full_matrices=False)
    comps = vt[:N_PCA].T  # (36, 4)
    flip = np.sign(comps[np.abs(comps).argmax(0), np.arange(N_PCA)])
    return mu, comps * flip

def _inject(T, kind, j_rng, sigma, months_row):
    """Return a perturbed copy of one transformed curve row and the anomaly params."""
    t = T.copy()
    a = j_rng.uniform(1.5, 6.0) * j_rng.choice([-1.0, 1.0])
    sbar = sigma.mean()
    k = np.arange(N_TENORS)
    if kind == "kink":
        j = j_rng.integers(1, N_TENORS - 1)
        t[j] += a * sigma[j]
    elif kind == "broken_spread":
        j = j_rng.integers(1, 30)
        t[j:] += a * sbar
    elif kind == "front_twist":
        t[:6] += a * sigma[:6] * np.array([1.0, 0.6, 0.2, -0.2, -0.4, -0.5])
    elif kind == "seasonal_hump":
        t[:24] += a * sbar * np.cos(2 * np.pi * (months_row[:24] - 1) / 12.0)
    elif kind == "back_bend":
        t[12:] += a * sbar * ((k[12:] - 12) / 23.0) ** 2 * 2.0
    return t

def process_market(mkt, wide, ref_wide, cfg):
    kind = cfg["transform"]
    P = wide.to_numpy(dtype=np.float64)
    ok = np.isfinite(P).all(1)
    if kind == "LOG":
        ok &= (P > 0).all(1)
    wide = wide[ok]
    dates = wide.index
    P = P[ok]
    T = transform(P, kind)
    if ref_wide is not None:
        rcfg = market_config(cfg["basis_ref"], ref_wide)
        rP = ref_wide.reindex(dates).to_numpy(dtype=np.float64)
        T_ref = transform(np.where(rP > 0, rP, np.nan), "LOG") if rcfg["transform"] == "LOG" else transform(rP, "ASINH")
    else:
        T_ref = None

    raw = raw_features(T, dates, T_ref)
    n = len(T)

    # Splits
    is_fit = dates <= TRAIN_END
    is_es = (dates >= EARLY_STOP[0]) & (dates <= EARLY_STOP[1])
    is_cal = (dates >= CALIBRATION[0]) & (dates <= CALIBRATION[1])
    is_test = dates >= TEST_START
    valid = np.isfinite(raw["scalar"]).all(1) & np.isfinite(raw["dshape"]).all(1)

    vol20 = pd.Series(raw["dM1"]).rolling(STRESS_WINDOW, min_periods=10).std().to_numpy()
    med_vol = np.nanmedian(vol20[is_fit & valid])
    stressed = vol20 >= cfg["stress_mult"] * med_vol

    fit_idx = np.where(is_fit & valid & ~stressed)[0]
    calm_fallback = len(fit_idx) < MIN_TRAIN_CURVES
    if calm_fallback:
        fit_idx = np.where(is_fit & valid)[0]
    es_idx = np.where(is_es & valid)[0]
    cal_idx = np.where(is_cal & valid & ~stressed)[0]
    test_idx = np.where(is_test & valid)[0]
    if min(len(fit_idx), len(es_idx), len(cal_idx), len(test_idx)) == 0:
        print(f"  {mkt}: skipped (empty split: fit={len(fit_idx)} es={len(es_idx)} cal={len(cal_idx)} test={len(test_idx)})")
        return None

    # Frozen transforms fitted on FIT rows
    pca_mu, pca_comps = _pca(raw["dshape"][fit_idx])

    def assemble(scalar, shape, dshape):
        pca = np.nan_to_num(dshape - pca_mu) @ pca_comps
        return np.concatenate([scalar, pca], axis=1)

    feat_raw = assemble(raw["scalar"], raw["shape"], raw["dshape"])
    sc = {
        "feat": _robust(feat_raw[fit_idx]),
        "shape": _robust(raw["shape"][fit_idx]),
        "dshape": _robust(raw["dshape"][fit_idx]),
        "curve": _robust(T[fit_idx]),
    }

    def scale(name, x):
        med, mad = sc[name]
        z = (x - med) / mad
        return np.clip(z, -FEATURE_CLIP, FEATURE_CLIP) if name == "feat" else z

    arrays = dict(
        dates=np.asarray(dates.values.astype("datetime64[D]")),
        price=P, T=T, level=T.mean(1),
        months=np.asarray(dates.month, dtype=np.int64),
        curve=scale("curve", T), shape=scale("shape", raw["shape"]),
        dshape=scale("dshape", np.nan_to_num(raw["dshape"])),
        feat=scale("feat", np.nan_to_num(feat_raw)), feat_raw=feat_raw,
        fit_idx=fit_idx, es_idx=es_idx, cal_idx=cal_idx, test_idx=test_idx,
        stressed=stressed,
    )

    # Injected anomalies on test days. Features are recomputed on a trailing window
    # of the perturbed series so every derived feature sees the anomaly consistently.
    rng = np.random.default_rng(INJECT_SEED + sum(map(ord, mkt)))
    sigma = np.nanstd(raw["dshape"][fit_idx], axis=0)
    months_all = tenor_months(dates)
    cand = test_idx[test_idx >= Z_WINDOW + 30]
    inj = {key: [] for key in ["curve", "shape", "dshape", "feat", "level", "months", "price", "T", "type", "row"]}
    for ti, akind in enumerate(ANOMALY_TYPES):
        for _ in range(INJECT_PER_TYPE if len(cand) else 0):
            r = int(rng.choice(cand))
            lo = r - (Z_WINDOW + 30)
            Ts = T[lo:r + 1].copy()
            Ts[-1] = _inject(Ts[-1], akind, rng, sigma, months_all[r])
            rs = raw_features(Ts, dates[lo:r + 1], None if T_ref is None else T_ref[lo:r + 1])
            f = assemble(rs["scalar"][-1:], rs["shape"][-1:], rs["dshape"][-1:])
            inj["curve"].append(scale("curve", Ts[-1:])[0])
            inj["shape"].append(scale("shape", rs["shape"][-1:])[0])
            inj["dshape"].append(scale("dshape", np.nan_to_num(rs["dshape"][-1:]))[0])
            inj["feat"].append(scale("feat", np.nan_to_num(f))[0])
            inj["level"].append(Ts[-1].mean())
            inj["months"].append(dates[r].month)
            inj["T"].append(Ts[-1])
            inj["price"].append(inverse_transform(Ts[-1], kind))
            inj["type"].append(ti)
            inj["row"].append(r)
    for key, v in inj.items():
        arrays["inj_" + key] = np.asarray(v)

    for name, (med, mad) in sc.items():
        arrays[f"scale_{name}_med"] = med
        arrays[f"scale_{name}_mad"] = mad
    arrays["pca_mu"] = pca_mu
    arrays["pca_comps"] = pca_comps

    meta = dict(market=mkt, **cfg, n_rows=n, n_fit=len(fit_idx), n_early_stop=len(es_idx),
                n_calibration=len(cal_idx), n_test=len(test_idx), n_injected=len(inj["row"]),
                calm_fallback=bool(calm_fallback), first_date=str(dates[0].date()),
                last_date=str(dates[-1].date()))
    return arrays, meta

def prepare(curves_path=None, seed=0):
    os.makedirs(DATA_DIR, exist_ok=True)
    if curves_path is None:
        print("No --curves given: generating synthetic curves for the 7 FERA markets")
        synthesize_curves(seed).to_csv(RAW_CURVES, index=False)
        curves_path = RAW_CURVES
    wides = read_curves(curves_path)
    metas = {}
    for mkt, wide in wides.items():
        cfg = market_config(mkt, wide)
        ref = cfg.get("basis_ref")
        if ref is not None and ref not in wides:
            cfg["basis_ref"] = ref = None
        out = process_market(mkt, wide, wides[ref] if ref else None, cfg)
        if out is None:
            continue
        arrays, meta = out
        np.savez_compressed(os.path.join(DATA_DIR, f"{mkt}.npz"), **arrays)
        metas[mkt] = meta
        print(f"  {mkt:12s} {cfg['transform']:5s} rows={meta['n_rows']:5d} fit={meta['n_fit']:5d} "
              f"es={meta['n_early_stop']:4d} cal={meta['n_calibration']:4d} test={meta['n_test']:4d} "
              f"injected={meta['n_injected']:4d}{' (calm fallback)' if meta['calm_fallback'] else ''}")
    with open(os.path.join(DATA_DIR, "meta.json"), "w") as f:
        json.dump(metas, f, indent=2)

# ---------------------------------------------------------------------------
# Runtime utilities (imported by train.py)
# ---------------------------------------------------------------------------

def list_markets():
    with open(os.path.join(DATA_DIR, "meta.json")) as f:
        return list(json.load(f).keys())

class MarketData:
    """All arrays for one market. batch(split) returns float32 model inputs/targets.

    Batch keys (N rows):
      curve  (N, 36)  T(price) robust-scaled per tenor (level included)
      shape  (N, 36)  level-free shape, robust-scaled per tenor   <- curve reconstruction target
      dshape (N, 36)  aligned day change of the shape, robust-scaled
      feat   (N, 14)  scalar features in FEATURES order, robust-scaled <- feature reconstruction target
      month  (N,)     calendar month of the trade date (1..12)
    """

    def __init__(self, market):
        self.market = market
        with open(os.path.join(DATA_DIR, "meta.json")) as f:
            self.meta = json.load(f)[market]
        self.a = dict(np.load(os.path.join(DATA_DIR, f"{market}.npz")))
        self.transform = self.meta["transform"]

    def indices(self, split):
        return {"fit": self.a["fit_idx"], "early_stop": self.a["es_idx"],
                "calibration": self.a["cal_idx"], "test": self.a["test_idx"],
                "all": np.arange(len(self.a["T"]))}[split]

    def batch(self, split):
        if split == "injected":
            a, p = self.a, "inj_"
            return dict(curve=a[p + "curve"].astype(np.float32), shape=a[p + "shape"].astype(np.float32),
                        dshape=a[p + "dshape"].astype(np.float32), feat=a[p + "feat"].astype(np.float32),
                        month=a[p + "months"].astype(np.int64))
        idx = self.indices(split)
        return dict(curve=self.a["curve"][idx].astype(np.float32), shape=self.a["shape"][idx].astype(np.float32),
                    dshape=self.a["dshape"][idx].astype(np.float32), feat=self.a["feat"][idx].astype(np.float32),
                    month=self.a["months"][idx].astype(np.int64))

    def shape_to_price(self, shape_scaled, level):
        """Re-anchor a (reconstructed) scaled shape at the observed level -> native price."""
        t = shape_scaled * self.a["scale_shape_mad"] + self.a["scale_shape_med"] + np.asarray(level)[:, None]
        return inverse_transform(t, self.transform)

    def level(self, split):
        if split == "injected":
            return self.a["inj_level"]
        return self.a["level"][self.indices(split)]

    def price(self, split):
        if split == "injected":
            return self.a["inj_price"]
        return self.a["price"][self.indices(split)]

def load_market(market):
    return MarketData(market)

# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE — this is the fixed metric)
# ---------------------------------------------------------------------------

def _rms(x):
    return np.sqrt(np.mean(np.square(x), axis=1))

def average_precision(scores, labels):
    order = np.argsort(-scores, kind="mergesort")
    l = labels[order]
    tp = np.cumsum(l)
    prec = tp / np.arange(1, len(l) + 1)
    return float((prec * l).sum() / max(l.sum(), 1))

def roc_auc(scores, labels):
    ranks = pd.Series(scores).rank().to_numpy()
    npos, nneg = labels.sum(), (1 - labels).sum()
    return float((ranks[labels == 1].sum() - npos * (npos + 1) / 2) / max(npos * nneg, 1))

def _errors(batch, out):
    shape_hat, feat_hat = np.asarray(out[0], np.float64), np.asarray(out[1], np.float64)
    return _rms(batch["shape"] - shape_hat), _rms(batch["feat"] - feat_hat), shape_hat, feat_hat

def evaluate_market(data, reconstruct_fn):
    """reconstruct_fn(batch) -> (shape_hat (N,36), feat_hat (N,14)[, z (N,L)]) in scaled units."""
    cal, test, inj = data.batch("calibration"), data.batch("test"), data.batch("injected")
    c_cal, f_cal, _, _ = _errors(cal, reconstruct_fn(cal))
    c_test, f_test, shape_hat, _ = _errors(test, reconstruct_fn(test))
    c_inj, f_inj, _, _ = _errors(inj, reconstruct_fn(inj))

    bar_c = max(np.quantile(c_cal, BAR_QUANTILE), 1e-8)
    bar_f = max(np.quantile(f_cal, BAR_QUANTILE), 1e-8)
    score = lambda c, f: np.maximum(c / bar_c, f / bar_f)
    s_test, s_inj = score(c_test, f_test), score(c_inj, f_inj)
    scores = np.concatenate([s_test, s_inj])
    labels = np.concatenate([np.zeros(len(s_test)), np.ones(len(s_inj))])

    y = test["shape"].astype(np.float64)
    ev = 1.0 - np.sum((y - shape_hat) ** 2) / np.sum((y - y.mean(0)) ** 2)
    p_true = data.price("test")
    p_hat = data.shape_to_price(shape_hat, data.level("test"))
    price_mape = float(np.median(np.abs(p_hat - p_true) / np.maximum(np.abs(p_true).mean(1, keepdims=True), 1e-8)))
    ap = average_precision(scores, labels)
    per_type = {}
    types = data.a["inj_type"]
    for ti, name in enumerate(ANOMALY_TYPES):
        m = types == ti
        if m.any():
            per_type[name] = float((s_inj[m] >= 1.0).mean())
    return dict(
        score=ap if ev >= EV_FLOOR else 0.0, auprc=ap, auroc=roc_auc(scores, labels),
        recon_ev=float(ev), far=float((s_test >= 1.0).mean()), recall=float((s_inj >= 1.0).mean()),
        price_mape=price_mape, curve_bar=float(bar_c), feat_bar=float(bar_f), recall_by_type=per_type,
    )

def evaluate(reconstruct_fns):
    """reconstruct_fns: {market: fn}. Returns (val_score, per-market results).

    val_score = mean over markets of AUPRC (clean test days vs injected shape anomalies),
    where the anomaly score is max(curve RMS / bar, feature RMS / bar) with bars = q99 of
    calibration errors; a market whose clean-test shape EV < EV_FLOOR scores 0. Higher is better.
    """
    results = {m: evaluate_market(load_market(m), fn) for m, fn in reconstruct_fns.items()}
    return float(np.mean([r["score"] for r in results.values()])), results

# ---------------------------------------------------------------------------
# Outputs: price-curve + feature reconstruction per day, Shape AE scores, plots
# ---------------------------------------------------------------------------

def _mahal(z, mu, cov_inv):
    d = z - mu
    return np.sqrt(np.maximum(np.einsum("ni,ij,nj->n", d, cov_inv, d), 0))

def export_outputs(market, reconstruct_fn, out_dir=OUTPUT_DIR, split="test", plot=True):
    """Write per-day reconstruction and scores for `split` to out_dir/<market>_*.csv (+ a plot)."""
    os.makedirs(out_dir, exist_ok=True)
    data = load_market(market)
    b = data.batch(split)
    out = reconstruct_fn(b)
    shape_hat, feat_hat = np.asarray(out[0], np.float64), np.asarray(out[1], np.float64)
    z = np.asarray(out[2], np.float64) if len(out) > 2 and out[2] is not None else None
    idx = data.indices(split)
    dates = pd.DatetimeIndex(data.a["dates"][idx])

    res = b["shape"] - shape_hat
    fres = b["feat"] - feat_hat
    cal = data.batch("calibration")
    cal_out = reconstruct_fn(cal)
    c_cal, f_cal, _, _ = _errors(cal, cal_out)
    bar_c, bar_f = np.quantile(c_cal, BAR_QUANTILE), np.quantile(f_cal, BAR_QUANTILE)

    df = pd.DataFrame(index=dates)
    df.index.name = "TRADE_DATE"
    df["A_GLOBAL"] = _rms(res)
    for r, sl in REGIONS.items():
        df["A_" + r] = _rms(res[:, sl])
    kink = np.abs(res[:, 1:-1] - (res[:, :-2] + res[:, 2:]) / 2)
    df["A_BUTTERFLY_MAX"] = kink.max(1)
    df["A_FEAT"] = _rms(fres)
    df["LEVEL_BAR"], df["FEAT_BAR"] = bar_c, bar_f
    df["BAR_MULTIPLE"] = np.maximum(df["A_GLOBAL"] / bar_c, df["A_FEAT"] / bar_f)
    df["DECISION"] = (df["BAR_MULTIPLE"] >= 1.0).astype(int)
    df["SEVERITY"] = pd.cut(df["BAR_MULTIPLE"], [-np.inf, 1, 1.5, 2, np.inf],
                            labels=["NORMAL", "MEDIUM", "HIGH", "CRITICAL"], right=False).astype(str)
    region_rms = df[["A_" + r for r in REGIONS]].to_numpy()
    df["SEVERITY_DRIVER_REGION"] = np.array(list(REGIONS))[region_rms.argmax(1)]
    df["SEVERITY_DRIVER_FEATURE"] = np.array(FEATURES)[np.abs(fres).argmax(1)]
    if z is not None and len(cal_out) > 2 and cal_out[2] is not None:
        zc = np.asarray(cal_out[2], np.float64)
        L = zc.shape[1]
        mu = zc.mean(0)
        cov_inv = np.linalg.pinv(np.cov(zc, rowvar=False).reshape(L, L) + 1e-4 * np.eye(L))
        df["A_MAHAL"] = _mahal(z, mu, cov_inv)
        dzc = np.diff(zc, axis=0)
        dcov_inv = np.linalg.pinv(np.cov(dzc, rowvar=False).reshape(L, L) + 1e-4 * np.eye(L))
        dz = np.vstack([np.full((1, L), np.nan), np.diff(z, axis=0)])
        df["TRANSITION_SCORE"] = _mahal(np.nan_to_num(dz), np.zeros(L), dcov_inv)
        df.loc[df.index[0], "TRANSITION_SCORE"] = np.nan
        for i in range(L):
            df[f"Z{i + 1}"] = z[:, i]
    df.to_csv(os.path.join(out_dir, f"{market}_scores.csv"))

    p_true = data.price(split)
    p_hat = data.shape_to_price(shape_hat, data.level(split))
    tenors = [f"M{i}" for i in range(1, N_TENORS + 1)]
    long = pd.concat({
        "PRICE": pd.DataFrame(p_true, index=dates, columns=tenors),
        "PRICE_RECON": pd.DataFrame(p_hat, index=dates, columns=tenors),
        "SHAPE_SCALED_RESIDUAL": pd.DataFrame(res, index=dates, columns=tenors),
    }, axis=1)
    long.to_csv(os.path.join(out_dir, f"{market}_curve_recon.csv"))
    feats = pd.concat({
        "FEATURE": pd.DataFrame(b["feat"], index=dates, columns=FEATURES),
        "FEATURE_RECON": pd.DataFrame(feat_hat, index=dates, columns=FEATURES),
        "FEATURE_RESIDUAL": pd.DataFrame(fres, index=dates, columns=FEATURES),
    }, axis=1)
    feats.to_csv(os.path.join(out_dir, f"{market}_feature_recon.csv"))

    if plot:
        _plot_day(market, dates[-1], p_true[-1], p_hat[-1], res[-1], fres[-1], df.iloc[-1],
                  os.path.join(out_dir, f"{market}_latest.png"))
    return df

def _plot_day(market, date, p_true, p_hat, res, fres, row, path):
    try:
        import matplotlib
    except ImportError:
        print("matplotlib not installed, skipping plot")
        return
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), gridspec_kw=dict(width_ratios=[1.2, 1, 1]))
    x = np.arange(1, N_TENORS + 1)
    ax = axes[0]
    ax.plot(x, p_true, "o-", ms=3, label="observed")
    ax.plot(x, p_hat, "s--", ms=3, label="reconstruction (re-anchored at level)")
    ax.set_title(f"{market} {date.date()}  price curve (A_GLOBAL {row['A_GLOBAL']:.3f}, bar x{row['BAR_MULTIPLE']:.2f})")
    ax.set_xlabel("tenor M1..M36"); ax.set_ylabel("price (native)"); ax.legend()
    ax = axes[1]
    ax.bar(x, res, color=["#d9534f" if abs(v) > 1 else "#3b82f6" for v in res])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_title("Curve reconstruction residual per tenor (scaled)"); ax.set_xlabel("tenor")
    ax = axes[2]
    order = np.argsort(np.abs(fres))
    ax.barh([FEATURE_LABELS[FEATURES[i]] for i in order], fres[order], color="#3b82f6")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_title(f"Feature reconstruction (A_FEAT {row['A_FEAT']:.3f})")
    ax.set_xlabel("reconstruction residual (normalized feature space)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare forward-curve data for Shape AE autoresearch")
    parser.add_argument("--curves", type=str, default=None,
                        help="Path to curves (.parquet/.csv, long or wide). Omit for synthetic data.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for synthetic curves")
    args = parser.parse_args()
    print(f"Cache directory: {CACHE_DIR}")
    prepare(args.curves, args.seed)
    print("Done! Ready to train.")
