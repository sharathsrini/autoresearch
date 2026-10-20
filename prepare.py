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
import re
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

CACHE_DIR = os.environ.get("SHAPE_AE_CACHE", os.path.join(os.path.expanduser("~"), ".cache", "autoresearch", "shape_ae"))
DATA_DIR = os.path.join(CACHE_DIR, "markets")
OUTPUT_DIR = os.path.join(CACHE_DIR, "outputs")

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
        import data_sources as ds
        ents = ds.load_config()["snowflake"]["entities"]
        df["MARKET"] = df["ENTITY_ID"].map(lambda e: ents.get(int(e), str(e)) if pd.notna(e) else e)
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
    M(k+1) is today's M(k) (ALIGN mode). Today's M36 was yesterday's unquoted M37; it is
    estimated as M25 + (M36 - M24): same calendar month a year earlier plus the annual step,
    which keeps both the seasonal pattern and the slope. NaN after a gap."""
    dates = pd.DatetimeIndex(dates)
    prev = np.full_like(T, np.nan)
    prev[1:] = T[:-1]
    roll = np.zeros(len(T), bool)
    roll[1:] = dates.month[1:] != dates.month[:-1]
    last = prev[roll, -1].copy()
    prev[roll, :-1] = prev[roll, 1:]
    prev[roll, -1] = prev[roll, 23] + (last - prev[roll, 22])  # M25 + (M36 - M24), in yesterday's tenors
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
    """Return a perturbed copy of one transformed curve row."""
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

def process_market(mkt, wide, ref_wide, cfg, store=None):
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

    # Feature store: clean-day values replace the computed ones for the mapped scalar features.
    # Injected days later get store value + (perturbed computed - clean computed).
    scalar_computed = raw["scalar"].copy()
    store_cols, parity = [], None
    if store is not None:
        from data_sources import feature_parity
        S = store.reindex(dates)
        store_cols = [c for c in FEATURES[:-N_PCA] if c in S.columns]
        comp = pd.DataFrame(scalar_computed, index=dates, columns=FEATURES[:-N_PCA])
        parity = feature_parity(comp, S[store_cols])
        for c in store_cols:
            raw["scalar"][:, FEATURES.index(c)] = S[c].to_numpy(dtype=np.float64)

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
    cal_all_idx = np.where(is_cal & valid)[0]
    test_idx = np.where(is_test & valid)[0]
    eval_idx = np.where(valid & (dates >= EARLY_STOP[0]))[0]
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
        cal_all_idx=cal_all_idx, eval_idx=eval_idx, valid=valid,
        stressed=stressed,
    )

    # Injected anomalies on test days. Features are recomputed on a trailing window
    # of the perturbed series so every derived feature sees the anomaly consistently.
    rng = np.random.default_rng(INJECT_SEED + sum(map(ord, mkt)))
    sigma = np.nanstd(raw["dshape"][fit_idx], axis=0)
    months_all = tenor_months(dates)
    cand = test_idx[test_idx >= Z_WINDOW + 30]
    inj = {key: [] for key in ["curve", "shape", "dshape", "feat", "level", "months", "price", "T", "type", "row",
                               "truth"]}
    for ti, akind in enumerate(ANOMALY_TYPES):
        for _ in range(INJECT_PER_TYPE if len(cand) else 0):
            r = int(rng.choice(cand))
            lo = r - (Z_WINDOW + 30)
            Ts = T[lo:r + 1].copy()
            Ts[-1] = _inject(Ts[-1], akind, rng, sigma, months_all[r])
            rs = raw_features(Ts, dates[lo:r + 1], None if T_ref is None else T_ref[lo:r + 1])
            store_offset = raw["scalar"][r] - scalar_computed[r]  # 0 for computed features
            f = assemble(rs["scalar"][-1:] + store_offset, rs["shape"][-1:], rs["dshape"][-1:])
            inj["curve"].append(scale("curve", Ts[-1:])[0])
            inj["shape"].append(scale("shape", rs["shape"][-1:])[0])
            d_inj = scale("dshape", np.nan_to_num(rs["dshape"][-1:]))[0]
            inj["dshape"].append(d_inj)
            # True tenors = where the anomaly moved the (level-free, robust-scaled) day change the
            # model sees by at least half its largest move. For a broken spread that is the shorter
            # side of the break; for a front twist both legs of the twist.
            moved = np.abs(d_inj - arrays["dshape"][r])
            inj["truth"].append(moved >= 0.5 * moved.max())
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
                last_date=str(dates[-1].date()), store_features=store_cols, feature_parity=parity)
    return arrays, meta

def _wides_from_long(df):
    out = {}
    for mkt, g in df.groupby("MARKET"):
        w = g.pivot_table(index="TRADE_DATE", columns="MONTH_OFFSET", values="VALUE_NATIVE", aggfunc="last")
        out[mkt] = w.reindex(columns=range(1, N_TENORS + 1)).sort_index()
    return out

def _long_from_file(path):
    rows = []
    for mkt, w in read_curves(path).items():
        d = w.stack().rename("VALUE_NATIVE").reset_index()
        d.columns = ["TRADE_DATE", "MONTH_OFFSET", "VALUE_NATIVE"]
        d["MARKET"] = mkt
        rows.append(d)
    return pd.concat(rows, ignore_index=True)

def acquire(source, curves_path=None, snapshot=None, seed=0, markets=None):
    """Get curves (+ store features) from a source, snapshot them, return the snapshot id."""
    import data_sources as ds
    cfg = ds.load_config()
    if markets:
        cfg["active_markets"] = list(markets)
    if snapshot:
        return snapshot, cfg
    feats = None
    if source == "synthetic":
        curves, origin = synthesize_curves(seed), f"synthetic(seed={seed})"
    elif source == "file":
        curves, origin = _long_from_file(curves_path), f"file:{os.path.abspath(curves_path)}"
    elif source == "snowflake":
        pull = ds.markets_to_pull(cfg, MARKETS)
        print(f"Pulling {pull} from Snowflake {cfg['snowflake']['curves']['table']} ...")
        conn = ds.connect(cfg)
        try:
            curves = ds.pull_curves(conn, cfg, pull)
            if cfg["features"]["source"] == "snowflake":
                feats = ds.pull_features(conn, cfg, cfg["active_markets"])
        finally:
            conn.close()
        origin = f"snowflake:{cfg['snowflake']['curves']['table']}"
    else:
        raise ValueError(source)
    snap = ds.save_snapshot(curves, feats, source=origin,
                            extra=dict(config_curves=cfg["snowflake"]["curves"] if source == "snowflake" else None))
    return snap, cfg

def prepare(source="synthetic", curves_path=None, snapshot=None, seed=0, markets=None, allow_dq_fail=False):
    import shutil
    import data_sources as ds
    snap, cfg = acquire(source, curves_path, snapshot, seed, markets)
    curves, feats, manifest = ds.load_snapshot(snap)
    print(f"Snapshot {snap}  ({manifest['source']}, created {manifest['created_utc']})")
    wides = _wides_from_long(curves)
    active = [m for m in cfg["active_markets"] if m in wides]
    missing = [m for m in cfg["active_markets"] if m not in wides]
    if missing:
        print(f"WARNING: active markets not in snapshot: {missing}")
    if not active:
        raise SystemExit("No active market has data. Check active_markets in fera.toml / --markets.")

    report = ds.check_curves(curves[curves["MARKET"].isin(wides)],
                             {m: market_config(m, w)["transform"] for m, w in wides.items()})
    ds.print_dq(report)
    fatal = {m: r["fatal"] for m, r in report.items() if r["fatal"] and m in active}
    if fatal and not allow_dq_fail:
        raise SystemExit(f"Data-quality check failed for {list(fatal)} (use --allow-dq-fail to override)")

    if os.path.isdir(DATA_DIR):
        shutil.rmtree(DATA_DIR)  # never mix markets from different snapshots
    os.makedirs(DATA_DIR)
    store_by_mkt = {}
    if feats is not None:
        for m, g in feats.groupby("MARKET"):
            store_by_mkt[m] = g.drop(columns="MARKET").set_index("TRADE_DATE").sort_index()
    metas = {}
    for mkt in active:
        wide = wides[mkt]
        mcfg = market_config(mkt, wide)
        ref = mcfg.get("basis_ref")
        if ref is not None and ref not in wides:
            print(f"  {mkt}: basis reference {ref} not in snapshot, basis features set to 0")
            mcfg["basis_ref"] = ref = None
        out = process_market(mkt, wide, wides[ref] if ref else None, mcfg, store_by_mkt.get(mkt))
        if out is None:
            continue
        arrays, meta = out
        meta.update(snapshot_id=snap, data_quality=report.get(mkt))
        np.savez_compressed(os.path.join(DATA_DIR, f"{mkt}.npz"), **arrays)
        metas[mkt] = meta
        print(f"  {mkt:12s} {mcfg['transform']:5s} rows={meta['n_rows']:5d} fit={meta['n_fit']:5d} "
              f"es={meta['n_early_stop']:4d} cal={meta['n_calibration']:4d} test={meta['n_test']:4d} "
              f"injected={meta['n_injected']:4d}{' (calm fallback)' if meta['calm_fallback'] else ''}"
              f"{' store=' + ','.join(meta['store_features']) if meta['store_features'] else ''}")
        if meta["feature_parity"]:
            for c, v in meta["feature_parity"].items():
                print(f"  {'':12s} parity {c:16s} n={v['n']:5d} corr={v.get('corr', float('nan')):.3f} "
                      f"std_ratio={v.get('std_ratio', float('nan')):.3f}")
    if not metas:
        raise SystemExit("No market could be prepared.")
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
                "calibration": self.a["cal_idx"], "calibration_all": self.a["cal_all_idx"],
                "test": self.a["test_idx"], "eval": self.a["eval_idx"],
                "all": np.arange(len(self.a["T"]))}[split]

    def dates(self, split):
        return pd.DatetimeIndex(self.a["dates"][self.indices(split)])

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
#
# The model rebuilds three things: the level-free shape (state), the roll-aligned day change of the
# shape (dshape: sudden moves) and the 14 features. Per-tenor evidence combines both curve targets:
#   e_k = sqrt((r_shape_k^2 + r_dshape_k^2) / 2), each residual divided by its calibration RMS.
# Explanation is part of the metric. A flagged day is explained by the model's residuals:
#   - tenor residuals and feature residuals f_j are divided by their calibration RMS, so a
#     tenor or feature the model always rebuilds poorly cannot dominate the driver
#   - region scores are normalised by their own calibration quantile, so short regions
#     (R1 = 3 tenors) are not favoured over long ones (R4 = 24 tenors)
# Bars are set so that ALERT_TARGET of all calibration days alert (the desk's 5-10% band).
#
# Per market:
#   injected:  S_inj = recall x (0.7 tenor_acc + 0.3 feature_acc)      on injected anomalies
#   events:    S_ev  = recall x (0.7 tenor_acc + 0.3 feature_acc)      on labelled EVENTs (dev half)
#   blend:     S = (1 - w) S_inj + w S_ev,   w = n_dev_events / (n_dev_events + EVENT_PRIOR)
#   gates:     S = 0 if clean-test shape EV < EV_FLOOR; S x alert_factor, where the alert rate is
#              measured on all out-of-calibration days (early-stop + test): factor 1 inside
#              ALERT_BAND, falling linearly to 0 at 0% and at ALERT_ZERO
# val_score = mean of S over markets. Higher is better, max 1.

ALERT_TARGET = 0.075          # share of calibration days that alert
ALERT_BAND = (0.05, 0.10)     # desk's acceptable alert rate on out-of-calibration days
ALERT_ZERO = 0.20             # alert factor reaches 0 here (and at 0%)
W_TENOR, W_FEATURE = 0.7, 0.3 # weights of tenor vs feature attribution in the explanation score
EVENT_PRIOR = 20              # events weight w = n / (n + EVENT_PRIOR)
FEATURE_TRUTH_MIN = 0.5       # injected days: a feature counts as moved if |delta| / its scale >= this
REGION_QUANTILE = 0.90        # region normaliser = this quantile of calibration region RMS
EVENTS_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "events", "events.csv")
EVENT_HOLDOUT_ENV = "SHAPE_AE_EVENT_HOLDOUT"  # set to 1 (human only) to also score the held-out half

def _rms(x):
    return np.sqrt(np.mean(np.square(x), axis=1))

def average_precision(scores, labels):
    order = np.argsort(-scores, kind="mergesort")
    l = labels[order]
    tp = np.cumsum(l)
    prec = tp / np.arange(1, len(l) + 1)
    return float((prec * l).sum() / max(l.sum(), 1))

def tenors_to_str(mask):
    """Boolean tenor mask -> 'M1-M3,M13'."""
    idx = np.where(mask)[0] + 1
    if len(idx) == 0:
        return ""
    parts, start, prev = [], idx[0], idx[0]
    for k in list(idx[1:]) + [None]:
        if k is not None and k == prev + 1:
            prev = k
            continue
        parts.append(f"M{start}" if start == prev else f"M{start}-M{prev}")
        if k is not None:
            start = prev = k
    return ",".join(parts)

def _residuals(batch, out):
    """out = (shape_hat, dshape_hat, feat_hat[, z]) -> residuals of the three targets."""
    return (batch["shape"].astype(np.float64) - np.asarray(out[0], np.float64),
            batch["dshape"].astype(np.float64) - np.asarray(out[1], np.float64),
            batch["feat"].astype(np.float64) - np.asarray(out[2], np.float64))

def alert_factor(rate):
    lo, hi = ALERT_BAND
    if rate < lo:
        return rate / lo
    if rate > hi:
        return max(0.0, (ALERT_ZERO - rate) / (ALERT_ZERO - hi))
    return 1.0

class Scorer:
    """Frozen per-market scoring built from the model's calibration reconstructions."""

    def __init__(self, data, reconstruct_fn):
        self.data, self.fn = data, reconstruct_fn
        rs, rd, f = _residuals(data.batch("calibration"), reconstruct_fn(data.batch("calibration")))
        self.shape_scale = np.maximum(np.sqrt(np.mean(rs ** 2, 0)), 1e-6)
        self.dshape_scale = np.maximum(np.sqrt(np.mean(rd ** 2, 0)), 1e-6)
        self.feat_scale = np.maximum(np.sqrt(np.mean(f ** 2, 0)), 1e-6)
        rn = self._tenor_evidence(rs, rd)
        self.region_scale = {k: max(np.quantile(_rms(rn[:, sl]), REGION_QUANTILE), 1e-6)
                             for k, sl in REGIONS.items()}
        rsa, rda, fa = _residuals(data.batch("calibration_all"), reconstruct_fn(data.batch("calibration_all")))
        c, fe = _rms(self._tenor_evidence(rsa, rda)), _rms(fa / self.feat_scale)
        lo, hi = 0.5, 1.0
        for _ in range(50):  # same quantile q for both bars, chosen so that ALERT_TARGET of days alert
            q = (lo + hi) / 2
            bc, bf = max(np.quantile(c, q), 1e-9), max(np.quantile(fe, q), 1e-9)
            rate = np.mean(np.maximum(c / bc, fe / bf) >= 1.0)
            lo, hi = (q, hi) if rate > ALERT_TARGET else (lo, q)
        self.bar_curve, self.bar_feat, self.bar_quantile = bc, bf, q

    def _tenor_evidence(self, rs, rd):
        """Signed per-tenor evidence: magnitude combines shape and day-change residuals, sign of the larger."""
        a, b = rs / self.shape_scale, rd / self.dshape_scale
        return np.sqrt((a ** 2 + b ** 2) / 2) * np.where(np.abs(a) >= np.abs(b), np.sign(a), np.sign(b))

    def score(self, batch, out=None):
        out = self.fn(batch) if out is None else out
        r, rd, f = _residuals(batch, out)
        rn, fn = self._tenor_evidence(r, rd), f / self.feat_scale
        curve, feat = _rms(rn), _rms(fn)
        ratio = np.maximum(curve / self.bar_curve, feat / self.bar_feat)
        regions = np.stack([_rms(rn[:, sl]) / self.region_scale[k] for k, sl in REGIONS.items()], 1)
        return dict(out=out, r=r, rd=rd, f=f, rn=rn, fn=fn, curve=curve, feat=feat, ratio=ratio, flag=ratio >= 1.0,
                    regions=regions, driver_region=regions.argmax(1), driver_feature=np.abs(fn).argmax(1))

def tenor_overlap(rn_row, truth):
    """Share of the k true tenors found among the k largest normalised residuals."""
    k = int(truth.sum())
    if k == 0:
        return np.nan
    top = np.argsort(-np.abs(rn_row))[:k]
    return float(truth[top].sum() / k)

def driver_tenors(rn_row):
    """Tenors a trader is pointed to: |normalised residual| >= max(2, half the largest)."""
    a = np.abs(rn_row)
    return a >= max(2.0, 0.5 * a.max())

def _explain_score(tenor_acc, feat_acc):
    t = np.nanmean(tenor_acc) if len(tenor_acc) and np.isfinite(tenor_acc).any() else 0.0
    f = np.nanmean(feat_acc) if len(feat_acc) and np.isfinite(feat_acc).any() else t
    return W_TENOR * t + W_FEATURE * f, t, f

def load_events(market, holdout=False):
    """Labelled EVENT / NOT_EVENT rows of a market; dev or held-out half (stable hash split)."""
    import hashlib
    if not os.path.exists(EVENTS_CSV):
        return pd.DataFrame()
    ev = pd.read_csv(EVENTS_CSV, dtype=str).fillna("")
    ev = ev[(ev["market"] == market) & ev["label"].isin(["EVENT", "NOT_EVENT"])].copy()
    if ev.empty:
        return ev
    half = ev["candidate_id"].map(lambda c: int(hashlib.sha1(c.encode()).hexdigest(), 16) % 2 == 1)
    ev = ev[half == holdout]
    for c in ("start", "peak", "end"):
        ev[c] = pd.to_datetime(ev[c])
    return ev

def evaluate_events(data, scorer, holdout=False):
    ev = load_events(data.market, holdout)
    res = dict(n_events=0, n_not_events=0, recall=np.nan, tenor_acc=np.nan, feature_acc=np.nan,
               not_event_flag_rate=np.nan, score=0.0)
    if ev.empty:
        return res
    dates = data.dates("eval")
    sc = scorer.score(data.batch("eval"))
    feats = {f: i for i, f in enumerate(FEATURES)}
    detected, t_acc, f_acc, ne_flag = [], [], [], []
    for _, e in ev.iterrows():
        rows = np.where((dates >= e["start"]) & (dates <= e["end"]))[0]
        if len(rows) == 0:
            continue  # outside the evaluation window (e.g. in the training years)
        if e["label"] == "NOT_EVENT":
            ne_flag.append(bool(sc["flag"][rows].any()))
            continue
        hit = sc["flag"][rows]
        detected.append(bool(hit.any()))
        if hit.any():
            i = rows[hit][np.argmax(sc["ratio"][rows][hit])]
            truth = np.array([c == "1" for c in e["tenor_mask"]])
            t_acc.append(tenor_overlap(sc["rn"][i], truth))
            labelled = [feats[x] for x in re.split(r"[;,\s]+", e["label_features"]) if x in feats]
            f_acc.append(float(sc["driver_feature"][i] in labelled) if labelled else np.nan)
    res.update(n_events=len(detected), n_not_events=len(ne_flag))
    if detected:
        recall = float(np.mean(detected))
        expl, t, f = _explain_score(np.array(t_acc), np.array(f_acc))
        res.update(recall=recall, tenor_acc=t, feature_acc=f, score=recall * expl)
    if ne_flag:
        res["not_event_flag_rate"] = float(np.mean(ne_flag))
    return res

def evaluate_market(data, reconstruct_fn, holdout=None):
    """reconstruct_fn(batch) -> (shape_hat (N,36), dshape_hat (N,36), feat_hat (N,14)[, z (N,L)]), scaled units."""
    if holdout is None:
        holdout = os.environ.get(EVENT_HOLDOUT_ENV) == "1"
    scorer = Scorer(data, reconstruct_fn)
    test, inj = data.batch("test"), data.batch("injected")
    st, si = scorer.score(test), scorer.score(inj)
    sesp = scorer.score(data.batch("early_stop"))

    # Reconstruction quality on clean test days (shape EV, price error)
    y = test["shape"].astype(np.float64)
    shape_hat = np.asarray(st["out"][0], np.float64)
    ev = 1.0 - np.sum((y - shape_hat) ** 2) / np.sum((y - y.mean(0)) ** 2)
    p_true = data.price("test")
    p_hat = data.shape_to_price(shape_hat, data.level("test"))
    price_mape = float(np.median(np.abs(p_hat - p_true) / np.maximum(np.abs(p_true).mean(1, keepdims=True), 1e-8)))
    alert_rate = float(np.concatenate([sesp["flag"], st["flag"]]).mean())  # out-of-calibration days
    yd = test["dshape"].astype(np.float64)
    ev_dshape = 1.0 - np.sum((yd - np.asarray(st["out"][1], np.float64)) ** 2) / np.sum((yd - yd.mean(0)) ** 2)

    # Injected anomalies: detection and attribution
    truth = data.a["inj_truth"].astype(bool)
    types = data.a["inj_type"]
    dfeat = (inj["feat"] - data.a["feat"][data.a["inj_row"]]) / scorer.feat_scale
    flag = si["flag"]
    t_acc = np.array([tenor_overlap(si["rn"][i], truth[i]) for i in range(len(flag))])
    f_acc = np.full(len(flag), np.nan)
    for i in range(len(flag)):
        moved = np.argsort(-np.abs(dfeat[i]))[:2]
        moved = moved[np.abs(dfeat[i][moved]) >= FEATURE_TRUTH_MIN]
        if len(moved):
            f_acc[i] = float(si["driver_feature"][i] in moved)
    recall = float(flag.mean())
    expl, t_mean, f_mean = _explain_score(t_acc[flag], f_acc[flag])
    s_inj = recall * expl
    region_of_truth = np.stack([truth[:, sl].mean(1) for sl in REGIONS.values()], 1).argmax(1)
    region_acc = float((si["driver_region"][flag] == region_of_truth[flag]).mean()) if flag.any() else 0.0
    by_type = {}
    for ti, name in enumerate(ANOMALY_TYPES):
        m = types == ti
        if m.any():
            fm = m & flag
            by_type[name] = dict(recall=round(float(flag[m].mean()), 3),
                                 tenor_acc=round(float(np.nanmean(t_acc[fm])), 3) if fm.any() else None)

    # Region bias diagnostic: driver-region shares on flagged clean test days
    shares = np.bincount(st["driver_region"][st["flag"]], minlength=len(REGIONS)) / max(st["flag"].sum(), 1)

    # Labelled events
    ev_dev = evaluate_events(data, scorer, holdout=False)
    w = ev_dev["n_events"] / (ev_dev["n_events"] + EVENT_PRIOR)
    blended = (1 - w) * s_inj + w * ev_dev["score"]
    a_factor = alert_factor(alert_rate)
    gates = dict(ev_floor=bool(ev >= EV_FLOOR), alert_factor=round(a_factor, 3))
    scores = np.concatenate([st["ratio"], si["ratio"]])
    labels = np.concatenate([np.zeros(len(st["ratio"])), np.ones(len(si["ratio"]))])
    res = dict(
        score=float(blended * a_factor if gates["ev_floor"] else 0.0), gates=gates,
        inj_score=float(s_inj), recall=recall, tenor_acc=float(t_mean), feature_acc=float(f_mean),
        region_acc=region_acc, recall_by_type=by_type,
        events=ev_dev, event_weight=float(w),
        recon_ev=float(ev), dshape_ev=float(ev_dshape), price_mape=price_mape, alert_rate=alert_rate,
        auprc=average_precision(scores, labels),
        region_share={k: round(float(v), 3) for k, v in zip(REGIONS, shares)},
        bar_quantile=float(scorer.bar_quantile),
    )
    if holdout:
        res["events_holdout"] = evaluate_events(data, scorer, holdout=True)
    return res

def evaluate(reconstruct_fns):
    """reconstruct_fns: {market: fn}. Returns (val_score, per-market results). See the header above."""
    results = {m: evaluate_market(load_market(m), fn) for m, fn in reconstruct_fns.items()}
    return float(np.mean([r["score"] for r in results.values()])), results

# ---------------------------------------------------------------------------
# Outputs: price-curve + feature reconstruction per day, explanation, Shape AE scores, plots
# ---------------------------------------------------------------------------

def _mahal(z, mu, cov_inv):
    d = z - mu
    return np.sqrt(np.maximum(np.einsum("ni,ij,nj->n", d, cov_inv, d), 0))

def export_outputs(market, reconstruct_fn, out_dir=OUTPUT_DIR, split="eval", plot=True):
    """Write per-day scores, explanation and reconstructions for `split` to out_dir/<market>_*.csv."""
    os.makedirs(out_dir, exist_ok=True)
    data = load_market(market)
    scorer = Scorer(data, reconstruct_fn)
    b = data.batch(split)
    sc = scorer.score(b)
    out = sc["out"]
    shape_hat, feat_hat = np.asarray(out[0], np.float64), np.asarray(out[2], np.float64)
    z = np.asarray(out[3], np.float64) if len(out) > 3 and out[3] is not None else None
    dates = data.dates(split)
    p_true = data.price(split)
    p_hat = data.shape_to_price(shape_hat, data.level(split))
    gap = p_true - p_hat

    df = pd.DataFrame(index=dates)
    df.index.name = "TRADE_DATE"
    df["A_GLOBAL"] = _rms(sc["r"])                      # raw RMS shape residual (scaled units), as in v3.1
    df["A_CURVE"] = sc["curve"]                         # normalised curve error (drives the decision)
    df["A_FEAT"] = sc["feat"]                           # normalised feature error
    for k, (name, _) in enumerate(REGIONS.items()):
        df["A_" + name] = sc["regions"][:, k]           # region score / its calibration q90
    kink = np.abs(sc["r"][:, 1:-1] - (sc["r"][:, :-2] + sc["r"][:, 2:]) / 2)
    df["A_BUTTERFLY_MAX"] = kink.max(1)
    df["CURVE_BAR"], df["FEAT_BAR"] = scorer.bar_curve, scorer.bar_feat
    df["BAR_MULTIPLE"] = sc["ratio"]
    df["DECISION"] = sc["flag"].astype(int)
    df["TRIGGER"] = np.where(~sc["flag"], "NONE",
                             np.where(sc["curve"] / scorer.bar_curve >= sc["feat"] / scorer.bar_feat, "CURVE", "FEATURE"))
    df["SEVERITY"] = pd.cut(df["BAR_MULTIPLE"], [-np.inf, 1, 1.5, 2, np.inf],
                            labels=["NORMAL", "MEDIUM", "HIGH", "CRITICAL"], right=False).astype(str)
    df["DRIVER_REGION"] = np.array(list(REGIONS))[sc["driver_region"]]
    df["DRIVER_TENORS"] = [tenors_to_str(driver_tenors(row)) for row in sc["rn"]]
    top = np.abs(sc["rn"]).argmax(1)
    df["DRIVER_TENOR"] = [f"M{k + 1}" for k in top]
    df["DRIVER_TENOR_GAP_NATIVE"] = gap[np.arange(len(top)), top]  # observed - reconstructed, price units
    order = np.argsort(-np.abs(sc["fn"]), axis=1)
    df["DRIVER_FEATURE"] = np.array(FEATURES)[order[:, 0]]
    df["DRIVER_FEATURE_2"] = np.array(FEATURES)[order[:, 1]]
    df["DRIVER_FEATURE_RESID"] = sc["fn"][np.arange(len(order)), order[:, 0]]
    df["EXPLANATION"] = [
        "" if not fl else
        (f"curve shape: {ten or tt} ({reg}), {tt} {g:+.2f} vs rebuilt normal; feature {FEATURE_LABELS[f1]} {fr:+.1f}"
         if trg == "CURVE" else
         f"feature: {FEATURE_LABELS[f1]} {fr:+.1f} vs rebuilt normal; largest curve gap {tt} {g:+.2f}")
        for fl, trg, ten, reg, tt, g, f1, fr in zip(sc["flag"], df["TRIGGER"], df["DRIVER_TENORS"], df["DRIVER_REGION"],
                                                     df["DRIVER_TENOR"], df["DRIVER_TENOR_GAP_NATIVE"],
                                                     df["DRIVER_FEATURE"], df["DRIVER_FEATURE_RESID"])]
    cal_out = scorer.fn(data.batch("calibration"))
    if z is not None and len(cal_out) > 3 and cal_out[3] is not None:
        zc = np.asarray(cal_out[3], np.float64)
        L = zc.shape[1]
        cov_inv = np.linalg.pinv(np.cov(zc, rowvar=False).reshape(L, L) + 1e-4 * np.eye(L))
        df["A_MAHAL"] = _mahal(z, zc.mean(0), cov_inv)
        dcov_inv = np.linalg.pinv(np.cov(np.diff(zc, axis=0), rowvar=False).reshape(L, L) + 1e-4 * np.eye(L))
        dz = np.vstack([np.zeros((1, L)), np.diff(z, axis=0)])
        ts = _mahal(dz, np.zeros(L), dcov_inv)
        ts[0] = np.nan
        ts[1:][(dates[1:] - dates[:-1]).days > MAX_GAP_DAYS] = np.nan
        df["TRANSITION_SCORE"] = ts
        for i in range(L):
            df[f"Z{i + 1}"] = z[:, i]
    df.to_csv(os.path.join(out_dir, f"{market}_scores.csv"))

    tenors = [f"M{i}" for i in range(1, N_TENORS + 1)]
    pd.concat({
        "PRICE": pd.DataFrame(p_true, index=dates, columns=tenors),
        "PRICE_RECON": pd.DataFrame(p_hat, index=dates, columns=tenors),
        "GAP_NATIVE": pd.DataFrame(gap, index=dates, columns=tenors),
        "RESID_NORMALISED": pd.DataFrame(sc["rn"], index=dates, columns=tenors),
    }, axis=1).to_csv(os.path.join(out_dir, f"{market}_curve_recon.csv"))
    pd.concat({
        "FEATURE": pd.DataFrame(b["feat"], index=dates, columns=FEATURES),
        "FEATURE_RECON": pd.DataFrame(feat_hat, index=dates, columns=FEATURES),
        "RESID_NORMALISED": pd.DataFrame(sc["fn"], index=dates, columns=FEATURES),
    }, axis=1).to_csv(os.path.join(out_dir, f"{market}_feature_recon.csv"))

    if plot:
        flagged = np.where(sc["flag"])[0]
        i = flagged[-1] if len(flagged) else len(dates) - 1  # latest flagged day, else the latest day
        _plot_day(market, dates[i], p_true[i], p_hat[i], sc["rn"][i], sc["fn"][i], df.iloc[i],
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
    ax.set_title(f"{market} {date.date()}  bar x{row['BAR_MULTIPLE']:.2f} ({row['SEVERITY']})")
    ax.set_xlabel("tenor M1..M36"); ax.set_ylabel("price (native)"); ax.legend()
    ax = axes[1]
    drv = driver_tenors(res)
    ax.bar(x, res, color=np.where(drv, "#d9534f", "#3b82f6"))
    ax.axhline(0, color="k", lw=0.8)
    ax.set_title(f"Normalised residual per tenor (red = drivers {row['DRIVER_TENORS'] or '-'})")
    ax.set_xlabel("tenor")
    ax = axes[2]
    order = np.argsort(np.abs(fres))
    ax.barh([FEATURE_LABELS[FEATURES[i]] for i in order], fres[order], color="#3b82f6")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_title(f"Feature reconstruction (A_FEAT {row['A_FEAT']:.2f})")
    ax.set_xlabel("normalised reconstruction residual")
    if row["EXPLANATION"]:
        fig.suptitle(row["EXPLANATION"], fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare forward-curve data for Shape AE autoresearch")
    parser.add_argument("--source", choices=["synthetic", "file", "snowflake"], default=None,
                        help="Where curves come from (default: file if --curves else synthetic)")
    parser.add_argument("--curves", type=str, default=None, help="Curves file (.parquet/.csv, long or wide)")
    parser.add_argument("--snapshot", type=str, default=None, help="Re-prepare an existing snapshot id")
    parser.add_argument("--markets", type=str, default=None,
                        help="Comma-separated markets to prepare (default: active_markets in fera.toml)")
    parser.add_argument("--seed", type=int, default=0, help="Seed for synthetic curves")
    parser.add_argument("--allow-dq-fail", action="store_true", help="Prepare despite fatal data-quality findings")
    parser.add_argument("--list-snapshots", action="store_true")
    args = parser.parse_args()
    if args.list_snapshots:
        import data_sources as ds
        print("\n".join(ds.list_snapshots()) or "(none)")
        raise SystemExit(0)
    source = args.source or ("file" if args.curves else "synthetic")
    print(f"Cache directory: {CACHE_DIR}")
    prepare(source, args.curves, args.snapshot, args.seed,
            args.markets.split(",") if args.markets else None, args.allow_dq_fail)
    print("Done! Ready to train.")
