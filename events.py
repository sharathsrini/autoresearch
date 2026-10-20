"""
Event catalogue for Shape AE: mine candidate events, label them in a spreadsheet, load the labels.

    python events.py mine                       # candidates from the prepared snapshot -> events/candidates_<SNAP>.csv
    python events.py mine --model-scores snowflake   # + live Shape AE alert days (AN_FERA.MODEL_SCORES)
    python events.py mine --model-scores scores.csv  # + Shape AE alerts from an export
    python events.py load events/candidates_<SNAP>.csv   # validate labels -> events/events.csv
    python events.py stats

Candidates come from two independent sources:
  CURVE     model-free: robust z of the roll-aligned daily shape change per tenor, of the
            tenor-kink (butterfly) and of the spread features, against a trailing 252-day scale
  SHAPE_AE  alert days (DECISION = 1) published by the live Shape AE (MODEL_ID 3)
plus CONTROL: random quiet days, to check how many events the miner itself misses.

Suggested tenors / features are in suggested_* columns; the label_* columns are left blank on
purpose so the labeller decides (copy a suggestion when you agree with it).
"""

import os
import re
import sys
import json
import argparse

import numpy as np
import pandas as pd

import prepare
import data_sources as ds

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
EVENTS_DIR = os.path.join(REPO_DIR, "events")
EVENTS_FILE = os.path.join(EVENTS_DIR, "events.csv")
PLOT_DIR = os.path.join(prepare.CACHE_DIR, "event_plots")

# Mining parameters (fixed so candidate lists are reproducible)
SCALE_WINDOW, SCALE_MIN = 252, 126   # trailing window for robust z
Z_TENOR = 6.0                        # tenor / kink z that makes a day a candidate
Z_FEATURE = 6.0                      # spread-feature z that makes a day a candidate
EPISODE_GAP = 3                      # trading days between candidate days still in one episode
MAX_PER_MARKET = 200                 # cap on candidates per market (highest priority kept)
N_CONTROLS = 20                      # random quiet days per market
CONTROL_SEED = 99
MINED_FEATURES = ["front_slope_d", "slope_d", "back_slope_d", "curvature_d", "winter_summer_d",
                  "basis_d", "return_z"]

LABELS = ["EVENT", "NOT_EVENT", "DATA_ERROR", "UNSURE"]
ANOMALY_TYPES = ["kink", "broken_spread", "front_twist", "seasonal_hump", "back_bend", "tilt",
                 "regime_shift", "other"]
SHAPE_AE_OUTPUTS = {161: "A_GLOBAL", 168: "SEVERITY_DRIVER", 185: "REVIEW_SCORE", 188: "DECISION"}
LABEL_COLUMNS = ["label", "label_tenors", "label_features", "label_type", "label_notes", "labeller"]

# ---------------------------------------------------------------------------
# Tenor-set helpers ("M1-M3,M13" <-> boolean mask)
# ---------------------------------------------------------------------------

tenors_to_str = prepare.tenors_to_str

_TENOR_TOKEN = re.compile(r"^M(\d{1,2})(?:-M(\d{1,2}))?$")

def str_to_tenors(text, n=prepare.N_TENORS):
    """'M1-M3, M13' -> bool mask. Raises ValueError on bad syntax."""
    mask = np.zeros(n, bool)
    for tok in str(text).replace(" ", "").upper().split(","):
        if not tok:
            continue
        m = _TENOR_TOKEN.match(tok)
        if not m:
            raise ValueError(f"bad tenor token {tok!r} (use e.g. M7-M9,M13)")
        a = int(m.group(1))
        b = int(m.group(2) or a)
        if not (1 <= a <= b <= n):
            raise ValueError(f"tenor range {tok!r} outside M1..M{n}")
        mask[a - 1:b] = True
    return mask

# ---------------------------------------------------------------------------
# Curve-only mining
# ---------------------------------------------------------------------------

def trailing_robust_z(x):
    """(x - trailing median) / trailing IQR-scale, both from prior days only. x: (n,) or (n, k)."""
    df = pd.DataFrame(x)
    roll = df.rolling(SCALE_WINDOW, min_periods=SCALE_MIN)
    med = roll.median().shift(1)
    iqr = (roll.quantile(0.75) - roll.quantile(0.25)).shift(1) / 1.349
    z = (df - med) / iqr.where(iqr > 1e-12)
    out = z.to_numpy()
    return out[:, 0] if np.ndim(x) == 1 else out

def curve_evidence(wide, ref_wide, kind, ref_kind):
    """Per-day mining evidence for one market. Returns dict of arrays aligned with wide.index."""
    P = wide.to_numpy(float)
    ok = np.isfinite(P).all(1) & ((P > 0).all(1) if kind == "LOG" else True)
    wide = wide[ok]
    P = P[ok]
    dates = wide.index
    T = prepare.transform(P, kind)
    T_ref = None
    if ref_wide is not None:
        rP = ref_wide.reindex(dates).to_numpy(float)
        T_ref = prepare.transform(np.where(rP > 0, rP, np.nan) if ref_kind == "LOG" else rP, ref_kind)
    raw = prepare.raw_features(T, dates, T_ref)
    z_tenor = trailing_robust_z(raw["dshape"])
    d = raw["dshape"]
    kink = d[:, 1:-1] - (d[:, :-2] + d[:, 2:]) / 2
    z_kink = np.full_like(z_tenor, np.nan)
    z_kink[:, 1:-1] = trailing_robust_z(kink)
    names = prepare.FEATURES[:-prepare.N_PCA]
    z_feat = trailing_robust_z(np.stack([raw["scalar"][:, names.index(f)] for f in MINED_FEATURES], 1))
    a_t = np.nan_to_num(np.abs(z_tenor))
    a_k = np.nan_to_num(np.abs(z_kink))
    a_f = np.nan_to_num(np.abs(z_feat))
    score = np.maximum(np.maximum(a_t.max(1), a_k.max(1)) / Z_TENOR, a_f.max(1) / Z_FEATURE)
    warm = np.isnan(z_tenor).all(1)
    score[warm] = np.nan
    prev = prepare.aligned_previous(P, dates)  # native units, roll-aligned
    dP = P - prev
    with np.errstate(invalid="ignore"):
        rel_move = dP - np.nanmean(np.where(np.isfinite(dP), dP, 0), axis=1, keepdims=True)  # vs parallel shift
    return dict(dates=dates, P=P, prev=prev, score=score, z_tenor=z_tenor, z_kink=z_kink, z_feat=z_feat,
                rel_move=rel_move)

def episodes(day_idx, gap=EPISODE_GAP):
    """Group sorted row indices into runs where consecutive members are <= gap rows apart."""
    out = []
    for i in day_idx:
        if out and i - out[-1][-1] <= gap:
            out[-1].append(i)
        else:
            out.append([i])
    return out

def describe_day(ev, i):
    """Suggested tenors, features, type hint and native-unit move for row i."""
    zt = np.nan_to_num(ev["z_tenor"][i])
    zk = np.nan_to_num(ev["z_kink"][i])
    zf = np.nan_to_num(ev["z_feat"][i])
    a = np.maximum(np.abs(zt), np.abs(zk))
    thr = max(3.0, 0.5 * a.max())
    mask = a >= thr
    top = int(np.argmax(a))
    order = np.argsort(-np.abs(zf))[:3]
    feats = ";".join(f"{MINED_FEATURES[j]}({zf[j]:+.1f})" for j in order if abs(zf[j]) >= 3.0)
    tenor_strength = max(np.abs(zt).max(), np.abs(zk).max()) / Z_TENOR
    feat_strength = np.abs(zf).max() / Z_FEATURE
    if np.abs(zk).max() >= np.abs(zt).max() and np.abs(zk).max() >= 0.8 * Z_TENOR and mask.sum() <= 3:
        hint = "kink"
    elif feat_strength >= tenor_strength:
        hint = {"front_slope_d": "front_twist", "slope_d": "tilt", "back_slope_d": "back_bend",
                "curvature_d": "front_twist", "winter_summer_d": "seasonal_hump",
                "basis_d": "other", "return_z": "regime_shift"}[MINED_FEATURES[order[0]]]
    elif mask[:6].mean() > 0.5 and not mask[12:].any():
        hint = "front_twist"
    elif mask[12:].mean() > 0.5:
        hint = "back_bend"
    else:
        hint = "broken_spread"
    reverts = False
    if i + 1 < len(zt):
        nxt = np.nan_to_num(ev["z_tenor"][i + 1])
        reverts = bool(abs(zt[top]) >= Z_TENOR and nxt[top] * zt[top] < 0 and abs(nxt[top]) >= 0.6 * abs(zt[top]))
    return dict(
        suggested_tenors=tenors_to_str(mask), suggested_features=feats, type_hint=hint,
        peak_tenor=f"M{top + 1}", peak_tenor_z=round(float(zt[top] if abs(zt[top]) >= abs(zk[top]) else zk[top]), 2),
        peak_tenor_move_native=round(float(ev["rel_move"][i, top]), 4),
        m1_move_native=round(float(ev["P"][i, 0] - ev["prev"][i, 0]), 4),
        reverts_next_day=reverts,
    )

def mine_curve(market, ev):
    s = ev["score"]
    days = np.where(np.nan_to_num(s) >= 1.0)[0]
    out = []
    for ep in episodes(days):
        peak = ep[int(np.argmax(s[ep]))]
        out.append(dict(market=market, start=ev["dates"][ep[0]], end=ev["dates"][ep[-1]],
                        peak=ev["dates"][peak], n_days=len(ep), curve_score=round(float(s[peak]), 2)))
    return out

# ---------------------------------------------------------------------------
# Shape AE seeds
# ---------------------------------------------------------------------------

def pull_model_scores(cfg, markets):
    ms, sf = cfg["snowflake"]["model_scores"], cfg["snowflake"]
    name_to_id = {v: k for k, v in sf["entities"].items()}
    e_sql, params = ds._in_clause("e", [name_to_id[m] for m in markets])
    o_sql, o_params = ds._in_clause("o", list(SHAPE_AE_OUTPUTS))
    params.update(o_params, model=ms["model_id"], start=sf["start_date"])
    i = ds._ident
    sql = (f"SELECT {i(ms['date_col'])} AS TRADE_DATE, {i(ms['entity_col'])} AS ENTITY_ID, "
           f"{i(ms['output_col'])} AS OUTPUT_ID, {i(ms['number_col'])} AS VALUE_NUMBER, "
           f"{i(ms['string_col'])} AS VALUE_STRING FROM {i(ms['table'])} "
           f"WHERE {i(ms['model_col'])} = %(model)s AND {i(ms['output_col'])} IN ({o_sql}) "
           f"AND {i(ms['entity_col'])} IN ({e_sql}) AND {i(ms['date_col'])} >= %(start)s")
    conn = ds.connect(cfg)
    try:
        df = ds._query_df(conn, sql, params)
    finally:
        conn.close()
    df["MARKET"] = df["ENTITY_ID"].astype(int).map(sf["entities"])
    return df

def read_model_scores_file(path, cfg):
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    df.columns = [c.upper() for c in df.columns]
    if "MARKET" not in df.columns:
        df["MARKET"] = df["ENTITY_ID"].astype(int).map(cfg["snowflake"]["entities"])
    if "VALUE_NUMBER" not in df.columns and "VALUE" in df.columns:
        df["VALUE_NUMBER"] = pd.to_numeric(df["VALUE"], errors="coerce")
        df["VALUE_STRING"] = df["VALUE"].astype(str)
    return df

def shape_ae_table(df):
    """Long model scores -> per (market, date) wide table with DECISION, REVIEW_SCORE, A_GLOBAL, SEVERITY_DRIVER."""
    df = df.copy()
    df["TRADE_DATE"] = pd.to_datetime(df["TRADE_DATE"]).dt.normalize()
    df["OUTPUT_ID"] = df["OUTPUT_ID"].astype(int)
    num = df[df["OUTPUT_ID"].isin([161, 185, 188])].pivot_table(
        index=["MARKET", "TRADE_DATE"], columns="OUTPUT_ID", values="VALUE_NUMBER", aggfunc="last")
    strs = df[df["OUTPUT_ID"] == 168].groupby(["MARKET", "TRADE_DATE"])["VALUE_STRING"].last()
    t = num.rename(columns=SHAPE_AE_OUTPUTS)
    t["SEVERITY_DRIVER"] = strs
    for c in SHAPE_AE_OUTPUTS.values():
        if c not in t.columns:
            t[c] = np.nan
    return t.reset_index()

def mine_shape_ae(market, table, dates):
    t = table[table["MARKET"] == market].set_index("TRADE_DATE").reindex(dates)
    alert = (t["DECISION"].fillna(0) >= 1).to_numpy()
    out = []
    for ep in episodes(np.where(alert)[0]):
        rs = t["REVIEW_SCORE"].to_numpy()[ep]
        peak = ep[int(np.nanargmax(rs))] if np.isfinite(rs).any() else ep[0]
        drivers = t["SEVERITY_DRIVER"].iloc[ep].dropna()
        out.append(dict(market=market, start=dates[ep[0]], end=dates[ep[-1]], peak=dates[peak],
                        shape_ae_alert_days=len(ep),
                        shape_ae_max_bar_multiple=round(float(np.nanmax(rs)), 3) if np.isfinite(rs).any() else np.nan,
                        shape_ae_driver=drivers.mode().iloc[0] if len(drivers) else ""))
    return out

# ---------------------------------------------------------------------------
# Merge, controls, output
# ---------------------------------------------------------------------------

def merge_sources(curve_eps, ae_eps, tol_days=2):
    """Overlapping curve and Shape AE episodes (within tol_days) become one candidate."""
    tol = pd.Timedelta(days=tol_days)
    cands = [dict(c, source="CURVE") for c in curve_eps]
    for a in ae_eps:
        hit = [c for c in cands if c["start"] - tol <= a["end"] and a["start"] <= c["end"] + tol]
        if hit:
            c = hit[0]
            c["start"], c["end"] = min(c["start"], a["start"]), max(c["end"], a["end"])
            c["source"] = "CURVE+SHAPE_AE"
            for k in ("shape_ae_alert_days", "shape_ae_max_bar_multiple", "shape_ae_driver"):
                c[k] = a[k]
        else:
            cands.append(dict(a, source="SHAPE_AE"))
    return cands

def mine(snapshot=None, markets=None, model_scores=None, max_per_market=MAX_PER_MARKET, plots=False,
         carry_labels=None):
    cfg = ds.load_config()
    if snapshot is None:
        snapshot = json.load(open(os.path.join(prepare.DATA_DIR, "meta.json")))
        snapshot = next(iter(snapshot.values()))["snapshot_id"]
    curves, _, manifest = ds.load_snapshot(snapshot)
    wides = prepare._wides_from_long(curves)
    markets = markets or [m for m in cfg["active_markets"] if m in wides]
    table = None
    if model_scores == "snowflake":
        table = shape_ae_table(pull_model_scores(cfg, markets))
    elif model_scores:
        table = shape_ae_table(read_model_scores_file(model_scores, cfg))

    rows = []
    rng = np.random.default_rng(CONTROL_SEED)
    for mkt in markets:
        mcfg = prepare.market_config(mkt, wides[mkt])
        ref = mcfg.get("basis_ref") if mcfg.get("basis_ref") in wides else None
        ref_kind = prepare.market_config(ref, wides[ref])["transform"] if ref else None
        ev = curve_evidence(wides[mkt], wides[ref] if ref else None, mcfg["transform"], ref_kind)
        ae = mine_shape_ae(mkt, table, ev["dates"]) if table is not None else []
        cands = merge_sources(mine_curve(mkt, ev), ae)
        prio = {"CURVE+SHAPE_AE": 0, "CURVE": 1, "SHAPE_AE": 2}
        cands.sort(key=lambda c: (prio[c["source"]], -c.get("curve_score", 0) or 0,
                                  -(c.get("shape_ae_max_bar_multiple") or 0)))
        dropped = max(0, len(cands) - max_per_market)
        cands = cands[:max_per_market]
        # controls: quiet days far from any candidate
        busy = np.zeros(len(ev["dates"]), bool)
        pos = {d: i for i, d in enumerate(ev["dates"])}
        for c in cands:
            busy[max(0, pos[c["start"]] - 5):pos[c["end"]] + 6] = True
        quiet = np.where(~busy & (np.nan_to_num(ev["score"], nan=9) < 0.3))[0]
        for i in rng.choice(quiet, size=min(N_CONTROLS, len(quiet)), replace=False):
            cands.append(dict(market=mkt, start=ev["dates"][i], end=ev["dates"][i], peak=ev["dates"][i],
                              n_days=1, source="CONTROL"))
        for c in cands:
            i = pos[c["peak"]]
            c.update(describe_day(ev, i))
            c["curve_score"] = round(float(np.nan_to_num(ev["score"][i])), 2)
            if plots:
                c["plot_file"] = plot_candidate(mkt, ev, i, c)
        n_src = pd.Series([c["source"] for c in cands]).value_counts().to_dict()
        print(f"  {mkt:12s} candidates={len(cands)} {n_src}" + (f" (dropped {dropped} lowest-priority)" if dropped else ""))
        rows += cands

    df = pd.DataFrame(rows)
    for c in ["n_days", "shape_ae_alert_days", "shape_ae_max_bar_multiple", "shape_ae_driver", "plot_file"]:
        if c not in df.columns:
            df[c] = np.nan
    df["candidate_id"] = df["market"] + "_" + pd.to_datetime(df["peak"]).dt.strftime("%Y%m%d")
    df["snapshot_id"] = snapshot
    for c in ["start", "end", "peak"]:
        df[c] = pd.to_datetime(df[c]).dt.date
    for c in LABEL_COLUMNS:
        df[c] = ""
    if carry_labels:
        old = pd.read_csv(carry_labels, dtype=str).fillna("").set_index("candidate_id")
        hit = df["candidate_id"].isin(old.index)
        for c in LABEL_COLUMNS:
            if c in old.columns:
                df.loc[hit, c] = df.loc[hit, "candidate_id"].map(old[c])
        print(f"  carried labels for {int(hit.sum())} of {len(old)} previous candidates")
    cols = ["candidate_id", "market", "start", "peak", "end", "n_days", "source", "curve_score",
            "suggested_tenors", "suggested_features", "type_hint", "peak_tenor", "peak_tenor_z",
            "peak_tenor_move_native", "m1_move_native", "reverts_next_day", "shape_ae_alert_days",
            "shape_ae_max_bar_multiple", "shape_ae_driver", "plot_file"] + LABEL_COLUMNS + ["snapshot_id"]
    os.makedirs(EVENTS_DIR, exist_ok=True)
    path = os.path.join(EVENTS_DIR, f"candidates_{snapshot}.csv")
    df[cols].to_csv(path, index=False)
    print(f"Wrote {len(df)} candidates to {path}")
    return path

def plot_candidate(market, ev, i, c):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return ""
    os.makedirs(PLOT_DIR, exist_ok=True)
    x = np.arange(1, prepare.N_TENORS + 1)
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.2))
    ax[0].plot(x, ev["P"][i], "o-", ms=3, label=str(ev["dates"][i].date()))
    ax[0].plot(x, ev["prev"][i], "s--", ms=3, label="previous day (roll-aligned)")
    j = max(0, i - 20)
    ax[0].plot(x, ev["P"][j], ":", label=f"{ev['dates'][j].date()} (-20d)")
    ax[0].set_title(f"{market} {c['source']} score {c.get('curve_score', 0)}")
    ax[0].legend(fontsize=8)
    ax[1].bar(x, np.nan_to_num(ev["z_tenor"][i]), color="#3b82f6", label="shape-change z")
    ax[1].plot(x, np.nan_to_num(ev["z_kink"][i]), "r.", label="kink z")
    ax[1].axhline(Z_TENOR, color="k", lw=0.5); ax[1].axhline(-Z_TENOR, color="k", lw=0.5)
    ax[1].set_title(f"suggested tenors: {c.get('suggested_tenors') or '-'}"); ax[1].legend(fontsize=8)
    ax[2].barh(MINED_FEATURES, np.nan_to_num(ev["z_feat"][i]), color="#3b82f6")
    ax[2].axvline(Z_FEATURE, color="k", lw=0.5); ax[2].axvline(-Z_FEATURE, color="k", lw=0.5)
    ax[2].set_title("feature z")
    fig.tight_layout()
    name = f"{market}_{ev['dates'][i]:%Y%m%d}.png"
    fig.savefig(os.path.join(PLOT_DIR, name), dpi=90)
    plt.close(fig)
    return name

# ---------------------------------------------------------------------------
# Load labels
# ---------------------------------------------------------------------------

def validate(df):
    errors = []
    feats = set(prepare.FEATURES)
    for n, r in df.iterrows():
        row = n + 2  # spreadsheet row (header = 1)
        lab = r["label"].strip().upper()
        if not lab:
            continue
        if lab not in LABELS:
            errors.append(f"row {row} {r['candidate_id']}: label {lab!r} not in {LABELS}")
            continue
        try:
            mask = str_to_tenors(r["label_tenors"])
        except ValueError as e:
            errors.append(f"row {row} {r['candidate_id']}: {e}")
            continue
        if lab == "EVENT" and not mask.any():
            errors.append(f"row {row} {r['candidate_id']}: EVENT needs label_tenors (e.g. M7-M9)")
        for f in [x for x in re.split(r"[;,\s]+", r["label_features"]) if x]:
            if f not in feats:
                errors.append(f"row {row} {r['candidate_id']}: unknown feature {f!r}")
        t = r["label_type"].strip().lower()
        if t and t not in ANOMALY_TYPES:
            errors.append(f"row {row} {r['candidate_id']}: label_type {t!r} not in {ANOMALY_TYPES}")
        try:
            s, p, e = (pd.Timestamp(r[c]) for c in ("start", "peak", "end"))
            if not s <= p <= e:
                errors.append(f"row {row} {r['candidate_id']}: need start <= peak <= end")
        except Exception:
            errors.append(f"row {row} {r['candidate_id']}: bad date")
    return errors

def load(path, out=None):
    out = out or EVENTS_FILE
    df = pd.read_csv(path, dtype=str).fillna("")
    missing = [c for c in ["candidate_id", "market", "start", "peak", "end"] + LABEL_COLUMNS if c not in df.columns]
    if missing:
        raise SystemExit(f"{path}: missing columns {missing}")
    if df["candidate_id"].duplicated().any():
        raise SystemExit(f"duplicate candidate_id: {df.loc[df['candidate_id'].duplicated(), 'candidate_id'].tolist()}")
    errors = validate(df)
    if errors:
        print("\n".join(errors))
        raise SystemExit(f"{len(errors)} label errors, nothing written")
    lab = df[df["label"].str.strip() != ""].copy()
    lab["label"] = lab["label"].str.strip().str.upper()
    lab["label_type"] = lab["label_type"].str.strip().str.lower()
    lab["tenor_mask"] = ["".join("1" if b else "0" for b in str_to_tenors(t)) for t in lab["label_tenors"]]
    lab["label_tenors"] = [tenors_to_str(str_to_tenors(t)) for t in lab["label_tenors"]]
    keep = ["candidate_id", "market", "start", "peak", "end", "source", "label", "label_tenors", "tenor_mask",
            "label_features", "label_type", "label_notes", "labeller", "snapshot_id"]
    lab = lab[[c for c in keep if c in lab.columns]]
    if os.path.exists(out):  # merge: new labels replace old ones with the same candidate_id
        old = pd.read_csv(out, dtype=str).fillna("")
        lab = pd.concat([old[~old["candidate_id"].isin(lab["candidate_id"])], lab], ignore_index=True)
    lab = lab.sort_values(["market", "peak"]).reset_index(drop=True)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    lab.to_csv(out, index=False)
    print(f"Wrote {len(lab)} labelled candidates to {out}")
    stats(out)

def stats(path=None):
    path = path or EVENTS_FILE
    if not os.path.exists(path):
        print("No labelled events yet.")
        return
    df = pd.read_csv(path, dtype=str).fillna("")
    print(pd.crosstab(df["market"], df["label"], margins=True).to_string())
    ev = df[df["label"] == "EVENT"]
    if len(ev):
        years = pd.to_datetime(ev["peak"]).dt.year
        print("\nEVENTs per year:\n" + pd.crosstab(ev["market"], years).to_string())
        masks = np.array([[c == "1" for c in m] for m in ev["tenor_mask"]])
        regions = {r: masks[:, sl].any(1).sum() for r, sl in prepare.REGIONS.items()}
        print("\nEVENTs touching each region: " + ", ".join(f"{r}={n}" for r, n in regions.items()))
    if "source" in df.columns:
        print("\nLabel by candidate source:\n" + pd.crosstab(df["source"], df["label"]).to_string())

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Shape AE event catalogue")
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("mine", help="mine candidate events into a CSV for labelling")
    m.add_argument("--snapshot", default=None, help="snapshot id (default: the prepared one)")
    m.add_argument("--markets", default=None, help="comma-separated (default: active_markets)")
    m.add_argument("--model-scores", default=None,
                   help="'snowflake' or a CSV/parquet export of Shape AE MODEL_SCORES (omit: curve-only)")
    m.add_argument("--max-per-market", type=int, default=MAX_PER_MARKET)
    m.add_argument("--plots", action="store_true", help=f"write one PNG per candidate to {PLOT_DIR}")
    m.add_argument("--carry-labels", default=None, help="previous labelled CSV: copy labels by candidate_id")
    l = sub.add_parser("load", help="validate a labelled candidates CSV and merge it into events/events.csv")
    l.add_argument("path")
    sub.add_parser("stats", help="summary of events/events.csv")
    args = ap.parse_args()
    if args.cmd == "mine":
        mine(args.snapshot, args.markets.split(",") if args.markets else None, args.model_scores,
             args.max_per_market, args.plots, args.carry_labels)
    elif args.cmd == "load":
        load(args.path)
    else:
        stats()
