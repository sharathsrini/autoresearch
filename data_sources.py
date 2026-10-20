"""
Data sources for Shape AE autoresearch (fixed, do not modify from the agent loop).

- Snowflake pull of the long curve view (VW_ML_LONG) and, optionally, the feature store
- Versioned snapshots: every pull is written once under a content-hash id and reused,
  so an experiment always names the exact data it ran on
- Data-quality report on the curves before anything is trained on them

Usage (via prepare.py):
    python prepare.py --source snowflake          # pull, snapshot, prepare
    python prepare.py --snapshot SNAP_1a2b...     # re-prepare an existing snapshot
"""

import os
import re
import json
import hashlib
import datetime as dt

import numpy as np
import pandas as pd

try:
    import tomllib
except ImportError:  # Python 3.10
    import tomli as tomllib

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(REPO_DIR, "fera.toml")
SNAPSHOT_ROOT = os.path.join(os.environ.get("SHAPE_AE_CACHE", os.path.join(
    os.path.expanduser("~"), ".cache", "autoresearch", "shape_ae")), "snapshots")
SPCS_TOKEN = "/snowflake/session/token"
LONG_COLUMNS = ["TRADE_DATE", "MARKET", "MONTH_OFFSET", "VALUE_NATIVE"]

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path=CONFIG_PATH):
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    cfg["snowflake"]["entities"] = {int(k): v for k, v in cfg["snowflake"]["entities"].items()}
    return cfg

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*(\.[A-Za-z_][A-Za-z0-9_$]*){0,2}$")

def _ident(name):
    """Config-supplied identifiers go into SQL text, so allow only plain (qualified) names."""
    if not _IDENT.match(name):
        raise ValueError(f"invalid Snowflake identifier in fera.toml: {name!r}")
    return name

# ---------------------------------------------------------------------------
# Snowflake connection
# ---------------------------------------------------------------------------

def connect(cfg):
    import snowflake.connector
    sf = cfg["snowflake"]
    opts = {k: sf[k] for k in ("warehouse", "role", "database") if sf.get(k)}
    if os.path.exists(SPCS_TOKEN):
        with open(SPCS_TOKEN) as f:
            token = f.read().strip()
        return snowflake.connector.connect(
            host=os.environ["SNOWFLAKE_HOST"], account=os.environ["SNOWFLAKE_ACCOUNT"],
            authenticator="oauth", token=token, **opts)
    if sf.get("snowsql_connection"):
        return snowflake.connector.connect(**{**snowsql_params(sf["snowsql_connection"]), **opts})
    if sf.get("connection_name"):
        return snowflake.connector.connect(connection_name=sf["connection_name"], **opts)
    env = {k[len("SNOWFLAKE_"):].lower(): v for k, v in os.environ.items()
           if k.startswith("SNOWFLAKE_") and k[len("SNOWFLAKE_"):].lower() in
           ("account", "user", "password", "authenticator", "private_key_file", "warehouse", "role", "database")}
    if "account" not in env:
        raise RuntimeError("No Snowflake credentials: set connection_name in fera.toml, "
                           "SNOWFLAKE_ACCOUNT/SNOWFLAKE_USER/... env vars, or run inside SPCS.")
    return snowflake.connector.connect(**{**env, **opts})

SNOWSQL_CONFIG = os.path.join(os.path.expanduser("~"), ".snowsql", "config")
_SNOWSQL_KEYS = {"accountname": "account", "username": "user", "password": "password",
                 "authenticator": "authenticator", "rolename": "role", "warehousename": "warehouse",
                 "dbname": "database", "schemaname": "schema", "private_key_path": "private_key_file",
                 "host": "host", "port": "port", "region": "region"}

def snowsql_params(name, path=SNOWSQL_CONFIG):
    """Connector kwargs from a SnowSQL config section: [connections.<name>] ('default' = [connections])."""
    import configparser
    cp = configparser.ConfigParser(interpolation=None)
    if not cp.read(os.path.expanduser(path)):
        raise RuntimeError(f"SnowSQL config not found: {path}")
    section = "connections" if name == "default" else f"connections.{name}"
    if section not in cp:
        raise RuntimeError(f"[{section}] not in {path}; sections: {cp.sections()}")
    out = {}
    for k, v in cp[section].items():
        if k in _SNOWSQL_KEYS:
            out[_SNOWSQL_KEYS[k]] = v.strip().strip('"').strip("'")
    if "private_key_file" in out:
        out["private_key_file"] = os.path.expanduser(out["private_key_file"])
        if os.environ.get("PRIVATE_KEY_PASSPHRASE"):  # same env var SnowSQL uses
            out["private_key_file_pwd"] = os.environ["PRIVATE_KEY_PASSPHRASE"]
    if "account" not in out:
        raise RuntimeError(f"[{section}] in {path} has no accountname")
    return out

def _query_df(conn, sql, params):
    cur = conn.cursor()
    try:
        cur.execute(sql, params)
        try:
            df = cur.fetch_pandas_all()
        except Exception:  # connector without pandas/pyarrow extras
            df = pd.DataFrame(cur.fetchall(), columns=[c[0] for c in cur.description])
    finally:
        cur.close()
    df.columns = [c.upper() for c in df.columns]
    return df

def _in_clause(prefix, values):
    names = [f"{prefix}{i}" for i in range(len(values))]
    return ", ".join(f"%({n})s" for n in names), dict(zip(names, values))

# ---------------------------------------------------------------------------
# Pulls
# ---------------------------------------------------------------------------

def markets_to_pull(cfg, market_configs):
    """Active markets plus the markets their basis features refer to."""
    active = list(cfg["active_markets"])
    refs = [market_configs.get(m, {}).get("basis_ref") for m in active]
    return list(dict.fromkeys(active + [r for r in refs if r]))

def pull_curves(conn, cfg, markets):
    sf, cv = cfg["snowflake"], cfg["snowflake"]["curves"]
    name_to_id = {v: k for k, v in sf["entities"].items()}
    missing = [m for m in markets if m not in name_to_id]
    if missing:
        raise ValueError(f"markets without an ENTITY_ID in fera.toml [snowflake.entities]: {missing}")
    in_sql, params = _in_clause("e", [name_to_id[m] for m in markets])
    params["start"] = sf["start_date"]
    sql = (f"SELECT {_ident(cv['date_col'])} AS TRADE_DATE, {_ident(cv['entity_col'])} AS ENTITY_ID, "
           f"{_ident(cv['tenor_col'])} AS MONTH_OFFSET, {_ident(cv['value_col'])} AS VALUE_NATIVE "
           f"FROM {_ident(cv['table'])} "
           f"WHERE {_ident(cv['entity_col'])} IN ({in_sql}) "
           f"AND {_ident(cv['tenor_col'])} BETWEEN 1 AND 36 "
           f"AND {_ident(cv['date_col'])} >= %(start)s")
    if cv.get("extra_filter"):
        sql += f" AND ({cv['extra_filter']})"
    df = _query_df(conn, sql, params)
    df["MARKET"] = df["ENTITY_ID"].astype(int).map(sf["entities"])
    return normalize_long(df)

def pull_features(conn, cfg, markets):
    fs = cfg["features"]["snowflake"]
    cols = fs.get("columns", {})
    if not fs.get("table") or not cols:
        raise ValueError("features.source = 'snowflake' needs [features.snowflake] table and columns")
    sf = cfg["snowflake"]
    name_to_id = {v: k for k, v in sf["entities"].items()}
    in_sql, params = _in_clause("e", [name_to_id[m] for m in markets])
    params["start"] = sf["start_date"]
    base = (f"FROM {_ident(fs['table'])} WHERE {_ident(fs['entity_col'])} IN ({in_sql}) "
            f"AND {_ident(fs['date_col'])} >= %(start)s")
    if fs.get("format", "wide") == "wide":
        sel = ", ".join(f"{_ident(c)} AS {_ident(f.upper())}" for f, c in cols.items())
        df = _query_df(conn, f"SELECT {_ident(fs['date_col'])} AS TRADE_DATE, "
                             f"{_ident(fs['entity_col'])} AS ENTITY_ID, {sel} {base}", params)
    else:
        n_sql, n_params = _in_clause("f", list(cols.values()))
        params.update(n_params)
        df = _query_df(conn, f"SELECT {_ident(fs['date_col'])} AS TRADE_DATE, {_ident(fs['entity_col'])} AS ENTITY_ID, "
                             f"{_ident(fs['name_col'])} AS NAME, {_ident(fs['value_col'])} AS VALUE "
                             f"{base} AND {_ident(fs['name_col'])} IN ({n_sql})", params)
        inv = {v: k.upper() for k, v in cols.items()}
        df["NAME"] = df["NAME"].map(inv)
        df = df.pivot_table(index=["TRADE_DATE", "ENTITY_ID"], columns="NAME", values="VALUE",
                            aggfunc="last").reset_index()
    df["MARKET"] = df["ENTITY_ID"].astype(int).map(sf["entities"])
    df["TRADE_DATE"] = pd.to_datetime(df["TRADE_DATE"])
    df.columns = [c.lower() if c.upper() in {k.upper() for k in cols} else c for c in df.columns]
    return df.drop(columns="ENTITY_ID").sort_values(["MARKET", "TRADE_DATE"]).reset_index(drop=True)

def normalize_long(df):
    df = df[LONG_COLUMNS].copy()
    df["TRADE_DATE"] = pd.to_datetime(df["TRADE_DATE"]).dt.normalize().astype("datetime64[ns]")
    df["MARKET"] = df["MARKET"].astype(str)
    df["MONTH_OFFSET"] = df["MONTH_OFFSET"].astype(int)
    df["VALUE_NATIVE"] = pd.to_numeric(df["VALUE_NATIVE"], errors="coerce").astype(float)
    return df.sort_values(["MARKET", "TRADE_DATE", "MONTH_OFFSET"], kind="mergesort").reset_index(drop=True)

# ---------------------------------------------------------------------------
# Snapshots (content-addressed)
# ---------------------------------------------------------------------------

def _hash_frame(df):
    h = hashlib.sha256()
    h.update(pd.util.hash_pandas_object(df, index=False).values.tobytes())
    h.update(",".join(df.columns).encode())
    return h.hexdigest()

def _write_frame(df, path_noext):
    try:
        df.to_parquet(path_noext + ".parquet", index=False)
        return os.path.basename(path_noext) + ".parquet"
    except ImportError:
        df.to_csv(path_noext + ".csv.gz", index=False)
        return os.path.basename(path_noext) + ".csv.gz"

def _read_frame(path):
    # round_trip: the default fast CSV float parser is not bit-exact, which would break the hash
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path, float_precision="round_trip")
    if "TRADE_DATE" in df.columns:
        df["TRADE_DATE"] = pd.to_datetime(df["TRADE_DATE"])
    return df

def save_snapshot(curves_long, features=None, source="", extra=None):
    """Write a snapshot once; identical content always maps to the same id."""
    curves_long = normalize_long(curves_long)
    curves_sha = digest = _hash_frame(curves_long)
    if features is not None:
        digest = hashlib.sha256((digest + _hash_frame(features)).encode()).hexdigest()
    snap_id = f"SNAP_{digest[:16].upper()}"
    d = os.path.join(SNAPSHOT_ROOT, snap_id)
    if os.path.exists(os.path.join(d, "manifest.json")):
        return snap_id
    os.makedirs(d, exist_ok=True)
    files = {"curves": _write_frame(curves_long, os.path.join(d, "curves_long"))}
    if features is not None:
        files["features"] = _write_frame(features, os.path.join(d, "features"))
    manifest = dict(
        snapshot_id=snap_id, sha256=digest, curves_sha256=curves_sha, source=source, files=files,
        created_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        markets={m: dict(rows=int(len(g)), dates=int(g["TRADE_DATE"].nunique()),
                         first=str(g["TRADE_DATE"].min().date()), last=str(g["TRADE_DATE"].max().date()))
                 for m, g in curves_long.groupby("MARKET")},
        **(extra or {}))
    with open(os.path.join(d, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return snap_id

def load_snapshot(snap_id):
    d = os.path.join(SNAPSHOT_ROOT, snap_id)
    with open(os.path.join(d, "manifest.json")) as f:
        manifest = json.load(f)
    curves = _read_frame(os.path.join(d, manifest["files"]["curves"]))
    curves = normalize_long(curves)
    if _hash_frame(curves) != manifest["curves_sha256"]:
        raise RuntimeError(f"snapshot {snap_id} content does not match its hash (modified on disk?)")
    feats = _read_frame(os.path.join(d, manifest["files"]["features"])) if "features" in manifest["files"] else None
    return curves, feats, manifest

def list_snapshots():
    if not os.path.isdir(SNAPSHOT_ROOT):
        return []
    return sorted(os.listdir(SNAPSHOT_ROOT))

# ---------------------------------------------------------------------------
# Data quality
# ---------------------------------------------------------------------------

def check_curves(curves_long, transforms, n_tenors=36, jump_z=15.0):
    """Per-market data-quality report on a long curve frame.

    transforms: {market: "LOG" | "ASINH"}. Returns {market: dict}; 'fatal' lists blocking problems.
    """
    report = {}
    for mkt, g in curves_long.groupby("MARKET"):
        r = {}
        dup = g.duplicated(["TRADE_DATE", "MONTH_OFFSET"], keep=False)
        r["duplicate_keys"] = int(dup.sum())
        w = g.pivot_table(index="TRADE_DATE", columns="MONTH_OFFSET", values="VALUE_NATIVE", aggfunc="last")
        w = w.reindex(columns=range(1, n_tenors + 1)).sort_index()
        P = w.to_numpy(float)
        r["dates"] = int(len(w))
        r["first"], r["last"] = str(w.index.min().date()), str(w.index.max().date())
        r["weekend_dates"] = int((w.index.dayofweek >= 5).sum())
        complete = np.isfinite(P).all(1)
        r["incomplete_rows"] = int((~complete).sum())
        kind = transforms.get(mkt, "LOG")
        r["non_positive_rows"] = int(((P <= 0) & np.isfinite(P)).any(1).sum()) if kind == "LOG" else 0
        same = np.zeros(len(P), bool)
        same[1:] = np.all(np.isclose(P[1:], P[:-1], equal_nan=False), axis=1)
        r["stale_rows"] = int(same.sum())
        gaps = np.diff(w.index.values).astype("timedelta64[D]").astype(int)
        r["gaps_gt_5d"] = int((gaps > 5).sum())
        r["max_gap_days"] = int(gaps.max()) if len(gaps) else 0
        # Bad-tick screen: robust z of the tenor-wise day change (no roll alignment; roll days excluded)
        with np.errstate(invalid="ignore", divide="ignore"):
            T = np.log(P) if kind == "LOG" else np.arcsinh(P / 10.0)
        d = np.diff(T, axis=0)
        roll = np.asarray(w.index.month[1:] != w.index.month[:-1])
        d[roll] = np.nan
        med = np.nanmedian(d, axis=0)
        mad = np.nanmedian(np.abs(d - med), axis=0) * 1.4826
        z = np.abs(d - med) / np.where(mad > 0, mad, np.nan)
        hit = np.nanmax(np.where(np.isfinite(z), z, 0), axis=1) > jump_z
        r["jump_outlier_days"] = int(hit.sum())
        r["jump_outlier_examples"] = [str(x.date()) for x in w.index[1:][hit][:10]]
        fatal = []
        if r["duplicate_keys"]:
            fatal.append("duplicate (date, tenor) keys: fix the view filter (extra_filter) first")
        if complete.sum() < 500:
            fatal.append(f"only {int(complete.sum())} complete curves")
        r["fatal"] = fatal
        report[mkt] = r
    return report

def print_dq(report):
    cols = ["dates", "first", "last", "incomplete_rows", "non_positive_rows", "duplicate_keys",
            "stale_rows", "gaps_gt_5d", "weekend_dates", "jump_outlier_days"]
    print("Data quality:")
    print("  " + f"{'market':12s} " + " ".join(f"{c[:12]:>12s}" for c in cols))
    for m, r in report.items():
        print("  " + f"{m:12s} " + " ".join(f"{str(r[c]):>12s}" for c in cols))
        if r["jump_outlier_examples"]:
            print(f"  {'':12s} jump outliers e.g. {', '.join(r['jump_outlier_examples'][:5])}")
        for msg in r["fatal"]:
            print(f"  {'':12s} FATAL: {msg}")

def feature_parity(computed, store):
    """Compare in-repo features with the store on common dates: corr and scale ratio per feature."""
    out = {}
    for c in store.columns:
        if c not in computed.columns:
            continue
        a, b = computed[c], store[c]
        m = a.notna() & b.notna()
        if m.sum() < 30:
            out[c] = dict(n=int(m.sum()))
            continue
        out[c] = dict(n=int(m.sum()), corr=float(np.corrcoef(a[m], b[m])[0, 1]),
                      std_ratio=float(b[m].std() / a[m].std()) if a[m].std() > 0 else float("nan"))
    return out
