"""
Tests for the data layer (no Snowflake account and no torch needed).

    python tests/test_data_sources.py      # or: pytest tests/
"""

import os
import sys
import re
import types
import tempfile
import importlib

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ["SHAPE_AE_CACHE"] = tempfile.mkdtemp(prefix="shape_ae_test_")

import prepare  # noqa: E402
import data_sources as ds  # noqa: E402

ENTITY = {1: "TTF", 2: "THE", 3: "JKM", 5: "POWER_DE_BL", 7: "EUA", 8: "NORDIC", 9: "CSS_DE_BL"}
_SYNTH = None

def synth():
    global _SYNTH
    if _SYNTH is None:
        _SYNTH = prepare.synthesize_curves(seed=7)
    return _SYNTH.copy()

# ---------------------------------------------------------------------------
# Fake Snowflake connector: answers the two queries from synthetic data
# ---------------------------------------------------------------------------

class FakeCursor:
    def __init__(self, conn):
        self.conn = conn
        self.description = None
        self._df = None

    def execute(self, sql, params):
        self.conn.queries.append((sql, dict(params)))
        ids = [v for k, v in params.items() if k.startswith("e")]
        if "VW_ML_LONG" in sql:
            df = synth()
            df = df[df["MARKET"].isin([ENTITY[i] for i in ids])]
            df["ENTITY_ID"] = df["MARKET"].map({v: k for k, v in ENTITY.items()})
            self._df = df[["TRADE_DATE", "ENTITY_ID", "MONTH_OFFSET", "VALUE_NATIVE"]]
        else:
            df = self.conn.store_frame[self.conn.store_frame["ENTITY_ID"].isin(ids)]
            aliases = dict(re.findall(r"(\w+) AS (\w+)", sql))  # apply SELECT aliases like Snowflake
            self._df = df[list(aliases)].rename(columns=aliases)
        self.description = [(c,) for c in self._df.columns]

    def fetch_pandas_all(self):
        return self._df.reset_index(drop=True)

    def close(self):
        pass

class FakeConn:
    def __init__(self, store_frame=None):
        self.queries = []
        self.store_frame = store_frame

    def cursor(self):
        return FakeCursor(self)

    def close(self):
        pass

def install_fake_connector(conn):
    connector = types.ModuleType("snowflake.connector")
    connector.connect = lambda **kw: conn
    pkg = types.ModuleType("snowflake")
    pkg.connector = connector
    sys.modules["snowflake"] = pkg
    sys.modules["snowflake.connector"] = connector

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_config_loads():
    cfg = ds.load_config()
    assert cfg["snowflake"]["entities"][1] == "TTF"
    assert cfg["active_markets"] == ["TTF", "THE"]

def test_identifier_guard():
    assert ds._ident("RW_CF_FERA.VW_ML_LONG") == "RW_CF_FERA.VW_ML_LONG"
    for bad in ["x; DROP TABLE y", "a b", "a.b.c.d", "1abc", "x--"]:
        try:
            ds._ident(bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad!r}")

def test_markets_to_pull_adds_basis_refs():
    cfg = ds.load_config()
    cfg["active_markets"] = ["JKM", "THE"]
    assert ds.markets_to_pull(cfg, prepare.MARKETS) == ["JKM", "THE", "TTF"]

def test_pull_curves_binds_and_maps():
    conn = FakeConn()
    cfg = ds.load_config()
    df = ds.pull_curves(conn, cfg, ["TTF", "THE"])
    sql, params = conn.queries[0]
    assert "%(e0)s, %(e1)s" in sql and params["e0"] == 1 and params["e1"] == 2
    assert "BETWEEN 1 AND 36" in sql
    assert set(df["MARKET"]) == {"TTF", "THE"}
    assert list(df.columns) == ds.LONG_COLUMNS

def test_snapshot_is_content_addressed_and_verified():
    df = synth()
    df = df[df["MARKET"] == "TTF"]
    a = ds.save_snapshot(df, source="t")
    b = ds.save_snapshot(df.sample(frac=1.0, random_state=0), source="t")  # row order must not matter
    assert a == b
    curves, feats, man = ds.load_snapshot(a)
    assert feats is None and man["markets"]["TTF"]["dates"] == df["TRADE_DATE"].nunique()
    df2 = df.copy()
    df2.iloc[0, df2.columns.get_loc("VALUE_NATIVE")] += 1.0
    assert ds.save_snapshot(df2, source="t") != a
    # tamper with the stored file -> load must fail
    d = os.path.join(ds.SNAPSHOT_ROOT, a)
    path = os.path.join(d, man["files"]["curves"])
    tampered = ds._read_frame(path)
    tampered.loc[0, "VALUE_NATIVE"] += 1.0
    (tampered.to_parquet(path, index=False) if path.endswith(".parquet") else tampered.to_csv(path, index=False))
    try:
        ds.load_snapshot(a)
    except RuntimeError:
        return
    raise AssertionError("tampered snapshot loaded")

def test_dq_flags_problems():
    df = synth()
    df = df[df["MARKET"] == "TTF"].copy()
    dates = sorted(df["TRADE_DATE"].unique())
    dup = df[df["TRADE_DATE"] == dates[100]].head(3)
    stale_src = df[df["TRADE_DATE"] == dates[200]].copy()
    df = df[df["TRADE_DATE"] != dates[201]]
    stale = stale_src.assign(TRADE_DATE=dates[201])
    tick = (df["TRADE_DATE"] == dates[300]) & (df["MONTH_OFFSET"] == 7)
    df.loc[tick, "VALUE_NATIVE"] *= 3.0
    df = pd.concat([df, dup, stale])
    r = ds.check_curves(df, {"TTF": "LOG"})["TTF"]
    assert r["duplicate_keys"] == 6 and r["fatal"]
    assert r["stale_rows"] >= 1
    assert str(pd.Timestamp(dates[300]).date()) in r["jump_outlier_examples"]

def test_prepare_synthetic_gas():
    prepare.prepare("synthetic", seed=3, markets=["TTF", "THE"])
    meta = prepare.list_markets()
    assert meta == ["TTF", "THE"]
    d = prepare.load_market("TTF")
    a = d.a
    # splits are disjoint, ordered and respect the embargo
    dates = pd.DatetimeIndex(a["dates"])
    fit, es, cal, test = (dates[a[k]] for k in ["fit_idx", "es_idx", "cal_idx", "test_idx"])
    assert fit.max() <= prepare.TRAIN_END < es.min()
    assert es.max() < cal.min() and cal.max() < test.min()
    assert (es.min() - fit.max()).days >= 35 and (test.min() - cal.max()).days >= 35
    # scalers are fitted on FIT rows only
    med = np.median(a["T"][a["fit_idx"]], axis=0)
    assert np.allclose(med, a["scale_curve_med"])
    assert d.meta["snapshot_id"].startswith("SNAP_")
    # the price re-anchoring is an exact inverse of the shape scaling
    b = d.batch("test")
    assert np.allclose(d.shape_to_price(b["shape"].astype(np.float64), d.level("test")), d.price("test"), rtol=1e-5)

def test_snowflake_flow_with_feature_store():
    # Store = affine transform of the computed features, so parity is exact and offsets are testable
    prepare.prepare("synthetic", seed=7, markets=["TTF", "THE"])
    clean = {m: prepare.load_market(m) for m in ["TTF", "THE"]}
    rows = []
    for m, d in clean.items():
        fr = d.a["feat_raw"]
        rows.append(pd.DataFrame({
            "TRADE_DATE": pd.DatetimeIndex(d.a["dates"]),
            "ENTITY_ID": {"TTF": 1, "THE": 2}[m],
            "STORE_RET_Z": 2.0 * fr[:, prepare.FEATURES.index("return_z")] + 1.0,
            "STORE_SLOPE": 2.0 * fr[:, prepare.FEATURES.index("slope_d")] + 1.0,
        }))
    store = pd.concat(rows, ignore_index=True)
    conn = FakeConn(store)
    install_fake_connector(conn)
    cfg_text = open(ds.CONFIG_PATH).read()
    cfg_text = cfg_text.replace('source = "computed"', 'source = "snowflake"')
    cfg_text = cfg_text.replace('connection_name = ""', 'connection_name = "fake"')
    cfg_text = cfg_text.replace('table = ""                    # e.g.', 'table = "AN_FERA.FS"  # e.g.')
    cfg_text = cfg_text.replace('# return_z = "RETURN_Z"', 'return_z = "STORE_RET_Z"')
    cfg_text = cfg_text.replace('# rv_21d = "RV_21D"', 'slope_d = "STORE_SLOPE"')
    tmp_cfg = os.path.join(os.environ["SHAPE_AE_CACHE"], "fera.toml")
    with open(tmp_cfg, "w") as f:
        f.write(cfg_text)
    old = ds.CONFIG_PATH
    ds.CONFIG_PATH = tmp_cfg
    ds.load_config.__defaults__ = (tmp_cfg,)
    try:
        prepare.prepare("snowflake", markets=["TTF", "THE"])
    finally:
        ds.CONFIG_PATH = old
        ds.load_config.__defaults__ = (old,)
    d = prepare.load_market("TTF")
    assert sorted(d.meta["store_features"]) == ["return_z", "slope_d"]
    par = d.meta["feature_parity"]["return_z"]
    assert par["corr"] > 0.999 and abs(par["std_ratio"] - 2.0) < 1e-6
    j = prepare.FEATURES.index("return_z")
    # back-end bend leaves M1 untouched -> injected return_z must equal the clean store value of that day
    bb = d.a["inj_type"] == prepare.ANOMALY_TYPES.index("back_bend")
    rows_ = d.a["inj_row"][bb]
    assert np.allclose(d.a["inj_feat"][bb, j], d.a["feat"][rows_, j], atol=1e-9)
    # and store values really are what the model sees on clean days
    k = d.a["test_idx"][:50]
    assert np.allclose(d.a["feat_raw"][k, j], 2.0 * clean["TTF"].a["feat_raw"][k, j] + 1.0)

def test_snowsql_config_mapping():
    path = os.path.join(os.environ["SHAPE_AE_CACHE"], "snowsql_config")
    with open(path, "w") as f:
        f.write('[connections]\naccountname = "acct-default"\nusername = u0\n\n'
                '[connections.fera]\naccountname = "xy12345.eu-central-1"\nusername = "SHARATH"\n'
                'authenticator = externalbrowser\nrolename = FERA_RW\nwarehousename = WH_FERA\n'
                'private_key_path = "~/keys/rsa_key.p8"\nlog_file = ~/x.log\n')
    p = ds.snowsql_params("fera", path)
    assert p["account"] == "xy12345.eu-central-1" and p["user"] == "SHARATH"
    assert p["authenticator"] == "externalbrowser" and p["role"] == "FERA_RW" and p["warehouse"] == "WH_FERA"
    assert p["private_key_file"].endswith("/keys/rsa_key.p8") and not p["private_key_file"].startswith("~")
    assert "log_file" not in p
    assert ds.snowsql_params("default", path)["account"] == "acct-default"

def test_feature_store_requires_mapping():
    cfg = ds.load_config()
    try:
        ds.pull_features(FakeConn(), cfg, ["TTF"])
    except ValueError:
        return
    raise AssertionError("empty feature mapping accepted")

if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as e:
                fails += 1
                import traceback
                traceback.print_exc()
                print(f"FAIL {name}: {e!r}")
    sys.exit(1 if fails else 0)
