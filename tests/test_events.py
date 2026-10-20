"""
Tests for the event miner and label loader (no Snowflake, no torch).

    python tests/test_events.py
"""

import os
import sys
import tempfile

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ["SHAPE_AE_CACHE"] = tempfile.mkdtemp(prefix="shape_ae_events_")

import prepare  # noqa: E402
import data_sources as ds  # noqa: E402
import events  # noqa: E402

events.EVENTS_DIR = os.path.join(os.environ["SHAPE_AE_CACHE"], "events")
events.EVENTS_FILE = os.path.join(events.EVENTS_DIR, "events.csv")

# Planted shape events on TTF: (date, tenor slice 0-based, size in units of the tenor's daily shape-change std)
PLANTED = [("2018-03-14", slice(6, 9), 10.0), ("2020-10-21", slice(0, 2), -10.0),
           ("2023-06-07", slice(20, 36), 9.0), ("2025-11-19", slice(11, 12), 12.0)]

def planted_snapshot():
    df = prepare.synthesize_curves(seed=11)
    df = df[df["MARKET"].isin(["TTF", "THE"])].copy()
    w = prepare._wides_from_long(df[df["MARKET"] == "TTF"])["TTF"]
    T = np.log(w.to_numpy())
    sd = np.nanstd(prepare.raw_features(T, w.index)["dshape"], axis=0)
    for date, sl, k in PLANTED:
        i = w.index.get_loc(pd.Timestamp(date))
        # a step that persists (level-free): shift the slice, keep the rest
        T[i:, sl] += k * sd[sl]
    w.loc[:, :] = np.exp(T)
    long = w.stack().rename("VALUE_NATIVE").reset_index()
    long.columns = ["TRADE_DATE", "MONTH_OFFSET", "VALUE_NATIVE"]
    long["MARKET"] = "TTF"
    df = pd.concat([df[df["MARKET"] == "THE"], long], ignore_index=True)
    return ds.save_snapshot(df, source="test-planted")

def test_tenor_strings_roundtrip():
    for s in ["M1-M3", "M7-M9,M13", "M36", "M1,M3,M5-M6"]:
        assert events.tenors_to_str(events.str_to_tenors(s)) == s
    assert events.tenors_to_str(events.str_to_tenors(" m1 - m3 ".replace(" ", ""))) == "M1-M3"
    for bad in ["M0", "M37", "M5-M3", "X1", "M1-"]:
        try:
            events.str_to_tenors(bad)
        except ValueError:
            continue
        raise AssertionError(bad)

def test_miner_finds_planted_events_and_tenors():
    snap = planted_snapshot()
    path = events.mine(snapshot=snap, markets=["TTF", "THE"])
    df = pd.read_csv(path)
    ttf = df[(df["market"] == "TTF") & (df["source"] == "CURVE")]
    for date, sl, _ in PLANTED:
        hit = ttf[(pd.to_datetime(ttf["start"]) <= date) & (pd.to_datetime(ttf["end"]) >= date)]
        assert len(hit) == 1, f"planted event {date} not mined"
        sug = events.str_to_tenors(hit["suggested_tenors"].iloc[0])
        truth = np.zeros(36, bool)
        truth[sl] = True
        assert (sug & truth).sum() / truth.sum() >= 0.6, (date, hit["suggested_tenors"].iloc[0])
    # controls are quiet and not inside candidates
    ctl = df[df["source"] == "CONTROL"]
    assert len(ctl) == 2 * events.N_CONTROLS and (ctl["curve_score"] <= 0.3).all()

def test_shape_ae_seeds_merge():
    snap = planted_snapshot()
    rows = []
    for d, dec, rs, drv in [("2018-03-14", 1, 1.8, "LEVEL:A_R3_NEARMID"), ("2018-03-15", 1, 1.2, "LEVEL:A_R3_NEARMID"),
                            ("2024-01-10", 1, 2.5, "TRANSITION:A_R1_PROMPT")]:
        for oid, num, st in [(188, dec, None), (185, rs, None), (161, 0.5, None), (168, None, drv)]:
            rows.append(dict(TRADE_DATE=d, ENTITY_ID=1, OUTPUT_ID=oid, VALUE_NUMBER=num, VALUE_STRING=st))
    f = os.path.join(os.environ["SHAPE_AE_CACHE"], "scores.csv")
    pd.DataFrame(rows).to_csv(f, index=False)
    df = pd.read_csv(events.mine(snapshot=snap, markets=["TTF"], model_scores=f))
    both = df[df["source"] == "CURVE+SHAPE_AE"]
    assert len(both) == 1 and both["shape_ae_alert_days"].iloc[0] == 2
    assert both["shape_ae_driver"].iloc[0] == "LEVEL:A_R3_NEARMID"
    only = df[df["source"] == "SHAPE_AE"]
    assert len(only) == 1 and only["peak"].iloc[0] == "2024-01-10"
    assert isinstance(only["suggested_tenors"].iloc[0], str) or pd.isna(only["suggested_tenors"].iloc[0])

def test_load_validates_and_merges():
    snap = planted_snapshot()
    path = events.mine(snapshot=snap, markets=["TTF"])
    df = pd.read_csv(path, dtype=str).fillna("")
    df.loc[0, ["label", "label_tenors", "label_features", "label_type"]] = ["event", "m7-m9", "slope_d;curvature_d", "Broken_Spread"]
    df.loc[1, ["label", "label_tenors"]] = ["NOT_EVENT", ""]
    bad = df.copy()
    bad.loc[2, ["label", "label_tenors"]] = ["EVENT", ""]          # EVENT without tenors
    bad.loc[3, ["label", "label_features"]] = ["UNSURE", "slope"]   # unknown feature
    bad.loc[4, "label"] = "MAYBE"                                   # bad label
    bp = path.replace(".csv", "_bad.csv")
    bad.to_csv(bp, index=False)
    try:
        events.load(bp)
        raise AssertionError("bad labels accepted")
    except SystemExit as e:
        assert "3 label errors" in str(e)
    assert not os.path.exists(events.EVENTS_FILE)
    gp = path.replace(".csv", "_good.csv")
    df.to_csv(gp, index=False)
    events.load(gp)
    out = pd.read_csv(events.EVENTS_FILE, dtype=str).fillna("")
    assert len(out) == 2
    ev = out[out["label"] == "EVENT"].iloc[0]
    assert ev["label_tenors"] == "M7-M9" and ev["tenor_mask"] == "0" * 6 + "111" + "0" * 27
    assert ev["label_type"] == "broken_spread"
    # relabel: the newer label replaces the old one, others kept
    df.loc[0, "label"] = "NOT_EVENT"
    df.to_csv(gp, index=False)
    events.load(gp)
    out = pd.read_csv(events.EVENTS_FILE, dtype=str)
    assert len(out) == 2 and (out["label"] == "NOT_EVENT").all()
    # carry labels into a re-mine
    p2 = events.mine(snapshot=snap, markets=["TTF"], carry_labels=gp)
    again = pd.read_csv(p2, dtype=str).fillna("")
    assert (again["label"] != "").sum() == 2

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
