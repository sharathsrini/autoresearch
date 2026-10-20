"""
Tests for the fixed evaluation (no torch: models are numpy functions).

    python tests/test_metric.py
"""

import os
import sys
import tempfile

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ["SHAPE_AE_CACHE"] = tempfile.mkdtemp(prefix="shape_ae_metric_")

import prepare as P  # noqa: E402

P.EVENTS_CSV = os.path.join(os.environ["SHAPE_AE_CACHE"], "events.csv")  # no real labels in tests
_READY = False

def ready():
    global _READY
    if not _READY:
        P.prepare("synthetic", seed=5, markets=["TTF", "THE"])
        _READY = True

def pca_fn(market, L=4):
    d = P.load_market(market)
    b = d.batch("fit")
    X = np.concatenate([b["shape"], b["dshape"], b["feat"]], 1)
    mu = X.mean(0)
    V = np.linalg.svd(X - mu, full_matrices=False)[2][:L].T

    def fn(bb):
        x = np.concatenate([bb["shape"], bb["dshape"], bb["feat"]], 1)
        z = (x - mu) @ V
        r = z @ V.T + mu
        return r[:, :36], r[:, 36:72], r[:, 72:], z
    return fn

def identity_fn(bb):
    return bb["shape"], bb["dshape"], bb["feat"], None

def test_alert_factor():
    assert P.alert_factor(0.075) == 1.0 and P.alert_factor(0.05) == 1.0 and P.alert_factor(0.10) == 1.0
    assert abs(P.alert_factor(0.025) - 0.5) < 1e-12 and P.alert_factor(0.0) == 0.0
    assert abs(P.alert_factor(0.15) - 0.5) < 1e-12 and P.alert_factor(0.30) == 0.0

def test_copying_the_input_scores_zero():
    ready()
    s, res = P.evaluate({m: identity_fn for m in ["TTF", "THE"]})
    assert s == 0.0, res

def test_bars_hit_alert_target_on_calibration():
    ready()
    d = P.load_market("TTF")
    sc = P.Scorer(d, pca_fn("TTF"))
    rate = sc.score(d.batch("calibration_all"))["flag"].mean()
    assert abs(rate - P.ALERT_TARGET) <= 0.01, rate

def test_oracle_detects_and_explains_injected():
    """A model that rebuilds the clean curve of each injected day exposes exactly the anomaly."""
    ready()
    d = P.load_market("TTF")
    base = pca_fn("TTF")
    inj = d.batch("injected")
    rows = d.a["inj_row"]

    def oracle(bb):
        if bb["shape"].shape == inj["shape"].shape and np.array_equal(bb["shape"], inj["shape"]):
            # rebuild the clean day: residual = anomaly + the base model's residual on the clean day
            clean = {k: d.a[k][rows].astype(np.float32) for k in ("shape", "dshape", "feat")}
            return base(clean)
        return base(bb)
    r = P.evaluate_market(d, oracle)
    # ceiling, not perfection: the smallest injections (1.5x a daily move) sit inside normal noise
    assert r["recall"] >= 0.85, r["recall"]
    assert r["tenor_acc"] >= 0.75, r["tenor_acc"]
    assert r["recall_by_type"]["kink"]["tenor_acc"] >= 0.9
    assert min(v["recall"] for k, v in r["recall_by_type"].items() if k != "kink") >= 0.95

def test_region_scores_not_biased_to_short_regions():
    ready()
    d = P.load_market("TTF")
    sc = P.Scorer(d, pca_fn("TTF"))
    s = sc.score(d.batch("calibration"))
    shares = np.bincount(s["driver_region"], minlength=4) / len(s["driver_region"])
    assert shares.max() < 0.5 and shares.min() > 0.08, shares

def _write_events(rows):
    pd.DataFrame(rows).to_csv(P.EVENTS_CSV, index=False)

def test_events_scored_with_dev_holdout_split():
    ready()
    d = P.load_market("TTF")
    base = pca_fn("TTF")
    dates = d.dates("eval")
    ev_rows, bumps = [], {}
    for k, i in enumerate(range(60, 60 + 16 * 20, 20)):
        mask = np.zeros(36, bool)
        mask[6:9] = True
        ev_rows.append(dict(candidate_id=f"TTF_{dates[i]:%Y%m%d}", market="TTF", start=str(dates[i].date()),
                            peak=str(dates[i].date()), end=str(dates[i].date()), label="EVENT",
                            label_tenors="M7-M9", tenor_mask="".join("1" if b else "0" for b in mask),
                            label_features="slope_d", label_type="other"))
        bumps[i] = mask
    _write_events(ev_rows)

    def fn(bb):
        out = list(base(bb))
        if bb["shape"].shape[0] == len(dates) and np.array_equal(bb["shape"], d.batch("eval")["shape"]):
            out[1] = out[1].copy()
            out[2] = out[2].copy()
            for i, m in bumps.items():
                out[1][i, m] -= 40.0  # model "fails to rebuild" the labelled tenors that day
                out[2][i, P.FEATURES.index("slope_d")] -= 40.0
        return tuple(out)
    try:
        r = P.evaluate_market(d, fn, holdout=True)
    finally:
        os.remove(P.EVENTS_CSV)
    dev, hold = r["events"], r["events_holdout"]
    assert dev["n_events"] + hold["n_events"] == 16 and 3 <= dev["n_events"] <= 13
    assert dev["recall"] == 1.0 and dev["tenor_acc"] == 1.0 and dev["feature_acc"] == 1.0
    assert abs(r["event_weight"] - dev["n_events"] / (dev["n_events"] + P.EVENT_PRIOR)) < 1e-12

def test_export_outputs_explain_flagged_days():
    ready()
    out = os.path.join(os.environ["SHAPE_AE_CACHE"], "out")
    df = P.export_outputs("TTF", pca_fn("TTF"), out_dir=out, plot=False)
    flagged = df[df["DECISION"] == 1]
    assert len(flagged) > 0 and (flagged["EXPLANATION"] != "").all()
    assert (df.loc[df["DECISION"] == 0, "EXPLANATION"] == "").all()
    assert set(flagged["TRIGGER"]) <= {"CURVE", "FEATURE"}
    for f in ["TTF_scores.csv", "TTF_curve_recon.csv", "TTF_feature_recon.csv"]:
        assert os.path.exists(os.path.join(out, f))
    assert {"A_MAHAL", "TRANSITION_SCORE", "DRIVER_TENOR_GAP_NATIVE"} <= set(df.columns)

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
