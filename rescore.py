"""Rescore every saved model in runs/ on the val window with scale-invariant
metrics, so row-std and logz runs can be compared apples-to-apples.

For each run:
  - rebuild CurveShapeAE from config.json (with defaults for older runs that
    pre-date the activation/input_scale/hidden_dims fields)
  - load model.pt
  - run forward in the model's *own* preprocessing space (row_std or
    log_robust_z) on the 2024-H2 val window
  - compute:
      val_metric_raw       (recomputed from the model; should match config)
      mean_row_energy_val  (sum_i x_i^2 averaged over val rows)
      var_explained_pooled (1 - mean(Sum_resid^2) / mean(Sum_x^2))
      frac_unexplained     (mean over rows of  Sum_resid^2 / max(Sum_x^2, eps))
      corr_with_F11        (Pearson of per-row recon energy vs F11_Z, val)
      top20_jaccard_F11    (Jaccard overlap of top-20 val dates by AE score
                            with top-20 val dates by |F11_Z|)
      top10_jaccard_F11, top40_jaccard_F11
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/home/user/autoresearch")
import prepare  # fixed module; provides walk_forward_split etc.
from train import CurveShapeAE, _ACT  # the architecture the agent edits

ROOT = Path("/home/user/autoresearch")
RUNS = ROOT / "runs"
OUT = ROOT / "rescored.tsv"

# ---- preprocessing: mirror what train.py did, per input_scale -----------
def preprocess_val(train_raw, val_raw, scale):
    if scale == "log_robust_z":
        tl = np.log(np.maximum(train_raw, 1e-8))
        vl = np.log(np.maximum(val_raw, 1e-8))
        med = np.nanmedian(tl, axis=0)
        mad = np.nanmedian(np.abs(tl - med), axis=0)
        s = np.maximum(mad * 1.4826, 1e-6)
        return ((vl - med) / s).astype(np.float32)
    train_X, _, _ = prepare.row_standardize(train_raw)  # not used here
    val_X, _, _ = prepare.row_standardize(val_raw)
    return val_X

def infer_hidden_dims(cfg):
    hd = cfg.get("hidden_dims")
    if hd:
        return tuple(int(x) for x in str(hd).split(",") if str(x).strip())
    return (int(cfg.get("hidden1", 24)), int(cfg.get("hidden2", 12)))

def infer_input_scale(cfg):
    s = cfg.get("input_scale")
    if s in ("row_std", "log_robust_z"):
        return s
    return "row_std"  # pre-logz runs

def infer_activation(cfg):
    a = cfg.get("activation")
    if a in _ACT:
        return a
    return "tanh"  # original code default

def build_and_load(cfg, state_path):
    model = CurveShapeAE(
        n_tenors=prepare.N_TENORS,
        hidden_dims=infer_hidden_dims(cfg),
        bottleneck=int(cfg["bottleneck"]),
        activation=infer_activation(cfg),
    )
    sd = torch.load(state_path, map_location="cpu", weights_only=True)
    try:
        model.load_state_dict(sd)
    except RuntimeError as e:
        # architecture mismatch — bubble up so caller can skip
        raise
    model.eval()
    return model

def jaccard_topk(score_a, score_b, dates, k):
    if len(dates) < k:
        k = len(dates)
    idx_a = np.argsort(-np.abs(score_a))[:k]
    idx_b = np.argsort(-np.abs(score_b))[:k]
    a = set(dates[idx_a])
    b = set(dates[idx_b])
    if not a and not b:
        return float("nan")
    return len(a & b) / len(a | b)

def main():
    print("loading panels and building val window…")
    panels = prepare.load_panels()
    df = prepare.build_curve_dataset("TTF", panels=panels)
    train_df, val_df, _ = prepare.walk_forward_split(df)
    train_df = prepare.filter_training_rows(train_df)
    train_raw = train_df[prepare.TENORS].values
    val_raw = val_df[prepare.TENORS].values
    F11_Z = val_df["F11_Z"].values
    val_dates = val_df.index.values

    print(f"val rows: {len(val_raw)}  train rows: {len(train_raw)}")

    # F11-only reference metrics
    F11_top10 = set(val_dates[np.argsort(-np.abs(F11_Z))[:10]])
    F11_top20 = set(val_dates[np.argsort(-np.abs(F11_Z))[:20]])
    F11_top40 = set(val_dates[np.argsort(-np.abs(F11_Z))[:40]])

    # cache preprocessings — they only depend on input_scale
    val_cache = {
        "row_std": preprocess_val(train_raw, val_raw, "row_std"),
        "log_robust_z": preprocess_val(train_raw, val_raw, "log_robust_z"),
    }

    rows = []
    skipped = []
    for d in sorted(RUNS.iterdir()):
        cfg_p = d / "config.json"
        sd_p = d / "model.pt"
        if not cfg_p.exists() or not sd_p.exists():
            continue
        cfg = json.load(open(cfg_p))
        scale = infer_input_scale(cfg)
        try:
            model = build_and_load(cfg, sd_p)
        except Exception as e:
            skipped.append((d.name, str(e)))
            continue

        val_X = val_cache[scale]
        with torch.no_grad():
            x = torch.from_numpy(val_X)
            x_hat, z = model(x)
            resid = (x - x_hat).numpy()

        per_row_resid_energy = (resid ** 2).sum(axis=1)
        per_row_input_energy = (val_X ** 2).sum(axis=1)

        val_metric_raw = float(per_row_resid_energy.mean())
        mean_row_energy = float(per_row_input_energy.mean())
        var_explained_pooled = 1.0 - val_metric_raw / max(mean_row_energy, 1e-12)
        eps = 1e-12
        frac_unexplained = float(np.mean(per_row_resid_energy
                                         / np.maximum(per_row_input_energy, eps)))
        corr = prepare._pearson(per_row_resid_energy, F11_Z)

        top10 = jaccard_topk(per_row_resid_energy, np.abs(F11_Z), val_dates, 10)
        top20 = jaccard_topk(per_row_resid_energy, np.abs(F11_Z), val_dates, 20)
        top40 = jaccard_topk(per_row_resid_energy, np.abs(F11_Z), val_dates, 40)

        rows.append({
            "tag": d.name,
            "input_scale": scale,
            "bottleneck": int(cfg.get("bottleneck", -1)),
            "hidden_dims": ",".join(str(x) for x in infer_hidden_dims(cfg)),
            "activation": infer_activation(cfg),
            "optimizer": cfg.get("optimizer", "adam"),
            "wd": float(cfg.get("weight_decay", 0.0) or 0.0),
            "contractive": float(cfg.get("contractive", 0.0) or 0.0),
            "val_metric_raw": val_metric_raw,
            "mean_row_energy": mean_row_energy,
            "var_explained_pooled": var_explained_pooled,
            "frac_unexplained": frac_unexplained,
            "corr_with_F11": corr,
            "top10_jaccard": top10,
            "top20_jaccard": top20,
            "top40_jaccard": top40,
            "logged_val_metric": float(cfg.get("metrics", {}).get("val_metric", float("nan"))),
            "logged_kept": bool(cfg.get("kept", False)),
        })

    # write tsv
    headers = [
        "tag", "input_scale", "bottleneck", "hidden_dims", "activation",
        "optimizer", "wd", "contractive",
        "val_metric_raw", "mean_row_energy",
        "var_explained_pooled", "frac_unexplained",
        "corr_with_F11", "top10_jaccard", "top20_jaccard", "top40_jaccard",
        "logged_val_metric", "logged_kept",
    ]
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\t".join(headers) + "\n")
        for r in rows:
            f.write("\t".join(
                f"{r[h]:.6f}" if isinstance(r[h], float) else str(r[h])
                for h in headers
            ) + "\n")
    print(f"wrote {OUT}: {len(rows)} runs scored, {len(skipped)} skipped")
    if skipped:
        print("skipped (architecture mismatch / load error):")
        for tag, err in skipped[:5]:
            print(f"  {tag}: {err[:120]}")

    # summary table
    rows.sort(key=lambda r: -r["var_explained_pooled"])
    print("\n=== Top 12 by variance-explained (pooled), per input space ===")
    print(f"{'tag':30s}  {'scale':14s}  {'b':>3s} {'hidden':>8s}  "
          f"{'var_exp':>8s}  {'frac_un':>8s}  {'corr_F11':>9s}  "
          f"{'jac20':>6s}  {'val_raw':>9s}")
    for r in rows[:12]:
        print(f"{r['tag']:30s}  {r['input_scale']:14s}  {r['bottleneck']:>3d} "
              f"{r['hidden_dims']:>8s}  {r['var_explained_pooled']*100:>7.3f}%  "
              f"{r['frac_unexplained']*100:>7.3f}%  {r['corr_with_F11']:>9.3f}  "
              f"{r['top20_jaccard']:>6.3f}  {r['val_metric_raw']:>9.4f}")

    # Best per (input_scale)
    print("\n=== Best per input_scale by var_explained_pooled ===")
    for scale in ("row_std", "log_robust_z"):
        cands = [r for r in rows if r["input_scale"] == scale]
        if not cands:
            continue
        b = max(cands, key=lambda r: r["var_explained_pooled"])
        print(f"  {scale:14s}: {b['tag']:28s}  var_exp={b['var_explained_pooled']*100:.3f}%  "
              f"frac_un={b['frac_unexplained']*100:.3f}%  "
              f"corr_F11={b['corr_with_F11']:.3f}  jac20={b['top20_jaccard']:.3f}")

    # Differentiation-from-F11 leaderboard: among models in top half by
    # var_explained, lowest corr_with_F11 wins.
    median_ve = np.median([r["var_explained_pooled"] for r in rows])
    pool = [r for r in rows if r["var_explained_pooled"] >= median_ve]
    pool.sort(key=lambda r: abs(r["corr_with_F11"]))
    print(f"\n=== Best differentiation-from-F11 among top-half-by-var_explained "
          f"(n={len(pool)}) ===")
    for r in pool[:8]:
        print(f"  {r['tag']:30s}  scale={r['input_scale']:14s}  "
              f"var_exp={r['var_explained_pooled']*100:.3f}%  "
              f"corr_F11={r['corr_with_F11']:+.3f}  jac20={r['top20_jaccard']:.3f}")

if __name__ == "__main__":
    main()
