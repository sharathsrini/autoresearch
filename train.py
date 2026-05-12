"""
Curve-shape autoencoder for FERA forward-curve anomaly detection.
This is the ONLY file the autoresearch agent edits.

Run: python train.py --tag baseline
"""

import os
import json
import time
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import prepare


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class CurveShapeAE(nn.Module):
    def __init__(self, n_tenors=36, hidden_dims=(24, 12), bottleneck=4):
        super().__init__()
        dims = [n_tenors, *hidden_dims, bottleneck]
        enc = []
        for i in range(len(dims) - 1):
            enc.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                enc.append(nn.Tanh())
        self.encoder = nn.Sequential(*enc)
        rev = list(reversed(dims))
        dec = []
        for i in range(len(rev) - 1):
            dec.append(nn.Linear(rev[i], rev[i + 1]))
            if i < len(rev) - 2:
                dec.append(nn.Tanh())
        self.decoder = nn.Sequential(*dec)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_run(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    panels = prepare.load_panels()
    df = prepare.build_curve_dataset(args.market, panels=panels)
    train_df, val_df, _test_df = prepare.walk_forward_split(df)
    train_df = prepare.filter_training_rows(train_df)

    if len(train_df) == 0 or len(val_df) == 0:
        raise RuntimeError(f"empty split: train={len(train_df)} val={len(val_df)}")

    train_X, _, _ = prepare.row_standardize(train_df[prepare.TENORS].values)
    val_X,   _, _ = prepare.row_standardize(val_df[prepare.TENORS].values)
    F11_Z_val = val_df["F11_Z"].values

    train_loader, _val_loader = prepare.make_loaders(train_X, val_X, args.batch_size)

    if args.hidden_dims:
        hidden_dims = tuple(int(x) for x in args.hidden_dims.split(",") if x.strip())
    else:
        hidden_dims = (args.hidden1, args.hidden2)
    model = CurveShapeAE(
        n_tenors=prepare.N_TENORS,
        hidden_dims=hidden_dims,
        bottleneck=args.bottleneck,
    )
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Per-element MSE summed over 36 tenors, mean over batch — per spec.
    def loss_fn(x_hat, x):
        return ((x_hat - x) ** 2).sum(dim=1).mean()

    t0 = time.time()
    budget = args.time_budget_s
    train_mse_running = float("nan")
    for epoch in range(args.epochs):
        model.train()
        epoch_loss_sum, epoch_n = 0.0, 0
        for (xb,) in train_loader:
            x_hat, _z = model(xb)
            loss = loss_fn(x_hat, xb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss_sum += float(loss.item()) * xb.size(0)
            epoch_n += xb.size(0)
        train_mse_running = epoch_loss_sum / max(epoch_n, 1)
        if (epoch + 1) % max(1, args.log_every) == 0:
            print(f"epoch {epoch+1:3d}/{args.epochs}  train_mse={train_mse_running:.6f}  "
                  f"elapsed={time.time()-t0:.1f}s")
        if time.time() - t0 > budget:
            print(f"time budget hit at epoch {epoch+1}")
            break

    metrics = prepare.compute_metric(model, val_X, F11_Z_val, train_mse_running)
    return model, metrics


def _read_best_val_metric(results_tsv_path):
    if not os.path.exists(results_tsv_path):
        return None
    best = None
    with open(results_tsv_path, "r", encoding="utf-8") as f:
        lines = f.read().strip().splitlines()
    if len(lines) <= 1:
        return None
    for line in lines[1:]:
        parts = line.split("\t")
        # timestamp tag val_metric train_mse latent_std_min corr_with_F11 kept
        if len(parts) < 7:
            continue
        kept = parts[6].strip() == "1"
        if not kept:
            continue
        try:
            v = float(parts[2])
        except ValueError:
            continue
        if best is None or v < best:
            best = v
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--market",       type=str,   default="TTF")
    p.add_argument("--bottleneck",   type=int,   default=4)
    p.add_argument("--hidden1",      type=int,   default=24)
    p.add_argument("--hidden2",      type=int,   default=12)
    p.add_argument("--hidden_dims",  type=str,   default="24",
                   help="comma-separated hidden layer sizes (e.g. '24' or '32,20,12')")
    p.add_argument("--epochs",       type=int,   default=200)
    p.add_argument("--batch_size",   type=int,   default=128)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--seed",         type=int,   default=0)
    p.add_argument("--log_every",    type=int,   default=20)
    p.add_argument("--time_budget_s",type=int,   default=180)  # 3 minutes / spec
    p.add_argument("--tag",          type=str,   required=True)
    args = p.parse_args()

    repo_root = Path(__file__).resolve().parent
    results_tsv = repo_root / "results.tsv"

    print(f"[run] tag={args.tag} market={args.market} bottleneck={args.bottleneck}")
    model, metrics = train_one_run(args)

    best = _read_best_val_metric(str(results_tsv))
    kept, reasons = prepare.evaluate_guards(metrics, best)

    print(f"[metric] val_metric={metrics['val_metric']:.6f}  "
          f"train_mse={metrics['train_mse']:.6f}  "
          f"latent_std_min={metrics['latent_std_min']:.6f}  "
          f"corr_with_F11={metrics['corr_with_F11']:.6f}")
    print(f"[guards] kept={kept} best_prev={best} reasons={reasons}")

    run_dir = repo_root / "runs" / args.tag
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), run_dir / "model.pt")
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump({**vars(args), "metrics": metrics, "kept": kept, "reasons": reasons}, f, indent=2)

    prepare.append_result(str(results_tsv), args.tag, metrics, kept)
    print(f"[done] appended to {results_tsv}")


if __name__ == "__main__":
    main()
