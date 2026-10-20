"""
Shape AE autoresearch training script. Single file, one model per market.

A multi-channel 1-D conv autoencoder over the 36-tenor forward curve, fused with an
MLP over the 14 scalar curve features. One latent code z rebuilds three targets:
  - the level-free curve shape (-> native price curve, re-anchored at the observed level)
  - the roll-aligned day change of the shape (dshape: where the curve moved today)
  - the scalar feature vector (Slope Δ, Curvature Δ, RV 21d, PCA factors, ...)
prepare.py turns the residuals into the decision and its explanation (tenors, feature).

Usage: uv run train.py
"""

import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare import (N_TENORS, N_FEATURES, TIME_BUDGET, list_markets, load_market,
                     evaluate, export_outputs, OUTPUT_DIR)

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

# Inputs
CURVE_CHANNELS = ("shape", "dshape")  # any of "shape", "dshape", "curve" (T(price) incl. level)
USE_FEATURES = True                   # feed the 14 scalar features into the encoder

# Model
HIDDEN = 32             # first conv width (second conv is 2x)
LATENT_DIM = 6          # bottleneck size L
FEAT_HIDDEN = 64        # feature encoder / decoder width

# Optimization
LR = 8e-4
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 128
GRAD_CLIP = 1.0
HUBER_DELTA = 1.0
DENOISE_STD = 0.03      # input noise (denoising AE), targets stay clean
DSHAPE_LOSS_WEIGHT = 1.0  # weight of day-change reconstruction vs shape reconstruction
FEAT_LOSS_WEIGHT = 1.0    # weight of feature reconstruction vs shape reconstruction
MAX_EPOCHS = 200
PATIENCE = 15           # early stop on the early-stop split
SEED = 1729

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ShapeAE(nn.Module):
    def __init__(self, in_channels, n_features, hidden=HIDDEN, latent=LATENT_DIM, feat_hidden=FEAT_HIDDEN,
                 use_features=USE_FEATURES):
        super().__init__()
        h2 = 2 * hidden
        self.use_features = use_features
        self.h2 = h2
        # curve encoder (stride 1, no pooling: length stays 36)
        self.enc_conv = nn.Sequential(
            nn.Conv1d(in_channels, hidden, 5, padding=2), nn.GELU(),
            nn.Conv1d(hidden, h2, 3, padding=1), nn.GELU(),
        )
        # feature encoder
        enc_in = h2 * N_TENORS
        if use_features:
            self.enc_feat = nn.Sequential(nn.Linear(n_features, feat_hidden), nn.GELU())
            enc_in += feat_hidden
        self.to_latent = nn.Linear(enc_in, latent)
        # curve decoder (mirror)
        self.from_latent = nn.Linear(latent, h2 * N_TENORS)
        self.dec_conv = nn.Sequential(
            nn.GELU(),
            nn.Conv1d(h2, hidden, 3, padding=1), nn.GELU(),
            nn.Conv1d(hidden, 2, 5, padding=2),  # channel 0: shape, channel 1: dshape
        )
        # feature decoder
        self.dec_feat = nn.Sequential(nn.Linear(latent, feat_hidden), nn.GELU(), nn.Linear(feat_hidden, n_features))

    def encode(self, curve, feat):
        h = self.enc_conv(curve).flatten(1)
        if self.use_features:
            h = torch.cat([h, self.enc_feat(feat)], dim=1)
        return self.to_latent(h)

    def decode(self, z):
        h = self.dec_conv(self.from_latent(z).view(-1, self.h2, N_TENORS))
        return h[:, 0], h[:, 1], self.dec_feat(z)

    def forward(self, curve, feat):
        z = self.encode(curve, feat)
        shape_hat, dshape_hat, feat_hat = self.decode(z)
        return shape_hat, dshape_hat, feat_hat, z

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def to_tensors(batch, device):
    curve = np.stack([batch[c] for c in CURVE_CHANNELS], axis=1)  # (N, C, 36)
    t = lambda x: torch.from_numpy(np.ascontiguousarray(x)).to(device)
    return t(curve), t(batch["feat"]), t(batch["shape"]), t(batch["dshape"])

def loss_fn(model, curve, feat, shape, dshape, noise=0.0):
    if noise > 0:
        curve = curve + noise * torch.randn_like(curve)
        feat = feat + noise * torch.randn_like(feat)
    shape_hat, dshape_hat, feat_hat, _ = model(curve, feat)
    ls = F.huber_loss(shape_hat, shape, delta=HUBER_DELTA)
    ld = F.huber_loss(dshape_hat, dshape, delta=HUBER_DELTA)
    lf = F.huber_loss(feat_hat, feat, delta=HUBER_DELTA)
    return ls + DSHAPE_LOSS_WEIGHT * ld + FEAT_LOSS_WEIGHT * lf

def make_reconstruct_fn(model, device):
    @torch.no_grad()
    def fn(batch):
        model.eval()
        curve, feat, _, _ = to_tensors(batch, device)
        return tuple(x.cpu().numpy() for x in model(curve, feat))  # shape, dshape, feat, z
    return fn

# ---------------------------------------------------------------------------
# Training (one model per market, equal share of the time budget)
# ---------------------------------------------------------------------------

def train_market(market, budget_s, device):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    data = load_market(market)
    fit = to_tensors(data.batch("fit"), device)
    es = to_tensors(data.batch("early_stop"), device)
    curve = fit[0]

    model = ShapeAE(len(CURVE_CHANNELS), N_FEATURES).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    n = curve.shape[0]
    best, best_state, bad, epoch, steps = math.inf, None, 0, 0, 0
    t0 = time.time()
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            loss = loss_fn(model, *(x[idx] for x in fit), noise=DENOISE_STD)
            if not torch.isfinite(loss):
                print("FAIL")
                exit(1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            steps += 1
        model.eval()
        with torch.no_grad():
            val = loss_fn(model, *es)
        if val.item() < best:
            best, bad = val.item(), 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if bad >= PATIENCE or time.time() - t0 >= budget_s:
            break
    model.load_state_dict(best_state)
    print(f"  {market:12s} fit={n:5d} epochs={epoch:3d} steps={steps:5d} best_es_loss={best:.5f} "
          f"({time.time() - t0:.1f}s)")
    return model, steps

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

t_start = time.time()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision("high")
markets = list_markets()
print(f"Device: {device} | markets: {markets} | time budget: {TIME_BUDGET}s")

per_market_budget = TIME_BUDGET / len(markets)
models, total_steps = {}, 0
t_train = time.time()
for m in markets:
    models[m], s = train_market(m, per_market_budget, device)
    total_steps += s
training_seconds = time.time() - t_train

fns = {m: make_reconstruct_fn(models[m], device) for m in markets}
val_score, results = evaluate(fns)
for m in markets:
    export_outputs(m, fns[m])

num_params = sum(p.numel() for p in models[markets[0]].parameters())
peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024 if device.type == "cuda" else 0.0
mean = lambda k: float(np.mean([r[k] for r in results.values()]))

print("---")
for m, r in results.items():
    e = r["events"]
    print(f"market {m:12s} score={r['score']:.4f} inj={r['inj_score']:.4f} recall={r['recall']:.3f} "
          f"tenor_acc={r['tenor_acc']:.3f} feature_acc={r['feature_acc']:.3f} alert_rate={r['alert_rate']:.3f} "
          f"alert_factor={r['gates']['alert_factor']:.2f} ev={r['recon_ev']:.3f} events={e['n_events']} "
          f"event_recall={e['recall']:.3f} regions={r['region_share']}")
    print(f"  by type: " + "  ".join(f"{k}={v['recall']:.2f}/{v['tenor_acc'] if v['tenor_acc'] is not None else 0:.2f}"
                                   for k, v in r["recall_by_type"].items()) + "  (recall/tenor_acc)")
print(f"val_score:        {val_score:.6f}")
print(f"mean_recall:      {mean('recall'):.6f}")
print(f"mean_tenor_acc:   {mean('tenor_acc'):.6f}")
print(f"mean_feature_acc: {mean('feature_acc'):.6f}")
print(f"mean_alert_rate:  {mean('alert_rate'):.6f}")
print(f"mean_recon_ev:    {mean('recon_ev'):.6f}")
print(f"mean_auprc:       {mean('auprc'):.6f}")
print(f"price_mape:       {mean('price_mape'):.6f}")
print(f"n_dev_events:     {sum(r['events']['n_events'] for r in results.values())}")
print(f"training_seconds: {training_seconds:.1f}")
print(f"total_seconds:    {time.time() - t_start:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"num_steps:        {total_steps}")
print(f"num_params_K:     {num_params / 1e3:.1f}")
print(f"latent_dim:       {LATENT_DIM}")
print(f"outputs:          {OUTPUT_DIR}")
