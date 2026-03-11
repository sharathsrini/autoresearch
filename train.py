"""
Time series anomaly detection autoresearch script. Single-GPU, single-file.
Usage: uv run train.py
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import gc
import math
import time
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare import WINDOW_SIZE, N_FEATURES, TIME_BUDGET, make_dataloader, evaluate_f1

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    window_size: int = WINDOW_SIZE
    n_features: int = N_FEATURES
    hidden_dim: int = 128         # LSTM hidden dimension
    latent_dim: int = 32          # bottleneck dimension
    n_layers: int = 2             # number of LSTM layers
    dropout: float = 0.1         # dropout rate

class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=config.n_features,
            hidden_size=config.hidden_dim,
            num_layers=config.n_layers,
            batch_first=True,
            dropout=config.dropout if config.n_layers > 1 else 0.0,
        )
        self.fc_latent = nn.Linear(config.hidden_dim, config.latent_dim)

    def forward(self, x):
        # x: (batch, window_size, n_features)
        _, (h_n, _) = self.lstm(x)
        # h_n: (n_layers, batch, hidden_dim) — take last layer
        h_last = h_n[-1]  # (batch, hidden_dim)
        z = self.fc_latent(h_last)  # (batch, latent_dim)
        return z


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.window_size = config.window_size
        self.hidden_dim = config.hidden_dim
        self.fc_expand = nn.Linear(config.latent_dim, config.hidden_dim)
        self.lstm = nn.LSTM(
            input_size=config.hidden_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.n_layers,
            batch_first=True,
            dropout=config.dropout if config.n_layers > 1 else 0.0,
        )
        self.fc_out = nn.Linear(config.hidden_dim, config.n_features)

    def forward(self, z):
        # z: (batch, latent_dim)
        h = self.fc_expand(z)  # (batch, hidden_dim)
        # Repeat across time steps
        h_repeated = h.unsqueeze(1).repeat(1, self.window_size, 1)  # (batch, window_size, hidden_dim)
        out, _ = self.lstm(h_repeated)  # (batch, window_size, hidden_dim)
        recon = self.fc_out(out)  # (batch, window_size, n_features)
        return recon


class LSTMAutoencoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

    def forward(self, x):
        """Forward pass. Returns reconstruction."""
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon

    def compute_loss(self, x, reduction='mean'):
        """Compute reconstruction loss (MSE)."""
        recon = self.forward(x)
        if reduction == 'none':
            # Per-sample loss: mean over (window_size, n_features)
            return ((x - recon) ** 2).mean(dim=(1, 2))
        return F.mse_loss(recon, x, reduction=reduction)

    @torch.no_grad()
    def compute_anomaly_scores(self, x):
        """Compute anomaly scores for a batch of windows.

        Args:
            x: (batch, window_size, n_features)

        Returns:
            scores: (batch,) — higher = more anomalous
        """
        recon = self.forward(x)
        # Per-window reconstruction error (mean over time and features)
        scores = ((x - recon) ** 2).mean(dim=(1, 2))
        return scores

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

# Model architecture
HIDDEN_DIM = 128          # LSTM hidden dimension
LATENT_DIM = 32           # bottleneck dimension
N_LAYERS = 2              # number of LSTM layers
DROPOUT = 0.1             # dropout rate

# Optimization
LEARNING_RATE = 1e-3      # Adam learning rate
WEIGHT_DECAY = 1e-5       # L2 regularization
BATCH_SIZE = 64           # training batch size
WARMUP_RATIO = 0.05       # fraction of time budget for LR warmup
WARMDOWN_RATIO = 0.3      # fraction of time budget for LR cooldown
FINAL_LR_FRAC = 0.01      # final LR as fraction of initial

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

t_start = time.time()
torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = ModelConfig(
    window_size=WINDOW_SIZE,
    n_features=N_FEATURES,
    hidden_dim=HIDDEN_DIM,
    latent_dim=LATENT_DIM,
    n_layers=N_LAYERS,
    dropout=DROPOUT,
)
print(f"Model config: {asdict(config)}")

model = LSTMAutoencoder(config).to(device)

num_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {num_params:,}")

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

train_loader = make_dataloader("train", BATCH_SIZE, WINDOW_SIZE, stride=1,
                                shuffle=True, device=device)

print(f"Time budget: {TIME_BUDGET}s")
print(f"Device: {device}")

# LR schedule
def get_lr_multiplier(progress):
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    elif progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

t_start_training = time.time()
smooth_train_loss = 0
total_training_time = 0
step = 0
warmup_steps = 5  # skip first N steps for timing (compilation, etc.)

model.train()
while True:
    t0 = time.time()

    windows, labels, epoch = next(train_loader)

    # Forward + backward
    loss = model.compute_loss(windows)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    train_loss_f = loss.item()

    # Fast fail
    if math.isnan(train_loss_f) or train_loss_f > 1e6:
        print("FAIL")
        exit(1)

    t1 = time.time()
    dt = t1 - t0

    if step > warmup_steps:
        total_training_time += dt

    # LR schedule
    progress = min(total_training_time / TIME_BUDGET, 1.0)
    lrm = get_lr_multiplier(progress)
    for pg in optimizer.param_groups:
        pg['lr'] = LEARNING_RATE * lrm

    # Logging
    ema_beta = 0.95
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1))
    pct_done = 100 * progress
    remaining = max(0, TIME_BUDGET - total_training_time)

    if step % 50 == 0:
        print(f"\rstep {step:05d} ({pct_done:.1f}%) | loss: {debiased_smooth_loss:.6f} | lr: {LEARNING_RATE * lrm:.2e} | dt: {dt*1000:.0f}ms | epoch: {epoch} | remaining: {remaining:.0f}s    ", end="", flush=True)

    # GC management
    if step == 0:
        gc.collect()

    step += 1

    if step > warmup_steps and total_training_time >= TIME_BUDGET:
        break

print()  # newline after \r training log

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

model.eval()

val_f1, val_threshold, val_precision, val_recall = evaluate_f1(model, device=str(device), split="val")
test_f1, test_threshold, test_precision, test_recall = evaluate_f1(model, device=str(device), split="test")

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------

t_end = time.time()
peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0

print("---")
print(f"val_f1:           {val_f1:.6f}")
print(f"val_precision:    {val_precision:.6f}")
print(f"val_recall:       {val_recall:.6f}")
print(f"val_threshold:    {val_threshold:.6f}")
print(f"test_f1:          {test_f1:.6f}")
print(f"test_precision:   {test_precision:.6f}")
print(f"test_recall:      {test_recall:.6f}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"num_steps:        {step}")
print(f"num_params:       {num_params:,}")
print(f"final_train_loss: {debiased_smooth_loss:.6f}")
