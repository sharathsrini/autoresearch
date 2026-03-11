"""
Time series anomaly detection autoresearch script. Single-GPU, single-file.
Supports both ML models (sklearn-based) and DL models (PyTorch-based).
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
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from prepare import (WINDOW_SIZE, N_FEATURES, TIME_BUDGET,
                     make_dataloader, load_data, create_windows, evaluate_f1)

# ---------------------------------------------------------------------------
# Choose which model to use: "lstm_ae" or "isolation_forest" or "lof"
# ---------------------------------------------------------------------------
MODEL_TYPE = "lstm_ae"

# ---------------------------------------------------------------------------
# Statistical / ML Models (sklearn-based, no GPU needed)
# ---------------------------------------------------------------------------

class SklearnAnomalyDetector:
    """Wrapper that makes any sklearn outlier detector compatible with evaluate_f1.

    The sklearn model operates on flattened windows: (batch, window_size * n_features).
    It is trained on normal data only, then scores new windows.
    """

    def __init__(self, sklearn_model):
        self.sklearn_model = sklearn_model
        self._fitted = False

    def fit(self, windows_np):
        """Fit on normal training windows. windows_np: (n_windows, window_size, n_features)."""
        flat = windows_np.reshape(len(windows_np), -1)
        print(f"Fitting {self.sklearn_model.__class__.__name__} on {len(flat)} windows...")
        self.sklearn_model.fit(flat)
        self._fitted = True

    def eval(self):
        """No-op for compatibility with the evaluation harness."""
        pass

    def compute_anomaly_scores(self, x):
        """Score windows. x: torch tensor (batch, window_size, n_features).
        Returns: torch tensor (batch,) — higher = more anomalous."""
        x_np = x.cpu().numpy()
        flat = x_np.reshape(len(x_np), -1)
        # sklearn: score_samples returns negative anomaly score (lower = more anomalous)
        # We negate so higher = more anomalous (matching evaluate_f1's expectation)
        scores = -self.sklearn_model.score_samples(flat)
        return torch.from_numpy(scores.astype(np.float32))


class IsolationForestDetector(SklearnAnomalyDetector):
    def __init__(self, n_estimators=200, max_samples="auto", contamination="auto",
                 max_features=1.0, random_state=42):
        model = IsolationForest(
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination=contamination,
            max_features=max_features,
            random_state=random_state,
            n_jobs=-1,
        )
        super().__init__(model)


class LOFDetector:
    """Local Outlier Factor detector.

    LOF requires a different approach since it uses novelty=True for scoring
    new data (as opposed to inductive mode).
    """

    def __init__(self, n_neighbors=20, contamination="auto", metric="euclidean"):
        self.lof = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination,
            metric=metric,
            novelty=True,
            n_jobs=-1,
        )
        self._fitted = False

    def fit(self, windows_np):
        flat = windows_np.reshape(len(windows_np), -1)
        print(f"Fitting LOF on {len(flat)} windows...")
        self.lof.fit(flat)
        self._fitted = True

    def eval(self):
        pass

    def compute_anomaly_scores(self, x):
        x_np = x.cpu().numpy()
        flat = x_np.reshape(len(x_np), -1)
        scores = -self.lof.score_samples(flat)
        return torch.from_numpy(scores.astype(np.float32))


# ---------------------------------------------------------------------------
# Deep Learning Models (PyTorch-based, GPU accelerated)
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
        _, (h_n, _) = self.lstm(x)
        h_last = h_n[-1]
        z = self.fc_latent(h_last)
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
        h = self.fc_expand(z)
        h_repeated = h.unsqueeze(1).repeat(1, self.window_size, 1)
        out, _ = self.lstm(h_repeated)
        recon = self.fc_out(out)
        return recon


class LSTMAutoencoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon

    def compute_loss(self, x, reduction='mean'):
        recon = self.forward(x)
        if reduction == 'none':
            return ((x - recon) ** 2).mean(dim=(1, 2))
        return F.mse_loss(recon, x, reduction=reduction)

    @torch.no_grad()
    def compute_anomaly_scores(self, x):
        """Anomaly score = reconstruction error. Higher = more anomalous."""
        recon = self.forward(x)
        scores = ((x - recon) ** 2).mean(dim=(1, 2))
        return scores


# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

# --- Isolation Forest hyperparameters ---
IF_N_ESTIMATORS = 200         # number of trees
IF_MAX_SAMPLES = "auto"       # samples per tree ("auto" = min(256, n_samples))
IF_MAX_FEATURES = 1.0         # features per tree (1.0 = all)
IF_CONTAMINATION = "auto"     # expected anomaly fraction

# --- LOF hyperparameters ---
LOF_N_NEIGHBORS = 20          # number of neighbors
LOF_CONTAMINATION = "auto"    # expected anomaly fraction
LOF_METRIC = "euclidean"      # distance metric

# --- LSTM-AE hyperparameters ---
HIDDEN_DIM = 128              # LSTM hidden dimension
LATENT_DIM = 32               # bottleneck dimension
N_LAYERS = 2                  # number of LSTM layers
DROPOUT = 0.1                 # dropout rate

# --- Optimization (DL models only) ---
LEARNING_RATE = 1e-3          # Adam learning rate
WEIGHT_DECAY = 1e-5           # L2 regularization
BATCH_SIZE = 64               # training batch size
WARMUP_RATIO = 0.05           # fraction of time budget for LR warmup
WARMDOWN_RATIO = 0.3          # fraction of time budget for LR cooldown
FINAL_LR_FRAC = 0.01          # final LR as fraction of initial

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

t_start = time.time()
torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

print(f"Model type: {MODEL_TYPE}")
print(f"Time budget: {TIME_BUDGET}s")
print(f"Device: {device}")

# ---------------------------------------------------------------------------
# Model-specific training
# ---------------------------------------------------------------------------

if MODEL_TYPE in ("isolation_forest", "lof"):
    # -----------------------------------------------------------------------
    # Statistical / ML model path (no GPU training loop needed)
    # -----------------------------------------------------------------------

    # Load and prepare training windows
    train_data_raw, train_labels_raw = load_data("train")
    train_data_raw2, _ = load_data("train")  # for normalization stats
    mean = train_data_raw2.mean(axis=0)
    std = train_data_raw2.std(axis=0)
    std[std < 1e-8] = 1.0
    train_norm = (train_data_raw - mean) / std

    train_windows, train_wlabels, _ = create_windows(train_norm, train_labels_raw,
                                                      WINDOW_SIZE, stride=1)
    # Only normal windows for training
    normal_mask = train_wlabels == 0
    train_windows = train_windows[normal_mask]

    # Subsample if too many windows (sklearn doesn't scale to millions)
    max_train_windows = 50000
    if len(train_windows) > max_train_windows:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(train_windows), max_train_windows, replace=False)
        train_windows = train_windows[idx]
    print(f"Training windows: {len(train_windows)}")

    t_start_training = time.time()

    if MODEL_TYPE == "isolation_forest":
        model = IsolationForestDetector(
            n_estimators=IF_N_ESTIMATORS,
            max_samples=IF_MAX_SAMPLES,
            max_features=IF_MAX_FEATURES,
            contamination=IF_CONTAMINATION,
        )
    else:  # lof
        model = LOFDetector(
            n_neighbors=LOF_N_NEIGHBORS,
            contamination=LOF_CONTAMINATION,
            metric=LOF_METRIC,
        )

    model.fit(train_windows)
    total_training_time = time.time() - t_start_training
    num_params = 0
    step = 1
    debiased_smooth_loss = 0.0
    print(f"Fitting completed in {total_training_time:.1f}s")

else:
    # -----------------------------------------------------------------------
    # Deep Learning model path (GPU training loop)
    # -----------------------------------------------------------------------

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

    # LR schedule
    def get_lr_multiplier(progress):
        if progress < WARMUP_RATIO:
            return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
        elif progress < 1.0 - WARMDOWN_RATIO:
            return 1.0
        else:
            cooldown = (1.0 - progress) / WARMDOWN_RATIO
            return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC

    # Training loop
    t_start_training = time.time()
    smooth_train_loss = 0
    total_training_time = 0
    step = 0
    warmup_steps = 5

    model.train()
    while True:
        t0 = time.time()

        windows, labels, epoch = next(train_loader)
        loss = model.compute_loss(windows)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss_f = loss.item()

        if math.isnan(train_loss_f) or train_loss_f > 1e6:
            print("FAIL")
            exit(1)

        t1 = time.time()
        dt = t1 - t0

        if step > warmup_steps:
            total_training_time += dt

        progress = min(total_training_time / TIME_BUDGET, 1.0)
        lrm = get_lr_multiplier(progress)
        for pg in optimizer.param_groups:
            pg['lr'] = LEARNING_RATE * lrm

        ema_beta = 0.95
        smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
        debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1))
        pct_done = 100 * progress
        remaining = max(0, TIME_BUDGET - total_training_time)

        if step % 50 == 0:
            print(f"\rstep {step:05d} ({pct_done:.1f}%) | loss: {debiased_smooth_loss:.6f} | lr: {LEARNING_RATE * lrm:.2e} | dt: {dt*1000:.0f}ms | epoch: {epoch} | remaining: {remaining:.0f}s    ", end="", flush=True)

        if step == 0:
            gc.collect()

        step += 1

        if step > warmup_steps and total_training_time >= TIME_BUDGET:
            break

    print()

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
print(f"model_type:       {MODEL_TYPE}")
