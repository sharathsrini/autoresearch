"""
One-time data preparation for time series anomaly detection autoresearch.
Generates synthetic multivariate time series with injected anomalies,
or loads user-provided CSV data.

Usage:
    python prepare.py                    # generate synthetic data
    python prepare.py --data-dir ./mydata  # use custom CSV files

Data is stored in ~/.cache/autoresearch_ad/.
"""

import os
import sys
import math
import argparse
import json

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

WINDOW_SIZE = 100        # sliding window length
STRIDE = 1               # stride for sliding windows
N_FEATURES = 25          # number of features in multivariate time series
TIME_BUDGET = 300        # training time budget in seconds (5 minutes)
ANOMALY_RATIO = 0.05     # fraction of points that are anomalous

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch_ad")
DATA_DIR = os.path.join(CACHE_DIR, "data")

# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

def generate_normal_series(length, n_features, seed=42):
    """Generate normal multivariate time series with realistic patterns."""
    rng = np.random.RandomState(seed)

    t = np.arange(length, dtype=np.float64)
    data = np.zeros((length, n_features), dtype=np.float64)

    for f in range(n_features):
        # Base: mix of sinusoidal components with different frequencies
        freq1 = rng.uniform(0.001, 0.01)
        freq2 = rng.uniform(0.01, 0.05)
        freq3 = rng.uniform(0.05, 0.15)
        amp1 = rng.uniform(0.5, 2.0)
        amp2 = rng.uniform(0.2, 1.0)
        amp3 = rng.uniform(0.1, 0.5)
        phase1 = rng.uniform(0, 2 * np.pi)
        phase2 = rng.uniform(0, 2 * np.pi)
        phase3 = rng.uniform(0, 2 * np.pi)

        signal = (amp1 * np.sin(2 * np.pi * freq1 * t + phase1) +
                  amp2 * np.sin(2 * np.pi * freq2 * t + phase2) +
                  amp3 * np.sin(2 * np.pi * freq3 * t + phase3))

        # Add slow trend
        trend_slope = rng.uniform(-0.0005, 0.0005)
        signal += trend_slope * t

        # Add Gaussian noise
        noise_std = rng.uniform(0.05, 0.3)
        signal += rng.normal(0, noise_std, length)

        # Inter-feature correlations: some features are correlated
        if f > 0 and rng.random() < 0.4:
            src = rng.randint(0, f)
            corr_weight = rng.uniform(0.2, 0.6)
            signal += corr_weight * data[:, src]

        data[:, f] = signal

    return data


def inject_anomalies(data, anomaly_ratio, seed=123):
    """Inject various types of anomalies into time series data.

    Returns:
        data_with_anomalies: modified data
        labels: binary array (1 = anomaly)
    """
    rng = np.random.RandomState(seed)
    length, n_features = data.shape
    labels = np.zeros(length, dtype=np.int32)
    data_anom = data.copy()

    n_anomalous_points = int(length * anomaly_ratio)

    # We inject anomalies as contiguous segments of various types
    anomaly_types = ['spike', 'level_shift', 'variance_change', 'trend_change', 'contextual']
    points_placed = 0

    while points_placed < n_anomalous_points:
        atype = rng.choice(anomaly_types)
        seg_len = rng.randint(10, min(80, n_anomalous_points - points_placed + 1))
        if seg_len <= 0:
            break

        # Find a non-anomalous region to place the anomaly
        max_tries = 100
        for _ in range(max_tries):
            start = rng.randint(WINDOW_SIZE, length - seg_len)
            if labels[start:start + seg_len].sum() == 0:
                break
        else:
            break

        # Pick subset of features to affect
        n_affected = rng.randint(1, max(2, n_features // 3))
        affected_features = rng.choice(n_features, n_affected, replace=False)

        if atype == 'spike':
            for f in affected_features:
                std = np.std(data[:, f])
                spikes = rng.choice(seg_len, min(seg_len // 2, 5), replace=False)
                for s in spikes:
                    data_anom[start + s, f] += rng.choice([-1, 1]) * rng.uniform(4, 8) * std

        elif atype == 'level_shift':
            for f in affected_features:
                std = np.std(data[:, f])
                shift = rng.choice([-1, 1]) * rng.uniform(3, 6) * std
                data_anom[start:start + seg_len, f] += shift

        elif atype == 'variance_change':
            for f in affected_features:
                std = np.std(data[:, f])
                noise = rng.normal(0, std * rng.uniform(3, 6), seg_len)
                data_anom[start:start + seg_len, f] += noise

        elif atype == 'trend_change':
            for f in affected_features:
                std = np.std(data[:, f])
                slope = rng.choice([-1, 1]) * rng.uniform(0.05, 0.2) * std
                trend = slope * np.arange(seg_len)
                data_anom[start:start + seg_len, f] += trend

        elif atype == 'contextual':
            for f in affected_features:
                std = np.std(data[:, f])
                data_anom[start:start + seg_len, f] += rng.normal(0, std * 2, seg_len)

        labels[start:start + seg_len] = 1
        points_placed += seg_len

    return data_anom, labels


def generate_dataset(train_length=50000, test_length=20000, n_features=N_FEATURES):
    """Generate full train/test dataset with anomalies only in test."""
    # Training data: normal only (for reconstruction-based methods)
    train_data = generate_normal_series(train_length, n_features, seed=42)
    train_labels = np.zeros(train_length, dtype=np.int32)

    # Test data: normal base + injected anomalies
    test_data_clean = generate_normal_series(test_length, n_features, seed=99)
    test_data, test_labels = inject_anomalies(test_data_clean, ANOMALY_RATIO, seed=123)

    # Validation data: separate normal + anomalies for metric evaluation
    val_data_clean = generate_normal_series(test_length, n_features, seed=77)
    val_data, val_labels = inject_anomalies(val_data_clean, ANOMALY_RATIO, seed=456)

    return {
        'train_data': train_data,
        'train_labels': train_labels,
        'test_data': test_data,
        'test_labels': test_labels,
        'val_data': val_data,
        'val_labels': val_labels,
    }


def save_dataset(dataset, data_dir=DATA_DIR):
    """Save dataset as .npy files."""
    os.makedirs(data_dir, exist_ok=True)
    for key, arr in dataset.items():
        np.save(os.path.join(data_dir, f"{key}.npy"), arr)

    # Save metadata
    meta = {
        'n_features': dataset['train_data'].shape[1],
        'train_length': len(dataset['train_data']),
        'test_length': len(dataset['test_data']),
        'val_length': len(dataset['val_data']),
        'anomaly_ratio_test': float(dataset['test_labels'].mean()),
        'anomaly_ratio_val': float(dataset['val_labels'].mean()),
        'window_size': WINDOW_SIZE,
    }
    with open(os.path.join(data_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Dataset saved to {data_dir}")
    print(f"  Train: {meta['train_length']} points, {meta['n_features']} features")
    print(f"  Test:  {meta['test_length']} points, anomaly ratio: {meta['anomaly_ratio_test']:.3f}")
    print(f"  Val:   {meta['val_length']} points, anomaly ratio: {meta['anomaly_ratio_val']:.3f}")


# ---------------------------------------------------------------------------
# Runtime utilities (imported by train.py)
# ---------------------------------------------------------------------------

def load_data(split, data_dir=DATA_DIR):
    """Load data and labels for a split ('train', 'test', 'val')."""
    data = np.load(os.path.join(data_dir, f"{split}_data.npy"))
    labels = np.load(os.path.join(data_dir, f"{split}_labels.npy"))
    return data, labels


def get_metadata(data_dir=DATA_DIR):
    """Load dataset metadata."""
    with open(os.path.join(data_dir, "metadata.json"), "r") as f:
        return json.load(f)


def normalize_data(train_data, *other_data):
    """Z-score normalization fitted on training data.

    Returns:
        normalized_train, *normalized_others, mean, std
    """
    mean = train_data.mean(axis=0)
    std = train_data.std(axis=0)
    std[std < 1e-8] = 1.0  # avoid division by zero

    result = [(train_data - mean) / std]
    for d in other_data:
        result.append((d - mean) / std)
    result.extend([mean, std])
    return tuple(result)


def create_windows(data, labels, window_size=WINDOW_SIZE, stride=STRIDE):
    """Create sliding windows from time series data.

    Args:
        data: (length, n_features) array
        labels: (length,) array of binary labels

    Returns:
        windows: (n_windows, window_size, n_features) array
        window_labels: (n_windows,) array — 1 if ANY point in window is anomalous
        point_labels: (n_windows, window_size) array — per-point labels within each window
    """
    length = len(data)
    n_windows = (length - window_size) // stride + 1

    windows = np.zeros((n_windows, window_size, data.shape[1]), dtype=np.float32)
    window_labels = np.zeros(n_windows, dtype=np.int32)
    point_labels = np.zeros((n_windows, window_size), dtype=np.int32)

    for i in range(n_windows):
        start = i * stride
        end = start + window_size
        windows[i] = data[start:end]
        point_labels[i] = labels[start:end]
        window_labels[i] = 1 if labels[start:end].any() else 0

    return windows, window_labels, point_labels


def make_dataloader(split, batch_size, window_size=WINDOW_SIZE, stride=STRIDE,
                    shuffle=True, device="cuda"):
    """Create an infinite dataloader yielding (windows, window_labels) batches.

    For training split: only yields normal windows (label=0).
    For val/test split: yields all windows.
    """
    data, labels = load_data(split)

    # Normalize using training statistics
    train_data, _ = load_data("train")
    mean = train_data.mean(axis=0)
    std = train_data.std(axis=0)
    std[std < 1e-8] = 1.0
    data_norm = (data - mean) / std

    windows, window_labels, point_labels = create_windows(data_norm, labels, window_size, stride)

    if split == "train":
        # Only normal windows for training reconstruction
        normal_mask = window_labels == 0
        windows = windows[normal_mask]
        window_labels = window_labels[normal_mask]
        point_labels = point_labels[normal_mask]

    windows_t = torch.from_numpy(windows).float()
    labels_t = torch.from_numpy(window_labels).long()

    n = len(windows_t)
    indices = np.arange(n)

    epoch = 0
    while True:
        if shuffle:
            np.random.shuffle(indices)
        epoch += 1
        for start in range(0, n - batch_size + 1, batch_size):
            idx = indices[start:start + batch_size]
            batch_w = windows_t[idx].to(device, non_blocking=True)
            batch_l = labels_t[idx].to(device, non_blocking=True)
            yield batch_w, batch_l, epoch


# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE -- this is the fixed metric)
# ---------------------------------------------------------------------------

def _point_adjust_labels(pred, true):
    """Point-adjust: if any point in a contiguous anomaly segment is detected,
    mark the entire segment as detected. This is standard in the literature
    (Xu et al., 2018; Su et al., 2019)."""
    adjusted_pred = pred.copy()
    anomaly_segments = []
    in_segment = False
    start = 0

    for i in range(len(true)):
        if true[i] == 1 and not in_segment:
            in_segment = True
            start = i
        elif true[i] == 0 and in_segment:
            in_segment = False
            anomaly_segments.append((start, i))
    if in_segment:
        anomaly_segments.append((start, len(true)))

    for seg_start, seg_end in anomaly_segments:
        if adjusted_pred[seg_start:seg_end].any():
            adjusted_pred[seg_start:seg_end] = 1

    return adjusted_pred


@torch.no_grad()
def evaluate_f1(model, device="cuda", split="val"):
    """Evaluate anomaly detection using best F1 score (point-adjusted).

    The model must implement:
        anomaly_scores = model.compute_anomaly_scores(windows)
    where windows is (batch, window_size, n_features) and anomaly_scores is (batch,).

    Higher anomaly_score = more anomalous.
    We search over thresholds to find the best F1.

    Returns:
        best_f1: float (higher is better, this is the primary metric)
        best_threshold: float
        precision: float at best threshold
        recall: float at best threshold
    """
    data, labels = load_data(split)
    train_data, _ = load_data("train")
    mean = train_data.mean(axis=0)
    std = train_data.std(axis=0)
    std[std < 1e-8] = 1.0
    data_norm = (data - mean) / std

    windows, window_labels, point_labels = create_windows(
        data_norm, labels, WINDOW_SIZE, stride=1
    )

    windows_t = torch.from_numpy(windows).float().to(device)

    # Compute anomaly scores in batches
    batch_size = 256
    all_scores = []
    for i in range(0, len(windows_t), batch_size):
        batch = windows_t[i:i + batch_size]
        scores = model.compute_anomaly_scores(batch)
        all_scores.append(scores.cpu())
    all_scores = torch.cat(all_scores).numpy()

    # Map window-level scores back to point-level scores (max over overlapping windows)
    n_points = len(labels)
    point_scores = np.full(n_points, -np.inf)
    for i in range(len(all_scores)):
        start = i
        end = start + WINDOW_SIZE
        point_scores[start:end] = np.maximum(point_scores[start:end], all_scores[i])

    # Replace -inf with min score for points not covered by any window
    valid_mask = point_scores > -np.inf
    if valid_mask.any():
        min_score = point_scores[valid_mask].min()
        point_scores[~valid_mask] = min_score

    # Search thresholds for best F1 (point-adjusted)
    # Use percentiles of the score distribution as candidate thresholds
    percentiles = np.arange(90, 100, 0.5)
    thresholds = np.percentile(point_scores, percentiles)
    # Also add some evenly spaced thresholds
    extra = np.linspace(np.percentile(point_scores, 85), point_scores.max(), 50)
    thresholds = np.unique(np.concatenate([thresholds, extra]))

    best_f1 = 0.0
    best_threshold = thresholds[0]
    best_prec = 0.0
    best_rec = 0.0

    for thresh in thresholds:
        pred = (point_scores >= thresh).astype(np.int32)
        adj_pred = _point_adjust_labels(pred, labels)

        if adj_pred.sum() == 0:
            continue

        f1 = f1_score(labels, adj_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
            best_prec = precision_score(labels, adj_pred, zero_division=0)
            best_rec = recall_score(labels, adj_pred, zero_division=0)

    return best_f1, best_threshold, best_prec, best_rec


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data for anomaly detection autoresearch")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Path to custom CSV data directory. If not provided, generates synthetic data.")
    parser.add_argument("--train-length", type=int, default=50000,
                        help="Length of training time series (synthetic mode)")
    parser.add_argument("--test-length", type=int, default=20000,
                        help="Length of test/val time series (synthetic mode)")
    parser.add_argument("--n-features", type=int, default=N_FEATURES,
                        help="Number of features (synthetic mode)")
    args = parser.parse_args()

    print(f"Cache directory: {CACHE_DIR}")
    print()

    if args.data_dir:
        # Load custom data
        print(f"Loading custom data from {args.data_dir}...")
        import pandas as pd
        train_df = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
        test_df = pd.read_csv(os.path.join(args.data_dir, "test.csv"))
        test_labels_df = pd.read_csv(os.path.join(args.data_dir, "test_labels.csv"))

        # Expect: train.csv has feature columns only (all normal)
        # test.csv has feature columns, test_labels.csv has a single 'label' column
        train_data = train_df.values.astype(np.float64)
        test_data = test_df.values.astype(np.float64)
        test_labels = test_labels_df.values.flatten().astype(np.int32)

        # Use last 40% of test as validation
        split_idx = int(len(test_data) * 0.6)
        val_data = test_data[split_idx:]
        val_labels = test_labels[split_idx:]
        test_data = test_data[:split_idx]
        test_labels = test_labels[:split_idx]

        dataset = {
            'train_data': train_data,
            'train_labels': np.zeros(len(train_data), dtype=np.int32),
            'test_data': test_data,
            'test_labels': test_labels,
            'val_data': val_data,
            'val_labels': val_labels,
        }
    else:
        # Generate synthetic data
        print("Generating synthetic multivariate time series data...")
        dataset = generate_dataset(
            train_length=args.train_length,
            test_length=args.test_length,
            n_features=args.n_features,
        )

    save_dataset(dataset)
    print()
    print("Done! Ready to train.")
