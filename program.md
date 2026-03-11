# autoresearch — time series anomaly detection

This is an experiment to have the LLM autonomously research time series anomaly detection algorithms.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar11`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, data generation, normalization, windowing, evaluation. Do not modify.
   - `train.py` — the file you modify. Model architecture, optimizer, training loop.
4. **Verify data exists**: Check that `~/.cache/autoresearch_ad/data/` contains `.npy` files and `metadata.json`. If not, tell the human to run `uv run prepare.py`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU. The training script runs for a **fixed time budget of 5 minutes** (wall clock training time, excluding startup/eval overhead). You launch it simply as: `uv run train.py`.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, loss functions, anomaly scoring strategies, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation, data loading, normalization, windowing, and constants (time budget, window size, etc).
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Modify the evaluation harness. The `evaluate_f1` function in `prepare.py` is the ground truth metric.

**The goal is simple: get the highest val_f1.** Since the time budget is fixed, you don't need to worry about training time — it's always 5 minutes. Everything is fair game: change the architecture (LSTM, Transformer, CNN, GAN, VAE, etc.), the optimizer, the hyperparameters, the batch size, the model size, the loss function, the anomaly scoring method. The only constraint is that the code runs without crashing and finishes within the time budget.

**Model interface contract**: Your model MUST implement `compute_anomaly_scores(windows)` that takes `(batch, window_size, n_features)` and returns `(batch,)` scores where higher = more anomalous. This is what `evaluate_f1` calls.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful val_f1 gains, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Output format

Once the script finishes it prints a summary like this:

```
---
val_f1:           0.750000
val_precision:    0.800000
val_recall:       0.700000
val_threshold:    0.123456
test_f1:          0.740000
test_precision:   0.790000
test_recall:      0.690000
training_seconds: 300.1
total_seconds:    310.5
peak_vram_mb:     2048.0
num_steps:        15000
num_params:       500,000
final_train_loss: 0.012345
```

You can extract the key metric from the log file:

```
grep "^val_f1:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	val_f1	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. val_f1 achieved (e.g. 0.750000) — use 0.000000 for crashes
3. peak memory in GB, round to .1f (e.g. 2.0 — divide peak_vram_mb by 1024) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	val_f1	memory_gb	status	description
a1b2c3d	0.750000	2.0	keep	baseline LSTM-AE
b2c3d4e	0.780000	2.1	keep	increase hidden_dim to 256
c3d4e5f	0.740000	2.0	discard	switch to GRU
d4e5f6g	0.000000	0.0	crash	Transformer encoder (OOM)
```

## Research directions to explore

Here are promising research directions for the agent (non-exhaustive):

**Architecture changes:**
- Transformer-based autoencoders (attention over time steps)
- 1D Convolutional autoencoders (temporal convolutions)
- Variational autoencoders (VAE) with KL divergence
- GAN-based approaches (adversarial training)
- Temporal convolutional networks (TCN)
- Hybrid CNN-LSTM architectures
- Multi-scale architectures (different window resolutions)

**Loss function innovations:**
- Contrastive learning losses
- Adversarial reconstruction loss
- Feature-wise reconstruction weighting
- Temporal consistency loss
- KL divergence (for VAE)

**Anomaly scoring innovations:**
- Reconstruction error + latent space distance
- Mahalanobis distance in latent space
- Ensemble scoring (combine multiple signals)
- Attention-based scoring (which features contribute most)

**Training improvements:**
- Learning rate scheduling variations
- Gradient clipping
- Mixed precision training
- Data augmentation (jitter, scaling, masking)

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar11`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `uv run train.py > run.log 2>&1`
5. Read out the results: `grep "^val_f1:\|^peak_vram_mb:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If val_f1 improved (higher), you "advance" the branch, keeping the git commit
9. If val_f1 is equal or worse, you git reset back to where you started

**Timeout**: Each experiment should take ~5 minutes total (+ startup/eval overhead). If a run exceeds 10 minutes, kill it and treat it as a failure.

**Crashes**: If a run crashes, use your judgment: fix if it's a simple bug, skip if fundamentally broken.

**NEVER STOP**: Once the experiment loop has begun, do NOT pause to ask the human if you should continue. The human might be asleep. You are autonomous. If you run out of ideas, think harder — try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.
