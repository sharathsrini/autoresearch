# autoresearch: Shape AE

This is an experiment to have the LLM do its own research: find the best autoencoder for **forward-curve shape anomaly detection**.

The model takes the raw forward curve of one market (M1..M36) together with feature channels of that curve. It has to rebuild **both** the price curve and the 14 scalar curve features from a narrow latent code. A day it cannot rebuild well is a shape anomaly.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `oct19`). The branch `autoresearch/<tag>` must not already exist. This is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from the current main branch.
3. **Read the in-scope files**: the repo is small. Read these files for full context:
   - `README.md`: repository context.
   - `prepare.py`: fixed constants, data prep, features, splits, scalers, anomaly injection, evaluation and output export. Do not modify.
   - `train.py`: the file you modify. Model architecture, inputs used, loss, optimizer, training loop.
4. **Verify data exists**: check that `~/.cache/autoresearch/shape_ae/markets/` contains `meta.json` and one `.npz` per market. If not, tell the human to run `uv run prepare.py` (synthetic curves) or `uv run prepare.py --curves <file>` (real curves).
5. **Initialize results.tsv**: create `results.tsv` with just the header row. The baseline is recorded after the first run.
6. **Confirm and go**: confirm the setup looks good.

Once you get confirmation, kick off the experimentation.

## The problem

Per market, every trade date has:

| Input | Shape | Meaning |
|---|---|---|
| `shape` | 36 | T(price) minus the curve's own mean (level removed), robust-scaled per tenor. **Curve reconstruction target.** |
| `dshape` | 36 | Day change of `shape`, roll-aligned (ALIGN mode), robust-scaled per tenor. |
| `curve` | 36 | T(price) robust-scaled per tenor, level included (the raw curve). |
| `feat` | 14 | Return z, RV 21d, Corr 20d, Front-slope Δ, Slope Δ, Back-slope Δ, Curvature Δ, Winter–summer Δ, Basis Δ, Basis level z, PCA factors 1-4. Robust-scaled. **Feature reconstruction target.** |
| `month` | 1 | Calendar month of the trade date. |

T is LOG for gas, LNG and EUA, and ASINH(p/10) for power and the clean spark spread (CSS). The price curve is reconstructed by re-anchoring the rebuilt shape at the observed level: `price_hat = T⁻¹(shape_hat · MAD + median + level)`.

Splits (one fold, the live fold of the deployed model):

- FIT: data up to 2023-12-31, calm days only (stressed days are dropped).
- Early-stop: 2024-02-05 to 2024-09-30.
- Calibration: 2024-11-05 to 2025-06-30, calm days. Sets the anomaly bars.
- Test: from 2025-08-05.

Scalers, PCA and the calm filter are fitted on FIT only.

## The metric

`evaluate()` in `prepare.py` is the ground truth. For each market:

1. Your model rebuilds `shape` and `feat` for calibration days, clean test days and **injected test days**. Injected days are copies of real test days with one shape anomaly added to the transformed curve: a single-tenor kink, a broken spread, a front twist, a seasonal-hump distortion or a back-end bend. Amplitudes are 1.5-6× the typical daily shape change. All features are recomputed from the perturbed curve.
2. Curve error = RMS of `shape - shape_hat`. Feature error = RMS of `feat - feat_hat`.
3. Bars = q0.99 of the calibration errors. Anomaly score = `max(curve_err / curve_bar, feat_err / feat_bar)`.
4. Market score = **AUPRC** of that score, with clean test days as negatives and injected days as positives. A market scores **0** if the clean-test shape explained variance is below 0.80. The model must still rebuild normal curves.

**`val_score` = mean market score. Higher is better.** The maximum is 1.0.

The tension is the whole game: a wide bottleneck rebuilds everything, including the anomalies, which kills detection. A narrow one cannot rebuild normal curves, which fails the EV floor and raises false alarms. Look for architectures that model *normal* shape tightly and nothing else.

## Experimentation

Each run trains one model per market inside a **fixed total time budget of 120 s** (`TIME_BUDGET` in `prepare.py`, split equally across markets). Then it evaluates and writes the outputs. Launch it with: `uv run train.py`. It runs on GPU if one is available, otherwise on CPU.

**What you CAN do:**
- Modify `train.py`. It is the only file you edit. Everything is fair game: which input channels to use (`shape`, `dshape`, `curve`, `month`), how the features are fused, the encoder/decoder (conv, dilations, attention over tenors, MLP), latent size, latent regularisation (VAE/KL, sparsity, contractive, Mahalanobis whitening), denoising/masking schemes, loss (Huber/MSE/weighted per tenor, curve vs feature weight), optimizer, schedule, early stopping, ensembling and seeds.
- Train any way you like on the `fit` split, and use `early_stop` for model selection.

**What you CANNOT do:**
- Modify `prepare.py`. It holds the fixed data, splits, scalers, anomaly injection and evaluation.
- Train on, or tune against, the `calibration`, `test` or `injected` splits. Only `fit` and `early_stop` may touch the weights or the hyperparameter selection inside a run. Using the others is cheating, and the result is invalid.
- Change what is reconstructed. `reconstruct_fn` must return `(shape_hat, feat_hat, z)` in the scaled units of `batch["shape"]` and `batch["feat"]`.
- Install new packages or add dependencies. Use only what is in `pyproject.toml`.

**Simplicity criterion**: all else being equal, simpler is better. A +0.002 val_score that adds 30 lines of hacky code is probably not worth it. Equal score with less code is a win. Prefer changes that make sense for curve shape (tenor locality, roll alignment, seasonality, level invariance).

**The first run**: always establish the baseline first by running the training script as is.

## Output format

When the script finishes it prints a per-market table and a summary like this:

```
market TTF          score=0.9312 auprc=0.9312 auroc=0.9420 ev=0.9810 far=0.006 recall=0.512 price_mape=0.00041
...
---
val_score:        0.912345
mean_auprc:       0.912345
mean_auroc:       0.930000
mean_recon_ev:    0.975000
mean_far:         0.010000
mean_recall:      0.480000
price_mape:       0.000400
training_seconds: 118.3
total_seconds:    125.0
peak_vram_mb:     40.1
num_steps:        12345
num_params_K:     46.2
latent_dim:       6
```

Extract the key metric with:

```
grep "^val_score:\|^mean_recon_ev:\|^mean_far:" run.log
```

Watch `mean_far` (clean test days above the bar) and `mean_recon_ev` too. A gain in val_score that pushes EV toward the 0.80 floor, or doubles the false-alarm rate, deserves a second look.

Each run also writes outputs to `~/.cache/autoresearch/shape_ae/outputs/`:

- `<MKT>_scores.csv`: A_GLOBAL, A_R1..A_R4 regions, A_BUTTERFLY_MAX, A_FEAT, BAR_MULTIPLE, DECISION, SEVERITY, driver region and feature, A_MAHAL, TRANSITION_SCORE and the latent code.
- `<MKT>_curve_recon.csv`: price vs reconstructed price per tenor, plus the scaled residual.
- `<MKT>_feature_recon.csv`: features vs reconstruction, plus the residual.
- `<MKT>_latest.png`: price curve vs reconstruction, residual per tenor, and the feature residual bars for the latest day.

## Logging results

When an experiment is done, log it to `results.tsv`. Use tabs, not commas, because commas break the descriptions.

The TSV has a header row and 6 columns:

```
commit	val_score	recon_ev	far	status	description
```

1. git commit hash (short, 7 chars)
2. val_score achieved (e.g. 0.912345). Use 0.000000 for crashes.
3. mean_recon_ev (e.g. 0.975). Use 0.000 for crashes.
4. mean_far (e.g. 0.010). Use 0.000 for crashes.
5. status: `keep`, `discard`, or `crash`
6. a short description of what the experiment tried

Example:

```
commit	val_score	recon_ev	far	status	description
a1b2c3d	0.912345	0.975	0.010	keep	baseline
b2c3d4e	0.925100	0.968	0.009	keep	latent 6 -> 4
c3d4e5f	0.901000	0.990	0.020	discard	add curve (level) channel
d4e5f6g	0.000000	0.000	0.000	crash	attention encoder (shape bug)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/oct19`).

LOOP FOREVER:

1. Look at the git state: the current branch and commit.
2. Change `train.py` to try an experimental idea.
3. git commit
4. Run the experiment: `uv run train.py > run.log 2>&1`. Redirect everything; do NOT use tee or let the output flood your context.
5. Read the results: `grep "^val_score:\|^mean_recon_ev:\|^mean_far:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the stack trace and attempt a fix. If you can't get it working after a few attempts, give up on that idea.
7. Record the results in the TSV. Do not commit `results.tsv`; leave it untracked by git.
8. If val_score improved (higher), "advance" the branch by keeping the commit.
9. If val_score is equal or worse, `git reset` back to where you started.

You are a fully autonomous researcher: try things out, keep what works and discard what doesn't. Rewind the branch only very sparingly, if ever.

**Timeout**: each experiment should take about 2-3 minutes in total. If a run exceeds 6 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: if the cause is something dumb and easy to fix (a typo, a shape mismatch), fix it and re-run. If the idea itself is broken, skip it, log `crash`, and move on.

**NEVER STOP**: once the experiment loop has begun after the initial setup, do NOT pause to ask the human whether to continue. The human might be asleep and expects you to keep working until you are manually stopped. If you run out of ideas, think harder: per-tenor loss weights, dilated convs over tenors, tenor positional embeddings, conditioning on calendar month (winter/summer humps roll through the tenors), separate curve/feature latents, a VAE with KL warmup, masked-tenor denoising (hide some tenors and rebuild them), contrastive or whitened latents, small ensembles over seeds, sharper early stopping, combining previous near-misses. The loop runs until the human interrupts you.
