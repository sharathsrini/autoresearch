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
4. **Verify data exists**: check that `~/.cache/autoresearch/shape_ae/markets/` contains `meta.json` and one `.npz` per market. If not, tell the human to run `uv run prepare.py --source snowflake` (real curves), `uv run prepare.py --curves <file>`, or `uv run prepare.py` (synthetic). Note the `snapshot_id` in `meta.json`. All runs in one session must use the same snapshot, otherwise their scores are not comparable. `fera.toml`, `data_sources.py` and `tests/` are fixed like `prepare.py`.
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

`evaluate()` in `prepare.py` is the ground truth. The aim is a model whose flags are **explainable**: when it flags a day, the residuals must point at the tenors and the feature that actually moved.

The model rebuilds three targets: `shape` (the state of the curve), `dshape` (where the curve moved today) and `feat`. `reconstruct_fn` returns `(shape_hat, dshape_hat, feat_hat, z)`. Then `prepare.py`, which you cannot change, does the following:

1. **Normalises the residuals.** Each tenor's evidence is `sqrt((r_shape² + r_dshape²)/2)`, with each residual divided by its RMS on calibration days. Feature residuals are divided the same way. A tenor or feature that is always rebuilt poorly therefore cannot dominate. Regions are normalised by their own calibration q90, so short regions are not favoured.
2. **Sets the bars** so that 7.5% of all calibration days alert, the desk's 5–10% band. A day is flagged when `max(curve RMS / bar, feature RMS / bar) ≥ 1`.
3. **Scores explanations.** On flagged days, *tenor accuracy* is the share of the true tenors found among the k largest tenor evidences (k = number of true tenors). *Feature accuracy* is whether the largest feature residual is one of the features that really moved.
4. **Scores per market:**
   - **injected anomalies** (kink, broken spread, front twist, seasonal hump, back-end bend, on real test days): `S_inj = recall × (0.7·tenor_acc + 0.3·feature_acc)`
   - **labelled events** (`events/events.csv`, the dev half only): `S_ev`, with the same formula against the labelled tenors and features
   - **blend**: `S = (1−w)·S_inj + w·S_ev`, with `w = n_events / (n_events + 20)`. The event weight grows as the human labels more events.
   - **gates**: `S = 0` if the clean-test shape EV < 0.80. `S` is multiplied by an **alert factor**: 1 when the alert rate on out-of-calibration days (early-stop + test) is within 5–10%, falling linearly to 0 at 0% and at 20%. This punishes models that drift into alarm storms or go silent outside calibration.

**`val_score` = mean of `S` over markets. Higher is better.** For orientation on synthetic data:
- a linear PCA autoencoder scores about 0.42;
- an oracle that rebuilds the clean version of every injected day scores about 0.6.

The smallest injections sit inside normal noise, so 1.0 is not reachable.

What makes a model better here:
- normal curves are rebuilt tightly, and only the anomalous tenors are left in the residual;
- residuals are localised: a kink at M8 must not smear across M5–M12;
- the alert rate stays stable outside the calibration window.

A wide bottleneck rebuilds the anomalies too, and recall collapses. Simply copying the input scores exactly 0.

## Experimentation

Each run trains one model per market inside a **fixed total time budget of 120 s** (`TIME_BUDGET` in `prepare.py`, split equally across markets). Then it evaluates and writes the outputs. Launch it with: `uv run train.py`. It runs on GPU if one is available, otherwise on CPU.

**What you CAN do:**
- Modify `train.py`. It is the only file you edit. Everything is fair game: which input channels to use (`shape`, `dshape`, `curve`, `month`), how the features are fused, the encoder/decoder (conv, dilations, attention over tenors, MLP), latent size, latent regularisation (VAE/KL, sparsity, contractive, Mahalanobis whitening), denoising/masking schemes, loss (Huber/MSE/weighted per tenor, curve vs feature weight), optimizer, schedule, early stopping, ensembling and seeds.
- Train any way you like on the `fit` split, and use `early_stop` for model selection.

**What you CANNOT do:**
- Modify `prepare.py`. It holds the fixed data, splits, scalers, anomaly injection and evaluation.
- Train on, or tune against, the `calibration`, `calibration_all`, `test`, `eval` or `injected` splits. Only `fit` and `early_stop` may touch the weights or the hyperparameter selection inside a run. Using the others is cheating, and the result is invalid.
- Change what is reconstructed. `reconstruct_fn` must return `(shape_hat, dshape_hat, feat_hat, z)` in the scaled units of `batch["shape"]`, `batch["dshape"]` and `batch["feat"]`.
- Read `events/events.csv`, or set `SHAPE_AE_EVENT_HOLDOUT`. The held-out half of the events is for the human only.
- Install new packages or add dependencies. Use only what is in `pyproject.toml`.

**Simplicity criterion**: all else being equal, simpler is better. A +0.002 val_score that adds 30 lines of hacky code is probably not worth it. Equal score with less code is a win. Prefer changes that make sense for curve shape (tenor locality, roll alignment, seasonality, level invariance).

**The first run**: always establish the baseline first by running the training script as is.

## Output format

When the script finishes it prints a per-market breakdown and a summary like this:

```
market TTF          score=0.4120 inj=0.4120 recall=0.780 tenor_acc=0.610 feature_acc=0.700 alert_rate=0.068 alert_factor=1.00 ev=0.975 events=0 event_recall=nan regions={...}
  by type: kink=0.40/0.98  broken_spread=0.85/0.66  front_twist=0.93/0.45  seasonal_hump=1.00/0.75  back_bend=0.90/0.45  (recall/tenor_acc)
...
---
val_score:        0.412000
mean_recall:      0.780000
mean_tenor_acc:   0.610000
mean_feature_acc: 0.700000
mean_alert_rate:  0.068000
mean_recon_ev:    0.975000
...
```

Extract the key metric with:

```
grep "^val_score:\|^mean_tenor_acc:\|^mean_alert_rate:" run.log
```

The per-type line shows where to look. Low kink recall means the model smooths kinks back in. Low tenor accuracy on back bends means back-end residuals smear.

Each run also writes outputs to `~/.cache/autoresearch/shape_ae/outputs/`:

- `<MKT>_scores.csv` (every day from 2024-02): DECISION, SEVERITY, TRIGGER (CURVE/FEATURE), DRIVER_TENORS (e.g. `M7-M9`), DRIVER_TENOR and its gap in price units, DRIVER_FEATURE(_2), DRIVER_REGION, an EXPLANATION sentence, A_GLOBAL, A_CURVE, A_FEAT, regions, A_MAHAL, TRANSITION_SCORE and the latent code.
- `<MKT>_curve_recon.csv`: price vs reconstructed price per tenor, the gap in price units, and the normalised residual.
- `<MKT>_feature_recon.csv`: features vs reconstruction, plus the normalised residual.
- `<MKT>_latest.png`: the latest flagged day: price curve vs reconstruction, driver tenors in red, feature residual bars.

## Logging results

When an experiment is done, log it to `results.tsv`. Use tabs, not commas, because commas break the descriptions.

The TSV has a header row and 6 columns:

```
commit	val_score	tenor_acc	alert_rate	status	description
```

1. git commit hash (short, 7 chars)
2. val_score achieved (e.g. 0.412345). Use 0.000000 for crashes.
3. mean_tenor_acc (e.g. 0.610). Use 0.000 for crashes.
4. mean_alert_rate (e.g. 0.068). Use 0.000 for crashes.
5. status: `keep`, `discard`, or `crash`
6. a short description of what the experiment tried

Example:

```
commit	val_score	tenor_acc	alert_rate	status	description
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
5. Read the results: `grep "^val_score:\|^mean_tenor_acc:\|^mean_alert_rate:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the stack trace and attempt a fix. If you can't get it working after a few attempts, give up on that idea.
7. Record the results in the TSV. Do not commit `results.tsv`; leave it untracked by git.
8. If val_score improved (higher), "advance" the branch by keeping the commit.
9. If val_score is equal or worse, `git reset` back to where you started.

You are a fully autonomous researcher: try things out, keep what works and discard what doesn't. Rewind the branch only very sparingly, if ever.

**Timeout**: each experiment should take about 2-3 minutes in total. If a run exceeds 6 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: if the cause is something dumb and easy to fix (a typo, a shape mismatch), fix it and re-run. If the idea itself is broken, skip it, log `crash`, and move on.

**NEVER STOP**: once the experiment loop has begun after the initial setup, do NOT pause to ask the human whether to continue. The human might be asleep and expects you to keep working until you are manually stopped. If you run out of ideas, think harder: per-tenor loss weights, dilated convs over tenors, tenor positional embeddings, conditioning on calendar month (winter/summer humps roll through the tenors), separate curve/feature latents, a VAE with KL warmup, masked-tenor denoising (hide some tenors and rebuild them), contrastive or whitened latents, small ensembles over seeds, sharper early stopping, combining previous near-misses. The loop runs until the human interrupts you.
