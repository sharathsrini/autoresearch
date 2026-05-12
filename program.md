# autoresearch — FERA curve-shape autoencoder

You are an autonomous research agent. Your job is to lower a single number
(`val_metric`) on a small autoencoder that detects anomalies in energy
forward curves. Iterate via short experiments, one git branch each.

## Repo orientation

Three files matter:

- **`prepare.py`** — fixed. Reads CSVs, builds the per-market curve dataset,
  applies the walk-forward split, filters the training window, row-standardizes
  inputs, exposes `compute_metric` and the kept-guards. **Do not modify.**
- **`train.py`** — the only file you edit. Contains `CurveShapeAE`, the
  training loop, optimizer wiring, CLI args.
- **`program.md`** — these instructions.

Everything else (CSV data, `runs/`, `results.tsv`, `pyproject.toml`,
`requirements.txt`) is fixed scaffolding.

Data inputs (read by `prepare.py`, do not touch):
`ml_wide.csv`, `ml_long.csv`, `curve_features.csv`, `tenor_features.csv`,
`cross-market-features.csv`. Lookup is `data/<name>` first, then repo root.

## The single metric

Lower is better:

```
val_metric = mean over val rows of  sum_i (x_i - x_hat_i)^2
```

Val window = 2024-07-01 .. 2024-12-31, with **no anomaly filtering**.

## Kept-guards (verbatim)

A run is **kept** only if **all four** hold versus the current best in
`results.tsv`:

1. `val_metric` is strictly lower than the prior best kept `val_metric`.
2. `train_mse > 1e-4`  — memorization guard. If the AE has collapsed training
   loss to ~0, it has overfit.
3. `latent_std_min > 0.05` — collapse guard. `latent_std_min = min over
   bottleneck dims of std(z) across val rows`.
4. `corr_with_F11 < 0.92` — differentiation guard. Pearson correlation of the
   per-row val anomaly score against `F11_PCA_RECONSTRUCTION_ERROR_Z` on val
   dates. Above 0.92 means the AE adds nothing over the existing linear PCA
   baseline (F11) and has failed its job.

If any guard fails, log the run, revert the branch.

`train.py` already calls `prepare.evaluate_guards` and writes `kept` to
`results.tsv`. Trust that. Read the printed `[guards]` line.

## Git protocol (one branch per experiment)

```
git checkout master
git pull --ff-only                          # if remote exists
git checkout -b autoresearch/<tag>
# edit train.py
python train.py --tag <tag>
# inspect the [metric] and [guards] lines; check the last row of results.tsv
```

If `kept=1`:
```
git add train.py results.tsv runs/<tag>/
git commit -m "<tag>: <one-line summary> (val_metric=<value>)"
git checkout master
git merge --no-ff autoresearch/<tag>
```

If `kept=0`:
```
# results.tsv still gets the row (we keep the log of failed attempts).
git checkout master
git checkout -- train.py                    # revert your changes
git branch -D autoresearch/<tag>            # discard the branch
# optionally: stage results.tsv on master so the failed-run row persists
```

**One axis per branch.** Do not change architecture and optimizer in the
same run — you will not know which knob moved the metric.

## Search directions (ordered by expected payoff)

Run them roughly in this order. Each direction is one or more branches; one
branch = one config change.

a. **Bottleneck width sweep:** 2, 4, 6, 8, 12. Default is 6.
b. **Encoder/decoder depth:** add or remove a hidden layer (keep symmetric).
c. **Hidden widths:** try `hidden1, hidden2` in {16,12}, {20,12}, {24,12}
   (default), {32,16}, {48,24}.
d. **Activation:** tanh (default) vs gelu vs leaky_relu.
e. **Optimizer:** Adam (default) vs AdamW. `lr` in {5e-4, 1e-3, 2e-3}.
   `weight_decay` in {0, 1e-5, 1e-4}.
f. **Loss weighting:** uniform (default) vs inverse-volatility per tenor —
   compute `per_tenor_std = train_X.std(axis=0)`, weight each squared
   residual by `1 / max(per_tenor_std, eps)`. Affects training loss only;
   `val_metric` stays unweighted (`prepare.compute_metric` is fixed).
g. **Latent regularizer:** contractive penalty on the encoder Jacobian.
   Coefficient in {0, 1e-4, 1e-3}. Add to the training loss only.
h. **Optional input swap:** row-standardize (default) vs (log-price, then
   per-tenor robust z). Only try after (a)-(g) plateau.

## Stop conditions

Stop when **either** holds:
- 50 branches attempted, OR
- 3 hours of wall clock elapsed since the first branch in this session.

## Conventions

- Time budget per run: 3 minutes wall clock (`--time_budget_s 180`,
  enforced inside `train.py`). Pick `epochs`, `batch_size` that fit.
- CPU only. Do not introduce `.cuda()`, `torch.compile`, or device flags.
- Do not install new packages. Use what's in `requirements.txt`.
- If a run crashes or NaNs, treat it as not-kept and revert the branch.
- `runs/<tag>/model.pt` and `runs/<tag>/config.json` are written automatically.
- The `--tag` arg should match the branch's tag (the bit after
  `autoresearch/`).

That's it. Start with `git checkout -b autoresearch/<your-first-tag>` and
make one small change.
