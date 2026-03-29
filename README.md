# FERA Tenor IF — Final Version

This version is scoped strictly to **pointwise tenor-level anomaly detection** using **Isolation Forest** on the engineered tenor feature CSV.

## What changed

- keeps the model family fixed to Isolation Forest
- treats the **held-out test split** as the primary evaluation slice
- uses **leakage-safe causal imputation** with training-only fallback medians
- filters rows using **active-feature completeness**
- fits **per-market models** and drops market-specific constant features automatically
- writes richer summaries including:
  - test anomaly rate
  - overall anomaly rate
  - precision proxy
  - per-market test rates
  - tenor-bucket test rates
  - spread diagnostics

## Default feature set

- `F02_LOG_RETURN_MAD_Z`
- `F03_VOLATILITY_SURPRISE`
- `F04_PERCENTILE_RANK_Z`
- `F06_CROSS_TENOR_RESIDUAL_MAD_Z`
- `F07_BUTTERFLY_CURVATURE_MAD_Z`

## Run

```bash
uv run python run_tenor_if_only.py \
  --input "/mnt/data/Untitled 15_2026-03-29-0307.csv" \
  --output-dir "./fera_tenor_if_outputs"
```

## Optional

```bash
uv run python run_tenor_if_only.py \
  --input "/mnt/data/Untitled 15_2026-03-29-0307.csv" \
  --output-dir "./fera_tenor_if_outputs" \
  --threshold 0.95 \
  --split-date 2024-07-01 \
  --completeness-threshold 0.60 \
  --impute-window 63 \
  --n-estimators 300 \
  --max-samples auto \
  --max-features 1.0 \
  --contamination auto
```
