# Event catalogue: labelling guide

`events.csv` in this folder is the labelled catalogue. It is tracked in git, so every label change is reviewable. Candidate files (`candidates_<SNAP>.csv`) are generated and not tracked.

## Workflow

```bash
uv run prepare.py --source snowflake                    # snapshot + prepared GAS markets
uv run events.py mine --model-scores snowflake --plots  # -> events/candidates_<SNAP>.csv
# open the CSV in Excel / Sheets, fill the label_* columns, save as CSV
uv run events.py load events/candidates_<SNAP>.csv       # validate + merge into events/events.csv
uv run events.py stats
```

Re-mining after a new snapshot keeps your work: `events.py mine --carry-labels <your labelled csv>` copies labels by `candidate_id` (market + peak date).

## Columns you read

| Column | Meaning |
|---|---|
| `source` | `CURVE`: model-free miner. `SHAPE_AE`: live Shape AE alert. `CURVE+SHAPE_AE`: both. `CONTROL`: a random quiet day. Label controls too; an `EVENT` among controls means the miner missed it |
| `start` / `peak` / `end` | Episode dates. You may correct them |
| `curve_score` | Peak miner score. ≥ 1 means some tenor, kink or spread move reached z ≥ 6 against the trailing year |
| `suggested_tenors` | Tenors whose shape change is large on the peak day, e.g. `M7-M9,M13` |
| `suggested_features` | Spread features with \|z\| ≥ 3, with their z |
| `type_hint` | Heuristic guess at the anomaly type |
| `peak_tenor_move_native` | The peak tenor's move relative to the whole curve's move, in price units (e.g. €/MWh) |
| `m1_move_native` | The front-month move, in price units |
| `reverts_next_day` | The peak tenor jumps and comes back the next day. Often a bad tick: consider `DATA_ERROR` |
| `shape_ae_*` | Live model: alert days, max bar multiple, driver |
| `plot_file` | PNG in `~/.cache/autoresearch/shape_ae/event_plots/` (with `--plots`) |

## Columns you fill

| Column | Values |
|---|---|
| `label` | `EVENT`, `NOT_EVENT`, `DATA_ERROR`, `UNSURE`. Leave blank to skip the row |
| `label_tenors` | Required for `EVENT`. Tenor offsets on the peak day, e.g. `M1-M3` or `M7-M9,M13` |
| `label_features` | Optional, `;`-separated: `return_z`, `rv_21d`, `corr_20d`, `front_slope_d`, `slope_d`, `back_slope_d`, `curvature_d`, `winter_summer_d`, `basis_d`, `basis_level_z`, `pca_1`..`pca_4` |
| `label_type` | Optional: `kink`, `broken_spread`, `front_twist`, `seasonal_hump`, `back_bend`, `tilt`, `regime_shift`, `other` |
| `label_notes`, `labeller` | Free text and your initials |

## What counts as an EVENT

A day the desk would want flagged as an unusual curve shape: the shape changed in a way that matters, not just the level. Mark it `EVENT` even if the cause was fundamental (an outage, a storage announcement). The model should catch those.

- `NOT_EVENT`: normal behaviour that only looks large statistically, e.g. a calm-period z on a tiny move, or an expected seasonal roll.
- `DATA_ERROR`: a bad tick, stale settlement, or missing contract.
- `UNSURE`: excluded from scoring until it is resolved.

For `label_tenors`, mark the tenors where the shape actually moved, not the whole curve. If the curve moved in parallel and one strip lagged, the lagging strip is the event.
