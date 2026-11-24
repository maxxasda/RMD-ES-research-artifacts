# Data dictionary and analysis rules

The release contains 90 run-level CSV files: three scenarios, ten methods, and three independent runs for each scenario–method pair. Each file is named `sotif_search_results.csv`. `results/data_manifest.csv` provides the relative path, method, scenario, run ID, seed, file size, and SHA-256 hash for each CSV.

## CSV fields

Scenario-parameter columns vary by scenario. The shared evaluation columns include:

| Field | Meaning |
|---|---|
| `step` | Search-generation/step identifier. Multiple evaluated records can share one step. |
| `target_fit` | Internal search objective and ranking value. It is not a calibrated physical risk scale and should not be compared absolutely across scenarios. |
| `collision` | Collision indicator recorded by the simulation pipeline. |
| `min_TTC` | Minimum time-to-collision value, in seconds, computed by the original evaluation pipeline. |
| `min_distance` | Auxiliary distance-related output present in some scenario CSVs. It is not a substitute for a geometric collision-distance measure. |

## Reported rates

- **Collision rate:** records with `collision=true` divided by all records in that run.
- **Target-valid collision rate:** records with `collision=true` and `target_fit - min_TTC < 0`, divided by all records in that run. This follows the project's analysis rule; it is not a standard calibrated safety metric.
- **Low-TTC coverage:** records with `min_TTC < 1 s` and `< 3 s`, respectively, divided by all records in that run.
- **Aggregation:** calculate each metric per run, then take the arithmetic mean across three runs. The script does not report standard deviations or statistical significance.

## Convergence processing

The script sorts numeric `step` values, groups all CSV records belonging to the same step, updates the observed best-so-far only at complete-step endpoints, carries that minimum forward, averages the three runs, and truncates at 2300 valid evaluations. It does not invent within-generation ordering or interpolate intermediate simulator values. Only the six main methods are plotted; ablations are summarized in the ablation table.
