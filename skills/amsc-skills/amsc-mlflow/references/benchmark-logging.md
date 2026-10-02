# Logging benchmark results

How `scripts/log_benchmark_results.py` lays out runs, and why. The layout generalizes the
BASE-Eval AXESS MLflow demo (`reproduce_tables.py`: one run per split × predictor ×
architecture, per-target R2/SMAPE/RMSE, aggregate metrics) to any benchmark that uses the
benchmark-builder results layout.

## Inputs

**A benchmark-builder results tree** (`--results`): `<results>/<split>/<model>/metrics.json`
as written by the benchmark's `score.py`:

```json
{"coverage": {"n_truth": 100, "n_scored": 98, ...},
 "groups": {"all": {"<output>": {"<metric>": 0.91, ...}, ...}, "<group>": {...}}}
```

A single-output benchmark may write `{"all": {"<metric>": value}}`. Both shapes are read.
Next to it, the script also picks up `METRICS.md`, plots (`.png/.svg/.pdf`), and
`<split>/predictions_<model>.meta.json` if present (LLM runs: model id, `model_args`,
task, n-shot, lm-eval version; these become params). The leaderboard `<results>/LEADERBOARD.md`
goes on the parent run.

**An eval bundle** (`--bundle`): the canonical, harness-independent JSON the genesis
`card-eval-updater` skill writes from one lm-evaluation-harness, Eval Factory or
NeMo-Skills run. Each result becomes `<benchmark>[/<variant>].<metric>` plus `_stderr`
and `num_samples`. A bundle with any partial (sample-limited) result is refused, matching
`card-eval-updater`'s own rule.

## Run layout

```
experiment  <benchmark>-<user>   (or --experiment / MLFLOW_EXPERIMENT_NAME)
└── parent run  "<benchmark> results <UTC time>"      tags: benchmark.name, owner, git.*, runtime.*
    │                                                 artifact: LEADERBOARD.md
    ├── child "test/gnn"        params: model, split [, model_id, task, n_shot, ...]
    │                           metrics: lut.r_squared, lut.smape, ..., coverage.n_scored, ...
    │                           tags: benchmark.split, benchmark.model, benchmark.model_role,
    │                                 benchmark.source, benchmark.source_sha256 [, benchmark.nonfinite_metrics]
    │                           artifacts: results/metrics.json, results/METRICS.md, plots, meta
    └── child "exemplar/gnn"    ...
```

Why this layout:

- **Child per (split, model).** Compare models within a split in the MLflow UI by
  filtering `tags.benchmark.split = 'test'`. That matches the AXESS demo's per-split runs
  and the benchmark's leaderboard rows.
- **Metric keys `<output>.<metric>`.** These mirror `metrics.json` and the AXESS demo's
  `{target}_{metric}`. Per-group metrics (`--all-groups`) are `<group>/<output>.<metric>`.
- **`benchmark.source_sha256`.** Every logged number traces to one exact `metrics.json`,
  the same traceability rule benchmark-builder applies to model cards. It also drives
  `--skip-existing`: a re-run after a partial failure logs only what's missing, and a
  changed `metrics.json` (a re-scored model) logs a new run instead of overwriting.
- **`benchmark.model_role`.** `reference` or `auxiliary`, mirroring benchmark-builder's
  leaderboard and `reference_solution/README.md`.
- **Non-finite values** (an R² of NaN at zero variance) are logged as-is and listed in
  `benchmark.nonfinite_metrics`, since the UI's charts drop them.
- **No recomputation.** The script never computes a metric. If the logged number is
  wrong, fix `score.py` and re-score, and the new sha256 produces a new run.

## Useful searches (MLflow UI or `mlflow.search_runs`)

| Goal | Filter |
|---|---|
| one benchmark's reference models on test | `tags.benchmark.name = 'wa-hls4ml' AND tags.benchmark.split = 'test' AND tags.benchmark.model_role = 'reference'` |
| runs from one code version | `tags.git.commit = '<sha>'` |
| a model across splits | `tags.benchmark.model = 'gnn'` |
| results from a cluster job | `tags.runtime.job_id = '<id>'` |
| good runs | `metrics.lut.r_squared > 0.9` (backquote keys with special characters: ``metrics.`lut.r_squared` ``) |

## What else to log, and where

- **Training runs** (when the benchmark includes training): log from the training code
  with `amsc_mlflow` (`references/training.md`). Tag them with the same `benchmark.name`,
  and record the training run ID in the registered model version (`register_model.py run`).
- **System metrics.** The AXESS demo enables `mlflow.enable_system_metrics_logging()`
  during evaluation (CPU/GPU/memory per run). This script logs finished results, so it
  doesn't. Enable it in the code that actually runs the models (`references/training.md`).
- **Predictions** (`--log-predictions`): useful for audits, but they can be large. Prefer
  keeping them on the benchmark's storage and logging only the `.meta.json`.
- **Tables.** The AXESS demo writes its Table 4/5 files locally and doesn't log them. Here
  `METRICS.md` and `LEADERBOARD.md` are logged as artifacts.
