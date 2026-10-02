# Instrumenting training and evaluation code

Source: the AmSC integration guide ("Core Point 1", "Asynchronous Logging", "HPC
Specific: Distributed Training on Frontier"), `intro-to-mlflow-pytorch`
(`experiment_tracking/`, `hyperparameter_search/`), and the AXESS demo.

## What to track for every training run

| Kind | Examples | Call |
|---|---|---|
| params | architecture (layers, hidden sizes), lr, batch size, epochs, optimizer, seed | `mlflow.log_params({...})` once |
| metrics | train/val loss, accuracy, per epoch | `mlflow.log_metrics({...}, step=epoch)` |
| tags | owner, project phase (`development`/`tuning`/`production`), git commit/branch/dirty, model class | `mlflow.set_tags(amsc_mlflow.git_tags() | {...})` |
| artifacts | confusion matrix, loss curves, sample outputs | `mlflow.log_artifact(path, artifact_path="plots")` |
| model | the trained model with its environment (`requirements.txt`/`conda.yaml` captured automatically) | `mlflow.<flavor>.log_model(model, name="model", signature=..., input_example=...)` |
| system metrics | CPU/GPU utilization, memory | `mlflow.enable_system_metrics_logging()` before the run (AXESS demo) |

`mlflow.note.content` (tag) holds a free-text note shown on the run page.

## The template

`assets/train_with_mlflow.py` is a complete, runnable example (a toy PyTorch regression)
showing all of the following. Copy its structure, not its model.

## Asynchronous logging

```python
mlflow = amsc_mlflow.configure(experiment=..., async_logging=True)   # mlflow.config.enable_async_logging()
...
amsc_mlflow.finish()                                                 # mlflow.flush_async_logging()
```

- Params, metrics and tags are submitted in the background, so the loop doesn't wait on
  the network.
- It **doesn't** make `log_artifact()` or model/checkpoint uploads non-blocking. Upload
  large files rarely (end of training, best checkpoint), not every epoch.
- Flush before the process exits, especially in a batch job, or the last metrics are lost.
  `rank_zero_run()` flushes on exit.

## Distributed training: rank 0 only

The guide's rule: only rank 0 calls MLflow, and it logs metrics already reduced across all
ranks, so the logged value reflects the global state.

```python
with amsc_mlflow.rank_zero_run(run_name=..., tags=...) as run:   # None on other ranks
    for epoch in range(epochs):
        loss_t = torch.tensor(local_loss, device=device)
        torch.distributed.all_reduce(loss_t, op=torch.distributed.ReduceOp.AVG)   # every rank
        if run:
            mlflow.log_metric("train_loss", loss_t.item(), step=epoch)
```

`amsc_mlflow.global_rank()` uses `torch.distributed` when initialized, else the launcher's
variables: `RANK` (torchrun), `SLURM_PROCID` (srun), `PMI_RANK`/`PMIX_RANK` (MPICH, Cray),
`OMPI_COMM_WORLD_RANK` (Open MPI), `PALS_RANKID` (Aurora's mpiexec). The collective must
run on every rank; only the logging is guarded.

## Hyperparameter sweeps

One MLflow run per configuration in the same experiment, with the configuration as
params, a `sweep.config_index` tag, and final values as unstepped metrics
(`final_val_accuracy`), so runs compare in the UI table (`intro-to-mlflow-pytorch`
`hyperparameter_search/run_sweep.py`). Nested runs under a parent "sweep" run keep large
sweeps tidy.

## Evaluating registered models (AXESS demo pattern)

The AXESS demo's evaluation (`reproduce_tables.py`):
1. loads each model from the registry by version (`models:/<name>/<version>`, resolving
   `latest` by searching versions);
2. evaluates per split and architecture, one run each, with system metrics on;
3. logs per-target metrics and their average.

In a benchmark-builder benchmark, the equivalent is:
- `predict.py` loads `models:/<name>@production` (or a pinned version);
- `score_all.sh` scores the predictions;
- `log_benchmark_results.py` logs the scores.

Pin a version number, not an alias, when reproducing published results: aliases move.

## Migrating from Weights & Biases

| W&B | MLflow |
|---|---|
| `wandb.init(project=...)` | `mlflow.set_experiment(...)` + `mlflow.start_run()` |
| `wandb.config.update(cfg)` | `mlflow.log_params(cfg)` |
| `wandb.log({"loss": v})` | `mlflow.log_metric("loss", v, step=step)` |
| `wandb.finish()` | leave the `start_run()` context |

`intro-to-mlflow-pytorch/migration_from_wandb/` converts existing W&B runs.

## LLM calls

For code that calls an LLM API, MLflow tracing records prompts and responses in the Traces
UI: `mlflow.openai.autolog(log_traces=True)` before the client calls (`intro-to-mlflow-pytorch`
`prompts_tracing/`). mlflow >= 3.16 also ships its own agent skill for tracing
(`mlflow/assistant/skills/instrumenting-with-mlflow-tracing/SKILL.md` inside the installed
package; MLflow prints its path on import). Use that for tracing details.
