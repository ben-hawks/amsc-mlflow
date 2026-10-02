---
name: amsc-mlflow
description: Track experiments, log benchmark results, and register models on the American Science Cloud (AmSC) / Genesis Mission MLflow service with the standard MLflow client. Use when a user wants to connect to AmSC MLflow with a MyAmSC Access Token, check connectivity, log a benchmark's scores or an LLM evaluation run, instrument a training script (async logging, rank-0 logging on HPC), register reference-model weights with staging/production aliases, load registered models by alias (models:/NAME@production), or run any of this from Slurm or PBS jobs on Perlmutter, Frontier, Aurora or Polaris (proxies, tokens in batch jobs, large uploads). Pairs with the benchmark-builder skill's results layout and the card-eval-updater eval bundle.
license: Apache-2.0
compatibility: Requires Python >= 3.10 and mlflow >= 3.13 (workspace support; tested with 3.16.1). Network access to the AmSC MLflow server (default https://mlflow-staging.american-science-cloud.org), through the site proxy on some HPC compute nodes. A MyAmSC account with an Access Token and workspace permission.
metadata:
  author: Ben Hawks, building on AmSC Model Services and BASE-Eval examples (see ATTRIBUTION.md)
  version: "0.1"
allowed-tools: Bash(python *) Bash(python3 *) Read Glob Grep
---

# AmSC MLflow

Use the American Science Cloud's hosted MLflow for experiment tracking and as the model
catalog, with the stock `mlflow` client. Authentication is MLflow's standard bearer token.
No custom headers, no monkey patching.

Scripts live in `scripts/`; run them with the skill directory as the working directory or
by absolute path. They share `scripts/amsc_mlflow.py`, which a user's own code can import
too.

| Script | Does |
|---|---|
| `scripts/check_connection.py` | prints the effective settings (never the token), checks read and write access, and names the fix for each failure |
| `scripts/log_benchmark_results.py` | logs a benchmark's scores: a benchmark-builder results tree (`<split>/<model>/metrics.json`) or a card-eval-updater `bundle.json` |
| `scripts/register_model.py` | registers a logged model, or manifest-checked checkpoint files as a pyfunc model; sets and moves aliases |
| `scripts/amsc_mlflow.py` | `configure()`, token handling and expiry, multipart uploads, rank-0 runs, git and runtime tags |

Read `references/setup-and-auth.md` before the first connection. Read
`references/hpc.md` before anything runs in a batch job.

**Related skill.** The genesis `amsc-python-client` skill reaches MLflow through the AmSC
gateway's REST proxy (`client.mlflow`). Use it for quick catalog-style calls from an
AmSC SDK session. Use this skill when code uses the `mlflow` library itself: training,
benchmark logging, logging and loading models.

## 1. Connect

1. **Token.** The user copies the **Access Token** (not the ID Token) from MyAmSC Profile.
   Have them set it themselves, without echoing it:
   ```bash
   printf 'Paste Access Token: '; read -s MLFLOW_TRACKING_TOKEN; printf '\n'; export MLFLOW_TRACKING_TOKEN
   ```
   Never ask the user to paste the token into the conversation. Never print it, log it,
   write it into code or commit it. For batch jobs, use a chmod-600 file named by
   `MLFLOW_TRACKING_TOKEN_FILE` (`references/hpc.md`).
2. **Server and workspace.**
   ```bash
   export MLFLOW_TRACKING_URI='https://mlflow-staging.american-science-cloud.org'   # the default
   export MLFLOW_WORKSPACE='modelservices'                                           # the default on AmSC
   export MLFLOW_EXPERIMENT_NAME="<project>-${USER}"
   ```
   Put the username in personal experiment names: creating runs in an experiment another
   user owns can return 403. Selecting a workspace doesn't grant access to it. The user
   needs the workspace user role (USE permission) from a platform admin.
3. **Check:** `python scripts/check_connection.py`. Exit codes: 0 ok, 2 no token, 3 token
   rejected (401) or expired, 4 no permission (403), 5 network/proxy. It prints the fix
   for each; relay it. Don't work around a 401/403 by switching to a local store without
   telling the user.

## 2. Log a benchmark's results

For a benchmark with benchmark-builder's layout, after `score_all.sh` has written
`reference_results/` (or `$BENCH_RESULTS`):

```bash
python scripts/log_benchmark_results.py --benchmark <name> --results reference_results \
    --repo <benchmark checkout> [--auxiliary <models>] [--dataset-revision <rev>] --dry-run
python scripts/log_benchmark_results.py --benchmark <name> --results reference_results --repo <checkout>
```

- Always show the `--dry-run` output first. It needs no server or token, and it's what the
  user approves.
- One parent run per invocation, and one child run per (split, model): metrics as
  `<output>.<metric>` plus `coverage.*`, the model's `metrics.json`/`METRICS.md`/plots as
  artifacts, and tags for benchmark, split, model, role (reference/auxiliary), git commit,
  runtime and source sha256.
- Re-running skips any (split, model) whose `metrics.json` sha256 is already logged.
- `--all-groups` adds per-group metrics; `--log-predictions` uploads prediction files.
  These may be large, so check the size first.
- For an LLM evaluation, pass the genesis `card-eval-updater` bundle instead:
  `--bundle bundle.json --split test`. Partial (sample-limited) runs are refused.

Numbers are copied from the scoring output, never recomputed or retyped. Record the parent
run ID in the benchmark's validation log (benchmark-builder: `VALIDATION.md` in its docs folder). The tag schema and the reasons behind it
are in `references/benchmark-logging.md`.

## 3. Instrument training or evaluation code

Import `amsc_mlflow` before `mlflow` (it turns on proxy multipart uploads, which MLflow
reads at import time):

```python
import amsc_mlflow
mlflow = amsc_mlflow.configure(experiment=f"myproj-{user}", async_logging=True)

with amsc_mlflow.rank_zero_run(run_name="train-lr1e-3", tags=amsc_mlflow.git_tags()) as run:
    if run:
        mlflow.log_params({"lr": lr, "batch_size": bs, "epochs": epochs})
    for epoch in range(epochs):
        loss = all_reduce_mean(loss)            # reduce across ranks FIRST
        if run:
            mlflow.log_metrics({"train_loss": loss, "val_loss": val}, step=epoch)
```

`assets/train_with_mlflow.py` is a complete, runnable template. The rules
(`references/training.md`):
- only rank 0 talks to MLflow;
- reduce metrics across ranks before logging;
- async logging covers params, metrics and tags, not artifacts or models;
- flush before exit (`rank_zero_run` does);
- log checkpoints sparingly.

## 4. Register and serve models

```bash
# a model already logged with mlflow.<flavor>.log_model
python scripts/register_model.py run --model-uri runs:/<run_id>/model --name <name> --alias staging
# checkpoint files from a benchmark's weights/MANIFEST.json (sha256-verified before upload)
python scripts/register_model.py files --manifest weights/MANIFEST.json --weights-dir <dir> \
    --files <file> [...] --name <benchmark>-<model> --loader <pkg>.mlflow_loader:load \
    --code-path src/<pkg> --benchmark <name> --alias staging
# promote after verification
python scripts/register_model.py alias --name <name> --from-alias staging --alias production
```

- Register new versions as `staging`. Move `production` only after the version has passed
  the benchmark's checks (benchmark-builder: golden tests and a run that reproduces
  `reference_results/`).
- Ask before moving `production`: consumers load `models:/<name>@production`.
- The loader contract, flavors vs pyfunc, and zero-copy options for very large models are
  in `references/model-registry.md`.

## 5. HPC jobs

Read `references/hpc.md`. In short:
- **Proxies.** Compute nodes on Frontier and ALCF systems need an outbound proxy (Frontier:
  `proxy.ccs.ornl.gov:3128`; ALCF: check its docs). Perlmutter needs none.
  `assets/mlflow_job_env.sh` sets these up per site.
- **Token expiry.** The token must outlive the job. `check_token_lifetime()` warns when it
  won't.
- **Log after the job when you can.** Results logging reads files, so run
  `log_benchmark_results.py` from a login node after the job instead of holding a token in
  the queue.
- **Large uploads.** Keep the multipart upload variables on, and consider keeping large
  checkpoints on shared storage.

## Troubleshooting

| Symptom | Meaning | Action |
|---|---|---|
| `401 Unauthorized: Bearer token required` / `Invalid Bearer Token` | no usable Access Token reached the gateway | set or refresh `MLFLOW_TRACKING_TOKEN` (Access, not ID, token) |
| `403 Permission denied` | token accepted, no permission for the workspace or action | ask for the workspace role; use your own experiment name; or an authorized workspace |
| `Active workspace ... the remote server does not support workspaces` | a workspace is set against a server without workspaces (e.g. a local server) | `MLFLOW_WORKSPACE=none`, or start that server with `--enable-workspaces` |
| run succeeded but isn't in the UI | the UI shows another workspace or experiment | select the same `MLFLOW_WORKSPACE` and experiment |
| hangs or connection errors on compute nodes | no route out | set the site proxy (`references/hpc.md`) |
| token looks garbled or binary | not a usable encoded JWT | don't use it; report it and get a valid Access Token |

Never disable TLS verification (`MLFLOW_TRACKING_INSECURE_TLS`, unverified SSL contexts)
unless a platform admin says the environment requires it.

## References

- [references/setup-and-auth.md](references/setup-and-auth.md): tokens, servers, workspaces, permissions
- [references/benchmark-logging.md](references/benchmark-logging.md): run layout and tag schema for benchmarks; benchmark-builder and card-eval-updater inputs
- [references/training.md](references/training.md): instrumentation, async logging, distributed rules, system metrics
- [references/model-registry.md](references/model-registry.md): registration, aliases, loaders, large models
- [references/hpc.md](references/hpc.md): per-site proxies, batch-job tokens, job snippets
- Upstream examples: AmSC `intro-to-mlflow-pytorch` and the BASE-Eval AXESS MLflow demo (ATTRIBUTION.md)
