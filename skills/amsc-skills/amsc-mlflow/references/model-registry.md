# Model Registry: registering, promoting and loading models

Source: the integration guide ("Core Point 2: The Model Catalog"),
`intro-to-mlflow-pytorch/model_registry/`, and the AXESS demo's
`register_rule4ml_models.py`.

## Concepts

- **Registered model.** A name (e.g. `Peregrine`, `rule4ml-gnn-v2`) with numbered
  **versions**. Each version points at a logged MLflow model.
- **Aliases.** Movable pointers to a version. The guide's lifecycle: `staging` for AmSC
  testing, `production` once verified. Consumers load by alias
  (`models:/<name>@production`), so promotion doesn't change their code. Older
  "stages" are deprecated in MLflow; use aliases.
- **Tags.** Both registered models and versions carry tags. The AXESS demo copies its
  provenance tags (weights family and version, source path, runtime versions) onto both.
  `register_model.py` does the same with the manifest's provenance and per-file sha256.

## Three ways to get a version

| Situation | How | Script |
|---|---|---|
| the training code logged the model (`mlflow.pytorch.log_model(...)`) | register its URI `runs:/<run_id>/model` | `register_model.py run` |
| checkpoint files exist outside MLflow (a release asset, a benchmark's `weights/`) | log them as artifacts of a pyfunc model with a loader, then register | `register_model.py files` |
| logging and registering in one call during training | `log_model(..., registered_model_name=...)` | (in the training code) |

### `files`: checkpoints from a benchmark's weights manifest

benchmark-builder benchmarks list their checkpoints in `weights/MANIFEST.json` (file →
url, sha256, bytes, role, model description). `register_model.py files`:

1. checks each requested file is listed, and that its sha256 matches **before** anything
   uploads;
2. logs a run whose model is the generic `ManifestModel` pyfunc (`scripts/manifest_pyfunc.py`,
   shipped inside the model) with the files as artifacts;
3. registers a new version, tagged `weights.sha256.<file>`, `weights.source`,
   `weights.source_commit`, `model.role`, `benchmark.name`, `git.*`, `runtime.*`, with the
   manifest's model description as the version description;
4. sets the requested aliases.

**The loader contract.** `--loader pkg.module:function` names a function in the benchmark's
own package. It takes `{file name: local path}` and returns an object with
`.predict(model_input)` (or a callable):

```python
# src/<pkg>/mlflow_loader.py
def load(artifacts):
    model = MyArchitecture(**HPARAMS)
    model.load_state_dict(torch.load(artifacts["model_final.pt"], map_location="cpu"), strict=True)
    stats = json.load(open(artifacts["normalization_stats.json"]))
    return Predictor(model, stats)          # .predict(DataFrame) -> DataFrame
```

Ship the package with `--code-path src/<pkg>`, or install it wherever the model is
loaded, and list its runtime dependencies with `--pip-requirement` so `mlflow` can rebuild
the environment. Load it back:

```python
model = mlflow.pyfunc.load_model("models:/<benchmark>-<model>@production")
preds = model.predict(df)
predictor = model.unwrap_python_model().model     # the loader's object, for non-DataFrame use
```

The AXESS demo uses the same idea with model-specific wrappers (`Rule4MLGNNWrapper`,
`Rule4MLMLPWrapper`). The demo's wrappers point at a local `rule4ml` checkout path and
load its bundled weights, so the registered model only loads on a machine with that
checkout. `ManifestModel` uploads the weight files themselves, so a registered version is
self-contained apart from the loader's package.

### When to use a native flavor instead

If the model is a plain PyTorch/Keras/sklearn model, log it with its flavor
(`mlflow.pytorch.log_model(model, name="model", signature=..., input_example=...)`) and
register with `register_model.py run`. You get signature validation and
`mlflow.pytorch.load_model()` without a loader. Use `files` when the inference procedure
needs more than the weights: preprocessing, derived statistics, post-processing,
several heads. benchmark-builder's reference solutions usually do.

## Promotion

```bash
python scripts/register_model.py alias --name <name> --from-alias staging --alias production
python scripts/register_model.py alias --name <name> --version 7 --alias production
```

Promote only after the version passed the benchmark's checks: the golden tests, and a
scoring run that reproduces `reference_results/`. Log that run with
`log_benchmark_results.py` and put its run ID in the version description. Moving
`production` changes what every consumer loads, so confirm with the user each time.

## Large models

- **Proxy multipart upload** is on by default via `amsc_mlflow`:
  `MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD=true`, 5 MB threshold, 100 MB chunks. Uploads
  through the AmSC gateway go in parts. Downloads (`load_model`, `download_artifacts`)
  reassemble them with no configuration. Uploads still block until done.
- **Artifacts on shared storage ("zero copy").** For checkpoints too large to upload, the
  AmSC examples (`relocate_artifact_location.py`) create an experiment whose
  `artifact_location` is a `file://` path on a shared filesystem (e.g. Lustre). Metadata
  goes to the server; files stay on site. Such models only load on machines that mount
  that path, and purge policies apply (Frontier `$MEMBERWORK`: 90 days). Use a
  non-purged project area. Prefer uploading when the size allows: it's what makes the
  registry the single source of truth.
