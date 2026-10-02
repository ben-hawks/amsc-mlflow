# Attribution

**Skill:** `amsc-mlflow`

**Author:** Ben Hawks. Developed in the `ben-hawks/genesis-mlflow` repository, laid out as
`skills/amsc-skills/amsc-mlflow/` so it can be proposed to
[AI-ModCon/genesis-skills](https://github.com/AI-ModCon/genesis-skills).

**License:** Apache-2.0 (`LICENSE` in this directory).

## Sources this skill builds on

- **AmSC MLflow Integration Guide**, American Science Cloud Model Services: the
  token, server, workspace, permission, HPC proxy, asynchronous logging, multipart upload,
  distributed-logging and model-catalog guidance. The skill follows the guide where it is
  explicit, and marks where it fills a gap (the Frontier `https_proxy`, the ALCF proxy
  values) as unverified.
- **`intro-to-mlflow-pytorch`**, AmSC Model Services
  (`gitlab.com/amsc2/ai-services/model-services/intro-to-mlflow-pytorch`): the
  connectivity test (`check_connection.py` is adapted from its `test_mlflow_tracking.py`),
  token handling, workspace selection, experiment tracking, model registry and alias
  patterns, the sweep layout, and the W&B migration table. Prompt-tracing example by
  Huihuo Zheng (ANL). No license file accompanied the copy used here. The skill
  re-implements the patterns rather than copying files, apart from the adapted
  connectivity test.
- **BASE-Eval AXESS MLflow demonstration**, ModCon Base Team
  (`AI-ModCon/BaseEval_AXESS_benchmark_demo`, Apache-2.0): registering benchmark model
  wrappers as pyfunc models with provenance tags on registered models and versions, and
  evaluating registered models with one run per split/predictor/architecture and system
  metrics. `log_benchmark_results.py` and `register_model.py` generalize those patterns.

## Differences from the sources

- The default server is the staging server named by the integration guide
  (`https://mlflow-staging.american-science-cloud.org`). The examples use the
  development server.
- No global TLS-verification bypass (`ssl._create_unverified_context`), which several
  example scripts set for dataset downloads.
- Registered pyfunc models upload their weight files (sha256-checked against a manifest)
  instead of pointing at a local checkout path.
