# genesis-mlflow: the `amsc-mlflow` agent skill

An [Agent Skill](https://agentskills.io) for using the **American Science Cloud (AmSC) /
Genesis Mission MLflow service** with the standard MLflow client:
- connecting with a MyAmSC Access Token;
- logging benchmark results;
- instrumenting training on HPC;
- registering and promoting models.

It follows the AmSC MLflow Integration Guide, generalizes the AmSC
`intro-to-mlflow-pytorch` examples and the BASE-Eval AXESS MLflow demo, and plugs into the
[benchmark-builder](https://github.com/ben-hawks/benchmark-builder) skill's results layout
and the genesis `card-eval-updater` eval bundle.

The skill is laid out as it would sit in
[AI-ModCon/genesis-skills](https://github.com/AI-ModCon/genesis-skills), under
`skills/amsc-skills/amsc-mlflow/`, and passes that repository's skill validator.

## Layout

```
skills/amsc-skills/amsc-mlflow/
├── SKILL.md                      the workflow the agent follows
├── ATTRIBUTION.md, LICENSE       authorship, sources, Apache-2.0
├── scripts/
│   ├── amsc_mlflow.py            configure(), token file + expiry, multipart uploads, rank 0, git/runtime tags
│   ├── check_connection.py       settings + read/write check, with a fix per failure (exit codes 0/2/3/4/5)
│   ├── log_benchmark_results.py  benchmark-builder results tree or eval bundle -> parent + per-(split, model) runs
│   ├── register_model.py         register a run's model or manifest-checked checkpoints; set/move aliases
│   └── manifest_pyfunc.py        generic pyfunc wrapper shipped with registered checkpoints
├── references/                   setup-and-auth, benchmark-logging, training, model-registry, hpc
└── assets/
    ├── env.example               every variable, documented
    ├── mlflow_job_env.sh         source in Slurm/PBS jobs: site proxy, token file, multipart
    └── train_with_mlflow.py      runnable PyTorch training template (rank 0, async, torchrun-ready)
tests/                            end-to-end tests against a local MLflow server (workspaces enabled)
```

## Install

Clone into your agent's skills directory, keeping only the skill directory:

```bash
git clone https://github.com/ben-hawks/genesis-mlflow
ln -s "$PWD/genesis-mlflow/skills/amsc-skills/amsc-mlflow" ~/.claude/skills/amsc-mlflow   # Claude Code
```

Python requirements: `mlflow>=3.13` (tested with 3.16.1). `train_with_mlflow.py` also
needs `torch`.

## Use

```bash
export MLFLOW_TRACKING_URI=https://mlflow-staging.american-science-cloud.org
export MLFLOW_WORKSPACE=modelservices
printf 'Paste Access Token: '; read -s MLFLOW_TRACKING_TOKEN; printf '\n'; export MLFLOW_TRACKING_TOKEN
python skills/amsc-skills/amsc-mlflow/scripts/check_connection.py
```

Then ask your agent to log a benchmark's results, instrument a training script, or register
a model. `SKILL.md` has the workflow.

## Test

```bash
python -m pip install -r requirements-dev.txt
python -m pytest tests -q
```

CI (`ci/github-actions-ci.yml`, move it to `.github/workflows/` to enable) runs these
tests plus the genesis-skills validator. The tests start a local `mlflow server` (SQLite backend, workspaces enabled, a
`modelservices` workspace) and run every script end to end against it. They don't contact
the AmSC server. `check_connection.py` against AmSC needs your own token, so run it
yourself.

## Merging into genesis-skills

Copy `skills/amsc-skills/amsc-mlflow/` into the catalog, then:
1. add `amsc-mlflow` to the root `README.md` tree (`amsc-skills` count and list) and to
   `NOTICE`, pointing at this skill's `ATTRIBUTION.md` and `LICENSE`;
2. mention it in `skills/amsc-skills/ATTRIBUTION.md` as a separately attributed skill (its
   authors differ from the AmSC Intelligent Interfaces Team's);
3. run `make verify-new-skill`;
4. keep `tests/` in this development repository; the catalog runs its own validator only.

## License

Apache-2.0 ([LICENSE](LICENSE)). See the skill's [ATTRIBUTION.md](skills/amsc-skills/amsc-mlflow/ATTRIBUTION.md)
for the sources it builds on.

## Acknowledgment

This work supports the U.S. Department of Energy Genesis Mission.
