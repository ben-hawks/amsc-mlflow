# Using AmSC MLflow from HPC jobs

Source: the integration guide ("HPC Network Configuration", "Large Model Uploads"), plus
site documentation where the guide is incomplete (marked). Site policies change, so check
the site's current docs and, for the scheduler and site facts, the genesis HPC skills
(`slurm`, `pbs`, `perlmutter`, `frontier`, `aurora`).

## Per-site network settings

| Site | Compute-node access to the AmSC server | Settings | Status |
|---|---|---|---|
| NERSC Perlmutter | outbound allowed | none | from the guide |
| OLCF Frontier | through the OLCF proxy | `export http_proxy=http://proxy.ccs.ornl.gov:3128 https_proxy=http://proxy.ccs.ornl.gov:3128 no_proxy=localhost,127.0.0.1,*.ccs.ornl.gov` | the guide gives `http_proxy` only. MLflow talks HTTPS, so `https_proxy` is needed too. Confirm the values in OLCF's current docs. **Unverified** by a run with this skill |
| ALCF (Polaris, Aurora) | through the ALCF proxy | `export http_proxy=http://proxy.alcf.anl.gov:3128 https_proxy=http://proxy.alcf.anl.gov:3128` | the guide's line is malformed (`export https://docs.alcf.anl.gov/polaris/getting-started/#proxy`); it points at that page. The values here are ALCF's usual proxy; confirm them on that page. **Unverified** by a run with this skill |

`assets/mlflow_job_env.sh` applies these by `AMSC_SITE` (or guesses the site from the
hostname), sets the MLflow variables, and reads the token from a file. Source it in job
scripts before Python starts. After using it on a site, record what worked in this table
(with the date) and drop "unverified".

## The token in batch jobs

A job may start hours after submission, and the Access Token expires.

1. **Prefer logging after the job.** `log_benchmark_results.py` only reads result files,
   so run it from a login node once the job has finished. No token sits in the queue, and
   an expired token costs nothing.
2. **When the job itself must log** (live training metrics, registering from the job):
   - write the token to a file only you can read, just before submitting:
     ```bash
     install -m 600 /dev/null ~/.amsc/mlflow_token    # creates it chmod 600
     printf 'Paste Access Token: '; read -s t; printf '%s' "$t" > ~/.amsc/mlflow_token; unset t
     ```
   - point jobs at it: `export MLFLOW_TRACKING_TOKEN_FILE=~/.amsc/mlflow_token` (the
     scripts and `amsc_mlflow.configure()` read it);
   - at job start, check the token outlives the job:
     `amsc_mlflow.check_token_lifetime(min_seconds=<wall time>)` warns, and
     `check_connection.py --no-write` fails fast on an expired token;
   - delete the file when done.

   Don't pass the token with `sbatch --export=MLFLOW_TRACKING_TOKEN=...` or
   `qsub -v MLFLOW_TRACKING_TOKEN=...`: it lands in the scheduler's job record. (`sbatch
   --export=ALL` copies an exported token silently; the file route makes the choice
   explicit.)
3. **If the token expires mid-run**, logging calls fail with 401. Make the training code
   tolerate that (keep training, write metrics locally) rather than crash the job, and log
   the local record afterwards.

## Large artifacts

`amsc_mlflow` turns on proxy multipart uploads before `mlflow` is imported
(`MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD=true`, 5 MB threshold, 100 MB chunks). In a job
script that runs code which imports `mlflow` itself, export them in the job script
(`assets/mlflow_job_env.sh` does). Upload big checkpoints once, at the end, from rank 0.
For artifacts too large to upload, see "Artifacts on shared storage" in
`references/model-registry.md`.

## Distributed jobs

Only rank 0 logs, after reducing metrics across ranks (`references/training.md`).
Launchers differ by site: `srun` on Slurm sites, `mpiexec` on Aurora/Polaris (PBS).
`amsc_mlflow.global_rank()` reads each one's rank variable.

## benchmark-builder job templates

benchmark-builder's generated `<hpc>/` directory (Slurm or PBS) runs `score_all.sh` as its
last job. To log from the cluster:
- source `mlflow_job_env.sh` in that job (or in `stack.sh`'s `bench_stack_env`);
- run `log_benchmark_results.py` after `score_all.sh`.

Or skip the cluster entirely and log from the login node afterwards (preferred).
