"""Shared helpers for using the American Science Cloud (AmSC) MLflow service.

Import this module instead of `mlflow` directly at the top of a script:

    import amsc_mlflow
    mlflow = amsc_mlflow.configure(experiment="my-benchmark")

It applies the AmSC MLflow Integration Guide's conventions:

- the tracking server comes from MLFLOW_TRACKING_URI (default: the AmSC staging server);
- authentication is MLflow's standard MLFLOW_TRACKING_TOKEN (a MyAmSC Access Token, sent as
  a bearer token by the MLflow client). The token is never printed. It can also be read from
  a file named by MLFLOW_TRACKING_TOKEN_FILE, which is how batch jobs get it without
  putting it on a command line;
- the workspace comes from MLFLOW_WORKSPACE and is selected with mlflow.set_workspace()
  when the client has it (MLflow >= 3.13);
- proxy multipart uploads for large artifacts are switched on **before mlflow is
  imported** (they're read at import time). Set AMSC_MLFLOW_MULTIPART=0 to leave them off.

Also: rank-0 detection for distributed jobs, git/runtime tags, and decoding a token's
expiry time (without verifying it) so a job can tell whether the token will outlive it.

Standard library only, apart from mlflow itself, which is imported lazily by configure().
"""

from __future__ import annotations

import base64
import contextlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_TRACKING_URI = "https://mlflow-staging.american-science-cloud.org"
DEFAULT_WORKSPACE = "modelservices"

MULTIPART_DEFAULTS = {
    "MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD": "true",
    "MLFLOW_MULTIPART_UPLOAD_MINIMUM_FILE_SIZE": "5242880",  # 5 MB: smaller files upload whole
    "MLFLOW_MULTIPART_UPLOAD_CHUNK_SIZE": "104857600",  # 100 MB per part
}


def enable_large_uploads() -> None:
    """Set the AmSC proxy multipart-upload variables unless the caller already set them.
    Must run before `import mlflow`; this module calls it at import time."""
    if os.environ.get("AMSC_MLFLOW_MULTIPART", "1") == "0":
        return
    for key, value in MULTIPART_DEFAULTS.items():
        os.environ.setdefault(key, value)


if "mlflow" in sys.modules and os.environ.get("AMSC_MLFLOW_MULTIPART", "1") != "0":
    print("amsc_mlflow: warning: mlflow was imported before amsc_mlflow, so the multipart "
          "upload settings may not apply; import amsc_mlflow first", file=sys.stderr)
enable_large_uploads()


# --------------------------------------------------------------------------- token


def is_remote(uri: str) -> bool:
    return uri.startswith(("http://", "https://"))


def tracking_uri() -> str:
    return os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI).rstrip("/")


def load_token() -> bool:
    """Make MLFLOW_TRACKING_TOKEN available, from the environment or from the file named
    by MLFLOW_TRACKING_TOKEN_FILE. Returns whether a token is set. Never prints it."""
    if os.environ.get("MLFLOW_TRACKING_TOKEN", "").strip():
        return True
    path = os.environ.get("MLFLOW_TRACKING_TOKEN_FILE")
    if path and Path(path).is_file():
        token = Path(path).read_text(encoding="utf-8").strip()
        if token:
            mode = Path(path).stat().st_mode & 0o077
            if mode:
                print(f"amsc_mlflow: warning: {path} is readable by group/others; "
                      "chmod 600 it", file=sys.stderr)
            os.environ["MLFLOW_TRACKING_TOKEN"] = token
            return True
    return False


def token_expiry(token: str | None = None) -> datetime | None:
    """The `exp` claim of a JWT access token, decoded WITHOUT verifying the signature (only
    to warn about expiry). None if the token isn't a JWT or has no exp."""
    token = token if token is not None else os.environ.get("MLFLOW_TRACKING_TOKEN", "")
    parts = token.strip().split(".")
    if len(parts) != 3:
        return None
    try:
        payload = parts[1] + "=" * (-len(parts[1]) % 4)
        claims = json.loads(base64.urlsafe_b64decode(payload))
        return datetime.fromtimestamp(int(claims["exp"]), tz=timezone.utc)
    except Exception:
        return None


def check_token_lifetime(min_seconds: float = 0) -> float | None:
    """Seconds until the token expires (negative if expired), or None if unknown. Warns
    when it expires within min_seconds (e.g. the job's wall time)."""
    exp = token_expiry()
    if exp is None:
        return None
    left = exp.timestamp() - time.time()
    if left <= 0:
        print(f"amsc_mlflow: the MLflow token expired at {exp.isoformat()}; get a new "
              "Access Token from MyAmSC Profile", file=sys.stderr)
    elif left < min_seconds:
        print(f"amsc_mlflow: warning: the MLflow token expires at {exp.isoformat()}, in "
              f"{left / 60:.0f} min, before this job's {min_seconds / 60:.0f} min end; log "
              "afterwards from a login node instead", file=sys.stderr)
    return left


# --------------------------------------------------------------------------- setup


def configure(experiment: str | None = None, workspace: str | None = None,
              require_token: bool = True, async_logging: bool = False):
    """Point the mlflow client at the AmSC server and return the `mlflow` module.

    - experiment: name to select (created if missing). Default: MLFLOW_EXPERIMENT_NAME, else
      no experiment is selected. Per the integration guide, include your username in
      personal experiment names: an experiment created by another user can return 403 when
      you try to create runs in it.
    - workspace: default MLFLOW_WORKSPACE, else "modelservices" on the AmSC server and none
      elsewhere. MLFLOW_WORKSPACE=none selects no workspace (the server's default).
      Selecting a workspace doesn't grant access to it; a 403 means you need the workspace
      role.
    - require_token: fail before any request when a remote server has no token. Local
      tracking URIs (file:, sqlite:, a path) never need one.
    - async_logging: mlflow.config.enable_async_logging(); call finish() before exit.
    """
    uri = tracking_uri()
    if is_remote(uri) and require_token and not load_token():
        raise SystemExit("amsc_mlflow: MLFLOW_TRACKING_TOKEN is not set. Copy the Access Token "
                         "(not the ID Token) from MyAmSC Profile and export it without "
                         "echoing it, e.g. `read -s MLFLOW_TRACKING_TOKEN; export "
                         "MLFLOW_TRACKING_TOKEN`, or point MLFLOW_TRACKING_TOKEN_FILE at a "
                         "chmod-600 file.")
    import mlflow

    mlflow.set_tracking_uri(uri)
    ws = workspace or effective_workspace()
    if ws and is_remote(uri):
        if hasattr(mlflow, "set_workspace"):
            mlflow.set_workspace(ws)
        else:
            print(f"amsc_mlflow: warning: mlflow {mlflow.__version__} has no set_workspace(); "
                  "runs go to the server's default workspace. Use mlflow >= 3.13.",
                  file=sys.stderr)
    name = experiment or os.environ.get("MLFLOW_EXPERIMENT_NAME")
    if name:
        mlflow.set_experiment(name)
    if async_logging:
        mlflow.config.enable_async_logging()
    return mlflow


def effective_workspace() -> str | None:
    """MLFLOW_WORKSPACE ("none" for no workspace), else "modelservices" when talking to the
    AmSC server, else None."""
    ws = os.environ.get("MLFLOW_WORKSPACE", "").strip()
    if ws.lower() == "none":
        return None
    if ws:
        return ws
    return DEFAULT_WORKSPACE if "american-science-cloud.org" in tracking_uri() else None


def finish() -> None:
    """Flush pending asynchronous logging. Call before a job or rank-0 process exits."""
    import mlflow

    with contextlib.suppress(Exception):
        mlflow.flush_async_logging()


def describe() -> dict:
    """The effective settings, safe to print (the token is reported as set/unset only)."""
    exp = token_expiry()
    return {
        "tracking_uri": tracking_uri(),
        "workspace": effective_workspace(),
        "experiment": os.environ.get("MLFLOW_EXPERIMENT_NAME"),
        "token": "set" if os.environ.get("MLFLOW_TRACKING_TOKEN", "").strip() else "not set",
        "token_expires": exp.isoformat() if exp else None,
        "multipart_upload": os.environ.get("MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD", "false"),
        "https_proxy": os.environ.get("https_proxy") or os.environ.get("HTTPS_PROXY"),
    }


# --------------------------------------------------------------------------- distributed


RANK_VARIABLES = ("RANK", "SLURM_PROCID", "PMI_RANK", "PMIX_RANK", "OMPI_COMM_WORLD_RANK",
                  "PALS_RANKID", "MPI_RANKID")


def global_rank() -> int:
    """This process's global rank: torch.distributed if initialized, else the launcher's
    environment (torchrun RANK, Slurm, MPICH/PMI, Open MPI, PALS on Aurora). 0 if none."""
    torch = sys.modules.get("torch")
    if torch is not None:
        with contextlib.suppress(Exception):
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                return int(torch.distributed.get_rank())
    for var in RANK_VARIABLES:
        value = os.environ.get(var)
        if value is not None and value.strip().isdigit():
            return int(value)
    return 0


def is_rank_zero() -> bool:
    return global_rank() == 0


@contextlib.contextmanager
def rank_zero_run(**start_run_kwargs):
    """`with rank_zero_run(run_name=...) as run:` starts an MLflow run on rank 0 only and
    yields None elsewhere, so every MLflow call can sit under `if run:`. Reduce metrics
    across ranks *before* logging them on rank 0. Flushes async logging on exit."""
    if not is_rank_zero():
        yield None
        return
    import mlflow

    with mlflow.start_run(**start_run_kwargs) as run:
        try:
            yield run
        finally:
            finish()


# --------------------------------------------------------------------------- tags


def git_tags(path: str | os.PathLike = ".") -> dict[str, str]:
    """git.commit, git.branch, git.dirty and git.remote for the checkout at path."""
    def git(*args: str) -> str:
        return subprocess.check_output(["git", "-C", str(path), *args],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    try:
        tags = {"git.commit": git("rev-parse", "HEAD"),
                "git.branch": git("rev-parse", "--abbrev-ref", "HEAD"),
                "git.dirty": "true" if git("status", "--porcelain") else "false"}
    except Exception:
        return {"git.commit": "unknown"}
    with contextlib.suppress(Exception):
        tags["git.remote"] = git("config", "--get", "remote.origin.url")
    return tags


def runtime_tags(*modules: str) -> dict[str, str]:
    """runtime.python, runtime.mlflow, runtime.<module> for each named module, and the
    host and HPC machine (BENCH_MACHINE, NERSC_HOST, LMOD_SYSTEM_NAME)."""
    import importlib

    tags = {"runtime.python": platform.python_version(), "runtime.host": platform.node()}
    for name in ("mlflow", *modules):
        try:
            tags[f"runtime.{name}"] = str(getattr(importlib.import_module(name), "__version__", "unknown"))
        except Exception:
            tags[f"runtime.{name}"] = "unavailable"
    machine = os.environ.get("BENCH_MACHINE") or os.environ.get("NERSC_HOST") or os.environ.get("LMOD_SYSTEM_NAME")
    if machine:
        tags["runtime.machine"] = machine
    for var in ("SLURM_JOB_ID", "PBS_JOBID"):
        if os.environ.get(var):
            tags["runtime.job_id"] = os.environ[var]
    return tags
