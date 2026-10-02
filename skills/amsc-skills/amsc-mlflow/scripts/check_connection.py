#!/usr/bin/env python3
"""Check that this machine can log to the AmSC MLflow server, and say why not if it can't.

    python scripts/check_connection.py [--experiment NAME] [--no-write]

Adapted from the AmSC intro-to-mlflow-pytorch `test_mlflow_tracking.py`. It prints the
effective settings (never the token, only whether one is set and when it expires), then:

1. lists one experiment (read access);
2. unless --no-write: creates a run in a per-user experiment, logs a param, a metric and a
   tag, reads them back, and prints the run ID.

Exit codes: 0 ok, 2 no token, 3 token rejected (401), 4 no permission (403),
5 network/proxy failure, 1 anything else. Each failure prints what to do.
"""

from __future__ import annotations

import argparse
import getpass
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import amsc_mlflow  # noqa: E402  (sets the multipart variables before mlflow is imported)

HINTS = {
    2: "Copy the Access Token (not the ID Token) from MyAmSC Profile and export it without "
       "echoing: `read -s MLFLOW_TRACKING_TOKEN; export MLFLOW_TRACKING_TOKEN`.",
    3: "The server rejected the token (401): it's missing, expired, or not an Access Token. "
       "Get a fresh Access Token from MyAmSC Profile. If the token looks garbled or binary, "
       "don't use it; report the token-rendering problem.",
    4: "The token was accepted but this account lacks permission for the workspace or action "
       "(403). Selecting a workspace doesn't grant access: ask the workspace admin for the "
       "workspace user role (USE permission), or try a workspace you're authorized for "
       "(e.g. MLFLOW_WORKSPACE=default). Using an experiment another user created can also "
       "give 403: put your username in the experiment name.",
    5: "Couldn't reach the server. On an HPC compute node, set the site's proxy (Frontier: "
       "http_proxy/https_proxy=http://proxy.ccs.ornl.gov:3128; ALCF: the proxy in ALCF's "
       "docs; Perlmutter: none needed). See references/hpc.md.",
}


def classify(exc: Exception) -> int:
    text = f"{type(exc).__name__}: {exc}"
    low = text.lower()
    if "401" in text or "unauthorized" in low:
        return 3
    if "403" in text or "permission denied" in low or "forbidden" in low:
        return 4
    if any(s in low for s in ("connectionerror", "max retries", "timed out", "timeout",
                              "name or service not known", "connection refused",
                              "connection reset", "proxyerror", "remote end closed",
                              "temporary failure in name resolution")):
        return 5
    return 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiment", help="default: MLFLOW_EXPERIMENT_NAME, else "
                   "mlflow-connectivity-test-<user>")
    p.add_argument("--no-write", action="store_true", help="only check read access")
    args = p.parse_args(argv)

    amsc_mlflow.load_token()
    info = amsc_mlflow.describe()
    for k, v in info.items():
        print(f"{k + ':':18} {v}")
    left = amsc_mlflow.check_token_lifetime()
    if left is not None and left <= 0:
        print(f"FAILED: {HINTS[3]}", file=sys.stderr)
        return 3

    experiment = args.experiment or os.environ.get("MLFLOW_EXPERIMENT_NAME") or \
        f"mlflow-connectivity-test-{getpass.getuser()}"
    try:
        mlflow = amsc_mlflow.configure(require_token=True)
    except SystemExit as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 2

    from mlflow import MlflowClient

    try:
        client = MlflowClient()
        client.search_experiments(max_results=1)
        print("read:              ok")
        if args.no_write:
            return 0
        exp = mlflow.set_experiment(experiment)
        run_name = datetime.now(timezone.utc).strftime("connectivity-%Y%m%dT%H%M%SZ")
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_param("connection_test", "passed")
            mlflow.log_metric("test_value", 1.0)
            mlflow.set_tag("purpose", "AmSC MLflow connectivity test (amsc-mlflow skill)")
            run_id = run.info.run_id
        saved = client.get_run(run_id)
        assert saved.data.params["connection_test"] == "passed"
        assert saved.data.metrics["test_value"] == 1.0
    except Exception as exc:
        code = classify(exc)
        print(f"FAILED ({type(exc).__name__}): {str(exc)[:500]}", file=sys.stderr)
        if code in HINTS:
            print(f"What to do: {HINTS[code]}", file=sys.stderr)
        return code

    print("write:             ok")
    print(f"experiment:        {experiment} ({exp.experiment_id})")
    print(f"run id:            {run_id}")
    print("SUCCESS: the server accepted the test run and returned it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
