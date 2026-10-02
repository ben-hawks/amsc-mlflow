#!/usr/bin/env python3
"""Log a benchmark's scored results to MLflow: one parent run per invocation, one child
run per (model, split).

    # a benchmark-builder results tree: <results>/<split>/<model>/metrics.json
    python scripts/log_benchmark_results.py --benchmark <name> --results reference_results \\
        [--models m1 m2] [--splits test exemplar] [--auxiliary m3] [--all-groups] \\
        [--repo .] [--dataset-revision <rev>] [--log-predictions] [--dry-run]

    # an LLM evaluation, via the eval bundle the genesis card-eval-updater skill writes
    python scripts/log_benchmark_results.py --benchmark <name> --bundle bundle.json [--split test]

Numbers are copied from the scoring output, never recomputed, and each child run records
the sha256 of the file it came from (tag `benchmark.source_sha256`), so a run traces to a
specific metrics.json. With --skip-existing (default on), a (model, split) whose source
sha256 already has a FINISHED run in the experiment is skipped, so re-running after a
partial failure doesn't duplicate runs.

Run layout (references/benchmark-logging.md):
- experiment: --experiment, else MLFLOW_EXPERIMENT_NAME, else "<benchmark>-<user>";
- parent run "<benchmark> results <UTC time>": tags benchmark.*, git.*, runtime.*;
  artifacts LEADERBOARD.md (if present);
- child run "<split>/<model>": params model, split, group; metrics
  "<output>.<metric>" from group "all" (and "<group>/<output>.<metric>" for other groups
  with --all-groups), "coverage.<key>"; artifacts: the model's metrics.json, METRICS.md,
  plots, the predictions .meta.json, and with --log-predictions the predictions file.

Non-finite values (NaN/inf) are logged as-is to MLflow but also listed in the tag
`benchmark.nonfinite_metrics`, since the MLflow UI shows them inconsistently.

--dry-run prints the runs, params, metrics and artifacts it would log and touches no
server (no token needed).
"""

from __future__ import annotations

import argparse
import getpass
import glob
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import amsc_mlflow  # noqa: E402

KEY_OK = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-. /")


def clean_key(key: str) -> str:
    """MLflow keys allow alphanumerics, _ - . space and /. Replace anything else."""
    return "".join(c if c in KEY_OK else "_" for c in str(key))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


# --------------------------------------------------------------------------- sources


def flatten_metrics(m: dict, all_groups: bool) -> dict[str, float]:
    """metrics.json (benchmark-builder: {"coverage": {...}, "groups": {group: {output:
    {metric: v}}}}, or {group: {metric: v}} for one output) -> flat MLflow metrics."""
    out: dict[str, float] = {}
    groups = m.get("groups", {k: v for k, v in m.items() if k != "coverage"})
    for group, body in groups.items():
        if group != "all" and not all_groups:
            continue
        prefix = "" if group == "all" else f"{group}/"
        if not isinstance(body, dict):
            continue
        for key, val in body.items():
            if isinstance(val, dict):
                for met, v in val.items():
                    if is_number(v):
                        out[clean_key(f"{prefix}{key}.{met}")] = float(v)
            elif is_number(val):
                out[clean_key(f"{prefix}{key}")] = float(val)
    for k, v in (m.get("coverage") or {}).items():
        if is_number(v):
            out[clean_key(f"coverage.{k}")] = float(v)
    return out


def from_results(results: Path, models, splits, all_groups, log_predictions):
    entries = []
    for path in sorted(results.glob("*/*/metrics.json")):
        split, model = path.parent.parent.name, path.parent.name
        if (splits and split not in splits) or (models and model not in models):
            continue
        with open(path) as f:
            metrics = flatten_metrics(json.load(f), all_groups)
        arts = [path] + [p for p in sorted(path.parent.iterdir())
                         if p.is_file() and p.name != "metrics.json"
                         and p.suffix.lower() in (".md", ".png", ".svg", ".pdf", ".json")]
        meta = path.parent.parent / f"predictions_{model}.meta.json"
        params = {"model": model, "split": split}
        if meta.exists():
            arts.append(meta)
            with open(meta) as f:
                for k, v in json.load(f).items():
                    if k in ("model_id", "model_args", "task", "task_version", "n_shot",
                             "lm_eval_version", "chat_template_sha", "output_type"):
                        params[k] = json.dumps(v) if isinstance(v, (dict, list)) else v
        if log_predictions:
            arts += [Path(p) for p in glob.glob(str(path.parent.parent / f"predictions_{model}.*"))
                     if not p.endswith(".meta.json")]
        entries.append({"model": model, "split": split, "source": path, "metrics": metrics,
                        "params": params, "artifacts": arts})
    if not entries:
        raise SystemExit(f"log_benchmark_results: no <split>/<model>/metrics.json under {results}"
                         + (f" for models {models}" if models else "")
                         + (f" splits {splits}" if splits else ""))
    return entries


def from_bundle(bundle_path: Path, split: str):
    """A card-eval-updater eval bundle (one harness run) -> one entry."""
    with open(bundle_path) as f:
        b = json.load(f)
    model = (b.get("model") or {}).get("id") or "unknown-model"
    metrics: dict[str, float] = {}
    partial = False
    for r in b.get("results", []):
        name = r["benchmark"] + (f"/{r['variant']}" if r.get("variant") else "")
        metrics[clean_key(f"{name}.{r['metric']}")] = float(r["value"])
        if is_number(r.get("stderr")):
            metrics[clean_key(f"{name}.{r['metric']}_stderr")] = float(r["stderr"])
        if is_number(r.get("num_samples")):
            metrics[clean_key(f"{name}.num_samples")] = float(r["num_samples"])
        partial = partial or bool(r.get("is_partial"))
    if partial:
        raise SystemExit("log_benchmark_results: the bundle holds a partial (sample-limited) run; "
                         "re-run the evaluation without a limit before logging it")
    run = b.get("run") or {}
    m = b.get("model") or {}
    params = {"model": model, "split": split, "harness": run.get("harness"),
              "harness_version": run.get("harness_version"),
              "model_id_provenance": m.get("id_provenance"), "dtype": m.get("dtype"),
              "num_parameters": m.get("num_parameters")}
    return [{"model": model, "split": split, "source": bundle_path, "metrics": metrics,
             "params": {k: v for k, v in params.items() if v is not None},
             "artifacts": [bundle_path]}]


# --------------------------------------------------------------------------- logging


def plan(args, entries):
    user = getpass.getuser()
    experiment = args.experiment or os.environ.get("MLFLOW_EXPERIMENT_NAME") or f"{args.benchmark}-{user}"
    common = {"benchmark.name": args.benchmark, "owner": user,
              **{k: v for k, v in amsc_mlflow.git_tags(args.repo).items()},
              **amsc_mlflow.runtime_tags()}
    if args.dataset_revision:
        common["benchmark.dataset_revision"] = args.dataset_revision
    for e in entries:
        e["sha256"] = sha256(e["source"])
        e["role"] = "auxiliary" if e["model"] in (args.auxiliary or []) else "reference"
        nonfinite = sorted(k for k, v in e["metrics"].items() if not math.isfinite(v))
        e["tags"] = {**common, "benchmark.split": e["split"], "benchmark.model": e["model"],
                     "benchmark.model_role": e["role"], "benchmark.source": str(e["source"]),
                     "benchmark.source_sha256": e["sha256"]}
        if nonfinite:
            e["tags"]["benchmark.nonfinite_metrics"] = ",".join(nonfinite)[:5000]
    return experiment, common


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--benchmark", required=True)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--results", type=Path, help="<results> holding <split>/<model>/metrics.json")
    src.add_argument("--bundle", type=Path, help="card-eval-updater eval bundle (bundle.json)")
    p.add_argument("--split", default="test", help="split label for --bundle")
    p.add_argument("--models", nargs="*")
    p.add_argument("--splits", nargs="*")
    p.add_argument("--auxiliary", nargs="*", help="models to tag benchmark.model_role=auxiliary")
    p.add_argument("--all-groups", action="store_true", help="also log per-group metrics")
    p.add_argument("--log-predictions", action="store_true", help="also upload predictions files")
    p.add_argument("--repo", default=".", help="benchmark checkout, for git tags")
    p.add_argument("--dataset-revision")
    p.add_argument("--experiment")
    p.add_argument("--run-name")
    p.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    entries = (from_results(args.results, args.models, args.splits, args.all_groups, args.log_predictions)
               if args.results else from_bundle(args.bundle, args.split))
    experiment, common = plan(args, entries)
    leaderboard = args.results / "LEADERBOARD.md" if args.results else None
    parent_name = args.run_name or f"{args.benchmark} results " + \
        datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")

    if args.dry_run:
        print(f"experiment: {experiment}\nparent run: {parent_name}")
        for k, v in sorted(common.items()):
            print(f"  tag {k} = {v}")
        if leaderboard and leaderboard.exists():
            print(f"  artifact {leaderboard}")
        for e in entries:
            print(f"child run: {e['split']}/{e['model']} ({e['role']}, source sha256 {e['sha256'][:12]})")
            for k, v in e["params"].items():
                print(f"  param {k} = {v}")
            for k, v in sorted(e["metrics"].items()):
                print(f"  metric {k} = {v}")
            for a in e["artifacts"]:
                print(f"  artifact {a}")
        return 0

    mlflow = amsc_mlflow.configure(experiment=experiment)
    from mlflow import MlflowClient

    client = MlflowClient()
    exp = mlflow.get_experiment_by_name(experiment)
    todo = []
    for e in entries:
        if args.skip_existing and exp is not None:
            found = client.search_runs(
                [exp.experiment_id],
                filter_string=f"tags.`benchmark.source_sha256` = '{e['sha256']}' "
                              f"and attributes.status = 'FINISHED'",
                max_results=1)
            if found:
                print(f"skip {e['split']}/{e['model']}: already logged as run {found[0].info.run_id}")
                continue
        todo.append(e)
    if not todo:
        print("nothing new to log")
        return 0

    with mlflow.start_run(run_name=parent_name, tags=common) as parent:
        if leaderboard and leaderboard.exists():
            mlflow.log_artifact(str(leaderboard))
        for e in todo:
            with mlflow.start_run(run_name=f"{e['split']}/{e['model']}", nested=True, tags=e["tags"]) as child:
                mlflow.log_params({k: str(v)[:6000] for k, v in e["params"].items()})
                mlflow.log_metrics(e["metrics"])
                for a in e["artifacts"]:
                    mlflow.log_artifact(str(a), artifact_path="results")
                print(f"logged {e['split']}/{e['model']}: run {child.info.run_id} "
                      f"({len(e['metrics'])} metrics)")
    print(f"parent run {parent.info.run_id} in experiment '{experiment}' at {amsc_mlflow.tracking_uri()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
