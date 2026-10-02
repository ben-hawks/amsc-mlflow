#!/usr/bin/env python3
"""Register models in the AmSC MLflow Model Registry and manage their aliases.

    # 1. a model already logged in a run (mlflow.<flavor>.log_model)
    python scripts/register_model.py run --model-uri runs:/<run_id>/model --name <name> [--alias staging]

    # 2. checkpoint files listed in a benchmark's weights/MANIFEST.json, as a pyfunc model
    python scripts/register_model.py files --manifest weights/MANIFEST.json --weights-dir $BENCH_WEIGHTS \\
        --files model_a_final.pt [stats.json ...] --name <benchmark>-<model> \\
        --loader <pkg>.mlflow_loader:load [--code-path src/<pkg>] [--pip-requirement ...] \\
        [--benchmark <name>] [--alias staging]

    # 3. move an alias (e.g. promote after verification)
    python scripts/register_model.py alias --name <name> --version 3 --alias production
    python scripts/register_model.py alias --name <name> --from-alias staging --alias production

`files` verifies every file's sha256 against the manifest **before** uploading and refuses
on a mismatch or a file the manifest doesn't list. It logs a run in the current experiment
holding the files and the generic ManifestModel wrapper (manifest_pyfunc.py, shipped with
the model), registers a new version, and tags the registered model and the version with
the manifest's provenance (source, source_commit, role, per-file sha256, the model
description) plus git and runtime tags. Large checkpoints upload through the AmSC proxy
multipart path (amsc_mlflow sets it up).

Aliases follow the integration guide: register as `staging` for AmSC testing, then promote
to `production` once verified. Consumers load "models:/<name>@production".
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import amsc_mlflow  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def tag_safe(value) -> str:
    return str(value)[:5000]


def set_aliases(client, name: str, version: str, aliases) -> None:
    for alias in aliases or []:
        client.set_registered_model_alias(name, alias, version)
        print(f"alias {name}@{alias} -> version {version}")


def cmd_run(args, mlflow, client) -> int:
    mv = mlflow.register_model(args.model_uri, args.name)
    tags = {**amsc_mlflow.runtime_tags(), **({"benchmark.name": args.benchmark} if args.benchmark else {})}
    for k, v in tags.items():
        client.set_model_version_tag(args.name, mv.version, k, tag_safe(v))
    if args.description:
        client.update_model_version(args.name, mv.version, description=args.description)
    print(f"registered {args.model_uri} as {args.name} version {mv.version}")
    set_aliases(client, args.name, mv.version, args.alias)
    return 0


def verify_files(manifest: dict, weights_dir: Path, files: list[str]) -> dict[str, dict]:
    listed = manifest.get("files", {})
    out, errors = {}, []
    for name in files:
        if name not in listed:
            errors.append(f"{name}: not listed in the manifest")
            continue
        path = weights_dir / name
        if not path.is_file():
            errors.append(f"{name}: missing at {path}")
            continue
        want = listed[name].get("sha256")
        got = sha256(path)
        if want and got != want:
            errors.append(f"{name}: sha256 {got} != manifest {want}")
            continue
        out[name] = {**listed[name], "path": str(path), "sha256": got}
    if errors:
        raise SystemExit("register_model: refusing to upload:\n  " + "\n  ".join(errors))
    return out


def cmd_files(args, mlflow, client) -> int:
    with open(args.manifest) as f:
        manifest = json.load(f)
    files = verify_files(manifest, Path(args.weights_dir), args.files)
    roles = {e.get("role", "reference") for e in files.values()}
    from manifest_pyfunc import ManifestModel

    tags = {
        "weights.source": manifest.get("source", "unknown"),
        "weights.source_commit": manifest.get("source_commit", "unknown"),
        "model.role": ",".join(sorted(roles)),
        **{f"weights.sha256.{n}": e["sha256"] for n, e in files.items()},
        **amsc_mlflow.git_tags(args.repo),
        **amsc_mlflow.runtime_tags(*args.runtime_module),
    }
    if args.benchmark:
        tags["benchmark.name"] = args.benchmark
    description = args.description or "\n".join(
        f"{n}: {e.get('model', '')}" for n, e in files.items()).strip()

    code_paths = [str(HERE / "manifest_pyfunc.py"), *args.code_path]
    pip = args.pip_requirement or [f"mlflow=={mlflow.__version__}"]
    with mlflow.start_run(run_name=f"register-{args.name}", tags=tags) as run:
        info = mlflow.pyfunc.log_model(
            name="model",
            python_model=ManifestModel(args.loader, list(files)),
            artifacts={n: e["path"] for n, e in files.items()},
            code_paths=code_paths,
            pip_requirements=pip,
            registered_model_name=args.name,
            metadata={"loader": args.loader, "files": {n: e["sha256"] for n, e in files.items()}},
        )
        run_id = run.info.run_id
    version = str(info.registered_model_version)
    for k, v in tags.items():
        client.set_model_version_tag(args.name, version, k, tag_safe(v))
    for k in ("benchmark.name", "weights.source", "model.role"):
        if k in tags:
            client.set_registered_model_tag(args.name, k, tag_safe(tags[k]))
    if description:
        client.update_model_version(args.name, version, description=description[:5000])
    print(f"registered {', '.join(files)} as {args.name} version {version} (run {run_id})")
    print(f"model uri: models:/{args.name}/{version}")
    set_aliases(client, args.name, version, args.alias)
    return 0


def cmd_alias(args, mlflow, client) -> int:
    if bool(args.version) == bool(args.from_alias):
        raise SystemExit("register_model alias: give exactly one of --version or --from-alias")
    version = args.version or client.get_model_version_by_alias(args.name, args.from_alias).version
    set_aliases(client, args.name, str(version), args.alias)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiment", help="experiment for the registration run (default: "
                   "MLFLOW_EXPERIMENT_NAME, else <name>-registry-<user>)")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="register a model logged in a run")
    r.add_argument("--model-uri", required=True, help="e.g. runs:/<run_id>/model")

    f = sub.add_parser("files", help="register manifest-listed checkpoint files as a pyfunc model")
    f.add_argument("--manifest", required=True)
    f.add_argument("--weights-dir", required=True)
    f.add_argument("--files", nargs="+", required=True, help="manifest file names to include")
    f.add_argument("--loader", required=True, help="package.module:function returning a predictor")
    f.add_argument("--code-path", nargs="*", default=[], help="extra code shipped with the model")
    f.add_argument("--pip-requirement", nargs="*", help="default: mlflow==<this version>")
    f.add_argument("--runtime-module", nargs="*", default=[], help="record these modules' versions")
    f.add_argument("--repo", default=".")

    a = sub.add_parser("alias", help="point an alias at a version")
    a.add_argument("--version")
    a.add_argument("--from-alias", help="copy the version another alias points to")

    for sp in (r, f, a):
        sp.add_argument("--name", required=True, help="registered model name")
        sp.add_argument("--alias", nargs="*", help="aliases to set, e.g. staging or production")
    for sp in (r, f):
        sp.add_argument("--benchmark")
        sp.add_argument("--description")
    args = p.parse_args(argv)
    if args.cmd == "alias" and not args.alias:
        raise SystemExit("register_model alias: --alias is required")

    import getpass
    import os
    experiment = (args.experiment or os.environ.get("MLFLOW_EXPERIMENT_NAME")
                  or f"{args.name}-registry-{getpass.getuser()}")
    mlflow = amsc_mlflow.configure(experiment=experiment if args.cmd == "files" else None)
    from mlflow import MlflowClient

    client = MlflowClient()
    return {"run": cmd_run, "files": cmd_files, "alias": cmd_alias}[args.cmd](args, mlflow, client)


if __name__ == "__main__":
    raise SystemExit(main())
