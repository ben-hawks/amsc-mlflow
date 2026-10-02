"""End-to-end tests of the amsc-mlflow scripts against a local MLflow server."""

from __future__ import annotations

import base64
import hashlib
import json
import sys
import time
from pathlib import Path

import pytest

from conftest import SCRIPTS, run_script

import amsc_mlflow


# --------------------------------------------------------------------------- helpers


def jwt(exp: float) -> str:
    def b64(d):
        return base64.urlsafe_b64encode(json.dumps(d).encode()).rstrip(b"=").decode()
    return f"{b64({'alg': 'none'})}.{b64({'exp': int(exp), 'sub': 'x'})}.sig"


def make_results(root: Path) -> Path:
    res = root / "reference_results"
    for split, model, r2 in [("test", "gnn", 0.9), ("test", "mlp", 0.3), ("exemplar", "gnn", -0.5)]:
        d = res / split / model
        d.mkdir(parents=True)
        (d / "metrics.json").write_text(json.dumps({
            "coverage": {"n_truth": 10, "n_scored": 9},
            "groups": {"all": {"lut": {"r_squared": r2, "smape": 12.5},
                               "ff": {"r_squared": float("nan"), "smape": 3.0}},
                       "dense": {"lut": {"r_squared": 0.1, "smape": 1.0}}}}))
        (d / "METRICS.md").write_text(f"# {model} on {split}\n")
    (res / "test" / "predictions_gnn.csv").write_text("sample_id,lut,ff\na,1,2\n")
    (res / "test" / "predictions_gnn.meta.json").write_text(json.dumps({"model_id": "gnn@v1", "n_shot": None}))
    (res / "LEADERBOARD.md").write_text("# leaderboard\n")
    return res


# --------------------------------------------------------------------------- amsc_mlflow


def test_token_expiry_decoded_without_printing(monkeypatch, capsys):
    monkeypatch.setenv("MLFLOW_TRACKING_TOKEN", jwt(time.time() + 600))
    assert amsc_mlflow.token_expiry() is not None
    left = amsc_mlflow.check_token_lifetime(min_seconds=3600)
    assert 0 < left <= 600
    err = capsys.readouterr().err
    assert "expires" in err and "dummy" not in err
    info = amsc_mlflow.describe()
    assert info["token"] == "set" and jwt(0)[:10] not in json.dumps(info)


def test_token_from_file(tmp_path, monkeypatch):
    f = tmp_path / "tok"
    f.write_text("abc\n")
    f.chmod(0o600)
    monkeypatch.delenv("MLFLOW_TRACKING_TOKEN", raising=False)
    monkeypatch.setenv("MLFLOW_TRACKING_TOKEN_FILE", str(f))
    assert amsc_mlflow.load_token()
    import os
    assert os.environ["MLFLOW_TRACKING_TOKEN"] == "abc"


@pytest.mark.parametrize("var,value,rank", [("SLURM_PROCID", "3", 3), ("PALS_RANKID", "0", 0),
                                            ("OMPI_COMM_WORLD_RANK", "1", 1)])
def test_rank_detection(monkeypatch, var, value, rank):
    for v in amsc_mlflow.RANK_VARIABLES:
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv(var, value)
    assert amsc_mlflow.global_rank() == rank
    assert amsc_mlflow.is_rank_zero() == (rank == 0)


def test_rank_zero_run_skips_other_ranks(monkeypatch):
    monkeypatch.setenv("RANK", "2")
    with amsc_mlflow.rank_zero_run(run_name="x") as run:
        assert run is None


def test_multipart_defaults_set_on_import():
    import os
    assert os.environ["MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD"] == "true"
    assert os.environ["MLFLOW_MULTIPART_UPLOAD_MINIMUM_FILE_SIZE"] == "5242880"


# --------------------------------------------------------------------------- check_connection


def test_check_connection_no_token(monkeypatch):
    r = run_script("check_connection.py", env_extra={"MLFLOW_TRACKING_TOKEN": "",
                                                     "MLFLOW_TRACKING_URI": "https://example.invalid"})
    assert r.returncode == 2, r.stderr


def test_check_connection_expired_token():
    r = run_script("check_connection.py", env_extra={"MLFLOW_TRACKING_TOKEN": jwt(time.time() - 60),
                                                     "MLFLOW_TRACKING_URI": "https://example.invalid"})
    assert r.returncode == 3, r.stderr


def test_check_connection_ok(env):
    r = run_script("check_connection.py", "--experiment", "conn-test")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "SUCCESS" in r.stdout and "dummy-token" not in r.stdout + r.stderr


def test_check_connection_network_failure(env):
    r = run_script("check_connection.py", env_extra={"MLFLOW_TRACKING_URI": "http://127.0.0.1:9",
                                                     "MLFLOW_HTTP_REQUEST_MAX_RETRIES": "0",
                                                     "MLFLOW_HTTP_REQUEST_TIMEOUT": "5"})
    assert r.returncode == 5, r.stdout + r.stderr


# --------------------------------------------------------------------------- log_benchmark_results


def test_log_results_dry_run_needs_no_server(tmp_path):
    res = make_results(tmp_path)
    r = run_script("log_benchmark_results.py", "--benchmark", "demo", "--results", str(res), "--dry-run",
                   env_extra={"MLFLOW_TRACKING_TOKEN": "", "MLFLOW_TRACKING_URI": "https://example.invalid"})
    assert r.returncode == 0, r.stderr
    assert "child run: test/gnn" in r.stdout and "metric lut.r_squared = 0.9" in r.stdout
    assert "dense/" not in r.stdout


def test_log_results_end_to_end_and_skip_existing(env, tmp_path):
    from mlflow import MlflowClient

    res = make_results(tmp_path)
    args = ["--benchmark", "demo", "--results", str(res), "--experiment", "demo-e2e",
            "--auxiliary", "mlp", "--all-groups", "--log-predictions"]
    r = run_script("log_benchmark_results.py", *args)
    assert r.returncode == 0, r.stdout + r.stderr
    mlflow_mod = __import__("mlflow")
    mlflow_mod.set_tracking_uri(env)
    mlflow_mod.set_workspace("modelservices")
    client = MlflowClient(env)
    exp = client.get_experiment_by_name("demo-e2e")
    runs = client.search_runs([exp.experiment_id])
    children = {x.data.tags["benchmark.split"] + "/" + x.data.tags["benchmark.model"]: x
                for x in runs if "benchmark.split" in x.data.tags}
    assert set(children) == {"test/gnn", "test/mlp", "exemplar/gnn"}
    gnn = children["test/gnn"]
    assert gnn.data.metrics["lut.r_squared"] == 0.9
    assert gnn.data.metrics["dense/lut.r_squared"] == 0.1
    assert gnn.data.metrics["coverage.n_scored"] == 9
    assert gnn.data.tags["benchmark.nonfinite_metrics"] == "ff.r_squared"
    assert gnn.data.params["model_id"] == "gnn@v1"
    src = res / "test" / "gnn" / "metrics.json"
    assert gnn.data.tags["benchmark.source_sha256"] == hashlib.sha256(src.read_bytes()).hexdigest()
    assert children["test/mlp"].data.tags["benchmark.model_role"] == "auxiliary"
    arts = {a.path for a in client.list_artifacts(gnn.info.run_id, "results")}
    assert {"results/metrics.json", "results/METRICS.md", "results/predictions_gnn.meta.json",
            "results/predictions_gnn.csv"} <= arts

    r2 = run_script("log_benchmark_results.py", *args)
    assert r2.returncode == 0 and "nothing new to log" in r2.stdout, r2.stdout + r2.stderr


def test_log_bundle(env):
    bundle = Path(__file__).parent / "fixtures" / "bundle.json"
    r = run_script("log_benchmark_results.py", "--benchmark", "beamline_qa", "--bundle", str(bundle),
                   "--experiment", "bundle-e2e")
    assert r.returncode == 0, r.stdout + r.stderr
    from mlflow import MlflowClient

    __import__("mlflow").set_tracking_uri(env)
    __import__("mlflow").set_workspace("modelservices")
    client = MlflowClient(env)
    exp = client.get_experiment_by_name("bundle-e2e")
    child = [x for x in client.search_runs([exp.experiment_id]) if "benchmark.model" in x.data.tags][0]
    assert child.data.tags["benchmark.model"] == "EleutherAI/pythia-14m"
    assert abs(child.data.metrics["beamline_qa_test.acc"] - 0.2916666666666667) < 1e-12
    assert "beamline_qa_test.acc_stderr" in child.data.metrics


def test_log_bundle_refuses_partial(tmp_path):
    b = json.loads((Path(__file__).parent / "fixtures" / "bundle.json").read_text())
    b["results"][0]["is_partial"] = True
    p = tmp_path / "b.json"
    p.write_text(json.dumps(b))
    r = run_script("log_benchmark_results.py", "--benchmark", "x", "--bundle", str(p), "--dry-run")
    assert r.returncode != 0 and "partial" in r.stderr


# --------------------------------------------------------------------------- register_model


def test_register_files_alias_and_load(env, tmp_path, monkeypatch):
    weights = tmp_path / "weights"
    weights.mkdir()
    (weights / "scale.json").write_text(json.dumps({"scale": 2.0}))
    digest = hashlib.sha256((weights / "scale.json").read_bytes()).hexdigest()
    manifest = tmp_path / "MANIFEST.json"
    manifest.write_text(json.dumps({"source": "https://example.org/rel", "source_commit": "abc",
                                    "files": {"scale.json": {"role": "reference", "sha256": digest,
                                                             "model": "Toy scaler"}}}))
    pkg = tmp_path / "toypkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "loader.py").write_text(
        "import json\n"
        "class P:\n"
        "    def __init__(self, s): self.s = s\n"
        "    def predict(self, df): return df * self.s\n"
        "def load(artifacts):\n"
        "    return P(json.load(open(artifacts['scale.json']))['scale'])\n")
    base = ["files", "--manifest", str(manifest), "--weights-dir", str(weights), "--files", "scale.json",
            "--name", "toy-scaler", "--loader", "toypkg.loader:load", "--code-path", str(pkg),
            "--benchmark", "demo", "--alias", "staging"]
    r = run_script("register_model.py", *base)
    assert r.returncode == 0, r.stdout + r.stderr

    r = run_script("register_model.py", "alias", "--name", "toy-scaler", "--from-alias", "staging",
                   "--alias", "production")
    assert r.returncode == 0, r.stdout + r.stderr

    import mlflow
    import pandas as pd
    from mlflow import MlflowClient

    mlflow.set_tracking_uri(env)
    mlflow.set_workspace("modelservices")
    client = MlflowClient(env)
    mv = client.get_model_version_by_alias("toy-scaler", "production")
    assert mv.tags["weights.sha256.scale.json"] == digest
    assert mv.tags["benchmark.name"] == "demo" and "Toy scaler" in mv.description
    monkeypatch.syspath_prepend(str(tmp_path))
    model = mlflow.pyfunc.load_model("models:/toy-scaler@production")
    out = model.predict(pd.DataFrame({"x": [1.0, 3.0]}))
    assert list(out["x"]) == [2.0, 6.0]

    (weights / "scale.json").write_text(json.dumps({"scale": 3.0}))
    r = run_script("register_model.py", *base)
    assert r.returncode != 0 and "sha256" in r.stderr


# --------------------------------------------------------------------------- training template


def test_training_template_logs_and_registers(tmp_path):
    pytest.importorskip("torch")
    import subprocess

    from conftest import SKILL

    uri = f"sqlite:///{tmp_path}/t.db"
    r = subprocess.run([sys.executable, str(SKILL / "assets" / "train_with_mlflow.py"), "--epochs", "2",
                        "--register", "toy-template"], capture_output=True, text=True, timeout=600,
                       env={**__import__("os").environ, "MLFLOW_TRACKING_URI": uri,
                            "MLFLOW_WORKSPACE": "none", "MLFLOW_DISABLE_AGENT_HINT": "1"})
    assert r.returncode == 0, r.stdout + r.stderr
    import mlflow
    from mlflow import MlflowClient

    mlflow.set_tracking_uri(uri)
    mv = MlflowClient(uri).get_model_version_by_alias("toy-template", "staging")
    run = MlflowClient(uri).get_run(mv.run_id)
    assert run.data.params["epochs"] == "2" and "train_loss" in run.data.metrics
    assert run.data.tags["project.phase"] == "development"
