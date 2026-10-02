"""Fixtures: a real MLflow tracking server on localhost (SQLite backend, proxied artifacts).

The skill's scripts treat any http(s) tracking URI as remote and require a token, so the
tests set a dummy MLFLOW_TRACKING_TOKEN; the local server ignores it. Nothing contacts the
AmSC server.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

SKILL = Path(__file__).resolve().parents[1] / "skills" / "amsc-skills" / "amsc-mlflow"
SCRIPTS = SKILL / "scripts"
sys.path.insert(0, str(SCRIPTS))


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def mlflow_server(tmp_path_factory):
    root = tmp_path_factory.mktemp("mlflow-server")
    port = _free_port()
    mlflow_bin = shutil.which("mlflow") or str(Path(sys.executable).parent / "mlflow")
    proc = subprocess.Popen(
        [mlflow_bin, "server", "--host", "127.0.0.1", "--port", str(port),
         "--backend-store-uri", f"sqlite:///{root}/mlflow.db",
         "--artifacts-destination", str(root / "artifacts"), "--workers", "1",
         "--enable-workspaces"],
        stdout=open(root / "server.log", "w"), stderr=subprocess.STDOUT)
    url = f"http://127.0.0.1:{port}"
    for _ in range(120):
        try:
            urllib.request.urlopen(url + "/health", timeout=1)
            break
        except Exception:
            time.sleep(0.5)
    else:
        proc.kill()
        raise RuntimeError("local mlflow server didn't start: " + (root / "server.log").read_text()[-2000:])
    # Mirror the AmSC layout: a named "modelservices" workspace next to the default one.
    import mlflow

    mlflow.set_tracking_uri(url)
    mlflow.create_workspace("modelservices")
    yield url
    proc.terminate()
    proc.wait(timeout=30)


@pytest.fixture
def env(mlflow_server, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", mlflow_server)
    monkeypatch.setenv("MLFLOW_TRACKING_TOKEN", "dummy-token-for-local-server")
    monkeypatch.setenv("MLFLOW_WORKSPACE", "modelservices")
    monkeypatch.delenv("MLFLOW_EXPERIMENT_NAME", raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    monkeypatch.setenv("MLFLOW_DISABLE_AGENT_HINT", "1")
    return mlflow_server


def run_script(name: str, *args: str, env_extra: dict | None = None) -> subprocess.CompletedProcess:
    e = dict(os.environ, **(env_extra or {}))
    return subprocess.run([sys.executable, str(SCRIPTS / name), *args], capture_output=True,
                          text=True, env=e, timeout=600)
