"""Generic MLflow pyfunc wrapper for checkpoint files registered by register_model.py.

Shipped inside each registered model (code_paths), so loading needs only this file plus
the loader's own package. The loader is a function the benchmark provides,
"package.module:function", taking {file name: local path} and returning an object with
.predict(model_input) (or a callable):

    # in the benchmark's package, e.g. src/<pkg>/mlflow_loader.py
    def load(artifacts):
        model = MyArchitecture()
        model.load_state_dict(torch.load(artifacts["model_final.pt"]), strict=True)
        return Predictor(model)          # .predict(DataFrame) -> DataFrame

Then, anywhere with the benchmark package installed:

    model = mlflow.pyfunc.load_model("models:/<name>@production")
    model.predict(df)
    raw = model.unwrap_python_model().model      # the loader's object
"""

from __future__ import annotations

import importlib

import mlflow.pyfunc


class ManifestModel(mlflow.pyfunc.PythonModel):
    def __init__(self, loader: str, files: list[str], options: dict | None = None):
        if ":" not in loader:
            raise ValueError(f"loader must be 'package.module:function', got {loader!r}")
        self.loader = loader
        self.files = list(files)
        self.options = dict(options or {})
        self.model = None

    def load_context(self, context):
        module, func = self.loader.split(":", 1)
        load = getattr(importlib.import_module(module), func)
        self.model = load({name: context.artifacts[name] for name in self.files}, **self.options)

    def predict(self, context, model_input, params=None):
        if self.model is None:
            raise RuntimeError("model not loaded: load_context() wasn't called")
        if hasattr(self.model, "predict"):
            return self.model.predict(model_input, **(params or {}))
        return self.model(model_input, **(params or {}))
