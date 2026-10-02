#!/usr/bin/env python3
"""Template: a PyTorch training script instrumented for the AmSC MLflow service.

    python train_with_mlflow.py --epochs 5 [--register <name>]                   # one process
    torchrun --nproc-per-node 4 train_with_mlflow.py --epochs 5                  # distributed
    srun python train_with_mlflow.py ...  /  mpiexec -n 12 python train_with_mlflow.py ...

What to copy from it (references/training.md):
- import amsc_mlflow before mlflow; configure() sets server, workspace, experiment, async;
- only rank 0 creates the run and logs (rank_zero_run yields None elsewhere);
- metrics are averaged across ranks (all_reduce) before rank 0 logs them;
- params once, metrics per epoch, git/runtime tags, a plot artifact, the model at the end;
- async logging is flushed when the run closes;
- optional registration of the logged model with the `staging` alias.

The model and data are a toy regression so the template runs anywhere on CPU. Replace
them; keep the structure. Needs torch and mlflow; amsc_mlflow.py from the skill's scripts/
must be importable (same directory or on PYTHONPATH).
"""

from __future__ import annotations

import argparse
import getpass
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import amsc_mlflow  # noqa: E402  (before mlflow: enables multipart uploads)

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402


def setup_distributed() -> tuple[int, int]:
    if int(os.environ.get("WORLD_SIZE", "1")) > 1 and not dist.is_initialized():
        dist.init_process_group(backend="gloo" if not torch.cuda.is_available() else "nccl")
    if dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def mean_across_ranks(value: float) -> float:
    if not dist.is_initialized():
        return value
    t = torch.tensor([value], dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item()) / dist.get_world_size()


def make_data(n: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 4, generator=g)
    y = x @ torch.tensor([[1.5], [-2.0], [0.5], [0.0]]) + 0.1 * torch.randn(n, 1, generator=g)
    return x, y


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--hidden", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--phase", default="development", choices=["development", "tuning", "production"])
    p.add_argument("--experiment", default=f"amsc-mlflow-template-{getpass.getuser()}")
    p.add_argument("--register", metavar="NAME", help="register the logged model as NAME@staging")
    args = p.parse_args(argv)

    rank, world = setup_distributed()
    torch.manual_seed(args.seed + rank)
    mlflow = amsc_mlflow.configure(experiment=args.experiment, async_logging=True) \
        if amsc_mlflow.is_rank_zero() else None

    x, y = make_data(2048, seed=args.seed)
    xv, yv = make_data(512, seed=args.seed + 1)
    shard = slice(rank, None, world)                     # each rank trains on its shard
    model = torch.nn.Sequential(torch.nn.Linear(4, args.hidden), torch.nn.ReLU(), torch.nn.Linear(args.hidden, 1))
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.MSELoss()

    tags = {"owner": getpass.getuser(), "project.phase": args.phase, "model.class": "ToyMLP",
            **amsc_mlflow.git_tags(), **amsc_mlflow.runtime_tags("torch")}
    with amsc_mlflow.rank_zero_run(run_name=f"toy-{args.phase}-lr{args.lr}", tags=tags) as run:
        if run:
            mlflow.log_params({"lr": args.lr, "batch_size": args.batch_size, "epochs": args.epochs,
                               "hidden": args.hidden, "optimizer": "Adam", "seed": args.seed,
                               "world_size": world})
        history = []
        for epoch in range(args.epochs):
            model.train()
            xs, ys = x[shard], y[shard]
            total = 0.0
            for i in range(0, len(xs), args.batch_size):
                opt.zero_grad()
                loss = loss_fn(model(xs[i:i + args.batch_size]), ys[i:i + args.batch_size])
                loss.backward()
                if dist.is_initialized():
                    for prm in model.parameters():           # average gradients (DDP does this for you)
                        dist.all_reduce(prm.grad, op=dist.ReduceOp.SUM)
                        prm.grad /= world
                opt.step()
                total += loss.item() * len(xs[i:i + args.batch_size])
            train_loss = mean_across_ranks(total / len(xs))   # every rank, before logging
            model.eval()
            with torch.no_grad():
                val_loss = loss_fn(model(xv), yv).item()
            history.append((train_loss, val_loss))
            if run:
                mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)
                print(f"epoch {epoch + 1}/{args.epochs} train {train_loss:.4f} val {val_loss:.4f}")

        if run:
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(5, 3))
                ax.plot([h[0] for h in history], label="train")
                ax.plot([h[1] for h in history], label="val")
                ax.set_xlabel("epoch")
                ax.set_ylabel("MSE")
                ax.legend()
                mlflow.log_figure(fig, "plots/loss.png")
                plt.close(fig)
            except ImportError:
                pass
            from mlflow.models import infer_signature

            example = xv[:4].numpy()
            info = mlflow.pytorch.log_model(model, name="model", input_example=example,
                                            signature=infer_signature(example, model(xv[:4]).detach().numpy()))
            print(f"run {run.info.run_id}; model {info.model_uri}")
    if mlflow is not None and args.register:
        mv = mlflow.register_model(info.model_uri, args.register)
        mlflow.MlflowClient().set_registered_model_alias(args.register, "staging", mv.version)
        print(f"registered {args.register} version {mv.version} as @staging")
    if dist.is_initialized():
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
