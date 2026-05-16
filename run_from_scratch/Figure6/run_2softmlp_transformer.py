from __future__ import annotations

import argparse
import csv
import fcntl
from pathlib import Path
import sys

import numpy as np
import optax

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "transformer_codebase"))
from model.transformerold import TransformerConfig  
from task.regression import FiniteSampler, LinearRegressionCorrect  
from trainminiold import train  

# Data saving csv collumn names 
FIELDNAMES = [
    "d",
    "K",
    "alpha",
    "tau",
    "run_index",
    "train_iters", 
    "avgerr_ICL",
    "avgerr_IDG",
]


def append_result(csv_path: Path, row: dict[str, float | int]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("a+", newline="") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        f.seek(0, 2)
        should_write_header = f.tell() == 0
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if should_write_header:
            writer.writeheader()
        writer.writerow(row)
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def scalar_to_float(value) -> float:
    return float(np.asarray(value).item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", type=int, required=True)
    parser.add_argument("--K", "--k", dest="K", type=int, required=True)
    parser.add_argument("--run_index", "--run-index", dest="run_index", type=int, default=0)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--tau", type=float, required=True)
    parser.add_argument("--train_iters", type=int, default=100000)
    parser.add_argument("--save_csv_filepath", type=Path, default="./saved.csv")
    args = parser.parse_args()

    N = int(args.alpha * args.d) # context length
    P = int(args.tau * args.d * args.d) # full pretraining batch size

    sigma = 0.1 # or rho = 0.01
    psi = 1 # an artifact of our old implemntations; this just says tasks have average norm 1*d
    h = args.d # hidden dimension, i.e. no rank compression or expansion, helps with stable expressivity

    ## TRAIN THE TRANSFORMER
    trainobject = FiniteSampler(
        n_points=N + 1,
        n_dims=args.d,
        eta_scale=sigma,
        w_scale=psi,
        diversity=args.K,
        batch_size=P,
        seed=None,
    )
    config = TransformerConfig(
        pos_emb=False,
        n_hidden=h,
        n_layers=2,
        n_mlp_layers=1,
        pure_linear_self_att=False,
    )
    state, hist = train(
        config,
        data_iter=iter(trainobject),
        loss="mse",
        batch_size=int(0.1 * P),
        test_every=1000,
        train_iters=args.train_iters,
        optim=optax.adamw,
        lr=1e-4,
    )

    ## TEST ON FRESH TASKS (ICL ERROR)
    testobject = LinearRegressionCorrect(
        n_points=N + 1,
        n_dims=args.d,
        eta_scale=sigma,
        w_scale=psi,
        batch_size=P,
        seed=None,
    )
    avgerr_ICL = 0
    loss_func = optax.squared_error
    numsamples = 1000
    for _ in range(numsamples):
        xs, labels = next(testobject)  # generates data
        logits = state.apply_fn(
            {"params": state.params},
            xs,
        )  # runs xs through transformer and makes predictions
        avgerr_ICL = avgerr_ICL + loss_func(logits, labels).mean()
    avgerr_ICL = avgerr_ICL / numsamples

    ## TEST ON SAME TASKS AS TRAINING (IDG ERROR)
    testobject = trainobject
    avgerr_IDG = 0
    loss_func = optax.squared_error
    numsamples = 1000
    for _ in range(numsamples):
        xs, labels = next(testobject)  # generates data
        logits = state.apply_fn(
            {"params": state.params},
            xs,
        )  # runs xs through transformer and makes predictions
        avgerr_IDG = avgerr_IDG + loss_func(logits, labels).mean()
    avgerr_IDG = avgerr_IDG / numsamples

    row = {
        "d": args.d,
        "K": args.K,
        "alpha": args.alpha,
        "tau": args.tau,
        "run_index": args.run_index,
        "train_iters": args.train_iters,
        "avgerr_ICL": scalar_to_float(avgerr_ICL),
        "avgerr_IDG": scalar_to_float(avgerr_IDG),
    }
    append_result(args.save_csv_filepath, row)
    print(row)


if __name__ == "__main__":
    main()
