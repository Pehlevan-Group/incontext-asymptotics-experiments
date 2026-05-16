from __future__ import annotations

import argparse
import csv
import fcntl
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "theory_runs"))

from common import *  


FIELDNAMES = [
    "d",
    "alpha",
    "kappa",
    "tau",
    "rho",
    "lam",
    "numavg",
    "icl_mean",
    "icl_std",
    "idg_mean",
    "idg_std",
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", type=int, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--kappa", type=float, required=True)
    parser.add_argument("--tau", type=float, required=True)
    parser.add_argument("--rho", type=float, required=True)
    parser.add_argument("--lam", type=float, required=True)
    parser.add_argument("--numavg", type=int, required=True)
    parser.add_argument("--save_csv_filepath", type=Path, default='./saved.csv')
    args = parser.parse_args()

    icl_mean, icl_std, idg_mean, idg_std = errors_from_DATA(  # noqa: F405
        args.d,
        args.alpha,
        args.kappa,
        args.tau,
        args.rho,
        args.lam,
        args.numavg,
    )

    row = {
        "d": args.d,
        "alpha": args.alpha,
        "kappa": args.kappa,
        "tau": args.tau,
        "rho": args.rho,
        "lam": args.lam,
        "numavg": args.numavg,
        "icl_mean": icl_mean,
        "icl_std": icl_std,
        "idg_mean": idg_mean,
        "idg_std": idg_std,
    }
    append_result(args.save_csv_filepath, row)
    print(row)


if __name__ == "__main__":
    main()
