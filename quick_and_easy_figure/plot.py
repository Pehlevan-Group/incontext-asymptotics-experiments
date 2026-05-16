import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import csv
import importlib.util
from pathlib import Path

# DEFINE STANDARD FORMATING FOR FIGURES USED THROUGHOUT PAPER
sns.set(style="white",font_scale=2.5,palette="colorblind")
plt.rcParams['lines.linewidth'] = 4.5
plt.rcParams["figure.figsize"] = (12, 10)

kappas_theory = np.logspace(np.log10(0.15), np.log10(40), 40)

BASE_DIR = Path(__file__).resolve().parent
SAVED_DATA_DIR = BASE_DIR / "saved_data"

def load_common():
    common_path = BASE_DIR.parent / "theory_runs" / "common.py"
    spec = importlib.util.spec_from_file_location("theory_common", common_path)
    common = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(common)
    return common

def load_csv(csv_path):
    with csv_path.open(newline="") as f:
        return [
            {key: float(value) for key, value in row.items()}
            for row in csv.DictReader(f)
        ]


def plot(name):
    dmmse_path = SAVED_DATA_DIR / f"{name}_dmmse.csv"
    gamma_path = SAVED_DATA_DIR / f"{name}_gamma.csv"
    if not dmmse_path.exists() or not gamma_path.exists():
        print(f"Data for experiment {name} not yet populated. Please run run.py.")
        return

    dmmse_rows = load_csv(dmmse_path)
    gamma_rows = load_csv(gamma_path)

    dmmse_rows.sort(key=lambda row: row["kappa"])
    gamma_rows.sort(key=lambda row: row["kappa"])

    gamma_kappa = np.array([row["kappa"] for row in gamma_rows])
    gamma_mean = np.array([row["gtask_mean"] for row in gamma_rows])
    gamma_std = np.array([row["gtask_std"] for row in gamma_rows])

    dmmse_kappa = np.array([row["kappa"] for row in dmmse_rows])
    dmmse_mean = np.array([row["gtask_mean"] for row in dmmse_rows])
    dmmse_std = np.array([row["gtask_std"] for row in dmmse_rows])

    alpha = gamma_rows[0]["alpha"]
    tau = gamma_rows[0]["tau"]
    rho = gamma_rows[0]["rho"]
    common = load_common()
    theory = np.array([
        common.ridgeless_icl(alpha, kappa, tau, rho)
        - common.ridgeless_idg(alpha, kappa, tau, rho)
        for kappa in kappas_theory
    ])

    fig, ax = plt.subplots()
    ax.plot(gamma_kappa, gamma_mean, marker="o", ms = 12, label="Finite data simulation")
    ax.fill_between(
        gamma_kappa,
        gamma_mean - gamma_std,
        gamma_mean + gamma_std,
        alpha=0.2,
    )
    ax.plot(dmmse_kappa, dmmse_mean, marker="s", ms=12, label="dMMSE")
    ax.plot(kappas_theory, theory, linestyle="--", label="Theory curve")

    plt.axvline(1, linestyle=':', linewidth=4, color='red', label=r"$\kappa$ = 1")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\kappa = k/d$")
    ax.set_ylabel(r"$g_{\text{task}}$")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(BASE_DIR / "saved_plots" / f"{name}_plot.png", dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default='test')
    args = parser.parse_args()
    plot(args.name)


if __name__ == "__main__":
    main()
