import numpy as np
import argparse
import csv
import time
from pathlib import Path

import run_gtask_from_data
import run_gtask_dmmse

def main() -> None:
    start_time = time.perf_counter()

    parser = argparse.ArgumentParser()
    parser.add_argument("--d", type=int, required=True)
    parser.add_argument("--name", type=str, default='test')
    args = parser.parse_args()

    gamma_csv_name = Path("saved_data") / f"{args.name}_gamma.csv"
    dmmse_csv_name = Path("saved_data") / f"{args.name}_dmmse.csv"

    d = args.d
    kappas = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.6, 2.0, 4.0, 8.0, 16.0, 32.0]
    alpha = 10; 
    tau = 10;
    rho = 0.01
    lam = 0.000001
    numavg_gamma = 20; numavg_dmmse = 1000

    for kappa in kappas:
        icl_mean, icl_std, idg_mean, idg_std = run_gtask_from_data.errors_from_DATA(
            d,
            alpha,
            kappa,
            tau,
            rho,
            lam,
            numavg_gamma,
        )
        run_gtask_from_data.append_result(
            gamma_csv_name,
            {
                "d": d,
                "alpha": alpha,
                "kappa": kappa,
                "tau": tau,
                "rho": rho,
                "lam": lam,
                "numavg": numavg_gamma,
                "gtask_mean": icl_mean - idg_mean,
                "gtask_std": np.sqrt(icl_std**2 + idg_std**2),
            },
        )
        
        icl_mean, icl_std, idg_mean, idg_std = run_gtask_dmmse.dmmse_estimator_errors(
            d,
            alpha,
            kappa,
            rho,
            numavg_dmmse,
        )
        run_gtask_dmmse.append_result(
            dmmse_csv_name,
            {
                "d": d,
                "alpha": alpha,
                "kappa": kappa,
                "rho": rho,
                "numavg": numavg_dmmse,
                "gtask_mean": icl_mean - idg_mean,
                "gtask_std": np.sqrt(icl_std**2 + idg_std**2),
            },
        )
    
    print(f"Data saved; took {time.perf_counter() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
