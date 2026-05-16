# Figure 2 data sweep parameters and instructions

To generate the data used for panel A of figure 1 (nonmonotonicity plot of IDG error against `alpha` for various `kappa`s), run `run_errors_from_data.py` for the follow sweeps of values

> `alphas = [1, 5.6, 6.15, 7.4, 100, 1000]`
> `kappas = [0.1, 0.3, 0.5, 0.7, 2]`
> `tau = 0.5`
> `d = 100`
> `rho = 0.5`
> `lam = 0.000001`

Note that only the IDG values from these runs will be used for panel A. 

To generate the data used for panel B, run `run_errors_from_data.py` for the follow sweeps of values

> `alphas = [2.35, 5.5, 9.62, 22.7, 100, 1000]`
> `kappas = [0.2, 0.4, 0.5, 0.6, 1.5]`
> `tau = 0.5`
> `d = 100`
> `rho = 0.1`
> `lam = 0.000001`

Only the ICL values will be useful for panel B.

To generate the data used for panel C, run `run_errors_from_data.py` for the follow sweeps of values

> `alphas = [2.35, 5.5, 9.62, 22.7, 100, 1000]`
> `kappas = [0.25, 0.75, 1, 1.1, 2]`
> `tau = 20`
> `d = 100`
> `rho = 0.5`
> `lam = 0.000001`

Only the ICL values will be useful for panel C.

`run_errors_from_data.py` can be called by 

```
python run_errors_from_data.py --d=100 --kappa=1 --alpha=1 --tau=5 --rho=0.01 --lam=0.00001 --numavg=10 
```

It returns 4 values, ICL mean error, ICL error stdv, IDG mean error, IDG error stdv, in that order. These values are saved to a csv file.

The data is added to the csv in a threading-safe way so that the calls to this python file over the parameter values listed above can be parallelised over, say, a job array. 