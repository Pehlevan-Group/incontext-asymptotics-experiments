# Figure 1 data sweep parameters and instructions

To generate the data used for panels A and B of figure 1 (ridgeless plots of ICL and IDG error), run `run_errors_from_data.py` for the follow sweeps of values

> `tau=[0.2, 0.5, 0.85, 1.15, 1.5, 2]`
> `alpha=[1,10,100]`
> `d=100, rho=0.01, kappa = 0.5`
> `lam=0.00001`

Both the ICL and IDG run values saved from these will be used for Panels A and B.

To generate the data used for panel C (ridge plot of ICL error), run `run_errors_from_data.py` for the follow sweeps of values

> `tau=[0.2, 0.5, 0.85, 1, 1.15, 1.5, 2]`
> `alpha=10`
> `kappa=-1`
> `rho=0.01, d=100`
> `lam=[0.00001, 0.1, 1]`

Note that only the ICL run values saved from these will be used for Panel C.

`run_errors_from_data.py` can be called by 

```
python run_errors_from_data.py --d=100 --kappa=1 --alpha=1 --tau=5 --rho=0.01 --lam=0.00001 --numavg=10 
```

It returns 4 values, ICL mean error, ICL error stdv, IDG mean error, IDG error stdv, in that order. These values are saved to a csv file.

The data is added to the csv in a threading-safe way so that the calls to this python file over the parameter values listed above can be parallelised over, say, a job array. 