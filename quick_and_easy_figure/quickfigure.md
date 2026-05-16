# Instructions

This is a proof-of-concept end-to-end plotting code meant to reproduce something akin to Figure 4 (phase transition from memorisation to generalisation in task diversity), but runable in reasonable time on CPU for demonstration purposes. We have both a data-creation function, `run.py`, and its corresponding plotting function

An example run would be 
```
conda activate incontext-asymptotics
python run.py --d=10 --name=myrun_d10
python plot.py --name=myrun_d10
```
Note that you want to make sure `run.py` finishes running first because `plot.py` will look for the files it's supposed to create in `saved_data.` The plotting function will create a plot in the folder `saved_plots`.

This code has $\kappa, \alpha, \tau, \rho, \lambda$ values already fixed. One can edit the `run.py` code directly if one wishes to use different values; for the purposes of a simple and easy demonstration we have hardcoded these.

The one key parameter for `run.py` is the dimension of the tokens $d$. Depending on how large this is, this code is feasible to run on CPU. 
- For $d=10$ this took us 00:18min on login node. 
- For $d=32$ this took us 04:20min on login node. 
- For $d=64$ this took us 73:13min on login node. 
As always, the larger the $d$ the better the concentration and fit to the theory curve. These runs could be done much faster by parallelising over the $\kappa$ values while running, but for simplicity of use we have not done this here. 