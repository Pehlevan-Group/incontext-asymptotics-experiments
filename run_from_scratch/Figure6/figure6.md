# Figure 6 instructions

This figure illustrates the transition between memorisation and generalisation against task diversity by plotting `g_task` against `kappa`. The architecture here will be two blocks of softmax attention and a single layer MLP, as described by the paper. 

To highlight both the transition and the concentration / correct scaling law of $k \propto d$, we use runs at $d=20, d=40, d=80.$ The $k$ values used for each experiment in Figure 6 are 
> $d=20$: `k = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 23, 28, 35, 45, 57, 69, 82, 96, 109, 123]`
> $d=40$: `k = [2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 40, 43, 46, 50, 54, 59, 64, 69, 75, 81, 88, 95, 103, 112, 121, 131, 142, 154, 167, 180, 195, 211, 229, 248, 268]`
> $d=80$: `k = [2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 75, 100, 142, 159, 178, 200, 225, 252, 283, 317, 356]`

The remaining parameters are 
> `alpha = 1`
> `tau = 10`
> `rho = 0.01`
> `train_iters = 100000`

For example, a use of this code that will run fairly quickly (but isn't trained long enough to be compared to the full experiments above we use) is 
```
python run_2softmlp_transformer.py --d 10 --K 20 --run_index 0 --alpha 1 --tau 10 --train_iters 100
``` 

The default `csv` file containing results is `./saved.csv`; change this to whatever you wish using the `--save_csv_filepath` flag.