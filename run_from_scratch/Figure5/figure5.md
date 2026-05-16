# Instructions for Figure 5

This figure contains multiple subfigures. Three are plots of double descent (i.e. ICL error against pretraining batch size, or equivalently, $\tau$). The final is a summary plot that tracks the locations of the double descent peaks for different architectures.

## Panels A, B, C: Double descent plots 
Here we compare three architectures: trainable linear attention, one block of softmax attention, and two blocks of softmax attention + mlp. We also compare over three values of $d=20,40,80$ to demonstrate the consistency of our proposed $n \propto d^2$ scaling rule. 

For all experiments, we have 
> `alpha = 1`
> `rho = 0.01`

Note that $k$ or $\kappa$ is no longer an argument for this code as we take "infinitely" many pretraining tasks for these runs, i.e. no finite-task limitatio.

The main code is done through `run_transformers.py.` An example run is 
```
python run_transformers.py --d 5 --run_index 0 --alpha 1 --tau 10 --train_iters 20 --model_type=4
``` 
The key new parameter here is `model_type` as this lets us toggle between the different models plotted in Figure 5. 
- `model_type = 1`: linear attention (panel A)
- `model_type = 2`: softmax attention (panel B)
- `model_type = 3`: 2x softmax attention + mlp (panel C)
- `model_type = 4`: 3x softmax attention + mlp (part of panel D)

### Linear attention (panel A)
For all $d$ values we use 

> `taus = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 2.0, 2.5, 3.0, 5.0]`

### Softmax attention (panel B)
> $d = 20$: `taus = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. , 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2. , 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3. , 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4. , 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5. , 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6. , 6.1]`
> $d = 40$: `taus = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. , 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2. , 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3. , 3.1, 3.5, 4. , 4.5, 5. , 5.5, 6. ]`
> $d = 80$: `taus = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. , 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2. , 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3. , 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4. , 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5. , 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6. , 6.1]`

### 2x (Softmax + MLP) (Panel C)
> $d = 20$: `taus = [0.5 ,  0.6 ,  0.7 ,  0.8 ,  0.9 ,  1.  ,  1.1 ,  1.2 ,  1.3 , 1.4 ,  1.5 ,  1.6 ,  1.7 ,  1.8 ,  1.9 ,  2.  ,  2.1 ,  2.2 , 2.3 ,  2.4 ,  2.5 ,  2.6 ,  2.7 ,  2.8 ,  2.9 ,  3.  ,  3.1 , 3.2 ,  3.3 ,  3.4 ,  3.5 ,  3.6 ,  3.7 ,  3.8 ,  3.9 ,  4.  , 4.1 ,  4.2 ,  4.3 ,  4.4 ,  4.5 ,  4.6 ,  4.7 ,  4.8 ,  4.9 , 5.  ,  5.1 ,  5.2 ,  5.3 ,  5.4 ,  5.5 ,  5.6 ,  5.7 ,  5.8 , 5.9 ,  6.  ,  6.1 ,  6.2 ,  6.3 ,  6.4 ,  6.5 ,  6.75,  7.  , 7.25,  7.5 ,  7.75,  8.  ,  8.25,  8.5 ,  8.75,  9.  ,  9.25, 9.5 ,  9.75, 10.  , 10.5 , 11.  , 11.5 , 12.  , 13.  , 14.  , 15.  ]`
> $d = 40$: `taus = [ 0.5 ,  0.55,  0.6 ,  0.65,  0.7 ,  0.75,  0.8 ,  0.85,  0.9 , 0.95,  1.  ,  1.05,  1.1 ,  1.15,  1.2 ,  1.25,  1.3 ,  1.35, 1.4 ,  1.45,  1.5 ,  1.75,  2.  ,  2.25,  2.5 ,  2.6 ,  2.7 , 2.8 ,  2.9 ,  3.  ,  3.25,  3.5 ,  3.75,  4.  ,  4.25,  4.5 , 4.75,  5.  ,  6.  ,  6.25,  6.5 ,  6.75,  7.  ,  7.25,  7.5 , 7.75,  8.  ,  8.25,  8.5 ,  8.75,  9.25,  9.5 ,  9.75, 10.  , 10.5 , 11.  , 11.5 , 12.  , 13.  , 14.  , 15.  ]`
> $d = 80$: `taus = [ 0.5 ,  0.55,  0.6 ,  0.65,  0.7 ,  0.75,  0.8 ,  0.85,  0.9 , 0.95,  1.  ,  1.05,  1.1 ,  1.15,  1.2 ,  1.25,  1.3 ,  1.35, 1.4 ,  1.45,  1.5 ,  1.75,  2.  ,  2.25,  2.5 ,  2.6 ,  2.7 , 2.8 ,  2.9 ,  3.  ,  3.25,  3.5 ,  3.75,  4.  ,  4.25,  4.5 , 4.75,  5.  ,  6.  ,  6.25,  6.5 ,  6.75,  7.  ,  7.25,  7.5 , 7.75,  8.  ,  8.25,  8.5 ,  8.75,  9.25,  9.5 ,  9.75, 10.  , 10.5 , 11.  , 11.5 , 12.  , 13.  , 14.  , 15.  ]`

## Panel D: interpolation threshold plot
To get the training losses from the run state `hist` object, one can append 
```
file_path = f'.train-file.pkl'
with open(file_path, 'wb') as fp:
    pickle.dump(hist, fp)
```
which saves the training losses over epochs to a `pkl` file. We omit this code for clarity of the code base. One may be interested in investigating the training losses to determine when the model is interpolating (0 training loss) or failing to interpolate (non 0 training loss). 

To determine the location of the interpolation threshold for the four models shown in Panel D, we sweep over the following $\tau$ values for each. Note that while the theory predicts interpolation at $\tau = 1$ for the linear transformer, we find that this value increases with model complexity. 
