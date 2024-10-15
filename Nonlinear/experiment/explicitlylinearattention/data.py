# Connection To Theory Paper
# n_points:     N + 1 = Length of a given context + 1 (includes the final x_{N+1} query vector)
# n_dims:       d = dimension of a token vectors
# eta_scale:    standard deviation of noise scalars eta
# w_scale:      standard deviation of beta vectors in isotropic case beta~N(0, sigma_beta I)
# batch_size:   P = number of contexts in prompt

import numpy as np
import random

# Implements DIVERSE ISOTROPIC case with C = I
class fulldiversitysampler:
    def __init__(self, n_points=6, n_dims=2, eta_scale=1, w_scale=1, data_cov=1, batch_size=128, seed=None) -> None:
        self.n_points = n_points # n_points = N+1 where N = context length, as n_points includes the (N+1)st query vector
        self.n_dims = n_dims # d = dimension of tokens
        self.w_scale = w_scale # sigma_beta
        self.eta_scale = eta_scale # noise sigma
        self.data_cov = data_cov # C = 1 usually but want to customise like 1/sqrt(alpha)
        self.batch_size = batch_size # P = number of contexts
        self.rng = np.random.default_rng(seed)
    
    def __next__(self):
        xs = (self.data_cov)*self.rng.normal(loc=0, scale = 1/np.sqrt(self.n_dims), size=(self.batch_size, self.n_points, self.n_dims))
        ws = self.rng.normal(loc=0, scale = self.w_scale, size=(self.batch_size, self.n_dims, 1))
        ys = xs @ ws + self.rng.normal(loc=0, scale = self.eta_scale, size=(self.batch_size, self.n_points, 1))
        Z = np.zeros((self.batch_size, self.n_points, self.n_dims + 1))
        Z[:,:,0:self.n_dims] = xs
        Z[:,:,-1] = ys.squeeze()
        Z[:,-1, self.n_dims] = 0 # padding for final context
	    
	    # returns the Z [x,y,x,y]... configuration and the true N+1 value for testing 
        return Z, ys[:,-1].squeeze()

    def __iter__(self):
        return self
