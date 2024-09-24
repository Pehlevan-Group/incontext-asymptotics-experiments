"""
In-context regression tasks

author: William Tong (wtong@g.harvard.edu)
modified by Mary Letey for 2024 In-Context Learning Project
"""

# <codecell>

# Connection To Theory Paper
# n_points:     N + 1 = Length of a given context + 1 (includes the final x_{N+1} query vector)
# n_dims:       d = dimension of a token vectors
# eta_scale:    standard deviation of noise scalars eta
# w_scale:      standard deviation of beta vectors in isotropic case beta~N(0, sigma_beta I)
# batch_size:   P = number of contexts in prompt

import numpy as np
import random

# Implements DIVERSE ISOTROPIC case with C = I
class LinearRegressionCorrect:
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

# We introduce a new parameter here as well
# diversity: K = number of distinct beta_k to be sampled UNIFORMLY for each context 
class FiniteSampler:
    def __init__(self, n_points=6, variable_context=False, context_shift_amount=0, n_dims=2, eta_scale=1, w_scale=1, diversity=6, batch_size=128, seed=None) -> None:
        self.n_points = n_points # n_points = N+1 where N = context length, as n_points includes the (N+1)st query vector
        self.variable_context = variable_context
        self.context_shift_amount = context_shift_amount # this is an alpha value not an N value!!
        self.n_dims = n_dims # d = dimension of tokens
        self.w_scale = w_scale # sigma_beta
        self.eta_scale = eta_scale # noise sigma
        self.batch_size = batch_size # P = number of contexts
        self.diversity = diversity
        self.rng = np.random.default_rng(None)
        # Now we fix a set of betas which will be sampled from during all other calls to iter or next once this object is instantiated. 
        # Once we get to the actual sampling, we will use
        self.E = self.rng.normal(loc=0, scale = self.w_scale, size=(self.n_dims, self.diversity)) 
    
    def __next__(self):
        if self.variable_context:
            context_length = self.n_points # force context length fixed.
        else:
            context_length = self.n_points
        print(context_length)
        uniform_ps = np.array([random.randrange(self.diversity) for _ in range(self.batch_size)])
        ws = np.array([self.E[:,uniform_ps[i]] for i in range(len(uniform_ps))]) 
        ws = ws[:,:,np.newaxis] # batch_size x n_dims x 1 as before
        xs = self.rng.normal(loc=0, scale = 1/np.sqrt(self.n_dims), size=(self.batch_size, context_length, self.n_dims))
        ys = xs @ ws + self.rng.normal(loc=0, scale = self.eta_scale, size=(self.batch_size, context_length, 1))
        Z = np.zeros((self.batch_size, context_length, self.n_dims + 1))
        Z[:,:,0:self.n_dims] = xs
        Z[:,:,-1] = ys.squeeze()
        Z[:,-1, self.n_dims] = 0 # padding for final context
    # returns the Z [x,y,x,y]... configuration and the true N+1 value for testing 
        return Z, ys[:,-1].squeeze()

    def __iter__(self):
        return self


class FiniteTasksVariableContext:
    def __init__(self, n_points=6, variable_context=False, context_shift_amount=0, n_dims=2, eta_scale=1, w_scale=1, diversity=6, batch_size=128, seed=None) -> None:
        self.n_points = n_points  # n_points = N+1 where N = context length, as n_points includes the (N+1)st query vector
        self.variable_context = variable_context
        self.context_shift_amount = context_shift_amount  # this is an alpha value, not an N value!!
        self.n_dims = n_dims  # d = dimension of tokens
        self.w_scale = w_scale  # sigma_beta
        self.eta_scale = eta_scale  # noise sigma
        self.batch_size = batch_size  # P = number of contexts
        self.diversity = diversity
        self.rng = np.random.default_rng(seed)
        # Now we fix a set of betas which will be sampled from during all other calls to iter or next once this object is instantiated.
        self.E = self.rng.normal(loc=0, scale=self.w_scale, size=(self.n_dims, self.diversity))

    def __next__(self):
        if self.variable_context:
            context_lengths = [int(random.uniform(max(int(0.1 * self.n_dims), self.n_points - int(self.n_dims * self.context_shift_amount)), 
                                                  self.n_points + int(self.n_dims * self.context_shift_amount))) + 1 for _ in range(self.batch_size)]
        else:
            context_lengths = [self.n_points + 1 for _ in range(self.batch_size)]  # All rows have the same context length

        uniform_ps = np.array([random.randrange(self.diversity) for _ in range(self.batch_size)])
        ws = np.array([self.E[:, uniform_ps[i]] for i in range(len(uniform_ps))]) 
        ws = ws[:, :, np.newaxis]  # batch_size x n_dims x 1

        # Generate variable-length xs
        xs_list = [self.rng.normal(loc=0, scale=1/np.sqrt(self.n_dims), size=(context_length, self.n_dims)) for context_length in context_lengths]
        
        # Generate ys based on xs
        ys_list = [xs @ ws[i] + self.rng.normal(loc=0, scale=self.eta_scale, size=(context_lengths[i], 1)) for i, xs in enumerate(xs_list)]
        
        # To return a padded array, we can create Z with the maximum context length, padding where necessary
        max_context_length = max(context_lengths)
        Z = np.zeros((self.batch_size, max_context_length, self.n_dims + 1))

        for i in range(self.batch_size):
            Z[i, :context_lengths[i], :self.n_dims] = xs_list[i]
            Z[i, :context_lengths[i], -1] = ys_list[i].squeeze()
            Z[i, context_lengths[i]:, -1] = 0  # Padding the rest with zeros for the last dimension

        return Z, np.array([ys[-1].squeeze() for ys in ys_list])
    
    def __iter__(self):
        return self