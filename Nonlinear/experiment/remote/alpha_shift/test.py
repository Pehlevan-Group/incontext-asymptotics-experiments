import numpy as np
import optax
import pickle
import sys
sys.path.append('../../')
sys.path.append('../../../')
from task.regression import FiniteTasksVariableContext

d = 10
myobject = FiniteTasksVariableContext(n_points=5, variable_context=True, context_shift_amount=1, n_dims = d, eta_scale = 0.1, w_scale= 1, diversity=1000, batch_size=7)
xs, labels = next(myobject)
print(xs[0])
print(xs[1])
print(xs[2])
print(xs.shape)
print(labels.shape)