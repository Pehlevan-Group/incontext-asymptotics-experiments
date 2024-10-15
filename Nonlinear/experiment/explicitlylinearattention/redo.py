import numpy as np
from data import fulldiversitysampler
from helpers import *
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys

name = sys.argv[1]
DIM = int(sys.argv[2])
LENGTH = 1*DIM
taus = np.linspace(0.2,2.2,21);
tauind = int(sys.argv[3])
avgind = int(sys.argv[4])
tau = taus[tauind]
errors = []
SAMPLE_SIZE = int(tau * DIM**2)
datamodel = fulldiversitysampler(LENGTH,DIM,0.1,1,1,SAMPLE_SIZE)
Z, YTrue = next(datamodel)
tracking, trained_V, trained_K, trained_Q = train_model(Z, YTrue, lr=0.001, epochs=1000)
# plt.plot(range(1000),tracking)
# plt.savefig('gradientsteps.png')
numtest = 10
avgerr = 0
for _ in range(numtest):
    ZTest, ytest = next(datamodel)
    yhat = linear_attention(ZTest,trained_V,trained_K,trained_Q)
    avgerr += np.mean((yhat - ytest) ** 2)

avgerr = avgerr/numtest
file_path = f'./{name}/error-{tauind}.txt'
with open(file_path, 'a') as file:
    file.write(f'{avgerr}\n')