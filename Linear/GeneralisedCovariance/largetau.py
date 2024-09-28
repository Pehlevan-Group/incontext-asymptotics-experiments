import numpy as np
import sys
from common import *

d = sys.argv[1]; d = int(d)
alpha = sys.argv[2]; alpha = float(alpha)
kappa = sys.argv[3]; kappa = float(kappa); K = np.int64(kappa * d)
tau = sys.argv[4]; tau = float(tau)
alphashift_low = sys.argv[5]; alphashift_low = float(alphashift_low)
alphashift_high = sys.argv[6]; alphashift_high = float(alphashift_high)
lengthvarbool = bool(sys.argv[7])

sigma_noise = 0.1; sigma_beta = 1; rho = (sigma_noise/sigma_beta)**2
lam = 0.1
numavg = 10;

print("d ", d)
print("alpha", alpha)
print("kappa",kappa)
print("tau",tau)
print(f'alpha shifts are {max(0.1,alpha-alphashift_low)} to {alpha+alphashift_high}')
print("lambda",lam)

rp = int(tau * d**2 / K)
n = rp * K

Alphas = np.linspace(0.1,5,50)
errvals = []

for dummy in range(numavg):
    print("iteration ",dummy)
    B = np.random.randn(K, d)*sigma_beta;
    norms = np.linalg.norm(B, axis=1)
    B = B / norms[:, np.newaxis] * np.sqrt(d)

    beta = np.repeat(B[np.newaxis, :, :], rp, axis=0).reshape(n, d)
    tau_max = 3
    Gamma = learn_Gamma_fast_NEW(beta, alpha, alphashift_low, alphashift_high, sigma_noise, lam, tau_max, lengthvarbool)

    Alpha_tests = [gen_err_analytical_NEW(Gamma, Alpha, np.zeros(d), sigma_beta**2 * np.eye(d), (sigma_noise/sigma_beta)**2) for Alpha in Alphas]
    print(Alpha_tests)
    errvals.append(Alpha_tests)

print("mean",list(np.mean(errvals,axis=0)))
print("std",list(np.std(errvals,axis=0)))