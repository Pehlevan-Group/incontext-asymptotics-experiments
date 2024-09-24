import numpy as np
import sys
from common import *

d = sys.argv[1]; d = int(d)
alpha = sys.argv[2]; alpha = float(alpha)
kappa = sys.argv[3]; kappa = float(kappa); K = np.int64(kappa * d)
tau = sys.argv[4]; tau = float(tau); P = int(tau*(d**2))
alphashift_low = sys.argv[5]; alphashift_low = float(alphashift_low)
alphashift_high = sys.argv[6]; alphashift_high = float(alphashift_high)

sigma_noise = 0.1; sigma_beta = 1; rho = (sigma_noise/sigma_beta)**2
lam = 0.000000000001

n_MC = 10  # Number of Monte Carlo runs

Alphas = np.linspace(0.1,5,50)
final_vals = []
for i in range(n_MC):
    H_z, yfinals = context_shift_compute_H_Z_andYs(P, d, int(alphashift_low*d), int(alphashift_low*d), K, rho)
    Gamma_star = compute_Gamma_star(P, d, H_z, yfinals, lam)
    Alpha_tests = [e_ICL_g_tr(Gamma_star, d, Alpha, rho) for Alpha in Alphas]
    print(Alpha_tests)
    final_vals.append(Alpha_tests)

final_vals = np.array(final_vals)
print(list(np.mean(final_vals, axis=0)))