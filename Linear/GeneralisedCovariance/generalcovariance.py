import numpy as np
import sys
from common import *

d = sys.argv[1]; d = int(d)
alpha = sys.argv[2]; alpha = float(alpha)
kappa = sys.argv[3]; kappa = float(kappa); K = np.int64(kappa * d)
taus = [0.1,0.3,0.6,0.9,1.1,1.4,1.7,2,2.5]
sigma_noise = 0.1; sigma_beta = 1; rho = (sigma_noise/sigma_beta)**2
lam = 0.000000000001

n_MC = 10  # Number of Monte Carlo runs

Alphas = np.linspace(0.1,10,100)
Ctrain = np.eye(d) #(np.ones((d,d)) + 4*np.eye(d))/d; 
Ctest1 = np.eye(d)
Ctest2 = np.ones((d,d))
Ctest3 = np.diag(np.tile([1, 3], d // 2 + 1)[:d])
Sigma = np.eye(d);
mu1 = np.zeros(d);
mu2 = np.ones(d);
tau_errors_1_1 = []
tau_errors_1_2 = []
tau_errors_1_3 = []
tau_errors_2_1 = []
tau_errors_2_2 = []
tau_errors_2_3 = []
for tau in taus:
    print(tau)
    P = int(tau*d*d)
    testvalue_1_1 = 0; testvalue_1_2 = 0; testvalue_1_3 = 0; testvalue_2_1 = 0; testvalue_2_2 = 0; testvalue_2_3 = 0; 
    for i in range(n_MC):
        H_z, yfinals = context_shift_compute_H_Z_andYs(P, d, int(alpha*d), int(alpha*d), K, rho, Ctrain)
        Gamma_star = compute_Gamma_star(P, d, H_z, yfinals, lam)
        testvalue_1_1 = testvalue_1_1 + e_ICL_g_tr_generalised(Gamma_star, d, alpha, rho, Sigma, Ctest1, mu1) 
        testvalue_1_2 = testvalue_1_2 + e_ICL_g_tr_generalised(Gamma_star, d, alpha, rho, Sigma, Ctest2, mu1) 
        testvalue_1_3 = testvalue_1_3 + e_ICL_g_tr_generalised(Gamma_star, d, alpha, rho, Sigma, Ctest3, mu1) 
        testvalue_2_1 = testvalue_2_1 + e_ICL_g_tr_generalised(Gamma_star, d, alpha, rho, Sigma, Ctest1, mu2) 
        testvalue_2_2 = testvalue_2_2 + e_ICL_g_tr_generalised(Gamma_star, d, alpha, rho, Sigma, Ctest2, mu2)
        testvalue_2_3 = testvalue_2_3 + e_ICL_g_tr_generalised(Gamma_star, d, alpha, rho, Sigma, Ctest3, mu2)
        print(i)
    tau_errors_1_1.append(testvalue_1_1/n_MC)
    tau_errors_1_2.append(testvalue_1_2/n_MC)
    tau_errors_1_3.append(testvalue_1_3/n_MC)
    tau_errors_2_1.append(testvalue_2_1/n_MC)
    tau_errors_2_2.append(testvalue_2_2/n_MC)
    tau_errors_2_3.append(testvalue_2_3/n_MC)

print('C1_mu1 =', tau_errors_1_1)
print('C2_mu1 =', tau_errors_1_2)
print('C3_mu1 =', tau_errors_1_3)
print('C1_mu2 =', tau_errors_2_1)
print('C2_mu2 =', tau_errors_2_2)
print('C3_mu2 =', tau_errors_2_3)