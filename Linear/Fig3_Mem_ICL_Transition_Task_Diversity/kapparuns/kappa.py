import numpy as np
import sys
#from helperfunctions import *
from common import *
from tqdm import tqdm

d = sys.argv[1]
alpha = sys.argv[2]
tau = sys.argv[3];
kappaind = int(sys.argv[4])-1

d = int(d)
alpha = float(alpha)
tau = float(tau)

print(d)
print(alpha)
print(tau)
l = int(alpha * d)  # Context length

rho = 0.01; sigma_noise = np.sqrt(rho); sigma_beta = 1;

lam = 0.000001

numavg = 10;
kappa_sim_ary = [0.2,0.5,1,10,50] #[0.1, 0.5]#, 1, 5, 10, 50, 100, 500] #[0.2, 0.4, 0.6, 1.5] #[0.2,0.5,1,10,50]
kappa = kappa_sim_ary[kappaind]

K = np.int64(kappa * d)
rp = np.int64(np.round(tau * d**2 / K))
n = rp * K
icl_dummy = []; 
for dummy in tqdm(range(numavg)):
  B = np.random.randn(K, d)*sigma_beta;
  beta = np.repeat(B[np.newaxis, :, :], rp, axis=0).reshape(n, d)
  tau_max = 3
  Gamma = learn_Gamma_fast_NEW(beta, alpha, sigma_noise, lam, tau_max)
  icl_dummy.append(gen_err_analytical_NEW(Gamma, alpha, np.zeros(d), sigma_beta**2 * np.eye(d), (sigma_noise/sigma_beta)**2))
  #idg_dummy.append(gen_err_analytical_NEW(Gamma, alpha, np.mean(B,axis=0), (B.T @ B)/K, (sigma_noise/sigma_beta)**2))
print('ANSWER:', np.mean(icl_dummy))
print('STD:', np.std(icl_dummy))

# icl_sim_ary = np.array(icl_sim_ary)
# idg_sim_ary = np.array(idg_sim_ary)
# print("iclmean",list(np.mean(icl_sim_ary,axis=1)))
# print("iclstd",list(np.std(icl_sim_ary,axis=1)))