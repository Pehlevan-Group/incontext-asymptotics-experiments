import numpy as np
import sys
from common import *
from tqdm import tqdm

directory = sys.argv[1]
# d = sys.argv[2]
kappa = sys.argv[2]
alphaind = int(sys.argv[3]) - 1
d_options = [100,100,100,100]

d = int(d_options[alphaind])
kappa = float(kappa)
print(d)
print(kappa)
K = int(kappa * d)  # Context length

rho = 0.1; sigma_noise = np.sqrt(rho); sigma_beta = 1; 
tau = 0.5;
lam = 0.000000001

numavg = 20;
alpha_sim_ary = [2.35, 5.5, 9.62, 22.7, 100, 1000]
#FIG 2A [1, 6.15, 100, 1000]
#FIG 2B [2.35, 9.62, 100, 1000]
#FIG 2C [1, 4.69, 16.78, 44.94, 79.06, 1000]

alpha = alpha_sim_ary[alphaind]
l = int(alpha * d)
rp = np.int64(np.round(tau * d**2 / K))
n = rp * K
icl_dummy = []; idg_dummy = [];
for dummy in tqdm(range(numavg)):
    B = np.random.randn(K, d)*sigma_beta;
    norms = np.linalg.norm(B, axis=1)
    B = B / norms[:, np.newaxis] * np.sqrt(d)
    beta = np.repeat(B[np.newaxis, :, :], rp, axis=0).reshape(n, d)
    tau_max = 3
    Gamma = learn_Gamma_fast_NEW(beta, alpha, sigma_noise, lam, tau_max)
    icl_dummy.append(gen_err_analytical_NEW(Gamma, alpha, np.zeros(d), sigma_beta**2 * np.eye(d), (sigma_noise/sigma_beta)**2))
    idg_dummy.append(gen_err_analytical_NEW(Gamma, alpha, np.mean(B,axis=0), (B.T @ B)/K, (sigma_noise/sigma_beta)**2))
print("alpha = ", alpha, " and icl = ", np.mean(np.array(icl_dummy)), "with deviation", np.std(np.array(icl_dummy)))
print("alpha = ", alpha, " and idg = ", np.mean(np.array(idg_dummy)), "with deviation", np.std(np.array(idg_dummy)))

filename = f'{directory}/icl_m.txt'
with open(filename, 'a') as file:
    file.write(f'[{alphaind}, {np.mean(np.array(icl_dummy))}],')
filename = f'{directory}/icl_s.txt'
with open(filename, 'a') as file:
    file.write(f'[{alphaind}, { np.std(np.array(icl_dummy))}],')
filename = f'{directory}/idg_m.txt'
with open(filename, 'a') as file:
    file.write(f'[{alphaind}, {np.mean(np.array(idg_dummy))}],')
filename = f'{directory}/idg_s.txt'
with open(filename, 'a') as file:
    file.write(f'[{alphaind}, {np.std(np.array(idg_dummy))}],')