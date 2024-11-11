import numpy as np
from tqdm import tqdm 
import sys

def S_W(c, alpha):
    return 2/(c+alpha-1 + np.sqrt((c+alpha-1)**2 + 4*c))

d_index = int(sys.argv[1])-1
directory = sys.argv[2]
alpha = float(sys.argv[3]) 
kappa = float(sys.argv[4])
sigma_noise = float(sys.argv[5])
nsim = int(sys.argv[6])
sigma_beta = 1

d_list = [40,60,80,100,120,140,160,180,200,220,240,260,280,300] 
d = d_list[d_index]
N = int(alpha * d)
K = int(kappa*d)


e_B_finite_ary = np.zeros(nsim)

print(f'd = {d}, alpha = {alpha}, kappa = {kappa}, rho = {sigma_noise**2}')

IsFinite = True
for i in tqdm(range(nsim)):
    B = np.random.randn(d, K)
    X = np.random.randn(d, N) / np.sqrt(d)
    if IsFinite:
        beta = B[:, np.random.randint(K)].reshape(d, 1)
    else:
        beta = np.random.randn(d, 1) * sigma_beta

    y = X.T @ beta + np.random.randn(N, 1) * sigma_noise

    # Bayesian estimator for the finite distribution
    c = -np.linalg.norm(y - X.T @ B, axis=0)**2/(2*sigma_noise**2)
    ec = np.exp(c - np.max(c))
    beta_hat_finite = B @ ec.reshape(K, 1) / np.sum(ec)

    xv = np.random.randn(d, 1)/np.sqrt(d)
    yv = (xv.T @ beta).item() + np.random.randn() * sigma_noise

    e_B_finite_ary[i] = ((xv.T @ beta_hat_finite).item() - yv)**2

ind = d_index
filename = f'{directory}/idg_dmmse_m.txt'
with open(filename, 'a') as file:
    file.write(f'[{ind}, {np.mean(e_B_finite_ary)}],')
filename = f'{directory}/idg_dmmse_s.txt'
with open(filename, 'a') as file:
    file.write(f'[{ind}, {np.std(e_B_finite_ary)}],')

IsFinite = False
e_B_finite_ary = np.zeros(nsim)
for i in range(nsim):
    B = np.random.randn(d, K)
    X = np.random.randn(d, N) / np.sqrt(d)
    if IsFinite:
        beta = B[:, np.random.randint(K)].reshape(d, 1)
    else:
        beta = np.random.randn(d, 1) * sigma_beta

    y = X.T @ beta + np.random.randn(N, 1) * sigma_noise

    # Bayesian estimator for the finite distribution
    c = -np.linalg.norm(y - X.T @ B, axis=0)**2/(2*sigma_noise**2)
    ec = np.exp(c - np.max(c))
    beta_hat_finite = B @ ec.reshape(K, 1) / np.sum(ec)

    xv = np.random.randn(d, 1)/np.sqrt(d)
    yv = (xv.T @ beta).item() + np.random.randn() * sigma_noise

    e_B_finite_ary[i] = ((xv.T @ beta_hat_finite).item() - yv)**2

ind = d_index
filename = f'{directory}/icl_dmmse_m.txt'
with open(filename, 'a') as file:
    file.write(f'[{ind}, {np.mean(e_B_finite_ary)}],')
filename = f'{directory}/icl_dmmse_s.txt'
with open(filename, 'a') as file:
    file.write(f'[{ind}, {np.std(e_B_finite_ary)}],')
