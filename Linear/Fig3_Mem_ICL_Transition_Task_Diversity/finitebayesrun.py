import numpy as np
from tqdm import tqdm 
import sys

def S_W(c, alpha):
    return 2/(c+alpha-1 + np.sqrt((c+alpha-1)**2 + 4*c))

d = 200
directory = sys.argv[1]
alpha = float(sys.argv[2]) 
kappaind = int(sys.argv[3]) - 1
sigma_noise = float(sys.argv[4])
sigma_beta = 1

K_array = list(np.int64(np.logspace(np.log10(0.01*d),np.log10(1000*d),50))); #list(np.int64(np.logspace(np.log10(0.05*d),np.log10(500*d),40)));
K = K_array[kappaind]
N = np.int64(alpha * d)

nsim = 500000

e_B_full_ary = np.zeros(nsim)
e_B_finite_ary = np.zeros(nsim)

# K = int(kappa*d);
print(f'kappa is {K/d}')
B = np.random.randn(d, K)

IsFinite = True
for i in range(nsim):
    X = np.random.randn(d, N) / np.sqrt(d)
    if IsFinite:
        beta = B[:, np.random.randint(K)].reshape(d, 1)
    else:
        beta = np.random.randn(d, 1) * sigma_beta

    y = X.T @ beta + np.random.randn(N, 1) * sigma_noise

    # Bayesian estimator for the Gaussian distribution
    beta_hat = np.linalg.solve(X @ X.T + sigma_noise**2/sigma_beta**2 * np.eye(d), X @ y)

    # Bayesian estimator for the finite distribution
    c = -np.linalg.norm(y - X.T @ B, axis=0)**2/(2*sigma_noise**2)
    ec = np.exp(c - np.max(c))
    beta_hat_finite = B @ ec.reshape(K, 1) / np.sum(ec)

    xv = np.random.randn(d, 1)/np.sqrt(d)
    yv = (xv.T @ beta).item() + np.random.randn() * sigma_noise

    e_B_full_ary[i] = ((xv.T @ beta_hat).item() - yv)**2
    e_B_finite_ary[i] = ((xv.T @ beta_hat_finite).item() - yv)**2

ind = kappaind
filename = f'{directory}/idg_dmmse_m.txt'
with open(filename, 'a') as file:
    file.write(f'{ind}, {np.mean(e_B_finite_ary)}\n')
filename = f'{directory}/idg_ridge_m.txt'
with open(filename, 'a') as file:
    file.write(f'{ind}, {np.mean(e_B_full_ary)}\n')
filename = f'{directory}/idg_dmmse_s.txt'
with open(filename, 'a') as file:
    file.write(f'{ind}, {np.std(e_B_finite_ary)}\n')
filename = f'{directory}/idg_ridge_s.txt'
with open(filename, 'a') as file:
    file.write(f'{ind}, {np.std(e_B_full_ary)}\n')

# IsFinite = False
# e_B_full_ary = np.zeros(nsim)
# e_B_finite_ary = np.zeros(nsim)
# for i in range(nsim):
#     X = np.random.randn(d, N) / np.sqrt(d)
#     if IsFinite:
#         beta = B[:, np.random.randint(K)].reshape(d, 1)
#     else:
#         beta = np.random.randn(d, 1) * sigma_beta

#     y = X.T @ beta + np.random.randn(N, 1) * sigma_noise

#     # Bayesian estimator for the Gaussian distribution
#     beta_hat = np.linalg.solve(X @ X.T + sigma_noise**2/sigma_beta**2 * np.eye(d), X @ y)

#     # Bayesian estimator for the finite distribution
#     c = -np.linalg.norm(y - X.T @ B, axis=0)**2/(2*sigma_noise**2)
#     ec = np.exp(c - np.max(c))
#     beta_hat_finite = B @ ec.reshape(K, 1) / np.sum(ec)

#     xv = np.random.randn(d, 1)/np.sqrt(d)
#     yv = (xv.T @ beta).item() + np.random.randn() * sigma_noise

#     e_B_full_ary[i] = ((xv.T @ beta_hat).item() - yv)**2
#     e_B_finite_ary[i] = ((xv.T @ beta_hat_finite).item() - yv)**2

# ind = kappaind
# filename = f'{directory}/icl_dmmse_m.txt'
# with open(filename, 'a') as file:
#     file.write(f'[{ind}, {np.mean(e_B_finite_ary)}],')
# filename = f'{directory}/icl_ridge_m.txt'
# with open(filename, 'a') as file:
#     file.write(f'[{ind}, {np.mean(e_B_full_ary)}],')
# filename = f'{directory}/icl_dmmse_s.txt'
# with open(filename, 'a') as file:
#     file.write(f'[{ind}, {np.std(e_B_finite_ary)}],')
# filename = f'{directory}/icl_ridge_s.txt'
# with open(filename, 'a') as file:
#     file.write(f'[{ind}, {np.std(e_B_full_ary)}],')
