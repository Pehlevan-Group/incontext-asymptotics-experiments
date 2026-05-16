
import numpy as np

def finite_bayes_estimator_from_sample(X, y, B, K, sigma_noise):
    c = -np.linalg.norm(y - X.T @ B, axis=0)**2/(2*sigma_noise**2)
    ec = np.exp(c - np.max(c))
    beta_hat_finite = B @ ec.reshape(K, 1) / np.sum(ec)
    return beta_hat_finite

def dmmse_estimator_errors(d, alpha, kappa, rho, nsim):
    N = int(alpha*d) # this is still just context length, we call it N here for some reason because it's the number of samples this estimator gets
    K = int(kappa*d) # task diversity
    sigma_noise = np.sqrt(rho) 

    B = np.random.randn(d, K) # these are our fixed omnicient batch of training task that the test data will be compared to
    
    icl_dummy = []
    idg_dummy = []

    for i in range(nsim):
        X = np.random.randn(d, N) / np.sqrt(d)
        beta_idg = B[:, np.random.randint(K)].reshape(d, 1) # old task
        beta_icl = np.random.randn(d, 1)  # new task

        y_idg = X.T @ beta_idg + np.random.randn(N, 1) * sigma_noise
        y_icl = X.T @ beta_icl + np.random.randn(N, 1) * sigma_noise

        beta_hat_idg = finite_bayes_estimator_from_sample(X, y_idg, B, K, sigma_noise)
        beta_hat_icl = finite_bayes_estimator_from_sample(X, y_icl, B, K, sigma_noise)

        # compare betas with beta_hats for icl and idg
        xv = np.random.randn(d, 1)/np.sqrt(d)
        yv_idg = (xv.T @ beta_idg).item() + np.random.randn() * sigma_noise
        yv_icl = (xv.T @ beta_icl).item() + np.random.randn() * sigma_noise

        icl_dummy.append(((xv.T @ beta_hat_icl).item() - yv_icl)**2)
        idg_dummy.append(((xv.T @ beta_hat_idg).item() - yv_idg)**2)
    
    icl_dummy = np.array(icl_dummy); idg_dummy = np.array(idg_dummy);
    return np.mean(icl_dummy), np.std(icl_dummy), np.mean(idg_dummy), np.std(idg_dummy)