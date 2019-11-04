import numpy as np
from BasicFunction.Gaussian_log_prob import grad_log_prob


def control_variate_linear(mcmc_samples, mcmc_gradients):
    dim_samples = mcmc_samples.shape[1]
    control = -0.5 * mcmc_gradients

    sc_matrix = np.concatenate((mcmc_samples, control), axis=1)
    sc_cov = np.cov(sc_matrix.T)
    Sigma_cs = sc_cov[0:dim_samples, dim_samples:dim_samples * 2].T
    Sigma_cc = sc_cov[dim_samples:dim_samples*2, dim_samples:dim_samples*2]
    zv = -np.linalg.inv(Sigma_cc) @ Sigma_cs @ control.T
    new_mcmc_samples = mcmc_samples + zv.T

    print('new_mcmc_samples variance: ')
    print(np.var(new_mcmc_samples, axis=0))
    print('old_mcmc_samples variance:')
    print(np.var(mcmc_samples, axis=0))

    return new_mcmc_samples


def control_variate_qudratic(mcmc_samples, mcmc_gradients):

    return 0
