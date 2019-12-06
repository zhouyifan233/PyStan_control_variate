import numpy as np
from PyStan_control_variate.BasicFunction.Gaussian_log_prob import grad_log_prob


def control_variate_linear(mcmc_samples, mcmc_gradients):
    dim = mcmc_samples.shape[1]
    control = -0.5 * mcmc_gradients

    sc_matrix = np.concatenate((mcmc_samples, control), axis=1)
    sc_cov = np.cov(sc_matrix.T)
    Sigma_cs = sc_cov[0:dim, dim:dim * 2].T
    Sigma_cc = sc_cov[dim:dim*2, dim:dim*2]
    zv = (-np.linalg.inv(Sigma_cc) @ Sigma_cs).T @ control.T
    new_mcmc_samples = mcmc_samples + zv.T

    print('new_mcmc_samples variance: ')
    print(np.var(new_mcmc_samples, axis=0))
    print('old_mcmc_samples variance:')
    print(np.var(mcmc_samples, axis=0))

    return new_mcmc_samples


def control_variate_quadratic(mcmc_samples, mcmc_gradients):
    num_mcmc = mcmc_samples.shape[0]
    dim = mcmc_samples.shape[1]
    dim_cp = int(0.5*dim*(dim-1))

    dim_control = dim+dim+dim_cp
    z = -0.5 * mcmc_gradients
    # It is weid here: should I use +0.5 or -0.5 ?
    control = np.concatenate((z, (mcmc_samples*mcmc_gradients - 0.5)), axis=1)
    control_parts = np.zeros((num_mcmc, dim_cp))
    for i in range(2, dim+1):
        for j in range(1, i):
            ind = int(0.5*(2*dim-j)*(j-1) + (i-j))
            control_parts[:,ind-1] = mcmc_samples[:,i-1]*mcmc_gradients[:,j-1] + mcmc_samples[:,j-1]*mcmc_gradients[:,i-1]
    control = np.concatenate((control, control_parts), axis=1)
    sc_matrix = np.concatenate((mcmc_samples.T, control.T), axis=0)
    sc_cov = np.cov(sc_matrix)
    Sigma_cs = sc_cov[0:dim, dim:dim+dim_control].T
    Sigma_cc = sc_cov[dim:dim+dim_control, dim:dim+dim_control]

    zv = (-np.linalg.inv(Sigma_cc) @ Sigma_cs).T @ control.T
    new_mcmc_samples = mcmc_samples + zv.T

    print('new_mcmc_samples variance: ')
    print(np.var(new_mcmc_samples, axis=0))
    print('old_mcmc_samples variance:')
    print(np.var(mcmc_samples, axis=0))

    return new_mcmc_samples
