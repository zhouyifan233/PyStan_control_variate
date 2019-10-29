import numpy as np
from BasicFunction.Gaussian_log_prob import grad_log_prob


def control_variate_linear_gaussian(fit_model, sample_par='y'):
    samples = fit_model.data
    samples = samples[sample_par]
    num_samples = samples.shape[0]
    if samples.ndim == 1:
        dim_samples = 1
        samples = samples.reshape((num_samples, 1))
    else:
        dim_samples = samples.shape[1]
    # Obtain gradient of log probability of all the samples
    mu = fit_model.extract('mu')
    mu = np.mean(mu['mu'], keepdims=True)
    sigma = fit_model.extract('sigma')
    sigma = np.mean(sigma['sigma'], keepdims=True)
    grad_log_prob_val = grad_log_prob(samples, mu, sigma)
    control = -0.5*grad_log_prob_val
    # sample and control matrix
    sc_matrix = np.concatenate((samples, control), axis=1)
    sc_cov = np.cov(sc_matrix.T)
    Sigma_cs = sc_cov[0:dim_samples, dim_samples:dim_samples*2]
    Sigma_cc = sc_cov[0:dim_samples, 0:dim_samples]
    zv = -np.linalg.inv(Sigma_cc)*Sigma_cs*control.T
    new_samples = samples.T + zv
    return new_samples.T


def control_variate_linear_mv_gaussian(fit_model, sample_par='y'):
    samples = fit_model.data
    samples = samples[sample_par]
    num_samples = samples.shape[0]
    if samples.ndim == 1:
        dim_samples = 1
        samples = samples.reshape((num_samples, 1))
    else:
        dim_samples = samples.shape[1]
    # Obtain gradient of log probability of all the samples
    mu = fit_model.extract('mu')
    mu = np.mean(mu['mu'], axis=0, keepdims=True)
    Sigma_extract = fit_model.extract(['Sigma_11', 'Sigma_12', 'Sigma_22'])
    Sigma_11 = np.mean(Sigma_extract['Sigma_11'], axis=0, keepdims=True)
    Sigma_12 = np.mean(Sigma_extract['Sigma_12'], axis=0, keepdims=True)
    Sigma_22 = np.mean(Sigma_extract['Sigma_22'], axis=0, keepdims=True)
    Sigma = np.asarray([[Sigma_11, Sigma_12], [Sigma_12, Sigma_22]])
    mu = np.squeeze(mu)
    Sigma = np.squeeze(Sigma)
    grad_log_prob_val = grad_log_prob(samples, mu, Sigma)
    control = -0.5 * grad_log_prob_val
    # sample and control matrix
    sc_matrix = np.concatenate((samples, control), axis=1)
    sc_cov = np.cov(sc_matrix.T)
    Sigma_cs = sc_cov[0:dim_samples, dim_samples:dim_samples * 2]
    Sigma_cc = sc_cov[0:dim_samples, 0:dim_samples]
    zv = -np.linalg.inv(Sigma_cc) @ Sigma_cs @ control.T
    new_samples = samples.T + zv
    return new_samples.T


def control_variate_quadratic_gaussian():
    return 0

