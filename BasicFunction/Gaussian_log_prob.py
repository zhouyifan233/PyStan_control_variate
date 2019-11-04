import numpy as np
from scipy.stats import multivariate_normal


def log_prob(y, mu, sigma):
    log_prob = []
    for y_ele in y:
        log_prob.extend(multivariate_normal.logpdf(y_ele, mean=mu, cov=sigma))
    log_prob = np.asarray(log_prob)

    return log_prob


def grad_log_prob(y, mu, sigma):
    log_prob = []
    if sigma.ndim == 1:
        # so the inverse matrix can work
        sigma = sigma.reshape((1, 1))
    for y_ele in y:
        log_prob.extend([np.linalg.inv(sigma)@np.expand_dims(y_ele-mu, axis=-1)])
    log_prob = np.asarray(log_prob)
    log_prob = np.squeeze(log_prob, axis=-1)

    return log_prob

