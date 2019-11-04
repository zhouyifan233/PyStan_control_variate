import numpy as np


def grad_log_prob(x, y, parameters):
    tmp = y - 1/(1 + np.exp(- x @ parameters))
    grad_val = x.T @ tmp

    return grad_val