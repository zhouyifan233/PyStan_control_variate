import pystan
import numpy as np
from ControlVariate.control_variate import control_variate_linear, control_variate_quadratic
from ControlVariate.plot_comparison import plot_comparison


def run_example():
    norm_code = """
    data {
        int<lower=0> n;
        vector[2] y[n];
    }
    transformed data {}
    parameters {
        vector[2] mu;
        real Sigma_11;
        real Sigma_22;
        real Sigma_12;
    }
    transformed parameters {
        matrix[2, 2] Sigma;
        Sigma[1, 1] = Sigma_11;
        Sigma[1, 2] = Sigma_12;
        Sigma[2, 1] = Sigma_12;
        Sigma[2, 2] = Sigma_22;
    }
    model {
        y ~ multi_normal(mu, Sigma);
    }
    generated quantities {}
    """
    sm = pystan.StanModel(model_code=norm_code)

    norm_dat = {
                 'n': 10000,
                 'y': np.random.multivariate_normal(np.asarray([100, 100]), np.diag([100, 100]), 10000),
                }

    fit = sm.sampling(data=norm_dat, chains=1, iter=1000, verbose=False, init=[{'mu': np.asarray([20, 20]), 'Sigma_11': 25, 'Sigma_12': 0, 'Sigma_22': 25}])

    # Extract parameters
    parameter_extract = fit.extract()
    mu = parameter_extract['mu']
    Sigma_11 = parameter_extract['Sigma_11']
    Sigma_22 = parameter_extract['Sigma_22']
    Sigma_12 = parameter_extract['Sigma_12']

    # Constraint mcmc samples
    mcmc_samples = []
    for i in range(len(mu)):
        mu_list = mu[i].tolist()
        mu_list.extend([Sigma_11[i], Sigma_22[i], Sigma_12[i]])
        mcmc_samples.append(mu_list)
    mcmc_samples = np.asarray(mcmc_samples)

    # Calculate gradients of the log-probability
    num_of_iter = mcmc_samples.shape[0]
    grad_log_prob_val = []
    for i in range(num_of_iter):
        grad_log_prob_val.append(fit.grad_log_prob(mcmc_samples[i], adjust_transform=False))
    grad_log_prob_val = np.asarray(grad_log_prob_val)

    # Run control variates
    cv_linear_mcmc_samples = control_variate_linear(mcmc_samples, grad_log_prob_val)
    cv_quad_mcmc_samples = control_variate_quadratic(mcmc_samples, grad_log_prob_val)
    plot_comparison(mcmc_samples, cv_linear_mcmc_samples, cv_quad_mcmc_samples, fig_name='mv_normal.png', fig_size=(8, 24))

    return mcmc_samples, cv_linear_mcmc_samples, cv_quad_mcmc_samples


if __name__ == "__main__":
    mcmc_samples, cv_linear_mcmc_samples, cv_quad_mcmc_samples = run_example()
