import pystan
import numpy as np
from ControlVariate.control_variate import control_variate_linear, control_variate_quadratic
from ControlVariate.plot_comparison import plot_comparison


def run_example():
    norm_code = """
    data {
        int<lower=0> n;
        real y[n];
    }
    transformed data {}
    parameters {
        real mu;
        real sigma;
    }
    transformed parameters {}
    model {
        y ~ normal(mu, sigma);
    }
    generated quantities {}
    """
    sm = pystan.StanModel(model_code=norm_code)

    norm_dat = {
                 'n': 500,
                 'y': np.random.normal(50, 100, 500),
                }

    #fit = pystan.stan(model_code=norm_code, data=norm_dat, iter=1000, chains=1)
    fit = sm.sampling(data=norm_dat, chains=1, iter=1000, verbose=True)

    # Extract parameters
    parameter_extract = fit.extract()
    mu = parameter_extract['mu']
    sigma = parameter_extract['sigma']

    # Constraint mcmc samples
    mcmc_samples = []
    for i in range(len(mu)):
        mcmc_samples.append([mu[i], sigma[i]])
    mcmc_samples = np.asarray(mcmc_samples)

    # Unconstraint mcmc samples. Not useful here.
    unconstrain_mcmc_samples = []
    for i in range(len(mu)):
        unconstrain_mcmc_samples.append(fit.unconstrain_pars({'mu': mu[i], 'sigma': sigma[i]}))
    unconstrain_mcmc_samples = np.asarray(unconstrain_mcmc_samples)

    # Calculate gradients of the log-probability
    # In this case, it seems unconstraint and constraint parameters are the same.
    num_of_iter = mcmc_samples.shape[0]
    grad_log_prob_val = []
    for i in range(num_of_iter):
        grad_log_prob_val.append(fit.grad_log_prob(mcmc_samples[i], adjust_transform=False))
    grad_log_prob_val = np.asarray(grad_log_prob_val)

    # Run control variates
    cv_linear_mcmc_samples = control_variate_linear(mcmc_samples, grad_log_prob_val)
    cv_quad_mcmc_samples = control_variate_quadratic(mcmc_samples, grad_log_prob_val)
    plot_comparison(mcmc_samples, cv_linear_mcmc_samples, cv_quad_mcmc_samples, fig_name='normal.png', fig_size=(8, 8))

    return mcmc_samples, cv_linear_mcmc_samples, cv_quad_mcmc_samples


if __name__ == "__main__":
    mcmc_samples, cv_linear_mcmc_samples, cv_quad_mcmc_samples = run_example()
