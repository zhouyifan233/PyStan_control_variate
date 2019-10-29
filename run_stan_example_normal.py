import pystan
import numpy as np
from ControlVariate.control_variate import control_variate_linear_mv_gaussian


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
             'n': 1000,
             'y': np.random.multivariate_normal(np.asarray([50, 100]), np.diag([49, 36]), 1000),
            }

#fit = pystan.stan(model_code=norm_code, data=norm_dat, iter=1000, chains=1)
fit = sm.sampling(data=norm_dat, chains=1, iter=100, verbose=False, init=[{'mu': np.asarray([20, 20]), 'Sigma_11': 9, 'Sigma_12': 0, 'Sigma_22': 9}])
print(fit.get_posterior_mean())
print(fit.get_logposterior())

yy = control_variate_linear_mv_gaussian(fit, 'y')
norm_dat['y'] = np.squeeze(yy)
fit = sm.sampling(data=norm_dat, chains=1, iter=100, verbose=False, init=[{'mu': np.asarray([20, 20]), 'Sigma_11': 9, 'Sigma_12': 0, 'Sigma_22': 9}])
print(fit.get_posterior_mean())
print(fit.get_logposterior())

# Parameter estimation in each iteration
data_all = fit.extract(['mu', 'sigma'], inc_warmup=True, permuted=False)
mu_all = data_all['mu']
sigma_all = data_all['sigma']


