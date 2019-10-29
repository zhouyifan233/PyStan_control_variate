import pystan
import numpy as np
from ControlVariate.control_variate import control_variate_linear_gaussian


norm_code = """
data {
    int<lower=0> n;
    real y[n];
}
transformed data {}
parameters {
    real<lower=0, upper=100> mu;
    real<lower=0, upper=10> sigma;
}
transformed parameters {}
model {
    y ~ normal(mu, sigma);
}
generated quantities {}
"""
sm = pystan.StanModel(model_code=norm_code)

norm_dat = {
             'n': 2000,
             'y': np.random.normal(50, 10, 2000),
            }

#fit = pystan.stan(model_code=norm_code, data=norm_dat, iter=1000, chains=1)
fit = sm.sampling(data=norm_dat, chains=1, iter=50, verbose=True, init=[{'mu':5, 'sigma':1}])
print(fit.get_posterior_mean())
print(fit.get_logposterior())

#y = norm_dat['y']
#print(np.mean(y))
#print(np.var(y))
yy = control_variate_linear_gaussian(fit, 'y')
#print(np.mean(yy))
#print(np.var(yy))
norm_dat['y'] = np.squeeze(yy)
fit = sm.sampling(data=norm_dat, chains=1, iter=50, verbose=True, init=[{'mu':5, 'sigma':1}])
print(fit.get_posterior_mean())
print(fit.get_logposterior())

# Calculate log-prob
# fit.log_prob(fit.unconstrain_pars({'mu':10, 'sigma':2}))

# Parameter estimation in each iteration
data_all = fit.extract(['mu', 'sigma'], inc_warmup=True, permuted=False)
mu_all = data_all['mu']
sigma_all = data_all['sigma']


