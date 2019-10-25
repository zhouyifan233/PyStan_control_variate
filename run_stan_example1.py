import pystan
import numpy as np


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

norm_dat = {
             'n': 50,
             'y': np.random.normal(10, 2, 50),
            }

#fit = pystan.stan(model_code=norm_code, data=norm_dat, iter=1000, chains=1)
sm = pystan.StanModel(model_code=norm_code)
fit = sm.sampling(data=norm_dat, chains=2, iter=3, verbose=True, init=[{'mu':0.1, 'sigma':0.1}, {'mu':5, 'sigma':5}])
print(fit)

# Calculate log-prob
# fit.log_prob(fit.unconstrain_pars({'mu':10, 'sigma':2}))

# Parameter estimation in each iteration
data_all = fit.extract(['mu', 'sigma'], inc_warmup=True, permuted=False)
mu_all = data_all['mu']
sigma_all = data_all['sigma']


