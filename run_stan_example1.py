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
             'n': 100,
             'y': np.random.normal(10, 2, 100),
            }

#fit = pystan.stan(model_code=norm_code, data=norm_dat, iter=1000, chains=1)
sm = pystan.StanModel(model_code=norm_code)
fit = sm.sampling(data=norm_dat, iter=1000)
print(fit)

# Calculate log-prob
# fit.log_prob(fit.unconstrain_pars({'mu':10, 'sigma':2}))
