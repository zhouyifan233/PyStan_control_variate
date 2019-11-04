import pystan
import numpy as np
import pandas as pd
from CalculateGradLogProb.logitModel import grad_log_prob
from ControlVariate.control_variate import control_variate_linear


log_reg_code = """
data {
    int<lower=0> n;
    int male[n];
    real weight[n];
    real height[n];
}
transformed data {}
parameters {
    real a;
    real b;
    real c;
}
transformed parameters {}
model {
    a ~ normal(0, 10);
    b ~ normal(0, 10);
    c ~ normal(0, 10);
    for(i in 1:n) {
        male[i] ~ bernoulli(inv_logit(a*weight[i] + b*height[i] + c));
  }
}
generated quantities {}
"""
sm = pystan.StanModel(model_code=log_reg_code)


df = pd.read_csv('data/HtWt.csv')
df.head()

log_reg_dat = {
             'n': len(df),
             'male': df.male,
             'height': df.height,
             'weight': df.weight
            }

fit = sm.sampling(data=log_reg_dat, iter=10000, chains=1)

# Extract parameters
parameter_extract = fit.extract()
a = parameter_extract['a']
b = parameter_extract['b']
c = parameter_extract['c']

# Constraint mcmc samples
mcmc_samples = []
for i in range(len(a)):
    mcmc_samples.append([a[i], b[i], c[i]])
mcmc_samples = np.asarray(mcmc_samples)

# Calculate gradients of the log-probability
num_of_iter = mcmc_samples.shape[0]
grad_log_prob_val = []
for i in range(num_of_iter):
    grad_log_prob_val.append(fit.grad_log_prob(mcmc_samples[i], adjust_transform=False))
grad_log_prob_val = np.asarray(grad_log_prob_val)
print(np.mean(grad_log_prob_val, axis=0))

"""
# Not using auto-diff: 
x1 = np.asarray(df.weight)
x2 = np.asarray(df.height)
x3 = np.ones_like(x1)
x = np.asarray([x1, x2, x3]).T
y = np.asarray(df.male)
num_of_iter = mcmc_samples.shape[0]
grad_log_prob_val_ = []
for i in range(num_of_iter):
    grad_log_prob_val_.append(grad_log_prob(x, y, mcmc_samples[i]))
grad_log_prob_val_ = np.asarray(grad_log_prob_val_)
print(np.mean(grad_log_prob_val_, axis=0))
"""

# Run control variates
yy = control_variate_linear(mcmc_samples, grad_log_prob_val)
