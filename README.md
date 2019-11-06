# PyStan_control_variate
Implementation of control variate with pystan

## Requirements:
Pystan >= 2.19

Tested on Ubuntu 19.04, Python 3.7.5

## Examples
1. Estimate mean and variance of a normal distribution:

python run_stan_example_normal.py

2. Estimate mean and covariance of a (2D) multivariate normal distribution:

python run_stan_example_mv_normal.py

3. Estimate 3 parameters using logistic regression, data is from PyMc3 example:

python run_stan_example_logistic_regression_1.py

4. Estimate 5 parameters using logistic regression, see the experiment, "swiss banknote", in [2]:

python run_stan_example_logistic_regression_2.py

## References:

[1] Mira, Antonietta, Reza Solgi, and Daniele Imparato. "Zero variance markov chain monte carlo for bayesian estimators." Statistics and Computing 23.5 (2013): 653-662.

[2] Papamarkou, Theodore, Antonietta Mira, and Mark Girolami. "Zero variance differential geometric Markov chain Monte Carlo algorithms." Bayesian Analysis 9.1 (2014): 97-128.

