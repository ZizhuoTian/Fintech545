from pickletools import optimize
import numpy as np
import pandas as pd
from scipy.optimize import minimize

import numpy as np

def sample_covariance_matrix(returns):
    return np.cov(returns, rowvar=False)

def ledoit_wolf_shrinkage(returns):
    n = returns.shape[1]
    sample_cov = sample_covariance_matrix(returns)
    mu = np.mean(returns, axis=0)
    delta = np.mean(np.square(returns - mu), axis=0)

    delta_bar = np.mean(delta)
    d_bar = np.sum(np.square(sample_cov - delta_bar * np.eye(n))) / n / (n-1)
    d2 = delta - delta_bar
    d2_bar = np.sum(np.square(d2)) / n
    gamma = d_bar / d2_bar

    shrinkage_param = np.min([1, np.max([0, (n - 2) / n * gamma])])
    shrunk_cov = shrinkage_param * sample_cov + (1 - shrinkage_param) * delta_bar * np.eye(n)

    return shrunk_cov

def fit_generalized_t(returns):
    sample_mean = np.mean(returns)
    sample_std = np.std(returns, ddof=1)
    sample_var = sample_std ** 2
    n = len(returns)
    p = 3

    def objective_function(params):
        nu, mu, sigma = params

    initial_guess = [2.1, sample_mean, sample_std]
    bounds = [(1e-6, None), (None, None), (1e-6, None)]

    result = minimize(objective_function, initial_guess, bounds=bounds)
    nu, mu, sigma = result.x
    return pd.Series({'mu': mu, 'sigma': sigma})



