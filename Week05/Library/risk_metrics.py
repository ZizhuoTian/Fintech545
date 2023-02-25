from scipy.stats import norm
from simulation_methods import monte_carlo_simulation
import numpy as np
import pandas as pd

def historical_var(returns, alpha):
    return -np.percentile(returns, alpha)

def parametric_var(returns, alpha):
    mu = np.mean(returns)
    std = np.std(returns)
    z_alpha = norm.ppf(1-alpha)
    return mu + z_alpha * std


def monte_carlo_var(returns, weights, params, alpha, n_sim):
    sample_mean = np.zeros(len(weights))
    for i, stock in enumerate(weights.index):
        sample_mean[i] = returns[stock].mean()
    sample_cov = np.zeros((len(weights), len(weights)))
    sample_cov = np.diag(np.full(len(weights), params[stock]['sigma']**2))
    L = np.linalg.cholesky(sample_cov)
    n_stocks = len(weights)
    eps = np.random.normal(size=(n_sim, n_stocks))
    z = eps @ L.T
    sim_returns = np.exp(z)
    sim_portfolio_returns = (sim_returns * weights.values).sum(axis=1)
    var_i = np.percentile(sim_portfolio_returns, alpha*100)
    return var_i

def historical_es(returns, alpha):
    var = historical_var(returns, alpha)
    return -np.mean(returns[returns < -var])

def parametric_es(returns, alpha):
    mu = np.mean(returns)
    std = np.std(returns)
    z_alpha = norm.ppf(1-alpha)
    return -(mu + std / alpha * norm.pdf(z_alpha))

def monte_carlo_es(returns, weights, params, alpha, n_sim):
    sorted_returns = np.sort(returns)
    var = np.percentile(sorted_returns, alpha * 100)
    portfolio_value = weights.sum()
    var *= portfolio_value
    worst_returns = sorted_returns[sorted_returns <= var]
    es = worst_returns.mean()
    es *= portfolio_value / (1 - alpha)
    simulated_returns = np.random.normal(params.T['mu'].values, params.T['sigma'].values, size=(n_sim, len(weights)))
    simulated_portfolio_returns = (simulated_returns * weights.values).sum(axis=1)
    sorted_simulated_returns = np.sort(simulated_portfolio_returns)
    var_sim = np.percentile(sorted_simulated_returns, alpha * 100)
    es = sorted_simulated_returns[sorted_simulated_returns <= var_sim].mean()
    es *= portfolio_value / (1 - alpha)
    return es

