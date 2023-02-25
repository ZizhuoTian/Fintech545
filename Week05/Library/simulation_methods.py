import numpy as np

def monte_carlo_simulation(mu, cov, n_sim):
    n_assets = len(mu)
    returns = np.random.multivariate_normal(mu, cov, size=n_sim)
    return returns
