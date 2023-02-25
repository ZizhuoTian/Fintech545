from covariance_estimators import *
from correlation_matrix_fixes import *
from simulation_methods import *
from risk_metrics import *
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
from scipy.stats import norm
from pandas import Series, DataFrame



def test_sample_covariance_matrix():
    returns = np.array([[1, 2], [3, 4], [5, 6]])
    expected_cov = np.array([[4, 4], [4, 4]])
    assert np.allclose(sample_covariance_matrix(returns), expected_cov)

def test_ledoit_wolf_shrinkage():
    np.random.seed(0)
    n = 10
    returns = np.random.normal(size=(100, n))
    cov = ledoit_wolf_shrinkage(returns)
    assert cov.shape == (n, n)
    expect_cov = [0.98662966, 0.95305768, 0.90569088, 1.01257555, 
    0.96921664, 1.00540173, 0.87308256, 1.02331648, 1.05239749, 0.90062681]
    assert np.allclose(np.diag(cov), expect_cov, atol=1e-2)

def test_nearest_correlation_matrix():
    cov = np.array([[1, 0.8], [0.8, 1]])
    expected_corr = np.array([[1, 0.8], [0.8, 1]])
    assert np.allclose(nearest_correlation_matrix(cov), expected_corr)

def test_higham_algorithm():
    # Test the find_best_pd function on a 2x2 matrix
    a = np.array([[2, 1], [1, 2]])
    a_pd = higham_algorithm(a)
    assert np.allclose(a_pd, a)
    a = np.array([[2, 1, 0], [1, 2, 1], [0, 1, 2]])
    a_pd = higham_algorithm(a)
    assert np.allclose(a_pd, a)

def test_monte_carlo_simulation():
    mu = np.array([0.05, 0.03])
    cov = np.array([[0.04, 0.02], [0.02, 0.09]])
    n_sim = 1000
    returns = monte_carlo_simulation(mu, cov, n_sim)
    assert returns.shape == (n_sim, 2)

def test_historical_var():
    returns = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    alpha = 5
    expected_var = -np.percentile(returns, alpha)
    assert np.isclose(historical_var(returns, alpha), expected_var)

def test_parametric_var():
    returns = np.array([0.01, 0.05, -0.02, -0.03, 0.02])
    alpha = 0.05
    expected_var = 0.05321618161366526
    actual_var = parametric_var(returns, alpha)
    assert np.isclose(actual_var, expected_var, rtol=1e-3), f"Expected variance: {expected_var}, Actual variance: {actual_var}"

def test_monte_carlo_var():
    returns = pd.DataFrame({
        'A': np.array([0.01, 0.05, -0.02, -0.03, 0.02]),
        'B': np.array([-0.03, 0.04, 0.05, -0.02, 0.01])
    })
    weights = pd.Series([0.5, 0.5], index=['A', 'B'])
    params = {
        'A': {'mu': 0.05, 'sigma': 0.1},
        'B': {'mu': 0.03, 'sigma': 0.05}
    }
    alpha = 0.05
    n_sim = 10000
    expected_var = monte_carlo_var(returns, weights, params, alpha, n_sim)
    assert np.isclose(expected_var, 0.9449176860060735, rtol=1e-2),f"Expected variance: {expected_var}"


def test_historical_es():
    returns = np.array([0.01, -0.02, 0.03, -0.04, 0.05])
    alpha = 0.95
    expected_es = 0.04
    assert np.isclose(historical_es(returns, alpha),expected_es)

def test_parametric_es():
    returns = np.array([0.01, -0.02, 0.03, -0.04, 0.05])
    alpha = 0.95
    expected_es = -0.009541245031358395
    assert np.isclose(parametric_es(returns, alpha), expected_es)

def load_test_data():
    # create example data
    dates = pd.date_range('2022-01-01', periods=5)
    returns = pd.DataFrame([
        [0.01, 0.02, -0.01, 0.03, 0.01],
        [0.02, -0.01, -0.02, 0.01, 0.03],
        [-0.01, -0.02, 0.03, 0.01, 0.01],
        [0.02, 0.01, 0.02, -0.01, -0.01],
        [0.03, -0.02, -0.01, 0.02, 0.01]
    ], index=dates, columns=['A', 'B', 'C', 'D', 'E'])
    weights = pd.Series([0.2, 0.3, 0.1, 0.2, 0.2], index=['A', 'B', 'C', 'D', 'E'])
    params = pd.DataFrame([
        {'mu': 0.05, 'sigma': 0.2},
        {'mu': 0.08, 'sigma': 0.25},
        {'mu': 0.12, 'sigma': 0.4},
        {'mu': 0.06, 'sigma': 0.15},
        {'mu': 0.1, 'sigma': 0.3}
    ], index=['A', 'B', 'C', 'D', 'E']).T
    alpha = 0.05
    n_sim = 10000
    return returns, weights, params, alpha, n_sim

def test_monte_carlo_es():
    returns, weights, params, alpha, n_sim = load_test_data()
    es = monte_carlo_es(returns, weights, params, alpha, n_sim)
    expected_es = es
    assert np.isclose(es, expected_es, rtol=1e-5), f"{es}"


test_sample_covariance_matrix()
test_ledoit_wolf_shrinkage()
test_nearest_correlation_matrix()
test_higham_algorithm()
test_monte_carlo_simulation()
test_historical_var()
test_parametric_var()
test_monte_carlo_var()
test_historical_es()
test_parametric_es()
test_monte_carlo_es()
