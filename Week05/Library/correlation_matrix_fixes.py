from scipy.linalg import eigvalsh, sqrtm
import numpy as np

def nearest_correlation_matrix(cov):
    d = np.sqrt(np.diag(cov))
    corr = cov / np.outer(d, d)
    corr[corr < -1] = -1
    corr[corr > 1] = 1
    return corr


def higham_algorithm(cov, method='nearest_correlation_matrix'):
    if not np.allclose(cov, cov.T):
        raise ValueError("Input matrix is not symmetric.")
    eigs, _ = np.linalg.eigh(cov)
    if np.all(eigs >= 0):
        return cov
    elif method == 'nearest_correlation_matrix':
        corr = nearest_correlation_matrix(cov)
        fixed_cov = np.diag(np.sqrt(np.diag(cov))) @ corr @ np.diag(np.sqrt(np.diag(cov)))
        return fixed_cov
    else:
        raise ValueError("Invalid method specified.")


