import numpy as np
from scipy.stats import norm

class GaussianCopula:
    def __init__(self, dim=None, corr=None):
        self.dim = dim
        self.corr = corr
        self.norms_inv = norm.ppf(np.linspace(1e-6, 1-1e-6, 10000))
        
    def fit(self, data):
        self.dim = data.shape[1]
        self.corr = np.corrcoef(data, rowvar=False)
        
    def sample(self, n=1):
        z = np.random.normal(size=(n, self.dim))
        z = norm.cdf(z)
        samples = []
        for i in range(n):
            r = np.linalg.cholesky(self.corr)
            x = np.matmul(r, z[i].reshape(self.dim, 1)).reshape(-1)
            samples.append(x)
        return np.array(samples)
