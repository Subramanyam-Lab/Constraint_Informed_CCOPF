import numpy as np
from sklearn.mixture import GaussianMixture

np.random.seed(125)


def generate_cauchy_samples(N, D, scale):
    return np.random.standard_cauchy(size=(N, D)) * scale


def fit_gaussian(samples, n_components=1):
    gmm = GaussianMixture(n_components=n_components)
    gmm.fit(samples)
    return gmm


def aggregate_parameters(means, covariances):
    ones = np.ones((means.shape[1], 1))
    agg_mean = (means @ ones).flatten()
    agg_var = (ones.T @ covariances @ ones).flatten()
    return agg_mean[0], agg_var[0]


N, D, scale = 5000, 5, 0.2
samples_5d = generate_cauchy_samples(N, D, scale)

# Approach 1: classical
gmm_5d = fit_gaussian(samples_5d)
mu_5d = gmm_5d.means_[0]
sigma_5d = gmm_5d.covariances_[0]

agg_mu_from_5d, agg_sigma_from_5d = aggregate_parameters(mu_5d.reshape(1, -1), sigma_5d)

# Approach 2: constraint-informed
samples_agg_1d = samples_5d.sum(axis=1).reshape(-1, 1)
gmm_1d = fit_gaussian(samples_agg_1d)
agg_mu_direct = gmm_1d.means_[0, 0]
agg_sigma_direct = gmm_1d.covariances_[0, 0]

print("Multivariate fit parameters:")
print("mu (5D):", mu_5d)
print("Sigma (5D):\n", sigma_5d)

print("\nTransformed Classical parameters from multi-dimensional fit:")
print("1ᵀ mu:", agg_mu_from_5d)
print("1ᵀ Sigma 1:", agg_sigma_from_5d)

print("\nConstraint-Informed fit parameters:")
print("mu:", agg_mu_direct)
print("Sigma:", agg_sigma_direct)