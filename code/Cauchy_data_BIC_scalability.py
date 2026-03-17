import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy.stats import norm
from scipy.special import logsumexp

###################################################
### Use this script to save the BIC values for 
### Synthetic-C datasets for d=10,50
###################################################

def generate_synthetic_cauchy(N, D, scale, seed=None):
    if seed is not None:
        np.random.seed(seed)
    samples = np.random.standard_cauchy(size=(N, D)) * scale
    sums = samples.sum(axis=1)
    return samples, sums


def fit_classical_projected(samples, n_components):
    D = samples.shape[1]
    gmm = GaussianMixture(n_components=n_components)
    gmm.fit(samples)

    weights = gmm.weights_
    means = gmm.means_
    covs = gmm.covariances_

    ones = np.ones((D, 1))

    m_1d = []
    s2_1d = []
    for k in range(n_components):
        m_k = means[k].sum()
        s2_k = float(ones.T @ covs[k] @ ones)
        m_1d.append(m_k)
        s2_1d.append(s2_k)

    return np.array(weights), np.array(m_1d), np.array(s2_1d)

#FixedEM, not used for plotting
class ZeroMean1D:

    def __init__(self, K, max_iter=200, tol=1e-4, seed=None):
        self.K = K
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed

    def fit(self, data):
        N = len(data)
        np.random.seed(self.seed)

        self.weights_ = np.ones(self.K) / self.K
        self.vars_ = np.ones(self.K)

        prev_ll = -np.inf

        for _ in range(self.max_iter):

            # E-step
            log_prob = np.zeros((N, self.K))
            for k in range(self.K):
                log_prob[:, k] = np.log(self.weights_[k]) + \
                                 norm.logpdf(data, loc=0, scale=np.sqrt(self.vars_[k]))

            log_norm = logsumexp(log_prob, axis=1, keepdims=True)
            resp = np.exp(log_prob - log_norm)
            ll = np.sum(log_norm)

            # M-step
            Nk = resp.sum(axis=0)
            self.weights_ = Nk / N

            for k in range(self.K):
                self.vars_[k] = np.sum(resp[:, k] * data**2) / Nk[k]

            if abs(ll - prev_ll) < self.tol:
                break

            prev_ll = ll

        self.loglik_ = ll
        return self



# Likelihood  and BIC calculations
def compute_log_likelihood_1d(data, weights, means, variances):
    ll = 0.0
    for x in data:
        p = sum(
            w * norm.pdf(x, loc=m, scale=np.sqrt(v))
            for w, m, v in zip(weights, means, variances)
        )
        ll += np.log(p)
    return ll


def bic_regular_1d(log_likelihood, K, N):
    n_params = 3 * K - 1
    return -2 * log_likelihood + n_params * np.log(N)


def bic_zero_mean_1d(log_likelihood, K, N):
    n_params = 2 * K - 1
    return -2 * log_likelihood + n_params * np.log(N)


def classical_bic_curve(samples, sums, K_max=10, n_init=10):
    N = len(sums)
    bic_vals = []

    for K in range(1, K_max + 1):
        best_bic = np.inf

        for _ in range(n_init):
            try:
                w, m, v = fit_classical_projected(samples, K)
                ll = compute_log_likelihood_1d(sums, w, m, v)
                bic = bic_regular_1d(ll, K, N)
                best_bic = min(best_bic, bic)
            except:
                continue

        bic_vals.append(best_bic)

    return np.array(bic_vals)


def ci_regular_bic_curve(sums, K_max=10, n_init=10):
    N = len(sums)
    bic_vals = []

    for K in range(1, K_max + 1):
        best_bic = np.inf

        for _ in range(n_init):
            try:
                gmm = GaussianMixture(n_components=K)
                gmm.fit(sums.reshape(-1, 1))
                ll = gmm.score(sums.reshape(-1, 1)) * N
                bic = bic_regular_1d(ll, K, N)
                best_bic = min(best_bic, bic)
            except:
                continue

        bic_vals.append(best_bic)

    return np.array(bic_vals)


def ci_zero_mean_bic_curve(sums, K_max=10, n_init=10):
    N = len(sums)
    bic_vals = []

    for K in range(1, K_max + 1):
        best_bic = np.inf

        for seed in range(n_init):
            try:
                model = ZeroMean1D(K, seed=seed)
                model.fit(sums)
                ll = model.loglik_
                bic = bic_zero_mean_1d(ll, K, N)
                best_bic = min(best_bic, bic)
            except:
                continue

        bic_vals.append(best_bic)

    return np.array(bic_vals)


def main():
    N = 10000
    scale = 0.01
    seed = 125
    K_max = 10

    results = {}

    # d=10
    samples10, sums10 = generate_synthetic_cauchy(N, 10, scale, seed)
    results["classical_10"] = classical_bic_curve(samples10, sums10, K_max)
    results["ci_regular_10"] = ci_regular_bic_curve(sums10, K_max)
    results["ci_zero_mean_10"] = ci_zero_mean_bic_curve(sums10, K_max)

    # d=50
    samples50, sums50 = generate_synthetic_cauchy(N, 50, scale, seed)
    results["classical_50"] = classical_bic_curve(samples50, sums50, K_max)
    results["ci_regular_50"] = ci_regular_bic_curve(sums50, K_max)
    results["ci_zero_mean_50"] = ci_zero_mean_bic_curve(sums50, K_max)

    np.save("../data/bic_results_all_methods.npy", results)

if __name__ == "__main__":
    main()