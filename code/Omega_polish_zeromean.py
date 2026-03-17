import os
import time
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from scipy.special import logsumexp

######################################################
### Use this script to do serial fitting of Omega of
### the constraint-informed-approach for Polish
### with FixedEM and output estimated parameters of 
### 10 different seeds
######################################################

def generate_cauchy_samples(N, D, gamma, seed=51):
    rng = np.random.RandomState(seed)
    return rng.standard_cauchy(size=(N, D)) * gamma

class ZeroMeanGMM1D:
    def __init__(self, n_components=3, max_iter=100, tol=1e-4,
                 reg_covar=1e-8, random_state=None):
        self.K = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.random_state = random_state

    def fit(self, X):
        X = X.reshape(-1, 1)
        N = len(X)
        rng = np.random.RandomState(self.random_state)
        self.weights_ = np.ones(self.K) / self.K
        self.tau2_ = rng.uniform(0.5*np.var(X), 1.5*np.var(X), size=self.K)
        lower_bound = -np.inf
        for _ in range(self.max_iter):
            logpdf = np.zeros((N, self.K))
            for k in range(self.K):
                var = self.tau2_[k]
                logpdf[:, k] = -0.5 * (np.log(2*np.pi*var) + (X.flatten()**2)/var)
            log_resp = logpdf + np.log(self.weights_)
            log_norm = logsumexp(log_resp, axis=1, keepdims=True)
            resp = np.exp(log_resp - log_norm)
            ll = log_norm.sum()
            Nk = resp.sum(axis=0)
            self.weights_ = Nk / N
            for k in range(self.K):
                self.tau2_[k] = ((resp[:, k] * (X.flatten()**2)).sum() / Nk[k])
                self.tau2_[k] = max(self.tau2_[k], self.reg_covar)
            if np.abs(ll - lower_bound) < self.tol * (1 + np.abs(ll)):
                break
            lower_bound = ll
        self.lower_bound_ = lower_bound
        return self

    def bic(self, X):
        N = len(X)
        K = self.K
        p = (K - 1) + K
        return -2*self.lower_bound_ + p*np.log(N)

def main():
    gamma = 0.01
    N = 8000
    D = 50
    K = 3
    n_init = 10

    for current_seed in range(42, 52):
        output_csv = f"../data/gmmPolish_omega_zeromean_{current_seed}.csv"
        xi = generate_cauchy_samples(N=N, D=D, gamma=gamma, seed=current_seed)
        Omega = xi.sum(axis=1)
        rows = []

        best_zm = None
        best_bic_zm = np.inf
        for r in range(n_init):
            zm = ZeroMeanGMM1D(n_components=K, random_state=100 + r)
            zm.fit(Omega)
            bic = zm.bic(Omega)
            if bic < best_bic_zm:
                best_bic_zm = bic
                best_zm = zm

        best_sk = None
        best_bic_sk = np.inf
        for r in range(n_init):
            gm = GaussianMixture(n_components=K, covariance_type="spherical", n_init=1, random_state=200 + r)
            gm.fit(Omega.reshape(-1,1))
            bic = gm.bic(Omega.reshape(-1,1))
            if bic < best_bic_sk:
                best_bic_sk = bic
                best_sk = gm

        for k in range(K):
            rows.append({
                "seed": current_seed,
                "approach": "zero_mean",
                "component": k+1,
                "weight": best_zm.weights_[k],
                "mean": 0.0,
                "variance": best_zm.tau2_[k],
                "bic": best_bic_zm
            })

        for k in range(K):
            rows.append({
                "seed": current_seed,
                "approach": "sklearn",
                "component": k+1,
                "weight": best_sk.weights_[k],
                "mean": best_sk.means_[k,0],
                "variance": best_sk.covariances_[k],
                "bic": best_bic_sk
            })

        df = pd.DataFrame(rows)
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"Processed seed {current_seed}: ZM BIC={best_bic_zm:.2f}, SK BIC={best_bic_sk:.2f}")

if __name__ == "__main__":
    main()