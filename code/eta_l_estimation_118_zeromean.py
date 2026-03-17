import os
import time
import numpy as np
import pandas as pd

from numpy.linalg import slogdet, cholesky
from sklearn.utils.extmath import row_norms
from sklearn.utils.validation import check_array, check_random_state
from scipy.special import logsumexp

######################################################
### Use this script to do serial fitting of eta_l of
### the constraint-informed-approach for 118-bus
### with FixedEM and output estimated parameters of 
### 10 different seeds
######################################################

def generate_cauchy_samples(N, D, gamma, seed=51):
    rng = np.random.RandomState(seed)
    return rng.standard_cauchy(size=(N, D)) * gamma


def load_wind_ptdf(csv_path):
    df = pd.read_csv(csv_path)
    line_ids = df["line_id"].astype(str).tolist()
    H = df.drop(columns=["line_id"]).values
    return H, line_ids


class ZeroMeanGMM:
    def __init__(self, n_components=2, covariance_type="tied_shape",
                 max_iter=100, tol=1e-4, reg_covar=1e-6, random_state=None):

        assert covariance_type in ("spherical", "tied_shape")

        self.n_components = n_components
        self.covariance_type = covariance_type
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.random_state = random_state

    def _log_gauss_zero_mean(self, X, tau2, C0=None, L0=None, logdetC0=0.0):
        N, d = X.shape
        K = tau2.shape[0]
        out = np.empty((N, K))

        if self.covariance_type == "spherical":
            q = row_norms(X, squared=True)
            for k in range(K):
                s2 = tau2[k]
                out[:, k] = -0.5 * (d*np.log(2*np.pi*s2) + q / s2)
        else:
            invL0 = np.linalg.inv(L0)
            Y = (invL0 @ X.T).T
            q = row_norms(Y, squared=True)
            for k in range(K):
                s2 = tau2[k]
                out[:, k] = -0.5 * (d*np.log(2*np.pi*s2) + logdetC0 + q / s2)

        return out

    def _initialize(self, X, rng):
        N, d = X.shape
        K = self.n_components

        self.weights_ = np.full(K, 1.0 / K)

        if self.covariance_type == "tied_shape":
            S = (X.T @ X) / N
            tr = np.trace(S)
            C0 = S if tr <= 0 else S * (d / tr)
            C0 = 0.5*(C0 + C0.T)
            C0.flat[::d+1] += self.reg_covar
            self.C0_ = C0
            self._L0_ = cholesky(self.C0_)
            self._logdetC0_ = slogdet(self.C0_)[1]
        else:
            self.C0_ = None
            self._L0_ = None
            self._logdetC0_ = 0.0

        scores = rng.normal(size=(N, K))
        R = np.exp(scores - logsumexp(scores, axis=1, keepdims=True))
        Nk = R.sum(axis=0) + 1e-12

        q = row_norms(X, squared=True)

        if self.covariance_type == "spherical":
            self.tau2_ = np.array([(R[:, k] @ q) / (Nk[k] * d)
                                   for k in range(K)]) + self.reg_covar
        else:
            invC0 = np.linalg.inv(self.C0_)
            Sk = [(X.T * R[:, k]) @ X / Nk[k] for k in range(K)]
            self.tau2_ = np.array([
                np.trace(invC0 @ Sk[k]) / d
                for k in range(K)
            ]) + self.reg_covar

    def _e_step(self, X):
        if self.covariance_type == "tied_shape":
            logpdf = self._log_gauss_zero_mean(
                X, self.tau2_, self.C0_, self._L0_, self._logdetC0_)
        else:
            logpdf = self._log_gauss_zero_mean(X, self.tau2_)

        log_resp = logpdf + np.log(self.weights_)
        log_norm = logsumexp(log_resp, axis=1, keepdims=True)
        resp = np.exp(log_resp - log_norm)
        ll = log_norm.sum()

        return resp, ll

    def _m_step(self, X, resp):
        N, d = X.shape
        K = self.n_components
        Nk = resp.sum(axis=0) + 1e-12

        self.weights_ = Nk / N
        q = row_norms(X, squared=True)

        if self.covariance_type == "spherical":
            tau2 = np.array([(resp[:, k] @ q) / (Nk[k] * d)
                             for k in range(K)])
            self.tau2_ = np.maximum(tau2, self.reg_covar)
        else:
            Sk = [(X.T * resp[:, k]) @ X / Nk[k] for k in range(K)]
            invC0 = np.linalg.inv(self.C0_)
            tau2 = np.array([
                np.trace(invC0 @ Sk[k]) / d
                for k in range(K)
            ])
            self.tau2_ = np.maximum(tau2, self.reg_covar)

    def fit(self, X):
        X = check_array(X)
        rng = check_random_state(self.random_state)
        self._initialize(X, rng)

        lower_bound = -np.inf

        for _ in range(self.max_iter):
            resp, ll = self._e_step(X)
            self._m_step(X, resp)

            if np.abs(ll - lower_bound) < self.tol * (1 + np.abs(ll)):
                break
            lower_bound = ll

        self.lower_bound_ = lower_bound
        return self

    def bic(self, X):
        N, d = X.shape
        K = self.n_components
        loglik = self.lower_bound_

        if self.covariance_type == "spherical":
            p = (K - 1) + K
        else:
            p = (K - 1) + K + (d*(d+1))//2 - 1

        return -2*loglik + p*np.log(N)

def fit_zero_mean_best(X, n_components=2, n_init=10):
    best_model = None
    best_bic = np.inf
    best_info = {}

    for cov_type in ("spherical", "tied_shape"):
        for r in range(n_init):
            seed = 1000 + r
            zm = ZeroMeanGMM(
                n_components=n_components,
                covariance_type=cov_type,
                random_state=seed
            )
            zm.fit(X)
            bic = zm.bic(X)

            if bic < best_bic:
                best_bic = bic
                best_model = zm
                best_info = {
                    "covariance_type": cov_type,
                    "seed": seed,
                    "bic": bic
                }

    return best_model, best_info


def main():
    ptdf_csv = "../data/H_wind_118_matrix.csv"
    base_output_path = "../data/"
    
    gamma = 0.01
    N = 8000
    seeds_to_run = range(42, 52) 

    if not os.path.exists(ptdf_csv):
        print(f"Error: File {ptdf_csv} not found.")
        return

    H_wind, line_ids = load_wind_ptdf(ptdf_csv)
    L, D = H_wind.shape
    print(f"Data Loaded: {L} lines, {D} wind units.")

    for current_seed in seeds_to_run:
        print(f"\n>>> Starting Seed: {current_seed}")
        output_csv = os.path.join(base_output_path, f"gmm118_zero_mean_eta_l_seed{current_seed}.csv")
        
        xi = generate_cauchy_samples(N=N, D=D, gamma=gamma, seed=current_seed)
        Omega = xi.sum(axis=1) 
        rows = []
        total_fit_time = 0.0
        seed_start = time.perf_counter()

        for i, line_id in enumerate(line_ids):
            line_start = time.perf_counter()

            Lambda_l = xi.dot(H_wind[i, :])
            X = np.column_stack((Omega, Lambda_l))

            gmm, info = fit_zero_mean_best(X, n_components=2, n_init=10)

            elapsed = time.perf_counter() - line_start
            total_fit_time += elapsed

            cov_type = info["covariance_type"]

            for k in range(gmm.n_components):
                if cov_type == "spherical":
                    Sigma = np.eye(2) * gmm.tau2_[k]
                else:
                    Sigma = gmm.tau2_[k] * gmm.C0_

                rows.append({
                    "gamma": gamma,
                    "generation_seed": current_seed,
                    "line_id": line_id,
                    "component": k + 1,
                    "weight": gmm.weights_[k],
                    "mean_Omega": 0.0,
                    "mean_Lambda": 0.0,
                    "cov_11": Sigma[0,0],
                    "cov_12": Sigma[0,1],
                    "cov_22": Sigma[1,1],
                    "covariance_type": cov_type,
                    "init_seed": info["seed"],
                    "bic": info["bic"],
                    "fit_time_seconds": elapsed
                })

        seed_elapsed = time.perf_counter() - seed_start
        df = pd.DataFrame(rows)
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df.to_csv(output_csv, index=False)

        print(f"Finished Seed {current_seed}:")
        print(f" - Saved to: {output_csv}")
        print(f" - Avg time/line: {total_fit_time / L:.4f} s")
        print(f" - Total seed time: {seed_elapsed:.2f} s")

    print("\nProcessing for all seeds complete.")

if __name__ == "__main__":
    main()