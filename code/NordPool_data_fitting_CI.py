import os
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from numpy.linalg import slogdet, cholesky, inv
from sklearn.utils.extmath import row_norms
from sklearn.utils.validation import check_array, check_random_state
from scipy.special import logsumexp

########################################################
### Partitioning the NordPool dataset into training
### and testing. Perform constraint-informed fitting
### on aggregate errors and 2D eta_ls with zeromeans,
### save the parameters for optimization
########################################################


class ZeroMeanGMM:
    def __init__(self, n_components=3, covariance_type="tied_shape",
                 max_iter=100, tol=1e-4, reg_covar=1e-6, random_state=None):
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
                out[:, k] = -0.5 * (d*np.log(2*np.pi*tau2[k]) + q / tau2[k])
        else:
            invL0 = inv(L0)
            Y = (invL0 @ X.T).T
            q = row_norms(Y, squared=True)
            for k in range(K):
                out[:, k] = -0.5 * (d*np.log(2*np.pi*tau2[k]) + logdetC0 + q / tau2[k])
        return out

    def _initialize(self, X, rng):
        N, d = X.shape
        K = self.n_components
        self.weights_ = np.full(K, 1.0 / K)
        if self.covariance_type == "tied_shape":
            S = (X.T @ X) / N
            tr = np.trace(S)
            C0 = S if tr <= 0 else S * (d / tr)
            C0 = 0.5*(C0 + C0.T) + np.eye(d) * self.reg_covar
            self.C0_ = C0 / (np.exp(slogdet(C0)[1])**(1/d)) if d > 1 else C0
            self._L0_ = cholesky(self.C0_)
            self._logdetC0_ = slogdet(self.C0_)[1]
        else:
            self.C0_, self._L0_, self._logdetC0_ = None, None, 0.0
        
        self.tau2_ = rng.uniform(0.1, 1.0, size=K)

    def fit(self, X):
        X = check_array(X)
        N, d = X.shape
        rng = check_random_state(self.random_state)
        self._initialize(X, rng)
        lower_bound = -np.inf
        for _ in range(self.max_iter):
            logpdf = self._log_gauss_zero_mean(X, self.tau2_, self.C0_, self._L0_, self._logdetC0_)
            log_resp = logpdf + np.log(self.weights_)
            log_norm = logsumexp(log_resp, axis=1, keepdims=True)
            resp = np.exp(log_resp - log_norm)
            ll = log_norm.sum()

            Nk = resp.sum(axis=0) + 1e-12
            self.weights_ = Nk / N
            q = row_norms(X, squared=True) if self.covariance_type == "spherical" else row_norms((inv(self._L0_) @ X.T).T, squared=True) if d > 1 else row_norms(X/np.sqrt(self.C0_), squared=True)
            self.tau2_ = np.array([(resp[:, k] @ q) / (Nk[k] * d) for k in range(self.n_components)])
            if np.abs(ll - lower_bound) < self.tol * (1 + np.abs(ll)): break
            lower_bound = ll
        self.lower_bound_ = lower_bound
        return self

    def bic(self, X):
        N, d = X.shape
        p = (self.n_components - 1) + self.n_components + (d*(d+1)//2 - 1 if self.covariance_type == "tied_shape" else 0)
        return -2 * self.lower_bound_ + p * np.log(N)

def fit_best_gmm(X, K, n_init=10, seed_base=42):
    best_m, best_bic = None, np.inf
    for cov_type in ["spherical", "tied_shape"]:
        for r in range(n_init):
            m = ZeroMeanGMM(n_components=K, covariance_type=cov_type, random_state=seed_base + r).fit(X)
            if m.bic(X) < best_bic:
                best_bic, best_m, best_cov = m.bic(X), m, cov_type
    return best_m, best_cov, best_bic

def main():
    input_data = "../data/15daysNordPoolFinalErrors.csv"
    ptdf_csv = "../data/H_wind_118_matrix.csv"
    
    df_raw = pd.read_csv(input_data)
    X_all = df_raw.iloc[:, -10:].replace([np.inf, -np.inf], np.nan).fillna(0).values
    
    H_wind = pd.read_csv(ptdf_csv, header=None).iloc[:, 1:].values
    ones_10 = np.ones(10)

    for seed in range(42, 52):
        print(f"--- Processing Seed {seed} ---")
        train_X, test_X = train_test_split(X_all, test_size=0.2, random_state=seed)
        pd.DataFrame(test_X).to_csv(f"../data/Nordtest_data_seed_{seed}.csv", index=False)

        Omega_train = train_X.sum(axis=1).reshape(-1, 1)
        best_om, cov_om, _ = fit_best_gmm(Omega_train, K=6, seed_base=seed)
        
        om_rows = [{"approach": "zero_mean", "component": k+1, "weight": best_om.weights_[k],
                    "variance": best_om.tau2_[k] * (best_om.C0_[0,0] if cov_om=="tied_shape" else 1.0)} for k in range(6)]
        pd.DataFrame(om_rows).to_csv(f"../data/gmm118Nord_omega_seed{seed}.csv", index=False)

        eta_rows = []
        for l_idx in range(H_wind.shape[0]):
            H_l = H_wind[l_idx, :]
            Lambda_l = train_X @ H_l
            X_2d = np.column_stack((Omega_train.flatten(), Lambda_l))
            
            best_eta, cov_eta, bic_val = fit_best_gmm(X_2d, K=3, seed_base=seed)
            
            for k in range(3):
                Sigma = best_eta.tau2_[k] * (best_eta.C0_ if cov_eta=="tied_shape" else np.eye(2))
                eta_rows.append({
                    "line_id": l_idx + 1, "component": k+1, "weight": best_eta.weights_[k],
                    "cov_11": Sigma[0,0], "cov_12": Sigma[0,1], "cov_22": Sigma[1,1],
                    "mean_Omega": 0.0, "mean_Lambda": 0.0, "bic": bic_val
                })
        pd.DataFrame(eta_rows).to_csv(f"../data/gmm118Nord_eta_l_seed{seed}.csv", index=False)

if __name__ == "__main__":
    main()