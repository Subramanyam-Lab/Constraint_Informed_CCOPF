import os
import time
import numpy as np
import pandas as pd

from numpy.linalg import slogdet, cholesky
from sklearn.mixture import GaussianMixture
from sklearn.utils.extmath import row_norms
from sklearn.utils.validation import check_array, check_random_state
from scipy.special import logsumexp  

def generate_cauchy_samples(N=2000, D=10, scale=0.02, seed=125):
    rng = np.random.RandomState(seed)
    return rng.standard_cauchy(size=(N, D)) * scale

def load_ptdf_matrix(csv_path):
    df = pd.read_csv(csv_path, index_col=0)
    return df.values, df.index.astype(str).tolist()

def softmax_rows(A):
    Z = A - logsumexp(A, axis=1, keepdims=True)
    return np.exp(Z)

# FixedEM at zero
class ZeroMeanGMM:
    def __init__(self, n_components=2, covariance_type="tied_shape",
                 max_iter=200, tol=1e-4, reg_covar=1e-6, random_state=None):
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
        R = softmax_rows(scores)
        Nk = R.sum(axis=0) + 1e-12
        if self.covariance_type == "spherical":
            q = row_norms(X, squared=True)
            self.tau2_ = np.array([(R[:, k] @ q) / (Nk[k] * d) for k in range(K)]) + self.reg_covar
        else:
            invC0 = np.linalg.inv(self.C0_)
            Sk = [(X.T * R[:, k]) @ X / Nk[k] for k in range(K)]
            self.tau2_ = np.array([(np.trace(invC0 @ Sk[k]) / d) for k in range(K)]) + self.reg_covar

    def _e_step(self, X):
        if self.covariance_type == "tied_shape":
            logpdf = self._log_gauss_zero_mean(X, self.tau2_, self.C0_, self._L0_, self._logdetC0_)
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

        if self.covariance_type == "spherical":
            q = row_norms(X, squared=True)
            tau2 = np.array([(resp[:, k] @ q) / (Nk[k] * d) for k in range(K)])
            self.tau2_ = np.maximum(tau2, self.reg_covar)
        else:
            Sk = [(X.T * resp[:, k]) @ X / Nk[k] for k in range(K)]
            invC0 = np.linalg.inv(self.C0_)
            tau2 = np.array([np.trace(invC0 @ Sk[k]) / d for k in range(K)])
            self.tau2_ = np.maximum(tau2, self.reg_covar)
            A = sum((Nk[k] / self.tau2_[k]) * Sk[k] for k in range(K))
            trA = np.trace(A)
            C0 = A * (d / trA) if trA > 0 else np.eye(d)
            C0 = 0.5*(C0 + C0.T)
            C0.flat[::d+1] += self.reg_covar
            self.C0_ = C0
            self._L0_ = cholesky(self.C0_)
            self._logdetC0_ = slogdet(self.C0_)[1]

    def fit(self, X):
        X = check_array(X, accept_sparse=False)
        rng = check_random_state(self.random_state)
        self._initialize(X, rng)
        lower_bound = -np.inf
        for it in range(self.max_iter):
            resp, ll = self._e_step(X)
            self._m_step(X, resp)
            if np.abs(ll - lower_bound) < self.tol * (1 + np.abs(ll)):
                break
            lower_bound = ll
        self.n_iter_ = it + 1
        self.lower_bound_ = lower_bound
        return self

    def bic(self, X):
        """BIC = -2*loglik + p*log(N)."""
        N, d = X.shape
        K = self.n_components
        loglik = self.lower_bound_
        if self.covariance_type == "spherical":
            p = (K - 1) + K  
        else:
            p = (K - 1) + K + (d*(d+1))//2 - 1  
        return -2*loglik + p*np.log(N)

# 10 runs for each approach
def fit_zero_mean_both_BICs(X, K=3, n_init=10, base_seed=0, max_iter=500):
    runs = []
    per_cov = {}
    for cov in ("spherical", "tied_shape"):
        best = None
        best_bic = np.inf
        for r in range(n_init):
            zm = ZeroMeanGMM(n_components=K, covariance_type=cov,
                             max_iter=max_iter, random_state=base_seed + 101*r)
            zm.fit(X)
            bic = zm.bic(X)
            runs.append(("zero_mean", cov, r, zm.lower_bound_, bic))
            if bic < best_bic:
                best_bic = bic
                best = dict(cov=cov, seed_id=r, ll=zm.lower_bound_, bic=bic, model=zm)
        per_cov[cov] = best
    winner_key = min(per_cov.keys(), key=lambda c: per_cov[c]["bic"])
    return per_cov, per_cov[winner_key], runs

def fit_sklearn_both_BICs(X, K=3, n_init=10, base_seed=0, max_iter=500):
    runs = []
    per_cov = {}
    for cov in ("tied", "spherical"):
        best = None
        best_bic = np.inf
        for r in range(n_init):
            gm = GaussianMixture(
                n_components=K,
                covariance_type=cov,
                n_init=1,                
                max_iter=max_iter,
                reg_covar=1e-6,
                random_state=base_seed + 313*r,
                init_params="kmeans",
            )
            gm.fit(X)
            ll  = gm.score(X) * len(X)   
            bic = gm.bic(X)
            runs.append(("sklearn", cov, r, ll, bic))
            if bic < best_bic:
                best_bic = bic
                best = dict(cov=cov, seed_id=r, ll=ll, bic=bic, model=gm)
        per_cov[cov] = best
    winner_key = min(per_cov.keys(), key=lambda c: per_cov[c]["bic"])
    return per_cov, per_cov[winner_key], runs


def _flatten_cov(Sigma):
    """Return dict of cov_ij entries for a dxd covariance matrix."""
    d = Sigma.shape[0]
    out = {}
    for i in range(d):
        for j in range(d):
            out[f"cov_{i+1}_{j+1}"] = float(Sigma[i, j])
    return out

def serialize_params(line_id, approach, cov_type, best_dict, d):
    rows = []
    model = best_dict["model"]
    K = model.n_components
    seed_id = best_dict["seed_id"]
    ll = best_dict["ll"]
    bic = best_dict["bic"]

    if approach == "zero_mean":
        # FixedMeans at 0
        means = np.zeros((K, d))
        weights = model.weights_
        if model.covariance_type == "spherical":
            # Σ_k = tau2_k * I spherical
            for k in range(K):
                Sigma = np.eye(d) * model.tau2_[k]
                row = {
                    "line_id": line_id,
                    "approach": approach,
                    "covariance_type": "spherical",
                    "component": k,
                    "seed_id": seed_id,
                    "loglik": ll,
                    "bic": bic,
                    "weight": float(weights[k]),
                    "tau2": float(model.tau2_[k]),
                }
                for j in range(d):
                    row[f"mean_{j+1}"] = float(means[k, j])
                row.update(_flatten_cov(Sigma))
                rows.append(row)
        else: # tied
            C0 = model.C0_
            C0_flat = {f"C0_{i+1}_{j+1}": float(C0[i, j]) for i in range(d) for j in range(d)}
            for k in range(K):
                Sigma = model.tau2_[k] * C0
                row = {
                    "line_id": line_id,
                    "approach": approach,
                    "covariance_type": "tied",
                    "component": k,
                    "seed_id": seed_id,
                    "loglik": ll,
                    "bic": bic,
                    "weight": float(model.weights_[k]),
                    "tau2": float(model.tau2_[k]),
                }
                for j in range(d):
                    row[f"mean_{j+1}"] = 0.0
                row.update(_flatten_cov(Sigma))
                row.update(C0_flat)
                rows.append(row)

    else:  # scikit-learn
        gm = model
        weights = gm.weights_
        means = gm.means_
        if gm.covariance_type == "tied":
            Sigma_shared = gm.covariances_
            for k in range(K):
                row = {
                    "line_id": line_id,
                    "approach": approach,
                    "covariance_type": "tied",
                    "component": k,
                    "seed_id": seed_id,
                    "loglik": ll,
                    "bic": bic,
                    "weight": float(weights[k]),
                }
                for j in range(d):
                    row[f"mean_{j+1}"] = float(means[k, j])
                row.update(_flatten_cov(Sigma_shared))
                rows.append(row)
        elif gm.covariance_type == "spherical":
            vars_per_comp = gm.covariances_  
            for k in range(K):
                Sigma = np.eye(d) * float(vars_per_comp[k])
                row = {
                    "line_id": line_id,
                    "approach": approach,
                    "covariance_type": "spherical",
                    "component": k,
                    "seed_id": seed_id,
                    "loglik": ll,
                    "bic": bic,
                    "weight": float(weights[k]),
                }
                for j in range(d):
                    row[f"mean_{j+1}"] = float(means[k, j])
                row.update(_flatten_cov(Sigma))
                rows.append(row)
        else:
            raise ValueError(f"Unhandled sklearn covariance_type: {gm.covariance_type}")

    return rows


def main():
    ptdf_csv     = "../data/ptdf_matrix.csv"
    wind_buses_1 = [10, 25, 26, 59, 65, 66, 69, 80, 100, 111]  
    n_components = 3
    n_init       = 10
    base_seed    = 0
    max_iter     = 500
    out_summary_csv = "../data/gmm_ll_bic_compare_by_line.csv"
    out_params_csv  = "../data/gmm_params_by_line.csv"

    D     = len(wind_buses_1)
    xi    = generate_cauchy_samples(N=2000, D=D, scale=0.02, seed=125)  
    Omega = xi.sum(axis=1)                                             

    H, line_ids = load_ptdf_matrix(ptdf_csv)   
    W_idx   = [b - 1 for b in wind_buses_1]    
    H_wind  = H[:, W_idx]                      

    summary_rows = []
    params_rows  = []

    print(f"Fitting {len(line_ids)} lines × (zero-mean & sklearn), each with {n_init} restarts…")

    for i, line_id in enumerate(line_ids):
        t0 = time.perf_counter()

        Lambda_l = xi.dot(H_wind[i, :])        
        X = np.column_stack((Omega, Lambda_l))  
        d = X.shape[1]

        # Fixedzero mean
        zm_per_cov, zm_best, _ = fit_zero_mean_both_BICs(
            X, K=n_components, n_init=n_init, base_seed=base_seed, max_iter=max_iter
        )
        zm_cov_choice = "tied" if zm_best["cov"] == "tied_shape" else "spherical"

        # Traditional EM
        sk_per_cov, sk_best, _ = fit_sklearn_both_BICs(
            X, K=n_components, n_init=n_init, base_seed=base_seed, max_iter=max_iter
        )

        t1 = time.perf_counter()

        summary_rows.append({
            "line_id": line_id,
            "approach": "zero_mean",
            "covariance_type": zm_cov_choice,    
            "best_seed_id": zm_best["seed_id"],
            "loglik": zm_best["ll"],
            "bic": zm_best["bic"],
            "bic_spherical": zm_per_cov["spherical"]["bic"],
            "bic_tied": zm_per_cov["tied_shape"]["bic"],
            "n_components": n_components,
            "fit_time_seconds": t1 - t0
        })
        summary_rows.append({
            "line_id": line_id,
            "approach": "sklearn",
            "covariance_type": sk_best["cov"],   
            "best_seed_id": sk_best["seed_id"],
            "loglik": sk_best["ll"],
            "bic": sk_best["bic"],
            "bic_spherical": sk_per_cov["spherical"]["bic"],
            "bic_tied": sk_per_cov["tied"]["bic"],
            "n_components": n_components,
            "fit_time_seconds": t1 - t0
        })

        params_rows.extend(
            serialize_params(
                line_id=line_id,
                approach="zero_mean",
                cov_type=zm_cov_choice,
                best_dict=zm_best,
                d=d,
            )
        )
        params_rows.extend(
            serialize_params(
                line_id=line_id,
                approach="sklearn",
                cov_type=sk_best["cov"],
                best_dict=sk_best,
                d=d,
            )
        )

    df_summary = pd.DataFrame(summary_rows)
    os.makedirs(os.path.dirname(out_summary_csv), exist_ok=True)
    df_summary.to_csv(out_summary_csv, index=False)

    df_params = pd.DataFrame(params_rows)
    os.makedirs(os.path.dirname(out_params_csv), exist_ok=True)
    df_params.to_csv(out_params_csv, index=False)

   
if __name__ == "__main__":
    main()
