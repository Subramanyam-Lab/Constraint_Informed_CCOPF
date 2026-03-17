import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from numpy.linalg import slogdet, cholesky, inv
from sklearn.utils.extmath import row_norms
from sklearn.utils.validation import check_array, check_random_state
from scipy.special import logsumexp

########################################################
### Perform classical fitting on 10-dimensional forecast errors
### save the parameters for optimization
########################################################
class ZeroMeanGMM10D:
    def __init__(self, n_components=5, max_iter=200, tol=1e-4, reg_covar=1e-6, random_state=None):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.random_state = random_state

    def _log_gauss_zero_mean(self, X, tau2, L0, logdetC0):
        N, d = X.shape
        K = self.n_components
        out = np.empty((N, K))
        invL0 = inv(L0)
        Y = (invL0 @ X.T).T
        q = row_norms(Y, squared=True)
        for k in range(K):
            out[:, k] = -0.5 * (d * np.log(2 * np.pi * tau2[k]) + logdetC0 + q / tau2[k])
        return out

    def fit(self, X):
        X = check_array(X)
        N, d = X.shape
        K = self.n_components
        rng = check_random_state(self.random_state)

        S = (X.T @ X) / N
        tr = np.trace(S)
        C0 = S * (d / tr) if tr > 0 else np.eye(d)
        C0 = 0.5 * (C0 + C0.T) + np.eye(d) * self.reg_covar
        
        self.C0_ = C0 / (np.exp(slogdet(C0)[1])**(1/d))
        self._L0_ = cholesky(self.C0_)
        self._logdetC0_ = slogdet(self.C0_)[1]

        self.weights_ = np.full(K, 1.0 / K)
        self.tau2_ = rng.uniform(0.1, 1.0, size=K)

        lower_bound = -np.inf
        invL0 = inv(self._L0_)
        q = row_norms((invL0 @ X.T).T, squared=True)

        for _ in range(self.max_iter):
            log_pdf = self._log_gauss_zero_mean(X, self.tau2_, self._L0_, self._logdetC0_)
            log_resp = log_pdf + np.log(self.weights_)
            log_norm = logsumexp(log_resp, axis=1, keepdims=True)
            resp = np.exp(log_resp - log_norm)
            ll = log_norm.sum()

            Nk = resp.sum(axis=0) + 1e-12
            self.weights_ = Nk / N
            self.tau2_ = np.array([(resp[:, k] @ q) / (Nk[k] * d) for k in range(K)])

            if np.abs(ll - lower_bound) < self.tol * (1 + np.abs(ll)): break
            lower_bound = ll
        return self

def main():
    input_data = "../data/15daysNordPoolFinalErrors.csv"
    df_raw = pd.read_csv(input_data)
    X_all = df_raw.iloc[:, -10:].replace([np.inf, -np.inf], np.nan).fillna(0).values

    for seed in range(42, 52):
        train_X, _ = train_test_split(X_all, test_size=0.2, random_state=seed)

        model = ZeroMeanGMM10D(n_components=6, random_state=seed).fit(train_X)

        c0_flattened = model.C0_.flatten()
        c0_cols = {f"C_{i}_{j}": c0_flattened[i*10 + j] for i in range(10) for j in range(10)}
        
        gmm_rows = []
        for k in range(model.n_components):
            row = {"weight": model.weights_[k], "tau2": model.tau2_[k]}
            row.update(c0_cols)
            gmm_rows.append(row)

        pd.DataFrame(gmm_rows).to_csv(f"../data/gmm118Nord_Classical_gmm_{seed}.csv", index=False)

if __name__ == "__main__":
    main()