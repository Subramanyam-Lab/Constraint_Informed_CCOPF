import pandas as pd
import numpy as np
from scipy.stats import norm, wasserstein_distance
import matplotlib.pyplot as plt

from sklearn.utils.extmath import row_norms
from sklearn.utils.validation import check_array, check_random_state
from scipy.special import logsumexp
from numpy.linalg import cholesky, slogdet

# Fixed Zero-mean GMM 
class ZeroMeanGMM:
    """
    K-component GMM Estimation with means fixed at 0 that doesn't get updated in each EM stap
    """
    def __init__(self, n_components=3, covariance_type="tied_shape",
                 max_iter=500, tol=1e-4, reg_covar=1e-6, random_state=None):
        assert covariance_type in ("spherical", "tied_shape")
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.random_state = random_state

    @staticmethod
    def _softmax_rows(A):
        Z = A - logsumexp(A, axis=1, keepdims=True)
        return np.exp(Z)

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
        R = self._softmax_rows(scores)
        Nk = R.sum(axis=0) + 1e-12
        if self.covariance_type == "spherical":
            q = row_norms(X, squared=True)
            self.tau2_ = np.array([(R[:, k] @ q) / (Nk[k] * d) for k in range(K)]) + self.reg_covar
        else:
            invC0 = np.linalg.inv(self.C0_)
            Sk = [(X.T * R[:, k]) @ X / Nk[k] for k in range(K)]
            self.tau2_ = np.array([np.trace(invC0 @ Sk[k]) / d for k in range(K)]) + self.reg_covar

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
        prev = -np.inf
        for _ in range(self.max_iter):
            resp, ll = self._e_step(X)
            self._m_step(X, resp)
            if np.abs(ll - prev) < self.tol * (1 + np.abs(ll)):
                break
            prev = ll
        self.lower_bound_ = ll
        return self


# Load datafiles
def load_and_prepare_error_data(file_path: str, regions: list) -> np.ndarray:
    df = pd.read_csv(file_path)
    df = df.replace({',': '', ' ': ''}, regex=True)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    for r in regions:
        df[f'{r}-Error'] = (
            (df[f'{r}-Actual'] - df[f'{r}-Forecast']) / df[f'{r}-Actual']
        )

    err_cols = [f'{r}-Error' for r in regions]
    df[err_cols] = df[err_cols].replace({np.inf: np.nan, -np.inf: -10})
    error_data = df[err_cols].dropna().values

    return error_data


# Compare with tradition EM from scikit leanr
def fit_classical_projected_zero_mean(error_data: np.ndarray,
                                      n_components: int = 3,
                                      covariance_type: str = "tied_shape",
                                      random_state: int = 100):
    N, D = error_data.shape
    zm = ZeroMeanGMM(n_components=n_components,
                     covariance_type=covariance_type,
                     max_iter=500, tol=1e-4, reg_covar=1e-6,
                     random_state=random_state).fit(error_data)
    weights = zm.weights_
    means_1d = np.zeros(n_components)

    ones = np.ones(D)
    if zm.covariance_type == "spherical":
        vars_1d = zm.tau2_ * D
    else:
        c = ones @ zm.C0_ @ ones  
        vars_1d = zm.tau2_ * c
    stds_1d = np.sqrt(vars_1d)

    return zm, weights, means_1d, stds_1d


def fit_constraint_informed_zero_mean_1d(aggregate_errors: np.ndarray,
                                         n_components: int = 3,
                                         random_state: int = 100):
    X = aggregate_errors.reshape(-1, 1)
    zm = ZeroMeanGMM(n_components=n_components,
                     covariance_type="spherical",  # 1D tied_shape ≡ spherical
                     max_iter=500, tol=1e-4, reg_covar=1e-6,
                     random_state=random_state).fit(X)
    weights = zm.weights_
    means = np.zeros(n_components)
    variances = zm.tau2_.copy()
    return weights, means, variances, zm


# Compute log-likelihood for performance evaluation
def sample_classical_projected(weights: np.ndarray,
                               mu_proj: np.ndarray,
                               sigma_proj: np.ndarray,
                               n: int) -> np.ndarray:
    comps = np.random.choice(len(weights), size=n, p=weights)
    return np.array([np.random.normal(mu_proj[c], sigma_proj[c]) for c in comps])


def sample_constraint(weights: np.ndarray, means: np.ndarray,
                      variances: np.ndarray, n: int) -> np.ndarray:
    comps = np.random.choice(len(weights), size=n, p=weights)
    return np.array([np.random.normal(means[c], np.sqrt(variances[c])) for c in comps])


def compute_log_likelihood_1d(data: np.ndarray,
                              weights: np.ndarray,
                              means: np.ndarray,
                              covs: np.ndarray) -> float:
    ll = 0.0
    for x in data:
        p = 0.0
        for w, m, c in zip(weights, means, covs):
            p += w * norm.pdf(x, loc=m, scale=np.sqrt(c))
        ll += np.log(max(p, 1e-300))  # avoid log(0)
    return ll


def plot_nordpool_zero_mean(aggregate_errors: np.ndarray,
                            weights_ci: np.ndarray, means_ci: np.ndarray, vars_ci: np.ndarray,
                            weights_cl: np.ndarray, mu_proj: np.ndarray, sigma_proj: np.ndarray):
    x = np.linspace(aggregate_errors.min(), aggregate_errors.max(), 2000)

    pdf_ci = np.zeros_like(x)
    for w, m, v in zip(weights_ci, means_ci, vars_ci):
        pdf_ci += w * norm.pdf(x, loc=m, scale=np.sqrt(v))

    pdf_cl = np.zeros_like(x)
    for w, m, s in zip(weights_cl, mu_proj, sigma_proj):
        pdf_cl += w * norm.pdf(x, loc=m, scale=s)

    plt.figure(figsize=(8, 8))
    hist = plt.hist(
        aggregate_errors.flatten(),
        bins=2000, density=True, color='skyblue',
        edgecolor='black', alpha=0.5, label='Original Data Histogram'
    )
    line_ci, = plt.plot(
        x, pdf_ci, 'forestgreen', linestyle='--', linewidth=3,
        label='Constraint-Informed Zero-Mean GMM (K=3)'
    )
    line_cl, = plt.plot(
        x, pdf_cl, 'darkorange', linestyle=':', linewidth=3,
        label='Classical Zero-Mean GMM (K=3)'
    )

    handles = [hist[2][0], line_cl, line_ci]
    labels = [
        'Original Data Histogram',
        'Classical Zero-Mean (K=3)',
        'Constraint-Informed Zero-Mean (K=3)'
    ]
    plt.legend(handles=handles, labels=labels, fontsize='x-large')
    plt.title("NordPool — Zero-Mean GMMs", fontsize=18)
    plt.xlabel("Aggregate Forecast Errors", fontsize=18)
    plt.ylabel("Density", fontsize=18)
    plt.xlim(aggregate_errors.min(), aggregate_errors.max())
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    file_path = '../data/15daysNordPoolFinal.csv'
    regions = ['50Hz', 'AMP', 'AT', 'PL', 'TBW', 'TTG', 'TTGf', '50Hzf', 'FR', 'FRf']

    error_data = load_and_prepare_error_data(file_path, regions)
    aggregate_errors = error_data.sum(axis=1)

    # Constraint-informed FixedZeroMean 
    weights_ci, means_ci, vars_ci, zm_ci = fit_constraint_informed_zero_mean_1d(
        aggregate_errors, n_components=3, random_state=100
    )

    # Classical FixedZeroMean 
    zm_cl, weights_cl, mu_proj, sigma_proj = fit_classical_projected_zero_mean(
        error_data, n_components=3, covariance_type="tied_shape", random_state=100
    )
    samples_ci = sample_constraint(weights_ci, means_ci, vars_ci, len(aggregate_errors))
    samples_cl = sample_classical_projected(weights_cl, mu_proj, sigma_proj, len(aggregate_errors))

    wd_ci = wasserstein_distance(aggregate_errors.flatten(), samples_ci)
    wd_cl = wasserstein_distance(aggregate_errors.flatten(), samples_cl)
    print(f"Wasserstein Distance (constraint_informed zero-mean): {wd_ci}")
    print(f"Wasserstein Distance (projected_classical zero-mean): {wd_cl}")

    ll_ci = compute_log_likelihood_1d(aggregate_errors, weights_ci, means_ci, vars_ci)
    ll_cl = compute_log_likelihood_1d(aggregate_errors, weights_cl, mu_proj, sigma_proj**2)
    print("=== Log-Likelihood on Aggregate Data (Zero-Mean Models) ===")
    print(f"Constraint-Informed Zero-Mean GMM Log-Likelihood: {ll_ci}")
    print(f"Classical Zero-Mean (1^T Transformed) Log-Likelihood: {ll_cl}")

    plot_nordpool_zero_mean(aggregate_errors, weights_ci, means_ci, vars_ci,
                            weights_cl, mu_proj, sigma_proj)


if __name__ == "__main__":
    main()
