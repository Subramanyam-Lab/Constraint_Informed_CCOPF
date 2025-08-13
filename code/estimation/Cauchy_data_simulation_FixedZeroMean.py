import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.extmath import row_norms
from sklearn.utils.validation import check_array, check_random_state
from scipy.special import logsumexp
from numpy.linalg import cholesky, slogdet
from scipy.stats import norm, wasserstein_distance


class ZeroMeanGMM:
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
            C0 = 0.5 * (C0 + C0.T)
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
            C0 = 0.5 * (C0 + C0.T)
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
        self.lower_bound_ = ll
        return self

    @property
    def n_components_(self):
        return self.n_components

def generate_synthetic_cauchy(N: int, D: int, scale: float, seed: int = None):
    if seed is not None:
        np.random.seed(seed)
    samples = np.random.standard_cauchy(size=(N, D)) * scale
    sums = samples.sum(axis=1)
    return samples, sums

# Classical fit 
def fit_classical_projected_zero_mean(samples_10d: np.ndarray,
                                      n_components: int = 3,
                                      covariance_type: str = "tied_shape",
                                      random_state: int = 0):
    N, D = samples_10d.shape
    zm = ZeroMeanGMM(n_components=n_components,
                     covariance_type=covariance_type,
                     max_iter=500,
                     tol=1e-4,
                     reg_covar=1e-6,
                     random_state=random_state).fit(samples_10d)
    w = zm.weights_
    means_1d = np.zeros(n_components)

    ones = np.ones(D)
    vars_1d = np.empty(n_components)
    if zm.covariance_type == "spherical":
        vars_1d = zm.tau2_ * D
    else:
        c = ones @ zm.C0_ @ ones
        vars_1d = zm.tau2_ * c

    return w, means_1d, vars_1d, zm


# Constraint-informed fit
def fit_constraint_informed_zero_mean_1d(sums_1d: np.ndarray,
                                         n_components: int = 3,
                                         covariance_type: str = "spherical",
                                         random_state: int = 0):
   
    X = sums_1d.reshape(-1, 1)
    cov = "spherical" if X.shape[1] == 1 else covariance_type
    zm = ZeroMeanGMM(n_components=n_components,
                     covariance_type=cov,
                     max_iter=500,
                     tol=1e-4,
                     reg_covar=1e-6,
                     random_state=random_state).fit(X)
    w = zm.weights_
    means = np.zeros(n_components)
    if zm.covariance_type == "spherical":
        vars_ = zm.tau2_.copy()
    else:
        vars_ = zm.tau2_.copy()
    return w, means, vars_, zm


def pdf_from_params_1d(x, weights, means, variances):
    pdf = np.zeros_like(x, dtype=float)
    for w, m, v in zip(weights, means, variances):
        pdf += w * norm.pdf(x, loc=m, scale=np.sqrt(v))
    return pdf

def plot_cauchy_comparison_zero_mean(sums: np.ndarray,
                                     cl_w: np.ndarray, cl_m: np.ndarray, cl_v: np.ndarray,
                                     ci_w: np.ndarray, ci_m: np.ndarray, ci_v: np.ndarray):
    plt.figure(figsize=(8, 8))
    plt.hist(sums, bins=5000, density=True, alpha=0.5, color='skyblue',
             edgecolor='black', label='Original Data Histogram')

    x_plot = np.linspace(-5, 5, 2000)

    pdf_classical = pdf_from_params_1d(x_plot, cl_w, cl_m, cl_v)
    plt.plot(x_plot, pdf_classical, linestyle=':', lw=3, label='Classical (Zero-Mean, K=3)')

    pdf_ci = pdf_from_params_1d(x_plot, ci_w, ci_m, ci_v)
    plt.plot(x_plot, pdf_ci, linestyle='--', lw=3, label='Constraint-Informed (Zero-Mean, K=3)')

    plt.xlim(-5, 5)
    plt.title("Synthetic Cauchy — Zero-Mean GMMs", fontsize=18)
    plt.xlabel("Aggregate Forecast Errors", fontsize=18)
    plt.ylabel("Density", fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(loc='upper left', fontsize='x-large')
    plt.grid(True)
    plt.show()

# Log-likelihood for perforance evaluation
def sample_from_1d_mixture(n_samples: int,
                           weights: np.ndarray,
                           means: np.ndarray,
                           stds: np.ndarray):
    components = np.random.choice(len(weights), size=n_samples, p=weights)
    return np.array([np.random.normal(means[c], stds[c]) for c in components])

def compute_log_likelihood_1d(data: np.ndarray,
                              weights: np.ndarray,
                              means: np.ndarray,
                              covariances: np.ndarray):
    ll = 0.0
    for x in data:
        p = sum(w * norm.pdf(x, loc=m, scale=np.sqrt(c))
                for w, m, c in zip(weights, means, covariances))
        ll += np.log(max(p, 1e-300))
    return ll


def main():
    N, D, scale, seed = 10000, 10, 0.02, 125
    samples_10d, sums_1d = generate_synthetic_cauchy(N, D, scale, seed)

    #Classical with zero-mean
    cl_w, cl_m, cl_v, cl_model = fit_classical_projected_zero_mean(
        samples_10d, n_components=3, covariance_type="tied_shape", random_state=0
    )

    ci_w, ci_m, ci_v, ci_model = fit_constraint_informed_zero_mean_1d(
        sums_1d, n_components=3, covariance_type="spherical", random_state=0
    )

    plot_cauchy_comparison_zero_mean(sums_1d, cl_w, cl_m, cl_v, ci_w, ci_m, ci_v)

    ftt = sample_from_1d_mixture(1000, cl_w, cl_m, np.sqrt(cl_v))
    ttf = sample_from_1d_mixture(1000, ci_w, ci_m, np.sqrt(ci_v))

    wd_ftt = wasserstein_distance(sums_1d, ftt)
    wd_ttf = wasserstein_distance(sums_1d, ttf)

    print("Wasserstein distance (classical zero-mean vs. data):", wd_ftt)
    print("Wasserstein distance (constraint-informed zero-mean vs. data):", wd_ttf)

    ll_ci = compute_log_likelihood_1d(sums_1d, ci_w, ci_m, ci_v)
    ll_cl = compute_log_likelihood_1d(sums_1d, cl_w, cl_m, cl_v)
    print(f"Constraint-Informed Zero-Mean GMM Log-Likelihood: {ll_ci}")
    print(f"Classical Zero-Mean  Log-Likelihood: {ll_cl}")

if __name__ == "__main__":
    main()
