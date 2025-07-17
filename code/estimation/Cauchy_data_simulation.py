import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy.stats import norm, wasserstein_distance


def generate_synthetic_cauchy(N: int, D: int, scale: float, seed: int = None):
    """
    Generate N×D Cauchy samples with given scale, return D dimensional samples and 1D aggregate
    """
    if seed is not None:
        np.random.seed(seed)
    samples = np.random.standard_cauchy(size=(N, D)) * scale
    sums = samples.sum(axis=1)
    return samples, sums


def fit_classical_projected(samples: np.ndarray, n_components: int = 3):
    """
    Classical fitting and then project into 1D
    """
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


def fit_constraint_informed_gmm1d(sums: np.ndarray, n_components: int = 3):
    """
    Constraint-informed fitting on 1D aggregate
    """
    gmm = GaussianMixture(n_components=n_components)
    gmm.fit(sums.reshape(-1, 1))
    return (
        gmm.weights_,
        gmm.means_.ravel(),
        gmm.covariances_.ravel(),
        gmm
    )


def plot_cauchy_comparison(sums: np.ndarray,
                           weights_classical: np.ndarray,
                           means_classical: np.ndarray,
                           vars_classical: np.ndarray,
                           gmm_ci: GaussianMixture):
    plt.figure(figsize=(8, 8))
    plt.hist(
        sums,
        bins=5000,
        density=True,
        alpha=0.5,
        color='skyblue',
        edgecolor='black',
        label='Original Data Histogram'
    )

    x_plot = np.linspace(-5, 5, 2000)

    pdf_classical = np.zeros_like(x_plot)
    for w, m, v in zip(weights_classical, means_classical, vars_classical):
        pdf_classical += w * norm.pdf(x_plot, loc=m, scale=np.sqrt(v))
    plt.plot(
        x_plot,
        pdf_classical,
        'darkorange',
        linestyle=':',
        lw=3,
        label='Classical GMM (K=3)'
    )

    logprob_ci = gmm_ci.score_samples(x_plot.reshape(-1, 1))
    pdf_ci = np.exp(logprob_ci)
    plt.plot(
        x_plot,
        pdf_ci,
        'forestgreen',
        linestyle='--',
        lw=3,
        label='Constraint-Informed GMM (K=3)'
    )

    plt.xlim(-5, 5)
    plt.title("Synthetic Cauchy", fontsize=18)
    plt.xlabel("Aggregate Forecast Errors", fontsize=18)
    plt.ylabel("Density", fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(loc='upper left', fontsize='x-large')
    plt.grid(True)
    plt.show()


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
        ll += np.log(p)
    return ll


def main():
    N, D, scale, seed = 10000, 10, 0.02, 125

    samples_10d, sums_1d = generate_synthetic_cauchy(N, D, scale, seed)

    weights_cl, means_cl, vars_cl = fit_classical_projected(samples_10d, n_components=3)
    weights_ci, means_ci, covs_ci, gmm_ci = fit_constraint_informed_gmm1d(sums_1d, n_components=3)
    plot_cauchy_comparison(sums_1d, weights_cl, means_cl, vars_cl, gmm_ci)
    ftt = sample_from_1d_mixture(1000, weights_cl, means_cl, np.sqrt(vars_cl))
    ttf, _ = gmm_ci.sample(1000)
    ttf = ttf.ravel()

    wd_ftt = wasserstein_distance(sums_1d, ftt)
    wd_ttf = wasserstein_distance(sums_1d, ttf)

    print("Wasserstein distance (classical GMM vs. data):", wd_ftt)
    print("Wasserstein distance (constraint_informed GMM   vs. data):", wd_ttf)

    print("=== Log‑Likelihood on Aggregate Data ===")
    ll_ci = compute_log_likelihood_1d(sums_1d, weights_ci, means_ci, covs_ci)
    ll_cl = compute_log_likelihood_1d(sums_1d, weights_cl, means_cl, vars_cl)
    print(f"Constraint‑Informed GMM Log‑Likelihood: {ll_ci}")
    print(f"Classical GMM (1^T Transformed) Log‑Likelihood: {ll_cl}")


if __name__ == "__main__":
    main()
