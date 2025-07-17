import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy.stats import norm


def generate_synthetic_gaussian(N: int, D: int, mu: float, sigma: float, seed: int = None):
    """
    Generate N×D samples from Gaussian, return D-dimensional samples and their 1D aggregate.
    """
    if seed is not None:
        np.random.seed(seed)
    samples = np.random.normal(mu, sigma, size=(N, D))
    sums = samples.sum(axis=1)
    return samples, sums


def fit_classical(samples: np.ndarray, n_components: int = 1, random_state: int = None):
    """
    Classical way of fitting D-dimensional samples
    """
    D = samples.shape[1]
    gmm = GaussianMixture(n_components=n_components, random_state=random_state)
    gmm.fit(samples)

    mean_vec = gmm.means_[0]
    cov_mat = gmm.covariances_[0]
    ones = np.ones(D)

    mean_1d = ones @ mean_vec
    var_1d = ones @ cov_mat @ ones
    std_1d = np.sqrt(var_1d)

    return mean_1d, std_1d, gmm


def fit_constraint_informed(sums: np.ndarray, ddof: int = 1):
    """
    Constraint-informed way of fitting 1D aggregate
    """
    mean_1d = np.mean(sums)
    std_1d = np.std(sums, ddof=ddof)
    return mean_1d, std_1d


def compute_log_likelihood(data: np.ndarray, loc: float, scale: float):
    return norm.logpdf(data, loc=loc, scale=scale).sum()


def plot_aggregate_comparison(sums: np.ndarray,
                              classical_params: tuple,
                              ci_params: tuple):
    mean_cl, std_cl = classical_params
    mean_ci, std_ci = ci_params

    plt.figure(figsize=(8, 8))
    plt.hist(
        sums,
        bins=70,
        density=True,
        alpha=0.5,
        color='skyblue',
        edgecolor='black',
        label='Original Data'
    )

    x_vals = np.linspace(sums.min(), sums.max(), 200)
    pdf_cl = norm.pdf(x_vals, loc=mean_cl, scale=std_cl)
    pdf_ci = norm.pdf(x_vals, loc=mean_ci, scale=std_ci)

    plt.plot(
        x_vals,
        pdf_ci,
        color='forestgreen',
        linestyle='--',
        linewidth=3,
        label='Constraint-Informed Gaussian',
        zorder=4
    )

    plt.plot(
        x_vals,
        pdf_cl,
        color='darkorange',
        linestyle=':',
        linewidth=3,
        label='Classical Gaussian',
        zorder=5
    )

    plt.xlim(-0.75, 0.2)
    plt.title("Synthetic Gaussian", fontsize=18)
    plt.xlabel("Aggregate Forecast Errors", fontsize=18)
    plt.ylabel("Density", fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(loc='upper left', fontsize='x-large')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    # Parameters
    N, D = 10_000, 10
    mu, sigma = -0.024, 0.036
    seed = 42
    samples_10d, sums_1d = generate_synthetic_gaussian(N, D, mu, sigma, seed)
    mean_cl, std_cl, _gmm = fit_classical(samples_10d, n_components=1, random_state=seed)
    mean_ci, std_ci = fit_constraint_informed(sums_1d)
    ll_classical = compute_log_likelihood(sums_1d, mean_cl, std_cl)
    ll_ci = compute_log_likelihood(sums_1d, mean_ci, std_ci)

    print(f"Total log-likelihood (classical fit):           {ll_classical:.2f}")
    print(f"Total log-likelihood (constraint-informed fit): {ll_ci:.2f}")

    plot_aggregate_comparison(
        sums_1d,
        classical_params=(mean_cl, std_cl),
        ci_params=(mean_ci, std_ci)
    )


if __name__ == "__main__":
    main()
