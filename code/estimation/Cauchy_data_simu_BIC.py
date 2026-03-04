import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy.stats import norm, wasserstein_distance


def generate_synthetic_cauchy(N: int, D: int, scale: float, seed: int = None):
    if seed is not None:
        np.random.seed(seed)
    samples = np.random.standard_cauchy(size=(N, D)) * scale
    sums = samples.sum(axis=1)
    return samples, sums

# Classical fitting
def fit_classical_projected(samples: np.ndarray, n_components: int):
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



# Constraint-informed fitting
def fit_constraint_informed_gmm1d(sums: np.ndarray, n_components: int):
    gmm = GaussianMixture(n_components=n_components)
    gmm.fit(sums.reshape(-1, 1))
    return gmm

def compute_log_likelihood_1d(data, weights, means, variances):
    ll = 0.0
    for x in data:
        p = sum(
            w * norm.pdf(x, loc=m, scale=np.sqrt(v))
            for w, m, v in zip(weights, means, variances)
        )
        ll += np.log(p)
    return ll


def bic_1d(log_likelihood, K, N):
    n_params = 3 * K - 1   # weights + means + variances
    return -2 * log_likelihood + n_params * np.log(N)



# BIC for classical and constraint-informed
def classical_bic_curve(samples_10d, sums_1d, K_max=10, n_init=10):
    N = len(sums_1d)
    bic_vals = []

    for K in range(1, K_max + 1):
        best_bic = np.inf

        for _ in range(n_init):
            try:
                w, m, v = fit_classical_projected(samples_10d, K)
                ll = compute_log_likelihood_1d(sums_1d, w, m, v)
                bic = bic_1d(ll, K, N)
                best_bic = min(best_bic, bic)
            except Exception:
                continue

        bic_vals.append(best_bic)
        print(f"[Classical] K={K:2d}, best BIC={best_bic:.2f}")

    return np.array(bic_vals)


def ci_bic_curve(sums_1d, K_max=10, n_init=10):
    N = len(sums_1d)
    bic_vals = []

    for K in range(1, K_max + 1):
        best_bic = np.inf

        for _ in range(n_init):
            try:
                gmm = GaussianMixture(n_components=K, n_init=1)
                gmm.fit(sums_1d.reshape(-1, 1))
                ll = gmm.score(sums_1d.reshape(-1, 1)) * N
                bic = bic_1d(ll, K, N)
                best_bic = min(best_bic, bic)
            except Exception:
                continue

        bic_vals.append(best_bic)
        print(f"[CI]        K={K:2d}, best BIC={best_bic:.2f}")

    return np.array(bic_vals)


# Plot
def plot_bic_vs_k(bic_classical, bic_ci):
    K = np.arange(1, len(bic_classical) + 1)

    # Determine y-range based on CI only
    finite_ci = bic_ci[np.isfinite(bic_ci)]
    ymax = np.max(finite_ci) * 1.15
    ymin = np.min(finite_ci) * 0.95

    # Cap classical values at ymax
    bic_classical_plot = bic_classical.copy()
    bic_classical_plot[~np.isfinite(bic_classical_plot) |
                       (bic_classical_plot > ymax)] = ymax

    plt.figure(figsize=(8, 6))

    
    plt.plot(
        K[:3],
        bic_classical_plot[:3],
        linestyle='--',
        linewidth=3,
        marker='o',
        color='darkorange',
        label='Classical'
    )

    plt.plot(
        K[2:],
        bic_classical_plot[2:],
        linestyle='-',
        linewidth=3,
        marker='o',
        color='darkorange'
    )

    plt.plot(
        K,
        bic_ci,
        linestyle='-',
        linewidth=3,
        marker='s',
        color='forestgreen',
        label='Constraint-Informed'
    )

    plt.ylim(ymin, ymax)
    plt.xticks(K)
    plt.xlabel("Number of GMM Components K", fontsize=14)
    plt.ylabel("BIC*", fontsize=14)
    plt.title("BIC* vs K", fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.show()


def main():
    N, D, scale, seed = 10000, 50, 0.01, 125
    K_max = 10
    n_init = 10

    samples_10d, sums_1d = generate_synthetic_cauchy(N, D, scale, seed)

    print("\n=== Classical GMM BIC ===")
    bic_classical = classical_bic_curve(
        samples_10d, sums_1d, K_max, n_init
    )

    print("\n=== Constraint-Informed GMM BIC ===")
    bic_ci = ci_bic_curve(
        sums_1d, K_max, n_init
    )

    plot_bic_vs_k(bic_classical, bic_ci)


if __name__ == "__main__":
    main()
