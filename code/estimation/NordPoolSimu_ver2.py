import pandas as pd
import numpy as np
from scipy.stats import norm, wasserstein_distance
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt


def load_and_prepare_error_data(file_path: str, regions: list) -> np.ndarray:
    """
    Read the NordPool CSV, clean data, return forecast errors
    """
    df = pd.read_csv(file_path)
    df = df.replace({',': '', ' ': ''}, regex=True)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    for r in regions:
        df[f'{r}-Error'] = (
            (df[f'{r}-Actual'] - df[f'{r}-Forecast'])
            / df[f'{r}-Actual']
        )

    err_cols = [f'{r}-Error' for r in regions]
    df[err_cols] = df[err_cols].replace({np.inf: np.nan, -np.inf: -10})
    error_data = df[err_cols].dropna().values

    return error_data


def fit_constraint_informed(aggregate_errors: np.ndarray,
                            n_components: int = 3,
                            random_state: int = 100) -> GaussianMixture:
    """
    Constraint-informed fitting on 1D aggregate
    """
    gmm = GaussianMixture(n_components=n_components, random_state=random_state)
    gmm.fit(aggregate_errors.reshape(-1, 1))
    return gmm


def fit_classical_projected(error_data: np.ndarray,
                            n_components: int = 3,
                            random_state: int = 100):
    """
    Classical fitting with transformation of D-dimensional parameters
    """
    gmm = GaussianMixture(n_components=n_components, random_state=random_state)
    gmm.fit(error_data)
    weights = gmm.weights_
    mu_proj = gmm.means_.sum(axis=1)
    sigma_proj = np.sqrt(np.einsum(
        'ij,ijk,ik->i',
        np.ones((n_components, error_data.shape[1])),
        gmm.covariances_,
        np.ones((n_components, error_data.shape[1]))
    ))
    return gmm, weights, mu_proj, sigma_proj


def sample_constraint(gmm: GaussianMixture, n: int) -> np.ndarray:
    return gmm.sample(n)[0].flatten()


def sample_classical_projected(weights: np.ndarray,
                               mu_proj: np.ndarray,
                               sigma_proj: np.ndarray,
                               n: int) -> np.ndarray:
    comps = np.random.choice(len(weights), size=n, p=weights)
    return np.array([
        np.random.normal(mu_proj[c], sigma_proj[c])
        for c in comps
    ])


def compute_wasserstein(aggregate_errors: np.ndarray,
                        samples_ci: np.ndarray,
                        samples_cl: np.ndarray):
    wd_ci = wasserstein_distance(aggregate_errors.flatten(), samples_ci)
    wd_cl = wasserstein_distance(aggregate_errors.flatten(), samples_cl)
    return wd_ci, wd_cl


def compute_log_likelihood_1d(data: np.ndarray,
                              weights: np.ndarray,
                              means: np.ndarray,
                              covs: np.ndarray) -> float:
    ll = 0.0
    for x in data:
        p = sum(w * norm.pdf(x, loc=m, scale=np.sqrt(c))
                for w, m, c in zip(weights, means, covs))
        ll += np.log(p)
    return ll


def plot_nordpool(aggregate_errors: np.ndarray,
                  gmm_ci: GaussianMixture,
                  weights_cl: np.ndarray,
                  mu_proj: np.ndarray,
                  sigma_proj: np.ndarray):
    x = np.linspace(aggregate_errors.min(), aggregate_errors.max(), 1000)

    pdf_ci = np.exp(gmm_ci.score_samples(x.reshape(-1, 1)))
    pdf_cl = sum(
        w * (1 / (np.sqrt(2 * np.pi) * s)) * np.exp(-0.5 * ((x - m) / s) ** 2)
        for w, m, s in zip(weights_cl, mu_proj, sigma_proj)
    )

    plt.figure(figsize=(8, 8))
    hist = plt.hist(
        aggregate_errors.flatten(),
        bins=2000,
        density=True,
        color='skyblue',
        edgecolor='black',
        alpha=0.5,
        label='Original Data Histogram'
    )
    line_ci, = plt.plot(
        x, pdf_ci,
        'forestgreen', linestyle='--', linewidth=3,
        label='Constraint-Informed GMM (K=3)'
    )
    line_cl, = plt.plot(
        x, pdf_cl,
        'darkorange', linestyle=':', linewidth=3,
        label='Classical GMM (K=3)'
    )

    handles = [hist[2][0], line_cl, line_ci]
    labels = [
        'Original Data Histogram',
        'Classical GMM (K=3)',
        'Constraint-Informed GMM (K=3)'
    ]
    plt.legend(handles=handles, labels=labels, fontsize='x-large')
    plt.title("NordPool", fontsize=18)
    plt.xlabel("Aggregate Forecast Errors", fontsize=18)
    plt.ylabel("Density", fontsize=18)
    plt.xlim(-8, 3)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    file_path = '../data/15daysNordPoolFinal.csv'
    regions = ['50Hz', 'AMP', 'AT', 'PL', 'TBW', 'TTG', 'TTGf',
               '50Hzf', 'FR', 'FRf']

    error_data = load_and_prepare_error_data(file_path, regions)

    aggregate_errors = error_data.sum(axis=1)

    gmm_ci = fit_constraint_informed(aggregate_errors)
    _, weights_cl, mu_proj, sigma_proj = fit_classical_projected(error_data)

    # Optional, alternative metric, cmopute wasserstein distance between sets of samples
    samples_ci = sample_constraint(gmm_ci, len(aggregate_errors))
    samples_cl = sample_classical_projected(
        weights_cl, mu_proj, sigma_proj, len(aggregate_errors)
    )

    wd_ci, wd_cl = compute_wasserstein(
        aggregate_errors, samples_ci, samples_cl
    )
    print(f"Wasserstein Distance (constraint_informed GMM): {wd_ci}")
    print(f"Wasserstein Distance (Projected_classical GMM): {wd_cl}")

    ll_ci = compute_log_likelihood_1d(
        aggregate_errors,
        gmm_ci.weights_,
        gmm_ci.means_.flatten(),
        gmm_ci.covariances_.flatten()
    )
    ll_cl = compute_log_likelihood_1d(
        aggregate_errors,
        weights_cl,
        mu_proj,
        sigma_proj**2
    )
    print("=== Log-Likelihood on Aggregate Data ===")
    print(f"Constraint-Informed GMM Log-Likelihood: {ll_ci}")
    print(f"Classical GMM (1^T Transformed) Log-Likelihood: {ll_cl}")

    plot_nordpool(aggregate_errors, gmm_ci, weights_cl, mu_proj, sigma_proj)


if __name__ == "__main__":
    main()
