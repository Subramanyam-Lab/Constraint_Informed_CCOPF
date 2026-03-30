import pandas as pd
import numpy as np
from scipy.stats import norm, wasserstein_distance
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt


### Clean NordPool data file
def load_and_prepare_error_data(file_path: str, regions: list) -> np.ndarray:
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
                            n_components: int = 6,
                            random_state: int = 100) -> GaussianMixture:
    gmm = GaussianMixture(n_components=n_components, random_state=random_state)
    gmm.fit(aggregate_errors.reshape(-1, 1))
    return gmm


def fit_classical_projected(error_data: np.ndarray,
                            n_components: int = 9,
                            random_state: int = 100):
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
        label='Constraint-Informed GMM (K=6)'
    )
    line_cl, = plt.plot(
        x, pdf_cl,
        'darkorange', linestyle=':', linewidth=3,
        label='Classical GMM (K=9)'
    )

    handles = [hist[2][0], line_cl, line_ci]
    labels = [
        'Original Data Histogram',
        'Classical GMM (K=9)',
        'Constraint-Informed GMM (K=6)'
    ]
    plt.legend(handles=handles, labels=labels, fontsize='x-large')
    plt.title("NordPool", fontsize=18)
    #plt.xlabel("Aggregate Forecast Errors", fontsize=18)
    plt.ylabel("Density", fontsize=18)
    plt.xlim(-8, 3)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(True)
    #plt.tight_layout()
    plt.savefig("../figures/nordpool_density.png", dpi=150, bbox_inches="tight")
    plt.show()


def main():
    file_path = '../data/15daysNordPoolFinalErrors.csv'
    regions = ['50Hz', 'AMP', 'AT', 'PL', 'TBW', 'TTG', 'TTGf',
               '50Hzf', 'FR', 'FRf']

    error_data = load_and_prepare_error_data(file_path, regions)

    aggregate_errors = error_data.sum(axis=1)

    gmm_ci = fit_constraint_informed(aggregate_errors)
    _, weights_cl, mu_proj, sigma_proj = fit_classical_projected(error_data)
    plot_nordpool(aggregate_errors, gmm_ci, weights_cl, mu_proj, sigma_proj)


if __name__ == "__main__":
    main()
