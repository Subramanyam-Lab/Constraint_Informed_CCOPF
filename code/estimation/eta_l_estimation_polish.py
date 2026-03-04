import os
import time
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture


def generate_cauchy_samples(N, D, gamma, seed=42):
    np.random.seed(seed)
    return np.random.standard_cauchy(size=(N, D)) * gamma

# Load Hl submatrix 
def load_wind_ptdf(csv_path):
    df = pd.read_csv(csv_path)
    line_ids = df["line_id"].astype(str).tolist()
    H = df.drop(columns=["line_id"]).values
    return H, line_ids

# Fit either a tied or spherical 2d GMM based on BIC scores
def fit_best_2d_gmm_by_bic(
    omega,
    lambda_l,
    n_components=3,
    covariance_types=("tied", "spherical"),
    n_init=10,
    base_seed=0,
    max_iter=100,
):
    X = np.column_stack((omega, lambda_l))

    best_gmm = None
    best_bic = np.inf
    best_info = {}

    for cov_type in covariance_types:
        for i in range(n_init):
            seed = base_seed + i
            gmm = GaussianMixture(
                n_components=n_components,
                covariance_type=cov_type,
                random_state=seed,
                max_iter=max_iter,
                init_params="kmeans",
            )
            gmm.fit(X)
            bic = gmm.bic(X)

            if bic < best_bic:
                best_bic = bic
                best_gmm = gmm
                best_info = {
                    "covariance_type": cov_type,
                    "seed": seed,
                    "bic": bic,
                }

    return best_gmm, best_info


def main():

    ptdf_csv = "../../data/H_wind_matrix.csv"
    output_csv = "../../data/gmm_gamma_001_results.csv"

    gamma = 0.01
    N = 8000

    H_wind, line_ids = load_wind_ptdf(ptdf_csv)
    L, D = H_wind.shape

    print(f"Lines: {L}, Wind units: {D}")

    xi = generate_cauchy_samples(N=N, D=D, gamma=gamma)
    Omega = xi.sum(axis=1)

    rows = []
    total_fit_time = 0.0
    overall_start = time.perf_counter()

    ####################################################################################
    ### Main Loop for fitting, for each line, generate the 2d sampels using Hl matrix ##
    ####################################################################################
    for i, line_id in enumerate(line_ids):

        line_start = time.perf_counter()

        Lambda_l = xi.dot(H_wind[i, :])

        gmm, info = fit_best_2d_gmm_by_bic(
            Omega,
            Lambda_l,
            n_components=3,
            covariance_types=("tied", "spherical"),
            n_init=10,
        )

        elapsed = time.perf_counter() - line_start
        total_fit_time += elapsed

        cov_type = info["covariance_type"]

        for k in range(gmm.n_components):

            if cov_type == "tied":
                cov_mat = gmm.covariances_
                cov00 = cov_mat[0, 0]
                cov01 = cov_mat[0, 1]
                cov11 = cov_mat[1, 1]
            else:
                var = gmm.covariances_[k]
                cov00 = var
                cov01 = 0.0
                cov11 = var

            rows.append({
                "gamma": gamma,
                "line_id": line_id,
                "component": k + 1,
                "weight": gmm.weights_[k],
                "mean_Omega": gmm.means_[k, 0],
                "mean_Lambda": gmm.means_[k, 1],
                "cov_Omega_Omega": cov00,
                "cov_Omega_Lambda": cov01,
                "cov_Lambda_Lambda": cov11,
                "covariance_type": cov_type,
                "seed": info["seed"],
                "bic": info["bic"],
                "fit_time_seconds": elapsed
            })

    overall_elapsed = time.perf_counter() - overall_start

    # Save results
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)

    print(f"Average time per line: {total_fit_time / L:.4f} ")
    print(f"Total time for serial fitting: {overall_elapsed:.2f} ")


if __name__ == "__main__":
    main()