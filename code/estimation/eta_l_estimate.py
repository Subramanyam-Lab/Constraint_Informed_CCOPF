import os
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

def generate_cauchy_samples(N=2000, D=10, scale=0.02, seed=125):
    """Generate N×D Cauchy wind‐error samples."""
    np.random.seed(seed)
    return np.random.standard_cauchy(size=(N, D)) * scale

def load_ptdf_matrix(csv_path):
    """
    Load PTDF from CSV and return
      - H: (L×B) 
      - line_ids: list of lines
    """
    df = pd.read_csv(csv_path, index_col=0)
    return df.values, df.index.astype(str).tolist()

def fit_2d_gmm(omega, lambda_l, n_components=3, random_state=0, max_iter=500):
    """Fit a 2-dimensional GMM on eta_l and return the fitted model."""
    samples = np.column_stack((omega, lambda_l))
    gmm = GaussianMixture(n_components=n_components,
                          covariance_type='full',
                          random_state=random_state,
                          max_iter=max_iter)
    gmm.fit(samples)
    return gmm

def main():
    ptdf_csv       = "../data/ptdf_matrix.csv"
    wind_buses_1   = [10,25,26,59,65,66,69,80,100,111]  # bus id that has wind units
    n_components   = 3
    output_csv     = "../data/gmm_all_lines.csv"

    xi    = generate_cauchy_samples()         
    Omega = xi.sum(axis=1)                    

    H, line_ids = load_ptdf_matrix(ptdf_csv)  
    W_idx       = [b-1 for b in wind_buses_1]  
    H_wind      = H[:, W_idx]                 

    # Fit 2D GMM to each line
    rows = []
    for i, line_id in enumerate(line_ids):
        Lambda_l = xi.dot(H_wind[i, :])       
        gmm      = fit_2d_gmm(Omega, Lambda_l, n_components)

        # .cCSV cannot store matrices, so instead, save 2 entries from the mean estimate and the 3 entries from covariance estimates
        for k in range(n_components):
            rows.append({
                "line_id":           line_id,
                "component":         k+1,
                "weight":            gmm.weights_[k],
                "mean_Omega":        gmm.means_[k, 0],
                "mean_Lambda":       gmm.means_[k, 1],
                "cov_Omega_Omega":   gmm.covariances_[k, 0, 0],
                "cov_Omega_Lambda":  gmm.covariances_[k, 0, 1],
                "cov_Lambda_Lambda": gmm.covariances_[k, 1, 1],
            })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Saved all GMM parameters to {output_csv}")

if __name__ == "__main__":
    main()
