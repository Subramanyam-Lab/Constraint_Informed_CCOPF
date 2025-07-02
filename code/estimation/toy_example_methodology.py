import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.mixture import GaussianMixture

#This script is used to plot the figure and get the estimators in section III of the paper.
#One can change the dimension (number of wind units), the Cauchy parameters, number of GMM components,
#and number of different iterations for the EM algorithm

def generate_cauchy_samples(N, D, scale):
    """Generate N samples of D-dimensional Cauchy forecast errors, and the data samples for Omega."""
    samples_5d = np.random.standard_cauchy(size=(N, D)) * scale
    aggregated_1d = samples_5d.sum(axis=1).reshape(-1, 1)
    return samples_5d, aggregated_1d

def best_gmm(X, K, n_inits):
    """Fit a K-component GMM to data samples (bxi, and Omega) with n_inits different EM starts, return model with lowest BIC."""
    best_bic, best_model = np.inf, None
    for seed in range(n_inits):
        gmm = GaussianMixture(
            n_components=K,
            covariance_type='full',
            random_state=seed,
            max_iter=500
        ).fit(X)
        bic = gmm.bic(X)
        if bic < best_bic:
            best_bic, best_model = bic, gmm
    return best_model

def project_gmm_to_1d(gmm, D):
    """Given a D‐dimensional GMM parameters, transform them to 1D using the bone^top transformation, this
    corresponds to the classical approach of obtaining statistical estimators."""
    ones = np.ones((D, 1))
    means_1d = (gmm.means_ @ ones).flatten()
    vars_1d = np.array([
        ones.T @ gmm.covariances_[k] @ ones
        for k in range(gmm.n_components)
    ]).flatten()
    weights = gmm.weights_
    return means_1d, vars_1d, weights

def plot_densities(aggregated, params_5d, params_1d, gaussian_params):
    """
    Plot the histogram of "true" aggregate system-wide errors and the PDF curves of:
    - Classical approach (projected D-dimensional GMM to 1D)
    - Constraint‐informed approach (1D GMM on Omega)
    - Single Gaussian (special case when both approaches agree)
    """
    means_5d, vars_5d, weights_5d = params_5d
    means_1d, vars_1d, weights_1d = params_1d
    mean_g, var_g = gaussian_params

    plt.figure(figsize=(8, 5))
    plt.hist(aggregated.flatten(), bins=25000, density=True,
             alpha=0.6, color='skyblue', label='Forecast Error Samples')

    x_vals = np.linspace(-10, 10, 1000)
    # Classical projected GMM from D-dimensional
    pdf_5d = sum(w * norm.pdf(x_vals, loc=m, scale=np.sqrt(v))
                 for w, m, v in zip(weights_5d, means_5d, vars_5d))
    plt.plot(x_vals, pdf_5d, color='darkorange', linestyle=':',
             lw=3, label='Classical GMM')
    # Constraint-informed 1D GMM
    pdf_1d = sum(w * norm.pdf(x_vals, loc=m, scale=np.sqrt(v))
                 for w, m, v in zip(weights_1d, means_1d, vars_1d))
    plt.plot(x_vals, pdf_1d, color='forestgreen', linestyle='--',
             lw=3, label='Constraint-Informed GMM')
    # Single Gaussian
    pdf_g = norm.pdf(x_vals, loc=mean_g, scale=np.sqrt(var_g))
    plt.plot(x_vals, pdf_g, color='red', linestyle='-', lw=3,
             label='Gaussian Fit')

    plt.xlabel('Aggregate Forecast Error', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.xlim(-10, 10)
    plt.title('Aggregated Forecast Errors Fitted with Different Approaches',
              fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    np.random.seed(42)
    N, D, scale, K, n_inits = 5000, 5, 0.2, 3, 10

    samples_5d, aggregated_1d = generate_cauchy_samples(N, D, scale)

    gmm_5d = best_gmm(samples_5d, K, n_inits)
    params_5d = project_gmm_to_1d(gmm_5d, D)

    gmm_1d = best_gmm(aggregated_1d, K, n_inits)
    params_1d = (gmm_1d.means_.flatten(),
                 gmm_1d.covariances_.flatten(),
                 gmm_1d.weights_)
    
    mean_g, var_g = aggregated_1d.mean(), aggregated_1d.var()

    plot_densities(aggregated_1d, params_5d, params_1d, (mean_g, var_g))

    means_5d, vars_5d, w_5d = params_5d
    print("===== Classical GMM  =====")
    for i, (m, v, w) in enumerate(zip(means_5d, vars_5d, w_5d), 1):
        print(f"Component {i}: Mean = {m:.4f}, Variance = {v:.4f}, Weight = {w:.4f}")

    means_1d, vars_1d, w_1d = params_1d
    print("\n===== Constraint-Informed GMM  =====")
    for i, (m, v, w) in enumerate(zip(means_1d, vars_1d, w_1d), 1):
        print(f"Component {i}: Mean = {m:.4f}, Variance = {v:.4f}, Weight = {w:.4f}")

    print("\n===== Gaussian Fit =====")
    print(f"Mean = {mean_g:.4f}, Variance = {var_g:.4f}")

if __name__ == "__main__":
    main()
