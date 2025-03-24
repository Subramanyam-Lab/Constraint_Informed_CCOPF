import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from scipy.stats import norm, wasserstein_distance

np.random.seed(58)
# ---------------------------
# 1) Load the error data CSV file
# ---------------------------
error_file_path = '/Users/tianyangyi/Desktop/Constraint_Informed_CCOPF/data/Corrected_Forecast_Errors_for_All_Locations.csv'
error_df = pd.read_csv(error_file_path)

# Identify the error columns
error_columns = [col for col in error_df.columns if '_ERROR' in col]

# Extract the error data (11D vectors)
error_data = error_df[error_columns].dropna().values

# ---------------------------
# 2) Compute 1D sums (Aggregate Errors)
# ---------------------------
aggregate_errors = error_data.sum(axis=1).reshape(-1, 1)

# ---------------------------
# 3A) Theoretical GMM: Fit a 2-component GMM in 11D and Project to 1D
# ---------------------------
gmm_11d = GaussianMixture(n_components=2, random_state=42)
gmm_11d.fit(error_data)

# Extract GMM parameters
weights_11d = gmm_11d.weights_
means_11d = gmm_11d.means_         # shape (2, 11)
covs_11d = gmm_11d.covariances_    # shape (2, 11, 11)

# Project the 11D GMM to 1D
ones_vector = np.ones(11)
projected_weights = weights_11d
projected_means = [ones_vector @ means_11d[k] for k in range(2)]
projected_stds = [np.sqrt(ones_vector @ covs_11d[k] @ ones_vector) for k in range(2)]

# ---------------------------
# 3B) Empirical GMM: Fit a 2-component GMM directly on 1D sums
# ---------------------------
gmm_1d = GaussianMixture(n_components=2, random_state=42)
gmm_1d.fit(aggregate_errors)

empirical_weights = gmm_1d.weights_
empirical_means = gmm_1d.means_.flatten()
empirical_stds = np.sqrt(gmm_1d.covariances_).flatten()

# ---------------------------
# 4) Compare with Wasserstein Distance
# ---------------------------
# Sampling for Wasserstein Distance Comparison
def sample_from_1d_mixture(n_samples, weights, means, stds):
    components = np.random.choice(range(len(weights)), size=n_samples, p=weights)
    samples = np.array([np.random.normal(means[c], stds[c]) for c in components])
    return samples

theoretical_samples_1d = sample_from_1d_mixture(1000, projected_weights, projected_means, projected_stds)
empirical_samples_1d, _ = gmm_1d.sample(1000)
empirical_samples_1d = empirical_samples_1d.ravel()

# Wasserstein Distance Calculation
wd_theoretical = wasserstein_distance(aggregate_errors.flatten(), theoretical_samples_1d)
wd_empirical = wasserstein_distance(aggregate_errors.flatten(), empirical_samples_1d)

print("Wasserstein Distance (11D -> 1D GMM) vs. Data:", wd_theoretical)
print("Wasserstein Distance (1D GMM) vs. Data:", wd_empirical)

# ---------------------------
# 5) Plot Histograms with GMM PDFs
# ---------------------------
plt.figure(figsize=(8,8))

# Histogram of the actual aggregate errors
plt.hist(aggregate_errors, bins=40, density=True, alpha=0.5, color='skyblue', edgecolor='black',
         label='Actual Aggregate Errors (Data)')

# X-axis for PDF plotting
x_values = np.linspace(aggregate_errors.min(), aggregate_errors.max(), 1000)

# Theoretical GMM PDF (Projected from 11D)
pdf_theoretical = sum(
    w * norm.pdf(x_values, loc=m, scale=s)
    for w, m, s in zip(projected_weights, projected_means, projected_stds)
)

# Empirical GMM PDF (Fitted directly on 1D sums)
pdf_empirical = sum(
    w * norm.pdf(x_values, loc=m, scale=s)
    for w, m, s in zip(empirical_weights, empirical_means, empirical_stds)
)

# Plotting the PDFs
plt.plot(x_values, pdf_theoretical, 'darkorange', linestyle=':',linewidth=3, label='Fit-then-Transform GMM (K=2)')
plt.plot(x_values, pdf_empirical, 'forestgreen', linestyle='--',linewidth=3, label='Transform-then-Fit GMM (K=2)')

# Plot Formatting
plt.title("GMM(K=2) Distributed Load Forecast Errors", fontsize = 14)
plt.xlabel("Aggregate Load Forecast Errors", fontsize = 14)
plt.ylabel("Density", fontsize = 14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend()
plt.grid(True)

plt.show()
