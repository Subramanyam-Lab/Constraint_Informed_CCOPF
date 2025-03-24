import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wasserstein_distance 

file_path = '/Users/tianyangyi/Desktop/Constraint_Informed_CCOPF/data/15daysNordPoolFinal.csv'  
df = pd.read_csv(file_path)

df_cleaned = df.replace({',': '', ' ': ''}, regex=True)
for col in df_cleaned.columns:
    df_cleaned[col] = pd.to_numeric(df_cleaned[col])

# Compute forecast errors for each region
regions = ['50Hz', 'AMP', 'AT', 'PL', 'TBW', 'TTG', 'TTGf', '50Hzf', 'FR', 'FRf']
for region in regions:
    df_cleaned[f'{region}-Error'] = (df_cleaned[f'{region}-Actual'] - df_cleaned[f'{region}-Forecast']) / df_cleaned[f'{region}-Actual']

df_cleaned.to_csv('/Users/tianyangyi/Desktop/Constraint_Informed_CCOPF/data/15daysNordPoolFinalErrors.csv', index=False)

# Extract normalized errors for the 10 regions
error_columns = [f'{region}-Error' for region in regions]
df_cleaned[error_columns] = df_cleaned[error_columns].replace([np.inf], np.nan)
df_cleaned[error_columns] = df_cleaned[error_columns].replace([-np.inf], -10)
error_data = df_cleaned[error_columns].dropna().values

# Constraint-informed GMM 
aggregate_errors = error_data.sum(axis=1).reshape(-1, 1)
gmm_constraint_informed = GaussianMixture(n_components=3, random_state=100)
gmm_constraint_informed.fit(aggregate_errors)
samples_constraint_informed = gmm_constraint_informed.sample(n_samples=len(aggregate_errors))[0].flatten()
print(gmm_constraint_informed.means_)
print(gmm_constraint_informed.covariances_)
print(gmm_constraint_informed.weights_)

# Classical GMM
gmm_classical = GaussianMixture(n_components=3, random_state=100)
gmm_classical.fit(error_data)
samples_classical = gmm_classical.sample(n_samples=len(aggregate_errors))[0]
samples_classical_projected = samples_classical.sum(axis=1)

wasserstein_constraint_informed = wasserstein_distance(aggregate_errors.flatten(), samples_constraint_informed)
wasserstein_classical = wasserstein_distance(aggregate_errors.flatten(), samples_classical_projected)

print(f"Wasserstein Distance (constraint_informed GMM): {wasserstein_constraint_informed}")
print(f"Wasserstein Distance (Projected_classical GMM): {wasserstein_classical}")

# Project the_classical GMM to constraint_informed using 1^T * mu and sqrt(1^T * Sigma * 1)
mu_projected = gmm_classical.means_.sum(axis=1)
sigma_projected = np.sqrt(np.einsum('ij,ijk,ik->i', np.ones((3, 10)), gmm_classical.covariances_, np.ones((3, 10))))
weights_projected = gmm_classical.weights_
print(mu_projected)
print(sigma_projected)
print(weights_projected)


x = np.linspace(aggregate_errors.min(), aggregate_errors.max(), 1000)
# PDF of the constraint_informed GMM
pdf_constraint_informed = np.exp(gmm_constraint_informed.score_samples(x.reshape(-1, 1)))
# PDF of the projected_classical GMM
pdf_classical_projected = sum(
    w * (1 / (np.sqrt(2 * np.pi) * s)) * np.exp(-0.5 * ((x - m) / s) ** 2)
    for w, m, s in zip(weights_projected, mu_projected, sigma_projected)
)
plt.figure(figsize=(8, 8))
hist = plt.hist(aggregate_errors.flatten(), bins=2000, density=True, color='skyblue',  edgecolor='black', alpha=0.5, label='Original Data Histogram')
gmm_constraint, = plt.plot(x, pdf_constraint_informed, 'forestgreen', linestyle='--', linewidth=3, label='Constraint-Informed GMM (K=3)')
gmm_classical, = plt.plot(x, pdf_classical_projected, 'darkorange', linestyle=':', linewidth=3, label='Classical GMM (K=3)')
handles = [hist[2][0], gmm_classical, gmm_constraint]  
labels = ['Original Data Histogram', 'Classical GMM (K=3)', 'Constraint-Informed GMM (K=3)']
plt.legend(handles=handles, labels=labels, fontsize='x-large')
plt.title("Estimation of NordPool Forecast Errors", fontsize=18)
plt.xlabel("Aggregate Forecast Errors", fontsize=18)
plt.ylabel("Density", fontsize=18)
plt.xlim(-8,3)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.grid(True)

plt.show()