import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

# Parameters for the Cauchy distribution
location = 0  
scale = 0.07   

# Sample 2000 data points from the Cauchy distribution
np.random.seed(28)
data = np.random.standard_cauchy(size=1500) * scale + location

gmm = GaussianMixture(n_components=2, random_state=42)
data_reshaped = data.reshape(-1, 1)
gmm.fit(data_reshaped)

weights = [0.98, 0.02]
means = [0, -0.2]
variances = [0.01, 0.35]
std_devs = np.sqrt(variances)

# Generate density values from the custom GMM
x = np.linspace(-5, 5, 1000).reshape(-1, 1)
log_density = gmm.score_samples(x)
density = np.exp(log_density)

density_custom = (
    weights[0] * (1 / (std_devs[0] * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - means[0]) / std_devs[0]) ** 2) +
    weights[1] * (1 / (std_devs[1] * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - means[1]) / std_devs[1]) ** 2)
)

plt.figure(figsize=(10, 6))
plt.hist(data, bins=1000, density=True, color = 'yellow',edgecolor='black', alpha=0.7, label='Original Data')
plt.plot(x, density, color='blue', linewidth=2, label='Fit-then-Transform')
plt.plot(x, density_custom, color='red', linewidth=2, label='Transform-then-Fit')
plt.xlabel('Forecast Error', fontsize=20)
plt.ylabel('Density', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title('Aggregate System-Wide Forecast Error', fontsize=20)
plt.xlim(-5, 5)
plt.legend(fontsize=16)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()