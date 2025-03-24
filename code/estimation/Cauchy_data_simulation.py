import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy.stats import norm
from scipy.stats import wasserstein_distance

np.random.seed(125)
N = 2000
D = 10
scale = 0.02

# Original 10D random vectors
samples_10d = np.random.standard_cauchy(size=(N, D)) * scale

# Classical Approach
gmm_10d = GaussianMixture(n_components=2)
gmm_10d.fit(samples_10d)

# Extract the fitted GMM parameters in 10D
weights_10d = gmm_10d.weights_          
means_10d   = gmm_10d.means_            
covs_10d    = gmm_10d.covariances_      
ones_10 = np.ones((D, 1))  
m_1d = []
s2_1d = []

for k in range(2):
    mu_k  = means_10d[k]          
    Sigma_k = covs_10d[k]         
    m_k = mu_k.sum()
    s_k2 = ones_10.T @ Sigma_k @ ones_10  
    s_k2 = s_k2.item()  

    m_1d.append(m_k)
    s2_1d.append(s_k2)

weights_1d_classical = weights_10d.copy()  
means_1d_classical   = np.array(m_1d)      
vars_1d_classical    = np.array(s2_1d)     

# Constraint-Informed
sums_1d = samples_10d.sum(axis=1)  
gmm_1d_constraint_informed = GaussianMixture(n_components=2)
gmm_1d_constraint_informed.fit(sums_1d.reshape(-1, 1))

weights_1d_constraint_informed = gmm_1d_constraint_informed.weights_
means_1d_constraint_informed   = gmm_1d_constraint_informed.means_.ravel()     
vars_1d_constraint_informed    = gmm_1d_constraint_informed.covariances_.ravel()

plt.figure(figsize=(8,8))
counts, bins, _ = plt.hist(sums_1d, bins=2000, density=True, alpha=0.5, 
                           color='skyblue', edgecolor='black', label='Original Data Histogram')

x_plot = np.linspace(-5,5,2000)
pdf_classical = np.zeros_like(x_plot)
for w, m, v in zip(weights_1d_classical, means_1d_classical, vars_1d_classical):
    pdf_classical += w * norm.pdf(x_plot, loc=m, scale=np.sqrt(v))

plt.plot(x_plot, pdf_classical, 'darkorange', linestyle=':',lw=3, 
         label='Classical GMM (K=2)')
logprob_constraint_informed = gmm_1d_constraint_informed.score_samples(x_plot.reshape(-1, 1))
pdf_constraint_informed = np.exp(logprob_constraint_informed)
plt.plot(x_plot, pdf_constraint_informed, 'forestgreen', linestyle='--',lw=3, 
         label='Constraint-Informed GMM (K=2)')

plt.xlim(-5,5)
plt.title("Estimation of Synthetic Cauchy Forecast Errors", fontsize = 18)
plt.xlabel("Aggregate Wind Forecast Errors", fontsize = 18)
plt.ylabel("Density", fontsize = 18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(loc='upper left', fontsize='x-large')
plt.grid(True)
plt.show()

# Wasserstein Distance
def sample_from_1d_mixture(n_samples, weights, means, stds):
    components = np.random.choice(len(weights), size=n_samples, p=weights)
    samples = np.array([
        np.random.normal(means[c], stds[c]) for c in components
    ])
    return samples

ftt_samples_1d = sample_from_1d_mixture(
    n_samples=1000,
    weights=weights_1d_classical,
    means=means_1d_classical,
    stds=vars_1d_classical
)

ttf_samples_1d, _ = gmm_1d_constraint_informed.sample(1000)
ttf_samples_1d = ttf_samples_1d.ravel()

wd_ftt = wasserstein_distance(sums_1d, ftt_samples_1d)
wd_ttf = wasserstein_distance(sums_1d, ttf_samples_1d)

print("Wasserstein distance (classical GMM vs. data):", wd_ftt)
print("Wasserstein distance (constraint_informed GMM   vs. data):", wd_ttf)