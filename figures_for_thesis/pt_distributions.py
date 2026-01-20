import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Define a 1D grid for plotting
x = np.linspace(-15, 25, 10_000)

# Define two Gaussian components (means and stds)
mu1, sigma1 = -3, 0.7
mu2, sigma2 =  10, 1.5

# Mixture weights
w1, w2 = 0.5, 0.5

# Posterior (unnormalized): mixture of two Gaussians
posterior = w1 * norm.pdf(x, mu1, sigma1) + w2 * norm.pdf(x, mu2, sigma2)

# Function to temper the posterior: π_T(x) ∝ π(x)^(1/T)
def temper(p, T):
    p_T = p ** (1 / T)
    return p_T / np.trapz(p_T, x)  # normalize so it integrates to 1

# Temperatures
T1 = 1
T2 = 20

posterior_T1 = temper(posterior, T1)
posterior_T2 = temper(posterior, T2)

# Plot the results
plt.figure(figsize=(9,5))
plt.plot(x, posterior_T1, label="T = 1 (True posterior)", linewidth=2)
plt.plot(x, posterior_T2, label="T = 20 (Hot / flattened)", linewidth=2)
plt.title("Effect of Tempering on a Multimodal Posterior")
plt.xlabel("Parameter k")
plt.ylabel("Density")
plt.legend()
plt.grid(alpha=0.3)
plt.show()