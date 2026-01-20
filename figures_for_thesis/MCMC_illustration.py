import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# 1. Define a fake posterior
# ----------------------------
def posterior(k):
    """Unnormalized posterior density."""
    return (
        0.7 * np.exp(-0.5 * ((k - 1.5) / 0.4)**2) +
        0.3 * np.exp(-0.5 * ((k + 1.0) / 0.7)**2)
    )

# ----------------------------
# 2. Simple Metropolis sampler
# ----------------------------
np.random.seed(0)

n_samples = 5000
proposal_std = 0.5

chain = np.zeros(n_samples)
chain[0] = 0.0  # initial point

for i in range(1, n_samples):
    current = chain[i - 1]
    proposal = current + proposal_std * np.random.randn()

    accept_ratio = posterior(proposal) / posterior(current)
    if np.random.rand() < accept_ratio:
        chain[i] = proposal
    else:
        chain[i] = current

# ----------------------------
# 3. Plot posterior + samples
# ----------------------------
k_grid = np.linspace(-3, 3, 500)
p_grid = posterior(k_grid)

plt.figure(figsize=(6, 3))

# Posterior curve
plt.plot(k_grid, p_grid, color="black", lw=2, label="Posterior")

# MCMC samples
nbr_samples_param = 30
plt.scatter(
    chain[::nbr_samples_param],
    np.zeros_like(chain[::nbr_samples_param]),
    s=10,
    color="tab:blue",
    alpha=0.4,
    label="MCMC samples"
)

plt.yticks([])
plt.xlabel("Parameter k")
plt.legend(frameon=False)

plt.tight_layout()
plt.show()