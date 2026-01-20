import numpy as np
import matplotlib.pyplot as plt

# Define p values
p_vals = np.linspace(0.01, 0.1, 500)

alpha = 0.99

# Compute required n
n_vals = np.log(1 - alpha) / np.log(1 - p_vals)

plt.figure(figsize=(6,4))
plt.plot(p_vals, n_vals)
plt.xlabel("Success probability p")
plt.ylabel("Required n for 99.9% confidence")
plt.title("Multistart count n vs success probability p")
plt.grid(True)

plt.show(block=True)