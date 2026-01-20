import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
from typing import Callable, Sequence
import jax

def scale_image_to_mean(I, target_mean=1.0):
    """Scale image so that its mean becomes target_mean."""
    current_mean = jnp.mean(I)
    return I * (target_mean / current_mean)

def add_poisson(I_clean_norm, scale=20_000.0, seed=0):
    key = jax.random.PRNGKey(seed)
    lam = I_clean_norm * scale
    return jax.random.poisson(key, lam=lam)

# def add_poisson_noise(I_scaled, seed=0, threshold=20.0):
#     """
#     Apply hybrid Poisson/Gaussian noise to a scaled image using NumPy.

#     Parameters
#     ----------
#     I_scaled : np.ndarray
#         Scaled intensity image (float64), units: photons/pixel.
#     seed : int
#         Random seed for reproducibility.
#     threshold : float
#         Cutoff for using exact Poisson vs Gaussian approximation.

#     Returns
#     -------
#     np.ndarray
#         Noisy image, float64
#     """
#     rng = np.random.default_rng(seed) # Same randomness every run
#     I_noisy = np.zeros_like(I_scaled)

#     for idx, pixel_value in np.ndenumerate(I_scaled): # One pixel at a time, row by row: idx is the pixel index and pixel_value is the pixel value
#         if pixel_value < 0:
#             raise ValueError("Negative intensity not allowed.")
#         elif pixel_value <= threshold:
#             # Inverse transform sampling of Poisson
#             p_threshold = np.exp(-pixel_value) # Small lambdas will give large probabilities here -> unlikely to stay in while loop below
#             k = 0
#             p = 1.0
#             while p > p_threshold:
#                 k += 1 # Add more noise each time p has not reached below p_threshold
#                 p *= rng.random()
#             I_noisy[idx] = max(k - 1, 0)
#         else: # For really large pixel values, Gaussian noise is good model
#             val = pixel_value + np.sqrt(pixel_value) * rng.standard_normal()
#             I_noisy[idx] = max(np.round(val), 0)

#     return I_noisy


# def add_noise_to_image(I_clean_norm, poisson_scale=10.0, noise_magnitude=0.1, seed=0):
#     """
#     Add noise to a clean image normalized to [0,1].

#     Parameters
#     ----------
#     I_clean_norm : jnp.ndarray
#         Clean image from simulator, already normalized to [0, 1].
#     poisson_scale : float
#         Scale to simulate Poisson counts (e.g. 5 → max ≈ 5 photons/pixel).
#         Controls stochasticity of Poisson noise.
#     noise_magnitude : float
#         Final scaling of noise relative to signal (e.g. 1.0 → same strength as signal).
#     seed : int
#         RNG seed

#     Returns
#     -------
#     jnp.ndarray
#         Noisy observed image, normalized to [0, 1].
#     """
#     # Step 1: simulate photon counts
#     I_scaled_np = np.asarray(I_clean_norm * poisson_scale) # Scaled enough to trigger noise according to Cyrille's rule
#     I_noise= add_poisson_noise(I_scaled_np, seed=seed)

#     # Step 2: normalize noise to [0,1] and scale
#     I_noise_norm = I_noise / np.max(I_noise)
#     I_noise_scaled = I_noise_norm * noise_magnitude # Here we decide how noise whould be weighed compared to signal with param noise_magnitude

#     # Step 3: add noise and renormalize
#     I_combined = np.asarray(I_clean_norm) + I_noise_scaled # Normalized + weighted noise
#     I_normalized = I_combined / np.max(I_combined) # Renormalize

#     return jnp.array(I_combined)
