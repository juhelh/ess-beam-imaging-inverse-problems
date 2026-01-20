import numpy as np
import jax.numpy as jnp
from tqdm import tqdm
from typing import Tuple, List
import os

from configuration.configuration import make_configuration
from minimization.solve_minimization_10D import simulate_image, raster_position
from minimization.visualize_simulations import visualize_image
from noise_simulator.noise import add_poisson

import matplotlib.pyplot as plt

def sample_gaussian_with_rejection(rng, mean, std, lower, upper, max_trials=5000):
    """
    Draw once from N(mean, std^2) with rejection if outside [lower, upper].
    """
    for _ in range(max_trials):
        x = rng.normal(mean, std)
        if np.all(x >= lower) and np.all(x <= upper):
            return x
    raise RuntimeError("Rejection sampling failed: too many out-of-bound samples.")


# ---------------------------------------------------------------
# Plotting utilities
# ---------------------------------------------------------------

def plot_simulation_diagnostics(I: jnp.ndarray, k: np.ndarray, T: float):
    """
    Display image with overlaid parameters for visual inspection.
    (10 parameters version)
    """
    names = ["Ax","Ay","sigx","sigy","cx","cy","fx","fy","phix","phiy"]

    plt.figure(figsize=(5, 4))
    plt.imshow(I, cmap='inferno')
    plt.colorbar()

    label = f"T = {T*1e3:.2f} ms\n"
    label += "\n".join([f"{name} = {val:.2f}" for name, val in zip(names, k)])

    plt.title(label, fontsize=8)
    plt.axis('off')
    plt.tight_layout()
    plt.show(block=True)


def plot_raster_trajectory(k, T, dt, cmap='plasma'):
    """
    Plot the raster path (gx(t), gy(t)) over time,
    now using full 10 parameters including phix, phiy.
    """
    Ax, Ay, _, _, cx, cy, fx, fy, phix, phiy = k
    t_vals = jnp.arange(0, T, dt)
    gx, gy = raster_position(t_vals, Ax, Ay, fx, fy, cx, cy, phix, phiy)
    gx, gy, t_vals = np.array(gx), np.array(gy), np.array(t_vals)

    plt.figure(figsize=(5, 4))
    plt.plot(gx, gy, '-', color='gray', lw=0.5, alpha=0.7)

    scatter = plt.scatter(gx, gy, c=t_vals, s=1, cmap=cmap)
    cbar = plt.colorbar(scatter)
    cbar.set_label("time (s)", rotation=270, labelpad=10)

    plt.title(f"Raster trajectory (T = {T*1e3:.2f} ms)")
    plt.xlabel("x (beam center)")
    plt.ylabel("y (beam center)")
    plt.axis("equal")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.show(block=True)


# ---------------------------------------------------------------
# Test set generation (irrational only)
# ---------------------------------------------------------------

def make_test_set(
    X: jnp.ndarray,
    Y: jnp.ndarray,
    dt: float,
    param_bounds: Tuple[np.ndarray, np.ndarray],
    t_bounds: Tuple[float, float],
    num_samples: int,
    rng: np.random.Generator = np.random.default_rng(1),
    distribution: str = "uniform",
    k_nominal: np.ndarray = None,
    std_vec: np.ndarray = None
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, List[bool]]:
    
    
    lower, upper = param_bounds

    if distribution == "uniform":
        ks = rng.uniform(lower, upper, size=(num_samples, 10))

    elif distribution == "gaussian":
        if k_nominal is None or std_vec is None:
            raise ValueError("Gaussian sampling requires k_nominal and std_vec.")

        ks = np.zeros((num_samples, 10))
        for i in range(num_samples):
            ks[i] = sample_gaussian_with_rejection(
                rng, k_nominal, std_vec, lower, upper
            )

    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    # Integration times
    t_min, t_max = t_bounds
    T_all = np.geomspace(t_min, t_max, num=num_samples) * 1e-3  # seconds

    rational_flags = [False] * num_samples  # always irrational
    scale = 20_000.0

    images = []
    for i in tqdm(range(num_samples), desc="Simulating test set"):
        T = T_all[i]
        k = ks[i]

        t_vals = jnp.arange(0, T, dt)

        I_clean = simulate_image(jnp.array(X), jnp.array(Y), jnp.array(t_vals), jnp.array(k))
        I_obs = add_poisson(I_clean, scale=scale)
        I_obs /= scale

        visualize_image(I_obs, X, Y)

        images.append(I_obs)

        # Diagnostics
        plot_raster_trajectory(k, T, dt)
        # plot_simulation_diagnostics(I_obs, k, T)

    return jnp.array(ks), jnp.stack(images), jnp.array(T_all), rational_flags


# ---------------------------------------------------------------
# Save function
# ---------------------------------------------------------------

def save_test_set(k_all, I_all, T_all, rational_flags,
                  folder="test_all_algorithms", filename="test_set_10D_gaussian.npz"):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)

    np.savez(
        path,
        k_all=np.array(k_all),
        I_all=np.array(I_all),
        T_all=np.array(T_all),
        rational_flags=np.array(rational_flags),
    )
    print(f"Saved test set to: {path}")


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main():
    X, Y, _, dt, _ = make_configuration(sampling_rate=1_000_000)

    k_nominal = np.array([
        60.0,   # Ax
        20.0,   # Ay
        13.5,   # sigx
        5.05,   # sigy
        0.0,    # cx
        0.0,    # cy
        39.55,  # fx
        29.05,  # fy
        0.0,    # phix (no nominal → still centered at zero)
        0.0     # phiy
    ], dtype=float)

    frac = 0.20
    # Std = 10% except fx, fy = 1%
    std_vec = np.array([
        frac * 60.0,
        frac * 20.0,
        frac * 13.5,
        frac * 5.05,
        frac * 1.0,      # cx (10% of 1 just to avoid zero, can be changed)
        frac * 1.0,      # cy
        0.05 * 39.55,
        0.05 * 29.05,
        frac * 1.0,      # phix: rough scale
        frac * 1.0       # phiy
    ])

    #                      Ax  Ay  sx  sy  cx  cy   fx   fy   phix  phiy
    lower = np.array([0,   0,  2,  2, -20, -20,  20,  20, -np.pi, -np.pi], dtype=float)
    upper = np.array([65, 25, 20, 20,  20,  20,  50,  50,  np.pi,  np.pi], dtype=float)

    t_bounds = (0.05, 3.0)  # ms
    num_samples = 10

    k_all, I_all, T_all, rational_flags = make_test_set(
        X, Y, dt,
        param_bounds=(lower, upper),
        t_bounds=t_bounds,
        num_samples=num_samples,
        distribution="gaussian",
        k_nominal=k_nominal,
        std_vec=std_vec
    )

    print("\n=== Summary (10D) ===")
    for i in range(num_samples):
        print(f"Sample {i:02d}: fx={k_all[i,6]:.2f}, fy={k_all[i,7]:.2f}, "
              f"phix={k_all[i,8]:.2f}, phiy={k_all[i,9]:.2f} (irrational)")

    save_test_set(k_all, I_all, T_all, rational_flags)


if __name__ == "__main__":
    main()