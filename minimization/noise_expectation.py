import jax
import jax.numpy as jnp
from jax import hessian
from functools import partial
from minimization.solve_minimization_8D import simulate_image
from minimization.visualize_simulations import visualize_image
from configuration.configuration import make_configuration
from statistical_methods.MCMC_tests_speedup import bin_image_and_grid

def normalize_image(I: jnp.ndarray, total_count: float) -> jnp.ndarray:
    """Scale image so that total photon count matches target value."""
    return I / jnp.sum(I) * total_count

def add_poisson_noise(key: jax.random.PRNGKey, I: jnp.ndarray) -> jnp.ndarray:
    """Apply elementwise Poisson noise."""
    return jax.random.poisson(key, lam=I)

def compute_normalized_loss(I_est: jnp.ndarray, I_obs: jnp.ndarray) -> float:
    """Compute normalized squared loss."""
    x = jnp.ravel(I_est)
    y = jnp.ravel(I_obs)
    return jnp.sum((x - y)**2) / jnp.sum(y**2)

def expected_loss_via_hessian(x_flat: jnp.ndarray) -> float:
    """Compute 0.5 * Tr(H @ diag(x)) using the second-order approximation."""

    def loss_fn(y_flat):
        return jnp.sum((x_flat - y_flat)**2) / jnp.sum(y_flat**2)

    H = hessian(loss_fn)(x_flat)
    return 0.5 * jnp.trace(H @ jnp.diag(x_flat))

def compare_losses(
    key: jax.random.PRNGKey,
    k_true: jnp.ndarray,
    total_count: float,
    X, Y, t_vals
) -> dict:
    """Generate image, add noise, compute both losses, and compare."""

    # Simulate clean image
    I_clean = simulate_image(X, Y, t_vals, k_true)
    I_scaled = normalize_image(I_clean, total_count)

    # Add Poisson noise
    key, subkey = jax.random.split(key)
    I_noisy = add_poisson_noise(subkey, I_scaled)

    # Flattened clean image
    x_flat = jnp.ravel(I_scaled)

    # Compute both losses
    empirical = compute_normalized_loss(I_scaled, I_noisy)
    expected = expected_loss_via_hessian(x_flat)
    relative_error = jnp.abs(empirical - expected) / expected

    return {
        "empirical_loss": float(empirical),
        "expected_loss": float(expected),
        "relative_error": float(relative_error),
    }

def main():
    X, Y, t_vals, DT, k_true = make_configuration(pulse_duration_ms=3)

    k_true = jnp.array([
        60.0, 20.0, 13.5, 5.05,
         0.0,  0.0, 39.55, 29.05
    ], dtype=jnp.float32)
    
    I_obs = simulate_image(X, Y, t_vals, k_true)
    # Binning
    BIN_FACTOR = 8  # or 4, etc.
    I_obs, X, Y = bin_image_and_grid(I_obs, X, Y, factor=BIN_FACTOR)

    # Normalize
    I_obs = I_obs / jnp.max(I_obs)
    visualize_image(I_obs, X, Y)
    
    key = jax.random.PRNGKey(42)
    total_photon_count = 20_000

    result = compare_losses(key, k_true, total_photon_count, X, Y, t_vals)

    print(f"Empirical loss:        {result['empirical_loss']:.6f}")
    print(f"Expected (approx) loss:{result['expected_loss']:.6f}")
    print(f"Relative error:        {100 * result['relative_error']:.2f}%")

if __name__ == "__main__":
    main()