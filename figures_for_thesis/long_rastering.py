import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax.scipy.special import erf
from configuration.configuration import make_configuration
from minimization.solve_minimization_8D import simulate_image


def compute_I_infinity(X, Y, k):
    Ax, Ay, sigx, sigy, cx, cy, fx, fy = k
    term_x = erf((X - cx + Ax) / (jnp.sqrt(2) * sigx)) - erf((X - cx - Ax) / (jnp.sqrt(2) * sigx))
    term_y = erf((Y - cy + Ay) / (jnp.sqrt(2) * sigy)) - erf((Y - cy - Ay) / (jnp.sqrt(2) * sigy))
    return (1.0 / (16 * Ax * Ay)) * term_x * term_y


def show_convergence_to_limit():
    # Frequencies with irrational ratio
    fx, fy = 1.0, jnp.sqrt(2)
    pulse_lengths = [1, 10.0, 100.0]  # in ms

    # Fixed configuration (grid stays the same)
    X, Y, _, _, k = make_configuration(pulse_duration_ms=1.0)
    I_inf = compute_I_infinity(X, Y, k)

    fig, axs = plt.subplots(2, 3, figsize=(14, 5))
    for i, T in enumerate(pulse_lengths):
        # Setup with same fx, fy but different time
        X, Y, t_vals, _, k_tmp = make_configuration(pulse_duration_ms=T)
        I_sim = simulate_image(X, Y, t_vals, k_tmp)
        I_sim = I_sim / (T * 1e-3)  # convert ms → s and take time average

        # Plot simulated image
        im0 = axs[0, i].imshow(I_sim, cmap="inferno", origin="lower")
        axs[0, i].set_title(f"Simulated image ({T:.0f} ms)")
        plt.colorbar(im0, ax=axs[0, i], fraction=0.05, pad=0.04)

        # Relative difference (normalized by local magnitude of I_inf)
        relative_diff = (I_sim - I_inf) / (I_inf + 1e-12)  # avoid division by zero
        im1 = axs[1, i].imshow(relative_diff, cmap="seismic", origin="lower", vmin=-0.01, vmax=0.01)
        axs[1, i].set_title("Relative error $(I - I_\\infty)/I_\\infty$")
        plt.colorbar(im1, ax=axs[1, i], fraction=0.05, pad=0.04)

        # Quantitative L2 relative error
        error_norm = jnp.linalg.norm(I_sim - I_inf) / jnp.linalg.norm(I_inf)
        print(f"T = {T:.0f} ms → relative L2 error = {error_norm:.3e}")

    plt.suptitle("Convergence toward the long-time limit: simulation vs. analytical", fontsize=14)
    plt.tight_layout()
    plt.show(block=True)


# Run the visualization
show_convergence_to_limit()