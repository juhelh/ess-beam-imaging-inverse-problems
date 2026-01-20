import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import grad, hessian, vmap, jit
from scipy.optimize import minimize, Bounds
from configuration.configuration import make_configuration
import os
from noise_simulator.noise import add_poisson
from minimization.visualize_simulations import visualize_image


def triangle_wave(f, t, phi=0.0):
    phase = (f * t - 0.25 + phi / (2 * jnp.pi)) % 1.0
    return 4.0 * jnp.abs(phase - 0.5) - 1.0

# def triangle_wave(f, t, phi=0.0):
#         return (2.0 / jnp.pi) * jnp.arcsin(jnp.sin(2.0 * jnp.pi * f * t + phi))


def triangle_wave_2(f, t, phi=0.0, eps=1e-12):
    phase = (f * t - 0.25 + phi / (2 * jnp.pi)) % 1.0
    # Soft absolute value: sqrt(x^2 + eps)
    soft_abs = jnp.sqrt((phase - 0.5)**2 + eps)
    return 4.0 * soft_abs - 1.0


def raster_position(t, Ax, Ay, fx, fy, cx, cy, phix, phiy):
    fx_Hz = fx * 1000
    fy_Hz = fy * 1000

    x_pos = Ax * triangle_wave(fx_Hz, t, phix) + cx
    y_pos = Ay * triangle_wave(fy_Hz, t, phiy) + cy
    return x_pos, y_pos


def beam_intensity_single_t(x, y, t, k):
    Ax, Ay, sigx, sigy, cx, cy, fx, fy, phix, phiy = k
    gx, gy = raster_position(t, Ax, Ay, fx, fy, cx, cy, phix, phiy)

    normalizing_const = 1.0 / (2 * jnp.pi * sigx * sigy)
    exponent = -0.5 * (((x - gx) / sigx) ** 2 + ((y - gy) / sigy) ** 2)
    I = normalizing_const * jnp.exp(exponent)
    return I


def beam_intensity_batch(x, y, t_vals, k):
    batched_function = vmap(beam_intensity_single_t, in_axes=(None, None, 0, None))
    return batched_function(x, y, t_vals, k)


@jit
def simulate_image(x, y, t_vals, k):
    dt = t_vals[1] - t_vals[0]
    I_stack = beam_intensity_batch(x, y, t_vals, k)
    I_accumulated = jnp.sum(I_stack, axis=0) * dt
    I_norm = I_accumulated / jnp.max(I_accumulated)

    return I_norm


@jit
def loss_function(k, I_obs, X, Y, t_vals):
    I_sim = simulate_image(X, Y, t_vals, k)
    num = jnp.sum((I_sim - I_obs) ** 2)
    denom = jnp.sum(I_obs ** 2)
    return num / denom


def loss_gradient(k, I_obs, X, Y, t_vals):
    return jit(grad(loss_function))(k, I_obs, X, Y, t_vals)


def loss_hessian(k, I_obs, X, Y, t_vals):
    return hessian(loss_function)(k, I_obs, X, Y, t_vals)


def estimate_parameters_BFGS(
    I_obs,
    X,
    Y,
    t_vals,
    k0,
    loss_function=loss_function,
    maxiter = 50,
    verbose=True,
    bounds=None  # NEW: Optional bounds argument
):
    """
    Estimate beam parameters using L-BFGS-B optimization.

    Parameters
    ----------
    I_obs : array
        Observed intensity image.
    X, Y : arrays
        Coordinate grids.
    t_vals : array
        Time points for raster simulation.
    k0 : array
        Initial guess for parameters.
    loss_function : callable, optional
        Function computing the loss. Default is global `loss_function`.
    verbose : bool, optional
        If True, print iteration details. Default is True.
    bounds : tuple of arrays, optional
        Tuple (lower, upper) with bounds for parameters. Default is None.

    Returns
    -------
    result : OptimizeResult
        Result object from scipy.optimize.minimize.
    """
    if verbose:
        print("=== Starting parameter estimation ===")

    def loss(k):
        return loss_function(k, I_obs, X, Y, t_vals)

    loss_grad = grad(loss)
    iteration = {'count': 0}

    def callback(k):
        if not verbose:
            return
        iteration['count'] += 1
        L = loss(k)
        g = loss_grad(k)
        g_norm = jnp.linalg.norm(g)

        print(f"--- Iteration {iteration['count']} ---")
        print(f"Loss:         {L:.6e}")
        print(f"‖grad‖:       {g_norm:.6e}")
        print(f"Current k:    {np.array2string(k, precision=3)}")
        print(f"Gradient:     {np.array2string(g, precision=3)}\n")

    # Prepare bounds object if provided
    scipy_bounds = Bounds(*bounds) if bounds is not None else None

    result = minimize(
        fun=loss,
        x0=jnp.array(k0),
        jac=loss_grad,
        method='L-BFGS-B',
        bounds=scipy_bounds,  # <- use bounds here
        callback=callback,
        options={'disp': verbose, 'maxiter': maxiter}
    )

    return result



def report_estimation_result(result, k_true):
    print("\n=== Full Estimation Result ===")
    print("Success:", result.success)
    print("Final loss:", result.fun)
    print("Recovered k:", result.x)
    print("True k:     ", k_true)
    print("Error norm: ", jnp.linalg.norm(result.x - k_true))


def visualize_estimation_result(I_obs, X, Y, t_vals, k_est):
    """
    Visualize observed vs simulated image using estimated parameters,
    and show their pixelwise difference.
    """
    I_est = simulate_image(X, Y, t_vals, k_est)
    I_diff = I_est - I_obs

    # Compute some difference stats
    l2_norm = jnp.linalg.norm(I_diff)
    rel_l2 = l2_norm / jnp.linalg.norm(I_obs)
    max_abs = jnp.max(jnp.abs(I_diff))

    print("\n=== Image Comparison Metrics ===")
    print(f"L2 norm         = {l2_norm:.4e}")
    print(f"Relative L2 err = {rel_l2:.4e}")
    print(f"Max abs error   = {max_abs:.4e}")

    # === Plot: observed, estimated, difference ===
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    im0 = axs[0].imshow(I_obs, cmap='inferno', origin='lower')
    axs[0].set_title("Observed image")
    plt.colorbar(im0, ax=axs[0], shrink=0.8)

    im1 = axs[1].imshow(I_est, cmap='inferno', origin='lower')
    axs[1].set_title("Simulated from estimated k")
    plt.colorbar(im1, ax=axs[1], shrink=0.8)

    im2 = axs[2].imshow(I_diff, cmap='seismic', origin='lower')
    axs[2].set_title("Difference image (I_est - I_obs)")
    plt.colorbar(im2, ax=axs[2], shrink=0.8)

    plt.suptitle("Parameter Estimation Result (fx/fy included)", fontsize=14)
    plt.tight_layout()
    plt.show(block=True)

# ---------------------------------------------------------
# Gaussian-with-rejection sampler
# ---------------------------------------------------------
def sample_gaussian_with_rejection(rng, mean, std, lower, upper, max_trials=5000):
    """
    Draw a single 10D parameter vector from N(mean, std^2)
    with rejection if outside bounds.
    """
    for _ in range(max_trials):
        x = rng.normal(mean, std)
        if np.all(x >= lower) and np.all(x <= upper):
            return x
    raise RuntimeError("Rejection sampling failed: no valid sample.")


# ---------------------------------------------------------
# Main: generate 3 images
# ---------------------------------------------------------
def main():
    # -----------------------------------------------------
    # 1. Setup grid, time discretization
    # -----------------------------------------------------
    X, Y, _, dt, _ = make_configuration(sampling_rate=1_000_000)

    pulse_duration = 50e-6
    t_vals = jnp.arange(0, pulse_duration, dt)

    # -----------------------------------------------------
    # 2. Bounds
    # -----------------------------------------------------
    lower = np.array([0,   0,  2,  2, -20, -20,  20,  20, -np.pi, -np.pi])
    upper = np.array([65, 25, 20, 20,  20,  20,  50,  50,  np.pi,  np.pi])


    k_true = jnp.array([
        60.0,   # Ax (horizontal raster amplitude)
        20.0,   # Ay (vertical raster amplitude)
        13.5,   # sigx (horizontal width)
        5.05,   # sigy (vertical width)
        0.0,   # cx (horizontal offset)
        0.0,   # cy (vertical offset)
        39_55,   # fx (horizontal frequency)
        29_05    # fy (vertical frequency)
    ])


    pulse_duration = 50e-6
    sampling_rate = 1_000_000
    t_vals = jnp.linspace(0, pulse_duration, int(pulse_duration * sampling_rate))

    I_obs = simulate_image(X, Y, t_vals, k_true)

    perturbation = jnp.array([
        5.0, -3.0, 1.0, -1.0, 2.0, -1.5,
        1, -3,
        0.1, 0.1
    ])
    k0 = k_true + perturbation

    result = estimate_parameters_BFGS(I_obs, X, Y, t_vals, k0)
    report_estimation_result(result, k_true)
    visualize_estimation_result(I_obs, X, Y, t_vals, result.x)


if __name__ == '__main__':
    main()
