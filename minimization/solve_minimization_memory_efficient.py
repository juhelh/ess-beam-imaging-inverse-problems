import jax
jax.config.update("jax_platform_name", "gpu")

import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import grad, hessian, vmap, jit
import jax
from scipy.optimize import minimize, Bounds
# === Grid ===
from configuration.config_grid import X, Y  # Or define manually if standalone
PARAM_NAMES = ["Ax", "Ay", "sigx", "sigy", "cx", "cy", "fx", "fy"]

print("JAX is using:", jax.default_backend())

import jax
import jax.numpy as jnp
from jax import jit, grad, remat, lax, value_and_grad
from functools import partial

def triangle_wave(f, t, eps=1e-12):
    phase = (2*jnp.pi*f * t) % 1.0 # 2*jnp.pi*
    soft_abs = jnp.sqrt((phase - 0.5)**2 + eps)
    return 4.0 * soft_abs - 1.0

def raster_position(t, Ax, Ay, fx, fy, cx, cy):
    fx_Hz = fx * 1000
    fy_Hz = fy * 1000

    x_pos = Ax * triangle_wave(fx_Hz, t) + cx
    y_pos = Ay * triangle_wave(fy_Hz, t) + cy
    return x_pos, y_pos

# ---------- beam simulation ---------- #

@remat
def beam_intensity_single_t(x, y, t, k, fx=None, fy=None):
    if fx is None or fy is None:
        Ax, Ay, sigx, sigy, cx, cy, fx, fy = k
    else:
        Ax, Ay, sigx, sigy, cx, cy = k

    gx, gy = raster_position(t, Ax, Ay, fx, fy, cx, cy)
    normal_const = 1.0 / (2 * jnp.pi * sigx * sigy)
    exponent = -0.5 * (((x - gx) / sigx) ** 2 + ((y - gy) / sigy) ** 2)
    return normal_const * jnp.exp(exponent)

def beam_intensity_batch(x, y, t_vals, k, fx=None, fy=None):
    def step(I_accum, t):
        I_t = beam_intensity_single_t(x, y, t, k, fx=fx, fy=fy)
        return I_accum + I_t, None

    I0 = jnp.zeros_like(x)
    I_final, _ = lax.scan(step, I0, t_vals)
    return I_final

@jit
def simulate_image(x, y, t_vals, k, fx=None, fy=None):
    dt = t_vals[1] - t_vals[0]
    I_accum = beam_intensity_batch(x, y, t_vals, k, fx=fx, fy=fy) * dt
    return I_accum / jnp.max(I_accum)

# ---------- loss & grad, JIT once ---------- #

def _loss_only(k_partial, I_obs, X, Y, t_vals, fx=None, fy=None):
    I_sim = simulate_image(X, Y, t_vals, k_partial, fx=fx, fy=fy)
    return jnp.sum((I_sim - I_obs) ** 2) / jnp.sum(I_obs ** 2)

loss_jit = partial(jit, static_argnums=(1, 2, 3))(lambda k, I_obs, X, Y, t_vals: _loss_only(k, I_obs, X, Y, t_vals))
value_and_grad_j = partial(jit, static_argnums=(1, 2, 3))(value_and_grad(_loss_only, argnums=0))




def estimate_parameters_BFGS(
    I_obs,
    X,
    Y,
    t_vals,
    k0,
    loss_function=loss_jit,
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


def report_estimation_result(result, k_true, I_obs=None, X=None, Y=None, t_vals=None):
    """
    Print a summary report comparing estimated vs true parameters, including gradient at solution.

    Parameters
    ----------
    result : scipy.optimize.OptimizeResult
        Result object from the optimizer (e.g. from minimize)
    k_true : array-like
        True beam parameters (same length as result.x)
    I_obs, X, Y, t_vals : optional
        If provided, compute gradient at result.x
    """
    print("\n=== Full Estimation Result ===")
    print("Success:", result.success)
    print("Final loss:", result.fun)
    print("Recovered k:", result.x)
    print("True k:     ", k_true)
    print("Error norm: ", jnp.linalg.norm(result.x - k_true))

    # Add gradient info
    print("Gradient at final estimate:", result.jac)
    print("‖Gradient‖ =", jnp.linalg.norm(result.jac))

def report_estimation_result(result, k_true, I_obs=None, X=None, Y=None, t_vals=None):
    """
    Print a summary report comparing estimated vs true parameters, including gradient at solution.

    Parameters
    ----------
    result : scipy.optimize.OptimizeResult
        Result object from the optimizer (e.g. from minimize)
    k_true : array-like
        True beam parameters (can be longer than result.x if partial optimization)
    I_obs, X, Y, t_vals : optional
        If provided, compute gradient at result.x
    """
    print("\n=== Full Estimation Result ===")
    print("Success:", result.success)
    print("Final loss:", result.fun)

    # Determine how many parameters were optimized
    n_opt = len(result.x)
    k_true_trimmed = jnp.array(k_true[-n_opt:])

    print("Recovered k:", result.x)
    print("True k:     ", k_true_trimmed)
    print("Error norm: ", jnp.linalg.norm(result.x - k_true_trimmed))

    # Gradient info
    print("Gradient at final estimate:", result.jac)
    print("‖Gradient‖ =", jnp.linalg.norm(result.jac))

def visualize_estimation_result(I_obs, X, Y, t_vals, k_est, fx=None, fy=None):
    """
    Visualize observed vs simulated image using estimated parameters,
    and show their pixelwise difference.

    Supports both full 8-parameter or 6-parameter (with fixed fx, fy) versions.
    """
    I_est = simulate_image(X, Y, t_vals, k_est, fx=fx, fy=fy)
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

    plt.suptitle("Parameter Estimation Result", fontsize=14)
    plt.tight_layout()
    plt.show(block=True)

def sample_initial_guesses(k_true, param_ranges, n_samples=100, scale=1.0, seed=0):
    """
    Sample random initial guesses around `k_true` within scaled bounds.

    Parameters
    ----------
    k_true : array-like
        The true parameter values (shape [D])
    param_ranges : array, shape (D, 2)
        Each row is [lower_bound, upper_bound] for that parameter
    scale : float
        How much to expand (or shrink) the sampling bounds around k_true
    n_samples : int
        Number of samples to generate
    seed : int
        RNG seed

    Returns
    -------
    samples : array, shape (n_samples, D)
        Random initial parameter vectors
    """

    rng = np.random.default_rng(seed)

    lower, upper = param_ranges[:, 0], param_ranges[:, 1]
    center = k_true
    half_width = (upper - lower) / 2.0
    new_lower = center - scale * half_width
    new_upper = center + scale * half_width

    samples = rng.uniform(low=new_lower, high=new_upper, size=(n_samples, len(k_true)))
    return jnp.array(samples)

def run_many_initializations(I_obs, X, Y, t_vals, k_true, k0_samples, estimate_fn, loss_threshold=0.01):
    results = []

    for i, k0 in enumerate(k0_samples):
        result = estimate_fn(I_obs, X, Y, t_vals, k0, verbose = False)
        error_norm = jnp.linalg.norm(result.x - k_true)
        final_loss = float(result.fun)
        is_success = final_loss < loss_threshold  

        results.append({
            'success': is_success,
            'final_loss': final_loss,
            'error_norm': float(error_norm),
            'n_iter': result.nit,
            'k0': k0,
            'k_est': result.x,
        })

        print(f"[{i+1}/{len(k0_samples)}] Success={is_success}, Final Loss={final_loss:.2e}, Error Norm={error_norm:.4e}")

    return results

def plot_convergence_distributions(results, loss_threshold=0.01):
    error_norms = [r['error_norm'] for r in results]
    final_losses = [r['final_loss'] for r in results]
    successes = [r['success'] for r in results]

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(error_norms, bins=30, color='skyblue', edgecolor='black')
    plt.title("Distribution of error norms")
    plt.xlabel("‖k_est - k_true‖")
    plt.ylabel("Count")

    plt.subplot(1, 2, 2)
    plt.hist(final_losses, bins=30, color='salmon', edgecolor='black')
    plt.axvline(x=loss_threshold, color='black', linestyle='--', label=f"Loss threshold = {loss_threshold}")
    plt.title("Distribution of final losses")
    plt.xlabel("Relative loss")
    plt.ylabel("Count")
    plt.legend()

    plt.suptitle(f"Sensitivity to Initialization ({len(results)} runs)")
    plt.tight_layout()
    plt.show(block=True)

    success_rate = sum(successes) / len(successes)
    print(f"\nSuccess rate (loss < {loss_threshold}): {success_rate:.2%}")


def main():
    # ==================================================
    # === TEST: Investigate gradient ===================
    # ==================================================
    pass

    # ==================================================
    # === TEST: Estimate all 8 parameters with L-BFGS ==
    # ==================================================

    fx = 39_550.0
    fy = 29_050.0   
    freq_scaling = 1000

    # === True parameters ===
    k_true = jnp.array([
        60.0,       # Ax
        20.0,       # Ay
        13.5,       # sigx
        5.05,       # sigy
        0.0,        # cx
        0.0,        # cy
        fx / freq_scaling,  # NOTE: scaled to have similar size to other parameters
        fy / freq_scaling 
    ])

    # === Short pulse time vector ===
    # NOTE: Try with full pulse here to show that it is NOT possible to estimate frequencies accurately!
    pulse_duration = 50e-6  # 50 microseconds: short pulse!
    sampling_rate = 1_000_000  # 1 MHz
    t_vals = jnp.linspace(0, pulse_duration, int(pulse_duration * sampling_rate))

    # === Simulate observed image ===
    I_obs = simulate_image(X, Y, t_vals, k_true)

    # === Initial guess: perturb true parameters ===
    perturbation = jnp.array([5.0, -3.0, 1.0, -1.0, 2.0, -1.5, 1000.0 / freq_scaling, -3000.0 / freq_scaling])
    k0 = k_true + perturbation

    # === Run estimation ===
    result = estimate_parameters_BFGS(I_obs, X, Y, t_vals, k0)
    report_estimation_result(result, k_true)
    k_est = result.x
    visualize_estimation_result(I_obs, X, Y, t_vals, k_est)

    # ==================================================
    # === TEST: Robustness compared to MCMC ============
    # ==================================================

    dim = 8
    prior_mean = np.array([60.0, 20.0, 13.5, 5.05, 0.0, 0.0, 39.55, 29.05])
    prior_std  = np.array([10.0, 5.0, 2.0, 1.0, 2.0, 2.0, 5.0, 5.0])

    # === Generate synthetic parameters with perturbation ===
    np.random.seed(42)  # For reproducibility
    perturbation = np.random.randn(dim) * prior_std
    pert_scale = 2
    k_true = jnp.array(prior_mean + perturbation*pert_scale)

    I_obs = simulate_image(X, Y, t_vals, k_true)

    # === Run estimation ===
    result = estimate_parameters_BFGS(I_obs, X, Y, t_vals, k0)
    
    print("=== Perturbation Summary ===")
    for i, name in enumerate(["Ax", "Ay", "sigx", "sigy", "cx", "cy", "fx", "fy"]):
        print(f"{name:>4}: prior = {prior_mean[i]:>6.2f}, perturbation = {perturbation[i]:>6.3f},  true = {k_true[i]:.3f}")
    
    report_estimation_result(result, k_true)
    k_est = result.x
    visualize_estimation_result(I_obs, X, Y, t_vals, k_est)

    # Special test: one single paramter off
    k_true = jnp.array(prior_mean + np.array([0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -20]))
    I_obs = simulate_image(X, Y, t_vals, k_true)

    result = estimate_parameters_BFGS(I_obs, X, Y, t_vals, k0)
    
    print("=== Perturbation Summary ===")
    for i, name in enumerate(["Ax", "Ay", "sigx", "sigy", "cx", "cy", "fx", "fy"]):
        print(f"{name:>4}: prior = {prior_mean[i]:>6.2f}, perturbation = {perturbation[i]:>6.3f},  true = {k_true[i]:.3f}")
    
    report_estimation_result(result, k_true)
    k_est = result.x
    visualize_estimation_result(I_obs, X, Y, t_vals, k_est)


    # ==================================================
    # === TEST: Many different initial points ==========
    # ==================================================

    # NOTE: Reusing variables from above like k_true 

    param_ranges = jnp.array([
        [50.0, 70.0],     # Ax
        [10.0, 30.0],     # Ay
        [11.0, 16.0],     # sigx
        [4.0,  6.0],      # sigy
        [-2.0, 2.0],      # cx
        [-2.0, 2.0],      # cy
        [34.5, 44.5],     # fx
        [24.5, 33.5],     # fy
    ])

    # --- Sample initial guesses with scale ---
    k0_samples = sample_initial_guesses(
        k_true=k_true,
        param_ranges=param_ranges,  # shape (D, 2)
        n_samples=100,
        scale=1.0,
    )

    # --- Run many fits ---
    results = run_many_initializations(
        I_obs, X, Y, t_vals,
        k_true=k_true,
        k0_samples=k0_samples,
        estimate_fn=estimate_parameters_BFGS
    )

    plot_convergence_distributions(results)

if __name__ == '__main__':
    main()