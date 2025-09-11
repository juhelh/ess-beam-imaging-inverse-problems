import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import grad, hessian, vmap, jit
from scipy.optimize import minimize, Bounds

def triangle_wave(f, t, N=5):
    # Fourier approximation: sum odd harmonics
    omega = 2 * jnp.pi * f
    terms = [(jnp.sin((2 * n - 1) * omega * t) / (2 * n - 1)**2) for n in range(1, N + 1)]
    return (8 / jnp.pi**2) * jnp.sum(jnp.stack(terms), axis=0)


def raster_position(t, Ax, Ay, fx, fy, cx, cy):
    fx_Hz = fx*1000
    fy_Hz = fy*1000

    x_pos = Ax * triangle_wave(fx_Hz, t) + cx
    y_pos = Ay * triangle_wave(fy_Hz, t) + cy
    return x_pos, y_pos


def beam_intensity_single_t(x, y, t, k):
    Ax, Ay, sigx, sigy, cx, cy, fx, fy = k
    gx, gy = raster_position(t, Ax, Ay, fx, fy, cx, cy)

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
    return I_accumulated


@jit
def loss_function(k, I_obs, X, Y, t_vals):
    I_sim = simulate_image(X, Y, t_vals, k)
    num = jnp.sum((I_sim - I_obs) ** 2)
    denom = jnp.sum(I_obs ** 2)
    loss = num / denom
    return loss


def loss_gradient(k, I_obs, X, Y, t_vals):
    return jit(grad(loss_function))(k, I_obs, X, Y, t_vals)


def loss_hessian(k, I_obs, X, Y, t_vals):
    return hessian(loss_function)(k, I_obs, X, Y, t_vals)


def estimate_parameters_BFGS(I_obs, X, Y, t_vals, k0):
    print("=== Starting parameter estimation ===")

    def loss(k):
        return loss_function(k, I_obs, X, Y, t_vals)

    loss_grad = grad(loss)
    iteration = {'count': 0}

    def callback(k):
        iteration['count'] += 1
        L = loss(k)
        g = loss_grad(k)
        g_norm = jnp.linalg.norm(g)

        print(f"--- Iteration {iteration['count']} ---")
        print(f"Loss:         {L:.6e}")
        print(f"‖grad‖:       {g_norm:.6e}")
        print(f"Current k:    {np.array2string(k, precision=3)}")
        print(f"Gradient:     {np.array2string(g, precision=3)}\n")

    result = minimize(
        fun=loss,
        x0=jnp.array(k0),
        jac=loss_grad,
        method='L-BFGS-B',
        callback=callback,
        options={'disp': True, 'maxiter': 100}
    )

    return result


def estimate_parameters_trustconstr(I_obs, X, Y, t_vals, k0):
    """
    Estimate beam parameters using trust-region constrained minimization.

    Parameters
    ----------
    I_obs : array (H, W)
        Observed accumulated beam image
    X, Y : 2D arrays
        Spatial meshgrid
    t_vals : 1D array
        Time samples used in the forward simulation
    k0 : array-like, shape (8,)
        Initial guess for parameters [Ax, Ay, sigx, sigy, cx, cy, fx, fy]

    Returns
    -------
    result : scipy.optimize.OptimizeResult
        Result object from optimizer
    """

    # Define bounds (same units as your model, fx and fy in Hz)
    bounds = Bounds(
        lb=jnp.array([0.0, 0.0, 1.0, 1.0, -10.0, -10.0, 30, 20]),
        ub=jnp.array([100.0, 100.0, 50.0, 50.0, 10.0, 10.0, 50, 40])
    )

    # Loss function and gradient (fx and fy are inside k now)
    def loss(k):
        return loss_function(k, I_obs, X, Y, t_vals)

    grad_fn = grad(loss)

    # Optional callback to monitor progress
    iteration = {'count': 0}

    def callback(k, state=None):
        iteration['count'] += 1
        L = loss(k)
        g = grad_fn(k)
        g_norm = jnp.linalg.norm(g)

        print(f"--- Iteration {iteration['count']} ---")
        print(f"Loss:         {L:.6e}")
        print(f"‖grad‖:       {g_norm:.6e}")
        print(f"Current k:    {np.array2string(k, precision=3)}")
        print(f"Gradient:     {np.array2string(g, precision=3)}\n")

    # Run optimization using trust-constr method
    result = minimize(
        fun=loss,
        x0=jnp.array(k0),
        jac=grad_fn,
        method='trust-constr',
        bounds=bounds,
        callback=callback,
        options={
            'verbose': 3,
            'gtol': 1e-8,
            'maxiter': 50,
        }
    )

    return result

def report_estimation_result(result, k_true):
    """
    Print a summary report comparing estimated vs true parameters.

    Parameters
    ----------
    result : scipy.optimize.OptimizeResult
        Result object from the optimizer (e.g. from minimize)
    k_true : array-like
        True beam parameters (same length as result.x)
    """
    print("\n=== Full Estimation Result ===")
    print("Success:", result.success)
    print("Final loss:", result.fun)
    print("Recovered k:", result.x)
    print("True k:     ", k_true)
    print("Error norm: ", jnp.linalg.norm(result.x - k_true))

def visualize_estimation_result(I_obs, X, Y, t_vals, k_est):
    """
    Visualize observed vs simulated image using estimated parameters.

    Parameters
    ----------
    I_obs : 2D array
        Observed accumulated image
    X, Y : 2D arrays
        Spatial grid (same shape as I_obs)
    t_vals : 1D array
        Time vector used in accumulation
    k_est : array-like
        Estimated beam parameters (including fx, fy)
    """
    I_est = simulate_image(X, Y, t_vals, k_est)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(I_obs, origin='lower', cmap='inferno')
    plt.title("Observed image")
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.imshow(I_est, origin='lower', cmap='inferno')
    plt.title("Simulated from estimated k")
    plt.colorbar()

    plt.suptitle("Parameter Estimation Result (fx/fy included)")
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
        result = estimate_fn(I_obs, X, Y, t_vals, k0)
        error_norm = jnp.linalg.norm(result.x - k_true)
        final_loss = float(result.fun)
        is_success = final_loss < loss_threshold  # ✅ New success criterion

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

    # === Grid ===
    from config_grid import X, Y  # Or define manually if standalone

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