
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax
from jax import grad, hessian, vmap, jit, random
from scipy.optimize import minimize, Bounds
from configuration.configuration import make_configuration
import os
from noise_simulator.noise import add_poisson
from minimization.visualize_simulations import visualize_image
from jax import lax
from jaxopt import LBFGSB


def triangle_wave(f, t, phi=0.0):
    phase = (f * t - 0.25 + phi / (2 * jnp.pi)) % 1.0
    return 4.0 * jnp.abs(phase - 0.5) - 1.0


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


beam_intensity_batch = jax.vmap(beam_intensity_single_t, in_axes=(None, None, 0, None))


@jit
def simulate_image(x, y, t_vals, k, blur_sigma=1.5):
    dt = t_vals[1] - t_vals[0]
    I_stack = beam_intensity_batch(x, y, t_vals, k)
    I_acc = jnp.sum(I_stack, axis=0) * dt

    # Normalize to max = 1.0 to keep scales reasonable
    I_norm = I_acc / jnp.max(I_acc)

    # Apply optical blur (camera PSF)
    #I_blur = apply_optical_blur(I_norm, sigma=blur_sigma)

    return I_norm

def compare_images(I_sim, I_obs, title="Image comparison"):
    """
    Visual debugging tool: compare simulated vs observed images.

    Parameters
    ----------
    I_sim : jnp.ndarray or np.ndarray
        Simulated image (normalized or not).
    I_obs : jnp.ndarray or np.ndarray
        Observed image.
    """

    # Convert JAX arrays to NumPy for plotting
    I_sim_np = np.array(I_sim)
    I_obs_np = np.array(I_obs)
    diff = I_sim_np - I_obs_np

    # --- Print stats ---
    sim_norm = np.linalg.norm(I_sim_np)
    obs_norm = np.linalg.norm(I_obs_np)
    mse = np.mean((I_sim_np - I_obs_np)**2)
    corr = np.corrcoef(I_sim_np.flatten(), I_obs_np.flatten())[0, 1]

    print("\n=== Image Comparison Debugging ===")
    print(f"Sim norm: {sim_norm:.4e}")
    print(f"Obs norm: {obs_norm:.4e}")
    print(f"MSE:      {mse:.4e}")
    print(f"Corr:     {corr:.4f}")

    # --- Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    im0 = ax.imshow(I_obs_np, cmap="inferno", origin="lower")
    ax.set_title("Observed")
    plt.colorbar(im0, ax=ax)

    ax = axes[1]
    im1 = ax.imshow(I_sim_np, cmap="inferno", origin="lower")
    ax.set_title("Simulated")
    plt.colorbar(im1, ax=ax)

    ax = axes[2]
    im2 = ax.imshow(diff, cmap="seismic", origin="lower")
    ax.set_title("Difference (sim - obs)")
    plt.colorbar(im2, ax=ax)

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def loss_function(k, I_obs, X, Y, t_vals):
    I_sim = simulate_image(X, Y, t_vals, k)

    # L2-normalize both to compare shape only
    I_sim = I_sim / jnp.linalg.norm(I_sim)
    I_obs_n = I_obs / jnp.linalg.norm(I_obs)

    return jnp.sum((I_sim - I_obs_n) ** 2)

def loss_regularized(k, I_obs, X, Y, t_vals,
                     Ax_ref=70.0, Ay_ref=30.0,
                     lambda_Ax=1e-2, lambda_Ay=1e-2):
    # data term (current shape-only L2)
    L_data = loss_function(k, I_obs, X, Y, t_vals)

    # amplitude prior term
    Ax, Ay = k[0], k[1]
    L_reg = lambda_Ax * (Ax - Ax_ref) ** 2 + lambda_Ay * (Ay - Ay_ref) ** 2

    return L_data + L_reg



def visualize_estimation_result(I_obs, X, Y, t_vals, k_est, photon_max=20_000):
    """
    Visualize observed vs simulated image using estimated parameters,
    applying Poisson noise ONLY here for fair comparison.
    """

    # 1 — deterministic simulated image (same as optimizer)
    I_clean = simulate_image(X, Y, t_vals, k_est)
    I_clean = np.array(I_clean)

    # 2 — Apply Poisson noise for fair visual comparison
    key = jax.random.PRNGKey(12345)
    lam = I_clean * photon_max
    I_noisy = jax.random.poisson(key, lam=lam) / photon_max
    I_est = np.array(I_noisy)

    I_est = np.array(I_clean)

    # 3 — Convert real observed to NumPy
    I_obs = np.array(I_obs)

    # 4 — Normalize both (L2 = 1)
    I_est /= np.linalg.norm(I_est)
    I_obs /= np.linalg.norm(I_obs)

    # 5 — Compute difference
    I_diff = I_est - I_obs

    # 6 — Metrics
    l2_norm = np.linalg.norm(I_diff)
    rel_l2 = l2_norm / np.linalg.norm(I_obs)
    max_abs = np.max(np.abs(I_diff))

    print("\n=== Image Comparison Metrics ===")
    print(f"L2 norm         = {l2_norm:.4e}")
    print(f"Relative L2 err = {rel_l2:.4e}")
    print(f"Max abs error   = {max_abs:.4e}")

    # 7 — Plot
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    im0 = axs[0].imshow(I_obs, cmap='inferno', origin='lower')
    axs[0].set_title("Observed image")
    plt.colorbar(im0, ax=axs[0], shrink=0.8)

    im1 = axs[1].imshow(I_est, cmap='inferno', origin='lower')
    axs[1].set_title(f"Simulated image")
    plt.colorbar(im1, ax=axs[1], shrink=0.8)

    im2 = axs[2].imshow(I_diff, cmap='seismic', origin='lower')
    axs[2].set_title("Difference image")
    plt.colorbar(im2, ax=axs[2], shrink=0.8)

    plt.suptitle("Final Comparison", fontsize=14)
    plt.tight_layout()
    plt.show(block=True)


def estimate_parameters_BFGS(
    I_obs,
    X,
    Y,
    t_vals,
    k0,
    loss_function=loss_function,
    maxiter=50,
    verbose=True,
    bounds=None,
    fixed_params=None   # dict {idx: value}
):
    """
    L-BFGS-B with optional fixed parameters (fx, fy, etc).
    """

    if verbose:
        print("=== Starting parameter estimation ===")

    if bounds is None:
        raise ValueError("Bounds must be provided when fixing parameters.")

    # ---------------------------
    # Initial guess & bounds
    # ---------------------------
    k0 = jnp.array(k0)
    lower = np.array([b[0] for b in bounds], dtype=float)
    upper = np.array([b[1] for b in bounds], dtype=float)   

    # Fix parameters by collapsing bounds and forcing k0
    if fixed_params is not None:
        for idx, val in fixed_params.items():
            lower[idx] = val
            upper[idx] = val
            k0 = k0.at[idx].set(val)

        if verbose:
            print(f"Fixed parameters: {fixed_params}")

    scipy_bounds = Bounds(lower, upper)

    # ---------------------------
    # Loss and masked gradient
    # ---------------------------
    def loss(k):
        return loss_function(k, I_obs, X, Y, t_vals)

    full_grad = jit(grad(loss))

    def masked_grad(k):
        k = jnp.array(k)
        g = full_grad(k)
        if fixed_params is not None:
            for idx in fixed_params.keys():
                g = g.at[idx].set(0.0)
        return g

    # ---------------------------
    # Callback printing
    # ---------------------------
    iteration = {"count": 0}

    def callback(k_np):
        if not verbose:
            return

        iteration["count"] += 1
        k = jnp.array(k_np)

        L_val = float(loss(k))
        g_np = np.array(masked_grad(k))
        g_norm = np.linalg.norm(g_np)

        print(f"--- Iteration {iteration['count']} ---")
        print(f"Loss:         {L_val:.6e}")
        print(f"‖grad‖:       {g_norm:.6e}")
        print(f"Current k:    {np.array2string(k_np, precision=3)}\n")

        # Optional: debug image comparison
        # I_sim = simulate_image(X, Y, t_vals, k)
        # compare_images(np.array(I_sim), np.array(I_obs),
        #                title=f"Iteration {iteration['count']}")

    # ---------------------------
    # SciPy optimizer
    # ---------------------------
    result = minimize(
        fun=lambda k: float(loss(jnp.array(k))),
        x0=np.array(k0),
        jac=lambda k: np.array(masked_grad(jnp.array(k))),
        method="L-BFGS-B",
        bounds=scipy_bounds
        #callback=callback,
        #options={"disp": verbose, "maxiter": maxiter},
    )

    return result



def estimate_parameters_BFGS_jaxopt(
    I_obs,
    X,
    Y,
    t_vals,
    k0,
    loss_function,
    maxiter=50,
    verbose=True,
    bounds=None,
    fixed_params=None,   # dict {idx: value}
):
    """
    JAXOPT L-BFGS-B with optional fixed parameters.
    Fully JAX-native, GPU-compatible.
    """

    if bounds is None:
        raise ValueError("Bounds must be provided.")

    if verbose:
        print("=== Starting parameter estimation (jaxopt) ===")

    # ---------------------------
    # Initial guess & bounds
    # ---------------------------
    k0 = jnp.asarray(k0)

    lower = jnp.array([b[0] for b in bounds], dtype=jnp.float32)
    upper = jnp.array([b[1] for b in bounds], dtype=jnp.float32)

    # Fix parameters by collapsing bounds and forcing k0
    if fixed_params is not None:
        mask = jnp.ones_like(k0)

        for idx, val in fixed_params.items():
            lower = lower.at[idx].set(val)
            upper = upper.at[idx].set(val)
            k0 = k0.at[idx].set(val)
            mask = mask.at[idx].set(0.0)

        if verbose:
            print(f"Fixed parameters: {fixed_params}")
    else:
        mask = jnp.ones_like(k0)

    bounds_jax = (lower, upper)

    # ---------------------------
    # Loss and masked gradient
    # ---------------------------
    def loss(k):
        return loss_function(k, I_obs, X, Y, t_vals)

    loss_and_grad = jax.value_and_grad(loss)

    def masked_loss_and_grad(k):
        val, g = loss_and_grad(k)
        return val, g * mask

    # ---------------------------
    # JAXOPT solver
    # ---------------------------
    solver = LBFGSB(
        fun=masked_loss_and_grad,
        value_and_grad=True,
        maxiter=maxiter,
        tol=1e-8,
    )

    result = solver.run(
        init_params=k0,
        bounds=bounds_jax,
    )

    k_est = result.params

    if verbose:
        final_loss = float(loss(k_est))
        grad_norm = float(jnp.linalg.norm(masked_loss_and_grad(k_est)[1]))

        print("=== Optimization finished ===")
        print(f"Final loss: {final_loss:.6e}")
        print(f"‖grad‖:     {grad_norm:.6e}")
        print(f"k*:         {np.array2string(np.array(k_est), precision=4)}")

    return result