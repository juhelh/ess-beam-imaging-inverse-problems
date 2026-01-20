import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp

from configuration.configuration import make_configuration
from minimization.visualize_simulations import visualize_image

# MCMC machinery
from statistical_methods.MCMC_runner_speedup import sample_from_prior, log_poisson_likelihood
from statistical_methods.MCMC_parallel_tempering import run_parallel_tempering

from scipy.ndimage import gaussian_filter

from real_images.preprocessing import (
    remove_background,
    tight_crop_bbox,
    crop_image,
    crop_grids,
    show_image,
    show_bbox,
)

from minimization.solve_minimization_10D_real import simulate_image

def load_real_images():
    """Load Cyrille's real raster images from the MAT file."""
    folder = os.path.dirname(__file__)
    path = os.path.join(folder, "data_laser_raster.mat")
    data = sio.loadmat(path)
    return data["im_all_roi"]   # shape (10, 181, 351)


# --------------------------------------------------
# Simple Gaussian-image loss + log-likelihood
# --------------------------------------------------

def image_loss(k, I_obs, X, Y, t_vals):
    """
    L2 loss between L2-normalized simulated and observed images.
    Deterministic; no Poisson noise here.
    """
    # Fix fx, fy internally to known values (Hz scale already applied in simulator)
    known_fx = 21.0
    known_fy = 15.0

    k = jnp.array(k)
    k_eff = k.at[6].set(known_fx).at[7].set(known_fy)

    I_sim = simulate_image(X, Y, t_vals, k_eff)

    # L2-normalize both
    I_sim = I_sim / jnp.linalg.norm(I_sim)
    I_obs = I_obs / jnp.linalg.norm(I_obs)

    return jnp.sum((I_sim - I_obs) ** 2)


def make_loglike(I_obs, X, Y, t_vals):
    """
    Wrap image_loss into a log-likelihood usable by MCMC.
    We just take loglike = -0.5 * loss; the constant scale is irrelevant.
    """
    def loglike(theta):
        L = image_loss(theta, I_obs, X, Y, t_vals)
        return -0.5 * L
    return jax.jit(loglike)


def build_beta_ladder(
    n_chains=300,
    beta_max=1.0,
    beta_cut1=0.97,
    beta_cut2=0.0026,
    beta_min=0.0025,
    frac_top=0.2,
    frac_mid=0.7,
    frac_bottom=0.1,
):
    if not (beta_max > beta_cut1 > beta_cut2 > beta_min):
        raise ValueError("Require beta_max > beta_cut1 > beta_cut2 > beta_min")

    n_top = int(n_chains * frac_top)
    n_mid = int(n_chains * frac_mid)
    n_bottom = n_chains - n_top - n_mid

    beta_top = jnp.linspace(beta_max, beta_cut1, n_top, endpoint=False)
    beta_mid = jnp.geomspace(beta_cut1, beta_cut2, n_mid, endpoint=False)
    beta_bottom = jnp.linspace(beta_cut2, beta_min, n_bottom, endpoint=True)

    betas = jnp.concatenate([beta_top, beta_mid, beta_bottom])
    betas = jnp.sort(betas)[::-1]
    return betas

# --------------------------------------------------
# Posterior summary + visualization for real images
# --------------------------------------------------


def summarize_mcmc(samples, param_names=None):
    samples = np.array(samples)
    n_params = samples.shape[1]

    fig, axes = plt.subplots(nrows=2, ncols=(n_params + 1) // 2, figsize=(18, 6))
    axes = axes.flatten()

    def circular_mean(phases):
        sin_sum = np.mean(np.sin(phases))
        cos_sum = np.mean(np.cos(phases))
        return (np.arctan2(sin_sum, cos_sum) + 2 * np.pi) % (2 * np.pi)

    def circular_std(phases):
        R = np.sqrt(np.mean(np.sin(phases))**2 + np.mean(np.cos(phases))**2)
        return np.sqrt(-2 * np.log(R))  # von Mises approx

    CIRCULAR_PARAMS = {"phi", "phix", "phiy"}

    print("\n=== Posterior Summary (real image) ===")
    for i in range(n_params):
        s = samples[:, i]
        name = param_names[i] if param_names else f"θ[{i}]"
        is_circular = name.lower() in CIRCULAR_PARAMS

        if is_circular:
            s = s % (2 * np.pi)
            mean = circular_mean(s)
            std = circular_std(s)
        else:
            mean = np.mean(s)
            std = np.std(s)

        print(f"{name:>5}: {mean:8.3f} ± {std:6.3f}")

        ax = axes[i]
        if is_circular:
            s_plot = s % (2 * np.pi)
            ax.hist(s_plot, bins=30, color="orange", edgecolor="black", alpha=0.8)
            ax.set_xlim(0, 2 * np.pi)
            ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
            ax.set_xticklabels(["0", "π/2", "π", "3π/2", "2π"])
        else:
            ax.hist(s, bins=30, color="steelblue", edgecolor="black", alpha=0.8)

        ax.set_title(name)
        ax.axvline(mean, color="black", linestyle="--", label="mean")
        ax.legend()

    for j in range(n_params, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.suptitle("Posterior Marginal Distributions (real image)", fontsize=16)
    plt.subplots_adjust(top=0.86)
    plt.show(block=True)


def visualize_posterior_mean(
    samples,
    I_obs,
    X,
    Y,
    t_vals,
    param_names,
    simulate_fn,
    total_counts_obs = 10**6
):
    """
    For real data: show Observed vs Posterior-mean simulated vs Difference.
    """
    try:
        samples = samples.block_until_ready()
    except AttributeError:
        pass

    samples = np.array(samples)
    I_obs = np.array(I_obs)
    X = np.array(X)
    Y = np.array(Y)
    t_vals = np.array(t_vals)

    # Posterior mean (with circular correction for phix, phiy)
    k_post = np.mean(samples, axis=0)
    k_std = np.std(samples, axis=0)

    def circular_mean(phases):
        sin_sum = np.mean(np.sin(phases))
        cos_sum = np.mean(np.cos(phases))
        return (np.arctan2(sin_sum, cos_sum) + 2 * np.pi) % (2 * np.pi)

    def circular_std(phases):
        R = np.sqrt(np.mean(np.sin(phases))**2 + np.mean(np.cos(phases))**2)
        return np.sqrt(-2 * np.log(R))

    CIRCULAR_PARAMS = {"phi", "phix", "phiy"}
    name_to_index = {name.lower(): i for i, name in enumerate(param_names)}
    for pname in CIRCULAR_PARAMS:
        if pname in name_to_index:
            idx = name_to_index[pname]
            s = samples[:, idx] % (2 * np.pi)
            k_post[idx] = circular_mean(s)
            k_std[idx] = circular_std(s)

    print("\nPosterior mean k (real image):")
    for name, m, s in zip(param_names, k_post, k_std):
        print(f"{name:>5}: {m:8.3f} ± {s:6.3f}")

    # Fix fx, fy before simulating (same trick as in loss)
    known_fx = 21.0
    known_fy = 15.0
    k_eff = k_post.copy()
    k_eff[6] = known_fx
    k_eff[7] = known_fy

    I_post = simulate_fn(X, Y, t_vals, k_eff)
    I_post /= jnp.sum(I_post)

    I_post = I_post * total_counts_obs

    # Debug
    print("Simulated total counts :", jnp.sum(I_post))
    print("Observed total counts :", jnp.sum(I_obs))

    I_diff = I_post - I_obs

    l2_norm = np.linalg.norm(I_diff)
    rel_l2 = l2_norm / np.linalg.norm(I_obs)
    max_abs = np.max(np.abs(I_diff))

    print("\n=== Image Comparison (posterior mean vs real) ===")
    print(f"L2 norm         = {l2_norm:.4e}")
    print(f"Relative L2 err = {rel_l2:.4e}")
    print(f"Max abs error   = {max_abs:.4e}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    im0 = axes[0].imshow(I_obs, cmap="inferno", origin="lower")
    axes[0].set_title("Observed image (normalized)")
    plt.colorbar(im0, ax=axes[0], shrink=0.8)

    im1 = axes[1].imshow(I_post, cmap="inferno", origin="lower")
    axes[1].set_title("Posterior mean simulated")
    plt.colorbar(im1, ax=axes[1], shrink=0.8)

    im2 = axes[2].imshow(I_diff, cmap="seismic", origin="lower")
    axes[2].set_title("Difference (sim - obs)")
    plt.colorbar(im2, ax=axes[2], shrink=0.8)

    plt.suptitle("Parallel Tempering Posterior (real image)", fontsize=14)
    plt.tight_layout()
    plt.show(block=True)

def load_custom_snapshot():
    """
    Load a single beam snapshot saved from EPICS acquisition.

    Returns
    -------
    I_obs : ndarray (H,W), float32
    meta  : dict with fx, fy, exposure_time
    """
    path = os.path.join(
        "beam_images",
        "beam_snapshot_1766149115.npz"
    )

    data = np.load(path)

    I_obs = data["image"].astype(np.float32)

    meta = {
        "fx": float(data["fx"]),              # kHz
        "fy": float(data["fy"]),              # kHz
        "exposure_time": float(data["exposure_time"])
    }

    print("\n=== Loaded custom snapshot metadata ===")
    for k, v in meta.items():
        print(f"{k:>14s} = {v}")

    return I_obs, meta

def bin_image_and_grid(I, X, Y, factor=2):
    """
    Downsample image I and grid (X, Y) by a given binning factor.
    Poisson-consistent: image is SUM-binned, coordinates are MEAN-binned.
    """

    def bin_sum(A):
        H, W = A.shape
        H_new = H // factor
        W_new = W // factor
        return (
            A[:H_new*factor, :W_new*factor]
            .reshape(H_new, factor, W_new, factor)
            .sum(axis=(1, 3))
        )

    def bin_mean(A):
        H, W = A.shape
        H_new = H // factor
        W_new = W // factor
        return (
            A[:H_new*factor, :W_new*factor]
            .reshape(H_new, factor, W_new, factor)
            .mean(axis=(1, 3))
        )

    I_binned = bin_sum(I)
    X_binned = bin_mean(X)
    Y_binned = bin_mean(Y)

    return I_binned, X_binned, Y_binned

# --------------------------------------------------
# Main: Parallel Tempering on a real image
# --------------------------------------------------

def run_mcmc_on_real_image(idx=0):
    print("\n=== Parallel Tempering MCMC on REAL image ===")

    # --------------------------------------------------
    # 1) Load image
    # --------------------------------------------------
    # If you want the MAT images, use these two lines instead:
    # imgs = load_real_images()
    # I_raw = imgs[idx].astype(np.float32)

    # Using EPICS snapshot (your current choice)
    I_raw, meta = load_custom_snapshot()
    I_clean = remove_background(I_raw, sigma=30)

    # Use the CLEANED image shape for configuration
    H0, W0 = I_clean.shape

    # --------------------------------------------------
    # 2) Configuration: match spatial grid to image shape
    # --------------------------------------------------
    # Use real exposure time if available (your snapshot provides it)
    # You earlier used exposure_time / 1_000_000 -> ms, assuming exposure_time is in microseconds.
    pulse_duration_sim = float(meta["exposure_time"]) / 1_000_000.0  # ms

    sampling_rate = 2_000_000  # Hz

    X, Y, t_vals, dt, _ = make_configuration(
        pulse_duration_ms=pulse_duration_sim,
        field_x_mm=float(W0),
        field_y_mm=float(H0),
        pixel_size_mm=1.0,
        sampling_rate=sampling_rate,
    )
    print(f"X/Y grid shape (full): {X.shape}, time steps: {len(t_vals)}")

    # --------------------------------------------------
    # 3) Tight crop in ORIGINAL resolution, then crop grids
    # --------------------------------------------------
    bbox = tight_crop_bbox(I_clean, frac=0.02, pad=10, min_h=32, min_w=32)
    show_bbox(I_clean, bbox, title="Tight signal ROI (full-res)")

    I_obs = crop_image(I_clean, bbox) 

    total_counts_obs = np.sum(I_obs)
    peak_obs = np.max(I_obs)

    print("Observed peak:", peak_obs)
    print("Observed total counts:", total_counts_obs)
    X, Y = crop_grids(X, Y, bbox)

    # Safety checks (catch empty crops early)
    assert I_obs.size > 0, "I_obs is empty after cropping!"
    assert X.size > 0 and Y.size > 0, "X/Y grids are empty after cropping!"

    # --------------------------------------------------
    # 4) Bin consistently (image + grids together)
    # --------------------------------------------------
    I_obs, X, Y = bin_image_and_grid(I_obs, X, Y, factor=1)
    show_image(I_obs, "Observed image (cropped + binned)")

    total_counts_obs = float(jnp.sum(I_obs))   # recompute AFTER binning
    peak_obs = float(np.max(I_obs))
    print("Observed peak (post-bin):", peak_obs)
    print("Observed total counts (post-bin):", total_counts_obs)

    print(f"I_obs shape after crop+bin: {I_obs.shape}")
    print(f"X/Y shape after crop+bin:  {X.shape}")

    # --------------------------------------------------
    # 5) Prior
    # --------------------------------------------------
    fx0 = float(meta["fx"])   # kHz (as saved in npz)
    fy0 = float(meta["fy"])   # kHz
    H, W = I_obs.shape

    Ax_max = 0.5 * W
    Ay_max = 0.5 * H

    lower = np.array([
        10.0, 10.0,          # Ax, Ay
        2.0,  2.0,           # sigx, sigy
    -5.0, -5.0,         # cx, cy
        0.95 * fx0, 0.95 * fy0,
        0.0,  0.0
    ], dtype=np.float32)

    upper = np.array([
        Ax_max, Ay_max,      # Ax, Ay  <-- FIX
        6.0,  6.0,         # sigx, sigy
        5.0,  5.0,         # cx, cy
        1.05 * fx0, 1.05 * fy0,
        2*np.pi, 2*np.pi
    ], dtype=np.float32)

    cx0 = 0.5 * (float(np.min(X)) + float(np.max(X)))
    cy0 = 0.5 * (float(np.min(Y)) + float(np.max(Y)))

    cx_half = 0.5 * (float(np.max(X)) - float(np.min(X)))
    cy_half = 0.5 * (float(np.max(Y)) - float(np.min(Y)))

    lower[4] = cx0 - 0.6 * cx_half
    upper[4] = cx0 + 0.6 * cx_half
    lower[5] = cy0 - 0.6 * cy_half
    upper[5] = cy0 + 0.6 * cy_half

    k_mean = jnp.array([
        0.4 * W,          # Ax: large enough to allow collapse
        0.4 * H,          # Ay
        6.0,              # sigx
        6.0,              # sigy
        cx0,              # cx
        cy0,              # cy
        fx0,              # fx (tight!)
        fy0,              # fy
        jnp.pi,              # phix
        jnp.pi               # phiy
    ], dtype=jnp.float32)

    k_std = jnp.array([
        0.30 * k_mean[0],   # Ax  (very broad)
        0.30 * k_mean[1],   # Ay
        0.40 * k_mean[2],   # sigx
        0.40 * k_mean[3],   # sigy
        0.5 * (upper[4] - lower[4]),  # cx
        0.5 * (upper[5] - lower[5]),  # cys
        0.02 * fx0,         # fx  (tight!)
        0.02 * fy0,         # fy
        5.0,                # phix
        5.0                 # phiy
    ], dtype=jnp.float32)

    prior_gaussian = dict(
        type="gaussian",
        mean=k_mean,
        std=k_std,
        lower=lower,
        upper=upper,
    )

    PARAM_NAMES = ["Ax", "Ay", "sigx", "sigy", "cx", "cy", "fx", "fy", "phix", "phiy"]

    # --------------------------------------------------
    # 6) Proposal std
    # --------------------------------------------------
    scale_factor = 1 / 200.0
    proposal_fractions = scale_factor * jnp.array(
        [0.02, 0.02, 0.01, 0.01, 0.02, 0.02, 0.03, 0.03, 0.05, 0.05],
        dtype=jnp.float32,
    )
    proposal_std = proposal_fractions * (upper - lower)

    # --------------------------------------------------
    # 7) Beta ladder
    # --------------------------------------------------
    n_chains = 300
    betas = build_beta_ladder(
        n_chains=n_chains,
        beta_max=1.0,
        beta_cut1=0.97,
        beta_cut2=0.0026,
        beta_min=0.0025,
        frac_top=0.2,
        frac_mid=0.7,
        frac_bottom=0.1,
    )

    # --------------------------------------------------
    # 8) Initial samples
    # --------------------------------------------------
    seed = 7
    key = jax.random.PRNGKey(seed)
    key, sub = jax.random.split(key)

    _, theta0_all = jax.vmap(lambda k: sample_from_prior(prior_gaussian, k))(
        jax.random.split(sub, n_chains)
    )

    # --------------------------------------------------
    # 9) Log-likelihood (Poisson)
    # --------------------------------------------------
    base_ll = lambda th: log_poisson_likelihood(th, I_obs, X, Y, t_vals, total_counts_obs=total_counts_obs)

    print("X range:", float(X.min()), float(X.max()), "mean", float(X.mean()))
    print("Y range:", float(Y.min()), float(Y.max()), "mean", float(Y.mean()))
    print("cx,cy:", float(k_mean[4]), float(k_mean[5]))
    # --------------------------------------------------
    # 10) Run Parallel Tempering
    # --------------------------------------------------
    samples, swap_rate, acc_rate_per_chain, swap_accept_means = run_parallel_tempering(
        key=key,
        theta0_all=theta0_all,
        prior=prior_gaussian,
        base_loglike_fn=base_ll,
        betas=betas,
        steps_per_swap=10,
        n_swap_intervals=2000,
        proposal_std=proposal_std,
        burn_in_steps=10_000,
    )

    # --- MASS CONSERVATION CHECK (REAL IMAGE) ---

    # 1) Observed photons (ground truth)
    N_obs = float(jnp.sum(I_obs))

    # 2) Posterior-mean parameters (with circular fix if needed)
    k_post = np.mean(samples, axis=0)
    k_eff = k_post.copy()


    # 3) Simulate raw intensity
    I_sim_raw = simulate_image(X, Y, t_vals, k_eff)

    # 4) Normalize ONCE to probability mass
    I_sim_prob = I_sim_raw / jnp.sum(I_sim_raw)

    # 5) Scale to observed photon count
    I_sim = N_obs * I_sim_prob

    # 6) Diagnostics
    print("\n=== MASS CONSERVATION CHECK ===")
    print(f"Observed total counts   : {N_obs:.6e}")
    print(f"Simulated total counts  : {float(jnp.sum(I_sim)):.6e}")
    print(f"Relative mismatch       : "
        f"{float((jnp.sum(I_sim) - N_obs) / N_obs):+.3e}")

    # --------------------------------------------------
    # 11) Summaries + plots
    # --------------------------------------------------
    print("\n=== Chain acceptance rates ===")
    for i, (beta, acc) in enumerate(zip(betas, acc_rate_per_chain)):
        print(f"Chain {i:02d} (β={float(beta):.4f}): acceptance = {100*acc:.2f}%")

    print("\n=== Swap acceptance rates ===")
    for i in range(len(swap_accept_means)):
        print(
            f"Swap pair {i:02d} (β={float(betas[i]):.4f} <-> β={float(betas[i+1]):.4f}): "
            f"swap rate = {swap_accept_means[i]*100:.2f}%"
        )

    summarize_mcmc(samples=samples, param_names=PARAM_NAMES)
    visualize_posterior_mean(
        samples=samples,
        I_obs=I_obs,
        X=X,
        Y=Y,
        t_vals=t_vals,
        param_names=PARAM_NAMES,
        simulate_fn=simulate_image,
        total_counts_obs=total_counts_obs
    )


def main():
    run_mcmc_on_real_image(idx=0)


if __name__ == "__main__":
    main()
