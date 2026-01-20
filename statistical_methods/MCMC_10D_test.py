from statistical_methods.MCMC_runner_speedup import *
from statistical_methods.MCMC_tests_speedup import *
from statistical_methods.MCMC_parallel_tempering import *

import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np
import jax
import jax.numpy as jnp

from minimization.solve_minimization_10D import simulate_image, loss_function

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

    # Normalize both before comparison
    I_post = I_post * total_counts_obs 
    I_obs_n = I_obs 

    I_diff = I_post - I_obs_n

    l2_norm = np.linalg.norm(I_diff)
    rel_l2 = l2_norm / np.linalg.norm(I_obs_n)
    max_abs = np.max(np.abs(I_diff))

    print("\n=== Image Comparison (posterior mean vs real) ===")
    print(f"L2 norm         = {l2_norm:.4e}")
    print(f"Relative L2 err = {rel_l2:.4e}")
    print(f"Max abs error   = {max_abs:.4e}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    im0 = axes[0].imshow(I_obs_n, cmap="inferno", origin="lower")
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


def main():
    print("\n=== Parallel Tempering (10D with phix, phiy) ===")
    seed = 4
    np_rng = np.random.default_rng(seed)
    key = jax.random.PRNGKey(seed)

    # Configuration
    X, Y, t_vals, DT, _ = make_configuration(pulse_duration_ms=0.05)

    # True parameters
    fx = 39_550.0
    fy = 29_050.0
    phix = np.pi / 6
    phiy = np.pi / 2.3
    freq_scaling = 1000

    k_true = jnp.array([
        60.0, 20.0, 13.5, 5.05,
         0.0,  0.0,
        fx / freq_scaling, fy / freq_scaling,
        phix, phiy
    ])
    PARAM_NAMES = ["Ax", "Ay", "sigx", "sigy", "cx", "cy", "fx", "fy", "phix", "phiy"]
    print("True k:", dict(zip(PARAM_NAMES, np.array(k_true))))

    lower = jnp.array([0, 0, 2, 2, -20, -20, 20, 20, 0.0, 0.0], dtype=jnp.float32)
    upper = jnp.array([60, 20, 20, 20, 20, 20, 50, 50, 2*np.pi, 2*np.pi], dtype=jnp.float32)

    # Observation
    I_clean = simulate_image(X, Y, t_vals, k_true)
    I_obs = add_poisson(I_clean, scale=20000.)
    I_obs /= 20_000.

    I_obs, X_bin, Y_bin = bin_image_and_grid(I_obs, X, Y, factor=4)
    visualize_image(I_obs, X_bin, Y_bin)

    # Prior: uniform
    prior_uniform = dict(type="uniform", lower=lower, upper=upper)

    # -------------------------
    # Proposal std (10D)
    # -------------------------
    scale_factor = 1 / 200.0
    proposal_fractions = scale_factor * jnp.array(
        [0.02, 0.02, 0.01, 0.01, 0.02, 0.02, 0.03, 0.03, 0.05, 0.05],
        dtype=jnp.float32,
    )
    proposal_std = proposal_fractions * (upper - lower)

    # -------------------------
    # Beta ladder (global)
    # -------------------------
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
    # Initial samples
    key, sub = jax.random.split(key)
    _, theta0_all = jax.vmap(lambda k: sample_from_prior(prior_uniform, k))(
        jax.random.split(sub, n_chains)
    )

    # Log-likelihood
    base_ll = lambda th: log_poisson_likelihood(th, I_obs, X_bin, Y_bin, t_vals)

    # Run PT
    samples, swap_rate, acc_rate_per_chain, swap_accept_means = run_parallel_tempering(
        key=key,
        theta0_all=theta0_all,
        prior=prior_uniform,
        base_loglike_fn=base_ll,
        betas=betas,
        steps_per_swap=10,
        n_swap_intervals=200,
        proposal_std=proposal_std,
        burn_in_steps=5_000
    )

    for i, (beta, acc) in enumerate(zip(betas, acc_rate_per_chain)):
        print(f"Chain {i:02d} (β={beta:.4f}): acceptance = {100*acc:.2f}%")
    for i in range(len(swap_accept_means)):
        print(f"Swap pair {i:02d} (β={betas[i]:.4f} <-> β={betas[i+1]:.4f}): swap rate = {swap_accept_means[i]*100:.2f}%")

    summarize_mcmc(samples=samples, param_names=PARAM_NAMES)
    visualize_posterior_mean(
        samples=samples,
        I_obs=I_obs,
        X=X_bin,
        Y=Y_bin,
        t_vals=t_vals,
        param_names=PARAM_NAMES,
        simulate_fn=simulate_image,
        total_counts_obs=20_000
    )


if __name__ == "__main__":
    main()