import numpy as np
import jax
import jax.numpy as jnp
from tqdm import tqdm
import matplotlib.pyplot as plt

from configuration.config_grid import X, Y
from minimization.solve_minimization_10D import simulate_image, loss_function

from statistical_methods.MCMC_runner_speedup import (
    sample_from_prior,
    log_poisson_likelihood,
)

from statistical_methods.MCMC_tests_speedup import (
    visualize_image,
    bin_image_and_grid,
)

from statistical_methods.MCMC_parallel_tempering import (
    run_parallel_tempering,
)

def summarize_mcmc(samples, true_theta=None, param_names=None):
    n_params = samples.shape[1]
    fig, axes = plt.subplots(nrows=2, ncols=(n_params + 1) // 2, figsize=(18, 6))
    axes = axes.flatten()

    # ---- Full 2π circular stats ----
    def circ_mean_2pi(ph):
        ph = ph % (2*np.pi)
        sin_sum = np.mean(np.sin(ph))
        cos_sum = np.mean(np.cos(ph))
        return (np.arctan2(sin_sum, cos_sum) + 2*np.pi) % (2*np.pi)

    def circ_std_2pi(ph):
        ph = ph % (2*np.pi)
        R = np.sqrt(np.mean(np.sin(ph))**2 + np.mean(np.cos(ph))**2)
        return np.sqrt(-2 * np.log(R + 1e-12))

    def circ_dist_2pi(a, b):
        """Shortest signed distance on full circle [0, 2π)."""
        return ((a - b + np.pi) % (2*np.pi)) - np.pi

    CIRC = {"phix", "phiy"}  # full circle variables

    print("\n=== Posterior Summary ===")
    for i in range(n_params):
        s = samples[:, i]
        name = param_names[i] if param_names else f"θ[{i}]"
        is_circ = name.lower() in CIRC

        if is_circ:
            s_mod = s % (2*np.pi)
            mean = circ_mean_2pi(s_mod)
            std = circ_std_2pi(s_mod)
        else:
            s_mod = s
            mean = np.mean(s_mod)
            std = np.std(s_mod)

        # ---- Print summary ----
        print(f"{name:>5}: {mean:8.3f} ± {std:6.3f}", end="")

        if true_theta is not None:
            if is_circ:
                tv = true_theta[i] % (2*np.pi)
                err = circ_dist_2pi(mean, tv)
            else:
                tv = true_theta[i]
                err = mean - tv
            print(f"   (true: {tv:.3f}, error: {err:.3f})")
        else:
            print()

        # ---- Plot ----
        ax = axes[i]
        if is_circ:
            ax.hist(s_mod, bins=30, color="orange", edgecolor="black", alpha=0.8)
            ax.set_xlim(0, 2*np.pi)
            ax.set_xticks([0, np.pi, 2*np.pi])
            ax.set_xticklabels(["0", "π", "2π"])
        else:
            ax.hist(s_mod, bins=30, color="steelblue", edgecolor="black", alpha=0.8)

        ax.set_title(name)
        ax.axvline(mean, color="black", linestyle="--", label="mean")
        if true_theta is not None:
            ax.axvline(tv, color="red", linestyle=":", label="true")
        ax.legend()

    # Hide any extra subplot axes
    for j in range(n_params, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.suptitle("Posterior Marginal Distributions", fontsize=16)
    plt.subplots_adjust(top=0.85)
    plt.show(block=True)

def summary_and_visualization(
    samples,
    k_true,
    I_obs,
    X,
    Y,
    t_vals,
    param_names,
    loss_fn,
    simulate_fn,
    title="Posterior mean vs observed image"
):
    try:
        print("blocking")
        samples = samples.block_until_ready()
        print("ready")
    except AttributeError:
        pass

    samples = np.array(samples)
    k_true = np.array(k_true)
    X = np.array(X)
    Y = np.array(Y)
    t_vals = np.array(t_vals)
    I_obs = np.array(I_obs)

    # ---- Full 2π circular stats ----
    def circular_mean(phases):
        phases = phases % (2*np.pi)
        sin_sum = np.mean(np.sin(phases))
        cos_sum = np.mean(np.cos(phases))
        return (np.arctan2(sin_sum, cos_sum) + 2*np.pi) % (2*np.pi)

    def circular_std(phases):
        phases = phases % (2*np.pi)
        R = np.sqrt(np.mean(np.sin(phases))**2 + np.mean(np.cos(phases))**2)
        return np.sqrt(-2 * np.log(R + 1e-12))

    # ---- Posterior mean and std ----
    k_post = np.mean(samples, axis=0)
    k_std  = np.std(samples, axis=0)

    CIRCULAR_PARAMS = {"phix", "phiy"}
    name_to_idx = {name.lower(): i for i, name in enumerate(param_names)}

    # Correct circular parameters
    for pname in CIRCULAR_PARAMS:
        if pname in name_to_idx:
            idx = name_to_idx[pname]
            s = samples[:, idx] % (2*np.pi)
            k_post[idx] = circular_mean(s)
            k_std[idx]  = circular_std(s)

    # ---- Print full summary ----
    print("\n=== Posterior Summary ===")
    summarize_mcmc(
        samples=samples,
        true_theta=k_true,
        param_names=param_names
    )

    # ---- Images ----
    I_post = simulate_fn(X, Y, t_vals, k_post)
    loss = float(loss_fn(k_post, I_obs, X, Y, t_vals))
    print(f"\nFinal loss (posterior mean vs I_obs): {loss:.6f}")

    I_true_clean = simulate_fn(X, Y, t_vals, k_true)
    I_diff = I_post - I_true_clean

    vmin, vmax = I_true_clean.min(), I_true_clean.max()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    im0 = axes[0].imshow(I_true_clean, extent=[X.min(), X.max(), Y.min(), Y.max()],
                         origin="lower", cmap="inferno", vmin=vmin, vmax=vmax)
    axes[0].set_title("True Image (clean)")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(I_post, extent=[X.min(), X.max(), Y.min(), Y.max()],
                         origin="lower", cmap="inferno", vmin=vmin, vmax=vmax)
    axes[1].set_title("Posterior Mean Image")
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(I_diff, extent=[X.min(), X.max(), Y.min(), Y.max()],
                         origin="lower", cmap="bwr")
    axes[2].set_title("Difference (Posterior - True)")
    plt.colorbar(im2, ax=axes[2])

    for ax in axes:
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    fig.suptitle(title)
    plt.tight_layout()
    plt.show(block=True)

    return k_post, k_std, loss


# ===========================================================
# Constants
# ===========================================================

PARAM_NAMES = ["Ax", "Ay", "σx", "σy", "cx", "cy", "fx", "fy", "phix", "phiy"]


# ===========================================================
# Beta ladder with top-linear, mid-geo, bottom-linear design
# ===========================================================

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


def wrap_params(k, param_names):
    k = np.array(k, dtype=float)
    for i, name in enumerate(param_names):
        if name.lower() in {"phix", "phiy"}:
            k[i] = k[i] % np.pi
    return k

def gaussian_prior_sample(key, mean, std, lower, upper):
    eps = jax.random.normal(key, shape=mean.shape)
    sample = mean + std * eps
    return jnp.clip(sample, lower, upper)


# ===========================================================
# Single-image PT driver
# ===========================================================

def run_parallel_tempering_for_image(
    I_obs,
    X,
    Y,
    t_vals,
    k_true,
    param_bounds,
    key,
    betas,
    proposal_std,
    n_chains=None,
    steps_per_swap=10,
    n_swap_intervals=None,
    burn_in_steps=None,
):
    """
    Run PT MCMC on one observation using fixed betas & proposal_std.
    """
    lower, upper = param_bounds

    # Bin & normalize
    I_obs, X_bin, Y_bin = bin_image_and_grid(I_obs, X, Y, factor=8)
    #I_obs /= jnp.max(I_obs)

    # # Prior
    # prior = dict(type="uniform", lower=lower, upper=upper)

   # === Nominal "best guess" parameters ===
    k_nominal = np.array([
        60.0, 20.0, 13.5, 5.05,
        0.0,  0.0,
        39.55, 29.05,
        0.0,  0.0
    ], dtype=float)

    # === Standard deviation ===
    frac = 0.20
    std_vec = np.array([
        frac * 60.0,
        frac * 20.0,
        frac * 13.5,
        frac * 5.05,
        frac * 1.0,
        frac * 1.0,
        0.05 * 39.55,
        0.05 * 29.05,
        frac * 1.0,
        frac * 1.0
    ], dtype=float)

    k_mean = jnp.array(k_nominal)
    k_std  = jnp.array(std_vec)

    # === Gaussian prior dict ===
    prior_gaussian = dict(
        type="gaussian",
        mean=k_mean,
        std=k_std,
        lower=lower,
        upper=upper,
    )

    # === Vectorized sampler over chains ===
    prior_sample_many = jax.vmap(gaussian_prior_sample,
                                in_axes=(0, None, None, None, None))

    # === Initialize thetas
    key, subkey = jax.random.split(key)
    keys0 = jax.random.split(subkey, n_chains)
    theta0_all = prior_sample_many(keys0, k_mean, k_std, lower, upper)

    visualize_image(I_obs, X_bin, Y_bin)

    # === Likelihood
    base_ll = lambda th: log_poisson_likelihood(th, I_obs, X_bin, Y_bin, t_vals)

    # === Run Parallel Tempering MCMC
    samples, swap_rate, acc_rate_per_chain, swap_rate_per_pair = run_parallel_tempering(
        key=key,
        theta0_all=theta0_all,
        prior=prior_gaussian,
        base_loglike_fn=base_ll,
        betas=betas,
        steps_per_swap=steps_per_swap,
        n_swap_intervals=n_swap_intervals,
        proposal_std=proposal_std,
        burn_in_steps=burn_in_steps,
    )

    # Diagnostics
    for i, (beta, acc) in enumerate(zip(betas, acc_rate_per_chain)):
        print(f"Chain {i:03d} (β={beta:.6f}): acceptance = {100*acc:.2f}%")

    for i, rate in enumerate(swap_rate_per_pair):
        print(
            f"Swap pair {i:03d} (β={betas[i]:.6f} <-> β={betas[i+1]:.6f}): "
            f"swap rate = {100*rate:.2f}%"
        )

    summary_and_visualization(
        samples=samples,
        k_true=k_true,
        I_obs=I_obs,
        X=X_bin,
        Y=Y_bin,
        t_vals=t_vals,
        param_names=PARAM_NAMES,
        loss_fn=loss_function,
        simulate_fn=simulate_image,
        title="Parallel Tempering Posterior",
    )

    post_mean = jnp.mean(samples, axis=0)
    return post_mean, samples, swap_rate


# ===========================================================
# Test-set runner
# ===========================================================

def run_parallel_tempering_on_test_set(I_all, T_all, k_true_all, param_bounds, seed=0):
    """
    Apply PT to all test samples with the same settings.
    """
    estimated_ks = []
    losses = []

    lower, upper = param_bounds

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

    print("Total chains:", n_chains)
    print("First 10 betas:", betas[:10])
    print("Last 10 betas:", betas[-10:])
    print("Is decreasing?", jnp.all(jnp.diff(betas) <= 0))

    key = jax.random.PRNGKey(seed)

    # -------------------------
    # Loop test images
    # -------------------------
    for i in tqdm(range(len(I_all)), desc="Running PT MCMC on test set"):
        I_obs = I_all[i]
        T = T_all[i]
        k_true = k_true_all[i]




        t_vals = jnp.arange(0, T, 1e-6)

        key, subkey = jax.random.split(key)
        k_est, _, _ = run_parallel_tempering_for_image(
            I_obs=I_obs,
            X=X,
            Y=Y,
            t_vals=t_vals,
            k_true=k_true,
            param_bounds=param_bounds,
            key=subkey,
            betas=betas,
            proposal_std=proposal_std,
            n_chains=n_chains,
            steps_per_swap=10,
            n_swap_intervals=4000,
            burn_in_steps=10_000,
        )

        loss = loss_function(k_est, I_obs, X, Y, t_vals)
        estimated_ks.append(k_est)
        losses.append(loss)

    return jnp.stack(estimated_ks), jnp.array(losses)


# ===========================================================
# Main
# ===========================================================

def main():
    data = np.load("test_all_algorithms/test_set_10D_gaussian.npz")
    I_all = data["I_all"]
    T_all = data["T_all"]
    k_true_all = data["k_all"]

    lower = np.array([0, 0, 2, 2, -20, -20, 20, 20, -np.pi, -np.pi], dtype=float)
    upper = np.array([65, 25, 20, 20, 20, 20, 50, 50,  np.pi,  np.pi], dtype=float)
    param_bounds = (lower, upper)

    estimated_ks, losses = run_parallel_tempering_on_test_set(
        I_all, T_all, k_true_all, param_bounds, seed=5
    )

    print("\n=== Parallel Tempering Summary ===")
    for i, (k_est, loss) in enumerate(zip(estimated_ks, losses)):
        print(
            f"Sample {i:02d}: loss = {loss:.4e}, "
            f"k = {np.round(k_est, 3)}"
        )

    np.savez(
        "test_all_algorithms/estimated_results_pt.npz",
        estimated_ks=estimated_ks,
        losses=losses,
    )


if __name__ == "__main__":
    main()