from statistical_methods.MCMC_runner_speedup import *
from statistical_methods.MCMC_tests_speedup import *
from statistical_methods.MCMC_parallel_tempering import *

import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np
import jax
import jax.numpy as jnp

from minimization.solve_minimization_10D import simulate_image, loss_function


# ============================================================
# 1. TRACE PLOTS
# ============================================================

def plot_traces(
    samples,
    param_names=None,
    chain_index=0,
    burn_in=0,
    thin=1,
    title="MCMC trace plots"
):
    """
    Plot MCMC trace plots in a 2 × 5 grid (for 10 parameters).
    Automatically adapts if number of parameters differs.
    """
    samples = np.array(samples)

    # Handle 2D vs 3D input
    if samples.ndim == 2:
        chain = samples
    elif samples.ndim == 3:
        chain = samples[chain_index]
    else:
        raise ValueError(f"Expected ndim 2 or 3, got {samples.shape}")

    # Burn-in and thinning
    chain = chain[burn_in::thin]
    n_samples, n_params = chain.shape

    if param_names is None:
        param_names = [f"θ[{i}]" for i in range(n_params)]

    # Determine grid shape
    n_cols = 5
    n_rows = int(np.ceil(n_params / n_cols))

    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=(18, 2.8 * n_rows),
        sharex=True
    )

    axes = axes.flatten()

    CIRCULAR_PARAMS = {"phi", "phix", "phiy"}
    x = np.arange(n_samples)

    for i in range(n_params):
        ax = axes[i]
        name = param_names[i]

        y = chain[:, i]
        if name.lower() in CIRCULAR_PARAMS:
            y = np.mod(y, 2 * np.pi)

        ax.plot(x, y, linewidth=0.6)
        ax.set_title(name, fontsize=10)
        ax.grid(alpha=0.3)

    # Hide unused axes if n_params < n_rows * n_cols
    for j in range(n_params, len(axes)):
        axes[j].axis("off")

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    plt.show()


def summarize_mcmc(samples, true_theta=None, param_names=None):
    n_params = samples.shape[1]
    fig, axes = plt.subplots(nrows=2, ncols=(n_params + 1) // 2, figsize=(18, 6))
    axes = axes.flatten()

    # ---- Circular stats for interval [0, π] ----
    def circ_mean_pi(ph):
        ph = ph % np.pi
        sin_sum = np.mean(np.sin(ph))
        cos_sum = np.mean(np.cos(ph))
        return (np.arctan2(sin_sum, cos_sum) + np.pi) % np.pi

    def circ_std_pi(ph):
        ph = ph % np.pi
        R = np.sqrt(np.mean(np.sin(ph))**2 + np.mean(np.cos(ph))**2)
        return np.sqrt(-2 * np.log(R))

    def circ_dist_pi(a, b):
        return ((a - b + (np.pi/2)) % np.pi) - (np.pi/2)

    CIRC = {"phix", "phiy"}

    print("\n=== Posterior Summary ===")
    for i in range(n_params):
        s = samples[:, i]
        name = param_names[i] if param_names else f"θ[{i}]"
        is_circ = name.lower() in CIRC

        if is_circ:
            s_mod = s % np.pi
            mean = circ_mean_pi(s_mod)
            std = circ_std_pi(s_mod)
        else:
            s_mod = s
            mean = np.mean(s)
            std = np.std(s)

        # ---- Print summary ----
        print(f"{name:>5}: {mean:8.3f} ± {std:6.3f}", end="")
        if true_theta is not None:
            if is_circ:
                tv = true_theta[i] % np.pi
                err = circ_dist_pi(mean, tv)
            else:
                tv = true_theta[i]
                err = mean - tv
            print(f"   (true: {tv:.3f}, error: {err:.3f})")
        else:
            print()

        # ---- Plot (all blue now) ----
        ax = axes[i]
        ax.hist(s_mod, bins=30, color="steelblue", edgecolor="black", alpha=0.8)

        if is_circ:
            ax.set_xlim(0, np.pi)
            ax.set_xticks([0, np.pi/2, np.pi])
            ax.set_xticklabels(["0", "π/2", "π"])

        ax.set_title(name)
        ax.axvline(mean, color="black", linestyle="--", label="mean")
        if true_theta is not None:
            ax.axvline(tv, color="red", linestyle=":", label="true")
        ax.legend()

    # Turn off unused axes
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

    def circular_mean(phases):
        sin_sum = np.mean(np.sin(phases))
        cos_sum = np.mean(np.cos(phases))
        return (np.arctan2(sin_sum, cos_sum) + 2 * np.pi) % (2 * np.pi)

    def circular_std(phases):
        R = np.sqrt(np.mean(np.sin(phases))**2 + np.mean(np.cos(phases))**2)
        return np.sqrt(-2 * np.log(R))  # von Mises approx

    k_post = np.mean(samples, axis=0)
    k_std  = np.std(samples, axis=0)

    # --- Correct circular stats for all relevant parameters ---
    CIRCULAR_PARAMS = {"phi", "phix", "phiy"}
    name_to_index = {name.lower(): i for i, name in enumerate(param_names)}
    for pname in CIRCULAR_PARAMS:
        if pname in name_to_index:
            idx = name_to_index[pname]
            s = samples[:, idx] % (2 * np.pi)
            k_post[idx] = circular_mean(s)
            k_std[idx] = circular_std(s)

    print("\n=== Posterior Summary ===")
    summarize_mcmc(
        samples=samples,
        true_theta=k_true,
        param_names=param_names
    )

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


PARAM_NAMES = ["Ax", "Ay", "σx", "σy", "cx", "cy", "fx", "fy", "phix", "phiy"]


def build_beta_ladder(
    n_chains=300,

    beta_max=1.0,       # always 1.0 for coldest
    beta_cut1=0.97,     # transition from dense linear to geometric
    beta_cut2=0.0026,   # transition from geometric to bottom linear
    beta_min=0.0025,    # hottest chain

    frac_top=0.5,       # linear region near β = 1.0
    frac_mid=0.4,       # geometric main region
    frac_bottom=0.1,    # linear tail near β_min
):
    """
    3-segment beta ladder:

      Top:    linear      beta_max → beta_cut1
      Mid:    geometric   beta_cut1 → beta_cut2
      Bottom: linear      beta_cut2 → beta_min

    Fractions specify how many chains go in each region.
    """

    # ---- SAFETY CONDITIONS ----
    if not (beta_max > beta_cut1 > beta_cut2 > beta_min):
        raise ValueError("Require beta_max > beta_cut1 > beta_cut2 > beta_min")

    # ---- Convert fractions to chain counts ----
    n_top = int(n_chains * frac_top)
    n_mid = int(n_chains * frac_mid)
    n_bottom = n_chains - n_top - n_mid  # force exact match

    # ---- 1. TOP region: linear 1.0 → 0.97 ----
    beta_top = jnp.linspace(beta_max, beta_cut1, n_top, endpoint=False)

    # ---- 2. MID region: geometric 0.97 → 0.0026 ----
    beta_mid = jnp.geomspace(beta_cut1, beta_cut2, n_mid, endpoint=False)

    # ---- 3. BOTTOM region: linear 0.0026 → 0.0025 ----
    beta_bottom = jnp.linspace(beta_cut2, beta_min, n_bottom, endpoint=True)

    # ---- Combine & ensure monotonicity ----
    betas = jnp.concatenate([beta_top, beta_mid, beta_bottom])
    betas = jnp.sort(betas)[::-1]  # enforce decreasing

    return betas


def autocorrelation(x):
    """
    Compute autocorrelation for lags 0..(N-1)
    """
    x = np.asarray(x)
    x = x - np.mean(x)
    N = len(x)

    # FFT for speed
    fftx = np.fft.fft(x, n=2*N)
    acf = np.fft.ifft(fftx * np.conjugate(fftx))[:N].real
    acf /= acf[0]
    return acf

def integrated_autocorrelation_time(acf):
    """
    Compute cumulative IACT as function of cutoff lag.
    """
    # τ_int(L) = 1 + 2 * sum_{l=1..L} acf[l]
    iact = 1 + 2 * np.cumsum(acf[1:])
    # Prepend τ_int(0)=1
    iact = np.concatenate([[1.0], iact])
    return iact

# --- Plotting function ---
def plot_acf_iact_for_param(param_samples, name, max_lag=5000):
    """
    param_samples: 1D array, samples from cold chain for one parameter
    """

    # Compute ACF
    acf = autocorrelation(param_samples)
    acf = acf[:max_lag]

    # Compute IACT
    iact = integrated_autocorrelation_time(acf)

    # Plot
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=False)

    ax[0].plot(acf)
    ax[0].set_title(f"Autocorrelation: {name}")
    ax[0].set_ylabel("ACF")
    ax[0].grid(True, alpha=0.3)

    ax[1].plot(iact)
    ax[1].set_title(f"Integrated autocorrelation time: {name}")
    ax[1].set_xlabel("Cutoff lag")
    ax[1].set_ylabel("IACT")
    ax[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show(block=True)

    return acf, iact

def diagnose_all_parameters(samples, param_names, max_lag=5000):
    samples = np.asarray(samples)

    # Handle 2D: (n_samples, n_params)
    if samples.ndim == 2:
        cold_chain = samples
    # Handle 3D: (n_chains, n_samples, n_params)
    elif samples.ndim == 3:
        cold_chain = samples[0]     # cold = β=1 chain
    else:
        raise ValueError(f"Expected samples with ndim=2 or 3, got {samples.shape}")

    all_acf = {}
    all_iact = {}

    for i, name in enumerate(param_names):
        print(f"\n=== Diagnostics for {name} ===")
        param_x = cold_chain[:, i]
        acf, iact = plot_acf_iact_for_param(param_x, name, max_lag=max_lag)
        all_acf[name] = acf
        all_iact[name] = iact

        plateau = iact[-1]
        ess = len(param_x) / plateau
        print(f"  IACT ≈ {plateau:.1f}")
        print(f"  ESS ≈ {ess:.1f} effective samples")

    return all_acf, all_iact

def print_betas(betas, decimals=12):
    for i, b in enumerate(betas):
        # format as fixed-precision float
        print(f"{i:04d}: {b:.{decimals}f}")


def plot_swap_rates(swap_rates, betas, title="Swap rates between adjacent chains"):
    swap_rates = np.array(swap_rates)
    n_pairs = len(swap_rates)
    x = np.arange(n_pairs)

    plt.figure(figsize=(12, 4))
    plt.bar(x, 100 * swap_rates, width=1.0, color="steelblue")

    # Target swap region
    plt.axhline(50, color="gray", linestyle="--", alpha=0.5)
    plt.axhline(70, color="gray", linestyle="--", alpha=0.5)

    plt.title(title)
    plt.xlabel("Swap pair index")
    plt.ylabel("Swap rate (%)")
    plt.ylim(0, 100)
    plt.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.show()


def plot_acceptance_rates(acc_rates, betas, title="Per-chain acceptance rates"):
    acc_rates = np.array(acc_rates)
    n = len(acc_rates)
    x = np.arange(n)

    plt.figure(figsize=(12, 4))
    plt.bar(x, 100 * acc_rates, width=1.0, color="steelblue")

    # Target acceptance region for Metropolis (10–30%)
    plt.axhline(10, color="gray", linestyle="--", alpha=0.5)
    plt.axhline(30, color="gray", linestyle="--", alpha=0.5)

    plt.title(title)
    plt.xlabel("Chain index")
    plt.ylabel("Acceptance rate (%)")
    plt.ylim(0, 100)
    plt.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.show()

def main():
    print("\n=== Parallel Tempering (10D with phix, phiy) ===")
    seed = 5
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
    upper = jnp.array([65, 25, 20, 20, 20, 20, 50, 50, np.pi, np.pi], dtype=jnp.float32)

    # Observation
    scale = 20_000
    I_clean = simulate_image(X, Y, t_vals, k_true)
    I_obs = add_poisson(I_clean, scale=scale)
    I_obs, X_bin, Y_bin = bin_image_and_grid(I_obs, X, Y, factor=8)
    #I_obs /= scale
    total_counts_obs = float(jnp.sum(I_obs))   # ✅ correct
    visualize_image(I_obs, X_bin, Y_bin)

    print("Total count: ", total_counts_obs)

    # Prior: uniform
    prior_uniform = dict(type="uniform", lower=lower, upper=upper)

    # Prior: Gaussian
    fraction = 0.5
    std_base = (upper - lower) * jnp.array(
        [fraction, fraction, fraction, fraction, fraction, fraction, 0.001, 0.001, fraction, fraction], dtype=jnp.float32
    )
    prior_gaussian = dict(
        type="gaussian",
        mean=k_true,
        std=std_base,
        lower=lower,
        upper=upper,
    )

    # Proposal std
    scale_factor = 1 / 200.0
    proposal_fractions = scale_factor * jnp.array(
        [0.02, 0.02, 0.01, 0.01, 0.02, 0.02, 0.03, 0.03, 0.05, 0.05],
        dtype=jnp.float32,
    )
    proposal_std = proposal_fractions * (upper - lower)

    n_chains = 300

    betas = build_beta_ladder(
        n_chains=n_chains,

        beta_max=1.0,       # always 1.0 for coldest
        beta_cut1=0.99,     # transition from dense linear to geometric
        beta_cut2=0.00255,   # transition from geometric to bottom linear
        beta_min=0.0025,    # hottest chain

        frac_top=0.1,       # linear region near β = 1.0
        frac_mid=0.8,       # geometric main region
        frac_bottom=0.1,    # linear tail near β_min
    )

    #betas = jnp.geomspace(1, 0.0025, n_chains, endpoint=True)

    print_betas(betas, decimals=18)

    # Optional sanity checks
    print("Total chains:", betas.size)
    print("First 10 betas:", betas[:10])
    print("Last 10 betas:", betas[-10:])
    print("Is decreasing?", jnp.all(jnp.diff(betas) <= 0.0))


    # Initial samples
    key, sub = jax.random.split(key)
    _, theta0_all = jax.vmap(lambda k: sample_from_prior(prior_uniform, k))(
        jax.random.split(sub, n_chains)
    )

    # Log-likelihood
    base_ll = lambda th: log_poisson_likelihood(th, I_obs, X_bin, Y_bin, t_vals, total_counts_obs)

    # Run PT
    samples, swap_rate, acc_rate_per_chain, swap_accept_means = run_parallel_tempering(
        key=key,
        theta0_all=theta0_all,
        prior=prior_uniform,
        base_loglike_fn=base_ll,
        betas=betas,
        steps_per_swap=10,
        n_swap_intervals=4000,
        proposal_std=proposal_std,
        burn_in_steps=10_000
    )

    for i, (beta, acc) in enumerate(zip(betas, acc_rate_per_chain)):
        print(f"Chain {i:02d} (β={beta:.4f}): acceptance = {100*acc:.2f}%")
    for i in range(len(swap_accept_means)):
        print(f"Swap pair {i:02d} (β={betas[i]:.4f} <-> β={betas[i+1]:.4f}): swap rate = {swap_accept_means[i]*100:.2f}%")


    plot_acceptance_rates(acc_rate_per_chain, betas)
    plot_swap_rates(swap_accept_means, betas)

    # ==============================================
    # Extract cold chain parameter trace
    # ==============================================

    # samples has shape (n_samples, dim)
    param_index = 0  # change this to inspect other ks

    trace = np.array(samples[:, param_index])

    print("\n=== Autocorrelation and IACT analysis ===")
    print(f"Parameter index: {param_index}")
    print(f"Number of samples: {len(trace)}")

    # Run diagnostics for all parameters
    #all_acf, all_iact = diagnose_all_parameters(samples, PARAM_NAMES, max_lag=5000)

    # ---- NEW: trace plots for the cold chain samples ----
    # If run_parallel_tempering already returns only cold-chain samples,
    # samples has shape (n_samples, n_params) and this just works.
    # If in the future you return all chains, this function will still
    # handle shape (n_chains, n_samples, n_params).
    plot_traces(
        samples=samples,
        param_names=PARAM_NAMES,
        chain_index=0,      # cold chain if samples is 3D; ignored if 2D
        burn_in=0,          # samples are already post-burn-in here
        thin=1,
        title="Trace plots for cold chain (β=1)"
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
        title="Parallel Tempering Posterior (10D)"
    )


if __name__ == "__main__":
    main()