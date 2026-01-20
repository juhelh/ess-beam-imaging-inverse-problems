# MCMC.py

import numpy as np
import jax.numpy as jnp
import jax.scipy as jsp
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from jax.scipy.special import gammaln
import jax
from jax.scipy.stats import norm

# === Beam model functions ===

# === Grid (from config) ===
from configuration.configuration import make_configuration

from minimization.visualize_simulations import visualize_image

import seaborn as sns
import pandas as pd
from typing import Tuple, Optional


# === Step 1: Define parameter names and dimension ===
PARAM_NAMES = ["Ax", "Ay", "sigx", "sigy", "cx", "cy", "fx", "fy"]
from minimization.solve_minimization_10D import simulate_image


@jax.jit
def log_gaussian_prior(theta: jnp.ndarray,
                       prior_mean: jnp.ndarray,
                       prior_std: jnp.ndarray):
    """
    Compute log prior density for theta under independent Gaussian priors.
    JIT-compiled, purely JAX.
    """
    z = (theta - prior_mean) / prior_std
    return -0.5 * jnp.sum(z**2)

# def log_gaussian_prior(theta, mean, std):
#     log_norm = -jnp.log(std * jnp.sqrt(2 * jnp.pi))
#     ll = log_norm - 0.5 * ((theta - mean) / std) ** 2
#     return jnp.sum(ll, axis=-1)     # keep batch axis

@jax.jit
def log_uniform_prior(theta: jnp.ndarray,
                      lower: jnp.ndarray,
                      upper: jnp.ndarray):
    """
    Log prior for uniform distribution within [lower, upper].
    Returns 0 if inside bounds, -inf if outside.
    """
    inside = jnp.all((theta >= lower) & (theta <= upper))
    return jnp.where(inside, 0.0, -jnp.inf)


# def log_uniform_prior(theta, lower, upper):
#     # theta (…, 8)
#     inside = jnp.all((theta >= lower) & (theta <= upper), axis=-1)   # per sample
#     return jnp.where(inside, 0.0, -jnp.inf)


@jax.jit
def log_prior(theta: jnp.ndarray, prior: dict):
    """
    Generic wrapper selecting prior type.
    """
    typ = prior["type"]

    if typ == "gaussian":
        return log_gaussian_prior(theta, prior["mean"], prior["std"])
    elif typ == "uniform":
        return log_uniform_prior(theta, prior["lower"], prior["upper"])
    else:
        raise ValueError(f"Unsupported prior type: {typ!r}")


def sample_from_prior(prior: dict, key: jax.Array) -> tuple[jax.Array, jnp.ndarray]:
    """
    Sample from a prior using JAX random functions.
    Works for Gaussian and Uniform priors.
    Returns a new RNG key and the sample.

    Parameters
    ----------
    prior : dict
        Must contain fields depending on type:
            - Gaussian: {"type": "gaussian", "mean", "std", "lower", "upper"}
            - Uniform:  {"type": "uniform",  "lower", "upper"}
    key : jax.Array
        PRNG key

    Returns
    -------
    key_out : jax.Array
        Updated PRNG key
    sample : jnp.ndarray
        Sample drawn from the prior
    """
    typ = prior["type"]

    if typ == "gaussian":
        mean, std = prior["mean"], prior["std"]
        lower, upper = prior["lower"], prior["upper"]

        # --- Body and cond for rejection sampling ---
        def cond_fun(state):
            _, _, accept = state
            return ~accept

        def body_fun(state):
            key, _, _ = state
            key, subkey = jax.random.split(key)
            sample = mean + std * jax.random.normal(subkey, shape=mean.shape)
            accept = jnp.all((sample >= lower) & (sample <= upper))
            return key, sample, accept

        # Initialize state
        init_state = (key, mean, False)
        key_out, sample, _ = jax.lax.while_loop(cond_fun, body_fun, init_state)
        return key_out, sample

    elif typ == "uniform":
        lower, upper = prior["lower"], prior["upper"]
        key, subkey = jax.random.split(key)
        sample = jax.random.uniform(subkey, shape=lower.shape, minval=lower, maxval=upper)
        return key, sample

    else:
        raise ValueError(f"Unsupported prior type: {typ!r}")

def sample_from_prior(prior: dict, key: jax.Array) -> tuple[jax.Array, jnp.ndarray]:
    """
    Sample from a prior using JAX random functions.
    Works for Gaussian (truncated) and Uniform priors.
    Returns a new RNG key and the sample.

    Parameters
    ----------
    prior : dict
        Must contain fields depending on type:
            - Gaussian: {"type": "gaussian", "mean", "std", "lower", "upper"}
            - Uniform:  {"type": "uniform",  "lower", "upper"}
    key : jax.Array
        PRNG key

    Returns
    -------
    key_out : jax.Array
        Updated PRNG key
    sample : jnp.ndarray
        Sample drawn from the prior
    """
    typ = prior["type"]

    if typ == "gaussian":
        mean, std = prior["mean"], prior["std"]
        lower, upper = prior["lower"], prior["upper"]

        key, subkey = jax.random.split(key)
        # Inverse transform sampling for truncated normal
        a = (lower - mean) / std
        b = (upper - mean) / std
        u = jax.random.uniform(subkey, shape=mean.shape)
        cdf_a = norm.cdf(a)
        cdf_b = norm.cdf(b)
        u_scaled = cdf_a + u * (cdf_b - cdf_a)
        z = norm.ppf(u_scaled)
        sample = mean + std * z
        return key, sample

    elif typ == "uniform":
        lower, upper = prior["lower"], prior["upper"]
        key, subkey = jax.random.split(key)
        sample = jax.random.uniform(subkey, shape=lower.shape, minval=lower, maxval=upper)
        return key, sample

    else:
        raise ValueError(f"Unsupported prior type: {typ!r}")

    
    
@jax.jit
def log_weighted_gaussian_likelihood(theta: jnp.ndarray,
                                     I_obs: jnp.ndarray,
                                     X: jnp.ndarray,
                                     Y: jnp.ndarray,
                                     t_vals: jnp.ndarray,
                                     noise_std: float = 0.03,
                                     scale: str = "per_pixel",
                                     epsilon: float = 1e-8):
    """
    Heteroscedastic Gaussian approximation to Poisson noise:
        Var(I) ≈ I_sim  (valid for large counts)
    """
    I_sim = simulate_image(X, Y, t_vals, theta)
    I_sim = jnp.clip(I_sim, epsilon)
    residual = I_obs - I_sim
    weighted_sq_error = (residual / noise_std) ** 2 / I_sim

    return jnp.where(
        scale == "per_pixel",
        -0.5 * jnp.mean(weighted_sq_error),
        -0.5 * jnp.sum(weighted_sq_error),
    )


@jax.jit
def log_poisson_likelihood(
    k: jnp.ndarray,
    I_obs: jnp.ndarray,
    X: jnp.ndarray,
    Y: jnp.ndarray,
    t_vals: jnp.ndarray,
    total_counts_obs: float,
    epsilon: float = 1e-8,
):
    # Forward model
    if len(k) == 10:
        I_sim = simulate_image(X, Y, t_vals, k)

    # Convert shape → probability mass
    I_sim = I_sim / jnp.sum(I_sim)

    # Scale to observed total photons
    I_sim = total_counts_obs * I_sim

    # Numerical safety
    I_sim = jnp.clip(I_sim, epsilon)

    # Poisson log-likelihood
    return jnp.sum(I_obs * jnp.log(I_sim) - I_sim)


def reflect(x, lower, upper):
    return jnp.where(
        x < lower, 2 * lower - x,
        jnp.where(x > upper, 2 * upper - x, x)
    )



def propose(key, theta, proposal_std, lower, upper):
    key, subkey = jax.random.split(key)
    eps = jax.random.normal(subkey, shape=theta.shape) * proposal_std
    theta_prop = reflect(theta + eps, lower, upper)
    return key, theta_prop


def choose_initial_theta(prior_mean: jnp.ndarray,
                         key: jax.Array,
                         prior_std: Optional[jnp.ndarray] = None) -> jnp.ndarray:
    """
    Simple initialization strategy: prior mean + Gaussian noise.
    """
    if prior_std is None:
        return prior_mean
    eps = jax.random.normal(key, shape=prior_mean.shape) * prior_std
    return prior_mean + eps

def make_log_likelihood(noise_model: str,
                        I_obs: jnp.ndarray,
                        X: jnp.ndarray,
                        Y: jnp.ndarray,
                        t_vals: jnp.ndarray,
                        noise_std: float = 0.03):
    """
    Return the appropriate log-likelihood function.
    """

    if noise_model == "poisson":
        return lambda theta: log_poisson_likelihood(theta, I_obs, X, Y, t_vals)
    elif noise_model == "weighted_gaussian":
        return lambda theta: log_weighted_gaussian_likelihood(theta, I_obs, X, Y, t_vals,
                                                              noise_std=noise_std, scale="per_pixel")
    else:
        raise ValueError(f"Unknown noise model: {noise_model}")


def run_mcmc_jax(key: jax.Array,
                 theta0: jnp.ndarray,
                 prior: dict,
                 log_likelihood_fn,
                 num_samples: int = 5000,
                 burn_in: int = 1000,
                 thin: int = 1,
                 proposal_std: jnp.ndarray = None,
                 proposal_fn=propose) -> tuple[jnp.ndarray, float]:
    """
    Fully JAX-based Metropolis–Hastings MCMC with lax.scan.

    Parameters
    ----------
    key : PRNGKey
        Random key for RNG
    theta0 : array, shape (D,)
        Initial parameter vector
    prior : dict
        Prior info (mean/std/lower/upper/type)
    log_likelihood_fn : callable
        Function returning log-likelihood for a given θ
    num_samples : int
        Number of posterior samples to collect
    burn_in : int
        Number of burn-in steps to discard
    thin : int
        Keep one sample every `thin` steps
    proposal_std : float or array
        Proposal standard deviation(s)

    Returns
    -------
    samples : array, shape (N_samples, D)
    accept_rate : float
    """

    lower, upper = prior["lower"], prior["upper"]

    # --- Choose log_prior_fn BEFORE JIT so JAX never sees strings ---
    if prior["type"] == "gaussian":
        log_prior_fn = lambda theta: log_gaussian_prior(theta, prior["mean"], prior["std"])
    elif prior["type"] == "uniform":
        log_prior_fn = lambda theta: log_uniform_prior(theta, prior["lower"], prior["upper"])
    else:
        raise ValueError(f"Unsupported prior type: {prior['type']}")

    # Define JAX-safe log posterior function (arrays only!)
    def log_posterior(theta):
        return log_prior_fn(theta) + log_likelihood_fn(theta)

    # --- MCMC transition function ---
    def mcmc_step(carry, _):
        key, theta, logp, accepted = carry
        key, subkey = jax.random.split(key)

        # Propose new candidate
        key, theta_prop = proposal_fn(subkey, theta, proposal_std, lower, upper)

        logp_prop = log_posterior(theta_prop)
        log_alpha = logp_prop - logp

        key, subkey = jax.random.split(key)
        u = jnp.log(jax.random.uniform(subkey))

        accept = u < log_alpha
        theta_new = jnp.where(accept, theta_prop, theta)
        logp_new = jnp.where(accept, logp_prop, logp)
        accepted_new = accepted + accept.astype(jnp.int32)

        return (key, theta_new, logp_new, accepted_new), theta_new

    # --- Initialize state ---
    logp0 = log_posterior(theta0)
    carry0 = (key, theta0, logp0, 0)

    # --- Run full chain ---
    total_steps = burn_in + num_samples * thin
    (key_final, theta_final, logp_final, accepted), chain = jax.lax.scan(
        mcmc_step, carry0, xs=None, length=total_steps
    )

    # --- Subsample post-burn-in ---
    keep_idx = jnp.arange(total_steps)[burn_in::thin]
    samples = chain[keep_idx]
    accept_rate = accepted / total_steps

    return samples, accept_rate

def run_multiple_mcmc_chains_jax(
    key: jax.Array,
    theta0_all: jnp.ndarray,
    prior: dict,
    log_likelihood_fn,
    num_samples: int = 2000,
    burn_in: int = 500,
    thin: int = 1,
    proposal_std: jnp.ndarray = None,
    proposal_fn=propose
):
    """
    Run multiple MCMC chains in parallel using JAX vmap.

    Parameters
    ----------
    key : PRNGKey
        RNG key to seed all chains
    theta0_all : array, shape (N_chains, D)
        Initial parameter vectors for all chains
    prior : dict
        Prior dictionary with mean/std/bounds
    log_likelihood_fn : callable
        Log-likelihood function of theta
    num_samples : int
        Number of samples to draw per chain
    burn_in : int
        Burn-in steps per chain
    thin : int
        Keep one sample every 'thin' steps
    proposal_std : array or scalar
        Proposal standard deviation(s)

    Returns
    -------
    samples_all : array, shape (N_chains, N_samples, D)
        Posterior samples for each chain
    accept_rates : array, shape (N_chains,)
        Acceptance rates
    """

    n_chains = theta0_all.shape[0]

    # Split master key into one key per chain
    keys = jax.random.split(key, n_chains)

    # Vectorize the single-chain sampler over keys and initial thetas
    batched_run = jax.vmap(
        lambda k, th0: run_mcmc_jax(
            key=k,
            theta0=th0,
            prior=prior,
            log_likelihood_fn=log_likelihood_fn,
            num_samples=num_samples,
            burn_in=burn_in,
            thin=thin,
            proposal_std=proposal_std,
            proposal_fn=proposal_fn
        ),
        in_axes=(0, 0),
        out_axes=(0, 0),
    )

    samples_all, accept_rates = batched_run(keys, theta0_all)

    # Compute global posterior mean and std
    all_samples_flat = samples_all.reshape(-1, samples_all.shape[-1])
    post_mean = jnp.mean(all_samples_flat, axis=0)
    post_std = jnp.std(all_samples_flat, axis=0)

    return samples_all, accept_rates, post_mean, post_std


def summarize_mcmc(samples, true_theta=None, param_names=None):
    """
    Plot histograms of marginals and report posterior stats.

    Parameters
    ----------
    samples : ndarray, shape (N_samples, D)
        MCMC posterior samples
    true_theta : array-like, optional
        True parameter vector for comparison
    param_names : list of str, optional
        Names of the parameters (len = D)
    """
    n_params = samples.shape[1]
    fig, axes = plt.subplots(nrows=2, ncols=(n_params + 1) // 2,
                             figsize=(18, 6))
    axes = axes.flatten()

    print("\n=== Posterior Summary ===")
    for i in range(n_params):
        s = samples[:, i]
        mean = np.mean(s)
        std = np.std(s)
        name = param_names[i] if param_names else f"θ[{i}]"

        print(f"{name:>5}: {mean:8.3f} ± {std:6.3f}", end="")
        if true_theta is not None:
            error = mean - true_theta[i]
            print(f"   (true: {true_theta[i]:.3f}, error: {error:.3f})")
        else:
            print()

        # Plot histogram
        ax = axes[i]
        ax.hist(s, bins=30, color="steelblue", edgecolor="black", alpha=0.8)
        ax.set_title(f"{name}")
        ax.axvline(mean, color="black", linestyle="--", label="mean")
        if true_theta is not None:
            ax.axvline(true_theta[i], color="red", linestyle=":", label="true")
        ax.legend()

    # Hide any unused axes
    for j in range(n_params, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.suptitle("Posterior Marginal Distributions", fontsize=16)
    plt.subplots_adjust(top=0.85)
    plt.show(block = True)


def plot_mcmc_error_distribution(results):
    errors = [r['error_norm'] for r in results]
    plt.figure(figsize=(8, 4))
    plt.hist(errors, bins=15, color='cornflowerblue', edgecolor='black')
    plt.axvline(x=5.0, linestyle='--', color='black', label="Success threshold")
    plt.xlabel("‖posterior mean - true θ‖")
    plt.ylabel("Count")
    plt.title("Distribution of posterior error norms over initial guesses")
    plt.legend()
    plt.tight_layout()
    plt.show()

    success_rate = sum(r['success'] for r in results) / len(results)
    print(f"\nSuccess rate (error < 5.0): {success_rate:.1%}")

def print_mcmc_result_table(results, k_true, param_names=None, max_rows=5):
    """
    Print table comparing true θ, initial guess, and posterior mean.
    """
    param_names = param_names or [f"θ[{i}]" for i in range(len(k_true))]
    print("\n=== Comparison Table: True θ vs Init θ vs Posterior Mean ===\n")

    rows_to_show = results[:max_rows]
    for i, r in enumerate(rows_to_show):
        print(f"--- Run {i+1} ---")
        print(f"{'Param':>6} | {'True':>10} | {'Initial':>10} | {'Posterior':>10} | {'Error':>10}")
        print("-" * 56)
        for j, name in enumerate(param_names):
            true_val = k_true[j]
            init_val = r['init_theta'][j]
            post_val = r['mean_theta'][j]
            err = post_val - true_val
            print(f"{name:>6} | {true_val:10.3f} | {init_val:10.3f} | {post_val:10.3f} | {err:10.3f}")
        print(f"{'‖error‖':>6} | {'':>10} | {'':>10} | {'':>10} | {r['error_norm']:10.3f}")
        print(f"{'accept':>6} | {'':>10} | {'':>10} | {'':>10} | {r['accept_rate']*100:9.1f}%")
        print()

def plot_pairwise(samples, param_names=None, true_theta=None, title="Pairwise Posterior Distributions"):
    """
    Plot pairwise relationships between posterior samples using a corner-style plot.

    Parameters
    ----------
    samples : ndarray, shape (N_samples, D)
        Posterior samples.
    param_names : list of str, optional
        List of parameter names.
    true_theta : array-like, optional
        True parameter values to overlay as red lines.
    title : str
        Title for the figure.
    """
    if param_names is None:
        param_names = [f"θ[{i}]" for i in range(samples.shape[1])]

    df = pd.DataFrame(samples, columns=param_names)

    # Plot using seaborn
    sns.set_theme(style="ticks", font_scale=1.2)
    g = sns.pairplot(df, diag_kind="kde", corner=True, plot_kws={"s": 8, "alpha": 0.5}, height=2.5)

    # Add vertical/horizontal lines for true values
    if true_theta is not None:
        for i, name in enumerate(param_names):
            if name in df.columns:
                ax = g.axes[i][i]
                ax.axvline(true_theta[i], color='red', linestyle='--', linewidth=1)

        for i in range(len(param_names)):
            for j in range(i):
                ax = g.axes[i][j]
                ax.axvline(true_theta[j], color='red', linestyle='--', linewidth=1)
                ax.axhline(true_theta[i], color='red', linestyle='--', linewidth=1)

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95, bottom=0.08)  # <-- this is the key line
    plt.show(block=True)

def plot_trace(samples, param_names=None):
    N, D = samples.shape
    fig, axes = plt.subplots(D, 1, figsize=(12, 2.5 * D), sharex=True)
    for i in range(D):
        ax = axes[i]
        ax.plot(samples[:, i], lw=0.5, alpha=0.7)
        name = param_names[i] if param_names else f"θ[{i}]"
        ax.set_ylabel(name)
    plt.xlabel("Sample index")
    plt.tight_layout()
    plt.show(block=True)
   

def main():
    pass

if __name__ == "__main__":
    main()