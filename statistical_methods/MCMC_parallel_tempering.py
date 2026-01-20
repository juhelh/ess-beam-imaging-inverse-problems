from statistical_methods.MCMC_runner_speedup import *
from statistical_methods.MCMC_tests_speedup import *
from statistical_methods.MCMC_with_gradient import mala_propose_wrapper


# -----------------------------------------------------------------------------
# Fully‑vectorised Parallel Tempering (Replica–Exchange MCMC)
# -----------------------------------------------------------------------------
#  • all replicas share one JIT‑compiled Metropolis kernel (vmap)
#  • swaps performed every `steps_per_swap` iterations
#  • no dynamic Python logic inside jitted code (string‑free priors, no dynamic
#    indexing) so it compiles once and runs fast on CPU/GPU/TPU
# -----------------------------------------------------------------------------


def run_parallel_tempering(
    *,
    key: jax.Array,
    theta0_all: jnp.ndarray,
    prior: dict,
    base_loglike_fn,
    betas: jnp.ndarray,
    steps_per_swap: int = 10,
    n_swap_intervals: int = 2000,
    proposal_std: float | jnp.ndarray = None,
    proposal_fn = propose,
    burn_in_steps: int = 1000,
):
    """
    Parallel tempering MCMC in JAX using random-walk Metropolis proposals.
    Gradient-free implementation suitable for thesis presentation.
    """
    n_chains, D = theta0_all.shape
    assert betas.shape[0] == n_chains, "betas length must equal number of chains"

    # --------------------------------------------------------------
    # Build per-chain proposal standard deviations
    # --------------------------------------------------------------
    if proposal_std is None:
        raise ValueError("proposal_std must be provided")

    proposal_std = jnp.asarray(proposal_std)

    if proposal_std.ndim == 1:
        # Base proposal for cold chain, scaled with temperature
        alpha = 0.5
        proposal_std_chain = proposal_std * betas[:, None] ** (-alpha)
    elif proposal_std.ndim == 2:
        assert proposal_std.shape == (n_chains, D), (
            f"proposal_std has shape {proposal_std.shape}, expected ({n_chains}, {D})"
        )
        proposal_std_chain = proposal_std
    else:
        raise ValueError(
            f"proposal_std must be 1D (D,) or 2D (n_chains, D), got shape {proposal_std.shape}"
        )

    # --------------------------------------------------------------
    # Build log-prior (string-free for JIT)
    # --------------------------------------------------------------
    if prior["type"] == "gaussian":
        log_prior_fn = lambda th: log_gaussian_prior(th, prior["mean"], prior["std"])
    elif prior["type"] == "uniform":
        log_prior_fn = lambda th: log_uniform_prior(th, prior["lower"], prior["upper"])
    else:
        raise ValueError("Unsupported prior type: " + prior["type"])

    # --------------------------------------------------------------
    # Untempered log-posterior (needed for swaps)
    # --------------------------------------------------------------
    def untempered_log_post(th):
        return log_prior_fn(th) + base_loglike_fn(th)

    # --------------------------------------------------------------
    # JIT-compiled MH step for all chains
    # --------------------------------------------------------------
    @jax.jit
    def mh_step_all_chains(keys, thetas):
        """Advance all chains one Metropolis–Hastings step."""

        def mh_step_one_chain(k, th, beta, std_vec):
            # Random-walk proposal with bounds
            k, th_prop = proposal_fn(
                k, th, std_vec, prior["lower"], prior["upper"]
            )

            log_post      = log_prior_fn(th)      + beta * base_loglike_fn(th)
            log_post_prop = log_prior_fn(th_prop) + beta * base_loglike_fn(th_prop)

            log_alpha = log_post_prop - log_post

            k, sub = jax.random.split(k)
            accept = jnp.log(jax.random.uniform(sub)) < log_alpha
            th_new = jnp.where(accept, th_prop, th)

            return k, th_new, accept

        keys_out, thetas_out, accepts = jax.vmap(mh_step_one_chain)(
            keys,
            thetas,
            betas,
            proposal_std_chain,
        )

        return keys_out, thetas_out, accepts


    # ------------------------------------------------------------------
    # 4.  Swap samples between chains (adjacent pairs, even/odd alternation)
    # ------------------------------------------------------------------
    def swap_samples(thetas, key, round_idx):

        even = (round_idx % 2) == 0

        if even:
            pairs_i = jnp.arange(0, n_chains - 1, 2)   # [0,2,4,...]
        else:
            pairs_i = jnp.arange(1, n_chains - 1, 2)   # [1,3,5,...]

        pairs_j = pairs_i + 1

        # Extract parameters for chains
        thetas_i = thetas[pairs_i]
        thetas_j = thetas[pairs_j]

        betas_i  = betas[pairs_i]
        betas_j  = betas[pairs_j]

        # Compute untempered log-posteriors
        logp_i = jax.vmap(untempered_log_post)(thetas_i)
        logp_j = jax.vmap(untempered_log_post)(thetas_j)

        # In swap ratio: delta = (logp_j - logp_i) * (beta_i - beta_j)
        deltas = (logp_j - logp_i) * (betas_i - betas_j)

        # Draw uniforms
        key, sub = jax.random.split(key)
        uniforms = jax.random.uniform(sub, deltas.shape)

        # Accept or reject swaps
        accept = jnp.log(uniforms) < deltas
        mask = accept[:, None]

        # Do the swap
        new_i = jnp.where(mask, thetas_j, thetas_i)
        new_j = jnp.where(mask, thetas_i, thetas_j)

        # Update main array
        thetas = thetas.at[pairs_i].set(new_i)
        thetas = thetas.at[pairs_j].set(new_j)

        return thetas, key, accept, pairs_i
        
    # ------------------------------------------------------------------
    # 5.  Main PT loop --------------------------------------------------
    # ------------------------------------------------------------------
    # Initialize keys for all chains
    key, sub = jax.random.split(key)
    keys = jax.random.split(sub, n_chains)

    thetas = theta0_all
    swap_rates = []
    cold_trace = []
    acceptance_history = []

    n_pairs_total = n_chains - 1  # All adjacent pairs
    swap_counts = jnp.zeros(n_pairs_total)
    swap_totals = jnp.zeros(n_pairs_total)

    round_idx = 0

    # ------------------------------------------------------------------
    #  Optional: Burn-in without swaps
    # ------------------------------------------------------------------
    if burn_in_steps > 0:
        print(f"Running burn-in for {burn_in_steps} steps (no swaps)...")
        for _ in range(burn_in_steps):
            keys, thetas, _ = mh_step_all_chains(keys, thetas)

    print("Burn-in done!")        

    for _ in range(n_swap_intervals):
        accepts_per_swap = []
        for _ in range(steps_per_swap):
            keys, thetas, accepts = mh_step_all_chains(keys, thetas)
            accepts_per_swap.append(accepts)
            cold_trace.append(thetas[0])  # cold chain trace

        # Convert list of (n_chains,) → array (steps_per_swap, n_chains)
        accepts_per_swap = jnp.stack(accepts_per_swap)
        mean_accepts_per_chain = jnp.mean(accepts_per_swap, axis=0)  # (n_chains,)
        acceptance_history.append(mean_accepts_per_chain)

        # --- Swap step ---
        key, subkey = jax.random.split(key)
        thetas, key, swap_accepts, pairs_i = swap_samples(thetas, subkey, round_idx)
        swap_rates.append(jnp.mean(swap_accepts))

        # update per-pair statistics
        swap_counts = swap_counts.at[pairs_i].add(1)
        swap_totals = swap_totals.at[pairs_i].add(swap_accepts.astype(jnp.float32))

        round_idx += 1

    # Stack and average over swap intervals
    acceptance_history = jnp.stack(acceptance_history)  # (n_swap_intervals, n_chains)
    mean_acceptance_per_chain = jnp.mean(acceptance_history, axis=0)  # (n_chains,)

    swap_accept_means = swap_totals / (swap_counts + 1e-8)

    samples_cold = jnp.stack(cold_trace)
    mean_swap_rate = float(jnp.mean(jnp.array(swap_rates)))

    return samples_cold, mean_swap_rate, mean_acceptance_per_chain, swap_accept_means


# JIT-compile the whole tempering loop.
run_parallel_tempering_jitted = jax.jit(
    run_parallel_tempering,
    static_argnames=("prior", "proposal_fn", "base_loglike_fn"),
)

def main():
        # ------------------------------------------------------------
        # 1.  Problem set-up
        # ------------------------------------------------------------
        print("\n=== Parallel–Tempering demo (JAX) ===")
        seed = 4
        np_rng = np.random.default_rng(seed)
        key    = jax.random.PRNGKey(seed)

        X, Y, t_vals, *_ = make_configuration(pulse_duration_ms=0.5)

        lower = jnp.array([0, 0, 2, 2, -20, -20, 25, 20], dtype=jnp.float32)
        upper = jnp.array([65, 25, 20, 20, 20, 20, 55, 40], dtype=jnp.float32)

        k_true = jnp.array(np_rng.uniform(low=lower, high=upper))
        print("True θ:", k_true)

        I_clean = simulate_image(X, Y, t_vals, k_true)
        I_obs   = add_poisson(I_clean, scale=20_000.)
        I_obs, X, Y = bin_image_and_grid(I_obs, X, Y, factor=8)
        I_obs  /= jnp.max(I_obs)                      # Normalize - in likelihood it is however scaled again!

        visualize_image(I_obs, X, Y)


        mean = (lower + upper) / 2
        std = jnp.full_like(lower, 10.0)
        
        prior_gaussian = dict(
            type  = "gaussian",
            mean  = mean,
            std   = std,
            lower = lower,
            upper = upper,
        )

        prior_uniform = dict(
            type  = "uniform",
            lower = lower,
            upper = upper,
        )

        # Choose priors
        prior = prior_uniform

        # How long are proposal steps should be
        proposal_std = 0.05 * (upper - lower)

        # ------------------------------------------------------------
        # 2.  Temperature ladder and initial states
        # ------------------------------------------------------------
        n_chains = 10
        betas = jnp.geomspace(1.0, 0.01, num=n_chains) # Goes from colder to hotter (beta is 1/T, so T: 1 to inf)
        n_chains = len(betas)

        key, sub = jax.random.split(key) # sub is the key to be used in computation now, key is for later computations
        _, theta0_all = jax.vmap(lambda k: sample_from_prior(prior, k))(jax.random.split(sub, n_chains)) # Split sub key into n_chains further branches to get different theta0 for each chain  

        # --- Log-posterior and gradient --- NOTE: not used right now!
        log_prior_fn = lambda th: log_gaussian_prior(th, mean, std)
        log_likelihood_fn = lambda th: log_poisson_likelihood(th, I_obs, X, Y, t_vals)
        scale_transform = 100.0  # Try different ones
        log_post_fn = lambda th: (log_prior_fn(th) + log_likelihood_fn(th)) / scale_transform
        step_size = 0.01
        proposal_fn = mala_propose_wrapper(log_post_fn, step_size)

        # ------------------------------------------------------------
        # 3.  Run parallel tempering
        # ------------------------------------------------------------
        base_ll = lambda th: log_poisson_likelihood(th, I_obs, X, Y, t_vals)

        cold_samples, swap_rate = run_parallel_tempering(
            key            = key,
            theta0_all     = theta0_all,
            prior          = prior,
            base_loglike_fn= base_ll,
            betas          = betas,
            steps_per_swap = 10,           # How long we go between swaps
            n_swap_intervals = 100,        # How many times we swap in total
            proposal_std   = proposal_std
        )

        print(f"\nMean swap-accept rate: {swap_rate:.2%}") # Important diagnostic, should be like 20 percent
        print("Posterior mean (cold chain):", jnp.mean(cold_samples, axis=0))
        print("Posterior std  (cold chain):", jnp.std (cold_samples, axis=0))

        # ------------------------------------------------------------
        # See posterior distribution and compare image to observed one
        # ------------------------------------------------------------
        summary_and_visualization(
            cold_samples,
            k_true,
            I_obs,
            X,
            Y,
            t_vals,
            PARAM_NAMES,
            loss_function,
            simulate_image,
            title="Parallel Tempering MCMC Posterior (cold chain)"
        )

if __name__ == '__main__':
    main()



# def run_parallel_tempering(
#     *,
#     key: jax.Array,
#     theta0_all: jnp.ndarray,
#     prior: dict,
#     base_loglike_fn,
#     betas: jnp.ndarray,
#     steps_per_swap: int = 50,
#     n_swap_intervals: int = 200,
#     proposal_std: float | jnp.ndarray = None,
#     proposal_fn = propose,
#     burn_in_steps: int = 1000,  # NEW
# ):
#     """
#     Parallel tempering in JAX, vectorized for full parallelism.
#     """
#     n_chains, D = theta0_all.shape
#     assert betas.shape[0] == n_chains, "betas length must equal number of chains"
#     # ------------------------------------------------------------------
#     # 1.  Build string‑free log‑prior (JIT can't deal with Python objects like dicts/strings)
#     # ------------------------------------------------------------------
#     if prior["type"] == "gaussian":
#         log_prior_fn = lambda th: log_gaussian_prior(th, prior["mean"], prior["std"])
#     elif prior["type"] == "uniform":
#         log_prior_fn = lambda th: log_uniform_prior(th, prior["lower"], prior["upper"])
#     else:
#         raise ValueError("Unsupported prior type: " + prior["type"])

#     # ------------------------------------------------------------------
#     # 2.  Make one log‑posterior function per temperature (beta)
#     # ------------------------------------------------------------------
#     log_post_fns = [
#         (lambda beta: (lambda th: log_prior_fn(th) + beta * base_loglike_fn(th)))(beta)
#         for beta in betas
#     ] # Gives a list of posterior functions: equivalent to looping over beta and defining a function for each one and appending to list

#     def untempered_log_post(theta):
#         return log_prior_fn(theta) + base_loglike_fn(theta)

#     # If gradient proposal is used, define the gradient functions for each beta here
#     if proposal_fn.__name__ == "mala_propose_wrapper":
#         grad_log_post_fns = [jax.grad(fn) for fn in log_post_fns]
#     else:
#         grad_log_post_fns = None

#     @jax.jit # JIT to build computation graph first run and then use that to optimize calculations on the hardware 
#     def mh_step_all_chains(keys, thetas):
#         """Takes in latest samples from all chains and advances them all one step
#         with Metropolis-Hastings logic. """

#         def mh_step_one_chain(k, th, beta, grad_fns):
#             """Takes the Metropolis-Hastings step."""
#             if grad_fns is not None:
#                 grad_log_post = grad_fns(th)
#                 k, th_prop = proposal_fn(k, th, grad_log_post, proposal_std)
#             else:
#                 k, th_prop = proposal_fn(k, th, proposal_std, prior["lower"], prior["upper"])

#             log_post      = log_prior_fn(th)      + beta * base_loglike_fn(th)
#             log_post_prop = log_prior_fn(th_prop) + beta * base_loglike_fn(th_prop)
#             log_alpha = log_post_prop - log_post

#             k, sub = jax.random.split(k)
#             accept = jnp.log(jax.random.uniform(sub)) < log_alpha
#             th_new = jnp.where(accept, th_prop, th)
#             return k, th_new, accept

#         keys_out, thetas_out, accepts = jax.vmap(mh_step_one_chain)(keys, thetas, betas, grad_log_post_fns)
#         return keys_out, thetas_out, accepts

#     # ------------------------------------------------------------------
#     # 4.  Swap samples between chains (adjacent pairs, even/odd alternation)
#     # ------------------------------------------------------------------
#     def swap_samples(thetas, keys, round_idx):
#         even = (round_idx % 2) == 0
#         idx = jnp.arange(n_chains) # Creates list with one index for each chain

#         # This is important for theoretical reasons: does not allow swapping e.g. chain 0 <-> 1 then 1 <-> 2. No overlap!
#         pairs_i = idx[:-1:2] if even else idx[1:-1:2]
#         pairs_j = idx[1::2]  if even else idx[2::2]

#         if pairs_i.size == 0:
#             return thetas, 0.0, keys  # If only two pairs, just skip swapping every other round

#         thetas_i, thetas_j = thetas[pairs_i], thetas[pairs_j] # Picks out the current sample at chain i and j
#         betas_i, betas_j = betas[pairs_i], betas[pairs_j] # Picks out the corresponding temperings

#         # # Log-posteriors for the i-chains and j-chains, all evaluated at current sample. Note that it should be the UNTEMPERED posteriors
#         # log_posteriors_i = jnp.stack([f(th) for f, th in zip(log_post_fns, thetas_i)], axis=0)
#         # log_posteriors_j = jnp.stack([f(th) for f, th in zip(log_post_fns, thetas_j)], axis=0)


#         # deltas = (log_posteriors_j - log_posteriors_i) * (betas_i - betas_j) # Gives the log ratio of posteriors between all adjacent pairs, so n_pairs/2 elements
        
#         # # Use those keys to get uniform random numbers
#         # uniforms = jax.vmap(lambda k: jax.random.uniform(k))(swap_keys)
#         # accept = jnp.log(uniforms) < deltas


#         # key_swap used for generating random numbers, key_remain used to replace the chain keys that were consumed here
#         # keys[0] means
#         key_swap, key_remain = jax.random.split(keys[0]) # keys[0] is (10,2). Gives randomness of first batch of 10 (steps_per_swap=10)
#         swap_keys = jax.random.split(key_swap, pairs_i.size)  # Get one key per "swap pair"

#         logp_i = jax.vmap(untempered_log_post)(thetas_i)  # shape (n_pairs,)
#         logp_j = jax.vmap(untempered_log_post)(thetas_j)

#         deltas = (logp_j - logp_i) * (betas_i - betas_j)

#         uniforms = jax.random.uniform(key_swap, shape=(pairs_i.size,))
#         accept   = jnp.log(uniforms) < deltas



#         # Swap where accept is True. 
#         # m[:, None] broadcasts m to right shape
#         def do_swaps(thetas_i, thetas_j, swap_mask):
#             swap_mask_expanded = swap_mask[:, None]  # Shape (n_pairs, 1), broadcastable so that swaps can be done in parallel over the entire batch of chains
#             new_thetas_i = jnp.where(swap_mask_expanded, thetas_j, thetas_i)
#             new_thetas_j = jnp.where(swap_mask_expanded, thetas_i, thetas_j)
#             return new_thetas_i, new_thetas_j 

#         new_thetas_i, new_thetas_j = do_swaps(thetas_i, thetas_j, accept) # The new thetas for all chains
#         thetas = thetas.at[pairs_i].set(new_thetas_i) # Update the full thetas vector for i-chains 
#         thetas = thetas.at[pairs_j].set(new_thetas_j) # Update the thetas for j-chains

#         acc_rate = jnp.mean(accept.astype(jnp.float32)) # Useful diagnostic

#         return thetas, acc_rate, accept, pairs_i
    
    

#     # ------------------------------------------------------------------
#     # 5.  Main PT loop --------------------------------------------------
#     # ------------------------------------------------------------------
#     # Initialize keys for all chains
#     key, sub = jax.random.split(key)
#     keys = jax.random.split(sub, n_chains)

#     thetas = theta0_all
#     swap_rates = []
#     cold_trace = []
#     round_idx  = 0
#     acceptance_history = []
#     swap_accepts_all = []

#     n_pairs_total = n_chains - 1  # All adjacent pairs
#     swap_counts   = jnp.zeros(n_pairs_total)
#     swap_totals   = jnp.zeros(n_pairs_total)

#     # ------------------------------------------------------------------
#     #  Optional: Burn-in without swaps
#     # ------------------------------------------------------------------
#     if burn_in_steps > 0:
#         print(f"Running burn-in for {burn_in_steps} steps (no swaps)...")
#         for _ in range(burn_in_steps):
#             keys, thetas, _ = mh_step_all_chains(keys, thetas)

#     print("Burn-in done!")        

#     for _ in range(n_swap_intervals):
#         accepts_per_swap = []
#         for _ in range(steps_per_swap):
#             keys, thetas, accepts = mh_step_all_chains(keys, thetas)
#             accepts_per_swap.append(accepts)
#             cold_trace.append(thetas[0])  # Could also store full trace per chain here

#         # Convert list of (n_chains,) → array (steps_per_swap, n_chains)
#         accepts_per_swap = jnp.stack(accepts_per_swap)
#         mean_accepts_per_chain = jnp.mean(accepts_per_swap, axis=0)  # (n_chains,)
#         acceptance_history.append(mean_accepts_per_chain)

#         # Swap step
#         key, subkey = jax.random.split(key)
#         n_pairs = n_chains // 2
#         swap_keys = jax.random.split(subkey, n_pairs)
        
#         thetas, acc_rate, swap_accepts, pairs_i = swap_samples(thetas, swap_keys, round_idx)
#         swap_rates.append(acc_rate)

#         # Convert boolean mask to float (0.0 or 1.0)
#         accepted_floats = swap_accepts.astype(jnp.float32)

#         # Update swap counts and totals using segment_sum
#         swap_counts = swap_counts.at[pairs_i].add(1)
#         swap_totals = swap_totals.at[pairs_i].add(accepted_floats)
        
#         round_idx += 1

#     # Stack and average over swap intervals
#     acceptance_history = jnp.stack(acceptance_history)  # shape: (n_swap_intervals, n_chains)
#     mean_acceptance_per_chain = jnp.mean(acceptance_history, axis=0)  # shape: (n_chains,)

#     swap_accept_means = swap_totals / (swap_counts + 1e-8)  # Prevent divide-by-zero

#     samples_cold = jnp.stack(cold_trace)
#     mean_swap_rate = float(jnp.mean(jnp.array(swap_rates)))

#     return samples_cold, mean_swap_rate, mean_acceptance_per_chain, swap_accept_means

# # JIT-compile the whole tempering loop.
# run_parallel_tempering_jitted = jax.jit(
#     run_parallel_tempering,
#     static_argnames=("prior", "proposal_fn", "base_loglike_fn"),
# )
