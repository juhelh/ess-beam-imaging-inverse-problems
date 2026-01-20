import numpy as np
import jax.numpy as jnp
from tqdm import tqdm

# === Updated imports for 10D model ===
from minimization.solve_minimization_10D import (
    estimate_parameters_BFGS,
    visualize_estimation_result,
    simulate_image,
    report_estimation_result
)
from minimization.visualize_simulations import visualize_image
from configuration.config_grid import X, Y  # grid is unchanged


def sample_gaussian_with_rejection(rng, mean, std, lower, upper, max_trials=5000):
    """
    Draw a single 10D vector from a Gaussian and reject if outside bounds.
    """
    for _ in range(max_trials):
        x = rng.normal(mean, std)
        if np.all(x >= lower) and np.all(x <= upper):
            return x
    raise RuntimeError("Gaussian rejection sampling failed: too many rejections.")

def bin_image_and_grid(I, X, Y, factor=2):
    """
    Downsample image I and grid (X, Y) by a given binning factor using average pooling.
    Works for arbitrary sizes by trimming before reshape.
    """

    def bin_array(A):
        A = np.array(A)  # ensure reshape-safe
        H, W = A.shape

        H_new = H // factor
        W_new = W // factor

        A_trim = A[:H_new*factor, :W_new*factor]
        A_bin = A_trim.reshape(H_new, factor, W_new, factor).mean(axis=(1, 3))

        return jnp.array(A_bin)

    I_binned = bin_array(I)
    X_binned = bin_array(X)
    Y_binned = bin_array(Y)

    return I_binned, X_binned, Y_binned

def estimate_parameters_for_image(
    I_obs,
    X,
    Y,
    t_vals,
    param_bounds,
    k_true,
    num_restarts=10,
    rng=np.random.default_rng(0),
    verbose=False,
    bin_factor=4,
    final_refinement=True,
    final_maxiter=200
):
    lower, upper = param_bounds
    dim = lower.shape[0]

    # --- Light binning to reduce memory ---
    if bin_factor > 1:
        I_obs_binned, Xb, Yb = bin_image_and_grid(I_obs, X, Y, factor=bin_factor)
    else:
        I_obs_binned, Xb, Yb = I_obs, X, Y

    # visualize_image(I_obs_binned, Xb, Yb)

    k_nominal = np.array([
        60.0, 20.0, 13.5, 5.05,
        0.0, 0.0,
        39.55, 29.05,
        0.0, 0.0
    ], dtype=float)

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
        np.pi,
        np.pi
    ], dtype=float)

    # --- Multistart coarse search ---
    results = []
    for i in range(num_restarts):
        print("restart: ", i)
        k0 = rng.uniform(lower, upper, size=(dim,))
        
        # # Gaussian initialization
        # k0 = sample_gaussian_with_rejection(
        # rng,
        # mean=k_nominal,
        # std=std_vec,
        # lower=lower,
        # upper=upper
        # )

        # # Override phases with uniform sampling
        # k0[8] = rng.uniform(-np.pi, np.pi)
        # k0[9] = rng.uniform(-np.pi, np.pi)

        result = estimate_parameters_BFGS(
            I_obs_binned, Xb, Yb, t_vals, k0,
            maxiter=50,
            bounds=param_bounds,
            verbose=verbose
        )
        results.append(result)

    # --- Select best coarse result ---
    best_result = min(results, key=lambda r: float(r.fun))
    best_k = best_result.x

    # --- Optional refinement step ---
    if final_refinement and bin_factor > 1:
        refined = estimate_parameters_BFGS(
            I_obs_binned, Xb, Yb, t_vals, best_k,
            maxiter=final_maxiter,
            bounds=param_bounds,
            verbose=True
        )
        best_result = refined
        best_k = refined.x

    # --- Diagnostics ---
    report_estimation_result(best_result, k_true)
    #visualize_estimation_result(I_obs, X, Y, t_vals, best_k)

    return best_k, best_result.fun


# -----------------------------------------------------------
# Run estimation on entire test set
# -----------------------------------------------------------
def run_estimation_on_test_set(
    I_all,
    T_all,
    param_bounds,
    k_true_all,
    max_num_restarts,
    rng=np.random.default_rng(0)
):
    estimated_ks = []
    losses = []

    # geometrically reduce # of restarts for long pulses
    num_restarts = np.round(
        np.geomspace(max_num_restarts, 1, num=len(T_all))
    ).astype(int)

    # Enforce a minimum number of restarts
    MIN_RESTARTS = 200
    num_restarts = np.maximum(num_restarts, MIN_RESTARTS)

    for i in tqdm(range(len(I_all)), desc="Estimating parameters for test set"):
        I_obs = I_all[i]
        T = T_all[i]
        t_vals = jnp.arange(0, T, 1e-6)  # dt = 1 μs, or replace with your dt

        k_hat, loss = estimate_parameters_for_image(
            I_obs,
            X, Y,
            t_vals,
            param_bounds,
            k_true_all[i],
            num_restarts=num_restarts[i],
            rng=rng,
            verbose=False
        )

        estimated_ks.append(k_hat)
        losses.append(loss)

    return jnp.stack(estimated_ks), jnp.array(losses)


# -----------------------------------------------------------
# Main
# -----------------------------------------------------------
def main():
    # Load new 10D test set
    data = np.load("test_all_algorithms/test_set_10D_uniform.npz")
    I_all = data["I_all"]
    T_all = data["T_all"]
    k_true_all = data["k_all"]

    # --- 10D parameter bounds ---
    lower = np.array([0, 0, 2, 2, -20, -20, 20, 20, -np.pi, -np.pi], dtype=float)
    upper = np.array([65, 25, 20, 20, 20, 20, 50, 50,  np.pi,  np.pi], dtype=float)
    param_bounds = (lower, upper)

    k_nominal = np.array([
        60.0, 20.0, 13.5, 5.05,
        0.0, 0.0,
        39.55, 29.05,
        0.0, 0.0
    ], dtype=float)

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

    # Run estimator
    estimated_ks, losses = run_estimation_on_test_set(
        I_all, T_all, param_bounds, k_true_all,
        max_num_restarts=200
    )

    print("\n=== Summary (10D estimation) ===")
    for i, (k, loss) in enumerate(zip(estimated_ks, losses)):
        print(f"Sample {i:02d}: loss = {loss:.4e}, estimated k = {np.round(k, 2)}")

    np.savez(
        "test_all_algorithms/estimated_results_10D.npz",
        estimated_ks=estimated_ks,
        losses=losses
    )


if __name__ == "__main__":
    main()