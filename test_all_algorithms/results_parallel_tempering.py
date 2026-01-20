import numpy as np
import jax.numpy as jnp

from configuration.config_grid import X, Y
from minimization.solve_minimization_10D import visualize_estimation_result

# === 10D parameter names ===
PARAM_NAMES = ["Ax", "Ay", "sigx", "sigy", "cx", "cy", "fx", "fy", "phix", "phiy"]
ANGLE_INDICES = [8, 9]   # phix, phiy


# ===============================================================
#  Helper functions
# ===============================================================

def circular_error_rad(est, true):
    """Wrapped error between two angles, in radians, in [-pi, pi]."""
    diff = est - true
    return jnp.arctan2(jnp.sin(diff), jnp.cos(diff))


def _relative_error(est, true, eps=1e-8):
    """Relative error, safe around zero."""
    est  = jnp.asarray(est)
    true = jnp.asarray(true)
    mask = jnp.abs(true) > eps
    rel = jnp.zeros_like(true)
    rel = rel.at[mask].set((est - true) / true)[mask]
    return rel


# ===============================================================
#  Summary 1: RMSE grouped by rationality (just like optimizer version)
# ===============================================================

def summarize_results(
    estimated_ks: jnp.ndarray,
    k_true_all: jnp.ndarray,
    rational_flags=None,
):
    assert estimated_ks.shape == k_true_all.shape

    if rational_flags is not None:
        rational_flags = np.array(rational_flags, dtype=bool)
        rational_idx   = np.where(rational_flags)[0]
        irrational_idx = np.where(~rational_flags)[0]
    else:
        # If no flags → everything is "irrational"
        rational_idx   = np.array([], dtype=int)
        irrational_idx = np.arange(len(estimated_ks))

    # ---------------------------------------------------------
    # Function printing stats for one subset
    # ---------------------------------------------------------
    def print_param_stats(label: str, idx: np.ndarray):
        if len(idx) == 0:
            print(f"\n{label} (no samples)")
            return

        print(f"\n{label} (n={len(idx)}):")

        for p, name in enumerate(PARAM_NAMES):
            est  = estimated_ks[idx, p]
            true = k_true_all[idx, p]

            if p in ANGLE_INDICES:
                # circular absolute error
                ang = jnp.array([circular_error_rad(e, t) for e, t in zip(est, true)])

                abs_rmse = float(jnp.sqrt(jnp.mean(ang**2)))
                abs_median = float(jnp.median(jnp.abs(ang)))

                print(f"  {name:<6s}: "
                      f"Abs-RMSE = {abs_rmse:7.3f} rad, "
                      f"median = {abs_median:7.3f} rad")

            else:
                abs_err = est - true
                abs_rmse = float(jnp.sqrt(jnp.mean(abs_err**2)))
                abs_median = float(jnp.median(jnp.abs(abs_err)))

                rel = _relative_error(est, true) * 100.0
                rel_rmse = float(jnp.sqrt(jnp.mean(rel**2)))
                rel_median = float(jnp.median(jnp.abs(rel)))

                print(f"  {name:<6s}: "
                      f"Rel-RMSE = {rel_rmse:7.3f} %, "
                      f"median = {rel_median:7.3f} %,   "
                      f"Abs-RMSE = {abs_rmse:7.3f}, "
                      f"median abs = {abs_median:7.3f}"
                )

    # === Print three blocks: all, rational, irrational
    print_param_stats("All samples", np.arange(len(estimated_ks)))
    print_param_stats("Rational freq ratio", rational_idx)
    print_param_stats("Irrational freq ratio", irrational_idx)


# ===============================================================
#  Summary 2: Per-sample absolute errors (like optimizer version)
# ===============================================================

def summarize_all_results_by_sample(
    estimated_ks: jnp.ndarray,
    k_true_all: jnp.ndarray,
    T_all: np.ndarray,
    losses: np.ndarray | None = None,
):
    print("\n=== Absolute estimation errors for all test images ===")
    print("Each row shows: pulse length T (ms), then *absolute errors* for all 10 parameters (phis in rad)\n")

    header = "Pulse T (ms) |  " + "  ".join(f"{p:>10s}" for p in PARAM_NAMES)
    if losses is not None:
        header += "  |   Loss"
    print(header)
    print("-" * len(header))

    for i in range(len(estimated_ks)):
        T_ms = float(T_all[i]) * 1e3

        est  = estimated_ks[i]
        true = k_true_all[i]

        errors = []
        for p in range(len(PARAM_NAMES)):
            if p in ANGLE_INDICES:
                e = float(circular_error_rad(est[p], true[p]))
            else:
                e = float(est[p] - true[p])
            errors.append(f"{e:+10.3f}")

        line = f"{T_ms:12.3f} |  " + "  ".join(errors)

        if losses is not None:
            line += f"  | {float(losses[i]):10.5f}"

        print(line)


# ===============================================================
#  Main script — identical style to optimizer script
# ===============================================================

def main():
    # === Load estimated MCMC results ===
    results = np.load("test_all_algorithms/estimated_results_pt.npz")
    estimated_ks = results["estimated_ks"]
    losses = results["losses"]

    # === Load true test set ===
    data = np.load("test_all_algorithms/test_set_10D_gaussian.npz")
    I_all = data["I_all"]
    T_all = data["T_all"]
    k_true_all = data["k_all"]
    rational_flags = data["rational_flags"]

    # === High-level summaries ===
    summarize_results(estimated_ks, k_true_all, rational_flags)

    summarize_all_results_by_sample(
        estimated_ks,
        k_true_all,
        T_all,
        losses,
    )

    # === Worst-case ===
    errors = jnp.linalg.norm(estimated_ks - k_true_all, axis=1)
    worst_idx = int(jnp.argmax(errors))

    print(f"\n=== Inspecting worst estimate (index {worst_idx}) ===")
    print("True k:     ", np.round(np.array(k_true_all[worst_idx]), 3))
    print("Estimated k:", np.round(np.array(estimated_ks[worst_idx]), 3))
    print("Loss:       ", float(losses[worst_idx]))
    print("‖Error‖:     ", float(errors[worst_idx]))

    # === Plot reconstruction ===
    T = float(T_all[worst_idx])
    t_vals = jnp.arange(0, T, 1e-6)

    visualize_estimation_result(
        I_obs=I_all[worst_idx],
        X=X,
        Y=Y,
        t_vals=t_vals,
        k_est=estimated_ks[worst_idx]
    )


if __name__ == "__main__":
    main()