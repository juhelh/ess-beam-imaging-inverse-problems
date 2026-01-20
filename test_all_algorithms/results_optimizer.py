import numpy as np
import jax.numpy as jnp
from typing import Sequence

# === PARAMETER NAMES FOR 10D ===
PARAM_NAMES = ["Ax", "Ay", "sigx", "sigy", "cx", "cy", "fx", "fy", "phix", "phiy"]

# === If you use visualize_estimation_result from 10D version ===
from configuration.config_grid import X, Y
from minimization.solve_minimization_10D import visualize_estimation_result


def _angle_error(est, true):
    """
    Circular error for angles in radians.
    Returns wrapped difference in degrees and percent of π.
    """
    diff = est - true
    wrapped = (diff + jnp.pi) % (2 * jnp.pi) - jnp.pi
    deg = wrapped * (180.0 / jnp.pi)
    percent_pi = 100.0 * (wrapped / jnp.pi)   # ±100% corresponds to ±π
    return deg, percent_pi

def _relative_error(est, true, eps=1e-8):
    """
    Relative error: (est - true)/true.
    Works for both scalars and arrays.
    """
    est  = jnp.asarray(est)
    true = jnp.asarray(true)

    mask = jnp.abs(true) > eps
    rel = jnp.zeros_like(true)

    rel_masked = (est - true) / true
    rel = rel.at[mask].set(rel_masked[mask])

    return rel


def summarize_results(
    estimated_ks: jnp.ndarray,
    k_true_all: jnp.ndarray,
    rational_flags=None,
):
    assert estimated_ks.shape == k_true_all.shape

    if rational_flags is not None:
        rational_flags = np.array(rational_flags, dtype=bool)
        irrational_idx = np.where(~rational_flags)[0]
        rational_idx   = np.where(rational_flags)[0]
    else:
        irrational_idx = np.arange(len(estimated_ks))
        rational_idx   = np.array([], dtype=int)

    def print_param_stats(name: str, idx: np.ndarray):
        if len(idx) == 0:
            print(f"\n{name} (no samples)")
            return

        print(f"\n{name} (n={len(idx)}):")

        for p in range(estimated_ks.shape[1]):
            est = estimated_ks[idx, p]
            true = k_true_all[idx, p]
            pname = PARAM_NAMES[p]

            if pname in ("phix", "phiy"):
                # ========= ANGLE PARAMETERS ===========
                # circular absolute errors (rad)
                ang_errs = jnp.array([
                    circular_error_rad(e, t) for e, t in zip(est, true)
                ])

                # RMSE and median in radians
                abs_rmse = float(jnp.sqrt(jnp.mean(ang_errs**2)))
                abs_med  = float(jnp.median(jnp.abs(ang_errs)))

                # Convert RMSE from mixed rad errors in summarize_results to rad
                print(f"  {pname:<6s}: "
                      f"Abs-RMSE = {abs_rmse:7.3f} rad, "
                      f"median = {abs_med:7.3f} rad")

            else:
                # ========= NON-ANGLE PARAMETERS ===========
                abs_err = est - true
                abs_rmse = float(jnp.sqrt(jnp.mean(abs_err**2)))
                abs_median = float(jnp.median(jnp.abs(abs_err)))

                # Relative error (%)
                rel = _relative_error(est, true) * 100.0
                rel_rmse = float(jnp.sqrt(jnp.mean(rel**2)))
                rel_median = float(jnp.median(jnp.abs(rel)))

                print(
                    f"  {pname:<6s}: "
                    f"Rel-RMSE = {rel_rmse:7.3f} %, "
                    f"median = {rel_median:7.3f} %,   "
                    f"Abs-RMSE = {abs_rmse:7.3f}, "
                    f"median abs = {abs_median:7.3f}"
                )

    print_param_stats("All samples", np.arange(len(estimated_ks)))
    print_param_stats("Rational freq ratio", rational_idx)
    print_param_stats("Irrational freq ratio", irrational_idx)


ANGLE_INDICES = [8, 9]   # phix, phiy indices in your 10D vector

def circular_error_rad(est, true):
    """Return wrapped circular error in radians, in [-pi, pi]."""
    diff = est - true
    return jnp.arctan2(jnp.sin(diff), jnp.cos(diff))


def summarize_all_results_by_sample(
    estimated_ks: jnp.ndarray,
    k_true_all: jnp.ndarray,
    T_all: np.ndarray,
    param_names: list[str],
    losses: np.ndarray | None = None,
):
    print("\n=== Absolute estimation errors for all test images ===")
    print("Each row shows: pulse length T (ms), then *absolute errors* (phis in rad)\n")

    header = (
        "Pulse T (ms) |  "
        + "  ".join(f"{p:>10s}" for p in param_names)
    )
    if losses is not None:
        header += "  |   Loss"
    print(header)
    print("-" * len(header))

    for i in range(len(estimated_ks)):
        pulse_ms = float(T_all[i]) * 1e3

        est  = estimated_ks[i]
        true = k_true_all[i]

        errors = []
        for j in range(len(param_names)):
            if j in ANGLE_INDICES:
                # circular absolute error in radians
                e = float(circular_error_rad(est[j], true[j]))
                errors.append(f"{e:+10.3f}")
            else:
                # absolute error in parameter units
                e = float(est[j] - true[j])
                errors.append(f"{e:+10.3f}")

        line = f"{pulse_ms:12.3f} |  " + "  ".join(errors)

        if losses is not None:
            line += f"  | {float(losses[i]):10.5f}"

        print(line)

# ===============================================================
#  Main script
# ===============================================================
def main():
    # === Load estimated results (10D) ===
    results = np.load("test_all_algorithms/estimated_results_10D.npz")
    estimated_ks = results["estimated_ks"]
    losses = results["losses"]

    # === Load test set (true ks) ===
    test_data = np.load("test_all_algorithms/test_set_10D_uniform.npz")
    I_all = test_data["I_all"]
    T_all = test_data["T_all"]
    k_true_all = test_data["k_all"]

    # print(k_true_all[3])
    # I_obs=I_all[3]
    # T = float(T_all[3])
    # t_vals = jnp.arange(0, T, 1e-6)
    # visualize_estimation_result(
    # I_obs=I_all[3],
    # X=X,
    # Y=Y,
    # t_vals=t_vals,
    # k_est=estimated_ks[3]
    # )

    print("\nIndex : A_y")
    for i, val in enumerate(k_true_all[:, 1]):
        print(f"{i:3d} : {val:.6f}")

    rational_flags = test_data["rational_flags"]  # all False in your new set

    # === Summaries ===
    summarize_results(estimated_ks, k_true_all, rational_flags)

    summarize_all_results_by_sample(
        estimated_ks=estimated_ks,
        k_true_all=k_true_all,
        T_all=T_all,
        param_names=PARAM_NAMES,
        losses=losses,
    )

    # === Identify worst case sample ===
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