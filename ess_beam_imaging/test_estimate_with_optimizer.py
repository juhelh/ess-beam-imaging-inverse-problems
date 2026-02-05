import numpy as np
import jax.numpy as jnp

from ess_beam_imaging.estimate import estimate_with_optimizer
from minimization.solve_minimization_10D import visualize_estimation_result 
import os


def load_image(path):
    """
    Load a single beam snapshot saved from EPICS acquisition.

    Returns
    -------
    image : ndarray (H, W), float32
    meta  : dict with fx, fy, exposure_time
    """

    data = np.load(path)

    image = data["image"].astype(np.float32)

    meta = {
        "fx": float(data["fx"]),              # kHz
        "fy": float(data["fy"]),              # kHz
        "exposure_time": float(data["exposure_time"]),
    }

    print("\n=== Loaded custom snapshot metadata ===")
    for k, v in meta.items():
        print(f"{k:>14s} = {v}")

    return image, meta

def test_real_image():
    """
    Run the full estimation pipeline on one recorded beam image.

    Loads a snapshot from disk, passes measured machine parameters
    (frequencies and pulse duration) to the estimator, applies a small
    regularization on Ax and Ay, and visualizes the fitted result
    against the observed image.
    """
    # Reference a saved image
    path = os.path.join(
    "beam_images",
    "beam_snapshot_1766149115.npz"
    )

    image, meta = load_image(path)
    ms_conversion = 1_000_000 # NOTE: This assumes the exposure time is always on the form it was here
    pulse_duration_ms = meta["exposure_time"]/ms_conversion

    # Known/measured information (in addition to image)
    meta_for_est = {
        "fx": meta["fx"],
        "fy": meta["fy"],
        "T": pulse_duration_ms,
    }

    # User choices
    config = {
    "debug": True,
    "regularization": {
        "params": {"Ax": 5e-4, "Ay": 5e-4}
    },
    "multistart": {
        "type": "gaussian",      # or "gaussian" later
        "n_starts": 50,
    }
}

    ks, losses, I_obs_cropped, t_vals, X, Y = estimate_with_optimizer(
        image,
        meta=meta_for_est,
        config=config,
    )

    # How many solutions to inspect
    n_show = min(20, len(ks))

    names = ["Ax", "Ay", "sigx", "sigy", "cx", "cy", "fx", "fy", "phix", "phiy"]

    print("\n=== Top solutions (sorted by loss) ===")
    for i in range(n_show):
        print(f"\n--- Solution {i+1} ---")
        print(f"Loss = {losses[i]:.6e}")
        for name, val in zip(names, ks[i]):
            print(f"{name:>6s} = {val:10.4f}")

        visualize_estimation_result(
            I_obs=np.array(I_obs_cropped),
            X=np.array(X),
            Y=np.array(Y),
            t_vals=np.array(t_vals),
            k_est=ks[i],
        )

def main():
    test_real_image()

if __name__ == "__main__":
    main()