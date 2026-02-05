import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import jax.numpy as jnp
import jax
from jax import lax

import time

from configuration.configuration import make_configuration

from minimization.solve_minimization_10D_real import (
    simulate_image,
    visualize_estimation_result,
    estimate_parameters_BFGS,
    loss_regularized,
    loss_function,
    estimate_parameters_BFGS_jaxopt
)

# --------------------------------------------------
# Data loading
# --------------------------------------------------

def load_long_pulse():
    """
    Load raster-14-image.mat.

    Returns
    -------
    I_obs : ndarray (H,W), float32
    meta  : dict with logged machine parameters
    """
    path = os.path.join(os.path.dirname(__file__), "raster-14-image.mat")
    data = sio.loadmat(path)

    I_obs = data["im_"].astype(np.float32)

    meta = {
        "H_amp_0": float(np.array(data["H_amp_0"]).squeeze()),
        "V_amp_0": float(np.array(data["V_amp_0"]).squeeze()),
        "H_off_0": float(np.array(data["H_off_0"]).squeeze()),
        "V_off_0": float(np.array(data["V_off_0"]).squeeze()),
        "Hfreq":   float(np.array(data["Hfreq"]).squeeze()),
        "Vfreq":   float(np.array(data["Vfreq"]).squeeze()),
    }

    print("\n=== Loaded raster-14 metadata ===")
    for k, v in meta.items():
        print(f"{k:>8s} = {v}")

    return I_obs, meta

def load_short_pulse():
    """
    Load raster-14-image.mat.

    Returns
    -------
    I_obs : ndarray (H,W), float32
    meta  : dict with logged machine parameters
    """
    folder = os.path.dirname(__file__)
    path = os.path.join(folder, "data_laser_raster.mat")
    data = sio.loadmat(path)

    I_obs_all = data["im_all_roi"].astype(np.float32)
    I_obs = I_obs_all[1]

    return I_obs


# --------------------------------------------------
# Preprocessing
# --------------------------------------------------

def remove_background(I, sigma=30):
    bg = gaussian_filter(I, sigma=sigma)
    return np.clip(I - bg, 0, None)


def crop_center(img, crop_h=181, crop_w=351):
    H, W = img.shape
    cy, cx = np.unravel_index(np.argmax(img), img.shape)

    x1 = max(0, int(cx - crop_w // 2))
    x2 = min(W, int(cx + crop_w // 2))
    y1 = max(0, int(cy - crop_h // 2))
    y2 = min(H, int(cy + crop_h // 2))

    return img[y1:y2, x1:x2]


def show_image(img, title=""):
    plt.figure(figsize=(6, 4))
    plt.imshow(img, cmap="inferno", origin="lower")
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.show()


def print_k_estimate(k_est):
    names = [
        "Ax", "Ay", "sigx", "sigy",
        "cx", "cy", "fx", "fy",
        "phix", "phiy",
    ]

    print("\n=== Final parameter estimate ===")
    for n, v in zip(names, k_est):
        print(f"{n:>6s} = {v:10.4f}")

def tight_crop_bbox(img, frac=0.02, pad=10, min_h=32, min_w=32):
    """
    Compute a tight crop bounding box around pixels above a threshold.

    Parameters
    ----------
    img : ndarray (H, W)
        Input image (NumPy).
    frac : float
        Threshold as a fraction of max(img). E.g. 0.02 means 2% of max.
    pad : int
        Padding (pixels) added around the detected bbox.
    min_h, min_w : int
        Minimum crop size (safety so we don't end up with tiny crops).

    Returns
    -------
    (y1, y2, x1, x2) : tuple of ints
        Slice indices suitable for img[y1:y2, x1:x2]
    """
    H, W = img.shape
    m = float(np.max(img))
    if m <= 0:
        # Fallback: no signal detected
        return 0, H, 0, W

    thr = frac * m
    mask = img > thr

    if not np.any(mask):
        # Fallback: nothing above threshold
        return 0, H, 0, W

    ys = np.where(np.any(mask, axis=1))[0]
    xs = np.where(np.any(mask, axis=0))[0]

    y1, y2 = int(ys[0]), int(ys[-1] + 1)
    x1, x2 = int(xs[0]), int(xs[-1] + 1)

    # Pad
    y1 = max(0, y1 - pad)
    y2 = min(H, y2 + pad)
    x1 = max(0, x1 - pad)
    x2 = min(W, x2 + pad)

    # Enforce minimum size by expanding around center if needed
    ch = y2 - y1
    cw = x2 - x1

    if ch < min_h:
        c = (y1 + y2) // 2
        y1 = max(0, c - min_h // 2)
        y2 = min(H, y1 + min_h)
        y1 = max(0, y2 - min_h)

    if cw < min_w:
        c = (x1 + x2) // 2
        x1 = max(0, c - min_w // 2)
        x2 = min(W, x1 + min_w)
        x1 = max(0, x2 - min_w)

    return y1, y2, x1, x2


def crop_image(img, bbox):
    y1, y2, x1, x2 = bbox
    return img[y1:y2, x1:x2]


def crop_grids(X, Y, bbox):
    """
    Crop X, Y grids (NumPy arrays) in the same way as the image.
    """
    y1, y2, x1, x2 = bbox
    return X[y1:y2, x1:x2], Y[y1:y2, x1:x2]

def run_l2_on_long_pulse():
    """
    Classical L2-based parameter estimation on raster-14 image,
    using SciPy/JAX BFGS (no neural loss).
    """

    print("\n=== Running L2 / classical BFGS on raster-14 ===")

    # --------------------------------------------------
    # 1) Load and preprocess image
    # --------------------------------------------------
    I_raw, meta = load_long_pulse()

    I_clean = remove_background(I_raw)
    I_obs = crop_center(I_clean)

    show_image(I_obs, "Observed raster-14 image (cropped)")

    # Normalize (important for stability)
    I_obs = I_obs / (np.linalg.norm(I_obs) + 1e-12)

    # --------------------------------------------------
    # 2) Build simulation grid (NumPy/JAX world)
    # --------------------------------------------------
    H, W = I_obs.shape

    pulse_duration_ms = 3.0
    sampling_rate = 1_000_000

    X, Y, t_vals, _, _ = make_configuration(
        pulse_duration_ms=pulse_duration_ms,
        sampling_rate=sampling_rate,
        field_x=float(W),
        field_y=float(H),
        pixel_size=1.0,
    )

    # Convert JAX → NumPy (IMPORTANT for SciPy BFGS)
    X = np.array(X)
    Y = np.array(Y)
    t_vals = np.array(t_vals)

    # --------------------------------------------------
    # 3) Initial guess
    # --------------------------------------------------
    # Frequencies are meaningful → use metadata (Hz → kHz)
    fx0 = meta["Hfreq"] / 1000.0
    fy0 = meta["Vfreq"] / 1000.0

    k0 = np.array([
        50.0,        # Ax
        50.0,        # Ay
        6.0,         # sigx
        6.0,         # sigy
        0.0,         # cx
        0.0,         # cy
        fx0,         # fx (kHz)
        fy0,         # fy (kHz)
        0.0,         # phix
        0.0,         # phiy
    ], dtype=np.float64)

    print("\nInitial guess k0:")
    print(k0)

    I_sim0 = simulate_image(X, Y, t_vals, k0)
    show_image(I_sim0, "Initial simulated image (L2 model)")

    # --------------------------------------------------
    # 4) Bounds (same philosophy as LPIPS run)
    # --------------------------------------------------
    lower = np.array([
        10, 10,
        2, 2,
        -50, -50,
        0.8 * fx0, 0.8 * fy0,
        0.0, 0.0
    ])

    upper = np.array([
        100, 100,
        20, 20,
        50, 50,
        1.2 * fx0, 1.2 * fy0,
        2*np.pi, 2*np.pi
    ])

    bounds = list(zip(lower, upper))

    # --------------------------------------------------
    # 5) Classical BFGS (pixel L2)
    # --------------------------------------------------
    result = estimate_parameters_BFGS(
        I_obs=I_obs,
        X=X,
        Y=Y,
        t_vals=t_vals,
        k0=k0,
        bounds=bounds,
        verbose=True,
        maxiter=100
    )

    k_est = result.x
    print_k_estimate(k_est)

    # --------------------------------------------------
    # 6) Visualization
    # --------------------------------------------------
    visualize_estimation_result(
        I_obs=I_obs,
        X=X,
        Y=Y,
        t_vals=t_vals,
        k_est=k_est,
    )

# --------------------------------------------------

def run_l2_short_pulse():
    # --------------------------------------------------
    # 1) Load and preprocess image
    # --------------------------------------------------
    I_raw= load_short_pulse()

    I_clean = remove_background(I_raw)
    I_obs = crop_center(I_clean)

    show_image(I_obs, "Observed raster-14 image (cropped)")

    # Normalize (important for stability)
    I_obs = I_obs / (np.linalg.norm(I_obs) + 1e-12)

    # --------------------------------------------------
    # 2) Build simulation grid (NumPy/JAX world)
    # --------------------------------------------------
    H, W = I_obs.shape

    pulse_duration_ms = 0.02
    sampling_rate = 1_000_000

    X, Y, t_vals, _, _ = make_configuration(
        pulse_duration_ms=pulse_duration_ms,
        sampling_rate=sampling_rate,
        field_x=float(W),
        field_y=float(H),
        pixel_size=1.0,
    )

    # Convert JAX → NumPy (IMPORTANT for SciPy BFGS)
    X = np.array(X)
    Y = np.array(Y)
    t_vals = np.array(t_vals)

    # --------------------------------------------------
    # 3) Initial guess
    # --------------------------------------------------
    # Frequencies are meaningful → use metadata (Hz → kHz)
    fx0 = 21
    fy0 = 21

    k0 = np.array([
        100.0,     # Ax
        26.0,     # Ay
        4.5,      # sigx
        4.5,      # sigy
        85.0,     # cx
        -10.0,      # cy
        21,    # fx 
        21,    # fy 
        4.9,      # phix
        5.2,      # phiy
    ], dtype=np.float64)

    print("\nInitial guess k0:")
    print(k0)

    I_sim0 = simulate_image(X, Y, t_vals, k0)
    show_image(I_sim0, "Initial simulated image (L2 model)")

    # Even looser
    lower = np.array([
    30,   10,     # Ax, Ay  (allow strong shrinking)
    2.0,  2.0,    # sigx, sigy
   -20,  -40,    # cx, cy  (allow large drift)
    20, 20,  # fx, fy  (still fairly tight)
    0.0,  0.0
    ])

    upper = np.array([
        140,  60,
        15.0, 15.0,
        140,  40,
        21, 21,
        2*np.pi, 2*np.pi
    ])

    bounds = list(zip(lower, upper))

    loss_function = lambda k, I_obs, X, Y, t_vals: loss_regularized(k, I_obs, X, Y, t_vals, Ax_ref=90.0, Ay_ref=40.0,lambda_Ax=1e-4, lambda_Ay=1e-4)

    # --------------------------------------------------
    # 5) Classical BFGS (pixel L2)
    # --------------------------------------------------
    result = estimate_parameters_BFGS(
        I_obs=I_obs,
        X=X, Y=Y,
        t_vals=t_vals,
        k0=k0,
        bounds=bounds,
        loss_function=loss_function,
        maxiter=150,
        verbose=True,
    )

    k_est = result.x
    print_k_estimate(k_est)

    # --------------------------------------------------
    # 6) Visualization
    # --------------------------------------------------
    visualize_estimation_result(
        I_obs=I_obs,
        X=X,
        Y=Y,
        t_vals=t_vals,
        k_est=k_est,
    )

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


def show_bbox(img, bbox, title=""):
    y1, y2, x1, x2 = bbox
    plt.figure(figsize=(6,4))
    plt.imshow(img, cmap="inferno", origin="lower")
    plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1,
                                     edgecolor="cyan", facecolor="none", linewidth=2))
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.show()

# ---------- helpers ----------
def project_to_bounds(k, lower, upper):
    return jnp.minimum(jnp.maximum(k, lower), upper)

def make_mask(dim, fixed_params):
    """fixed_params: dict {idx: value} or None"""
    mask = jnp.ones((dim,), dtype=jnp.float32)
    if fixed_params:
        for idx in fixed_params.keys():
            mask = mask.at[idx].set(0.0)
    return mask

# ---------- main entry ----------
def make_fast_solver(
    X, Y, t_vals,
    I_obs,                     # already cropped + normalized, shape (H,W)
    lower, upper,              # shape (D,)
    fixed_params=None,         # dict {idx: value}
    loss_fn=None,              # loss_fn(k, I_obs, X, Y, t_vals) -> scalar
):
    """
    Returns a jitted function solve(k0) -> (k_est, final_loss)

    IMPORTANT: X, Y, t_vals, I_obs, bounds must have fixed shapes for JIT caching.
    """

    D = lower.shape[0]
    mask = make_mask(D, fixed_params)

    # Convert fixed params into a vector (for forcing values)
    fixed_vec = jnp.zeros((D,), dtype=jnp.float32)
    if fixed_params:
        for idx, val in fixed_params.items():
            fixed_vec = fixed_vec.at[idx].set(val)

    def enforce_fixed(k):
        # k_fixed = mask*k + (1-mask)*fixed_vec
        return k * mask + fixed_vec * (1.0 - mask)

    # Precompute normalized observed once (avoid doing it inside loss repeatedly)
    I_obs = jnp.asarray(I_obs)
    I_obs_n = I_obs / (jnp.linalg.norm(I_obs) + 1e-12)

    # Wrap loss to (a) enforce fixed params (b) reuse I_obs_n
    def loss_only(k):
        k = enforce_fixed(project_to_bounds(k, lower, upper))
        return loss_fn(k, I_obs_n, X, Y, t_vals)

    loss_and_grad = jax.value_and_grad(loss_only)

    @jax.jit
    def solve_adam(
        k0,
        steps=60,
        lr=0.05,
        b1=0.9,
        b2=0.999,
        eps=1e-8,
    ):
        """
        Fully jitted fixed-step solver.
        Tune (steps, lr) for speed/accuracy tradeoff.
        """

        k = enforce_fixed(project_to_bounds(jnp.asarray(k0, dtype=jnp.float32), lower, upper))
        m = jnp.zeros_like(k)
        v = jnp.zeros_like(k)

        def body(i, state):
            k, m, v = state

            val, g = loss_and_grad(k)
            # Mask gradient so fixed params don't move
            g = g * mask

            # Adam update
            m = b1 * m + (1.0 - b1) * g
            v = b2 * v + (1.0 - b2) * (g * g)

            # Bias correction
            t = i + 1.0
            mhat = m / (1.0 - b1 ** t)
            vhat = v / (1.0 - b2 ** t)

            k = k - lr * mhat / (jnp.sqrt(vhat) + eps)
            k = enforce_fixed(project_to_bounds(k, lower, upper))

            return (k, m, v)

        k, m, v = lax.fori_loop(0, steps, body, (k, m, v))
        final_loss = loss_only(k)
        return k, final_loss

    return solve_adam    

def loss_l2_and_reg(k, I_obs_n, X, Y, t_vals,
                    Ax_ref=90.0, Ay_ref=40.0,
                    lambda_Ax=1e-4, lambda_Ay=1e-4):
    """
    JAX-friendly loss: shape-only L2 + quadratic priors on Ax, Ay.
    I_obs_n must be L2-normalized already.
    """
    # Simulate
    I_sim = simulate_image(X, Y, t_vals, k)

    # L2-normalize simulated image (shape only)
    I_sim = I_sim / (jnp.linalg.norm(I_sim) + 1e-12)

    # Data term
    L_data = jnp.sum((I_sim - I_obs_n) ** 2)

    # Regularization on Ax, Ay
    Ax, Ay = k[0], k[1]
    L_reg = lambda_Ax * (Ax - Ax_ref) ** 2 + lambda_Ay * (Ay - Ay_ref) ** 2

    return L_data + L_reg


def run_custom_image():
    """
    Fast JAX-jitted fixed-step Adam optimization on a newly acquired EPICS image.
    Uses fixed ROI (crop_center) + fixed sampling_rate = 2_000_000.
    """

    print("\n=== Running FAST JAX Adam on custom EPICS image ===")

    # --------------------------------------------------
    # 1) Load + preprocess
    # --------------------------------------------------
    I_raw, meta = load_custom_snapshot()

    I_clean = remove_background(I_raw)

    bbox = tight_crop_bbox(
        I_clean,
        frac=0.02,   # or tune (0.01–0.03)
        pad=10,
        min_h=32,
        min_w=32,
    )

    show_bbox(I_clean, bbox, title="Tight signal ROI")

    I_obs = crop_image(I_clean, bbox)

    show_image(I_obs, "Observed EPICS image (cropped)")

    # Normalize observed to L2=1 (do in NumPy here; make_fast_solver will also normalize,
    # but it's fine to keep it stable/consistent)
    I_obs = I_obs / (np.linalg.norm(I_obs) + 1e-12)

    # --------------------------------------------------
    # 2) Build simulation grid (JAX arrays, fixed shapes)
    # --------------------------------------------------
    H, W = I_obs.shape

    # Exposure time -> pulse duration (match your earlier convention)
    # Your earlier code did: pulse_duration_ms = exposure_time / 1_000_000
    # That implies exposure_time is in microseconds.
    pulse_duration_ms = meta["exposure_time"] / 1_000_000.0

    sampling_rate = 2_000_000

    X, Y, t_vals, _, _ = make_configuration(
        pulse_duration_ms=float(pulse_duration_ms),
        sampling_rate=int(sampling_rate),
        field_x=float(W),
        field_y=float(H),
        pixel_size=1.0,
    )

    # Keep as JAX float32 for speed
    X = jnp.asarray(X, dtype=jnp.float32)
    Y = jnp.asarray(Y, dtype=jnp.float32)
    t_vals = jnp.asarray(t_vals, dtype=jnp.float32)
    I_obs_j = jnp.asarray(I_obs, dtype=jnp.float32)

    # --------------------------------------------------
    # 3) Initial guess + bounds
    # --------------------------------------------------
    fx0 = float(meta["fx"])   # kHz (as saved in npz)
    fy0 = float(meta["fy"])   # kHz

    Ax0 = 40.0
    Ay0 = 30.0

    k0 = np.array([
        Ax0,     # Ax
        Ay0,     # Ay
        5.0,     # sigx
        5.0,     # sigy
        -5.0,    # cx
        -5.0,    # cy
        fx0,     # fx
        fy0,     # fy
        1.0,     # phix
        2.0,     # phiy
    ], dtype=np.float32)

    # Bounds (same idea as your earlier code)
    lower = np.array([
        10, 10,
        2, 2,
        -50, -50,
        0.95 * fx0, 0.95 * fy0,
        0.0, 0.0
    ], dtype=np.float32)

    upper = np.array([
        120, 80,
        20, 20,
        50, 50,
        1.05 * fx0, 1.05 * fy0,
        2*np.pi, 2*np.pi
    ], dtype=np.float32)

    # Convert to JAX arrays
    k0_j = jnp.asarray(k0, dtype=jnp.float32)
    lower_j = jnp.asarray(lower, dtype=jnp.float32)
    upper_j = jnp.asarray(upper, dtype=jnp.float32)

    # Fix fx, fy if you want (recommended for speed + stability)
    fixed_params = {6: jnp.float32(fx0), 7: jnp.float32(fy0)}
    # fixed_params = None

    # --------------------------------------------------
    # 4) Build the fast solver (compiled)
    # --------------------------------------------------
    # IMPORTANT: use make_fast_solver (the function you already have)
    solve_fast = make_fast_solver(
        X=X, Y=Y, t_vals=t_vals,
        I_obs=I_obs_j,
        lower=lower_j, upper=upper_j,
        fixed_params=fixed_params,
        loss_fn=lambda k, I_obs_n, X, Y, t: loss_l2_and_reg(
            k, I_obs_n, X, Y, t,
            Ax_ref=Ax0, Ay_ref=Ay0,
            lambda_Ax=5e-3, lambda_Ay=5e-3
        ),
    )

    # --------------------------------------------------
    # 5) Warmup compile (do once)
    # --------------------------------------------------
    k_tmp, L_tmp = solve_fast(k0_j, steps=5, lr=0.05)
    k_tmp.block_until_ready()

    # --------------------------------------------------
    # 6) Timed run
    # --------------------------------------------------
    t0 = time.perf_counter()
    
    k_est_j, L_final = solve_fast(k0_j, steps=50, lr=0.07)
    
    k_est_j.block_until_ready()
    t1 = time.perf_counter()

    k_est = np.asarray(k_est_j, dtype=np.float64)
    print(f"Final loss: {float(L_final):.6e}")

    print_k_estimate(k_est)
    print(f"\nJIT Adam solve time: {t1 - t0:.4f} s")

    # --------------------------------------------------
    # 7) Visualization 
    # --------------------------------------------------
    visualize_estimation_result(
        I_obs=np.array(I_obs_j),
        X=np.array(X),
        Y=np.array(Y),
        t_vals=np.array(t_vals),
        k_est=k_est,
    )


def run_custom_image_bfgs():
    print("\n=== Running BFGS (SciPy L-BFGS-B) on custom EPICS image ===")

    # --------------------------------------------------
    # 1) Load + preprocess
    # --------------------------------------------------
    I_raw, meta = load_custom_snapshot()

    I_clean = remove_background(I_raw)

    bbox = tight_crop_bbox(
        I_clean,
        frac=0.02,
        pad=10,
        min_h=32,
        min_w=32,
    )

    show_bbox(I_clean, bbox, title="Tight signal ROI")
    I_obs = crop_image(I_clean, bbox)
    show_image(I_obs, "Observed EPICS image (cropped)")

    # Normalize observed (shape-only)
    I_obs = I_obs / (np.linalg.norm(I_obs) + 1e-12)

    # --------------------------------------------------
    # 2) Build simulation grid (NumPy arrays for SciPy)
    # --------------------------------------------------
    H, W = I_obs.shape

    # Exposure time -> pulse duration (ms)
    pulse_duration_ms = meta["exposure_time"] / 1_000_000.0

    sampling_rate = 2_000_000

    X, Y, t_vals, _, _ = make_configuration(
        pulse_duration_ms=float(pulse_duration_ms),
        sampling_rate=int(sampling_rate),
        field_x=float(W),
        field_y=float(H),
        pixel_size=1.0,
    )

    # IMPORTANT: SciPy wants NumPy arrays
    X = np.array(X)
    Y = np.array(Y)
    t_vals = np.array(t_vals)

    # --------------------------------------------------
    # 3) Initial guess + bounds
    # --------------------------------------------------
    fx0 = float(meta["fx"])   # kHz
    fy0 = float(meta["fy"])   # kHz

    Ax0 = 40.0
    Ay0 = 30.0

    k0 = np.array([
        Ax0,     # Ax
        Ay0,     # Ay
        5.0,     # sigx
        5.0,     # sigy
        -5.0,    # cx
        -5.0,    # cy
        fx0,     # fx
        fy0,     # fy
        1.0,     # phix
        2.0,     # phiy
    ], dtype=np.float64)

    lower = np.array([
        10, 10,
        2, 2,
        -50, -50,
        0.95 * fx0, 0.95 * fy0,
        0.0, 0.0
    ], dtype=np.float64)

    upper = np.array([
        120, 80,
        20, 20,
        50, 50,
        1.05 * fx0, 1.05 * fy0,
        2*np.pi, 2*np.pi
    ], dtype=np.float64)

    bounds = list(zip(lower, upper))

    # --------------------------------------------------
    # 4) Loss with regularization (Ax, Ay)
    # --------------------------------------------------
    # NOTE: loss_regularized imported from solve_minimization_10D_real
    # (same signature style as your other functions)
    reg_loss = lambda k, I_obs, X, Y, t_vals: loss_regularized(
        k, I_obs, X, Y, t_vals,
        Ax_ref=Ax0,
        Ay_ref=Ay0,
        lambda_Ax=5e-3,
        lambda_Ay=5e-3
    )

    # --------------------------------------------------
    # 5) Run BFGS (SciPy L-BFGS-B)
    # --------------------------------------------------
    result = estimate_parameters_BFGS(
        I_obs=I_obs,
        X=X,
        Y=Y,
        t_vals=t_vals,
        k0=k0,
        bounds=bounds,
        loss_function=reg_loss,
        maxiter=100,
        verbose=True,
    )

    k_est = result.x
    print_k_estimate(k_est)

    # --------------------------------------------------
    # 6) Visualization
    # --------------------------------------------------
    visualize_estimation_result(
        I_obs=I_obs,
        X=X,
        Y=Y,
        t_vals=t_vals,
        k_est=k_est,
    )

# --------------------------------------------------

if __name__ == "__main__":
    run_custom_image_bfgs()
    run_custom_image()
    run_l2_short_pulse()
    run_l2_on_long_pulse()