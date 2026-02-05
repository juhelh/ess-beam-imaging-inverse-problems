import numpy as np
import jax.numpy as jnp

from ess_beam_imaging.preprocessing import (
    remove_background,
    tight_crop_bbox,
    crop_image,
    show_image,  # DEBUG
    show_bbox,   # DEBUG
)

from configuration.configuration import make_configuration
from minimization.solve_minimization_10D import loss_function, simulate_image, estimate_parameters_BFGS


def _parse_input(image, meta=None, config=None):
    """
    Prepare the raw inputs for estimation.

    This function:
    - Ensures the image is a 2D numerical array
    - Converts it to a consistent floating-point format
    - Extracts the pulse duration from metadata (if provided)
    """
    meta = {} if meta is None else meta
    config = {} if config is None else config

    image = jnp.asarray(image, dtype=jnp.float32)
    if image.ndim != 2:
        raise ValueError("image must be a 2D array")

    # Pulse duration in ms:
    # Priority: explicit pulse_duration_ms -> else derive from exposure_time (assumed microseconds) -> else fallback
    if "T" in meta:
        T = float(meta["T"])

    return {"image": image, "meta": meta, "config": config, "T": T}


def _preprocess_image(image_jnp, config=None):
    """
    Clean and isolate the beam signal in the image.

    This step:
    - Removes smooth background intensity
    - Automatically finds and crops the region containing the beam
    - Normalizes the cropped image so only its shape matters
    """
    config = {} if config is None else config
    debug = bool(config.get("debug", False))

    I_np = np.asarray(image_jnp)

    I_clean = remove_background(I_np)

    bbox = tight_crop_bbox(I_clean, frac=0.02, pad=10, min_h=32, min_w=32)

    if debug:
        show_bbox(I_clean, bbox, title="Tight signal ROI")

    I_crop = crop_image(I_clean, bbox)

    if debug:
        show_image(I_crop, title="Observed image (cropped)")

    I_crop = I_crop / (np.linalg.norm(I_crop) + 1e-12)

    I_obs = jnp.asarray(I_crop, dtype=jnp.float32)
    return I_obs, bbox


def _build_grid_and_time(I_obs, T, sampling_rate=2_000_000):
    """
    Construct the arrays for space and time in the beam model.

    This function creates:
    - A 2D spatial grid matching the image pixels
    - A time grid covering the beam pulse duration
    """
    H, W = I_obs.shape

    X, Y, t_vals, _, _ = make_configuration(
        pulse_duration_ms=T,
        sampling_rate=sampling_rate,
        field_x=float(W),
        field_y=float(H),
        pixel_size=1.0,
    )

    return (
        jnp.asarray(X, dtype=jnp.float32),
        jnp.asarray(Y, dtype=jnp.float32),
        jnp.asarray(t_vals, dtype=jnp.float32),
    )


def _init_guess_and_bounds(H, W, meta=None):
    """
    Choose initial values and limits for the beam parameters.

    The initial values are rough guesses based on the image size.
    If measured frequencies are available, they are used directly.

    The bounds limit how far each parameter is allowed to move,
    and known parameters can be fixed so they are not changed.
    """
    meta = {} if meta is None else meta

    Ax0 = 0.7 * W/2
    Ay0 = 0.7 * H/2

    fx0 = meta.get("fx", 39.55)
    fy0 = meta.get("fy", 29.05)

    k0 = np.array([Ax0, Ay0, 5.0, 5.0, 0.0, 0.0, fx0, fy0, 0.0, 0.0], dtype=np.float32)

    lower = np.array(
        [0.0, 0.0, 1.0, 1.0, -W / 2.0, -H / 2.0, 0.9 * fx0, 0.9 * fy0, -np.pi, -np.pi],
        dtype=np.float32,
    )
    upper = np.array(
        [float(W), float(H), 10.0, 10.0, W / 2.0, H / 2.0, 1.1 * fx0, 1.1 * fy0, np.pi, np.pi],
        dtype=np.float32,
    )

    # For fixed params lower = upper
    fixed_params = {}
    if "fx" in meta:
        lower[6] = upper[6] = fx0
        fixed_params[6] = fx0
    if "fy" in meta:
        lower[7] = upper[7] = fy0
        fixed_params[7] = fy0

    return (
        jnp.asarray(k0),
        jnp.asarray(lower),
        jnp.asarray(upper),
        fixed_params,
    )

def _config_regularization(k0, config):
    """
    Convert a high-level regularization specification into arrays used by the loss.

    - reg_mask: marks which parameters are regularized
    - reg_ref: reference values (taken from the initial guess k0)
    - reg_lambda: regularization strength per parameter
    """
    reg_cfg = config.get("regularization", None)
    if reg_cfg is None:
        return config

    param_index = {
        "Ax": 0,
        "Ay": 1,
        "sigx": 2,
        "sigy": 3,
        "cx": 4,
        "cy": 5,
        "fx": 6,
        "fy": 7,
        "phix": 8,
        "phiy": 9,
    }

    D = k0.shape[0]
    reg_mask = np.zeros(D, dtype=np.float32)
    reg_ref = np.zeros(D, dtype=np.float32)
    reg_lambda = np.zeros(D, dtype=np.float32)

    params = reg_cfg.get("params", {})

    for name, lam in params.items():
        if name not in param_index:
            raise ValueError(f"Unknown regularized parameter '{name}'")

        idx = param_index[name]
        reg_mask[idx] = 1.0
        reg_ref[idx] = float(k0[idx])      # reference from run function
        reg_lambda[idx] = float(lam)        # per-parameter lambda

    # Inject low-level fields expected by loss_regularized
    config = dict(config)
    config["reg_mask"] = reg_mask
    config["reg_ref"] = reg_ref
    config["reg_lambda"] = reg_lambda

    return config


def loss_regularized(k, I_obs, X, Y, t_vals, config):
    """
    Loss = data mismatch + optional quadratic regularization.

    Regularization is applied only if reg_mask is present in config.
    Reference values and strengths are read from config.
    """
    L_data = loss_function(k, I_obs, X, Y, t_vals)

    reg_mask = config.get("reg_mask", None)
    if reg_mask is None:
        return L_data

    reg_ref = jnp.asarray(config["reg_ref"])
    reg_mask = jnp.asarray(config["reg_mask"])
    reg_lambda = jnp.asarray(config["reg_lambda"])

    diff = (k - reg_ref) * reg_mask
    L_reg = jnp.sum(reg_lambda * diff**2)

    return L_data + L_reg

def generate_initial_points(k0, lower, upper, fixed_params, config):
    """
    Generate multiple initial guesses for the optimizer.

    Only parameters that are neither fixed nor regularized are randomized.
    """
    ms_cfg = config.get("multistart", None)
    if ms_cfg is None:
        return [k0]

    n = int(ms_cfg.get("n_starts", 1))
    mode = ms_cfg.get("type", "uniform")

    D = k0.shape[0]

    # Indices that must NOT be randomized
    fixed_idx = set(fixed_params.keys())
    reg_mask = config.get("reg_mask", None)
    reg_idx = set(np.where(reg_mask)[0]) if reg_mask is not None else set()

    free_idx = [i for i in range(D) if i not in fixed_idx and i not in reg_idx]

    starts = []

    if mode == "uniform":
        for _ in range(n):
            k = np.array(k0, copy=True)
            for i in free_idx:
                k[i] = lower[i] + (upper[i] - lower[i]) * np.random.rand()
            starts.append(jnp.asarray(k, dtype=jnp.float32))

    elif mode == "gaussian":
        std_scale = 0.5
        sigma = std_scale * (upper - lower)
        for _ in range(n):
            k = np.array(k0, copy=True)
            for i in free_idx:
                k[i] = k0[i] + sigma[i] * np.random.randn()
                k[i] = np.clip(k[i], lower[i], upper[i])
            starts.append(jnp.asarray(k, dtype=jnp.float32))

    else:
        raise ValueError(f"Unknown multistart type '{mode}'")

    return starts

def estimate_with_optimizer(image, meta=None, config=None):
    """
    Do parameter estimation using optimization.

    This function:
      1) Parses input image and metadata
      2) Crops and normalizes the image
      3) Builds a simulation grid and time axis
      4) Sets initial guesses and parameter bounds
      5) Runs bounded L-BFGS optimization
      6) Returns estimated beam parameters

    Parameters
    ----------
    image : ndarray (H, W)
        Raw camera image.
    meta : dict
        Machine metadata (e.g. fx, fy, exposure_time).
    config : dict
        Optional configuration (debug, regularization).

    Returns
    -------
    k_est : ndarray (10,)
        Estimated beam parameters.
    """
    # --------------------------------------------------
    # 1) Parse inputs and metadata
    # --------------------------------------------------
    parsed = _parse_input(image, meta=meta, config=config)

    I_obs = parsed["image"]
    T = parsed["T"]
    meta = parsed["meta"]
    config = parsed["config"]

    debug = bool(config.get("debug", False))


    # --------------------------------------------------
    # 2) Preprocess the observed image
    # --------------------------------------------------
    I_obs, bbox = _preprocess_image(I_obs, config=config)


    # --------------------------------------------------
    # 3) Build spatial and temporal grids
    # --------------------------------------------------
    X, Y, t_vals = _build_grid_and_time(I_obs, T)


    # --------------------------------------------------
    # 4) Initial guess, bounds, and fixed parameters
    # --------------------------------------------------
    H, W = I_obs.shape
    k0, lower, upper, fixed_params = _init_guess_and_bounds(H, W, meta)


    # --------------------------------------------------
    # 5) Define loss function and optional regularization
    # --------------------------------------------------
    if config.get("regularization", False):
        config = _config_regularization(k0, config)
        loss_fn = lambda k, I_obs, X, Y, t_vals: loss_regularized(
            k, I_obs, X, Y, t_vals, config
        )
    else:
        loss_fn = loss_function


    # --------------------------------------------------
    # 6) Optimization (with optional multistart)
    # --------------------------------------------------

    results = []   # will store (loss, k)

    # First run from user's best guess
    result = estimate_parameters_BFGS(
        I_obs=I_obs,
        X=X,
        Y=Y,
        t_vals=t_vals,
        k0=k0,
        loss_function=loss_fn,
        bounds=(lower, upper),
        maxiter=100,
        verbose=True,
    )

    results.append((result.fun, result.x))

    # Additional multistart runs
    k_starts = generate_initial_points(k0, lower, upper, fixed_params, config)

    for i, k_start in enumerate(k_starts):
        if debug:
            print(f"\n--- Multistart {i+1}/{len(k_starts)} ---")

        result = estimate_parameters_BFGS(
            I_obs=I_obs,
            X=X,
            Y=Y,
            t_vals=t_vals,
            k0=k_start,
            loss_function=loss_fn,
            bounds=(lower, upper),
            maxiter=100,
            verbose=debug,
        )

        results.append((result.fun, result.x))

    # --------------------------------------------------
    # 7) Rank and return solutions
    # --------------------------------------------------

    # Sort by loss (smallest first)
    results.sort(key=lambda x: x[0])

    # Best estimate
    best_loss, best_k = results[0]

    if debug:
        # Collect sorted losses and ks in lists
        losses = [float(L) for L, _ in results]
        ks = [np.asarray(k, dtype=np.float64) for _, k in results]

        return ks, losses, I_obs, t_vals, X, Y
    else:
        return np.asarray(best_k, dtype=np.float64)