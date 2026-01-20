import jax
import jax.numpy as jnp
from functools import partial
from jax import lax

# For real images
def make_configuration(
    pulse_duration_ms = 0.05,
    sampling_rate = 1_000_000,
    field_x_mm = 400.0,
    field_y_mm = 350.0,
    pixel_size_mm = 1.0
):

    # --- Spatial grid ---
    NX = int(field_x_mm / pixel_size_mm)
    NY = int(field_y_mm / pixel_size_mm)

    x = jnp.linspace(-field_x_mm/2, field_x_mm/2, NX)
    y = jnp.linspace(-field_y_mm/2, field_y_mm/2, NY)
    X, Y = jnp.meshgrid(x, y, indexing="xy")

    # --- Time grid ---
    DT = 1.0 / sampling_rate
    pulse_s = pulse_duration_ms / 1000.0
    N_time = int(pulse_s * sampling_rate)
    T_VALS = jnp.linspace(0.0, pulse_s, N_time)

    # --- Default "reference" parameters (unchanged) ---
    k_true = jnp.array([
        60.0,
        20.0,
        13.5,
        5.05,
        0.0,
        0.0,
        39.55,
        29.05
    ])

    return X, Y, T_VALS, DT, k_true

# # For simulations 
# def make_configuration(pulse_duration_ms = 0.05, sampling_rate = 1_000_000):
#     # --- Spatial field configuration ---
#     PIXEL_SIZE = 0.4  # mm
#     FIELD_X = 150.0   # mm (x: -75 to +75)
#     FIELD_Y = 60.0    # mm (y: -30 to +30)


#     NX = int(FIELD_X / PIXEL_SIZE)
#     NY = int(FIELD_Y / PIXEL_SIZE)

#     x = jnp.linspace(-FIELD_X / 2, FIELD_X / 2, NX)
#     y = jnp.linspace(-FIELD_Y / 2, FIELD_Y / 2, NY)
#     X, Y = jnp.meshgrid(x, y, indexing="xy")

#     # --- Time configuration ---
#     SAMPLING_RATE = sampling_rate  # Hz = 1 µs steps
#     DT = 1.0 / SAMPLING_RATE

#     pulse_duration_s = pulse_duration_ms / 1000.0
#     NUM_TIME_POINTS = int(pulse_duration_s * SAMPLING_RATE)
#     T_VALS = jnp.linspace(0.0, pulse_duration_s, NUM_TIME_POINTS)

#     # --- Ground truth beam parameters ---
#     # [Ax, Ay, sigx, sigy, cx, cy, fx, fy]
#     k_true = jnp.array([
#         60.0,   # Ax (horizontal raster amplitude)
#         20.0,   # Ay (vertical raster amplitude)
#         13.5,   # sigx (horizontal width)
#         5.05,   # sigy (vertical width)
#         0.0,   # cx (horizontal offset)
#         0.0,   # cy (vertical offset)
#         39_55,   # fx (horizontal frequency)
#         29_05    # fy (vertical frequency)
#     ])

#     return X, Y, T_VALS, DT, k_true