import jax.numpy as jnp

# --- Pixel & field of view ---
PIXEL_SIZE = 0.4  # mm per pixel
FIELD_X = 150.0   # mm (x range: -50 to +50)
FIELD_Y = 60.0    # mm (y range: -30 to +30)

NX = int(FIELD_X / PIXEL_SIZE)
NY = int(FIELD_Y / PIXEL_SIZE)

# --- Coordinate grid ---
x = jnp.linspace(-FIELD_X / 2, FIELD_X / 2, NX)
y = jnp.linspace(-FIELD_Y / 2, FIELD_Y / 2, NY)
X, Y = jnp.meshgrid(x, y, indexing="xy")

# --- Time setup ---
PULSE_DURATION = 2.86/1000  # ms to s
SAMPLING_RATE = 100_0000       # Hz
DT = 1.0 / SAMPLING_RATE      # 10 μs
NUM_TIME_POINTS = int(PULSE_DURATION * SAMPLING_RATE)
T_VALS = jnp.linspace(0.0, PULSE_DURATION, NUM_TIME_POINTS)