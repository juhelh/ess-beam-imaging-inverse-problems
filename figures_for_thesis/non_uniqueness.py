import matplotlib.pyplot as plt
import jax.numpy as jnp
from minimization.solve_minimization_8D import simulate_image, raster_position  
from configuration.configuration import make_configuration 

def show_nonuniqueness_large_loop(alpha=2):
    """
    Non-uniqueness demonstration with visibly spread-out raster loop.
    Uses (fx, fy) = (2, 3) and (4, 6) to show identical accumulated images.
    """

    # === Frequencies (kHz) giving a clear Lissajous-type loop ===
    fx_base = 2.0
    fy_base = 3.0
    N_x, N_y = 3, 2  # so that fy/fx = 3/2, closes after LCM(3,2)/fx = 6/2 = 3 ms
    lcm_cycles = jnp.lcm(N_x, N_y)
    T_full = lcm_cycles / fx_base * 1e-3  # seconds (since fx in kHz)

    # === Configuration ===
    X, Y, t_vals, dt, k1 = make_configuration(pulse_duration_ms=T_full * 1e3)

    # Replace with base (2,3) kHz
    k1 = k1.at[6].set(fx_base)
    k1 = k1.at[7].set(fy_base)

    # Scaled version (4,6) kHz
    k2 = k1.at[6].set(alpha * fx_base)
    k2 = k2.at[7].set(alpha * fy_base)

    # Adjust pulse time for scaled case
    T_full_scaled = T_full / alpha
    t_vals_scaled = jnp.linspace(0, T_full_scaled, len(t_vals))

    # === Simulate accumulated images ===
    I1 = simulate_image(X, Y, t_vals, k1)
    I2 = simulate_image(X, Y, t_vals_scaled, k2)
    diff = jnp.abs(I1 - I2)

    # === Compute raster trajectories for visualization ===
    t_vis = jnp.linspace(0, T_full, 1000)
    x1, y1 = raster_position(t_vis, k1[0], k1[1], k1[6], k1[7], k1[4], k1[5])
    x2, y2 = raster_position(t_vis, k2[0], k2[1], k2[6], k2[7], k2[4], k2[5])

    # === Print diagnostic info ===
    print(f"fx1={k1[6]:.1f}, fy1={k1[7]:.1f}, fy1/fx1={k1[7]/k1[6]:.2f}")
    print(f"fx2={k2[6]:.1f}, fy2={k2[7]:.1f}, fy2/fx2={k2[7]/k2[6]:.2f}")
    print(f"T1={T_full*1e3:.3f} ms, T2={T_full_scaled*1e3:.3f} ms")

    # === Plot ===
    fig, axs = plt.subplots(1, 4, figsize=(18, 5))

    im0 = axs[0].imshow(I1, cmap='inferno', origin='lower')
    axs[0].set_title(f"Image 1: fx={k1[6]:.1f}, fy={k1[7]:.1f}")
    plt.colorbar(im0, ax=axs[0], shrink=0.8)

    im1 = axs[1].imshow(I2, cmap='inferno', origin='lower')
    axs[1].set_title(f"Image 2: fx={k2[6]:.1f}, fy={k2[7]:.1f}")
    plt.colorbar(im1, ax=axs[1], shrink=0.8)

    im2 = axs[2].imshow(diff, cmap='seismic', origin='lower', vmin=-0.05, vmax=0.05)
    axs[2].set_title("Absolute difference")
    plt.colorbar(im2, ax=axs[2], shrink=0.8)

    # --- Raster trajectories (loop shape) ---
    axs[3].plot(x1, y1, lw=2, label="(2,3) kHz")
    axs[3].plot(x2, y2, "--", lw=2, label="(4,6) kHz")
    axs[3].set_title("Raster trajectories")
    axs[3].set_xlabel("x [mm]")
    axs[3].set_ylabel("y [mm]")
    axs[3].set_aspect('equal', adjustable='box')
    axs[3].legend()

    plt.suptitle("Non-uniqueness from frequency scaling (same fy/fx ratio)", fontsize=14)
    plt.tight_layout()
    plt.show()


# Run
show_nonuniqueness_large_loop()
