import numpy as np
import matplotlib.pyplot as plt
plt.ion()
from matplotlib.animation import FuncAnimation
from scipy.special import erf

def triangle_wave(f, t):
    """
    Triangle wave with amplitude in [-1, 1].

    tri(t; f) = (2/pi) * arcsin( sin(2*pi*f*t) )

    Parameters
    ----------
    f : float
        Frequency in Hz (cycles per second).
    t : float or np.ndarray
        Time(s). Scalar or array-like. Units must match 1/f.

    Returns
    -------
    float or np.ndarray
        Triangle wave evaluated at t, same shape as t.
    """
    return (2.0 / np.pi) * np.arcsin(np.sin(2.0 * np.pi * f * t))

def raster_position(t, Ax, Ay, fx, fy, cx, cy):
    """
    Compute the x, y raster coordinates at time t.

    Parameters
    ----------
    t  : float or np.ndarray
        Time(s)
    Ax : float
        Amplitude in x-direction
    Ay : float
        Amplitude in y-direction
    fx : float
        Frequency in x-direction
    fy : float
        Frequency in y-direction
    cx : float
        Center offset in x-direction
    cy : float
        Center offset in y-direction

    Returns
    -------
    Tuple[x, y] : ndarray or float
        Raster coordinates at time t
    """
    x_pos = Ax * triangle_wave(fx, t) + cx
    y_pos = Ay * triangle_wave(fy, t) + cy
    return x_pos, y_pos

def beam_intensity(x, y, t, k):
    """
    Compute 2D Gaussian beam intensity at (x, y, t) given beam parameters k.

    Parameters
    ----------
    x : float or np.ndarray
        x-coordinate(s)
    y : float or np.ndarray
        y-coordinate(s)
    t : float or np.ndarray
        Time(s)
    k : array-like of length 8
        Beam parameters: [Ax, Ay, sigx, sigy, cx, cy, fx, fy]

    Returns
    -------
    I : float or np.ndarray
        Beam intensity at each (x, y, t)
    """
    Ax, Ay, sigx, sigy, cx, cy, fx, fy = k
    gx, gy = raster_position(t, Ax, Ay, fx, fy, cx, cy)

    normalizing_const = 1.0 / (2 * np.pi * sigx * sigy)
    exponent = -0.5 * (((x - gx) / sigx) ** 2 + ((y - gy) / sigy) ** 2)

    I = normalizing_const * np.exp(exponent) # Will return a 2d array if x and y are vectors
    return I

def simulate_image(x, y, t_vals, k):
    """
    Simulate accumulated beam image over time.

    Parameters
    ----------
    x : np.ndarray
        x grid (2D)
    y : np.ndarray
        y grid (2D)
    t_vals : np.ndarray
        1D array of time points
    k : array-like
        Beam parameters [Ax, Ay, sigx, sigy, cx, cy, fx, fy]

    Returns
    -------
    I_acc : np.ndarray
        Accumulated intensity image over time, same shape as x/y
    """
    dt = t_vals[1] - t_vals[0]
    I_accumalated = np.zeros_like(x)

    for t in t_vals:
        I_accumalated += beam_intensity(x, y, t, k) * dt

    return I_accumalated

def animate_beam_accumulation(X, Y, t_vals, k, intensity_const=10.0):
    """
    Animate beam intensity over time and show live accumulation.

    Parameters
    ----------
    X, Y : 2D meshgrid
        Spatial grid
    t_vals : 1D array
        Time vector
    k : array-like
        Beam parameters [Ax, Ay, sigx, sigy, cx, cy, fx, fy]
    intensity_const : float
        Scaling constant for intensity (arbitrary energy unit)
    """
    print("Running test: simulate_image with accumulation and animation")

    dt = t_vals[1] - t_vals[0]
    T = len(t_vals)

    # Prepare accumulator
    I_acc = np.zeros_like(X)

    # Initial beam intensity (to set color scale)
    I0 = np.array(beam_intensity(X, Y, t_vals[0], k))
    vmax = np.max(I0)

    # Plot setup
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: current frame
    im_frame = ax1.imshow(
        np.zeros_like(X),
        extent=[X.min(), X.max(), Y.min(), Y.max()],
        origin='lower',
        aspect='equal',
        cmap='inferno',
        vmin=0, vmax=vmax
    )
    ax1.set_title("Beam intensity at time t")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")

    # Right: accumulated image
    im_acc = ax2.imshow(
        np.zeros_like(X),
        extent=[X.min(), X.max(), Y.min(), Y.max()],
        origin='lower',
        aspect='equal',
        cmap='inferno',
        vmin=0, vmax=vmax
    )
    ax2.set_title("Accumulated image so far")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    cb = plt.colorbar(im_acc, ax=ax2)
    cb.set_label("Accumulated energy")

    def update(frame):
        t = t_vals[frame]
        I = np.array(beam_intensity(X, Y, t, k))
        I_acc[:] += intensity_const * I * dt

        im_frame.set_data(I)
        im_acc.set_data(I_acc)

        ax1.set_title(f"Beam at t = {t:.2f} s")
        ax2.set_title(f"Accumulated image (step {frame+1}/{T})")
        return [im_frame, im_acc]

    ani = FuncAnimation(fig, update, frames=T, interval=30, blit=False)
    plt.tight_layout()
    plt.show()

    # Final image after animation
    plt.figure()
    plt.imshow(
        I_acc,
        extent=[X.min(), X.max(), Y.min(), Y.max()],
        origin='lower',
        aspect='equal',
        cmap='inferno'
    )
    plt.colorbar(label="Final accumulated energy")
    plt.title("Simulated beam image (fully integrated)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(False)
    plt.show()

def visualize_image(image, X, Y):
    plt.figure(figsize=(8, 5))
    plt.imshow(
        image,
        extent=[X.min(), X.max(), Y.min(), Y.max()],
        origin='lower',
        aspect='equal',
        cmap='inferno'
    )
    plt.colorbar(label="Accumulated intensity")
    plt.title("Simulated ESS Beam Image (Nominal Parameters)")
    plt.xlabel("x [mm]")
    plt.ylabel("y [mm]")
    plt.grid(False)
    plt.tight_layout()
    plt.show()

def show_erf_result():# Parameters
    Ax, Ay = 2.0, 1.5     # Half-widths of the raster box
    cx, cy = 0.0, 0.0     # Center of rastering
    sigx, sigy = 0.5, 0.5 # Beam standard deviations

    # Grid for plotting
    x = np.linspace(-4, 4, 300)
    y = np.linspace(-3, 3, 300)
    X, Y = np.meshgrid(x, y)

    # Final accumulated image expression from the analytic error function formula
    def I_inf(x, y, Ax, Ay, cx, cy, sigx, sigy):
        term_x = erf((x - cx + Ax) / (np.sqrt(2) * sigx)) - erf((x - cx - Ax) / (np.sqrt(2) * sigx))
        term_y = erf((y - cy + Ay) / (np.sqrt(2) * sigy)) - erf((y - cy - Ay) / (np.sqrt(2) * sigy))
        return 0.25 * term_x * term_y

    # Evaluate
    I = I_inf(X, Y, Ax, Ay, cx, cy, sigx, sigy)

    # Plot
    plt.figure(figsize=(6, 5))
    plt.contourf(X, Y, I, levels=50, cmap='magma')
    plt.colorbar(label=r'$I_\infty(x, y)$')
    plt.title(r'Accumulated Image $I_\infty(x, y)$ from 2D Rastered Gaussian')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    import matplotlib
    matplotlib.use("TkAgg")  # or "QtAgg", depending on OS

    # === Test: triangle wave ===
    print("Running test: triangle_wave")

    # Frequency of triangle wave
    f = 1.0  # Hz

    # Time vector: 0 to 2 seconds, 1000 points
    t = np.linspace(0, 2, 1000)

    # Evaluate triangle wave
    y = triangle_wave(f, t)

    # Plot
    plt.plot(t, y)
    plt.title(f"Triangle wave at {f} Hz")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()
    print()

        # === Test: raster_position ===
    print("Running test: raster_position")

    # Time vector
    T = 4000
    t = np.linspace(0, 2, T)  # Fine resolution

    # Raster parameters
    Ax, Ay = 20.0, 10.0
    fx, fy = 1.3, 5.7
    cx, cy = 0.0, 0.0

    # Compute raster trajectory
    x, y = raster_position(t, Ax, Ay, fx, fy, cx, cy)

    # Plot trajectory in XY space
    plt.plot(x, y)
    plt.title("Raster path in (x, y) space")
    plt.xlabel("x position")
    plt.ylabel("y position")
    plt.axis("equal")
    plt.grid(True)
    plt.show(block=True)

    # === Test: beam_intensity ===
    print("Running test: beam_intensity at fixed time")

    # Define 2D grid
    xvals = np.linspace(-30, 30, 200)
    yvals = np.linspace(-15, 15, 100)
    X, Y = np.meshgrid(xvals, yvals)

    # Beam parameters
    k = [20.0, 10.0, 4.0, 2.0, 0.0, 0.0, 1.3, 5.7]

    # Time (fixed)
    t = 0.5

    # Compute intensity
    I = np.array(beam_intensity(X, Y, t, k))

    # Plot
    plt.imshow(
        I,
        extent=[xvals[0], xvals[-1], yvals[0], yvals[-1]],
        origin='lower',
        aspect='equal',
        cmap='inferno'
    )
    plt.colorbar(label='Intensity')
    plt.title(f"Beam intensity at t = {t}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(False)
    plt.show()

    # === Test: beam_intensity animation ===
    print("Running test: beam_intensity animation")

    from matplotlib.animation import FuncAnimation

    # Grid
    xvals = np.linspace(-30, 30, 200)
    yvals = np.linspace(-15, 15, 100)
    X, Y = np.meshgrid(xvals, yvals)

    # Beam parameters
    k = [20.0, 10.0, 4.0, 2.0, 0.0, 0.0, 1.3, 5.7]

    # Time vector for animation
    T = 200
    t_vals = np.linspace(0, 2, T)
    
    I = np.array(beam_intensity(X, Y, t_vals[0], k))

    # Set up figure
    fig, ax = plt.subplots()
    im = ax.imshow(
        np.zeros_like(X),
        extent=[xvals[0], xvals[-1], yvals[0], yvals[-1]],
        origin='lower',
        aspect='equal',
        cmap='inferno',
        vmin=0,
        vmax=np.max(I)
    )
    
    ax.set_title("Beam intensity over time")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    cb = plt.colorbar(im, ax=ax)
    cb.set_label("Intensity")

    def update(frame):
        t = t_vals[frame]
        I = np.array(beam_intensity(X, Y, t, k))
        im.set_data(I)
        ax.set_title(f"Beam intensity at t = {t:.2f} s")
        return [im]

    ani = FuncAnimation(fig, update, frames=T, interval=50, blit=False)
    plt.show()


    # ==================================================
    # === TEST: simulate_image accumulated intensity ===
    # ==================================================
    print("Running test: simulate_image (accumulated image)")

    # Beam parameters reused: k = [20.0, 10.0, 4.0, 2.0, 0.0, 0.0, 1.3, 5.7]
    # Grid defined above

    T = 1000
    t_vals = np.linspace(0, 20, T)
    I_acc = simulate_image(X, Y, t_vals, k)

    plt.figure()
    plt.imshow(
        I_acc,
        extent=[xvals[0], xvals[-1], yvals[0], yvals[-1]],
        origin='lower',
        aspect='equal',
        cmap='inferno'
    )
    plt.colorbar(label="Accumulated energy")
    plt.title("Accumulated beam image over time")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(False)
    plt.show(block=True)

    # ==================================================
    # === TEST: 2D erf function ========================
    # ==================================================
    show_erf_result()

