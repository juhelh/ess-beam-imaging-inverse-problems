import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter



def normalize_l2(img, eps=1e-12):
    return img / (np.linalg.norm(img) + eps)

def remove_background(I, sigma=30):
    bg = gaussian_filter(I, sigma=sigma)
    return np.clip(I - bg, 0, None)
    
def tight_crop_bbox(img, frac=0.02, pad=10, min_h=32, min_w=32):
    H, W = img.shape
    m = float(np.max(img))
    if m <= 0:
        return 0, H, 0, W

    thr = frac * m
    mask = img > thr
    if not np.any(mask):
        return 0, H, 0, W

    ys = np.where(np.any(mask, axis=1))[0]
    xs = np.where(np.any(mask, axis=0))[0]

    y1, y2 = int(ys[0]), int(ys[-1] + 1)
    x1, x2 = int(xs[0]), int(xs[-1] + 1)

    y1 = max(0, y1 - pad)
    y2 = min(H, y2 + pad)
    x1 = max(0, x1 - pad)
    x2 = min(W, x2 + pad)

    if (y2 - y1) < min_h:
        c = (y1 + y2) // 2
        y1 = max(0, c - min_h // 2)
        y2 = min(H, y1 + min_h)
        y1 = max(0, y2 - min_h)

    if (x2 - x1) < min_w:
        c = (x1 + x2) // 2
        x1 = max(0, c - min_w // 2)
        x2 = min(W, x1 + min_w)
        x1 = max(0, x2 - min_w)

    return y1, y2, x1, x2


def crop_image(img, bbox):
    y1, y2, x1, x2 = bbox
    return img[y1:y2, x1:x2]


def crop_grids(X, Y, bbox):
    y1, y2, x1, x2 = bbox
    return X[y1:y2, x1:x2], Y[y1:y2, x1:x2]

# ----------------------------
# Debug / visualization only
# ----------------------------
def show_image(img, title=""):
    plt.figure(figsize=(6, 4))
    plt.imshow(img, cmap="inferno", origin="lower")
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.show(block=True)

# ----------------------------
# Debug / visualization only
# ----------------------------
def show_bbox(img, bbox, title=""):
    y1, y2, x1, x2 = bbox
    plt.figure(figsize=(6, 4))
    plt.imshow(img, cmap="inferno", origin="lower")
    plt.gca().add_patch(
        plt.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            edgecolor="cyan", facecolor="none", linewidth=2
        )
    )
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.show(block=True)