import numpy as np
import matplotlib.pyplot as plt
from epics import caget
import time

# --------------------
# PVs
# --------------------
pv_im  = 'LAB:IMG-TGT:Ctrl-AMC-110:BkgArrayData'
pv_fx  = 'HOR:C1_FRQ'
pv_fy  = 'VERT:C1_FRQ'
pv_exp = 'LAB:IMG-TGT:Ctrl-AMC-110:CamExpTime-SP'
pv_Ax =  'HOR:C1_AMP'
pv_Ay = 'VER:C1_AMP'

n = 512

# --------------------
# Read PVs (single snapshot)
# --------------------
im_raw = np.asarray(caget(pv_im))
fx     = float(caget(pv_fx))
fy     = float(caget(pv_fy))
exp    = float(caget(pv_exp))
ts     = time.time()

# --------------------
# Reshape image
# --------------------
image = im_raw[:n*n].reshape((n, n))

# --------------------
# Save everything in ONE data type
# --------------------
data = {
    "image": image,
    "fx": fx,
    "fy": fy,
    "exposure_time": exp
}

np.savez(f"beam_images/beam_snapshot_{int(ts)}.npz", **data)

# --------------------
# Optional visualization
# --------------------
plt.imshow(image, origin="lower", aspect="auto")
plt.colorbar()
plt.title(f"fx={fx:.3f}, fy={fy:.3f}, exp={exp}")
plt.savefig(f"beam_images/beam_snapshot_{int(ts)}.png", dpi=150)
plt.close()