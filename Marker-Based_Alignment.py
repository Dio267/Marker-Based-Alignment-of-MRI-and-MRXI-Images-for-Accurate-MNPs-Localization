# %%
## Libraries ## 
import imageio.v2 as iio 
import matplotlib.pyplot as plt 
import numpy as np 
import scipy.ndimage as ndi
from matplotlib.widgets import Slider

# %% 
## image read ##
phantom_vol = iio.volread(r'/Users/wiyon/OneDrive/Desktop/mag', format="DICOM")

print(phantom_vol.shape)
print(phantom_vol.dtype)

# %%
## Image cropping ##
phantom_marker = phantom_vol[42, 96:170, 65:139]
phantom_target = phantom_vol[42, 180:254, 65:139]
marker = phantom_vol[42, 112:120, 116:124]
top = phantom_vol[30, 180:254, 65:139]
bottom = phantom_vol[62, 180:254, 65:139]

# %%
## Image conversion ##
phantom_marker = phantom_marker.astype(float)
phantom_target = phantom_target.astype(float)
marker = marker.astype(float)
top = phantom_vol[30, 180:254, 65:139].astype(float).copy()
bottom = phantom_vol[62, 180:254, 65:139].astype(float).copy()

# %% 
## Image visualization ##
fig, axarr = plt.subplots(2, 3)
axarr[0,0].imshow(phantom_marker, cmap="gray")
axarr[0,1].imshow(phantom_target, cmap="gray")
axarr[1,0].imshow(marker, cmap="gray")
axarr[1,1].imshow(top, cmap="gray")
axarr[1,2].imshow(bottom, cmap="gray")
axarr[0,0].set_title("Mid Phantom (Instance 42)")
axarr[0,1].set_title("Bottom Phantom (Instance 42)")
axarr[1,0].set_title("Marker capsule (Instance 42)")
axarr[1,1].set_title("Top (Instance 30)")
axarr[1,2].set_title("Bottom (Instance 63)")
plt.show()

# %%
## Reusable function to browse any 3D volume slice-by-slice ##
def view_volume(volume, title="Volume", cmap="gray"):
    """
    volume: 3D numpy array, shape (z, y, x)
    Opens an interactive figure with a slider to scroll through z-slices.
    """
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)
    img = ax.imshow(volume[0], cmap=cmap)
    ax.set_title(f"{title} — z index = 0")

    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
    slider = Slider(ax_slider, "z", 0, volume.shape[0] - 1, valinit=0, valstep=1)

    def update(val):
        z = int(slider.val)
        img.set_data(volume[z])
        ax.set_title(f"{title} — z index = {z}")
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()
    return slider   # keep a reference so the slider stays interactive

# %%
## Crop the 3D sub-volume: z=30 through z=62 (real phantom, no markers yet) ##
z_start, z_end = 30, 63   # 63 exclusive, so index 62 is included
y_start, y_end = 180, 254
x_start, x_end = 65, 139

phantom = phantom_vol[z_start:z_end, y_start:y_end, x_start:x_end].astype(float).copy()

print(phantom.shape)   # should be (33, 74, 74)

# %%
## Paste all 4 markers into `phantom`, using LOCAL coordinates ##
x1, y1 = 1, 1
y2 = 64
x3, y3 = 64, 1

z_M1_M2_M3 = 30 - z_start   # = 0
z_M4       = 62 - z_start   # = 32

# M1, M2, M3 all sit at local z = 0
phantom[z_M1_M2_M3, y1:y1 + marker.shape[0], x1:x1 + marker.shape[1]] = marker  # M1
phantom[z_M1_M2_M3, y2:y2 + marker.shape[0], x1:x1 + marker.shape[1]] = marker  # M2
phantom[z_M1_M2_M3, y3:y3 + marker.shape[0], x3:x3 + marker.shape[1]] = marker  # M3

# M4 sits at local z = 32
phantom[z_M4, y1:y1 + marker.shape[0], x1:x1 + marker.shape[1]] = marker        # M4

# %%
## View the cropped phantom with markers pasted in ##
slider_ref = view_volume(phantom, title="Cropped phantom (z=30-62) with markers", cmap="gray")

# %%
## Rotate the full phantom (with markers) in xy, yz, zx ##
angle_xy = 7.91
angle_yz = 5.36
angle_zx = 7.67

rotated_phantom = phantom.copy()
# rotated_phantom = ndi.rotate(rotated_phantom, angle=angle_xy, axes=(1, 2), reshape=False, order=1, mode="nearest")
rotated_phantom = ndi.rotate(rotated_phantom, angle=angle_yz, axes=(0, 1), reshape=False, order=1, mode="nearest")
# rotated_phantom = ndi.rotate(rotated_phantom, angle=angle_zx, axes=(0, 2), reshape=False, order=1, mode="nearest")

print(rotated_phantom.shape)   # should still be (33, 74, 74)

# %%
## Top and bottom slices again, but from the ROTATED phantom ##
top_after = rotated_phantom[0, :, :]     # same index as your original top marker group
bottom_after = rotated_phantom[32, :, :]  # same index as your original bottom marker

fig, axarr = plt.subplots(1, 2, figsize=(10, 5))
axarr[0].imshow(top_after, cmap="gray")
axarr[0].set_title("Top slice (z index 0), after rotation")
axarr[1].imshow(bottom_after, cmap="gray")
axarr[1].set_title("Bottom slice (z index 32), after rotation")
plt.show()

# %%
## The centroids of the markers (top slices) ## 
# marker_centroid_top = [(z, 4.25, 4.25), # M1
#                     (z, 4.25, 68.5),    # M2
#                     (z, 68.5, 4.25),    # M3
#                     (z, 4.25, 4.25)]    # M4