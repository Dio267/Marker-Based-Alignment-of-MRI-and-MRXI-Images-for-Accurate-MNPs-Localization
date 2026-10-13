# Marker-Based Alignment of MRI and MRXI Images for Accurate MNPs Localization

# This code was made for my bachelor thesis with the title above.
# Author: Shafa Arthadia Wiyono
# The thesis is supervised / helped by Mishuk Mitra.
# Claude (Anthropic, Claude Sonnet 5.5) helped me to write and tidy this code.

# %%
## Libraries ##
import imageio.v2 as iio
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
from matplotlib.widgets import Slider
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
# %%
## Read the images ##
phantom_vol = iio.volread(r'/Users/wiyon/OneDrive/Desktop/mag', format="DICOM")
print(phantom_vol.shape)
print(phantom_vol.dtype)
# %%
## The bright marker patch (8x8 pixels)
marker = phantom_vol[42, 112:120, 116:124].astype(float)

# %%
## Slider viewer, to scroll through the z slices
def view_volume(volume, title="Volume", cmap="gray"):
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)
    img = ax.imshow(volume[0], cmap=cmap)
    ax.set_title(f"{title} — z index = 0")

    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
    slider = Slider(ax_slider, "z", 0, volume.shape[0] - 1, valinit=0, valstep=1)

    # this runs every time the slider is moved
    def update(val):
        z = int(slider.val)
        img.set_data(volume[z])
        ax.set_title(f"{title} — z index = {z}")
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()
    return slider

# %%
## Cut out the part of the volume that we need (z = 30 to 62)
z_start, z_end = 30, 63
y_start, y_end = 180, 254
x_start, x_end = 65, 139

phantom = phantom_vol[z_start:z_end, y_start:y_end, x_start:x_end].astype(float).copy()
print(phantom.shape)   
# %%
## Keep a clean copy, before the synthetic markers are pasted in
phantom_clean = phantom.copy()
# %%
## Paste the 4 synthetic markers into phantom (only for testing the angle method) 
x1, y1 = 3, 3
y2 = 62
x3, y3 = 62, 3

z_M123 = 30 - z_start
z_M4 = 62 - z_start

phantom[z_M123, y1:y1 + marker.shape[0], x1:x1 + marker.shape[1]] = marker  # M1
phantom[z_M123, y2:y2 + marker.shape[0], x1:x1 + marker.shape[1]] = marker  # M2
phantom[z_M123, y3:y3 + marker.shape[0], x3:x3 + marker.shape[1]] = marker  # M3
phantom[z_M4, y1:y1 + marker.shape[0], x1:x1 + marker.shape[1]] = marker    # M4
# %%
## Look at the phantom with the synthetic markers 
slider_ref = view_volume(phantom, title="Cropped phantom (with synthetic markers)")
# %%
## Rotate the phantom with the markers by known angles 
# small angles
# angle_xy = 2
# angle_yz = 1  
# angle_zx = 2

# middle angles
angle_xy = 7.91
angle_yz = 5.36
angle_zx = 7.67

# big angles
# angle_xy = 15
# angle_yz = 10
# angle_zx = 15

rotated_phantom = phantom.copy()
rotated_phantom = ndi.rotate(rotated_phantom, angle=angle_xy, axes=(1, 2), reshape=True, order=2, mode="nearest")
rotated_phantom = ndi.rotate(rotated_phantom, angle=angle_yz, axes=(0, 1), reshape=True, order=2, mode="nearest")
rotated_phantom = ndi.rotate(rotated_phantom, angle=angle_zx, axes=(0, 2), reshape=True, order=2, mode="nearest")
print(rotated_phantom.shape)
# %%
## Look at the rotated phantom 
slider_ref_rotated = view_volume(rotated_phantom, title="Rotated phantom (with synthetic markers)")
# %%
## Find the centers of the markers 
def find_marker_centroids_v2(volume, threshold_k=3, min_size=20, verbose=True):
    # everything brighter than mean + k * std is a marker
    threshold = volume.mean() + threshold_k * volume.std()
    mask = volume > threshold

    # label the blobs (diagonal neighbors count as connected)
    structure = np.ones((3, 3, 3))
    labeled, num_blobs = ndi.label(mask, structure=structure)
    sizes = ndi.sum(mask, labeled, range(1, num_blobs + 1))

    # only keep the blobs that are big enough
    real_blob_ids = []
    for i in range(num_blobs):
        if sizes[i] >= min_size:
            real_blob_ids.append(i + 1)

    if verbose:
        print(f"num_blobs raw: {num_blobs}, sizes: {sizes}")
        print(f"kept {len(real_blob_ids)} blobs after min_size={min_size} filter")

    centroids = ndi.center_of_mass(volume, labeled, real_blob_ids)
    centroid_list = []
    for c in centroids:
        centroid_list.append(np.array(c))
    return centroid_list

centroids_merged = find_marker_centroids_v2(rotated_phantom, threshold_k=3, min_size=20)
for i, c in enumerate(centroids_merged, start=1):
    print(f"Blob {i}: z={c[0]:.2f}, y={c[1]:.2f}, x={c[2]:.2f}")

# %%
## Marker positions (z, y, x), copied from the printout above 
# big angles
# M1_ref = np.array([9.91, 6.83, 64.12])
# M2_detected = np.array([16.06, 21.90, 6.76])
# M3_detected = np.array([9.40, 79.57, 20.82])
# M4_detected = np.array([65.84, 29.67, 20.28])

# small angles
# M1_ref = np.array([1.60, 9.27, 7.07])
# M2_detected = np.array([0.55, 7.20, 65.95])
# M3_detected = np.array([1.04, 68.21, 9.08])
# M4_detected = np.array([36, 10.10, 8.24])

# middle angles
M1_ref = np.array([8.27, 15.01, 7.13])
M2_detected = np.array([4.96, 73.76, 14.88])
M3_detected = np.array([4.80, 7.01, 65.51])
M4_detected = np.array([49.62, 18.86, 13.11])
# %%
## Plot: M1 is the reference, expected line vs actual line, and the angle ##
fig, axarr = plt.subplots(1, 3, figsize=(18, 6))
# M1 vs M2
ax = axarr[0]
z0 = round(M1_ref[0])
if z0 < phantom.shape[0]:
    ax.imshow(phantom[z0], cmap="gray")
ax.plot([M1_ref[2], M1_ref[2]], [M1_ref[1], M2_detected[1]], 'g--', linewidth=1.5, label="expected (M1's x)")
ax.plot([M1_ref[2], M2_detected[2]], [M1_ref[1], M2_detected[1]], color='orange', linewidth=1.5, label="actual path (M1→M2)")
ax.scatter(M1_ref[2], M1_ref[1], color='lime', s=60, label="M1 (ref)")
ax.scatter(M2_detected[2], M2_detected[1], color='cyan', s=60, label="M2 (actual)")
angle_12 = np.degrees(np.arctan2(M2_detected[2] - M1_ref[2], M2_detected[1] - M1_ref[1]))
ax.annotate(f"{angle_12:.2f}°", (M1_ref[2] + 3, M1_ref[1] + 8), color="yellow", fontsize=11)
ax.legend(fontsize=8); ax.set_title("M1 vs M2")

# M1 vs M3
ax = axarr[1]
if z0 < phantom.shape[0]:
    ax.imshow(phantom[z0], cmap="gray")
ax.plot([M1_ref[2], M3_detected[2]], [M1_ref[1], M1_ref[1]], 'g--', linewidth=1.5, label="expected (M1's y)")
ax.plot([M1_ref[2], M3_detected[2]], [M1_ref[1], M3_detected[1]], color='orange', linewidth=1.5, label="actual path (M1→M3)")
ax.scatter(M1_ref[2], M1_ref[1], color='lime', s=60, label="M1 (ref)")
ax.scatter(M3_detected[2], M3_detected[1], color='cyan', s=60, label="M3 (actual)")
angle_13 = np.degrees(np.arctan2(M3_detected[1] - M1_ref[1], M3_detected[2] - M1_ref[2]))
ax.annotate(f"{angle_13:.2f}°", (M1_ref[2] + 8, M1_ref[1] + 3), color="yellow", fontsize=11)
ax.legend(fontsize=8); ax.set_title("M1 vs M3")

# M1 vs M4
ax = axarr[2]
if z0 < phantom.shape[0]:
    ax.imshow(phantom[z0], cmap="gray")
ax.plot([M1_ref[2], M4_detected[2]], [M1_ref[1], M4_detected[1]], color='orange', linewidth=1.5, label="actual path (M1→M4)")
ax.scatter(M1_ref[2], M1_ref[1], color='lime', s=60, label="M1 (ref)")
ax.scatter(M4_detected[2], M4_detected[1], color='cyan', s=60, label="M4 (actual)")
drift = np.sqrt((M4_detected[2] - M1_ref[2]) ** 2 + (M4_detected[1] - M1_ref[1]) ** 2)
ax.annotate(f"drift = {drift:.2f}px", (M1_ref[2] + 3, M1_ref[1] + 8), color="yellow", fontsize=11)
ax.legend(fontsize=8); ax.set_title("M1 vs M4 (x,y drift)")

plt.tight_layout()
plt.show()
# %%
## Rotate the clean phantom (no markers) by the same angles
rotated_phantom_clean = phantom_clean.copy()
rotated_phantom_clean = ndi.rotate(rotated_phantom_clean, angle=angle_xy, axes=(1, 2), reshape=True, order=2, mode="nearest")
rotated_phantom_clean = ndi.rotate(rotated_phantom_clean, angle=angle_yz, axes=(0, 1), reshape=True, order=2, mode="nearest")
rotated_phantom_clean = ndi.rotate(rotated_phantom_clean, angle=angle_zx, axes=(0, 2), reshape=True, order=2, mode="nearest")
print(rotated_phantom_clean.shape)
# %%
## Rotate it back with the opposite angles (in reverse order) 
## this is the corrected phantom (no synthetic markers) 
corrected_phantom = rotated_phantom_clean.copy()
corrected_phantom = ndi.rotate(corrected_phantom, angle=-angle_zx, axes=(0, 2), reshape=True, order=2, mode="nearest")
corrected_phantom = ndi.rotate(corrected_phantom, angle=-angle_yz, axes=(0, 1), reshape=True, order=2, mode="nearest")
corrected_phantom = ndi.rotate(corrected_phantom, angle=-angle_xy, axes=(1, 2), reshape=True, order=2, mode="nearest")
print(corrected_phantom.shape)
# %%
## Look at the corrected phantom 
slider_corrected = view_volume(corrected_phantom, title="Corrected phantom (clean, no synthetic markers)")
# %%
## Segmentation at 1 mm: find the 4 capsules 
def segment_capsules(vol, k, dz, rows, cols, sigma=2, min_distance=15,
                     exclude_border=5, rel_depth=0.5, title=""):
    # average the slices k-dz ... k+dz, this gives less noise
    lo = max(k - dz, 0)
    hi = min(k + dz + 1, vol.shape[0])
    img = vol[lo:hi, rows, cols].mean(axis=0)

    # smooth the image, the capsules are dark so turn them into bright peaks
    smooth = ndi.gaussian_filter(img, sigma=sigma)
    depth = np.median(smooth) - smooth
    depth[depth < 0] = 0

    # find the 4 peaks (one for every capsule)
    peaks = peak_local_max(depth, min_distance=min_distance, num_peaks=4,
                           exclude_border=exclude_border)
    if len(peaks) != 4:
        raise ValueError(f"{title}: found {len(peaks)} peaks instead of 4, change k/dz, sigma or min_distance")
    print(f"{title}: capsule peaks (row, col) = {peaks.tolist()}")

    # watershed splits the image into 4 areas, one around every peak
    markers = np.zeros(img.shape, dtype=int)
    for i in range(4):
        markers[peaks[i][0], peaks[i][1]] = i + 1
    basins = watershed(-depth, markers)

    # in every area only keep the pixels that are deeper than half of the peak
    seg = np.zeros(img.shape, dtype=int)
    for i in range(4):
        r = peaks[i][0]
        c = peaks[i][1]
        region = (basins == i + 1) & (depth >= rel_depth * depth[r, c])
        region = ndi.binary_fill_holes(region)
        lab, n = ndi.label(region)
        if n > 1:
            region = lab == lab[r, c]    # more than one piece: keep the piece with the peak
        seg[region] = i + 1
    return img, seg

def disk_struct(radius):
    # small round shape (True inside the circle), used to round the capsules
    size = 2 * radius + 1
    disk = np.zeros((size, size), dtype=bool)
    for y in range(-radius, radius + 1):
        for x in range(-radius, radius + 1):
            if x * x + y * y <= radius * radius:
                disk[y + radius, x + radius] = True
    return disk

def round_regions(seg, radius=2):
    disk = disk_struct(radius)
    out = np.zeros_like(seg)
    for i in range(1, 5):
        # opening removes thin spikes, fill_holes closes holes
        region = ndi.binary_opening(seg == i, structure=disk)
        region = ndi.binary_fill_holes(region)
        lab, n = ndi.label(region)
        if n == 0:
            continue    # nothing left of this capsule
        if n > 1:
            # more than one piece: keep the biggest one
            biggest = 1
            biggest_size = 0
            for j in range(1, n + 1):
                size = (lab == j).sum()
                if size > biggest_size:
                    biggest = j
                    biggest_size = size
            region = lab == biggest
        out[region] = i
    return out
# %%
## Corrected phantom: segmentation at 1 mm 
img_corr, seg_corr = segment_capsules(
    corrected_phantom, k=30, dz=3,
    rows=slice(17, 87), cols=slice(17, 87), title="Corrected"
)
round_corr = round_regions(seg_corr, radius=2)
binary_1mm = round_corr > 0
# %%
## Rotated phantom (not corrected): segmentation at 1 mm
img_rot, seg_rot = segment_capsules(
    rotated_phantom_clean, k=21, dz=1,
    rows=slice(15, 75), cols=slice(11, 70), title="Rotated"
)
round_rot = round_regions(seg_rot, radius=2)
binary_1mm_rot = round_rot > 0
# %%
# MRI slices that are used for the segmentation 
# the segmentation itself averages the slices around them (k +/- dz)
slice_img_rot = rotated_phantom_clean[21, 15:75, 11:70]
slice_img_corr = corrected_phantom[30, 17:87, 17:87]

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(slice_img_rot, cmap="gray", interpolation="nearest")
axes[0].set_title("Rotated (uncorrected) MRI slice (z=21)")
axes[1].imshow(slice_img_corr, cmap="gray", interpolation="nearest")
axes[1].set_title("Corrected MRI slice (z=30)")
for ax in axes:
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_aspect("equal")

plt.tight_layout()
plt.show()
# %%
## Both 1 mm masks next to each other 
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(binary_1mm_rot, cmap="gray", interpolation="nearest", vmin=0, vmax=1)
axes[0].set_title("Rotated (uncorrected) – 1 mm segmentation")
axes[1].imshow(binary_1mm, cmap="gray", interpolation="nearest", vmin=0, vmax=1)
axes[1].set_title("Corrected – 1 mm segmentation")
for ax in axes:
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_aspect("equal")
plt.tight_layout()
plt.show()
# %%
## Make the 1 mm masks smaller: 5 mm voxels (majority vote) ##
def majority_vote_downsample(mask_1mm, block=5, threshold=0.5):
    h, w = mask_1mm.shape
    n_rows = h // block    # how many voxels fit in (the rest is cut off)
    n_cols = w // block
    result = np.zeros((n_rows, n_cols), dtype=bool)
    for i in range(n_rows):
        for j in range(n_cols):
            part = mask_1mm[i * block:(i + 1) * block, j * block:(j + 1) * block]
            if part.mean() >= threshold:
                result[i, j] = True
    return result

block = 5
binary_5mm = majority_vote_downsample(binary_1mm, block=block)
binary_5mm_rot = majority_vote_downsample(binary_1mm_rot, block=block)
print("Corrected mask: 1 mm", binary_1mm.shape, "-> 5 mm", binary_5mm.shape)
print("Rotated mask:   1 mm", binary_1mm_rot.shape, "-> 5 mm", binary_5mm_rot.shape)
# %%
## Put the 5 mm masks in the middle of the 20 x 20 MRXI grid ##
Nx, Ny = 20, 20

def center_fit(mask, target_h, target_w):
    out = np.zeros((target_h, target_w), dtype=bool)
    h, w = mask.shape
    if h > target_h or w > target_w:
        raise ValueError("Mask is larger than the target MRXI grid")
    top = (target_h - h) // 2
    left = (target_w - w) // 2
    out[top:top + h, left:left + w] = mask
    return out

mask_rotated_fitted = center_fit(binary_5mm_rot, Nx, Ny)
mask_corrected_fitted = center_fit(binary_5mm, Nx, Ny)
# %%
## Both 5 mm masks next to each other 
fig, axarr = plt.subplots(1, 2, figsize=(10, 5))
axarr[0].imshow(mask_rotated_fitted, cmap="gray", interpolation="nearest", vmin=0, vmax=1)
axarr[0].set_title("Rotated (uncorrected) – 5 mm MRXI mask")
axarr[1].imshow(mask_corrected_fitted, cmap="gray", interpolation="nearest", vmin=0, vmax=1)
axarr[1].set_title("Corrected – 5 mm MRXI mask")
for ax in axarr:
    ax.set_xlabel("x (5 mm voxels)")
    ax.set_ylabel("y (5 mm voxels)")
    ax.set_aspect("equal")
plt.tight_layout()
plt.show()
# %%
## Save the masks 
np.save(r"C:\Users\wiyon\OneDrive\Desktop\mask_rotated_fitted.npy", mask_rotated_fitted)
np.save(r"C:\Users\wiyon\OneDrive\Desktop\mask_corrected.npy", mask_corrected_fitted)
