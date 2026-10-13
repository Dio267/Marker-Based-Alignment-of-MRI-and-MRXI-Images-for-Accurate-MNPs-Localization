# Marker-Based Alignment of MRI and MRXI Images for Accurate MNPs Localization

Code for my bachelor thesis with the title above.

- Author: Shafa Arthadia Wiyono
- The thesis is supervised / helped by Mishuk Mitra.
- Claude (Anthropic, Claude Sonnet 5.5) helped me to write and tidy this code.

## What this is about

Magnetorelaxometry imaging (MRXI) measures how much iron there is, but it shows
no anatomy. MRI shows the anatomy and the capsules with iron as dark spots. To
use the MRI as a region of interest (ROI) for the MRXI reconstruction, the
phantom in the MRI must be aligned correctly. This script tests whether markers
can be used to find a tilt of the phantom, corrects the tilt, and makes the 5 mm
masks that the MRXI reconstruction uses as ROI.

## What the script does

The script `Marker-Based_Alignment.py` is made of cells (`# %%`) that are run one
after the other, for example in VS Code.

1. **Read the images.** It reads the MRI volume (DICOM) and cuts out the part
   with the phantom (z = 30 to 62, 33 x 74 x 74 voxels at 1 mm).
2. **Markers.** It keeps a clean copy of the phantom and makes a second copy with
   4 synthetic markers (M1, M2, M3 in the top slice and M4 in the bottom slice).
3. **Tilt.** It rotates the phantom with the markers by known angles (xy, yz, zx).
   There are three angle sets in the script: small, middle and big.
4. **Marker centres.** It finds the centres of the markers in the rotated
   phantom (everything brighter than mean + 3 x std, connected blobs).
5. **Angle check.** It uses M1 as reference and calculates the angle of M2 and
   M3 and the drift of M4. The plots show the expected line against the actual
   line.
6. **Corrected phantom.** It rotates the clean phantom (no markers) by the same
   angles to get a "tilted scan", and rotates it back with the opposite angles
   to get the corrected phantom.
7. **Capsule segmentation at 1 mm.** For the tilted and the corrected phantom it
   finds the 4 capsules:
   - averages a few slices and smooths the image,
   - takes the 4 darkest peaks (one per capsule),
   - splits the image with a watershed, one area per peak,
   - keeps the pixels that are deeper than half of the peak,
   - rounds the shapes.
8. **5 mm masks.** It makes the masks smaller to 5 mm voxels (majority vote of
   5 x 5 pixels), puts them in the middle of the 20 x 20 MRXI grid and saves them.

## Output

| File | Content |
|---|---|
| `mask_corrected.npy` | 20 x 20 mask of the corrected phantom |
| `mask_rotated_fitted.npy` | 20 x 20 mask of the tilted (not corrected) phantom |

The MRXI reconstruction loads these files as ROI.

## How to run

1. Install the libraries: `pip install imageio numpy scipy matplotlib scikit-image`
2. In the cell `## Read the images ##` set the path of your DICOM folder.
3. In the last cell set the folders where the masks are saved.
4. Run the cells one after the other.

## Notes

- The corrected phantom is made with the known angles. The angles estimated from
  the markers (step 5) are not used for it.
- The marker positions in step 5 are copied by hand from the printout of step 4.
  They belong to one angle set, so change them when the angles change.
- The slices and crop windows of the segmentation (step 7) are set for the middle
  angle set. Check them again for other angles.
- Saving the masks overwrites old files with the same names, for example the
  masks that were made with an Otsu threshold.
- The 5 mm masks are made of square voxels, so the round capsules look like
  small blocks there. The 1 mm masks keep the round shape.
