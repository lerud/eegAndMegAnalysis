# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: neuroTools
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt
import mne
import ipyniivue
from ipyniivue import NiiVue, SliceType
import ipywidgets as widgets
import nibabel as nib
import os
from IPython.display import Image
from IPython.display import display
import IPython.display
import time
import colorsys

# matplotlib.use("QtAgg")
# plt.ion()

# %%
subjects_dir = os.path.expandvars("$SUBJECTS_DIR")

# condition = "speech"
# condition = "clicks"
# condition = "ebResult"
# condition = "ebSpeechResult"
condition = "ebClickResult"


volDir = "/Users/karl/speechClickNiis/"

# structVol = "bet_MNI152_T1_0.5mm.nii.gz"
structVol = "mni152.nii.gz"

# timeRange = [.008, .009]
# timeRange = [.004, .0055]
# timeRange = [.008, .009]
# magThreshRatio = .6


# This below grouping is for looking at one response, as opposed to the masked diff

# timeRange = [.0015, .0035]
# magThreshRatio = .4
# crosshairs = [0, 0, 0]

# timeRange = [.0055, .0065]
# magThreshRatio = .3
# crosshairs = [0, 0, 0]

# # Speech wave V
# timeRange = [.008, .0095]
# magThreshRatio = .55
# # magThreshRatio = .1
# crosshairs = [5, -10, 0]

# timeRange = [.010, .012]
# # magThreshRatio = .15  # This threshold shows the speech SN10 pretty well
# magThreshRatio = .4
# crosshairs = [65, 0, 0]

# Late Na for click
timeRange = [0.018, 0.020]
magThreshRatio = 0.4
crosshairs = [0, 0, 0]

# This is a good late Na picture for speech
# timeRange = [.020, .0215]
# magThreshRatio = .4
# crosshairs = [65, 0, 0]

# timeRange = [.026, .028]
# magThreshRatio = .4
# crosshairs = [0, 0, 0]

# timeRange = [.031, .034]
# magThreshRatio = .4
# crosshairs = [0, 0, 0]

# timeRange = [.041, .042]
# magThreshRatio = .6
# crosshairs = [0, 0, 0]

# This is good for the P50 of speech TRF, unthresholded
# timeRange = [.0519, .0521]
# magThreshRatio = .6
# crosshairs = [65, 0, 0]

# Approximate P50 for clicks
# timeRange = [.0565, .0615]
# magThreshRatio = .35
# crosshairs = [65, 5, 6]


# This below grouping is for looking at the masked diff

# timeRange = [.007, .008]
# magThreshRatio = 1
# crosshairs = [0, 0, 0]

# timeRange = [.0105, .0115]
# magThreshRatio = .1
# crosshairs = [65, 0, 0]

# timeRange = [.018, .019]
# magThreshRatio = .06
# crosshairs = [15, -15, 0]

# timeRange = [.025, .029]
# magThreshRatio = .3
# crosshairs = [65, 0, 0]

# timeRange = [.0325, .0345]
# magThreshRatio = .5
# crosshairs = [65, 0, 0]

# timeRange = [.041, .044]
# magThreshRatio = .2
# crosshairs = [-60, -11, 4]

# timeRange = [.048, .05]
# magThreshRatio = .2
# crosshairs = [65, 0, 0]

# timeRange = [.025, .03]
# timeRange = [.0325, .0345]
# timeRange = [.041, .044]
# timeRange = [.05, .055]
# magThreshRatio = .5

# magThreshRatio = .6
# magThreshRatio = .06

volIndsAll = []
if len(timeRange) == 1:
    volInds = np.array([int(timeRange * 5000)])
else:
    volInds = np.arange(int(timeRange[0] * 5000), int(timeRange[1] * 5000))

volIndsAll.append(volInds)

# numFrames = 345
numFrames = 1
volumesAll = []
# azimuths = np.array([250, 220, 180, 160])
# elevations = np.array([30, 0, -20, -10])
# azimuths = np.array([180, 210, 220, 210, 190, 170, 160, 170])
# elevations = np.array([-10, 10, 20, 10, 0, 10, 15, 10])

overwriteTemps = True

# x = np.linspace(0, 1, numFrames)
# azCS = sp.interpolate.CubicSpline(np.linspace(0, 1, len(azimuths)), azimuths)
# elCS = sp.interpolate.CubicSpline(np.linspace(0, 1, len(elevations)), elevations)
# azimuthsAll = azCS(x)
# elevationsAll = elCS(x)

# # Loop for getting STD/RMS for each volume
# STDs = np.zeros(numFrames)
# for volInd in np.arange(numFrames):
#     volInds = np.array([volInd])
#     volSubdir = f"{condition}/magnitude/"

#     initialImg = nib.load(f"{volDir}{volSubdir}vol{volInds[0]:04d}.nii.gz")
#     sum_data = initialImg.get_fdata().astype(np.float64) # Use float64 for precision during averaging
#     for i in range(1, len(volInds)):
#         img = nib.load(f"{volDir}{volSubdir}vol{volInds[i]:04d}.nii.gz")
#         sum_data += img.get_fdata()

#     average_data = sum_data / len(volInds)
#     STDs[volInd] = np.std(average_data)

# STDsRange = np.max(STDs) - np.min(STDs)
# STDsAlt = STDs - np.min(STDs) + .1 * STDsRange  # Start the STDs positive and near zero to prepare for dB transform
# STDsAlt = 20 * np.log10(STDsAlt)  # Calculate the magnitude in decibels
# STDsAlt = STDsAlt - np.min(STDsAlt)  # Make the range of dBs start at zero
# STDsAlt /= np.max(STDsAlt)  # Make the range from zero to one
# STDsAlt = 1 - STDsAlt  # Reverse the numbers so close to zero become close to one, and vice versa
# STDsAlt *= (1 - magThreshRatio)  # Normalize it to the remainder of the ratio

STDsAlt = np.zeros(len(volInds))

# plt.plot(STDsAlt)

# %%
for frameInd in np.arange(numFrames):
    volInds = volIndsAll[frameInd]

    average_data_all = []
    average_img_all = []
    volSubdir = f"{condition}/magnitude/"

    initialImg = nib.load(f"{volDir}{volSubdir}vol{volInds[0]:04d}.nii.gz")
    sum_data = initialImg.get_fdata().astype(
        np.float64
    )  # Use float64 for precision during averaging
    affine = initialImg.affine
    header = initialImg.header
    for i in range(1, len(volInds)):
        img = nib.load(f"{volDir}{volSubdir}vol{volInds[i]:04d}.nii.gz")
        sum_data += img.get_fdata()

    average_data = sum_data / len(volInds)
    magMax = np.max(average_data)
    magThreshAbs = magMax * (magThreshRatio + STDsAlt[frameInd])
    maskInds = average_data <= magThreshAbs
    average_data[maskInds] = 0
    average_data_all.append(average_data)
    average_img = nib.Nifti1Image(average_data, affine, header)
    average_img_all.append(average_img)
    if overwriteTemps or not os.path.exists(f"{volDir}temp0_{volInds[0]:04d}.nii.gz"):
        nib.save(average_img, f"{volDir}temp0_{volInds[0]:04d}.nii.gz")

    volSubdir = f"{condition}/upDown/"
    # volSubdir = f"{condition}/frontBack/"

    initialImg = nib.load(f"{volDir}{volSubdir}vol{volInds[0]:04d}.nii.gz")
    sum_data = initialImg.get_fdata().astype(
        np.float64
    )  # Use float64 for precision during averaging
    affine = initialImg.affine
    header = initialImg.header
    for i in range(1, len(volInds)):
        img = nib.load(f"{volDir}{volSubdir}vol{volInds[i]:04d}.nii.gz")
        sum_data += img.get_fdata()

    average_data = sum_data / len(volInds)
    average_data[maskInds] = 0
    average_data_all.append(average_data)
    average_img = nib.Nifti1Image(average_data, affine, header)
    average_img_all.append(average_img)
    if overwriteTemps or not os.path.exists(f"{volDir}temp1_{volInds[0]:04d}.nii.gz"):
        nib.save(average_img, f"{volDir}temp1_{volInds[0]:04d}.nii.gz")

    volSubdir = f"{condition}/frontBack/"
    # volSubdir = f"{condition}/leftRight/"

    initialImg = nib.load(f"{volDir}{volSubdir}vol{volInds[0]:04d}.nii.gz")
    sum_data = initialImg.get_fdata().astype(
        np.float64
    )  # Use float64 for precision during averaging
    affine = initialImg.affine
    header = initialImg.header
    for i in range(1, len(volInds)):
        img = nib.load(f"{volDir}{volSubdir}vol{volInds[i]:04d}.nii.gz")
        sum_data += img.get_fdata()

    average_data = sum_data / len(volInds)
    average_data[maskInds] = 0
    average_data_all.append(average_data)
    average_img = nib.Nifti1Image(average_data, affine, header)
    average_img_all.append(average_img)
    if overwriteTemps or not os.path.exists(f"{volDir}temp2_{volInds[0]:04d}.nii.gz"):
        nib.save(average_img, f"{volDir}temp2_{volInds[0]:04d}.nii.gz")

    # volSubdir = f"{condition}/leftRight/"

    # initialImg = nib.load(f"{volDir}{volSubdir}vol{volInds[0]:04d}.nii.gz")
    # sum_data = initialImg.get_fdata().astype(np.float64) # Use float64 for precision during averaging
    # affine = initialImg.affine
    # header = initialImg.header
    # for i in range(1, len(volInds)):
    #     img = nib.load(f"{volDir}{volSubdir}vol{volInds[i]:04d}.nii.gz")
    #     sum_data += img.get_fdata()

    # average_data = sum_data / len(volInds)
    # average_data[maskInds] = 0
    # average_data_all.append(average_data)
    # average_img = nib.Nifti1Image(average_data, affine, header)
    # average_img_all.append(average_img)
    # if overwriteTemps or not os.path.exists(f"{volDir}temp2_{volInds[0]:04d}.nii.gz"):
    #     nib.save(average_img, f"{volDir}temp2_{volInds[0]:04d}.nii.gz")

    average_data = np.atan2(average_data_all[1], average_data_all[2])

    # Run these next two lines to rotate these phases backwards by 90 degrees to make red on top
    average_data = average_data + (np.pi / 2)
    average_data = np.atan2(np.sin(average_data), np.cos(average_data))

    average_data[maskInds] = -4
    average_data_all.append(average_data)
    average_img = nib.Nifti1Image(average_data, affine, header)
    average_img_all.append(average_img)
    if overwriteTemps or not os.path.exists(f"{volDir}temp3_{volInds[0]:04d}.nii.gz"):
        nib.save(average_img, f"{volDir}temp3_{volInds[0]:04d}.nii.gz")

    # # These are for sum of squares-style plots showing how much variance e.g. each dimension explains
    # average_data = (average_data_all[3]**2) / (average_data_all[1]**2 + average_data_all[2]**2 + average_data_all[3]**2)
    # average_data = (average_data_all[1]**2 + average_data_all[2]**2) / (average_data_all[1]**2 + average_data_all[2]**2 + average_data_all[3]**2)

    # average_data[maskInds] = 0
    # average_data_all.append(average_data)
    # average_img = nib.Nifti1Image(average_data, affine, header)
    # average_img_all.append(average_img)
    # if overwriteTemps or not os.path.exists(f"{volDir}temp3_{volInds[0]:04d}.nii.gz"):
    #     nib.save(average_img, f"{volDir}temp3_{volInds[0]:04d}.nii.gz")

    volumes = []

    volumeDict = {
        "path": f"{volDir}{structVol}",
        "colormap": "gray",
        # "visible": True,
        "opacity": 1,
    }
    volumes.append(volumeDict)

    average_data = average_data_all[0]
    volumeDict = {
        # "path": f"{volDir}{volSubdir}vol{volInd:04d}.nii.gz",
        "path": f"{volDir}temp0_{volInds[0]:04d}.nii.gz",
        "colormap": "warm",
        # "colormap": "red",
        # "colormapNegative": "blue",
        # "visible": True,
        "opacity": 0.5,
        # "cal_min": np.min(average_data[average_data>0]),
        # "cal_max": np.max(average_data[average_data>0]),
        # "cal_minNeg": np.max(average_data[average_data<0]),
        # "cal_maxNeg": np.min(average_data[average_data<0]),
        # "frame_4d": 52,
    }
    # volumes.append(volumeDict)

    average_data = average_data_all[1]
    volumeDict = {
        # "path": f"{volDir}{volSubdir}vol{volInd:04d}.nii.gz",
        "path": f"{volDir}temp1_{volInds[0]:04d}.nii.gz",
        "colormap": "warm",
        # "colormap": "blue",
        # "ColormapNegative": "blueNegative",
        # "visible": True,
        "opacity": 0.5,
        # "cal_min": np.min(average_data[average_data>0]),
        # "cal_max": np.max(average_data),
        # "cal_minNeg": np.min(average_data[average_data<0]),
        # "cal_maxNeg": np.max(average_data[average_data<0])
        # "cal_min": 1e-16,
        # "cal_max": np.max(average_data),
        # "cal_minNeg": np.min(average_data[average_data<0]),
        # "cal_maxNeg": -1e-16,
        # "frame_4d": 52,
    }
    # volumes.append(volumeDict)

    average_data = average_data_all[2]
    volumeDict = {
        # "path": f"{volDir}{volSubdir}vol{volInd:04d}.nii.gz",
        "path": f"{volDir}temp2_{volInds[0]:04d}.nii.gz",
        # "colormap": "warm",
        # "colormap": "green",
        "ColormapNegative": "violet",
        # "visible": True,
        "opacity": 0.5,
        # "cal_min": np.min(average_data[average_data>0]),
        # "cal_max": np.max(average_data),
        # "cal_minNeg": np.min(average_data[average_data<0]),
        # "cal_maxNeg": np.max(average_data[average_data<0])
        # "cal_min": 1e-16,
        # "cal_max": np.max(average_data),
        # "cal_minNeg": np.min(average_data[average_data<0]),
        # "cal_maxNeg": -1e-16,
        # "frame_4d": 52,
    }
    # volumes.append(volumeDict)

    # volumeDict = {
    #     # "path": f"{volDir}{volSubdir}vol{volInd:04d}.nii.gz",
    #     "path": f"/usr/local/fsl/data/atlases/HarvardOxford/HG.nii.gz",
    #     # "colormap": "warm",
    #     # "colormap": "green",
    #     "colormap": "gray",
    #     # "visible": True,
    #     "opacity": 0.5,
    #     # "cal_min": np.min(average_data[average_data>0]),
    #     # "cal_max": np.max(average_data),
    #     # "cal_minNeg": np.min(average_data[average_data<0]),
    #     # "cal_maxNeg": np.max(average_data[average_data<0])
    #     # "cal_min": 1e-16,
    #     # "cal_max": np.max(average_data),
    #     # "cal_minNeg": np.min(average_data[average_data<0]),
    #     # "cal_maxNeg": -1e-16,
    #     # "frame_4d": 52,
    # }
    # volumes.append(volumeDict)

    # volumeDict = {
    #     # "path": f"{volDir}{volSubdir}vol{volInd:04d}.nii.gz",
    #     "path": f"/usr/local/fsl/data/atlases/HarvardOxford/bs.nii.gz",
    #     # "colormap": "warm",
    #     # "colormap": "green",
    #     "colormap": "gray",
    #     # "visible": True,
    #     "opacity": 0.5,
    #     # "cal_min": np.min(average_data[average_data>0]),
    #     # "cal_max": np.max(average_data),
    #     # "cal_minNeg": np.min(average_data[average_data<0]),
    #     # "cal_maxNeg": np.max(average_data[average_data<0])
    #     # "cal_min": 1e-16,
    #     # "cal_max": np.max(average_data),
    #     # "cal_minNeg": np.min(average_data[average_data<0]),
    #     # "cal_maxNeg": -1e-16,
    #     # "frame_4d": 52,
    # }
    # volumes.append(volumeDict)

    # volumeDict = {
    #     # "path": f"{volDir}{volSubdir}vol{volInd:04d}.nii.gz",
    #     "path": f"/usr/local/fsl/data/atlases/HarvardOxford/rt.nii.gz",
    #     # "colormap": "warm",
    #     # "colormap": "green",
    #     "colormap": "gray",
    #     # "visible": True,
    #     "opacity": 0.5,
    #     # "cal_min": np.min(average_data[average_data>0]),
    #     # "cal_max": np.max(average_data),
    #     # "cal_minNeg": np.min(average_data[average_data<0]),
    #     # "cal_maxNeg": np.max(average_data[average_data<0])
    #     # "cal_min": 1e-16,
    #     # "cal_max": np.max(average_data),
    #     # "cal_minNeg": np.min(average_data[average_data<0]),
    #     # "cal_maxNeg": -1e-16,
    #     # "frame_4d": 52,
    # }
    # volumes.append(volumeDict)

    # volumeDict = {
    #     # "path": f"{volDir}{volSubdir}vol{volInd:04d}.nii.gz",
    #     "path": f"/usr/local/fsl/data/atlases/HarvardOxford/lt.nii.gz",
    #     # "colormap": "warm",
    #     # "colormap": "green",
    #     "colormap": "gray",
    #     # "visible": True,
    #     "opacity": 0.5,
    #     # "cal_min": np.min(average_data[average_data>0]),
    #     # "cal_max": np.max(average_data),
    #     # "cal_minNeg": np.min(average_data[average_data<0]),
    #     # "cal_maxNeg": np.max(average_data[average_data<0])
    #     # "cal_min": 1e-16,
    #     # "cal_max": np.max(average_data),
    #     # "cal_minNeg": np.min(average_data[average_data<0]),
    #     # "cal_maxNeg": -1e-16,
    #     # "frame_4d": 52,
    # }
    # volumes.append(volumeDict)

    volumeDict = {
        # "path": f"{volDir}{volSubdir}vol{volInd:04d}.nii.gz",
        "path": f"/usr/local/fsl/data/atlases/HarvardOxford/all.nii.gz",
        # "colormap": "warm",
        # "colormap": "green",
        "colormap": "copper",
        # "visible": True,
        "opacity": 0.75,
        # "cal_min": np.min(average_data[average_data>0]),
        # "cal_max": np.max(average_data),
        # "cal_minNeg": np.min(average_data[average_data<0]),
        # "cal_maxNeg": np.max(average_data[average_data<0])
        # "cal_min": 1e-16,
        # "cal_max": np.max(average_data),
        # "cal_minNeg": np.min(average_data[average_data<0]),
        # "cal_maxNeg": -1e-16,
        # "frame_4d": 52,
    }
    volumes.append(volumeDict)

    average_data = average_data_all[3]
    volumeDict = {
        # "path": f"{volDir}{volSubdir}vol{volInd:04d}.nii.gz",
        "path": f"{volDir}temp3_{volInds[0]:04d}.nii.gz",
        # "colormap": "red",
        "colormap": "hsv",
        # "colormap": "twilight",
        # "ColormapNegative": "violet",
        # "visible": True,
        "opacity": 0.75,
        "cal_min": -np.pi,
        "cal_max": np.pi,
        # "cal_minNeg": np.min(average_data[average_data<0]),
        # "cal_maxNeg": np.max(average_data[average_data<0])
        # "cal_min": np.min(average_data),
        # "cal_max": np.max(average_data),
        # "cal_minNeg": np.min(average_data[average_data<0]),
        # "cal_maxNeg": -1e-16,
        # "frame_4d": 52,
    }
    volumes.append(volumeDict)
    volumesAll.append(volumes)
    print(f"Done with {frameInd}")


# %%

# for i in np.arange(340, 345):
# # for i in np.arange(240, 345):
# # for i in np.arange(228, 229):
#     azimuth = azimuthsAll[i]
#     elevation = elevationsAll[i]
#     nv = NiiVue(is_orient_cube=False)
#     nv.opts.mesh_xray = 0.02
#     nv.height = 1000
#     nv.set_slice_type("RENDER")

#     nv.load_volumes(volumesAll[i])

#     nv.set_render_azimuth_elevation(azimuth, elevation)
#     # time.sleep(2)
#     display(nv)
#     time.sleep(2)
#     # nv.draw_scene()
#     nv.save_scene(f"test{i:04d}.png")
#     nv.close()


# %%
# volumes = [
# {
#     "path": f"{volDir}vol{volInd:04d}.nii.gz",
#     "colormap": "red",
#     "colormapNegative": "blue",
#     "visible": True,
#     "opacity": 0.5,
#     "cal_min": 0.05,
#     "cal_max": 1.0,
#     "cal_minNeg": -0.05,
#     "cal_maxNeg": -1.0
#     # "frame_4d": 52,
# },
# {
#     "path": f"/Users/karl/speechFrontBack/vol{volInd:04d}.nii.gz",
#     "colormap": "violet",
#     "colormapNegative": "green",
#     "visible": True,
#     "opacity": 0.5,
#     "cal_min": 0.05,
#     "cal_max": 2.0,
#     "cal_minNeg": -0.05,
#     "cal_maxNeg": -2.0
#     # "frame_4d": 52,
# },
# ]
# meshes = [
#     {
#         "path": f"{subjects_dir}/fsaverage/surf/lh.pial",
#         "colormap": "gray",
#         "visible": True,
#         "opacity": 1.0,
#     },
#     {
#         "path": f"{subjects_dir}/fsaverage/surf/rh.pial",
#         "colormap": "gray",
#         "visible": True,
#         "opacity": 1.0,
#     },
# ]


# volumes = [
#     {
#         "path": DATA_FOLDER / "mni152.nii.gz",
#         "colormap": "gray",
#         "visible": True,
#         "opacity": 1.0,
#     },
#     {
#         "path": DATA_FOLDER / "hippo.nii.gz",
#         "colormap": "red",
#         "visible": True,
#         "opacity": 1,
#     },
# ]
# nv = NiiVue()
# nv.load_volumes(volumes)
# # nv.add_volume(average_img_all[1])

# style = {"description_width": "initial"}
# string_widget = widgets.Text(
#     value="A 10 20 C -10 -30; S R X 0 R X -0",
#     description="Slice mosaic string",
#     style=style,
# )
# string_widget.layout.width = "auto"

# nv.opts.slice_mosaic_string = string_widget.value


# def update_view(*args):
#     """Update the mosaic according to the string in the widget."""
#     nv.opts.slice_mosaic_string = string_widget.value


# string_widget.observe(update_view, "value")

# display(string_widget)
# display(nv)
# nv.height = 800
# nv.opts.is_colorbar = True
# nv.set_interpolation = True


# %%
nv = NiiVue()
nv.load_volumes(volumesAll[0])
nv.set_slice_type(slice_type=SliceType.MULTIPLANAR)
# nv.set_slice_type(slice_type=SliceType.RENDER)
# nv.load_meshes(meshes)
# nv.add_volume(volumes[0])
# nv.add_volume(volumes[1])
# nv.add_volume(volumes[2])
# nv


# nv.volumes[1].colormap = "red"
# nv.volumes[2].colormap = "blue"
# nv.volumes[0].opacity = 0.4


nv.opts.show_3d_crosshair = True
nv.opts.crosshair_color = [1, 1, 1, 0.5]
nv.opts.crosshair_width = 0.75
nv.opts.back_color = [0, 0, 0, 1]
nv.opts.is_colorbar = True
nv.opts.view_mode_hot_key = "KeyN"
nv.set_clip_plane(20, 180, 180)
nv.opts.clip_plane_color = [0, 1, 1, 1]


nv.height = 800
display(nv)
time.sleep(3)
nv.move_crosshair_in_vox(crosshairs[0], crosshairs[1], crosshairs[2])

fig = plt.figure(figsize=[13, 11])
ax = fig.add_subplot(projection="polar")

rho = np.linspace(0, 1, 100)  # Radius of 1, distance from center to outer edge
# phi = np.linspace(0, 2*np.pi, 1000) # in radians, one full circle
phi = np.linspace(-np.pi / 2, np.pi * 3 / 2, 1000)  # in radians, one full circle

RHO, PHI = np.meshgrid(rho, phi)  # get every combination of rho and phi

h = (PHI - PHI.min()) / (
    PHI.max() - PHI.min()
)  # use angle to determine hue, normalized from 0-1
h = np.flip(h)
s = RHO  # saturation is set as a function of radias
v = np.ones_like(RHO)  # value is constant

# convert the np arrays to lists. This actually speeds up the colorsys call
h, s, v = h.flatten().tolist(), s.flatten().tolist(), v.flatten().tolist()
c = [colorsys.hsv_to_rgb(*x) for x in zip(h, s, v)]
c = np.array(c)

ax.scatter(PHI, RHO, c=c)
ax.set_theta_direction(-1)
ax.set_xticks(np.array([0, np.pi / 2, np.pi, np.pi * 3 / 2]))
ax.set_xticklabels(["     Ant", "Inf", "Post     ", "Sup"])
ax.yaxis.grid(False)
ax.set_yticks([])
ax.grid(False)
ax.tick_params(labelsize=46)
ax.tick_params(colors="white")
ax.spines["polar"].set_visible(False)
ax.set_facecolor("black")
fig.set_facecolor("black")
ax.set_title("Current Dipole Orientation", color="white", fontsize=60)
# ax.text(.37, .87, "Wave V", color="black", fontsize=48, transform=ax.transAxes)
# ax.text(.2, .14, "SN10", color="black", fontsize=48, transform=ax.transAxes)
# ax.text(.11, .23, "Na", color="black", fontsize=48, transform=ax.transAxes)
# ax.text(.72, .8, "Pa", color="black", fontsize=48, transform=ax.transAxes)
# ax.text(.07, .34, "Nb", color="black", fontsize=48, transform=ax.transAxes)
# ax.text(.77, .67, "P50", color="black", fontsize=48, transform=ax.transAxes)
# fig.savefig("/Users/karl/Dropbox/UMD/ARO2026/talk/colorMap.png", bbox_inches='tight', pad_inches=0)

nv2 = NiiVue()
nv2.load_volumes(volumesAll[0])
# nv2.set_slice_type(slice_type=SliceType.MULTIPLANAR)
nv2.set_slice_type(slice_type=SliceType.RENDER)
# nv.load_meshes(meshes)
# nv.add_volume(volumes[0])
# nv.add_volume(volumes[1])
# nv.add_volume(volumes[2])
# nv


# nv.volumes[1].colormap = "red"
# nv.volumes[2].colormap = "blue"
# nv.volumes[0].opacity = 0.4


nv2.opts.show_3d_crosshair = True
nv2.opts.crosshair_color = [1, 1, 1, 0.5]
nv2.opts.crosshair_width = 0.75
nv2.opts.back_color = [0, 0, 0, 1]
nv2.opts.is_colorbar = True
nv2.opts.view_mode_hot_key = "KeyN"
nv2.set_clip_plane(20, 180, 180)
nv2.opts.clip_plane_color = [0, 1, 1, 1]


nv2.height = 800
display(nv2)
# time.sleep(3)
# nv2.move_crosshair_in_vox(crosshairs[0], crosshairs[1], crosshairs[2])

# %%
nv.move_crosshair_in_vox(65, 0, 0)
nv2.move_crosshair_in_vox(65, 0, 0)

# %%
# nv = NiiVue(is_orient_cube=True)
# nv.opts.mesh_xray = 0.02
# nv.height = 1000
# nv.set_slice_type("RENDER")


# nv.load_volumes(volumes)

# nv.set_render_azimuth_elevation(250, 30)
# display(nv)
# nv.save_scene("test.png")

# nv.set_clip_plane([[0.1, 180, 20], [0.1, 0, -20]])
# nv.set_clip_plane(0.1, 180, 20)


# cutaway_check = widgets.Checkbox(value=False, description="Cutaway")


# def on_cutaway_change(change):
#     """Set cutaway."""
#     nv.set_cutaway(bool(change))


# cutaway_check.observe(on_cutaway_change, names="value")


# planes_options = ["0", "1", "2", "3", "4", "5", "6"]

# planes_dropdown = widgets.Dropdown(
#     options=planes_options,
#     value="2",
#     description="Planes:",
# )


# shade_options = ["none", "outside", "inside"]

# shade_dropdown = widgets.Dropdown(
#     options=shade_options,
#     value="outside",
#     description="Shade:",
# )


# def on_shade_change(change):
#     """Set planes shading."""
#     new_value = change["new"]
#     clr = list(nv.opts.clip_plane_color)

#     if new_value == "none":
#         clr[3] = 0
#     elif new_value == "outside":
#         clr[3] = 0.5
#     elif new_value == "inside":
#         clr[3] = -0.5

#     nv.set_clip_plane_color(clr)


# shade_dropdown.observe(on_shade_change, names="value")


# Display Viewer

# controls = widgets.HBox([planes_dropdown, shade_dropdown, cutaway_check])

# display(controls)

# nv.height = 1000
# nv.set_render_azimuth_elevation(250, 30)
# display(nv)
# nv.save_scene("test.png")


# %%

# time.sleep(1)

# %%
# from IPython.display import Image

# Create an Image object (replace with your actual image source)
# This example loads an image from a file for demonstration
# img = Image(filename='my_image.png')

# # Access the raw data and save it to a new file
# with open("saved_image.png", "wb") as f:
#     f.write(img.data)


# display(nv)
# plt.figure
# time.sleep(1)
# nv
# nv.save_scene(f"/Users/karl/test.png")
# for i in range(5):
#     nv.set_render_azimuth_elevation(i*50, 30)
#     display(nv);nv.save_scene(f"/Users/karl/test{i}.png")

# %%
# nv = NiiVue()
# nv.load_volumes(volumes)

# nv.opts.mesh_xray = 0.02
# nv.set_slice_type("RENDER")
# # nv.set_clip_plane([[0.1, 180, 20], [0.1, 0, -20]])
# nv.set_clip_plane(0.1, 180, 20)


# cutaway_check = widgets.Checkbox(value=False, description="Cutaway")


# def on_cutaway_change(change):
#     """Set cutaway."""
#     nv.set_cutaway(bool(change))


# cutaway_check.observe(on_cutaway_change, names="value")


# planes_options = ["0", "1", "2", "3", "4", "5", "6"]

# planes_dropdown = widgets.Dropdown(
#     options=planes_options,
#     value="2",
#     description="Planes:",
# )


# def on_planes_change(change):
#     """Set clipping planes based on dropdown selection."""
#     new_value = int(change["new"])
#     planes = [[2.0, 180, 20]]  # default

#     if new_value == 1:
#         planes = [[0.1, 180, 20]]
#     elif new_value == 2:
#         planes = [
#             [0.1, 180, 20],
#             [0.1, 0, -20],
#         ]
#     elif new_value == 3:
#         planes = [
#             [0.0, 90, 0],  # right center
#             [0.0, 0, -20],  # posterior oblique
#             [0.1, 0, -90],  # inferior
#         ]
#     elif new_value == 4:
#         planes = [
#             [0.3, 270, 0],  # left
#             [0.3, 90, 0],  # right
#             [0.0, 180, 0],  # anterior
#             [0.1, 0, 0],  # posterior
#         ]
#     elif new_value == 5:
#         planes = [
#             [0.4, 270, 0],  # left
#             [0.4, 90, 0],  # right
#             [0.4, 180, 0],  # anterior
#             [0.2, 0, 0],  # posterior
#             [0.1, 0, -90],  # inferior
#         ]
#     elif new_value == 6:
#         planes = [
#             [0.4, 270, 0],  # left
#             [-0.1, 90, 0],  # right
#             [0.4, 180, 0],  # anterior
#             [0.2, 0, 0],  # posterior
#             [0.1, 0, -90],  # inferior
#             [0.3, 0, 90],  # superior
#         ]
#     else:
#         planes = [[2.0, 180, 20]]  # default

#     nv.set_clip_plane(planes)


# planes_dropdown.observe(on_planes_change, names="value")


# shade_options = ["none", "outside", "inside"]

# shade_dropdown = widgets.Dropdown(
#     options=shade_options,
#     value="outside",
#     description="Shade:",
# )


# def on_shade_change(change):
#     """Set planes shading."""
#     new_value = change["new"]
#     clr = list(nv.opts.clip_plane_color)

#     if new_value == "none":
#         clr[3] = 0
#     elif new_value == "outside":
#         clr[3] = 0.5
#     elif new_value == "inside":
#         clr[3] = -0.5

#     nv.set_clip_plane_color(clr)


# shade_dropdown.observe(on_shade_change, names="value")

# # Display Viewer

# controls = widgets.HBox([planes_dropdown, shade_dropdown, cutaway_check])

# display(controls)
# display(nv)

# nv.height = 800


# %%
# xval = np.arange(0, 2 * np.pi, 0.01)
# # yval will represent the radius (constant for a solid wheel)
# yval = np.ones_like(xval)

# # 2. Get the 'hsv' colormap, which naturally cycles through hues
# colormap = plt.get_cmap('hsv')

# # 3. Normalize the data for the colormap to span the full range (0 to 2*pi)
# norm = matplotlib.colors.Normalize(0.0, 2 * np.pi)

# # 4. Create a subplot with polar coordinates
# fig = plt.figure(figsize=(6, 6))
# ax = fig.add_subplot(1, 1, 1, polar=True)

# # 5. Use scatter plot to draw the wheel
# # The 'c' argument maps the angle (xval) to the color using the colormap
# ax.scatter(xval, yval, c=xval, s=300, cmap=colormap, norm=norm, linewidths=3) #

# # 6. Customize the plot appearance
# # Remove the radial ticks (the Y-axis values)
# ax.set_yticks([])
# # Set the direction for zero angle (optional, makes red point upwards)
# ax.set_theta_zero_location('W')
# # Set the theta direction to clockwise (optional)
# ax.set_theta_direction(1)

# # 7. Display the plot
# plt.show()

# %%
# fig = plt.figure(figsize=[13, 11])
# ax = fig.add_subplot(projection="polar")

# rho = np.linspace(0,1,100) # Radius of 1, distance from center to outer edge
# # phi = np.linspace(0, 2*np.pi, 1000) # in radians, one full circle
# phi = np.linspace(-np.pi/2, np.pi*3/2, 1000) # in radians, one full circle

# RHO, PHI = np.meshgrid(rho,phi) # get every combination of rho and phi

# h = (PHI-PHI.min()) / (PHI.max()-PHI.min()) # use angle to determine hue, normalized from 0-1
# h = np.flip(h)
# s = RHO               # saturation is set as a function of radias
# v = np.ones_like(RHO) # value is constant

# # convert the np arrays to lists. This actually speeds up the colorsys call
# h,s,v = h.flatten().tolist(), s.flatten().tolist(), v.flatten().tolist()
# c = [colorsys.hsv_to_rgb(*x) for x in zip(h,s,v)]
# c = np.array(c)

# ax.scatter(PHI, RHO, c=c)
# ax.set_theta_direction(-1)
# ax.set_xticks(np.array([0, np.pi/2, np.pi, np.pi*3/2]))
# ax.set_xticklabels(["     Ant", "Inf", "Post     ", "Sup"])
# ax.yaxis.grid(False)
# ax.set_yticks([])
# ax.grid(False)
# ax.tick_params(labelsize=46)
# ax.tick_params(colors="black")
# ax.spines["polar"].set_visible(False)
# ax.set_facecolor("black")
# fig.set_facecolor("black")
# ax.set_title("Current Dipole Orientation", color="black", fontsize=60)
# # ax.text(.37, .87, "Wave V", color="black", fontsize=48, transform=ax.transAxes)
# # ax.text(.2, .14, "SN10", color="black", fontsize=48, transform=ax.transAxes)
# # ax.text(.11, .23, "Na", color="black", fontsize=48, transform=ax.transAxes)
# # ax.text(.72, .8, "Pa", color="black", fontsize=48, transform=ax.transAxes)
# # ax.text(.07, .34, "Nb", color="black", fontsize=48, transform=ax.transAxes)
# # ax.text(.77, .67, "P50", color="black", fontsize=48, transform=ax.transAxes)
# # _ = ax.axis('off')
# fig.savefig("/Users/karl/Downloads/testPngs/colorMap.png", bbox_inches='tight', pad_inches=0)

# %%
nv.colormaps()

# %%
