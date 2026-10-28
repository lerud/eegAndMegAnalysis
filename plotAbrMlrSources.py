# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: neuroTools
#     language: python
#     name: python3
# ---

# %%
import mne
import numpy as np
import eelbrain as eb
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt
import joblib
import os
import subprocess
import sys
import pickle
import glob
import time
import nibabel
import nilearn
import nilearn.image as nimg
import importlib
import pandas as pd
import seaborn as sb

from prepTrfObjs import *
from neuroAndSignalTools.utilityFuncs import doRsync
from neuroAndSignalTools.utilityFuncs import calcPC

matplotlib.use("QtAgg")
plt.ion()

# %%
subjects_dir = os.path.expandvars("$SUBJECTS_DIR")
parentDir = "/Volumes/Seagate/map/"

    
subjectsToAverageAll = [
    # "R3045",
    # "R3089",
    # "R3093",
    # "R3095",
    # "R3151",
    # "R2774",
    # "R3152",
    # "R2877",
    # "R3157",
    # "R2783",
    # "R3170",
    # "R3172",
    # "R3193",
    # "R3184",
    # "R3214",
    "R3265",
    "R3275",
    "R3247",
    "R3292",
    "R3293",
    "R3294",
    "R3303",
    "R3304",
    "R3306",
    "R3313",
    "R3317",
    "R3320",
]

# subjectsToAverageAll = ["R3303"]
clickList = ["R3265", "R3275", "R3247", "R3292", "R3293", "R3294", "R3303", "R3304", "R3306", "R3313", "R3317", "R3320"]
conditions = ["T: Quiet (real)", "T: All (real)", "D: All (real)", "M: Easy (real)", "M: Hard (real)", "T: Clicks complete (real)"]
lambdaInd = 0
# condInd = 0
# lenResponse = 850
lenResponse = 500
fs = 5000

makeNIfTIs = False

sourceParamsSuffix = "_depth0.8"  # This is the filename sourceParamsSuffix that is used for the main analysis in the paper, made with snr 9
# sourceParamsSuffix = "_snr3.0"
# sourceParamsSuffix = "_snr9_invMethod_sLORETA"
# sourceParamsSuffix = "_snr9_invMethod_eLORETA"
# sourceParamsSuffix = "_snr9_invMethod_dSPM"
# sourceParamsSuffix = "_snr11_invMethod_dSPM"

# sourceParamsSuffixClicks = "_depth0.8"  # This is the filename sourceParamsSuffix that is used for the main analysis in the paper, made with snr 9
# sourceParamsSuffix = "_snr3.0"
# sourceParamsSuffix = "_snr9_invMethod_sLORETA"
# sourceParamsSuffix = "_snr9_invMethod_eLORETA"
# sourceParamsSuffix = "_snr9_invMethod_dSPM"
sourceParamsSuffixClicks = sourceParamsSuffix

avgBrain = "fsaverage"
# avgBrain = "mni"

# saveSuffix = f"_mixed_vector_{avgBrain}"
# saveSuffix = f"_mixed_None_{avgBrain}"
# saveSuffix = f"_volOnly_vector_{avgBrain}"
# saveSuffix = f"_volOnly_None_{avgBrain}"

# saveSuffix = f"_mixed_vector"
# saveSuffix = f"_mixed_None"
saveSuffix = f"_volOnly_vector"
# saveSuffix = f"_volOnly_None"


# saveSuffix = f"_mixed_vector_{avgBrain}"
# saveSuffix = f"_mixed_None_{avgBrain}"
# saveSuffix = f"_volOnly_vector_{avgBrain}"
# saveSuffix = f"_volOnly_None_{avgBrain}"

# saveSuffix = f"_mixed_vector"
# saveSuffix = f"_mixed_None"
# saveSuffixClicks = f"_volOnly_vector_{avgBrain}_zeroPhase"
# saveSuffix = f"_volOnly_None"
saveSuffixClicks = saveSuffix

filePrefix = "newSources_"

subDirs = "/eegAndMeg/meg/"

pos = 8.0
spacing = "oct5"

doXhemi = False
    
bem_fs = eb.load.unpickle(f"{subjects_dir}/{avgBrain}/bem/conductorModel.pickle")
bem_fs_sym = eb.load.unpickle(f"{subjects_dir}/fsaverage_sym/bem/conductorModel.pickle")



# %%
def flipLeftRight(stc, src):
    if isinstance(stc, mne.VolVectorSourceEstimate):
        totalNumVertices_here = np.sum([src[i]["nuse"] for i in range(len(src))])
        tmin = stc.tmin
        tstep = stc.tstep
        n_times=len(stc.times)

        # stc_upDown, dirs_used = stc.project(directions=np.tile(np.array([0, 0, 1]), (totalNumVertices_here, 1)))
        stc_upDown = mne.VolSourceEstimate(
            data=stc.data[:, 2, :],
            vertices=[src[0]["vertno"]],
            tmin=tmin,
            tstep=tstep,
            subject="fsaverage_sym",  # Optional: FreeSurfer subject name
        )
        # stc_frontBack, dirs_used = stc.project(directions=np.tile(np.array([0, 1, 0]), (totalNumVertices_here, 1)))
        stc_frontBack = mne.VolSourceEstimate(
            data=stc.data[:, 1, :],
            vertices=[src[0]["vertno"]],
            tmin=tmin,
            tstep=tstep,
            subject="fsaverage_sym",  # Optional: FreeSurfer subject name
        )
        # stc_leftRight, dirs_used = stc.project(directions=np.tile(np.array([1, 0, 0]), (totalNumVertices_here, 1)))
        stc_leftRight = mne.VolSourceEstimate(
            data=stc.data[:, 0, :],
            vertices=[src[0]["vertno"]],
            tmin=tmin,
            tstep=tstep,
            subject="fsaverage_sym",  # Optional: FreeSurfer subject name
        )

        volume = stc_upDown.as_volume(src=src, dest="mri")
        data = volume.get_fdata()
        flipped_data = np.flip(data, axis=0)
        nx, ny, nz, nt = flipped_data.shape
        # data_2d_upDown = flipped_data.reshape(-1, nt)


        # 2. Extract vertex spatial coordinates from the source space
        # MNE source spaces store positions ('rr') in meters.
        # We extract the positions for the active vertices used in the volume source space.
        src_vol = src[0]
        vertices_m = src_vol['rr'][src_vol['vertno']] 

        # Convert meters to millimeters for NIfTI RAS space mapping
        vertices_mm = vertices_m * 1000.0

        # 3. Transform millimeter coordinates to NIfTI voxel indices (i, j, k)
        # We invert the NIfTI affine matrix to map real-world coords -> grid indices
        inv_affine = np.linalg.inv(volume.affine)

        # Add a 4th row of ones to the vertices for matrix multiplication (homogeneous coordinates)
        vertices_hom = np.hstack([vertices_mm, np.ones((len(vertices_mm), 1))])
        voxel_coords = (inv_affine @ vertices_hom.T).T

        # Round to the nearest integer index and cast to int
        voxel_indices = np.round(voxel_coords[:, :3]).astype(int)

        # 4. Extract data from the dense 4D grid back into 2D (vertices x time)
        # Extract only the voxels that match our original source space vertices
        data_2d_upDown = np.zeros((len(voxel_indices), nt))

        for i, (x, y, z) in enumerate(voxel_indices):
            # Ensure coordinates fall safely within image boundaries to prevent IndexError
            if (0 <= x < flipped_data.shape[0] and 
                0 <= y < flipped_data.shape[1] and 
                0 <= z < flipped_data.shape[2]):
                data_2d_upDown[i, :] = flipped_data[x, y, z, :]

        # 5. Recreate the VolSourceEstimate using original vertex indices
        # We pull the exact active vertex IDs from the original source space object
        original_vertices = [src_vol['vertno']]





        volume = stc_frontBack.as_volume(src=src, dest="mri")
        data = volume.get_fdata()
        flipped_data = np.flip(data, axis=0)
        # affine = np.eye(4)
        # flipped_frontBack = nimg.new_img_like(nibabel.Nifti1Image(np.zeros(flipped_data.shape), affine), flipped_data)
        # nx, ny, nz, nt = flipped_data.shape
        # data_2d_frontBack = flipped_data.reshape(-1, nt)
        # nonzero_idx_frontBack = np.nonzero(np.any(data_2d_frontBack != 0, axis=1))[0]
        # filtered_data_frontBack = data_2d[nonzero_idx]

        data_2d_frontBack = np.zeros((len(voxel_indices), nt))
        
        for i, (x, y, z) in enumerate(voxel_indices):
            # Ensure coordinates fall safely within image boundaries to prevent IndexError
            if (0 <= x < flipped_data.shape[0] and 
                0 <= y < flipped_data.shape[1] and 
                0 <= z < flipped_data.shape[2]):
                data_2d_frontBack[i, :] = flipped_data[x, y, z, :]




        volume = stc_leftRight.as_volume(src=src, dest="mri")
        data = volume.get_fdata()
        flipped_data = np.flip(data, axis=0)
        # affine = np.eye(4)
        # flipped_leftRight = nimg.new_img_like(nibabel.Nifti1Image(np.zeros(flipped_data.shape), affine), flipped_data)
        # nx, ny, nz, nt = flipped_data.shape
        # data_2d_leftRight = flipped_data.reshape(-1, nt)
        # nonzero_idx_leftRight = np.nonzero(np.any(data_2d_leftRight != 0, axis=1))[0]
        # filtered_data_leftRight = data_2d[nonzero_idx]

        data_2d_leftRight = np.zeros((len(voxel_indices), nt))
        
        for i, (x, y, z) in enumerate(voxel_indices):
            # Ensure coordinates fall safely within image boundaries to prevent IndexError
            if (0 <= x < flipped_data.shape[0] and 
                0 <= y < flipped_data.shape[1] and 
                0 <= z < flipped_data.shape[2]):
                data_2d_leftRight[i, :] = flipped_data[x, y, z, :]

        # lengths = np.array([len(nonzero_idx_upDown), len(nonzero_idx_frontBack), len(nonzero_idx_leftRight)])
        # nonzero_idx_toUse = [nonzero_idx_upDown, nonzero_idx_frontBack, nonzero_idx_leftRight][lengths.argmax()]

        # print(f"Each of three nonzero idx lengths are {lengths}")
        # print(f"and the total number of vertices for this source space should be {totalNumVertices_here}")
        # print("\n")

        data_out = np.zeros((totalNumVertices_here, 3, n_times))
        # data_out[:, 0, :] = data_2d_leftRight[src[0]["vertno"]]
        # data_out[:, 1, :] = data_2d_frontBack[src[0]["vertno"]]
        # data_out[:, 2, :] = data_2d_upDown[src[0]["vertno"]]
        data_out[:, 0, :] = data_2d_leftRight
        data_out[:, 1, :] = data_2d_frontBack
        data_out[:, 2, :] = data_2d_upDown

        stc_out = mne.VolVectorSourceEstimate(
            data=data_out,
            vertices=original_vertices,
            tmin=tmin,
            tstep=tstep,
            subject="fsaverage_sym",  # Optional: FreeSurfer subject name
        )

        return stc_out


        

# %%
if saveSuffix[:4] == "_vol":

    volume_label = None
    
    src_fs = mne.setup_volume_source_space(
        avgBrain,
        mri=subjects_dir + f"/{avgBrain}/mri/aparc.a2009s+aseg.mgz",
        pos=pos,
        bem=bem_fs,
        volume_label=volume_label,
        subjects_dir=subjects_dir,
        add_interpolator=True,  # just for speed, usually this should be True
        verbose=True,
    )
    
    src_fs_sym = mne.setup_volume_source_space(
        "fsaverage_sym",
        mri=subjects_dir + f"/fsaverage_sym/mri/aparc.a2009s+aseg.mgz",
        pos=pos,
        bem=bem_fs_sym,
        volume_label=volume_label,
        subjects_dir=subjects_dir,
        add_interpolator=True,  # just for speed, usually this should be True
        verbose=True,
    )

    totalNumVertices_sym = np.sum([src_fs_sym[i]["nuse"] for i in range(len(src_fs_sym))])

    
elif saveSuffix[:4] == "_mix":
    
    volume_label = [
        "Left-Thalamus-Proper",
        "Right-Thalamus-Proper",
        "Brain-Stem",
    ]
    
    src_surf_fs = mne.setup_source_space(
        subject=avgBrain, spacing=spacing, add_dist=False, subjects_dir=subjects_dir
    )    
    src_vol_fs = mne.setup_volume_source_space(
        avgBrain,
        mri=subjects_dir + f"/{avgBrain}/mri/aseg.mgz",
        pos=pos,
        bem=bem_fs,
        volume_label=volume_label,
        subjects_dir=subjects_dir,
        add_interpolator=True,  # just for speed, usually this should be True
        verbose=True,
    )
    src_fs = src_surf_fs + src_vol_fs


# volume_label = None
# bem_fs = eb.load.unpickle(f"{subjects_dir}/fsaverage/bem/conductorModel.pickle")
# src_fs = mne.setup_volume_source_space(
#     "fsaverage",
#     mri=subjects_dir + "/fsaverage/mri/aparc+aseg.mgz",
#     pos=5.0,
#     bem=bem_fs,
#     volume_label=volume_label,
#     subjects_dir=subjects_dir,
#     add_interpolator=True,  # just for speed, usually this should be True
#     verbose=True,
# )

totalNumVertices = np.sum([src_fs[i]["nuse"] for i in range(len(src_fs))])
    
# if saveSuffix == "_mixed_None" or saveSuffix == "_volOnly_None":
#     stcAll = np.zeros((totalNumVertices, lenResponse, len(subjectsToAverageAll), len(conditions)))
#     stcAllBeam = np.zeros((totalNumVertices, lenResponse, len(subjectsToAverageAll), len(conditions)))
# elif saveSuffix == "_mixed_vector" or saveSuffix == "_volOnly_vector":
#     stcAll = np.zeros((totalNumVertices, 3, lenResponse, len(subjectsToAverageAll), len(conditions)))
#     stcAllBeam = np.zeros((totalNumVertices, 3, lenResponse, len(subjectsToAverageAll), len(conditions)))
    
dipsAll = []
# peakTimeRanges = [[-.007, .07], [-.01, .01], [-.01, .01], [-.02, .02]]
# peakTimeRanges = [[-.008, .061], [-0.0105,  0.0585]]
peakTimeRanges = [[-.008, .061], [-.008, .061]]
stcSubsetsMNE = []
stcSubsetsBeam = []
stcSubsetsXhemi_orig = []
stcSubsetsXhemi_oppo = []
for i in range(len(peakTimeRanges)):
    lenT = len(np.arange(peakTimeRanges[i][0], peakTimeRanges[i][1], 1 / fs))
    if saveSuffix[9:13] == "None" or saveSuffix[7:11] == "None":
        stcSubsetsMNE.append(np.zeros((totalNumVertices, lenT, len(subjectsToAverageAll), len(conditions))))
        stcSubsetsBeam.append(np.zeros((totalNumVertices, lenT, len(subjectsToAverageAll), len(conditions))))
    elif saveSuffix[9:15] == "vector" or saveSuffix[7:13] == "vector":
        stcSubsetsMNE.append(np.zeros((totalNumVertices, 3, lenT, len(subjectsToAverageAll), len(conditions))))
        stcSubsetsBeam.append(np.zeros((totalNumVertices, 3, lenT, len(subjectsToAverageAll), len(conditions))))
        if doXhemi:
            stcSubsetsXhemi_orig.append(np.zeros((totalNumVertices_sym, 3, lenT, len(subjectsToAverageAll), 3)))  # For now, only going to do this for three conditions, mix easy and hard which together are the speech mix, and clicks, which we need two variables for, original hemispheres and opposite hemispheres
            stcSubsetsXhemi_oppo.append(np.zeros((totalNumVertices_sym, 3, lenT, len(subjectsToAverageAll), 3)))

waveVsAll = np.zeros((len(subjectsToAverageAll), len(conditions)))

for subjInd, subject in enumerate(subjectsToAverageAll):

    megLocation = parentDir + subject + subDirs

    dipsHere = []
    mixInd = 0
    for condInd in range(len(conditions)):

        if conditions[condInd][3] == "C":
            tempLocation = doRsync(f"{megLocation}{filePrefix}{conditions[condInd]}_lambda{lambdaInd}{sourceParamsSuffixClicks}{saveSuffixClicks}.pickle")
        else:
            tempLocation = doRsync(f"{megLocation}{filePrefix}{conditions[condInd]}_lambda{lambdaInd}{sourceParamsSuffix}{saveSuffix}.pickle")
        src, stc, stc_fs, dips, stcBeam, stcBeam_fs = eb.load.unpickle(tempLocation)

        if doXhemi:
            morph_sym = mne.compute_source_morph(
                src=src,
                subject_from=subject,
                subject_to="fsaverage_sym",
                src_to=src_fs_sym,
                subjects_dir=subjects_dir,
                niter_sdr=[5, 5, 2],
                niter_affine=[5, 5, 2],
                zooms=7,  # just for speed
                verbose=True,
            )
    
        print("\n")
        print(f"Done loading all source data for subject {subject}")
        print("\n")
        # stcAll[..., subjInd, condInd] = stc_fs.data
        # stcAllBeam[..., subjInd, condInd] = stcBeam_fs.data
        print("\n")

        subsetsMNE = []
        subsetsBeam = []  # This way we save an stc object for each time range, to use later for plotting after replacing the data with the averages
        subsetsXhemi_orig = []
        subsetsXhemi_oppo = []
        for i in range(len(peakTimeRanges)):

            # peakInd 0 will be simply time locked at TRF time, and peakInd 1 will be time locked to subject's wave V time
            if i == 0:
                timeInd = 0.008
            elif i == 1:
                timeInd = dips[0][0][0].times.mean()  # It should work to get the wave V time from the first dipole that was calculated, but those are not always correct from the previous step of source analysis, ...
                # timeInd = eb.load.unpickle(f"{parentDir}{subject}/timesFive.pickle")[condInd][0]  # ...so we can read the time directly from the manually-checked individual subject times matrix
                # timeInd = eb.load.unpickle(f"{parentDir}{subject}/timesFiveSourceFile.pickle")[condInd][0]  # ...so we can read the time directly from the manually-checked individual subject times matrix
                # timeInd = eb.load.unpickle(f"{parentDir}{subject}/timesFiveNew.pickle")[condInd][0]  # ...so we can read the time directly from the manually-checked individual subject times matrix
                # timeInd = eb.load.unpickle(f"{parentDir}{subject}/timesFiveUpdated.pickle")[condInd][0]  # ...so we can read the time directly from the manually-checked individual subject times matrix
                waveVsAll[subjInd, condInd] = timeInd
                
            subsetsMNE.append(stc_fs.copy().crop(tmin=timeInd+peakTimeRanges[i][0], tmax=timeInd+peakTimeRanges[i][1]))
            subsetsBeam.append(stcBeam_fs.copy().crop(tmin=timeInd+peakTimeRanges[i][0], tmax=timeInd+peakTimeRanges[i][1]))
            
            tempData = subsetsMNE[i].data
            # minLen = np.min((tempData.shape[-1], stcSubsetsMNE[i].shape[-3]))
            # tempData = tempData[..., :minLen]
            # stcSubsetsMNE[i][..., :tempData.shape[-1], subjInd, condInd] = tempData
            stcSubsetsMNE[i][..., subjInd, condInd] = tempData[..., :lenT]
            
            tempData = subsetsBeam[i].data
            # minLen = np.min((tempData.shape[-1], stcSubsetsBeam[i].shape[-3]))
            # tempData = tempData[..., :minLen]
            # stcSubsetsBeam[i][..., :tempData.shape[-1], subjInd, condInd] = tempData
            stcSubsetsBeam[i][..., subjInd, condInd] = tempData[..., :lenT]

            if doXhemi:
                if conditions[condInd][0] == "M" or conditions[condInd][3] == "C":  # If we are on either the mix easy or mix hard or the clicks, get the xhemi stcs, morphing straight from subject space to fs_sym space
                    stc_fs_sym = morph_sym.apply(stc)
                    # stc_fs_sym_xhemi = morph_sym_xhemi.apply(stc_fs_sym)  # This technique is not going to work for volume source solutions, so use NIfTI volumes instead
                    stc_fs_sym_xhemi = flipLeftRight(stc_fs_sym, src_fs_sym)                
                    subsetsXhemi_orig.append(stc_fs_sym.copy().crop(tmin=timeInd+peakTimeRanges[i][0], tmax=timeInd+peakTimeRanges[i][1]))
                    subsetsXhemi_oppo.append(stc_fs_sym_xhemi.copy().crop(tmin=timeInd+peakTimeRanges[i][0], tmax=timeInd+peakTimeRanges[i][1]))
                    tempData = subsetsXhemi_orig[i].data
                    stcSubsetsXhemi_orig[i][..., subjInd, mixInd] = tempData[..., :lenT]
                    tempData = subsetsXhemi_oppo[i].data
                    stcSubsetsXhemi_oppo[i][..., subjInd, mixInd] = tempData[..., :lenT]
                    print("Checking on Xhemi mats")
                    print(f"Condition {conditions[condInd]}")
                    print(f"len of stcSubsets for Xhemi is {len(stcSubsetsXhemi_orig)}")
                    print(f"size of stcSubset mats for Xhemi is {stcSubsetsXhemi_orig[i].shape}")
                    print("\n")

        if conditions[condInd][0] == "M" or conditions[condInd][3] == "C":
            mixInd += 1
    
            
        print(f"Done writing average-space stc to main matrices")
        dipsHere.append(dips)
    dipsAll.append(dipsHere)

if doXhemi:
    # Be super lazy and remake these here
    subsetsXhemi_orig = []
    subsetsXhemi_oppo = []
    for i in range(2):
        subsetsXhemi_orig.append(stc_fs_sym.copy().crop(tmin=timeInd+peakTimeRanges[i][0], tmax=timeInd+peakTimeRanges[i][1]))
        subsetsXhemi_oppo.append(stc_fs_sym_xhemi.copy().crop(tmin=timeInd+peakTimeRanges[i][0], tmax=timeInd+peakTimeRanges[i][1]))

# Since we read in both the mix easy and mix hard conditions, average them, move the average into the mix easy position, and then slide the click over one to keep it the last
for i in range(len(peakTimeRanges)):
    print(stcSubsetsMNE[i].shape)
    print(stcSubsetsBeam[i].shape)
    stcSubsetsMNE[i][..., 3] = stcSubsetsMNE[i][..., 3:5].mean(axis=-1)
    stcSubsetsMNE[i][..., 4] = stcSubsetsMNE[i][..., 5]
    stcSubsetsMNE[i] = stcSubsetsMNE[i][..., :5]
    stcSubsetsBeam[i][..., 3] = stcSubsetsBeam[i][..., 3:5].mean(axis=-1)
    stcSubsetsBeam[i][..., 4] = stcSubsetsBeam[i][..., 5]
    stcSubsetsBeam[i] = stcSubsetsBeam[i][..., :5]
    if doXhemi:
        stcSubsetsXhemi_orig[i][..., 0] = stcSubsetsXhemi_orig[i][..., 0:2].mean(axis=-1)  # Also average the xhemi stcs over the two mixes, easy and hard, which are the only two conditions for those that we should have here
        stcSubsetsXhemi_oppo[i][..., 0] = stcSubsetsXhemi_oppo[i][..., 0:2].mean(axis=-1)  # same thing for the left-right reversed
        stcSubsetsXhemi_orig[i][..., 1] = stcSubsetsXhemi_orig[i][..., 2]  # Move the clicks into the spot where the second mix was
        stcSubsetsXhemi_orig[i][..., 2] = stcSubsetsXhemi_orig[i][..., 0] - stcSubsetsXhemi_orig[i][..., 1]  # ...and take the speech minus click difference for the third spot
        stcSubsetsXhemi_oppo[i][..., 1] = stcSubsetsXhemi_oppo[i][..., 2]  # ...and do the same things for the left-right reversed responses
        stcSubsetsXhemi_oppo[i][..., 2] = stcSubsetsXhemi_oppo[i][..., 0] - stcSubsetsXhemi_oppo[i][..., 1]

    print(stcSubsetsMNE[i].shape)
    print(stcSubsetsBeam[i].shape)
    if doXhemi:
        print(stcSubsetsXhemi_orig[i].shape)
        print(stcSubsetsXhemi_oppo[i].shape)

waveVsAll[:, 3] = waveVsAll[:, 3:5].mean(axis=-1)
waveVsAll[:, 4] = waveVsAll[:, 5]
waveVsAll = waveVsAll[:, :5]

conditions = ["T: Quiet (real)", "T: All (real)", "D: All (real)", "M: All (real)", "T: Clicks complete (real)"]
    
    
    # morph = mne.compute_source_morph(
    # src,
    # subject_from=subject,
    # subject_to="fsaverage",
    # src_to=src_fs,
    # subjects_dir=subjects_dir,
    # niter_sdr=[5, 5, 2],
    # niter_affine=[5, 5, 2],
    # zooms=7,  # just for speed
    # verbose=True,
    # )

    # source_rr = np.concatenate([_src["rr"][_src["vertno"], :] for _src in src], axis=0)
    
    # dipoles = [dips[0][0][0], dips[0][1][0], dips[0][2][0], dips[0][3][0]]
    # for i in range(len(dipoles)):
    #     for j in range(len(dipoles[i].pos)):
    #         location = np.linalg.norm(dipoles[i].pos[j] - source_rr, axis=1).argmin()
    #         dipoles[i].pos[j] = source_rr[location, :]
    # stc_dips = mne.inverse_sparse.make_stc_from_dipoles(dipoles, src, verbose=None)
    # stcDips1 = morph.apply(stc_dips)
    # stcAllDips1[:, :, :, subjInd] = stcDips1.data
    
    # dipoles = [dips[0][0][0], dips[1][1][0], dips[1][2][0], dips[1][3][0]]
    # for i in range(len(dipoles)):
    #     for j in range(len(dipoles[i].pos)):
    #         location = np.linalg.norm(dipoles[i].pos[j] - source_rr, axis=1).argmin()
    #         dipoles[i].pos[j] = source_rr[location, :]
    # stc_dips = mne.inverse_sparse.make_stc_from_dipoles(dipoles, src, verbose=None)
    # stcDips1and2 = morph.apply(stc_dips)
    # stcAllDips1and2[:, :, :, subjInd] = stcDips1and2.data

# %%
# print(stcAll.shape)
# stcAllAvg = stcAll.mean(axis=3)
# print(stcAllBeam.shape)
# stcAllBeamAvg = stcAllBeam.mean(axis=3)


# print(stcAllDips1.shape)
# stcAllDips1Avg = stcAllDips1.mean(axis=3)
# stcDips1.data = stcAllDips1Avg
# print(stcAllDips1and2.shape)
# stcAllDips1and2Avg = stcAllDips1and2.mean(axis=3)
# stcDips1and2.data = stcAllDips1and2Avg

# %%
# initial_time = .0086

# condToPlot = 0
# if saveSuffix == "_mixed_None" or saveSuffix == "_volOnly_None":
#     axesToMean = 2
# elif saveSuffix == "_mixed_vector" or saveSuffix == "_volOnly_vector":
#     axesToMean = 3
# mneTitle = f"MNE: {conditions[condToPlot]}"
# beamTitle = f"Beamformer: {conditions[condToPlot]}"

# condToPlot = np.arange(len(conditions)).astype(int)
# if saveSuffix == "_mixed_None" or saveSuffix == "_volOnly_None":
#     axesToMean = (2, 3)
# elif saveSuffix == "_mixed_vector" or saveSuffix == "_volOnly_vector":
#     axesToMean = (3, 4)
# mneTitle = f"MNE: Average of conditions"
# beamTitle = f"Beamformer: Average of conditions"

# stc_fs.data = stcAll[..., condToPlot].mean(axis=axesToMean).squeeze()
# mneBrain = stc_fs.plot_3d(
#     hemi="both",
#     src=src_fs,
#     views="coronal",
#     initial_time=initial_time,
#     subjects_dir=subjects_dir,
#     brain_kwargs=dict(silhouette=True),
#     smoothing_steps=0,
#     show_traces=True,
# )
# mne.viz.set_3d_title(mneBrain, mneTitle, size=20)

# stcBeam_fs.data = stcAllBeam[..., condToPlot].mean(axis=axesToMean).squeeze()
# beamBrain = stcBeam_fs.plot_3d(
#     hemi="both",
#     src=src_fs,
#     views="coronal",
#     initial_time=initial_time,
#     subjects_dir=subjects_dir,
#     brain_kwargs=dict(silhouette=True),
#     smoothing_steps=0,
#     show_traces=True,
# )
# mne.viz.set_3d_title(beamBrain, beamTitle, size=20)

# %%
doInitialPlots = False
initial_time = None
peakInd = 1
takeNorms = False
brain_alpha = 0.4
surface = "white"

condToPlot = 3
condDiffToPlot = 4
# if saveSuffix == "_mixed_None" or saveSuffix == "_volOnly_None":
#     axesToMean = 2
# elif saveSuffix == "_mixed_vector" or saveSuffix == "_volOnly_vector":
#     axesToMean = 3


group1mat = stcSubsetsMNE[peakInd][..., condToPlot]
group2mat = stcSubsetsMNE[peakInd][..., condDiffToPlot]
group3mat = stcSubsetsMNE[peakInd][..., 1]
group4mat = stcSubsetsMNE[peakInd][..., 2]
firstTitleSubset = f"MNE: {conditions[condToPlot]}, peak number {peakInd}"
secondTitleSubset = f"MNE: {conditions[condDiffToPlot]}, peak number {peakInd}"
stcFirst_fs_here = subsetsMNE[peakInd].copy()
stcSecond_fs_here = subsetsMNE[peakInd].copy()
stcDiff_fs_here = subsetsMNE[peakInd].copy()

stcTarget_fs_here = subsetsMNE[peakInd].copy()
stcDistractor_fs_here = subsetsMNE[peakInd].copy()

# group1mat = stcSubsetsBeam[peakInd][..., condToPlot]
# group2mat = stcSubsetsBeam[peakInd][..., condDiffToPlot]
# firstTitleSubset = f"Beamformer: {conditions[condToPlot]}, peak number {peakInd}"
# secondTitleSubset = f"Beamformer: {conditions[condDiffToPlot]}, peak number {peakInd}"
# stcFirst_fs_here = subsetsBeam[peakInd].copy()
# stcSecond_fs_here = subsetsBeam[peakInd].copy()
# stcDiff_fs_here = subsetsBeam[peakInd].copy()


diffTitleSubset = f"Difference: {conditions[condToPlot]} - {conditions[condDiffToPlot]}, peak number {peakInd}"

group1mat /= np.std(group1mat)
group2mat /= np.std(group2mat)
group3mat /= np.std(group3mat)
group4mat /= np.std(group4mat)

if takeNorms:
    group1mat = np.linalg.norm(group1mat, axis=1)
    group2mat = np.linalg.norm(group2mat, axis=1)
    group3mat = np.linalg.norm(group3mat, axis=1)
    group4mat = np.linalg.norm(group4mat, axis=1)

    # This is just to initialize them, the data doesn't matter
    stcFirst_fs_here = mne.VolSourceEstimate(data = np.linalg.norm(stcFirst_fs_here.data, axis=1), vertices=stcFirst_fs_here.vertices, tmin=stcFirst_fs_here.tmin, tstep=stcFirst_fs_here.tstep)
    stcSecond_fs_here = mne.VolSourceEstimate(data = np.linalg.norm(stcSecond_fs_here.data, axis=1), vertices=stcSecond_fs_here.vertices, tmin=stcSecond_fs_here.tmin, tstep=stcSecond_fs_here.tstep)
    stcDiff_fs_here = mne.VolSourceEstimate(data = np.linalg.norm(stcDiff_fs_here.data, axis=1), vertices=stcDiff_fs_here.vertices, tmin=stcDiff_fs_here.tmin, tstep=stcDiff_fs_here.tstep)
    
    stcTarget_fs_here = mne.VolSourceEstimate(data = np.linalg.norm(stcTarget_fs_here.data, axis=1), vertices=stcTarget_fs_here.vertices, tmin=stcTarget_fs_here.tmin, tstep=stcTarget_fs_here.tstep)
    stcDistractor_fs_here = mne.VolSourceEstimate(data = np.linalg.norm(stcDistractor_fs_here.data, axis=1), vertices=stcDistractor_fs_here.vertices, tmin=stcDistractor_fs_here.tmin, tstep=stcDistractor_fs_here.tstep)

# condToPlot = np.arange(len(conditions)).astype(int)
# if saveSuffix == "_mixed_None" or saveSuffix == "_volOnly_None":
#     axesToMean = (2, 3)
# elif saveSuffix == "_mixed_vector" or saveSuffix == "_volOnly_vector":
#     axesToMean = (3, 4)
# mneTitleSubset = f"MNE: Average of conditions, peak number {peakInd + 1}"
# beamTitleSubset = f"Beamformer: Average of conditions, peak number {peakInd + 1}"


stcFirst_fs_here.data = group1mat.mean(axis=-1)
stcSecond_fs_here.data = group2mat.mean(axis=-1)
stcDiff_fs_here.data = (group1mat - group2mat).mean(axis=-1)
stcTarget_fs_here.data = group3mat.mean(axis=-1)
stcDistractor_fs_here.data = group4mat.mean(axis=-1)

if doInitialPlots:

    firstBrainSubset = mne.viz.plot_source_estimates(stcFirst_fs_here,
        subject=avgBrain,
        surface=surface,
        hemi="both",
        src=src_fs,
        views="coronal",
        initial_time=initial_time,
        subjects_dir=subjects_dir,
        brain_kwargs=dict(silhouette=True),
        smoothing_steps=7,
        show_traces=True,
        alpha=brain_alpha,
    )
    mne.viz.set_3d_title(firstBrainSubset, firstTitleSubset, size=20)
    
    secondBrainSubset = mne.viz.plot_source_estimates(stcSecond_fs_here,
        subject=avgBrain,
        surface=surface,
        hemi="both",
        src=src_fs,
        views="coronal",
        initial_time=initial_time,
        subjects_dir=subjects_dir,
        brain_kwargs=dict(silhouette=True),
        smoothing_steps=7,
        show_traces=True,
        alpha=brain_alpha,
    )
    mne.viz.set_3d_title(secondBrainSubset, secondTitleSubset, size=20)
    
    diffBrainSubset = mne.viz.plot_source_estimates(stcDiff_fs_here,
        subject=avgBrain,
        surface=surface,
        hemi="both",
        src=src_fs,
        views="coronal",
        initial_time=initial_time,
        subjects_dir=subjects_dir,
        brain_kwargs=dict(silhouette=True),
        smoothing_steps=7,
        show_traces=True,
        alpha=brain_alpha,
    )
    mne.viz.set_3d_title(diffBrainSubset, diffTitleSubset, size=20)

# fig = stc_fs_here.plot(initial_time=initial_time, src=src_fs, subjects_dir=subjects_dir)
# fig2 = stcBeam_fs_here.plot(initial_time=initial_time, src=src_fs, subjects_dir=subjects_dir)

# %%

# %%
# from nilearn.datasets import load_mni152_template
# from nilearn.plotting import plot_anat

# subjInd = subjectsToAverageAll.index("R3157")
# numDipoleFitsInd = 1
# timeInd = 0
# dipoleFitInd = 1
# subject = subjectsToAverageAll[subjInd]
# trans = f"/Volumes/Seagate/map/{subject}/eegAndMeg/eeg/eeg-trans.fif"
# dip = dipsAll[subjInd][numDipoleFitsInd][timeInd][dipoleFitInd]
# # dip.plot_locations(trans, subject, subjects_dir, mode="orthoview")
# mniPos = dip.to_mni(subject, trans, subjects_dir=subjects_dir)

# template = load_mni152_template(resolution=1)

# fig_template = plot_anat(
#     template, cut_coords=mniPos[0], title="Dipole loc. (MNI Space)"
# )

# %%
doTests = False
saveTests = False
loadTests = True

for peakInd in range(1, 2):
# for peakInd in range(1):
# for peakInd in range(2):
    
    numCondsAll = stcSubsetsMNE[peakInd].shape[-1]
    condSTDsMNE = [np.std(stcSubsetsMNE[peakInd][..., j]) for j in range(numCondsAll)]
    condSTDsBeam = [np.std(stcSubsetsBeam[peakInd][..., j]) for j in range(numCondsAll)]

    stcListsMNE = [[] for _ in range(numCondsAll)]
    stcListsBeam = [[] for _ in range(numCondsAll)]

    if doXhemi:
        numCondsXhemi = stcSubsetsXhemi_orig[peakInd].shape[-1]
        condSTDsXhemi_orig = [np.std(stcSubsetsXhemi_orig[peakInd][..., j]) for j in range(numCondsXhemi)]
        condSTDsXhemi_oppo = [np.std(stcSubsetsXhemi_oppo[peakInd][..., j]) for j in range(numCondsXhemi)]

        stcListsXhemi_orig = [[] for _ in range(numCondsXhemi)]
        stcListsXhemi_oppo = [[] for _ in range(numCondsXhemi)]

    
    srcSavePath = f"{subjects_dir}/{avgBrain}/mneVolSourceSpace-src.fif"
    src_fs.save(srcSavePath, overwrite=True)
        
    srcSavePath_sym = f"{subjects_dir}/fsaverage_sym/mneVolSourceSpace-src.fif"
    src_fs_sym.save(srcSavePath_sym, overwrite=True)
            
    
    for j in range(numCondsAll):
        for i in range(len(subjectsToAverageAll)):
            stcTemp = subsetsMNE[peakInd].copy()
            data = stcSubsetsMNE[peakInd][..., i, j].squeeze() / condSTDsMNE[j]
            stcTemp.data = data
            stcListsMNE[j].append(stcTemp)
    
            stcTemp = subsetsBeam[peakInd].copy()
            data = stcSubsetsBeam[peakInd][..., i, j].squeeze() / condSTDsBeam[j]
            stcTemp.data = data
            stcListsBeam[j].append(stcTemp)

    if doXhemi:
        for j in range(numCondsXhemi):
            for i in range(len(subjectsToAverageAll)):
                stcTemp = subsetsXhemi_orig[peakInd].copy()
                data = stcSubsetsXhemi_orig[peakInd][..., i, j].squeeze() / condSTDsXhemi_orig[j]
                stcTemp.data = data
                stcListsXhemi_orig[j].append(stcTemp)
            
                stcTemp = subsetsXhemi_oppo[peakInd].copy()
                data = stcSubsetsXhemi_oppo[peakInd][..., i, j].squeeze() / condSTDsXhemi_oppo[j]
                stcTemp.data = data
                stcListsXhemi_oppo[j].append(stcTemp)
    

    if doXhemi:  # This will create the proper speech minus click difference TRFs
        for i in range(len(subjectsToAverageAll)):
            stcListsXhemi_orig[2][i].data = stcListsXhemi_orig[0][i].data - stcListsXhemi_orig[1][i].data
            stcListsXhemi_oppo[2][i].data = stcListsXhemi_oppo[0][i].data - stcListsXhemi_oppo[1][i].data
    
    ebGroup0MNE = eb.load.mne.stc_ndvar(stc=stcListsMNE[0], subject=avgBrain, src="vol-8", subjects_dir=subjects_dir, sss_filename=srcSavePath, parc="aparc.a2009s+aseg")
    ebGroup0MNE_norm = ebGroup0MNE.norm("space")
    ebGroup1MNE = eb.load.mne.stc_ndvar(stc=stcListsMNE[1], subject=avgBrain, src="vol-8", subjects_dir=subjects_dir, sss_filename=srcSavePath, parc="aparc.a2009s+aseg")
    ebGroup1MNE_norm = ebGroup1MNE.norm("space")
    ebGroup2MNE = eb.load.mne.stc_ndvar(stc=stcListsMNE[2], subject=avgBrain, src="vol-8", subjects_dir=subjects_dir, sss_filename=srcSavePath, parc="aparc.a2009s+aseg")
    ebGroup2MNE_norm = ebGroup2MNE.norm("space")
    ebGroup3MNE = eb.load.mne.stc_ndvar(stc=stcListsMNE[3], subject=avgBrain, src="vol-8", subjects_dir=subjects_dir, sss_filename=srcSavePath, parc="aparc.a2009s+aseg")
    ebGroup3MNE_norm = ebGroup3MNE.norm("space")
    ebGroup4MNE = eb.load.mne.stc_ndvar(stc=stcListsMNE[4], subject=avgBrain, src="vol-8", subjects_dir=subjects_dir, sss_filename=srcSavePath, parc="aparc.a2009s+aseg")
    ebGroup4MNE_norm = ebGroup4MNE.norm("space")

    ebGroup0Beam = eb.load.mne.stc_ndvar(stc=stcListsBeam[0], subject=avgBrain, src="vol-8", subjects_dir=subjects_dir, sss_filename=srcSavePath, parc="aparc.a2009s+aseg")
    ebGroup0Beam_norm = ebGroup0Beam.norm("space")
    ebGroup1Beam = eb.load.mne.stc_ndvar(stc=stcListsBeam[1], subject=avgBrain, src="vol-8", subjects_dir=subjects_dir, sss_filename=srcSavePath, parc="aparc.a2009s+aseg")
    ebGroup1Beam_norm = ebGroup1Beam.norm("space")
    ebGroup2Beam = eb.load.mne.stc_ndvar(stc=stcListsBeam[2], subject=avgBrain, src="vol-8", subjects_dir=subjects_dir, sss_filename=srcSavePath, parc="aparc.a2009s+aseg")
    ebGroup2Beam_norm = ebGroup2Beam.norm("space")
    ebGroup3Beam = eb.load.mne.stc_ndvar(stc=stcListsBeam[3], subject=avgBrain, src="vol-8", subjects_dir=subjects_dir, sss_filename=srcSavePath, parc="aparc.a2009s+aseg")
    ebGroup3Beam_norm = ebGroup3Beam.norm("space")
    ebGroup4Beam = eb.load.mne.stc_ndvar(stc=stcListsBeam[4], subject=avgBrain, src="vol-8", subjects_dir=subjects_dir, sss_filename=srcSavePath, parc="aparc.a2009s+aseg")
    ebGroup4Beam_norm = ebGroup4Beam.norm("space")

    if doXhemi:
        ebGroupXhemiSpeech_orig = eb.load.mne.stc_ndvar(stc=stcListsXhemi_orig[0], subject="fsaverage_sym", src="vol-8", subjects_dir=subjects_dir, sss_filename=srcSavePath_sym, parc="aparc.a2009s+aseg")
        ebGroupXhemiSpeech_orig_norm = ebGroupXhemiSpeech_orig.norm("space")
        ebGroupXhemiSpeech_oppo = eb.load.mne.stc_ndvar(stc=stcListsXhemi_oppo[0], subject="fsaverage_sym", src="vol-8", subjects_dir=subjects_dir, sss_filename=srcSavePath_sym, parc="aparc.a2009s+aseg")
        ebGroupXhemiSpeech_oppo_norm = ebGroupXhemiSpeech_oppo.norm("space")
        ebGroupXhemiClick_orig = eb.load.mne.stc_ndvar(stc=stcListsXhemi_orig[1], subject="fsaverage_sym", src="vol-8", subjects_dir=subjects_dir, sss_filename=srcSavePath_sym, parc="aparc.a2009s+aseg")
        ebGroupXhemiClick_orig_norm = ebGroupXhemiClick_orig.norm("space")
        ebGroupXhemiClick_oppo = eb.load.mne.stc_ndvar(stc=stcListsXhemi_oppo[1], subject="fsaverage_sym", src="vol-8", subjects_dir=subjects_dir, sss_filename=srcSavePath_sym, parc="aparc.a2009s+aseg")
        ebGroupXhemiClick_oppo_norm = ebGroupXhemiClick_oppo.norm("space")
        ebGroupXhemiDiff_orig = eb.load.mne.stc_ndvar(stc=stcListsXhemi_orig[2], subject="fsaverage_sym", src="vol-8", subjects_dir=subjects_dir, sss_filename=srcSavePath_sym, parc="aparc.a2009s+aseg")
        ebGroupXhemiDiff_orig_norm = ebGroupXhemiDiff_orig.norm("space")
        ebGroupXhemiDiff_oppo = eb.load.mne.stc_ndvar(stc=stcListsXhemi_oppo[2], subject="fsaverage_sym", src="vol-8", subjects_dir=subjects_dir, sss_filename=srcSavePath_sym, parc="aparc.a2009s+aseg")
        ebGroupXhemiDiff_oppo_norm = ebGroupXhemiDiff_oppo.norm("space")

    if doTests:

        # # The original paired difference tests
        # ebTest_04MNE = eb.testnd.VectorDifferenceRelated(ebGroup0MNE, ebGroup4MNE, tfce=True)
        # ebTestNorm_04MNE = eb.testnd.TTestRelated(ebGroup0MNE_norm, ebGroup4MNE_norm, tfce=True)
        # ebTest_14MNE = eb.testnd.VectorDifferenceRelated(ebGroup1MNE, ebGroup4MNE, tfce=True)
        # ebTestNorm_14MNE = eb.testnd.TTestRelated(ebGroup1MNE_norm, ebGroup4MNE_norm, tfce=True)
        # ebTest_24MNE = eb.testnd.VectorDifferenceRelated(ebGroup2MNE, ebGroup4MNE, tfce=True)
        # ebTestNorm_24MNE = eb.testnd.TTestRelated(ebGroup2MNE_norm, ebGroup4MNE_norm, tfce=True)
        # ebTest_34MNE = eb.testnd.VectorDifferenceRelated(ebGroup3MNE, ebGroup4MNE, tfce=True)
        # ebTestNorm_34MNE = eb.testnd.TTestRelated(ebGroup3MNE_norm, ebGroup4MNE_norm, tfce=True)
        # ebTest_12MNE = eb.testnd.VectorDifferenceRelated(ebGroup1MNE, ebGroup2MNE, tfce=True)
        # ebTestNorm_12MNE = eb.testnd.TTestRelated(ebGroup1MNE_norm, ebGroup2MNE_norm, tfce=True)
        # ebTest_03MNE = eb.testnd.VectorDifferenceRelated(ebGroup0MNE, ebGroup3MNE, tfce=True)
        # ebTestNorm_03MNE = eb.testnd.TTestRelated(ebGroup0MNE_norm, ebGroup3MNE_norm, tfce=True)
        
        # ebTest_04Beam = eb.testnd.VectorDifferenceRelated(ebGroup0Beam, ebGroup4Beam, tfce=True)
        # ebTestNorm_04Beam = eb.testnd.TTestRelated(ebGroup0Beam_norm, ebGroup4Beam_norm, tfce=True)
        # ebTest_14Beam = eb.testnd.VectorDifferenceRelated(ebGroup1Beam, ebGroup4Beam, tfce=True)
        # ebTestNorm_14Beam = eb.testnd.TTestRelated(ebGroup1Beam_norm, ebGroup4Beam_norm, tfce=True)
        # ebTest_24Beam = eb.testnd.VectorDifferenceRelated(ebGroup2Beam, ebGroup4Beam, tfce=True)
        # ebTestNorm_24Beam = eb.testnd.TTestRelated(ebGroup2Beam_norm, ebGroup4Beam_norm, tfce=True)
        # ebTest_34Beam = eb.testnd.VectorDifferenceRelated(ebGroup3Beam, ebGroup4Beam, tfce=True)
        # ebTestNorm_34Beam = eb.testnd.TTestRelated(ebGroup3Beam_norm, ebGroup4Beam_norm, tfce=True)
        # ebTest_12Beam = eb.testnd.VectorDifferenceRelated(ebGroup1Beam, ebGroup2Beam, tfce=True)
        # ebTestNorm_12Beam = eb.testnd.TTestRelated(ebGroup1Beam_norm, ebGroup2Beam_norm, tfce=True)
        # ebTest_03Beam = eb.testnd.VectorDifferenceRelated(ebGroup0Beam, ebGroup3Beam, tfce=True)
        # ebTestNorm_03Beam = eb.testnd.TTestRelated(ebGroup0Beam_norm, ebGroup3Beam_norm, tfce=True)

        # ebTest_3MNE = eb.testnd.Vector(ebGroup3MNE, tfce=True)
        # ebTestNorm_3MNE = eb.testnd.TTestOneSample(ebGroup3MNE_norm, tfce=True)
        # ebTest_4MNE = eb.testnd.Vector(ebGroup4MNE, tfce=True)
        # ebTestNorm_4MNE = eb.testnd.TTestOneSample(ebGroup4MNE_norm, tfce=True)
        # ebTest_24MNE = eb.testnd.VectorDifferenceRelated(ebGroup2MNE, ebGroup4MNE, tfce=True)
        # ebTestNorm_24MNE = eb.testnd.TTestRelated(ebGroup2MNE_norm, ebGroup4MNE_norm, tfce=True)
        ebTest_34MNE = eb.testnd.VectorDifferenceRelated(ebGroup3MNE, ebGroup4MNE, tfce=True)
        # ebTestNorm_34MNE = eb.testnd.TTestRelated(ebGroup3MNE_norm, ebGroup4MNE_norm, tfce=True)
        # ebTest_12MNE = eb.testnd.VectorDifferenceRelated(ebGroup1MNE, ebGroup2MNE, tfce=True)
        # ebTestNorm_12MNE = eb.testnd.TTestRelated(ebGroup1MNE_norm, ebGroup2MNE_norm, tfce=True)
        # ebTest_03MNE = eb.testnd.VectorDifferenceRelated(ebGroup0MNE, ebGroup3MNE, tfce=True)
        # ebTestNorm_03MNE = eb.testnd.TTestRelated(ebGroup0MNE_norm, ebGroup3MNE_norm, tfce=True)

        # ebTest_XhemiSpeech = eb.testnd.VectorDifferenceRelated(ebGroupXhemiSpeech_orig, ebGroupXhemiSpeech_oppo, tfce=True)
        # ebTest_XhemiClick = eb.testnd.VectorDifferenceRelated(ebGroupXhemiClick_orig, ebGroupXhemiClick_oppo, tfce=True)
        # ebTest_XhemiDiff = eb.testnd.VectorDifferenceRelated(ebGroupXhemiDiff_orig, ebGroupXhemiDiff_oppo, tfce=True)
        

    if saveTests:

        # # The original paired difference tests
        # eb.save.pickle((
        # ebTest_04MNE,
        # ebTestNorm_04MNE,
        # ebTest_14MNE,
        # ebTestNorm_14MNE,
        # ebTest_24MNE,
        # ebTestNorm_24MNE,
        # ebTest_34MNE,
        # ebTestNorm_34MNE,
        # ebTest_12MNE,
        # ebTestNorm_12MNE,
        # ebTest_03MNE,
        # ebTestNorm_03MNE,
        # ebTest_04Beam,
        # ebTestNorm_04Beam,
        # ebTest_14Beam,
        # ebTestNorm_14Beam,
        # ebTest_24Beam,
        # ebTestNorm_24Beam,
        # ebTest_34Beam,
        # ebTestNorm_34Beam,
        # ebTest_12Beam,
        # ebTestNorm_12Beam,
        # ebTest_03Beam,
        # ebTestNorm_03Beam), f"/Volumes/Seagate/map/ebTestResultsPeakInd{peakInd}_{avgBrain}.pickle")
        
        # eb.save.pickle((
        # ebTest_3MNE,
        # ebTestNorm_3MNE,
        # ebTest_4MNE,
        # ebTestNorm_4MNE), f"/Volumes/Seagate/map/ebTestResults_single_PeakInd{peakInd}_{avgBrain}.pickle")

        eb.save.pickle((
        ebTest_34MNE,
        ebTestNorm_34MNE), f"/Volumes/Seagate/map/ebTestResults{sourceParamsSuffix}_PeakInd{peakInd}_{avgBrain}.pickle")

# %%
if loadTests:
    ebTests = []
    for i in range(2):
        ebTests.append(eb.load.unpickle(f"/Volumes/Seagate/map/ebTestResultsPeakInd{i}_{avgBrain}.pickle"))
        # ebTests.append(eb.load.unpickle(f"/Volumes/Seagate/map/ebTestResults{sourceParamsSuffix}_PeakInd{i}_{avgBrain}.pickle"))



# %%
####################################################
# Make sure to define this properly if LOADING tests
peakInd = 1
####################################################

if loadTests:
    # The original paired difference tests
    (ebTest_04MNE,
    ebTestNorm_04MNE,
    ebTest_14MNE,
    ebTestNorm_14MNE,
    ebTest_24MNE,
    ebTestNorm_24MNE,
    ebTest_34MNE,
    ebTestNorm_34MNE,
    ebTest_12MNE,
    ebTestNorm_12MNE,
    ebTest_03MNE,
    ebTestNorm_03MNE,
    ebTest_04Beam,
    ebTestNorm_04Beam,
    ebTest_14Beam,
    ebTestNorm_14Beam,
    ebTest_24Beam,
    ebTestNorm_24Beam,
    ebTest_34Beam,
    ebTestNorm_34Beam,
    ebTest_12Beam,
    ebTestNorm_12Beam,
    ebTest_03Beam,
    ebTestNorm_03Beam) = ebTests[peakInd]

    # (ebTest_3MNE,
    # ebTestNorm_3MNE,
    # ebTest_4MNE,
    # ebTestNorm_4MNE) = ebTests[peakInd]

    # (ebTest_34MNE,
    # ebTestNorm_34MNE) = ebTests[peakInd]


# ebTest = ebTest_XhemiDiff
ebTest = ebTest_34MNE
# ebTest = ebTest_4MNE



tmin = subsetsMNE[0].tmin
tstep = subsetsMNE[0].tstep

clusters = ebTest.find_clusters()
print(clusters)

vertices_all_list = ebGroup0MNE.source.vertices
# vertices_all_list = ebGroupXhemiSpeech_orig.source.vertices
# save.pickle(vertices_all_list, '/Users/yiwei/Dropbox/agrammatism/code/vertices_all_list.pickle')



res_diff = ebTest.masked_difference()
masked_res_diff = res_diff.x.filled(0)

# res_diff = ebTest.masked_parameter_map()
# masked_res_diff = res_diff.x.filled(0)

# masked_res_diff = ebTest.t2

# masked_res_diff = ebTest.p
# masked_res_diff.x = 1 / masked_res_diff.x

# masked_res_diff = ebTest.tfce_map



vertices_ROI = res_diff.source.vertices[0]

if len(masked_res_diff.shape) == 2:
    dipoles_all_timeseries = np.zeros((len(vertices_all_list[0]), masked_res_diff.shape[1]))
elif len(masked_res_diff.shape) == 3:
    dipoles_all_timeseries = np.zeros((len(vertices_all_list[0]), masked_res_diff.shape[1], masked_res_diff.shape[2]))

# Create a mapping from dipoleIndex to their positions
dipole_map = {vertex: idx for idx, vertex in enumerate(vertices_all_list[0])}

# Iterate through ROIVertices and update the new_array
for idx, vertex in enumerate(vertices_ROI):
    if vertex in dipole_map:
        dipoles_all_timeseries[dipole_map[vertex]] = masked_res_diff[idx]

if len(masked_res_diff.shape) == 2:
    stcEBresult = mne.VolSourceEstimate(data = dipoles_all_timeseries, vertices=vertices_all_list, tmin=tmin, tstep=tstep)
elif len(masked_res_diff.shape) == 3:
    stcEBresult = mne.VolVectorSourceEstimate(data = dipoles_all_timeseries, vertices=vertices_all_list, tmin=tmin, tstep=tstep)

if not doXhemi:
    srcForPlot = src_fs
    subjectForPlot = avgBrain
else:
    srcForPlot = src_fs_sym
    subjectForPlot = "fsaverage_sym"

ebBrain = mne.viz.plot_source_estimates(stcEBresult,
    subject=subjectForPlot,
    # surface=surface,
    surface="white",
    hemi="both",
    src=srcForPlot,
    views="coronal",
    initial_time=initial_time,
    subjects_dir=subjects_dir,
    brain_kwargs=dict(silhouette=True),
    smoothing_steps=7,
    show_traces=True,
    alpha=brain_alpha,
    # clim=dict(kind="value", lims=[2.0, 4.0, 6.0]),
)
mne.viz.set_3d_title(ebBrain, "Eelbrain significance test", size=20)


# stcSubsetsXhemi_orig[peakInd][..., 2] = stcSubsetsXhemi_orig[peakInd][..., 0] / stcSubsetsXhemi_orig[peakInd][..., 0].std() - stcSubsetsXhemi_orig[peakInd][..., 1] / stcSubsetsXhemi_orig[peakInd][..., 1].std()
# stcSubsetsXhemi_oppo[peakInd][..., 2] = stcSubsetsXhemi_oppo[peakInd][..., 0] / stcSubsetsXhemi_oppo[peakInd][..., 0].std() - stcSubsetsXhemi_oppo[peakInd][..., 1] / stcSubsetsXhemi_oppo[peakInd][..., 1].std()

# condIndForPlot = 2
# xhemi_orig_forPlot = subsetsXhemi_orig[peakInd].copy()
# xhemi_orig_forPlot.data = np.mean(stcSubsetsXhemi_orig[peakInd][..., condIndForPlot].squeeze() / np.std(stcSubsetsXhemi_orig[peakInd][..., condIndForPlot].squeeze()), axis=-1)
# ebBrain2 = mne.viz.plot_source_estimates(xhemi_orig_forPlot,
#     subject=subjectForPlot,
#     # surface=surface,
#     surface="pial",
#     hemi="both",
#     src=srcForPlot,
#     views="coronal",
#     initial_time=initial_time,
#     subjects_dir=subjects_dir,
#     brain_kwargs=dict(silhouette=True),
#     smoothing_steps=7,
#     show_traces=True,
#     alpha=brain_alpha,
# )

# xhemi_oppo_forPlot = subsetsXhemi_oppo[peakInd].copy()
# xhemi_oppo_forPlot.data = np.mean(stcSubsetsXhemi_oppo[peakInd][..., condIndForPlot].squeeze() / np.std(stcSubsetsXhemi_oppo[peakInd][..., condIndForPlot]), axis=-1)
# ebBrain3 = mne.viz.plot_source_estimates(xhemi_oppo_forPlot,
#     subject=subjectForPlot,
#     # surface=surface,
#     surface="pial",
#     hemi="both",
#     src=srcForPlot,
#     views="coronal",
#     initial_time=initial_time,
#     subjects_dir=subjects_dir,
#     brain_kwargs=dict(silhouette=True),
#     smoothing_steps=7,
#     show_traces=True,
#     alpha=brain_alpha,
# )
# mne.viz.set_3d_title(ebBrain, "Eelbrain significance test", size=20)

# brain2 = stcEBresult.plot_3d(
#     subject = avgBrain, 
#     hemi = "both",
#     subjects_dir=subjects_dir,
#     # clim = {'kind':'value',
#     #         'lims': [0,1.5e-11,3e-11]
#     #         },
#     src = src_fs,
#     smoothing_steps=3,
#     # brain_alpha=0.2,
#     cortex='low_contrast',
    
# )


stcEBresult.plot_3d(
    src=src_fs,
    subject=avgBrain,
    hemi="both",
    subjects_dir=subjects_dir,
    # initial_time=0.1,         # Time point to display
    
    # --- ARROW / QUIVER SETTINGS ---
    scale_factor=4.5,       # Change this to scale the size of the arrows up or down
    vector_alpha=0.8,         # Set opacity of the vector arrows (0.0 to 1.0)
    brain_alpha=0.3           # Lower brain opacity to make internal arrows easier to see
)

# %%
# adj = mne.spatial_src_adjacency(src_fs)

# # targetNormMNE = np.linalg.norm(stcSubsetsMNE[peakIndForStats][:, :, :, :, 1].T, axis=2)
# # clickNormMNE = np.linalg.norm(stcSubsetsMNE[peakIndForStats][:, :, :, :, 3].T, axis=2)
# # targetNormBeam = np.linalg.norm(stcSubsetsBeam[peakIndForStats][:, :, :, :, 1].T, axis=2)
# # clickNormBeam = np.linalg.norm(stcSubsetsBeam[peakIndForStats][:, :, :, :, 3].T, axis=2)

# stcFullMatsMNE = []
# stcFullMatsBeam = []
# stcFullMatsNormMNE = []
# stcFullMatsBeam = []
# stcFullMatsNormMNE = []
# stcFullMatsNormBeam = []

# for j in range(numConds):
#     tempMat = stcSubsetsMNE[peakInd][:, :, :, :, j].T
#     stcFullMatsMNE.append(tempMat / np.std(tempMat))
#     stcFullMatsNormMNE.append(np.linalg.norm(stcFullMatsMNE[j], axis=2))

#     tempMat = stcSubsetsBeam[peakInd][:, :, :, :, j].T
#     stcFullMatsBeam.append(tempMat / np.std(tempMat))
#     stcFullMatsNormBeam.append(np.linalg.norm(stcFullMatsBeam[j], axis=2))


# p_threshold = 0.05
# df = len(subjectsToAverageAll) - 1  # degrees of freedom for the test
# t_threshold = sp.stats.distributions.t.ppf(1 - p_threshold / 2, df=df)
# print(t_threshold)

# # stc_fs_here = subsetsMNE[peakIndForStats].copy()
# # stc_fs_here = mne.VolSourceEstimate(data=targetNormMNE.T.mean(axis=2), vertices=stc_fs_here.vertices, tmin=stc_fs_here.tmin, tstep=stc_fs_here.tstep, subject=stc_fs_here.subject)
# # mneBrainSubset = stc_fs_here.plot_3d(
# #     hemi="both",
# #     src=src_fs,
# #     views="coronal",
# #     initial_time=initial_time,
# #     subjects_dir=subjects_dir,
# #     brain_kwargs=dict(silhouette=True),
# #     smoothing_steps=7,
# #     show_traces=True,
# # )

# # This came from Google's AI, doesn't seem to work, or be compatible
# def custom_hotelling_t2_1samp(X):
#     """
#     Custom function to compute Hotelling's T^2 for MNE cluster test.
#     X has shape (n_observations, n_variables)
    
#     Note: For spatio-temporal data, X would be (n_subjects, n_times * n_vertices).
#     This function computes T^2 for one time/vertex point.
#     """
#     n_obs = X.shape[0]
#     n_vars = X.shape[1]
#     print("here are the dims:")
#     print(n_obs)
#     print(n_vars)
    
#     # if n_obs <= n_vars:
#     #     # Hotelling's T^2 requires n_obs > n_vars
#     #     # Handle this case (e.g., return NaNs or raise an error)
#     #     # For cluster testing across many points, NaNs are safer.
#     #     t2_val = np.nan
#     # else:
#     mu0 = np.zeros(n_vars)
#     mean_vec = np.mean(X, axis=0)
#     cov_mat = np.cov(X, rowvar=False) # Covariance matrix
    
#     # Hotelling's T^2 statistic
#     try:
#         inv_cov = sp.linalg.inv(cov_mat)
#         t2_val = n_obs * (mean_vec - mu0).T @ inv_cov @ (mean_vec - mu0)
#     except sp.linalg.LinAlgError:
#         t2_val = np.nan # Handle singular matrix issues

#     # MNE expects the function to work over the first dimension (observations)
#     # and return a single value for each spatio-temporal point.
#     # This conceptual example would need vectorization to work efficiently over all points.
#     # return t2_val, np.nan # Return T^2 statistic and a dummy p-value
#     # print("got here")
#     # print(t2_val)
#     return t2_val  # Return T^2 statistic and a dummy p-value












# resultsDiffMNE = mne.stats.spatio_temporal_cluster_1samp_test(sp.stats.zscore(targetNormMNE, axis=None)-sp.stats.zscore(clickNormMNE, axis=None), adjacency=adj, n_permutations=50)
# resultsTargetZeroMNE = mne.stats.spatio_temporal_cluster_1samp_test(sp.stats.zscore(targetNormMNE), adjacency=adj, n_permutations=50)
# resultsDiff = mne.stats.spatio_temporal_cluster_1samp_test(
#     stcFullMatsNormMNE[1] - stcFullMatsNormMNE[3], 
#     # adjacency=adj, 
#     n_permutations=50,
#     threshold=t_threshold,
# )
# resultsZero = mne.stats.spatio_temporal_cluster_1samp_test(
#     targetNormMNE, 
#     adjacency=adj, 
#     n_permutations=50,
#     threshold=t_threshold
# )
# resultsDiffBeam = mne.stats.spatio_temporal_cluster_1samp_test(targetNormBeam-clickNormBeam, adjacency=adj, n_permutations=50)
# resultsTargetZeroBeam = mne.stats.spatio_temporal_cluster_1samp_test(sp.stats.zscore(targetNormBeam), adjacency=adj, n_permutations=50)

# stc_all_cluster_vis = mne.stats.summarize_clusters_stc(
#     resultsDiff, vertices=src_fs, subject=avgBrain, tstep=1/5000
# )
# brain = stc_all_cluster_vis.plot_3d(
#     src=src_fs,
#     hemi="both",
#     views="lateral",
#     subjects_dir=subjects_dir,
#     time_label="temporal extent (s)",
# )
# plt.plot(resultsDiff[2])

# %%
testStc = stcSecond_fs_here
# testStc = stcSecond_fs_here
# testStc = stcEBresult
mgzLabel = "aparc.a2009s+aseg.mgz"
# mgzLabel = "aparc+aseg.mgz"

labels = mne.get_volume_labels_from_aseg(subjects_dir + f"/{avgBrain}/mri/{mgzLabel}")
test = mne.extract_label_time_course(testStc, subjects_dir + f"/{avgBrain}/mri/{mgzLabel}", src_fs)
testNorm = np.linalg.norm(test, axis=1)

# extractInds = [.0035, .0065]
# extractInds = [.0075, .0095]
# extractInds = [.0095, .011]
# extractInds = [.0086, .013]
# extractInds = [.0172, .0192]
# extractInds = [.023, .030]
# extractInds = [.0246, .0368]
# extractInds = [.03, .036]
# extractInds = [.042, .045]
# extractInds = [.0386, .0532]
extractInds = [.048, .053]
# extractInds = [.04, .06]
testPortion = testNorm[:, int(extractInds[0] * fs):int(extractInds[1] * fs)]
maxVals = np.max(testPortion, axis=-1)
sortedInds = np.argsort(maxVals)
start = -40
print(maxVals[sortedInds[start:]])
for i in np.arange(start, 0):
    print([labels[sortedInds[i]], maxVals[sortedInds[i]]])

# plt.plot(testPortion.T)

# %%
if makeNIfTIs:

    os.chdir("/Users/karl/eegAndMegAnalysis")
    origDir = os.getcwd()
    
    target_affine = np.eye(3) * 1.5

    # This is the original grouping of NIfTIs to create, for the paper and talk etc.
    # stcList = [stcFirst_fs_here, stcSecond_fs_here, stcDiff_fs_here, stcEBresult]
    # stcListDirs = ["/Users/karl/speechClickNiis/speech/", "/Users/karl/speechClickNiis/clicks/", "/Users/karl/speechClickNiis/diff/", "/Users/karl/speechClickNiis/ebResult/"]

    # This begins ad-hoc NIfTIs to create
    stcList = [stcEBresult]
    stcListDirs = ["/Users/karl/speechClickNiis/ebResult/"]
    # stcList = [stcFirst_test]
    
    for i, stcMember in enumerate(stcList):
        stc_upDown, dirs_used = stcMember.project(directions=np.tile(np.array([0, 0, 1]), (totalNumVertices, 1)))
        stc_frontBack, dirs_used = stcMember.project(directions=np.tile(np.array([0, 1, 0]), (totalNumVertices, 1)))
        stc_leftRight, dirs_used = stcMember.project(directions=np.tile(np.array([1, 0, 0]), (totalNumVertices, 1)))
        stcListHere = [stcMember, stc_upDown, stc_frontBack, stc_leftRight]
        # filenamesList = ["/Users/karl/stcSpeech_up.nii.gz", "/Users/karl/stcClicks_up.nii.gz", "/Users/karl/stcDiff_up.nii.gz", "/Users/karl/stcEBresult_up.nii.gz"]
        stcSubDirs = ["magnitude/", "upDown/", "frontBack/", "leftRight/"]
        
        for j, stcHere in enumerate(stcListHere):
        
            img = stcHere.as_volume(src=src_fs)
            
            upsampled_nii = nimg.resample_img(
                img, 
                target_affine=target_affine, 
                interpolation="continuous" # Use 'nearest' for label maps
            )
            filename = stcSubDirs[j][:-1]
            location = stcListDirs[i] + stcSubDirs[j]
            # Save the resulting upsampled image
            nibabel.save(upsampled_nii, location + filename + ".nii.gz")
            os.chdir(location)
            subprocess.run(["subject2mni", "-i", f"{filename}.nii.gz", "-m", f"{subjects_dir}/fsaverage/mri/simnibsParc/m2m_fsaverage", "-o", f"{filename}.nii.gz"])  # This SimNIBS function will automatically append "_MNI" into the filename and not just overwrite
            subprocess.run(["fslsplit", f"{filename}_MNI.nii.gz"])
            os.chdir(origDir)


# %%
# stds = np.std(stcFirst_fs_here.data, axis=-1).mean(axis=-1)
# test = calcPC(stcFirst_fs_here.data, True, None)
# test *= stds
# test = -test.T
# stcFirst_test = mne.VolSourceEstimate(data=test, vertices=vertices_all_list, tmin=tmin, tstep=tstep)

# stcFirst_test, dirs_used = stcFirst_fs_here.project(directions=np.tile(np.array([0, 1, 0]), (totalNumVertices, 1)))

labels = mne.get_volume_labels_from_aseg(subjects_dir + f"/{avgBrain}/mri/{mgzLabel}")
# labelSubset = ["CSF", "ctx_rh_G_cingul-Post-ventral", "ctx_lh_G_cingul-Post-ventral", 
#                "Right-Thalamus-Proper", "Left-Thalamus-Proper", "ctx_rh_G_temp_sup-G_T_transv", "ctx_lh_G_temp_sup-G_T_transv",
#                "ctx_rh_Lat_Fis-post", "ctx_lh_Lat_Fis-post", "ctx_rh_S_temporal_transverse", "ctx_lh_S_temporal_transverse"]
# labelSubset = ["CSF", "ctx_rh_G_cingul-Post-ventral", "ctx_lh_G_cingul-Post-ventral", "CC_Posterior", "CC_Mid_Posterior", "CC_Central", 
#                "Right-Thalamus-Proper", "Left-Thalamus-Proper", "ctx_rh_Lat_Fis-post", "ctx_lh_Lat_Fis-post",
#                "ctx_rh_G_temp_sup-G_T_transv", "ctx_lh_G_temp_sup-G_T_transv",
#                "ctx_rh_S_temporal_transverse", "ctx_lh_S_temporal_transverse"]
# labelSubset = ["ctx_rh_G_cingul-Post-ventral",
#                "ctx_rh_G_temp_sup-G_T_transv", "ctx_lh_G_temp_sup-G_T_transv"]
# labelSubset = ["ctx_rh_G_temp_sup-G_T_transv", "ctx_lh_G_temp_sup-G_T_transv"]

labelSubset = ["Midbrain", "Cortex"]
labelSubsetABR = ["CSF", "ctx_rh_G_cingul-Post-ventral", "ctx_lh_G_cingul-Post-ventral", "Right-Thalamus-Proper", "Left-Thalamus-Proper", "CC_Posterior"]
labelSubsetMLR = ["ctx_rh_G_temp_sup-G_T_transv", "ctx_lh_G_temp_sup-G_T_transv", "ctx_rh_S_temporal_transverse", "ctx_lh_S_temporal_transverse"]
# labelSubsetABR = ["CSF", "ctx_rh_G_cingul-Post-ventral", "ctx_lh_G_cingul-Post-ventral", "Right-Thalamus-Proper", "Left-Thalamus-Proper", "CC_Posterior"]
# labelSubsetMLR = ["ctx_lh_G_temp_sup-G_T_transv", "ctx_lh_S_temporal_transverse"]

t = stcFirst_fs_here.times * 1000

linewidth = 3.5
alpha = 0.85
zscoreROIs = True
zscoreAxis = 0
# zscoreAxis = None

projectToOne = False
projectToComplex = False

clusterTimes = [[clusters[0]["tstart"], clusters[0]["tstop"]], [clusters[1]["tstart"], clusters[1]["tstop"]], [clusters[2]["tstart"], clusters[2]["tstop"]], [clusters[8]["tstart"], clusters[8]["tstop"]]]
# clusterTimes = [[clusters[2]["tstart"], clusters[2]["tstop"]], [clusters[5]["tstart"], clusters[5]["tstop"]], [clusters[6]["tstart"], clusters[6]["tstop"]], [clusters[12]["tstart"], clusters[12]["tstop"]]]

if projectToOne:
    sourceTimeCourseSpeechABR = mne.extract_label_time_course(stcFirst_fs_here.copy().project(directions=np.tile(np.array([0, 0, 1]), (totalNumVertices, 1)))[0], [subjects_dir + f"/{avgBrain}/mri/{mgzLabel}", labelSubsetABR], src_fs, mode="auto")
    sourceTimeCourseSpeechMLR = mne.extract_label_time_course(stcFirst_fs_here.copy().project(directions=np.tile(np.array([0, 1, 1]), (totalNumVertices, 1)))[0], [subjects_dir + f"/{avgBrain}/mri/{mgzLabel}", labelSubsetMLR], src_fs, mode="auto")
elif projectToComplex:
    stcFirst_firstDim  = stcFirst_fs_here.copy().project(directions=np.tile(np.array([0, 0, 1]), (totalNumVertices, 1)))[0].data
    stcFirst_secondDim = stcFirst_fs_here.copy().project(directions=np.tile(np.array([0, 1, 0]), (totalNumVertices, 1)))[0].data
    sourceTimeCourseSpeechABRFirstDim  = mne.extract_label_time_course(stcFirst_fs_here.copy().project(directions=np.tile(np.array([0, 0, 1]), (totalNumVertices, 1)))[0], [subjects_dir + f"/{avgBrain}/mri/{mgzLabel}", labelSubsetABR], src_fs, mode="auto")
    sourceTimeCourseSpeechABRSecondDim = mne.extract_label_time_course(stcFirst_fs_here.copy().project(directions=np.tile(np.array([0, 1, 0]), (totalNumVertices, 1)))[0], [subjects_dir + f"/{avgBrain}/mri/{mgzLabel}", labelSubsetABR], src_fs, mode="auto")
    sourceTimeCourseSpeechMLRFirstDim  = mne.extract_label_time_course(stcFirst_fs_here.copy().project(directions=np.tile(np.array([0, 0, 1]), (totalNumVertices, 1)))[0], [subjects_dir + f"/{avgBrain}/mri/{mgzLabel}", labelSubsetMLR], src_fs, mode="auto")
    sourceTimeCourseSpeechMLRSecondDim = mne.extract_label_time_course(stcFirst_fs_here.copy().project(directions=np.tile(np.array([0, 1, 0]), (totalNumVertices, 1)))[0], [subjects_dir + f"/{avgBrain}/mri/{mgzLabel}", labelSubsetMLR], src_fs, mode="auto")
    
    sourceTimeCourseSpeechABR = np.atan2(sourceTimeCourseSpeechABRFirstDim, sourceTimeCourseSpeechABRSecondDim)
    # Run these next two lines to rotate these phases backwards by 90 degrees to make red on top
    sourceTimeCourseSpeechABR = sourceTimeCourseSpeechABR + (np.pi / 2)
    sourceTimeCourseSpeechABR = np.atan2(np.sin(sourceTimeCourseSpeechABR), np.cos(sourceTimeCourseSpeechABR))
    
    sourceTimeCourseSpeechMLR = np.atan2(sourceTimeCourseSpeechMLRFirstDim, sourceTimeCourseSpeechMLRSecondDim)
    # Run these next two lines to rotate these phases backwards by 90 degrees to make red on top
    sourceTimeCourseSpeechMLR = sourceTimeCourseSpeechMLR + (np.pi / 2)
    sourceTimeCourseSpeechMLR = np.atan2(np.sin(sourceTimeCourseSpeechMLR), np.cos(sourceTimeCourseSpeechMLR))
else:
    sourceTimeCourseSpeechABR = mne.extract_label_time_course(stcFirst_fs_here, [subjects_dir + f"/{avgBrain}/mri/{mgzLabel}", labelSubsetABR], src_fs, mode="auto")
    sourceTimeCourseSpeechMLR = mne.extract_label_time_course(stcFirst_fs_here, [subjects_dir + f"/{avgBrain}/mri/{mgzLabel}", labelSubsetMLR], src_fs, mode="auto")

if sourceTimeCourseSpeechABR.shape[1] == 3:
    sourceTimeCourseSpeechABR = calcPC(sourceTimeCourseSpeechABR, True, None)
else:
    sourceTimeCourseSpeechABR = sourceTimeCourseSpeechABR.T
if zscoreROIs:
    sourceTimeCourseSpeechABR = sp.stats.zscore(sourceTimeCourseSpeechABR, axis=zscoreAxis)
sourceTimeCourseSpeechABR = sourceTimeCourseSpeechABR.mean(axis=1)[:, np.newaxis]

if sourceTimeCourseSpeechMLR.shape[1] == 3:
    sourceTimeCourseSpeechMLR = calcPC(sourceTimeCourseSpeechMLR, True, None)
else:
    sourceTimeCourseSpeechMLR = sourceTimeCourseSpeechMLR.T
if zscoreROIs:
    sourceTimeCourseSpeechMLR = sp.stats.zscore(sourceTimeCourseSpeechMLR, axis=zscoreAxis)
sourceTimeCourseSpeechMLR = sourceTimeCourseSpeechMLR.mean(axis=1)[:, np.newaxis]
sourceTimeCourseSpeech = np.concatenate((sourceTimeCourseSpeechABR, sourceTimeCourseSpeechMLR), axis=1)

if projectToOne:
    sourceTimeCourseClickABR = mne.extract_label_time_course(stcSecond_fs_here.copy().project(directions=np.tile(np.array([0, 0, 1]), (totalNumVertices, 1)))[0], [subjects_dir + f"/{avgBrain}/mri/{mgzLabel}", labelSubsetABR], src_fs, mode="auto")
    sourceTimeCourseClickMLR = mne.extract_label_time_course(stcSecond_fs_here.copy().project(directions=np.tile(np.array([0, 1, 1]), (totalNumVertices, 1)))[0], [subjects_dir + f"/{avgBrain}/mri/{mgzLabel}", labelSubsetMLR], src_fs, mode="auto")
else:
    sourceTimeCourseClickABR = mne.extract_label_time_course(stcSecond_fs_here, [subjects_dir + f"/{avgBrain}/mri/{mgzLabel}", labelSubsetABR], src_fs, mode="auto")
    sourceTimeCourseClickMLR = mne.extract_label_time_course(stcSecond_fs_here, [subjects_dir + f"/{avgBrain}/mri/{mgzLabel}", labelSubsetMLR], src_fs, mode="auto")

if sourceTimeCourseClickABR.shape[1] == 3:
    sourceTimeCourseClickABR = calcPC(sourceTimeCourseClickABR, True, None)
else:
    sourceTimeCourseClickABR = sourceTimeCourseClickABR.T
if zscoreROIs:
    sourceTimeCourseClickABR = sp.stats.zscore(sourceTimeCourseClickABR, axis=zscoreAxis)
sourceTimeCourseClickABR = sourceTimeCourseClickABR.mean(axis=1)[:, np.newaxis]

if sourceTimeCourseClickMLR.shape[1] == 3:
    sourceTimeCourseClickMLR = calcPC(sourceTimeCourseClickMLR, True, None)
else:
    sourceTimeCourseClickMLR = sourceTimeCourseClickMLR.T
if zscoreROIs:
    sourceTimeCourseClickMLR = sp.stats.zscore(sourceTimeCourseClickMLR, axis=zscoreAxis)
sourceTimeCourseClickMLR = sourceTimeCourseClickMLR.mean(axis=1)[:, np.newaxis]
sourceTimeCourseClick = np.concatenate((sourceTimeCourseClickABR, sourceTimeCourseClickMLR), axis=1)

if projectToOne:
    sourceTimeCourseTargetABR = mne.extract_label_time_course(stcTarget_fs_here.copy().project(directions=np.tile(np.array([0, 0, 1]), (totalNumVertices, 1)))[0], [subjects_dir + f"/{avgBrain}/mri/{mgzLabel}", labelSubsetABR], src_fs, mode="auto")
    sourceTimeCourseTargetMLR = mne.extract_label_time_course(stcTarget_fs_here.copy().project(directions=np.tile(np.array([0, 1, 1]), (totalNumVertices, 1)))[0], [subjects_dir + f"/{avgBrain}/mri/{mgzLabel}", labelSubsetMLR], src_fs, mode="auto")
else:
    sourceTimeCourseTargetABR = mne.extract_label_time_course(stcTarget_fs_here, [subjects_dir + f"/{avgBrain}/mri/{mgzLabel}", labelSubsetABR], src_fs, mode="auto")
    sourceTimeCourseTargetMLR = mne.extract_label_time_course(stcTarget_fs_here, [subjects_dir + f"/{avgBrain}/mri/{mgzLabel}", labelSubsetMLR], src_fs, mode="auto")

if sourceTimeCourseTargetABR.shape[1] == 3:
    sourceTimeCourseTargetABR = calcPC(sourceTimeCourseTargetABR, True, None)
else:
    sourceTimeCourseTargetABR = sourceTimeCourseTargetABR.T
if zscoreROIs:
    sourceTimeCourseTargetABR = sp.stats.zscore(sourceTimeCourseTargetABR, axis=zscoreAxis)
sourceTimeCourseTargetABR = sourceTimeCourseTargetABR.mean(axis=1)[:, np.newaxis]

if sourceTimeCourseTargetMLR.shape[1] == 3:
    sourceTimeCourseTargetMLR = calcPC(sourceTimeCourseTargetMLR, True, None)
else:
    sourceTimeCourseTargetMLR = sourceTimeCourseTargetMLR.T
if zscoreROIs:
    sourceTimeCourseTargetMLR = sp.stats.zscore(sourceTimeCourseTargetMLR, axis=zscoreAxis)
sourceTimeCourseTargetMLR = sourceTimeCourseTargetMLR.mean(axis=1)[:, np.newaxis]
sourceTimeCourseTarget = np.concatenate((sourceTimeCourseTargetABR, sourceTimeCourseTargetMLR), axis=1)

if projectToOne:
    sourceTimeCourseDistractorABR = mne.extract_label_time_course(stcDistractor_fs_here.copy().project(directions=np.tile(np.array([0, 0, 1]), (totalNumVertices, 1)))[0], [subjects_dir + f"/{avgBrain}/mri/{mgzLabel}", labelSubsetABR], src_fs, mode="auto")
    sourceTimeCourseDistractorMLR = mne.extract_label_time_course(stcDistractor_fs_here.copy().project(directions=np.tile(np.array([0, 1, 1]), (totalNumVertices, 1)))[0], [subjects_dir + f"/{avgBrain}/mri/{mgzLabel}", labelSubsetMLR], src_fs, mode="auto")
else:
    sourceTimeCourseDistractorABR = mne.extract_label_time_course(stcDistractor_fs_here, [subjects_dir + f"/{avgBrain}/mri/{mgzLabel}", labelSubsetABR], src_fs, mode="auto")
    sourceTimeCourseDistractorMLR = mne.extract_label_time_course(stcDistractor_fs_here, [subjects_dir + f"/{avgBrain}/mri/{mgzLabel}", labelSubsetMLR], src_fs, mode="auto")

if sourceTimeCourseDistractorABR.shape[1] == 3:
    sourceTimeCourseDistractorABR = calcPC(sourceTimeCourseDistractorABR, True, None)
else:
    sourceTimeCourseDistractorABR = sourceTimeCourseDistractorABR.T
if zscoreROIs:
    sourceTimeCourseDistractorABR = sp.stats.zscore(sourceTimeCourseDistractorABR, axis=zscoreAxis)
sourceTimeCourseDistractorABR = sourceTimeCourseDistractorABR.mean(axis=1)[:, np.newaxis]

if sourceTimeCourseDistractorMLR.shape[1] == 3:
    sourceTimeCourseDistractorMLR = calcPC(sourceTimeCourseDistractorMLR, True, None)
else:
    sourceTimeCourseDistractorMLR = sourceTimeCourseDistractorMLR.T
if zscoreROIs:
    sourceTimeCourseDistractorMLR = sp.stats.zscore(sourceTimeCourseDistractorMLR, axis=zscoreAxis)
sourceTimeCourseDistractorMLR = sourceTimeCourseDistractorMLR.mean(axis=1)[:, np.newaxis]
sourceTimeCourseDistractor = np.concatenate((sourceTimeCourseDistractorABR, sourceTimeCourseDistractorMLR), axis=1)


# labelSubsetTemp = labelSubset
# labelSubsetTemp = [labelSubset[0]]

# mne.viz.plot_source_estimates(stcFirst_test,
#         subject=avgBrain,
#         surface=surface,
#         hemi="both",
#         src=src_fs,
#         views="coronal",
#         initial_time=initial_time,
#         subjects_dir=subjects_dir,
#         brain_kwargs=dict(silhouette=True),
#         smoothing_steps=7,
#         show_traces=True,
#         alpha=brain_alpha,
#     )
# mne.viz.plot_source_estimates(stcFirst_fs_here,
#         subject=avgBrain,
#         surface=surface,
#         hemi="both",
#         src=src_fs,
#         views="coronal",
#         initial_time=initial_time,
#         subjects_dir=subjects_dir,
#         brain_kwargs=dict(silhouette=True),
#         smoothing_steps=7,
#         show_traces=True,
#         alpha=brain_alpha,
#     )


# plt.figure()
# plt.plot(t, sourceTimeCourseSpeech)
# plt.legend(labelSubset)
# plt.figure()
# plt.plot(t, sourceTimeCourseClick)
# plt.legend(labelSubset)


cmapTarget = matplotlib.cm.get_cmap("Purples_r")
cmapDistractor = matplotlib.cm.get_cmap("Greens_r")
cmapSpeech = matplotlib.cm.get_cmap("Blues_r")
cmapClicks = matplotlib.cm.get_cmap("Oranges_r")
norm = plt.Normalize(vmin=0, vmax=1.5)

fig1, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=[12, 9])

for i, category in enumerate(labelSubset):
    # Get a color from the colormap using the normalized index
    color = cmapTarget(norm(i))
    # Plot dummy data/lines with the specific color and label
    ax1.plot(t, sourceTimeCourseTarget[:, i], color=color, label=category, linewidth=linewidth, alpha=alpha)
    # Get a color from the colormap using the normalized index
    color = cmapDistractor(norm(i))
    # Plot dummy data/lines with the specific color and label
    ax1.plot(t, sourceTimeCourseDistractor[:, i], color=color, label=category, linewidth=linewidth, alpha=alpha)

# Create the legend, which will automatically use the specified colors/labels
ax1.set_title("Target vs. Distractor source TRFs")
ax1.set_ylabel("Z-scored current")
# ax1.legend(["Medial/midbrain", "Right auditory cortex", "Left auditory cortex"], title="ROIs medial to lateral", loc="upper right", bbox_to_anchor=(1.12, 1))
# ax1.legend(["Medial/midbrain", "Right auditory cortex", "Left auditory cortex"], title="ROIs medial to lateral", loc="upper right")
ax1.legend(["Target Midbrain", "Distractor Midbrain", "Target Cortex", "Distractor Cortex"])
# ax1.legend(title="ROIs medial to lateral", bbox_to_anchor=(.9, 1))
plt.grid()



for i, category in enumerate(labelSubset):
    # Get a color from the colormap using the normalized index
    color = cmapSpeech(norm(i))
    # Plot dummy data/lines with the specific color and label
    ax2.plot(t, sourceTimeCourseSpeech[:, i], color=color, label=category, linewidth=linewidth, alpha=alpha)
    # Get a color from the colormap using the normalized index
    color = cmapClicks(norm(i))
    # Plot dummy data/lines with the specific color and label
    ax2.plot(t, sourceTimeCourseClick[:, i], color=color, label=category, linewidth=linewidth, alpha=alpha)

# Create the legend, which will automatically use the specified colors/labels
ax2.set_title("Speech vs. Click source TRFs")
ax2.set_xlabel("Time (ms)")
ax2.set_ylabel("Z-scored current")
ax2.legend(["Speech Midbrain", "Clicks Midbrain", "Speech Cortex", "Clicks Cortex"])
maxAbs = np.max((np.max(np.abs(sourceTimeCourseSpeech)), np.max(np.abs(sourceTimeCourseClick))))
for i, clusterTime in enumerate(clusterTimes):
    ax2.fill_betweenx((-maxAbs, maxAbs), clusterTime[0]*1000, clusterTime[1]*1000, color="silver", alpha=0.5)
# ax2.legend(title="ROIs medial to lateral")
plt.grid()
plt.show()
# fig1.legend(title="ROIs medial to lateral")


# %%
clusterInd = 2
# clusterInd = 3

sn10inds = [np.argmin(np.abs(t-clusterTimes[clusterInd][0]*1000)), np.argmin(np.abs(t-clusterTimes[clusterInd][1]*1000))]
# sn10inds = [np.argmin(np.abs(t-clusters[clusterInd]["tstart"]*1000)), np.argmin(np.abs(t-clusters[clusterInd]["tstop"]*1000))]
sn10stcsSpeech = np.zeros((sn10inds[1] - sn10inds[0], len(subjectsToAverageAll)))
sn10stcsClicks = np.zeros((sn10inds[1] - sn10inds[0], len(subjectsToAverageAll)))
for i in range(group1mat.shape[-1]):
    stcTemp = stcFirst_fs_here.copy()
    stcTemp.data = group1mat[..., sn10inds[0]:sn10inds[1], i]
    temp = mne.extract_label_time_course(stcTemp, [subjects_dir + f"/{avgBrain}/mri/{mgzLabel}", labelSubsetMLR], src_fs, mode="auto")
    temp = calcPC(temp, True, None)
    sn10stcsSpeech[:, i] = temp.mean(axis=1)
    
    stcTemp = stcSecond_fs_here.copy()
    stcTemp.data = group2mat[..., sn10inds[0]:sn10inds[1], i]
    temp = mne.extract_label_time_course(stcTemp, [subjects_dir + f"/{avgBrain}/mri/{mgzLabel}", labelSubsetMLR], src_fs, mode="auto")
    temp = calcPC(temp, True, None)
    sn10stcsClicks[:, i] = temp.mean(axis=1)


# %%
figSN10, (axSN10speech, axSN10clicks) = plt.subplots(1, 2, figsize=(8, 8))
t_sn10 = t[sn10inds[0]: sn10inds[1]]
# 1. Retrieve the colormap by name
cmap = plt.colormaps["viridis"]

# 2. Define how many colors you want
num_colors = len(subjectsToAverageAll)

# 3. Sample evenly across the 0.0 to 1.0 range
colors = cmap(np.linspace(0, 1, num_colors))
maxSN10 = np.max((np.max(sn10stcsSpeech), np.max(sn10stcsClicks)))
minSN10 = np.min((np.min(sn10stcsSpeech), np.min(sn10stcsClicks)))
for i in range(len(subjectsToAverageAll)):
    axSN10speech.plot(t_sn10, sn10stcsSpeech[:, i], linewidth=5, alpha=.85, color=colors[i])
    axSN10clicks.plot(t_sn10, sn10stcsClicks[:, i], linewidth=5, alpha=.85, color=colors[i])
axSN10speech.set_ylim((minSN10, maxSN10))
axSN10clicks.set_ylim((minSN10, maxSN10))
axSN10clicks.set_yticklabels([])
axSN10speech.set_xlabel("Time (ms)")
axSN10clicks.set_xlabel("Time (ms)")
axSN10speech.set_ylabel("Z-scored current")

# connectPoint = int(np.round(len(t_sn10) / 2))
connectPointInds = np.zeros((len(subjectsToAverageAll), 2))
connectPointVals = np.zeros((len(subjectsToAverageAll), 2))
for i in range(len(subjectsToAverageAll)):
    connectPointInds[i, 0] = np.argmin(sn10stcsSpeech[:, i])
    connectPointVals[i, 0] = np.min(sn10stcsSpeech[:, i])
    connectPointInds[i, 1] = np.argmin(sn10stcsClicks[:, i])
    connectPointVals[i, 1] = np.min(sn10stcsClicks[:, i])

for i in range(len(subjectsToAverageAll)):
    con = matplotlib.patches.ConnectionPatch(
        xyA=(t_sn10[int(connectPointInds[i, 0])], connectPointVals[i, 0]),
        xyB=(t_sn10[int(connectPointInds[i, 1])], connectPointVals[i, 1]),
        coordsA="data",
        coordsB="data",
        axesA=axSN10speech,
        axesB=axSN10clicks,
        # color=colors[i],
        color="black",
        linestyle="--",
        linewidth=1,
        # alpha=0.8,
    )
    figSN10.add_artist(con)

bounds = np.linspace(0, 12, 13)
cmap = plt.get_cmap("viridis", len(subjectsToAverageAll))
norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
sm_sn10 = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
sm_sn10.set_array([]) # Matplotlib requires an empty array initialization 
cbar = figSN10.colorbar(sm_sn10, ax=[axSN10speech, axSN10clicks], ticks=bounds, orientation="vertical", fraction=0.046, pad=0.04)
cbar.set_label("Subject")
figSN10.suptitle("SN10 cluster time: Cortical/MLR ROI TRFs")
axSN10speech.set_title("Speech")
axSN10clicks.set_title("Clicks")

sn10stats = sp.stats.ttest_rel(connectPointVals[:, 0], connectPointVals[:, 1])
print(sn10stats)

# np.random.seed(42)
# n_samples = 20
# before = np.random.normal(loc=50, scale=10, size=n_samples)
# # Simulate an increase in the 'after' condition with some random noise
# after = before + np.random.normal(loc=8, scale=5, size=n_samples)

# 2. Conduct the Paired Samples t-test
# t_stat, p_val = stats.ttest_rel(before, after)

# 3. Structure data for plotting
df = pd.DataFrame({"Speech": connectPointVals[:, 0], "Clicks": connectPointVals[:, 1], "Subject": range(len(subjectsToAverageAll))})
df_melted = pd.melt(
    df,
    id_vars=["Subject"],
    value_vars=["Speech", "Clicks"],
    var_name="Condition",
    value_name="Value",
)

# 4. Create the plot
plt.figure(figsize=(6, 6), dpi=100)
sb.set_theme(style="dark")

# Plot individual data points (stripplot)
sb.stripplot(
    data=df_melted,
    x="Condition",
    y="Value",
    hue="Condition",
    # palette="Set2",
    size=8,
    jitter=False,  # Keep points perfectly aligned on the x-axis to draw straight connecting lines
    legend=False,
)

# Plot boxplots to show distributions underneath
sb.boxplot(
    data=df_melted,
    x="Condition",
    y="Value",
    color="darkgray",
    width=0.4,
    boxprops=dict(alpha=0.3),
    showfliers=False,
)

# 5. Draw connecting lines between pairs
for i in range(len(subjectsToAverageAll)):
    plt.plot(
        [0, 1],
        [df.loc[i, "Speech"], df.loc[i, "Clicks"]],
        color="gray",
        linestyle="-",
        alpha=0.5,
        linewidth=2,
        zorder=1,  # Place lines behind the scatter points
    )

# 6. Add statistical annotation
# Position the text dynamically above the max data point
ypos = min(df_melted["Value"])
plt.text(
    x=0.5,
    y=ypos - (ypos * 0.15),
    s=f"Paired t-test:\nt = {sn10stats.statistic:.3f}\np = {sn10stats.pvalue:.4e}",
    horizontalalignment="center",
    verticalalignment="top",
    fontsize=11,
    fontweight="semibold",
    bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="gray", alpha=0.8),
)

# Formatting details
plt.title(
    "SN10 cluster time TRF minimums: Cortical/MLR ROIs",
    fontsize=13,
    pad=20,
    fontweight="bold",
)
# plt.xlabel("Experimental Condition", fontsize=11)
plt.ylabel("Z-scored current", fontsize=11)
plt.ylim(np.min(connectPointVals) - np.std(connectPointVals)/4, np.max(connectPointVals) + np.std(connectPointVals))

max_val = df_melted["Value"].max()
min_val = df_melted["Value"].min()
data_range = max_val - min_val

# Define bracket height thresholds
y_bracket = max_val + (data_range * 0.08)    # The ceiling bar level
y_tick = y_bracket - (data_range * 0.02)      # The downward tick tips
y_text = y_bracket + (data_range * 0.01)      # Text positioning above the line

# Draw the left tick, top bar, and right tick seamlessly
plt.plot([0, 0, 1, 1], [y_tick, y_bracket, y_bracket, y_tick], color="black", linewidth=1.5)

# Place the significance stars text center-aligned
plt.text(x=0.5, y=y_text, s="***", horizontalalignment="center", verticalalignment="bottom", fontsize=14, fontweight="bold")


plt.tight_layout()
plt.show()

# %%
stcAllSubjs = np.zeros((len(t), len(subjectsToAverageAll)))
for i, subject in enumerate(subjectsToAverageAll):
    stcTemp = stcFirst_fs_here.copy()
    stcTemp.data = group1mat[..., i]
    temp = mne.extract_label_time_course(stcTemp, [subjects_dir + f"/{avgBrain}/mri/{mgzLabel}", labelSubsetABR], src_fs, mode="auto")
    temp = calcPC(temp, True, None)
    stcAllSubjs[:, i] = temp.mean(axis=1)

plt.figure()
plt.plot(t, stcAllSubjs)
plt.plot(t, stcAllSubjs.mean(axis=-1), linewidth=2, alpha=.7, color="black")

# %%
stcAllSubjs = np.zeros((len(t), len(subjectsToAverageAll)))
for i, subject in enumerate(subjectsToAverageAll):
    stcTemp = stcFirst_fs_here.copy()
    stcTemp.data = group2mat[..., i]
    temp = mne.extract_label_time_course(stcTemp, [subjects_dir + f"/{avgBrain}/mri/{mgzLabel}", labelSubsetABR], src_fs, mode="auto")
    temp = calcPC(temp, True, None)
    stcAllSubjs[:, i] = temp.mean(axis=1)

plt.figure()
plt.plot(t, stcAllSubjs)
plt.plot(t, stcAllSubjs.mean(axis=-1), linewidth=2, alpha=.7, color="black")

# %%
# "CSF", "ctx_rh_G_cingul-Post-ventral", "ctx_lh_G_cingul-Post-ventral", "Right-Thalamus-Proper", "Left-Thalamus-Proper", "CC_Posterior"
# "ctx_rh_G_temp_sup-G_T_transv", "ctx_lh_G_temp_sup-G_T_transv", "ctx_rh_S_temporal_transverse", "ctx_lh_S_temporal_transverse"
# testROIs = ["Right-Thalamus-Proper", "Left-Thalamus-Proper"]
testROIs = ["ctx_rh_G_temp_sup-G_T_transv", "ctx_lh_G_temp_sup-G_T_transv", "ctx_rh_S_temporal_transverse", "ctx_lh_S_temporal_transverse", 
           "ctx_rh_S_circular_insula_inf", "ctx_lh_S_circular_insula_inf", "ctx_rh_Lat_Fis-post", "ctx_lh_Lat_Fis-post", 
           "ctx_rh_G_temp_sup-Plan_polar", "ctx_lh_G_temp_sup-Plan_polar",]# "ctx_rh_G_and_S_subcentral", "ctx_lh_G_and_S_subcentral"]

timePoint = int(.011 * 5000)

testDimsAllSubjs = []

for subjInd in range(12):

    stcSpeechHere = stcFirst_fs_here.copy()
    stcSpeechHere.data = stcSubsetsMNE[1][..., subjInd, 3]
    stcClicksHere = stcSecond_fs_here.copy()
    stcClicksHere.data = stcSubsetsMNE[1][..., subjInd, 4]
    testDim1Speech = []
    testDim2Speech = []
    testDim1Clicks = []
    testDim2Clicks = []
    for testROI in testROIs:
        testDim1Speech.append(stcSpeechHere.copy().project(directions=np.tile(np.array([0, 0, 1]), (totalNumVertices, 1)))[0].in_label(testROI, subjects_dir + f"/{avgBrain}/mri/{mgzLabel}", src_fs).data)
        testDim2Speech.append(stcSpeechHere.copy().project(directions=np.tile(np.array([0, 1, 0]), (totalNumVertices, 1)))[0].in_label(testROI, subjects_dir + f"/{avgBrain}/mri/{mgzLabel}", src_fs).data)
        testDim1Clicks.append(stcClicksHere.copy().project(directions=np.tile(np.array([0, 0, 1]), (totalNumVertices, 1)))[0].in_label(testROI, subjects_dir + f"/{avgBrain}/mri/{mgzLabel}", src_fs).data)
        testDim2Clicks.append(stcClicksHere.copy().project(directions=np.tile(np.array([0, 1, 0]), (totalNumVertices, 1)))[0].in_label(testROI, subjects_dir + f"/{avgBrain}/mri/{mgzLabel}", src_fs).data)
    testDim1Speech = np.concatenate(testDim1Speech, axis=0)
    testDim2Speech = np.concatenate(testDim2Speech, axis=0)
    testDim1Clicks = np.concatenate(testDim1Clicks, axis=0)
    testDim2Clicks = np.concatenate(testDim2Clicks, axis=0)

    testDimsAllSubjs.append([testDim1Speech, testDim2Speech, testDim1Clicks, testDim2Clicks])

    statsSpeech = sp.stats.directional_stats(np.stack((testDim1Speech[:, timePoint], testDim2Speech[:, timePoint]), axis=1))
    statsClicks = sp.stats.directional_stats(np.stack((testDim1Clicks[:, timePoint], testDim2Clicks[:, timePoint]), axis=1))
    print(f"Speech direction is {np.atan2(statsSpeech.mean_direction[0], statsSpeech.mean_direction[1])}, speech PLV is {statsSpeech.mean_resultant_length}")
    print(f"Clicks direction is {np.atan2(statsClicks.mean_direction[0], statsClicks.mean_direction[1])}, clicks PLV is {statsClicks.mean_resultant_length}")
    print(f"ROI combination has {testDim1Speech.shape[0]} total vertices")

# %%
labelsList = mne.get_volume_labels_from_aseg(subjects_dir + f"/{avgBrain}/mri/{mgzLabel}")
src_fs_new = mne.setup_volume_source_space(
    avgBrain,
    mri=subjects_dir + f"/{avgBrain}/mri/aparc.a2009s+aseg.mgz",
    pos=pos,
    bem=bem_fs,
    volume_label=labelsList,
    subjects_dir=subjects_dir,
    add_interpolator=True,  # just for speed, usually this should be True
    verbose=True,
)
labels = mne.get_volume_labels_from_src(src_fs_new, avgBrain, subjects_dir)

# %%
# testROIs = ["Thalamus-Proper-rh", "Thalamus-Proper-lh"]
# testROIs = ["ctx_rh_G_temp_sup-G_T_transv", "ctx_lh_G_temp_sup-G_T_transv", "ctx_rh_S_temporal_transverse", "ctx_lh_S_temporal_transverse", 
#            "ctx_rh_S_circular_insula_inf", "ctx_lh_S_circular_insula_inf", "ctx_rh_Lat_Fis-post", "ctx_lh_Lat_Fis-post", 
#            "ctx_rh_G_temp_sup-Plan_polar", "ctx_lh_G_temp_sup-Plan_polar",]
testROIsLeft = ["ctx_lh_G_temp_sup-G_T_transv", "ctx_lh_S_temporal_transverse",
                "ctx_lh_S_circular_insula_inf", "ctx_lh_Lat_Fis-post",
                "ctx_lh_G_pariet_inf-Supramar", 
               ]
testROIsRight = ["ctx_rh_G_temp_sup-G_T_transv", "ctx_rh_S_temporal_transverse",
                 "ctx_rh_S_circular_insula_inf", "ctx_rh_Lat_Fis-post",
                 "ctx_rh_G_pariet_inf-Supramar",
                ]
allVertsLeft = []
for testROI in testROIsLeft:
    for label in labels:
        if testROI == label.name:
            allVertsLeft.append(label.pos)
allVertsLeft = np.concatenate(allVertsLeft, axis=0)

allVertsRight = []
for testROI in testROIsRight:
    for label in labels:
        if testROI == label.name:
            allVertsRight.append(label.pos)
allVertsRight = np.concatenate(allVertsRight, axis=0)

# brain = mne.viz.Brain("fsaverage", hemi="lh", surf="white", alpha=0.5, background="black", subjects_dir=subjects_dir)
brain = mne.viz.plot_source_estimates(stcEBresult,
    subject=avgBrain,
    # surface=surface,
    surface="pial",
    hemi="both",
    src=src_fs,
    views="coronal",
    initial_time=0,
    subjects_dir=subjects_dir,
    brain_kwargs=dict(silhouette=True),
    smoothing_steps=7,
    show_traces=True,
    alpha=brain_alpha,
)
brain.add_foci(
    allVertsLeft*1000, 
    # coords_as_verts=True, 
    map_surface="pial",
    hemi="lh", 
    color="blue",
    scale_factor=0.5 # Optional: adjust size of plotted points
)
brain.add_foci(
    allVertsRight*1000, 
    # coords_as_verts=True, 
    map_surface="pial",
    hemi="rh", 
    color="blue",
    scale_factor=0.5 # Optional: adjust size of plotted points
)
# brain2 = mne.viz.plot_source_estimates(stcEBresult,
#     subject=avgBrain,
#     # surface=surface,
#     surface="pial",
#     hemi="both",
#     src=src_fs,
#     views="coronal",
#     initial_time=0,
#     subjects_dir=subjects_dir,
#     brain_kwargs=dict(silhouette=True),
#     smoothing_steps=7,
#     show_traces=True,
#     alpha=brain_alpha,
# )
# brain2.add_foci(
#     allVertsLeft*1000, 
#     # coords_as_verts=True, 
#     # map_surface="pial",
#     hemi="lh", 
#     color="blue",
#     scale_factor=0.5 # Optional: adjust size of plotted points
# )
# brain2.add_foci(
#     allVertsRight*1000, 
#     # coords_as_verts=True, 
#     # map_surface="pial",
#     hemi="rh", 
#     color="blue",
#     scale_factor=0.5 # Optional: adjust size of plotted points
# )

# %%
waveVstats = sp.stats.ttest_rel(waveVsAll[:, 3], waveVsAll[:, 4])
print(waveVstats)
print(f"Mean of speech is {waveVsAll[:, 3].mean()}, mean of clicks is {waveVsAll[:, 4].mean()}")
plt.figure()
plt.boxplot(waveVsAll)

# %%

# %%
