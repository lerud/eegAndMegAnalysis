# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: neuroTools
#     language: python
#     name: neurotools
# ---

# %%
import mne
import numpy as np
import eelbrain as eb
import scipy as sp
import mat73
import matplotlib
import matplotlib.pyplot as plt
import joblib
import os
import sys
import pickle
import glob
import time
import gc

# %%
fs = 5000

parentDir = "/Volumes/Seagate/map/"

subjectsToAverage = [
    "R3045",
    "R3089",
    "R3093",
    "R3095",
    "R3151",
    "R2774",
    "R3152",
    "R2877",
    "R3157",
    "R2783",
    "R3170",
    "R3172",
    "R3193",
    "R3184",
    "R3214",
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

badChanLists = [
    [["Fp2", "AF3", "FC1", "F3", "Fz", "Cz", "CP5", "P3", "CP6", "F7"], ["MEG 056", "MEG 086", "MEG 046"]],  # R3045
    [["P7", "CP6", "C4", "T7", "CP5", "P3", "P4", "O2", "Oz", "PO4", "F8"], ["MEG 056", "MEG 086"]],  # R3089
    [["Oz", "P8", "CP6", "Fp2", "F8"], ["MEG 056", "MEG 086"]],  # R3093
    [["P7", "T8", "O2", "PO4", "Fp2", "F4", "T7"], ["MEG 056", "MEG 086"]],  # R3095
    [["C3", "FC5", "P4", "Oz"], ["MEG 056", "MEG 086", "MEG 046"]],  # R3151
    [[
        "Fp1",
        "AF3",
        "F7",
        "F3",
        "Fz",
        "F4",
        "FC6",
        "C3",
        "CP5",
        "Pz",
        "CP6",
        "P8",
        "P3",
        "FC1",
        "FC5",
        "T7",
        "AF4",
        "PO4",
    ], ["MEG 056", "MEG 086"]],  # R2774
    [["T7", "C3", "P7", "Pz", "O1", "P8", "CP6", "Oz", "O2"], ["MEG 056", "MEG 086", "MEG 046", "MEG 062", "MEG 105"]],  # R3152
    [["CP5", "P7", "F3", "FC5", "C3", "P4", "FC6", "FC1", "P8"], ["MEG 056", "MEG 086", "MEG 046", "MEG 062"]],  # R2877
    [["CP6", "P8", "F4", "FC6", "T7", "C3", "Pz", "Oz", "O2", "P3"], ["MEG 056", "MEG 086", "MEG 046"]],  # R3157
    [["CP6", "P8", "T7", "CP5", "P7", "Oz", "FC5", "C3", "CP2"], ["MEG 056", "MEG 086", "MEG 046"]],  # R2783
    [["CP5", "P3", "F3", "P4", "T7", "Fp1"], ["MEG 056", "MEG 086", "MEG 046"]],  # R3170
    [["F8", "FC6", "C4", "T8", "CP6", "P8", "P4", "P7", "Pz", "CP5", "FC1", "F7", "F3", "Oz"], ["MEG 056", "MEG 086", "MEG 046", "MEG 157"]],  # R3172
    [None, ["MEG 056", "MEG 086", "MEG 046"]],  # R3193
    [["F3", "T7", "F4", "FC6", "P8", "FC2", "P4", "P7", "FC2"], ["MEG 056", "MEG 086", "MEG 046"]],  # R3184
    [["F8", "C3", "T8", "P7"], ["MEG 056", "MEG 086", "MEG 046"]],  # R3214
    [None, ["MEG 056", "MEG 086", "MEG 046", "MEG 062"]],  # R3265
    [["T7", "P7", "F7"], ["MEG 056", "MEG 086", "MEG 046", "MEG 157", "MEG 105"]],  # R3275
    [["P4", "F8", "F3"], ["MEG 056", "MEG 086", "MEG 046", "MEG 157"]],  # R3247
    [None, ["MEG 056", "MEG 086", "MEG 046"]],  # R3292
    [["Oz", "T7", "T8", "P3"], ["MEG 056", "MEG 086", "MEG 046"]],  # R3293
    [["CP6", "T8", "FC6", "T7", "FC5", "P7", "P8", "CP5", "F3", "Oz"], ["MEG 056", "MEG 086", "MEG 046"]],  # R3294
    [["CP5", "P7"], ["MEG 056", "MEG 086", "MEG 046"]],  # R3303
    [["FC1"], ["MEG 056", "MEG 086", "MEG 046"]],  # R3304
    [["CP6", "T8", "P8", "PO4", "PO3", "O1", "O2", "FC6", "F4", "P7"], ["MEG 056", "MEG 086", "MEG 046"]],  # R3306
    [["CP6", "P8", "FC5", "FC6", "F3", "T7", "PO3", "PO4", "O1", "T8", "F4", "CP5", "C3", "FC1"], ["MEG 056", "MEG 086", "MEG 046", "MEG 121"]],  # R3313
    [["CP6", "T8", "P7", "P8", "Oz", "T7"], ["MEG 056", "MEG 086", "MEG 046", "MEG 062"]],  # R3317
    [["P3", "F7", "FC5", "F8", "FC6", "F4"], ["MEG 056", "MEG 086", "MEG 046"]],  # R3320
]

subset = [i for i in range(len(subjectsToAverage))]
# subset = [26]
subjectsToAverage = [subjectsToAverage[i] for i in subset]
badChanLists = [badChanLists[i] for i in subset]

if fs == 5000:
    l_freq = 10
    h_freq = 300
elif fs == 500:
    l_freq = 2
    h_freq = 40

notchFreqs = np.arange(60, fs / 2, 60)

filterPhase = "minimum"

subDirs = "/eegAndMeg/eeg/"
subDirsMEG = "/eegAndMeg/meg/"

refs = ["EXG3", "EXG4"]

for subjInd, subject in enumerate(subjectsToAverage):

    badChans = badChanLists[subjInd][0]
    badChansMEG = badChanLists[subjInd][1]
    
    eegLocation = parentDir + subject + subDirs
    megLocation = parentDir + subject + subDirsMEG
    
    temp = sorted(glob.glob(f"{eegLocation}*baseline*bdf"))
    eegBaselineFile = temp[-1]
    temp = sorted(
        glob.glob(f"{megLocation}*baseline*fif")
    )  # Read both of these the same way, even though there really should never be more than one of either named this way
    megBaselineFile = temp[-1]
    
    eegBaselineRaw = mne.io.read_raw_bdf(eegBaselineFile, preload=True)
    eegBaselineRaw.resample(fs)
    megBaselineRaw = mne.io.read_raw_fif(megBaselineFile, preload=True)
    megBaselineRaw.resample(fs)
    
    # %%
    elp_ch_names = [
        "Mark1",
        "Mark2",
        "Mark3",
        "Mark4",
        "Mark5",
        "Fp1",
        "AF3",
        "F7",
        "F3",
        "FC1",
        "FC5",
        "T7",
        "C3",
        "CP1",
        "CP5",
        "P7",
        "P3",
        "Pz",
        "PO3",
        "O1",
        "Oz",
        "O2",
        "PO4",
        "P4",
        "P8",
        "CP6",
        "CP2",
        "C4",
        "T8",
        "FC6",
        "FC2",
        "F4",
        "F8",
        "AF4",
        "Fp2",
        "Fz",
        "Cz",
    ]
    
    channelFile = glob.glob(parentDir + subject + "/digitization/*eeg*.elp")
    if len(channelFile) == 0:
        channelFile = glob.glob(parentDir + subject + "/digitization/*EEG*.elp")
    print(f"Using the channel file {channelFile[0]}")
    
    digMontage = mne.channels.read_dig_polhemus_isotrak(
        channelFile[0], ch_names=elp_ch_names
    )
    changeChTypes = {
        "EXG1": "eog",
        "EXG2": "eog",
        "EXG3": "eog",
        "EXG4": "eog",
        "EXG5": "eog",
        "EXG6": "eog",
        "EXG7": "eog",
        "EXG8": "eog",
    }
    elpChangeChTypes = {
        "Mark1": "hpi",
        "Mark2": "hpi",
        "Mark3": "hpi",
        "Mark4": "hpi",
        "Mark5": "hpi",
    }
    
    eegBaselineRaw.set_channel_types(changeChTypes)
    eegBaselineRaw.set_montage(digMontage)
    
    if badChans is not None:
        eegBaselineRaw.info["bads"].extend(badChans)
        # baselineRaw.interpolate_bads()
    
    if badChansMEG is not None:
        megBaselineRaw.info["bads"].extend(badChansMEG)
        # baselineRaw.interpolate_bads()
    
    eegBaselineEvents = mne.find_events(eegBaselineRaw)
    eegStartTime = eegBaselineEvents[0, 0] / fs
    eegStopTime = eegBaselineEvents[1, 0] / fs
    eegDuration = eegStopTime - eegStartTime
    megBaselineEvents = mne.find_events(megBaselineRaw)
    megStartTime = megBaselineEvents[0, 0] / fs
    megStopTime = megBaselineEvents[1, 0] / fs
    # Crop both recordings to start at the trigger time (same time), and end at the same duration afterwards
    eegBaselineRaw.crop(tmin=eegStartTime, tmax=eegStartTime + eegDuration)
    megBaselineRaw.crop(tmin=megStartTime, tmax=megStartTime + eegDuration)
    
    # Now combine MEG and EEG baselines to do the rest of the computations and noise covariance calculation etc. together
    megBaselineRaw.add_channels([eegBaselineRaw], force_update_info=True)
    baselineRaw = megBaselineRaw
    
    baselineRaw.set_eeg_reference(ref_channels=refs)
    # baselineRaw.set_eeg_reference(projection=True)
    baselineRaw.notch_filter(notchFreqs, phase=filterPhase)
    
    # %%
    if l_freq is not None or h_freq is not None:
        baselineRaw.filter(l_freq=l_freq, h_freq=h_freq, phase=filterPhase)
        print("\n")
    
    # %%
    baselineCov = mne.compute_raw_covariance(baselineRaw, picks=["meg", "eeg"])

    eb.save.pickle(baselineCov, f"{megLocation}baselineCov_noProj_fs{fs}.pickle")
    








