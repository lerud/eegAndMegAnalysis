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
import scipy as sp
import glob
import os
import eelbrain as eb
import matplotlib
import matplotlib.pyplot as plt
import time

matplotlib.use("QtAgg")
plt.ion()

# %%
t0 = time.time()

subject = "R3303"
parentDir = "/Volumes/Seagate/map/"
eegLocation = f"{parentDir}{subject}/eegAndMeg/eeg/"
megLocation = f"{parentDir}{subject}/eegAndMeg/meg/"

# badChanList = ["T7", "C3", "P7", "Pz", "O1", "P8", "CP6", "Oz"]  # R3152
# MEG_bad_channels = ["MEG 056", "MEG 086", "MEG 046", "MEG 062"]  # R3152



# badChanList = None  # R3292
# MEG_bad_channels = ["MEG 056", "MEG 086", "MEG 046"]  # R3292

# badChanList = ["Oz", "T7", "T8", "P3"]  # R3293
# MEG_bad_channels = ["MEG 056", "MEG 086", "MEG 046"]  # R3293

badChanList = ["CP5", "P7"]  # R3303
MEG_bad_channels = ["MEG 056", "MEG 086", "MEG 046"]  # R3303

refs = ["EXG3", "EXG4"]
notchFreqs = np.arange(60, 8192, 60)
notchFreqsMeg = np.arange(60, 1000, 60)

rawEEG = mne.io.read_raw_fif(f"{eegLocation}{subject}_maintask.fif", preload=True)
rawMEG = mne.io.read_raw_fif(f"{megLocation}{subject}_maintask-raw.fif", preload=True)

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

rawEEG.set_channel_types(changeChTypes)
rawEEG.set_montage(digMontage)
# denoised.set_montage(easycap_montage)

if badChanList is not None:
    rawEEG.info["bads"].extend(badChanList)
    rawEEG.interpolate_bads(reset_bads=False)

for MEG_bad_channel in MEG_bad_channels:
    rawMEG.info["bads"].append(MEG_bad_channel)

rawMEG.interpolate_bads(reset_bads=False)

rawEEG.set_eeg_reference(ref_channels=refs)
rawEEG.notch_filter(notchFreqs)
rawMEG.notch_filter(notchFreqsMeg)

# %%
rawEEG.filter(l_freq=1500, h_freq=None, phase="minimum")
rawMEG.filter(l_freq=350, h_freq=None, phase="minimum")

# %%
eegNoiseCov = mne.compute_raw_covariance(rawEEG)
eegNoiseCov.plot(info=rawEEG.info, exclude="bads")
rawEEGdown = rawEEG.copy().resample(sfreq=rawMEG.info["sfreq"])

# %%
lenEEG = rawEEGdown.times[-1]
lenMEG = rawMEG.times[-1]
if lenEEG > lenMEG:
    rawEEGdown.crop(tmax=lenMEG)
else:
    rawMEG.crop(tmax=lenEEG)
rawMEEG = rawMEG.copy().add_channels([rawEEGdown], force_update_info=True)
megNoiseCov = mne.compute_raw_covariance(rawMEEG)

# %%
megNoiseCov.plot(rawMEEG.info, exclude="bads")
eegSize = eegNoiseCov.data.shape[0]
megNoiseCovAlt = megNoiseCov.copy()
megNoiseCovAlt.data[-eegSize:, -eegSize:] = eegNoiseCov.data
megNoiseCovAlt.plot(rawMEEG.info, exclude="bads")


# %%
t1 = time.time()
print(f"Total time taken was {t1-t0} seconds")

# %%
temp=megNoiseCovAlt.data[:154, :154]
mask=np.ones(temp.shape) - np.eye(temp.shape[0])
plt.imshow(temp*mask)

# %%
