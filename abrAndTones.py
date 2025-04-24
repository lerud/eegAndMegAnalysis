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
import pickle
import glob

from neuroAndSignalTools.freqAnalysis import *

matplotlib.use("QtAgg")
plt.ion()

# %%

subject = "R3265"
badChanList = None


runName = "trftrial"

# eventID = 130  # the usual
# eventID=32898  # R3089
# eventID=49282  # R3093

parentDir = "/Volumes/Seagate/map/"

eegLocation = parentDir + subject + "/eegAndMeg/eeg/"

bdfFilename = subject + "_" + runName + ".fif"


exgLabels = ["EXG1", "EXG2", "EXG3", "EXG4", "EXG5", "EXG6", "EXG7", "EXG8"]
notchFreqs = np.arange(60, 8192, 60)
refs = ["EXG3", "EXG4"]

abr_lFreq = 40
abr_hFreq = 1000
tones_lFreq = 3
tones_hFreq = 10

edgePad = 0.001

# %%
abrAndTones = mne.io.read_raw_fif(f"{eegLocation}{bdfFilename}", preload=True)
abrAndTones.notch_filter(notchFreqs)
abrAndTones.set_eeg_reference(ref_channels=refs)

events = mne.find_events(abrAndTones, shortest_event=0)
print("\n")
print("Here is a sample of the initial events matrix:")
print("\n")
print(events[:30, :])
print(events[-30:, :])
events[:, 2] = events[:, 2] - events[:, 1]
print("\n")
print("\n")
print("Here is a sample of the events matrix after subtracting the middle column:")
print("\n")
print(events[:30, :])
print(events[-30:, :])

# %%
# soundDelay = (
#     3.1 / 343 - 0.00275
# )  # Speed of sound delay in ear tubes, in seconds, minus the AN model delay compensation (2.75 ms according to Shan et al.). Convert this to samples below and add it to all trigger times
soundDelay = (
    3.1 / 343
)  # Speed of sound delay in ear tubes, in seconds. Convert this to samples below and add it to all trigger times
presentationDelay = 0.0017  # Length of time that Presentation triggers are early, relative to Triggy triggers, on average

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
# elpChangeChTypes = {
#     "Mark1": "hpi",
#     "Mark2": "hpi",
#     "Mark3": "hpi",
#     "Mark4": "hpi",
#     "Mark5": "hpi",
# }

abrAndTones.set_channel_types(changeChTypes)
abrAndTones.set_montage(digMontage)

if badChanList is not None:
    abrAndTones.info["bads"].extend(badChanList)
    abrAndTones.interpolate_bads()


# %%
abrAndTones.crop(
    tmin=soundDelay
    + presentationDelay
    + events[events[:, 2] == 142, 0][0] / abrAndTones.info["sfreq"],
    tmax=soundDelay
    + presentationDelay
    + events[events[:, 2] == 148, 0][0] / abrAndTones.info["sfreq"],
)

# %%
tonesRaw = abrAndTones.copy().resample(sfreq=500)
tonesRaw.filter(l_freq=tones_lFreq, h_freq=tones_hFreq)
tonesResponse = tonesRaw.get_data(picks="eeg").squeeze().T

toneTimes = eb.load.unpickle("regressorWork/toneTimes.pickle")
[stimFs, stim] = sp.io.wavfile.read("regressorWork/fullABRstim_triggersNegPosLeft.wav")
tonesRegressor = createPredictorTimeseries(
    toneTimes,
    tonesRaw.info["sfreq"],
    int(round(len(stim) * tonesRaw.info["sfreq"] / stimFs)),
)

print(
    f"""Length of cropped EEG response is {tonesRaw.times[-1]}, and length of corresponding regressor at the same sampling rate is {len(tonesRegressor) / tonesRaw.info["sfreq"]}"""
)
print(
    f"""Resampling EEG response to have length {len(tonesRegressor) / tonesRaw.info["sfreq"]} seconds"""
)

tonesResponse = sp.signal.resample(tonesResponse, len(tonesRegressor))

# %%
tonesRegressor = sp.stats.zscore(tonesRegressor)
tonesResponse = sp.stats.zscore(tonesResponse)

rfTones = mne.decoding.ReceptiveField(
    -0.1 - edgePad, 0.75 + edgePad, tonesRaw.info["sfreq"], estimator=0
)
rfTones.fit(tonesRegressor[:, np.newaxis], tonesResponse)
TRFsTones = np.moveaxis(
    rfTones.coef_[
        :,
        :,
        int(edgePad * tonesRaw.info["sfreq"]) : int(
            -edgePad * tonesRaw.info["sfreq"] - 1
        ),
    ],
    -1,
    0,
).squeeze()
tonesEvoked = mne.EvokedArray(TRFsTones.T, tonesRaw.pick("eeg").info, tmin=-0.1)

# %%
eventID = 130
eventsTones = np.zeros((len(toneTimes), 3))
eventsTones[:, 0] = np.round(toneTimes * tonesRaw.info["sfreq"]).astype(int)
eventsTones[:, 2] = eventID

event_dict_tones = {"Tones": eventID}
reject_criteria = dict(
    #    mag=4000e-15,  # 4000 fT
    #   grad=4000e-13,  # 4000 fT/cm
    # eeg=250e-6,  # 150 µV  # Often this threshold seems best
    eeg=550e-6,  # 150 µV  # Often this threshold seems best
    #    eeg=550e-5,  # 150 µV
    #    eog=250e-6,
)  # 250 µV
epochs = mne.Epochs(
    # tonesRaw.copy().resample(sfreq=tonesRaw.info["sfreq"] * tonesResponse.shape[0] / tonesRegressor.shape[0]),
    tonesRaw,
    eventsTones.astype(int),
    event_id=event_dict_tones,
    tmin=-0.1,  # These are ok for slower ERP
    tmax=0.75,
    #    tmin=-0.05,  # Try these for FFRish ERP?
    #    tmax=.2,
    reject=reject_criteria,
    preload=True,
)
toneEpochs = epochs["Tones"]
tonesEvokedMNE = toneEpochs.average()

# %%
tonesEvoked.plot_topo(color="r", legend=False)
tonesEvokedMNE.plot_topo(color="r", legend=False)

# %%
abrAndTones.filter(l_freq=abr_lFreq, h_freq=abr_hFreq)
abrResponse = abrAndTones.get_data(picks="eeg").squeeze().T

abrTimes, prevICIvalues = eb.load.unpickle("regressorWork/clickTimes.pickle")
[stimFs, stim] = sp.io.wavfile.read("regressorWork/fullABRstim_triggersNegPosLeft.wav")
values = 1 / np.array(prevICIvalues)
values[0] = 0
abrRegressor = createPredictorTimeseries(
    abrTimes,
    abrAndTones.info["sfreq"],
    int(round(len(stim) * abrAndTones.info["sfreq"] / stimFs)),
)
# abrRegressor = createPredictorTimeseries(abrTimes, abrAndTones.info["sfreq"], int(round(len(stim) * abrAndTones.info["sfreq"] / stimFs)), values=values)

print(
    f"""Length of cropped EEG response is {abrAndTones.times[-1]}, and length of corresponding regressor at the same sampling rate is {len(abrRegressor) / abrAndTones.info["sfreq"]}"""
)
print(
    f"""Resampling EEG response to have length {len(abrRegressor) / abrAndTones.info["sfreq"]} seconds"""
)

abrResponse = sp.signal.resample(abrResponse, len(abrRegressor))

# %%
# abrRegressor = sp.stats.zscore(abrRegressor)
# abrResponse = sp.stats.zscore(abrResponse)

rfAbr = mne.decoding.ReceptiveField(
    -0.01 - edgePad, 0.075 + edgePad, abrAndTones.info["sfreq"], estimator=0
)
rfAbr.fit(abrRegressor[:, np.newaxis], abrResponse)
TRFsAbr = np.moveaxis(
    rfAbr.coef_[
        :,
        :,
        int(edgePad * abrAndTones.info["sfreq"]) : int(
            -edgePad * abrAndTones.info["sfreq"] - 1
        ),
    ],
    -1,
    0,
).squeeze()
abrEvoked = mne.EvokedArray(TRFsAbr.T, abrAndTones.pick("eeg").info, tmin=-0.01)

# %%
abrEvoked.plot_topo(color="r", legend=False)

# %%
