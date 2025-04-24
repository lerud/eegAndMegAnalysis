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
# subject='R3151';badChanList=['C3','FC5']

# subject = "R2774"
# badChanList = [
#     "Fp1",
#     "AF3",
#     "F7",
#     "F3",
#     "Fz",
#     "F4",
#     "FC6",
#     "C3",
#     "CP5",
#     "Pz",
#     "CP6",
#     "P8",
#     "FC1",
#     "FC5",
#     "T7",
#     "AF4",
# ]

# subject='R3152';badChanList=['T7','C3','P7','Pz','O1','P8','CP6']
# subject='R2877';badChanList=['CP5','P7','F3','FC5','C3','P4','FC6','FC1','P8']

# subject = "R3214"
# badChanList = ["T8"]

subject = "R3265"
badChanList = None

# badChanList = ["CP6"]  # Keep an eye on CP2 and possibly others
# subject='R2783';badChanList=['T7','P7','CP5']

runName = "tones"
# runName='stacks'

eventID = 130  # the usual
# eventID=32898  # R3089
# eventID=49282  # R3093

parentDir = "/Volumes/Seagate/map/"

# eegLocation='/Users/karl/Dropbox/UMD/multilevel0/230908/'
# eegLocation='/Users/karl/Dropbox/UMD/R2881/eegAndMeg/eeg/'
# eegLocation='/Users/karl/map/R3045/eegAndMeg/eeg/'
# eegLocation = "/Users/karl/map/" + subject + "/eegAndMeg/eeg/"
eegLocation = parentDir + subject + "/eegAndMeg/eeg/"

# bdfFilename='multilevel0_tones.bdf'
# bdfFilename='R3045_tones.bdf'
# bdfFilename='R3045_stacks.bdf'

bdfFilename = subject + "_" + runName + ".fif"

# matFilename='tonesSnsTspcaOutput.mat'
# matFilename='stacksSnsTspcaOutput.mat'
matFilename = runName + "SnsTspcaOutput.mat"

exgLabels = ["EXG1", "EXG2", "EXG3", "EXG4", "EXG5", "EXG6", "EXG7", "EXG8"]
denoiseMatlab = False
refs = ["EXG3", "EXG4"]
# refs=["Fp1"]

edgePad = 0.01

# %%
tones = mne.io.read_raw_fif(f"{eegLocation}{bdfFilename}", preload=True)

# %%
if not denoiseMatlab:
    tonesDenoised = tones
    notchFreqs = np.arange(60, 8192, 60)
    tonesDenoised.notch_filter(notchFreqs)
    tonesDenoised.set_eeg_reference(ref_channels=refs)
else:
    preprocMat = mat73.loadmat(f"{eegLocation}{matFilename}")["full_output"]
    # print(preprocMat.shape)
    # print(preprocMat[32,1000000])
    # plt.plot(preprocMat[31,0:7300])
    exgs = tones.copy().pick_channels(exgLabels)[:][0]
    # [eeg channels, 8 exgs, trigger] -- this should match the raw info
    # don't use the 8 exgs anywhere; shape just needs to match the raw info
    data = np.concatenate([preprocMat[:32], exgs, preprocMat[32:33]], axis=0)
    tonesDenoised = mne.io.RawArray(data, tones.info, tones.first_samp)

# %%
events = mne.find_events(tonesDenoised, shortest_event=0)
tonesDenoised, events = tonesDenoised.resample(sfreq=500, events=events, verbose=True)
# events = mne.find_events(tonesDenoised)
# print(tonesDenoised.ch_names)
print(events[:30, :])
print(events[-30:, :])
events[:, 2] = events[:, 2] - events[:, 1]
print("\n")
print(events[:30, :])
print(events[-30:, :])

# %%
easycap_montage = mne.channels.make_standard_montage("biosemi32")
listChTypes = tonesDenoised.get_channel_types()
# listChTypes[32:40]=['eog','eog','eog','eog','eog','eog','eog','eog']
print(listChTypes)
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
tonesDenoised.set_channel_types(changeChTypes)
tonesDenoised.set_montage(easycap_montage)

# tonesDenoised.compute_psd(fmax=8192).plot(picks="data", exclude="bads")

# %%
# tonesDenoised.set_eeg_reference(ref_channels='average')

# %%
# tonesDenoised.filter(l_freq=0.1, h_freq=20)
tonesDenoised.filter(l_freq=3, h_freq=20)
# tonesDenoised.filter(l_freq=80,h_freq=630)
print(tonesDenoised.info)

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

tonesDenoised.set_channel_types(changeChTypes)
tonesDenoised.set_montage(digMontage)

if badChanList is not None:
    tonesDenoised.info["bads"].extend(badChanList)
    tonesDenoised.interpolate_bads()

# %%
# plt.figure()
# plt.plot(data[40,145700:146000])
# plt.show()

# %%
# This is supposed to be 130, but may be 32898 or other things...??

# event_dict = {"Tones": 130}
# event_dict = {"Tones": 32898}
event_dict = {"Tones": eventID}

# fig = mne.viz.plot_events(
#    events, event_id=event_dict, sfreq=tonesDenoised.info["sfreq"], first_samp=tonesDenoised.first_samp
# )

# %%
reject_criteria = dict(
    #    mag=4000e-15,  # 4000 fT
    #   grad=4000e-13,  # 4000 fT/cm
    # eeg=250e-6,  # 150 µV  # Often this threshold seems best
    eeg=550e-6,  # 150 µV  # Often this threshold seems best
    #    eeg=550e-5,  # 150 µV
    #    eog=250e-6,
)  # 250 µV

# %%
epochs = mne.Epochs(
    tonesDenoised,
    events,
    event_id=event_dict,
    tmin=-0.2,  # These are ok for slower ERP
    tmax=0.7,
    #    tmin=-0.05,  # Try these for FFRish ERP?
    #    tmax=.2,
    reject=reject_criteria,
    preload=True,
)

# %%
toneEpochs = epochs["Tones"]
# toneEpochs.ch_names

# toneEpochs.plot_image(picks=["FC1", "CP2"])

# %%
tonesEvoked = toneEpochs.average()

# %%
toneTimes = events[events[:, 2] == 130, 0] / tonesDenoised.info["sfreq"]
tonesRegressor = createPredictorTimeseries(
    toneTimes, tonesDenoised.info["sfreq"], tonesDenoised.get_data().shape[1]
)
tonesResponse = tonesDenoised.get_data(picks="eeg").T

tonesRegressor = sp.stats.zscore(tonesRegressor)
tonesRegressor = sp.ndimage.gaussian_filter1d(
    tonesRegressor, 0.007 * tonesDenoised.info["sfreq"]
)
tonesResponse = sp.stats.zscore(tonesResponse)

rfTones = mne.decoding.ReceptiveField(
    -0.2 - edgePad, 0.7 + edgePad, tonesDenoised.info["sfreq"], estimator=1e1
)
rfTones.fit(tonesRegressor[:, np.newaxis], tonesResponse)
TRFsTones = np.moveaxis(
    rfTones.coef_[
        :,
        :,
        int(edgePad * tonesDenoised.info["sfreq"]) : int(
            -edgePad * tonesDenoised.info["sfreq"] - 1
        ),
    ],
    -1,
    0,
).squeeze()
tonesTrfsEvoked = mne.EvokedArray(
    TRFsTones.T, tonesDenoised.pick("eeg").info, tmin=-0.2
)

# %%
tonesEvoked.pick_types(eeg=True).plot_topo(color="r", legend=False)
tonesTrfsEvoked.pick_types(eeg=True).plot_topo(color="r", legend=False)


# %%
# tonesEvokedSpec=tonesEvoked.compute_psd(method="welch", tmin=0, tmax=.25, fmin=50, fmax=550, n_per_seg=16384, n_fft=16384)
# tonesEvokedSpec.plot(picks="data", exclude="bads")
# tonesEvokedSpec.plot_topo(color="k", fig_facecolor="w", axis_facecolor="w")

# freqs = np.logspace(*np.log10([70, 250]), num=80)
# n_cycles = freqs / 8  # different number of cycle per frequency
# power, itc = mne.time_frequency.tfr_morlet(
#    toneEpochs,
#    freqs=freqs,
#    n_cycles=n_cycles,
#    use_fft=True,
#    return_itc=True,
#    decim=3,
#    n_jobs=None,
# )
# power.plot_topo(baseline=(-0.1, 0), mode="logratio", title="Average power")
# power.plot([20], baseline=(-0.1, 0), mode="logratio", title=power.ch_names[20])

# fig, axes = plt.subplots(1, 2, figsize=(7, 4), constrained_layout=True)
# topomap_kw = dict(
#    ch_type="eeg", tmin=0, tmax=.3, baseline=(-0.1, 0), mode="logratio", show=False
# )
# plot_dict = dict(F0=dict(fmin=90, fmax=110), Harm=dict(fmin=190, fmax=210))
# for ax, (title, fmin_fmax) in zip(axes, plot_dict.items()):
#    power.plot_topomap(**fmin_fmax, axes=ax, **topomap_kw)
#    ax.set_title(title)


# %%

# %%
# tonesDenoised.save('tonesDenoised.fif')

# %%
# eb.save.pickle(tonesEvoked,'tonesEvoked')
# eb.save.pickle(toneEpochs,'toneEpochs')

# %%
# onsetTimes=events[1:,0]/16384

# %%
# onsetDiffs=np.diff(onsetTimes)
# print(onsetTimes)
# print(onsetDiffs)

# %%
# plt.hist(onsetDiffs,bins=600)
# plt.show()

# %%
# data=tonesDenoised[31,:]
# plt.plot(data[1],data[0].T)
# plt.show()

# %%
# print(onsetTimes)

# %%
