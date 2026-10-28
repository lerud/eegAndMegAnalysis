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
import mne
import sklearn
import numpy as np
import eelbrain as eb
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt
import pickle
import glob
import time
import os

from neuroAndSignalTools.freqAnalysis import *

matplotlib.use("QtAgg")
plt.ion()

# %%
calculateNow = True  # Else it will skip the main computation block and load pickles at the end (as it would anyway after computation and saving)
saveSummaryPickle = True
plotting = True
saveFigs = True

# subject = "R3265"
# badChanList = None

# subject = "R3275"
# badChanList = None

# subject = "R3247"
# badChanList = ["AF3", "F3", "P4"]

# subject = "notSubjectR3283"
# badChanList = ["CP6", "P8", "T7"]

# subject = "R3292"
# badChanList = None

# subject = "R3293"
# badChanList = None

# subject = "R3294"
# badChanList = None

# subject = "R3303"
# badChanList = ["P7", "CP5"]

# subject = "R3304"
# badChanList = None

# subject = "R3306"
# badChanList = ["CP6", "P8"]

# subjects = ["R3265", "R3275", "R3247", "notSubjectR3283", "R3292", "R3293", "R3294", "R3303", "R3304", "R3306"]
# badChanLists = [None, None, ["AF3", "F3", "P4"], ["CP6", "P8", "T7"], None, None, None, ["P7", "CP5"], None, ["CP6", "P8"]]

subjects = [
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
    "R0286",
    "R3362",
]
badChanLists = [
    [None, ["MEG 056", "MEG 086", "MEG 046", "MEG 062"]],  # R3265
    [
        ["T7", "P7", "F7", "T8"],
        ["MEG 056", "MEG 086", "MEG 046", "MEG 157", "MEG 105"],
    ],  # R3275
    [["P4", "F8", "F3"], ["MEG 056", "MEG 086", "MEG 046", "MEG 157"]],  # R3247
    [["T8", "FC6"], ["MEG 056", "MEG 086", "MEG 046"]],  # R3292
    [["Oz", "T7", "T8", "P3"], ["MEG 056", "MEG 086", "MEG 046"]],  # R3293
    [
        ["CP6", "T8", "FC6", "T7", "FC5", "P7", "P8", "CP5", "F3", "Oz"],
        ["MEG 056", "MEG 086", "MEG 046"],
    ],  # R3294
    [["CP5", "P7"], ["MEG 056", "MEG 086", "MEG 046"]],  # R3303
    [["FC1"], ["MEG 056", "MEG 086", "MEG 046"]],  # R3304
    [
        ["CP6", "T8", "P8", "PO4", "PO3", "O1", "O2", "FC6", "F4", "P7"],
        ["MEG 056", "MEG 086", "MEG 046"],
    ],  # R3306
    [
        [
            "CP6",
            "P8",
            "FC5",
            "FC6",
            "F3",
            "T7",
            "PO3",
            "PO4",
            "O1",
            "T8",
            "F4",
            "CP5",
            "C3",
            "FC1",
        ],
        ["MEG 056", "MEG 086", "MEG 046", "MEG 121"],
    ],  # R3313
    [
        ["CP6", "T8", "P7", "P8", "Oz", "T7"],
        ["MEG 056", "MEG 086", "MEG 046", "MEG 062"],
    ],  # R3317
    [
        ["P3", "F7", "FC5", "F8", "FC6", "F4", "T8"],
        ["MEG 056", "MEG 086", "MEG 046"],
    ],  # R3320
    [["PO3"], ["MEG 056", "MEG 086", "MEG 046", "MEG 087"]],  # R0286
    [["P8"], ["MEG 056", "MEG 086", "MEG 062", "MEG 087", "MEG 105"]],  # R3362
]
# subset = np.array([13])
subset = np.arange(12)
subjects = [subjects[i] for i in subset]
badChanLists = [badChanLists[i] for i in subset]


useANmodel = True


gaussianSDabr = 0.0005
# gaussianSDabr = None
gaussianSDtones = 0.007
# gaussianSDtones = None

lambdas = np.array([0, 1e2, 1e4, 1e6, 1e8, 1e10, 1e12, 1e14, 1e16, 1e18, 1e20, 1e22])
# lambdaReg = 1e10
# lambdaReg = 0
# lambdaRegInd = 0

reg_type = "ridge"
# reg_type = "laplacian"

runName = "trftrial"

# eventID = 130  # the usual
# eventID=32898  # R3089
# eventID=49282  # R3093


windowLen = 1
numBins = 8
nDelays = 0

saveDir = "/Users/karl/Library/CloudStorage/GoogleDrive-lerud@umd.edu/My Drive/meegSources/filteringAndRegularization/"
parentDir = "/Volumes/Seagate/map/"
doMultTRFs = True


# filterPhase = "zero"
filterPhase = "minimum"

tminABR = -0.025
tmaxABR = 0.075
tminTone = -0.05
tmaxTone = 0.5

scoring = "corrcoef"
n_folds = 10
n_jobs = 2

startInd = 6

# %%
for lambdaRegInd, lambdaReg in enumerate(lambdas[startInd:], start=startInd):

    if calculateNow:

        for subCount, subject in enumerate(subjects):

            badChanList = badChanLists[subCount]

            # startTrigger = 256
            # stopTriggers = np.array([512, 660, 768, 916])
            # presentationDelay = 0  # Length of time that Presentation triggers are early, relative to Triggy triggers, on average
            # presStopCorrection = 0  # Length of time that Presentation stop triggers are late, relative to Triggy triggers and regressors, because of the added triggering click at the end of the wav file

            startTrigger = 142
            stopTriggers = np.array([148, 404, 660, 916])
            presentationDelay = 0.0017  # Length of time that Presentation triggers are early, relative to Triggy triggers, on average
            presStopCorrection = (
                -1 / 600
            )  # Length of time that Presentation stop triggers are late, relative to Triggy triggers and regressors, because of the added triggering click at the end of the wav file

            startTriggerMeg = 174
            stopTriggerMeg = 180
            # badChanListMeg = ["MEG 056", "MEG 086"]
            presentationDelayMeg = 0.0017  # Length of time that Presentation triggers are early, relative to Triggy triggers, on average
            presStopCorrectionMeg = (
                -1 / 600
            )  # Length of time that Presentation stop triggers are late, relative to Triggy triggers and regressors, because of the added triggering click at the end of the wav file

            soundDelay = (
                3.1 / 343
            )  # Speed of sound delay in ear tubes, in seconds. Convert this to samples below and add it to all trigger times

            if useANmodel:
                ANmodelDelay = (
                    -0.00275
                )  # the AN model delay compensation (2.75 ms according to Shan et al.)
            else:
                ANmodelDelay = 0

            eegLocation = parentDir + subject + "/eegAndMeg/eeg/"
            megLocation = parentDir + subject + "/eegAndMeg/meg/"

            bdfFilename = subject[-5:] + "_" + runName + ".fif"
            bdfFilenameMeg = subject[-5:] + "_" + runName + "-raw.fif"

            exgLabels = ["EXG1", "EXG2", "EXG3", "EXG4", "EXG5", "EXG6", "EXG7", "EXG8"]
            notchFreqs = np.arange(60, 8192, 60)
            notchFreqsMeg = np.arange(60, 1000, 60)
            # notchFreqs = np.array([60, 180, 300])
            # notchFreqsMeg = np.array([60, 180, 300])
            refs = ["EXG3", "EXG4"]

            # abr_lFreq = 20
            # abr_hFreq = 999
            abr_lFreq = 20
            abr_hFreq = 300
            abr_lFreq_forReg = 20
            abr_hFreq_forReg = 300
            tones_lFreq = 2
            tones_hFreq = 40
            tones_lFreq_forReg = 2
            tones_hFreq_forReg = 40

            edgePadFast = 0.04
            edgePadSlow = 0.1

            addNoiseBeforeFilter = False
            addNoiseAfterFilter = True
            noiseRatio = 2.5

            filterRegressors = True

            values = 1  # This is the values that go into the create predictor function, so the values of each discrete predictor up sample

            zScoreRegressors = True
            divByStds = False

            # Set the analysis sampling frequencies up here early
            abrFs = 5000
            toneFs = 500

            abrAndTones = mne.io.read_raw_fif(
                f"{eegLocation}{bdfFilename}", preload=True
            )
            abrAndTones.notch_filter(notchFreqs, phase=filterPhase)

            abrAndTonesMeg = mne.io.read_raw_fif(
                f"{megLocation}{bdfFilenameMeg}", preload=True
            )
            abrAndTonesMeg.notch_filter(notchFreqsMeg, phase=filterPhase)

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
            print(
                "Here is a sample of the events matrix after subtracting the middle column:"
            )
            print("\n")
            print(events[:30, :])
            print(events[-30:, :])

            eventsMeg = mne.find_events(abrAndTonesMeg, shortest_event=0)

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

            badChanListEeg = badChanList[0]
            badChanListMeg = badChanList[1]

            if badChanListEeg is not None:
                abrAndTones.info["bads"].extend(badChanListEeg)
                abrAndTones.interpolate_bads(reset_bads=False)

            if badChanListMeg is not None:
                abrAndTonesMeg.info["bads"].extend(badChanListMeg)
                abrAndTonesMeg.interpolate_bads(reset_bads=False)

            for i in range(len(stopTriggers)):
                if any(events[:, 2] == stopTriggers[i]):
                    stopTriggerInd = i

            stopTrigger = stopTriggers[stopTriggerInd]
            print(f"Using EEG stop trigger {stopTrigger}")

            # Add the AN model compensation here, but reverse-compensate for it below for epoching
            abrAndTones.crop(
                tmin=soundDelay
                + ANmodelDelay
                + presentationDelay
                + events[events[:, 2] == startTrigger, 0][-1]
                / abrAndTones.info["sfreq"],
                tmax=soundDelay
                + ANmodelDelay
                + presentationDelay
                + presStopCorrection
                + events[events[:, 2] == stopTrigger, 0][-1]
                / abrAndTones.info["sfreq"],
            ).pick("eeg")
            abrAndTonesMeg.crop(
                tmin=soundDelay
                + ANmodelDelay
                + presentationDelayMeg
                + eventsMeg[eventsMeg[:, 2] == startTriggerMeg, 0][-1]
                / abrAndTonesMeg.info["sfreq"],
                tmax=soundDelay
                + ANmodelDelay
                + presentationDelayMeg
                + presStopCorrectionMeg
                + eventsMeg[eventsMeg[:, 2] == stopTriggerMeg, 0][-1]
                / abrAndTonesMeg.info["sfreq"],
            ).pick("meg")

            print("\n")
            print("Now doing highpass noise covariances")
            print("\n")
            eegCopyForNoise = abrAndTones.copy().filter(
                l_freq=1500, h_freq=None, phase=filterPhase
            )
            eegNoiseCov = mne.compute_raw_covariance(eegCopyForNoise)
            eb.save.pickle(
                eegNoiseCov, eegLocation + "highpassNoiseCovariance_abrAndTones.pickle"
            )
            eegCopyForNoise.resample(
                sfreq=2000
            )  # Magic number 2000 here is hopefully the MEG sampling rate
            megCopyForNoise = abrAndTonesMeg.copy().filter(
                l_freq=350, h_freq=None, phase=filterPhase
            )
            lenEEG = eegCopyForNoise.times[-1]
            lenMEG = megCopyForNoise.times[-1]
            if lenEEG > lenMEG:
                eegCopyForNoise.crop(tmax=lenMEG)
            else:
                megCopyForNoise.crop(tmax=lenEEG)
            megCopyForNoise.add_channels([eegCopyForNoise], force_update_info=True)
            megNoiseCov = mne.compute_raw_covariance(megCopyForNoise)
            eegSize = eegNoiseCov.data.shape[0]
            megNoiseCov.data[-eegSize:, -eegSize:] = (
                eegNoiseCov.data
            )  # replace the EEG covariances with what we calculated above at the full sampling rate
            eb.save.pickle(
                megNoiseCov, megLocation + "highpassNoiseCovariance_abrAndTones.pickle"
            )
            del eegCopyForNoise
            del megCopyForNoise
            print("\n")
            print("Done with highpass noise covariances")
            print("\n")

            abrAndTones.resample(abrFs)
            abrAndTonesMeg.resample(abrFs)

            lenMeg = len(abrAndTonesMeg.times)
            lenEeg = len(abrAndTones.times)
            maxLenInd = np.argmax((lenMeg, lenEeg))
            if maxLenInd == 0:
                abrAndTonesMeg = mne.io.RawArray(
                    data=abrAndTonesMeg.get_data()[:, :lenEeg], info=abrAndTonesMeg.info
                )
            elif maxLenInd == 1:
                abrAndTones = mne.io.RawArray(
                    data=abrAndTones.get_data()[:, :lenMeg], info=abrAndTones.info
                )

            abrAndTonesMeg.add_channels([abrAndTones], force_update_info=True)
            abrAndTonesMEEG = abrAndTonesMeg

            tonesRawMEEG = abrAndTonesMEEG.copy().resample(
                sfreq=toneFs
            )  # Copy this first here, for use below

            abrAndTonesMEEG.filter(
                l_freq=abr_lFreq, h_freq=abr_hFreq, phase=filterPhase
            )
            dataCov = mne.compute_raw_covariance(abrAndTonesMEEG)
            eb.save.pickle(
                dataCov, f"{megLocation}dataCovariance_abrAndTones_fs{abrFs}.pickle"
            )
            # abrResponse = abrAndTones.get_data(picks="eeg").squeeze().T
            # abrResponseMeg = abrAndTonesMeg.get_data(picks="meg").squeeze().T
            abrResponseMEEG = abrAndTonesMEEG.get_data().squeeze().T

            abrTimes, _ = eb.load.unpickle("regressorWork/clickTimes.pickle")
            abrTimes = (
                abrTimes - ANmodelDelay
            )  # Un-compensate for the AN model here for epoching and the mTRF with multiple ICI bins
            [stimFs, stim] = sp.io.wavfile.read("regressorWork/fullABRstim.wav")

            prevICIvalues = np.diff(abrTimes, prepend=0)
            prevICIvalues[0] = prevICIvalues[1]

            leadupFreqs = np.zeros(prevICIvalues.shape)

            for i in range(len(leadupFreqs)):
                leftInd = np.max((i - windowLen + 1 - nDelays, 0))
                leadupFreqs[i] = 1 / sp.stats.hmean(
                    prevICIvalues[leftInd : i + 1 - nDelays]
                )
                # leadupFreqs[i] = np.mean(prevICIvalues[leftInd:i+1])

            # binInds = np.digitize(modifiedICIs, 1/np.logspace(np.log10(7.5), np.log10(230), numBins))
            # edges = np.histogram_bin_edges(modifiedICIs, numBins)
            # binInds = np.digitize(modifiedICIs, edges, numBins)

            sortInds = np.argsort(
                leadupFreqs
            )  # These inds would sort the frequency array from low to high in Hertz
            sortedLeadupFreqs = leadupFreqs[sortInds]

            groups = np.array_split(
                sortInds, numBins
            )  # This will group those inds into numBins groups of equal size
            freqBins = np.median(
                np.array(np.array_split(sortedLeadupFreqs, numBins)), axis=1
            )

            print(f"Frequency bin medians are {freqBins}")

            abrRegressorMEEG = []
            for i in range(
                numBins
            ):  # Do this numBins + 1 times, and the last one will be no binning, just all the click onsets

                abrRegressorMEEG.append(
                    createPredictorTimeseries(
                        abrTimes[groups[i]],
                        abrFs,
                        int(round(len(stim) * abrFs / stimFs)),
                        values=values,
                    )
                )

            if useANmodel:

                abrRegressorMEEGComplete = eb.load.unpickle(
                    "regressorWork/clickANmodelMeg.pickle"
                ).squeeze()  # This file was generated at 5,000 Hz sampling rate

            else:

                abrRegressorMEEGComplete = createPredictorTimeseries(
                    abrTimes,
                    abrFs,
                    int(round(len(stim) * abrFs / stimFs)),
                    values=values,
                )

            abrRegressorMEEG = np.array(abrRegressorMEEG).T

            if gaussianSDabr is not None:
                abrRegressorMEEG = sp.ndimage.gaussian_filter1d(
                    abrRegressorMEEG,
                    gaussianSDabr * abrFs,
                    axis=0,
                )
                if not useANmodel:
                    abrRegressorMEEGComplete = sp.ndimage.gaussian_filter1d(
                        abrRegressorMEEGComplete,
                        gaussianSDabr * abrFs,
                        axis=0,
                    )

            # plt.figure()
            # plt.plot(abrRegressorMeg)

            # abrRegressor = createPredictorTimeseries(abrTimes, abrAndTones.info["sfreq"], int(round(len(stim) * abrAndTones.info["sfreq"] / stimFs)), values=values)

            print(
                f"""Length of cropped M/EEG response is {abrAndTonesMEEG.times[-1]}, and length of corresponding regressor at the same sampling rate is {abrRegressorMEEG.shape[0] / abrFs}"""
            )
            ratio = (abrRegressorMEEG.shape[0] / abrFs) / abrAndTonesMEEG.times[-1]
            print(f"""This is a ratio of {ratio}""")
            print(
                f"""Resampling M/EEG response to have length {abrRegressorMEEG.shape[0] / abrFs} seconds"""
            )

            abrResponseMEEG = sp.signal.resample(
                abrResponseMEEG, abrRegressorMEEG.shape[0]
            )

            # abrRegressor = sp.stats.zscore(abrRegressor)
            # abrResponse = sp.stats.zscore(abrResponse)

            if addNoiseBeforeFilter:
                normFactor = noiseRatio * np.std(abrRegressorMEEG, axis=0)
                abrRegressorMEEG = abrRegressorMEEG + normFactor * (
                    np.random.random(abrRegressorMEEG.shape) * 2 - 1
                )

                normFactor = noiseRatio * np.std(abrRegressorMEEGComplete, axis=0)
                abrRegressorMEEGComplete = abrRegressorMEEGComplete + normFactor * (
                    np.random.random(abrRegressorMEEGComplete.shape) * 2 - 1
                )

            if filterRegressors:
                thisRegressor = abrRegressorMEEG
                regressorInfo = mne.create_info(
                    ch_names=thisRegressor.shape[-1], sfreq=abrFs, ch_types="stim"
                )
                raw = mne.io.RawArray(data=thisRegressor.T, info=regressorInfo)
                raw.notch_filter(
                    np.arange(60, abrFs / 2, 60), picks="all", phase=filterPhase
                )
                raw.filter(
                    l_freq=abr_lFreq_forReg,
                    h_freq=abr_hFreq_forReg,
                    picks="all",
                    phase=filterPhase,
                )
                abrRegressorMEEG = raw.get_data().T

                thisRegressor = abrRegressorMEEGComplete[:, np.newaxis]
                regressorInfo = mne.create_info(
                    ch_names=thisRegressor.shape[-1], sfreq=abrFs, ch_types="stim"
                )
                raw = mne.io.RawArray(data=thisRegressor.T, info=regressorInfo)
                raw.notch_filter(
                    np.arange(60, abrFs / 2, 60), picks="all", phase=filterPhase
                )
                raw.filter(
                    l_freq=abr_lFreq_forReg,
                    h_freq=abr_hFreq_forReg,
                    picks="all",
                    phase=filterPhase,
                )
                abrRegressorMEEGComplete = raw.get_data().T.squeeze()

            if addNoiseAfterFilter:
                normFactor = noiseRatio * np.std(abrRegressorMEEG, axis=0)
                abrRegressorMEEG = abrRegressorMEEG + normFactor * (
                    np.random.random(abrRegressorMEEG.shape) * 2 - 1
                )

                normFactor = noiseRatio * np.std(abrRegressorMEEGComplete, axis=0)
                abrRegressorMEEGComplete = abrRegressorMEEGComplete + normFactor * (
                    np.random.random(abrRegressorMEEGComplete.shape) * 2 - 1
                )

            if divByStds:
                abrRegressorMEEGSTD = np.std(abrRegressorMEEG, axis=0)
                abrRegressorMEEGCompleteSTD = np.std(abrRegressorMEEGComplete, axis=0)
            else:
                abrRegressorMEEGSTD = 1
                abrRegressorMEEGCompleteSTD = 1

            if zScoreRegressors:
                abrRegressorMEEG = sp.stats.zscore(abrRegressorMEEG)
                abrRegressorMEEGComplete = sp.stats.zscore(abrRegressorMEEGComplete)

            rfAbrMEEG = mne.decoding.ReceptiveField(
                tminABR - edgePadFast,
                tmaxABR + edgePadFast,
                abrFs,
                estimator=lambdaReg,
                scoring=scoring,
            )

            rfAbrMEEGComplete = mne.decoding.ReceptiveField(
                tminABR - edgePadFast,
                tmaxABR + edgePadFast,
                abrFs,
                estimator=lambdaReg,
                scoring=scoring,
            )

            t0 = time.time()

            if len(abrRegressorMEEG.shape) == 1:
                abrRegressorMEEG = abrRegressorMEEG[:, np.newaxis]

            abrRegressorMEEGComplete = abrRegressorMEEGComplete[:, np.newaxis]

            if doMultTRFs:
                rfAbrMEEG.fit(abrRegressorMEEG, abrResponseMEEG)
                scoreAbrMEEG = rfAbrMEEG.score(abrRegressorMEEG, abrResponseMEEG)
                # cv = sklearn.model_selection.KFold(n_splits=n_jobs, shuffle=False)
                # scoreAbrMEEGAll = mne.decoding.cross_val_multiscore(rfAbrMEEG, abrRegressorMEEG, abrResponseMEEG, cv=cv, n_jobs=n_jobs)
                # scoreAbrMEEGCV = np.mean(scoreAbrMEEGAll, axis=0)

            rfAbrMEEGComplete.fit(abrRegressorMEEGComplete, abrResponseMEEG)
            scoreAbrMEEGComplete = rfAbrMEEGComplete.score(
                abrRegressorMEEGComplete, abrResponseMEEG
            )
            cv = sklearn.model_selection.KFold(n_splits=n_folds, shuffle=False)
            scoreAbrMEEGCompleteAll = mne.decoding.cross_val_multiscore(
                rfAbrMEEGComplete,
                abrRegressorMEEGComplete,
                abrResponseMEEG,
                cv=cv,
                n_jobs=n_jobs,
            )
            scoreAbrMEEGCompleteCV = np.mean(scoreAbrMEEGCompleteAll, axis=0)

            if doMultTRFs:
                TRFAbrMEEG = (1 / abrRegressorMEEGSTD) * np.moveaxis(
                    rfAbrMEEG.coef_[
                        :,
                        :,
                        int(edgePadFast * abrFs) : int(-edgePadFast * abrFs - 1),
                    ],
                    -1,
                    0,
                ).squeeze()

            TRFAbrMEEGComplete = (1 / abrRegressorMEEGCompleteSTD) * np.moveaxis(
                rfAbrMEEGComplete.coef_[
                    :,
                    :,
                    int(edgePadFast * abrFs) : int(-edgePadFast * abrFs - 1),
                ],
                -1,
                0,
            ).squeeze()

            print("\n")
            print(f"Took {time.time() - t0} seconds to deconvolve ABR")
            print("\n")
            abrTrfsMEEG = mne.EvokedArray(
                TRFAbrMEEGComplete.T, abrAndTonesMEEG.info, tmin=tminABR
            )

            abrAndTones = mne.io.RawArray(
                abrResponseMEEG.T, abrAndTonesMEEG.info
            )  # Here we overwrite the original raw array variable that contained only the EEG

            eventID = 130
            # origFsAbr = abrAndTones.info["sfreq"]
            # abrAndTones.resample(sfreq=origFsAbr / ratio)
            eventsABRs = np.zeros((len(abrTimes), 3)).astype(int)
            eventsABRs[:, 0] = np.round(abrTimes * abrFs).astype(int)
            eventsABRs[:, 2] = eventID

            event_dict_abrs = {"ABRs": eventID}
            reject_criteria = dict(
                #    mag=4000e-15,  # 4000 fT
                #   grad=4000e-13,  # 4000 fT/cm
                # eeg=250e-6,  # 150 µV  # Often this threshold seems best
                eeg=550e-5,  # 150 µV  # Often this threshold seems best
                #    eeg=550e-5,  # 150 µV
                #    eog=250e-6,
            )  # 250 µV
            epochsABR = mne.Epochs(
                # tonesRaw.copy().resample(sfreq=tonesRaw.info["sfreq"] * tonesResponse.shape[0] / tonesRegressor.shape[0]),
                abrAndTones,
                eventsABRs,
                event_id=event_dict_abrs,
                tmin=tminABR,  # These are ok for slower ERP
                tmax=tmaxABR,
                #    tmin=-0.05,  # Try these for FFRish ERP?
                #    tmax=.2,
                reject=reject_criteria,
                preload=True,
            )
            abrEpochs = epochsABR["ABRs"]
            abrEvokedMEEG = abrEpochs.average()

            tonesRawMEEG.filter(
                l_freq=tones_lFreq, h_freq=tones_hFreq, phase=filterPhase
            )
            dataCov = mne.compute_raw_covariance(tonesRawMEEG)
            eb.save.pickle(
                dataCov, f"{megLocation}dataCovariance_abrAndTones_fs{toneFs}.pickle"
            )
            tonesResponseMEEG = tonesRawMEEG.get_data().squeeze().T

            toneTimes = eb.load.unpickle("regressorWork/toneTimes.pickle")
            toneTimes = (
                toneTimes - ANmodelDelay
            )  # Un-compensate for the AN model here for epoching
            [stimFs, stim] = sp.io.wavfile.read("regressorWork/fullABRstim.wav")
            tonesRegressor = createPredictorTimeseries(
                toneTimes,
                toneFs,
                int(round(len(stim) * toneFs / stimFs)),
                values=values,
            )

            if gaussianSDtones is not None:
                tonesRegressor = sp.ndimage.gaussian_filter1d(
                    tonesRegressor, gaussianSDtones * toneFs
                )

            ratio = len(tonesRegressor) / toneFs / tonesRawMEEG.times[-1]

            print(
                f"""Length of cropped M/EEG tones response is {tonesRawMEEG.times[-1]}, and length of corresponding regressor at the same sampling rate is {len(tonesRegressor) / toneFs}"""
            )
            print(f"""This is a ratio of {ratio}""")
            print(
                f"""Resampling M/EEG response to have length {len(tonesRegressor) / toneFs} seconds"""
            )

            tonesResponseMEEG = sp.signal.resample(
                tonesResponseMEEG, len(tonesRegressor)
            )

            if addNoiseBeforeFilter:
                normFactor = noiseRatio * np.std(tonesRegressor, axis=0)
                tonesRegressor = tonesRegressor + normFactor * (
                    np.random.random(tonesRegressor.shape) * 2 - 1
                )

            if filterRegressors:
                thisRegressor = tonesRegressor[:, np.newaxis]
                regressorInfo = mne.create_info(
                    ch_names=thisRegressor.shape[-1], sfreq=toneFs, ch_types="stim"
                )
                raw = mne.io.RawArray(data=thisRegressor.T, info=regressorInfo)
                raw.notch_filter(
                    np.arange(60, toneFs / 2, 60), picks="all", phase=filterPhase
                )
                raw.filter(
                    l_freq=tones_lFreq_forReg,
                    h_freq=tones_hFreq_forReg,
                    picks="all",
                    phase=filterPhase,
                )
                tonesRegressor = raw.get_data().T.squeeze()

            if addNoiseAfterFilter:
                normFactor = noiseRatio * np.std(tonesRegressor, axis=0)
                tonesRegressor = tonesRegressor + normFactor * (
                    np.random.random(tonesRegressor.shape) * 2 - 1
                )

            if divByStds:
                tonesRegressorSTD = np.std(tonesRegressor, axis=0)
            else:
                tonesRegressorSTD = 1

            if zScoreRegressors:
                tonesRegressor = sp.stats.zscore(tonesRegressor)

            rfTonesMEEG = mne.decoding.ReceptiveField(
                tminTone - edgePadSlow,
                tmaxTone + edgePadSlow,
                toneFs,
                estimator=lambdaReg,
                scoring=scoring,
            )

            t0 = time.time()
            rfTonesMEEG.fit(tonesRegressor[:, np.newaxis], tonesResponseMEEG)
            scoreTonesMEEG = rfTonesMEEG.score(
                tonesRegressor[:, np.newaxis], tonesResponseMEEG
            )
            cv = sklearn.model_selection.KFold(n_splits=n_folds, shuffle=False)
            scoreTonesMEEGAll = mne.decoding.cross_val_multiscore(
                rfTonesMEEG,
                tonesRegressor[:, np.newaxis],
                tonesResponseMEEG,
                cv=cv,
                n_jobs=n_jobs,
            )
            scoreTonesMEEGCV = np.mean(scoreTonesMEEGAll, axis=0)

            TRFTonesMEEG = (1 / tonesRegressorSTD) * np.moveaxis(
                rfTonesMEEG.coef_[
                    :,
                    :,
                    int(edgePadSlow * toneFs) : int(-edgePadSlow * toneFs - 1),
                ],
                -1,
                0,
            ).squeeze()

            print("\n")
            print(f"Took {time.time() - t0} seconds to deconvolve tone ERP")
            print("\n")
            tonesTrfsMEEG = mne.EvokedArray(
                TRFTonesMEEG.T, tonesRawMEEG.info, tmin=tminTone
            )

            tonesRaw = mne.io.RawArray(
                tonesResponseMEEG.T, tonesRawMEEG.info
            )  # This creates a new raw object with the slightly resampled data and original info with both MEG and EEG

            eventID = 130
            # origFsTones = tonesRaw.info["sfreq"]
            # tonesRaw.resample(sfreq=origFsTones / ratio)
            eventsTones = np.zeros((len(toneTimes), 3)).astype(int)
            eventsTones[:, 0] = np.round(toneTimes * toneFs).astype(int)
            eventsTones[:, 2] = eventID

            event_dict_tones = {"Tones": eventID}
            reject_criteria = dict(
                #    mag=4000e-15,  # 4000 fT
                #   grad=4000e-13,  # 4000 fT/cm
                # eeg=250e-6,  # 150 µV  # Often this threshold seems best
                eeg=550e-5,  # 150 µV  # Often this threshold seems best
                #    eeg=550e-5,  # 150 µV
                #    eog=250e-6,
            )  # 250 µV
            epochs = mne.Epochs(
                # tonesRaw.copy().resample(sfreq=tonesRaw.info["sfreq"] * tonesResponse.shape[0] / tonesRegressor.shape[0]),
                tonesRaw,
                eventsTones,
                event_id=event_dict_tones,
                tmin=tminTone,  # These are ok for slower ERP
                tmax=tmaxTone,
                #    tmin=-0.05,  # Try these for FFRish ERP?
                #    tmax=.2,
                reject=reject_criteria,
                preload=True,
            )
            toneEpochs = epochs["Tones"]
            tonesEvokedMEEG = toneEpochs.average()

            abrTrfsMEEG.set_montage(mne.channels.make_standard_montage("biosemi32"))
            abrEvokedMEEG.set_montage(mne.channels.make_standard_montage("biosemi32"))
            tonesTrfsMEEG.set_montage(mne.channels.make_standard_montage("biosemi32"))
            tonesEvokedMEEG.set_montage(mne.channels.make_standard_montage("biosemi32"))

            if doMultTRFs:
                multAbrTrfsMEEG = []
                for i in range(numBins):
                    comment = str(round(freqBins[i], 1))
                    # comment = None
                    multAbrTrfsMEEG.append(
                        mne.EvokedArray(
                            TRFAbrMEEG[:, :, i].squeeze().T,
                            abrAndTonesMEEG.info,
                            tmin=tminABR,
                            comment=comment,
                        ).set_montage(mne.channels.make_standard_montage("biosemi32"))
                    )

            STDs = np.ones(scoreAbrMEEGComplete.shape)[
                np.newaxis, :
            ]  # Save a value for data STDs to be compatible with other scripts, but since this data is not z-scored in the first place, it should just be all ones
            if saveSummaryPickle:
                if doMultTRFs:
                    eb.save.pickle(
                        (
                            abrTrfsMEEG,
                            abrEvokedMEEG,
                            tonesTrfsMEEG,
                            tonesEvokedMEEG,
                            multAbrTrfsMEEG,
                        ),
                        f"{megLocation}abrAndTones_{filterPhase}Phase_lambda{lambdaRegInd}.pickle",
                    )
                    eb.save.pickle(
                        (None, [abrTrfsMEEG], scoreAbrMEEGCompleteCV, None, None, STDs),
                        f"{megLocation}abrAndTones_clicksComplete_{filterPhase}Phase_lambda{lambdaRegInd}.pickle",
                    )
                    eb.save.pickle(
                        (None, multAbrTrfsMEEG, scoreAbrMEEG, None, None, STDs),
                        f"{megLocation}abrAndTones_clicksMTRF_{filterPhase}Phase_lambda{lambdaRegInd}.pickle",
                    )
                    eb.save.pickle(
                        (None, [tonesTrfsMEEG], scoreTonesMEEGCV, None, None, STDs),
                        f"{megLocation}abrAndTones_tones_{filterPhase}Phase_lambda{lambdaRegInd}.pickle",
                    )
                else:
                    eb.save.pickle(
                        (abrTrfsMEEG, abrEvokedMEEG, tonesTrfsMEEG, tonesEvokedMEEG),
                        f"{megLocation}abrAndTones_{filterPhase}Phase_lambda{lambdaRegInd}.pickle",
                    )
                    eb.save.pickle(
                        (None, [abrTrfsMEEG], scoreAbrMEEGCompleteCV, None, None, STDs),
                        f"{megLocation}abrAndTones_clicksComplete_{filterPhase}Phase_lambda{lambdaRegInd}.pickle",
                    )
                    eb.save.pickle(
                        (None, [tonesTrfsMEEG], scoreTonesMEEGCV, None, None, STDs),
                        f"{megLocation}abrAndTones_tones_{filterPhase}Phase_lambda{lambdaRegInd}.pickle",
                    )

# %%
if plotting:

    ci = True

    abrTrfsMEEGAll = [[]]
    abrEvokedMEEGAll = [[]]
    tonesTrfsMEEGAll = [[]]
    tonesEvokedMEEGAll = [[]]

    abrTrfsMEEGAllGA = []
    abrEvokedMEEGAllGA = []
    tonesTrfsMEEGAllGA = []
    tonesEvokedMEEGAllGA = []

    if doMultTRFs:
        multAbrTrfsMEEGAll = [[] for _ in range(numBins)]

    for subject in subjects:

        megLocation = f"/Volumes/Seagate/map/{subject}/eegAndMeg/meg/"
        if doMultTRFs:
            (
                abrTrfsMEEG,
                abrEvokedMEEG,
                tonesTrfsMEEG,
                tonesEvokedMEEG,
                multAbrTrfsMEEG,
            ) = eb.load.unpickle(
                f"{megLocation}abrAndTones_{filterPhase}Phase_lambda{lambdaRegInd}.pickle"
            )
        else:
            abrTrfsMEEG, abrEvokedMEEG, tonesTrfsMEEG, tonesEvokedMEEG = (
                eb.load.unpickle(
                    f"{megLocation}abrAndTones_{filterPhase}Phase_lambda{lambdaRegInd}.pickle"
                )
            )

        # Here, subjects are all appended to just the first element of a list, so that plot_compare_evokeds will give mean and ci over subjects, but there is only one evoked to "compare" in each plot
        abrTrfsMEEGAll[0].append(abrTrfsMEEG)
        abrEvokedMEEGAll[0].append(abrEvokedMEEG)
        tonesTrfsMEEGAll[0].append(tonesTrfsMEEG)
        tonesEvokedMEEGAll[0].append(tonesEvokedMEEG)

        abrTrfsMEEGAllGA.append(abrTrfsMEEG)
        abrEvokedMEEGAllGA.append(abrEvokedMEEG)
        tonesTrfsMEEGAllGA.append(tonesTrfsMEEG)
        tonesEvokedMEEGAllGA.append(tonesEvokedMEEG)

        if doMultTRFs:
            for i in range(numBins):
                multAbrTrfsMEEGAll[i].append(multAbrTrfsMEEG[i])

    figs = []

    cmap = ("Hz", "plasma")

    figs.append(
        mne.viz.plot_compare_evokeds(
            abrTrfsMEEGAll,
            picks="eeg",
            axes="topo",
            title="EEG ABR-MLR with TRF deconvolution",
            ci=ci,
        )[0]
    )
    figs.append(
        mne.viz.plot_compare_evokeds(
            abrTrfsMEEGAll,
            picks="meg",
            axes="topo",
            title="MEG ABR-MLR with TRF deconvolution",
            ci=ci,
        )[0]
    )
    figs.append(
        mne.viz.plot_compare_evokeds(
            abrEvokedMEEGAll,
            picks="eeg",
            axes="topo",
            title="EEG ABR-MLR with epoch and average",
            ci=ci,
        )[0]
    )
    figs.append(
        mne.viz.plot_compare_evokeds(
            abrEvokedMEEGAll,
            picks="meg",
            axes="topo",
            title="MEG ABR-MLR with epoch and average",
            ci=ci,
        )[0]
    )
    figs.append(
        mne.viz.plot_compare_evokeds(
            tonesTrfsMEEGAll,
            picks="eeg",
            axes="topo",
            title="EEG tone N1-P2 with TRF deconvolution",
            ci=ci,
        )[0]
    )
    figs.append(
        mne.viz.plot_compare_evokeds(
            tonesTrfsMEEGAll,
            picks="meg",
            axes="topo",
            title="MEG tone N1-P2 with TRF deconvolution",
            ci=ci,
        )[0]
    )
    figs.append(
        mne.viz.plot_compare_evokeds(
            tonesEvokedMEEGAll,
            picks="eeg",
            axes="topo",
            title="EEG tone N1-P2 with epoch and average",
            ci=ci,
        )[0]
    )
    figs.append(
        mne.viz.plot_compare_evokeds(
            tonesEvokedMEEGAll,
            picks="meg",
            axes="topo",
            title="MEG tone N1-P2 with epoch and average",
            ci=ci,
        )[0]
    )
    if doMultTRFs:
        figs.append(
            mne.viz.plot_compare_evokeds(
                multAbrTrfsMEEGAll,
                picks="eeg",
                axes="topo",
                cmap=cmap,
                legend=True,
                ci=False,
                title="EEG ABRs with binned interclick intervals",
            )[0]
        )
        figs.append(
            mne.viz.plot_compare_evokeds(
                multAbrTrfsMEEGAll,
                picks="meg",
                axes="topo",
                cmap=cmap,
                legend=True,
                ci=False,
                title="MEG ABRs with binned interclick intervals",
            )[0]
        )

    # figReg, axReg = plt.subplots(figsize=[10, 6])
    # axReg.plot(abrRegressorComplete[1300:3000])
    # figReg.suptitle("Regressor sample")
    # figs.append(figReg)
    if saveFigs:
        for i in figs:
            i.savefig(f"{saveDir}{i._suptitle.get_text()}.pdf")

# %%
if plotting:

    abrTrfsMEEGGA = mne.grand_average(
        abrTrfsMEEGAllGA, interpolate_bads=False, drop_bads=True
    )
    abrEvokedMEEGGA = mne.grand_average(
        abrEvokedMEEGAllGA, interpolate_bads=False, drop_bads=True
    )
    tonesTrfsMEEGGA = mne.grand_average(
        tonesTrfsMEEGAllGA, interpolate_bads=False, drop_bads=True
    )
    tonesEvokedMEEGGA = mne.grand_average(
        tonesEvokedMEEGAllGA, interpolate_bads=False, drop_bads=True
    )

    windowMultiplier = 1

    # timesEeg = np.array([.001, .003, .007, .0125, .0185, .0275, .042, .055])
    # averageEeg = np.array([.0006, .001, .001, .001, .001, .002, .005, .005])
    # timesMeg = np.array([.001, .003, .0065, .0125, .0195, .0275, .0425, .055])
    # averageMeg = np.array([.0006, .001, .001, .001, .001, .002, .005, .005]) * windowMultiplier

    timesEeg = np.array([0.001, 0.003, 0.0083, 0.011, 0.015, 0.02, 0.0275, 0.05])
    averageEeg = np.array([0.0006, 0.001, 0.001, 0.001, 0.001, 0.002, 0.005, 0.005])
    timesMeg = np.array([0.001, 0.003, 0.0083, 0.011, 0.015, 0.02, 0.0275, 0.05])
    averageMeg = (
        np.array([0.0006, 0.001, 0.001, 0.001, 0.001, 0.002, 0.005, 0.005])
        * windowMultiplier
    )

    figEeg, axsEeg = plt.subplots(2, 4, figsize=[20, 10])
    for i in range(len(timesEeg)):
        abrTrfsMEEGGA.plot_topomap(
            times=timesEeg[i],
            average=averageEeg[i],
            ch_type="eeg",
            axes=axsEeg[i // axsEeg.shape[1], i % axsEeg.shape[1]],
            colorbar=False,
        )
    # evokedValues[0].plot_topomap(times=timesEeg, average=averageEeg, ch_type="eeg")
    figEeg.suptitle(f"EEG topographies")
    figEeg.savefig(f"{saveDir}{figEeg._suptitle.get_text()}.pdf")

    figMeg, axsMeg = plt.subplots(2, 4, figsize=[20, 10])
    for i in range(len(timesMeg)):
        abrTrfsMEEGGA.plot_topomap(
            times=timesMeg[i],
            average=averageMeg[i],
            ch_type="mag",
            axes=axsMeg[i // axsMeg.shape[1], i % axsMeg.shape[1]],
            colorbar=False,
        )
    # evokedValues[0].plot_topomap(times=timesMeg, average=averageMeg, ch_type="mag")
    figMeg.suptitle(f"MEG topographies")
    figMeg.savefig(f"{saveDir}{figMeg._suptitle.get_text()}.pdf")

# %%
# %connect_info
