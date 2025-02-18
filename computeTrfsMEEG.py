# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.0
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

from neuroAndSignalTools.freqAnalysis import *
from neuroAndSignalTools.deconvGeneralized import *

matplotlib.use("QtAgg")
plt.ion()

# %load_ext autoreload
# %autoreload 2


def computeTrfs(
    parentDir,
    subject,
    presStopCorrection,
    badChanList,
    condition,
    typeOfRegressor,
    nameOfRegressor,
    regressorDir,
    trialsToAnalyze=None,
    filenameSuffix="",
):

    # %%
    t0overall = time.time()

    # presStopCorrection = None

    # # subject='R2881';badChanList=['P3','Fp1','CP6']

    # # subject = "R3045";badChanList=None;condition="A";presStopCorrection=0
    # # subject='R3089';badChanList=['P7','CP6','C4','T7','CP5','P3','P4','O2','Oz','PO4'];condition='B';presStopCorrection=0
    # # subject='R3093';badChanList=['Oz','P8','CP6','Fp2'];condition='C';presStopCorrection=0
    # # subject='R3095';badChanList=['P7','T8','O2','PO4'];condition='D';presStopCorrection=0  # and possibly O2 and PO4

    # # subject='R3151';badChanList=['C3','FC5','P4'];condition='A'
    # # subject='R2774';badChanList=['Fp1','AF3','F7','F3','Fz','F4','FC6','C3','CP5','Pz','CP6','P8','FC1','FC5','T7','AF4'];condition='B'
    # # subject='R3152';badChanList=['T7','C3','P7','Pz','O1','P8','CP6'];condition='C'
    # subject = "R2877"
    # # badChanList = ["CP5", "P7", "F3", "FC5", "C3", "P4", "FC6", "FC1", "P8"]
    # badChanList = None
    # condition = "D"

    # # subject='R3157';badChanList=['CP6'];condition='A'  # Keep an eye on CP2 and possibly others
    # # subject='R3157';badChanList=None;condition='A'  # Keep an eye on CP2 and possibly others
    # # subject='R2783';badChanList=['T7','P7','CP5'];condition='B'
    # # subject='R2783';badChanList=None;condition='B'

    doPlotting = False
    doSaving = True

    doParallel = True
    n_jobs = 16
    # n_jobs = 8
    # n_jobs = -1
    parallelBackend = "loky"
    # parallelBackend = "sequential"
    # parallelBackend = "threading"
    # parallelBackend = "multiprocessing"

    if presStopCorrection is None:
        doPresentation = False
        doTriggy = True
    else:
        doPresentation = True
        doTriggy = False

    subDirs = "/eegAndMeg/eeg/"
    subDirsMeg = "/eegAndMeg/meg/"
    # subDirs='/ffrTests/'

    # runName='tones'
    # runName='stacks'
    runName = "maintask"
    # runName='triggytest-erp'
    # runName='triggytest-trf'

    # eventIDpres=130  # the usual
    # eventIDtrig=256  # the usual
    # eventID=32898  # R3089
    # eventID=49282  # R3093

    presStartCodes = np.array([142])
    presStopCodes = np.array([148, 404, 660, 916])
    trigStartCodes = np.array([256])
    trigStopCodes = np.array([512, 660, 768, 916])

    if presStopCorrection is None:
        presStopCorrection = (
            1 / 600
        )  # This is the extra time in seconds that the Presentation durations will end up having, because of the triggering click added to the actual wav files

    # eegLocation='/Users/karl/Dropbox/UMD/multilevel0/230908/'
    # eegLocation='/Users/karl/Dropbox/UMD/R2881/eegAndMeg/eeg/'
    # eegLocation='/Users/karl/map/R3045/eegAndMeg/eeg/'
    eegLocation = parentDir + subject + subDirs
    megLocation = parentDir + subject + subDirsMeg

    # bdfFilename='multilevel0_tones.bdf'
    # bdfFilename='R3045_tones.bdf'
    # bdfFilename='R3045_stacks.bdf'

    # bdfFilename=subject+'_'+runName+'.bdf'
    fifFilename = subject + "_" + runName + ".fif"
    fifFilenameMeg = subject + "_" + runName + "-raw.fif"

    MEG_bad_channels = ["MEG 056", "MEG 086"]

    # matFilename='tonesSnsTspcaOutput.mat'
    # matFilename='stacksSnsTspcaOutput.mat'
    matFilename = runName + "SnsTspcaOutput.mat"

    # regressorDir='/Users/karl/map/stimAndPredictors/mixes/predictors/'
    # regressorDir = "/Users/karl/map/stimAndPredictors/targets/predictors/"
    # regressorDir='/Users/karl/map/stimAndPredictors/distractors/predictors/'

    # typeOfRegressor='mix'
    # typeOfRegressor = "target"
    # typeOfRegressor='distractor'

    # nameOfRegressor = "_ANmodel_correctedLevels"
    # nameOfRegressor='~gammatone-1'
    # nameOfRegressor='~gammatone-on-1'

    # ANmodelNames=['longTrialNoTrigger_forReg_1.pickle','longTrialNoTrigger_forReg_2.pickle','longTrialNoTrigger_forReg_3.pickle',
    #              'longTrialNoTrigger_forReg_4.pickle','longTrialNoTrigger_forReg_5.pickle']

    # regressorNames=['A_mix1_ANmodel.pickle','A_mix1_ANmodel.pickle','A_mix2_ANmodel.pickle','A_mix2_ANmodel.pickle','A_mix3_ANmodel.pickle','A_mix3_ANmodel.pickle','A_mix4_ANmodel.pickle','A_mix4_ANmodel.pickle',
    #              'A_mix5_ANmodel.pickle','A_mix5_ANmodel.pickle','A_mix6_ANmodel.pickle','A_mix6_ANmodel.pickle','A_mix7_ANmodel.pickle','A_mix7_ANmodel.pickle','A_mix8_ANmodel.pickle','A_mix8_ANmodel.pickle',
    #              'A_mix9_ANmodel.pickle','A_mix9_ANmodel.pickle','A_mix10_ANmodel.pickle','A_mix10_ANmodel.pickle','A_mix11_ANmodel.pickle','A_mix11_ANmodel.pickle','A_mix12_ANmodel.pickle','A_mix12_ANmodel.pickle',
    #              'A_mix13_ANmodel.pickle','A_mix13_ANmodel.pickle','A_mix14_ANmodel.pickle','A_mix14_ANmodel.pickle','A_mix15_ANmodel.pickle','A_mix15_ANmodel.pickle','A_mix16_ANmodel.pickle','A_mix16_ANmodel.pickle'
    #              ]

    regressorNames = [
        f"{condition}_{typeOfRegressor}1{nameOfRegressor}.pickle",
        f"{condition}_{typeOfRegressor}1{nameOfRegressor}.pickle",
        f"{condition}_{typeOfRegressor}2{nameOfRegressor}.pickle",
        f"{condition}_{typeOfRegressor}2{nameOfRegressor}.pickle",
        f"{condition}_{typeOfRegressor}3{nameOfRegressor}.pickle",
        f"{condition}_{typeOfRegressor}3{nameOfRegressor}.pickle",
        f"{condition}_{typeOfRegressor}4{nameOfRegressor}.pickle",
        f"{condition}_{typeOfRegressor}4{nameOfRegressor}.pickle",
        f"{condition}_{typeOfRegressor}5{nameOfRegressor}.pickle",
        f"{condition}_{typeOfRegressor}5{nameOfRegressor}.pickle",
        f"{condition}_{typeOfRegressor}6{nameOfRegressor}.pickle",
        f"{condition}_{typeOfRegressor}6{nameOfRegressor}.pickle",
        f"{condition}_{typeOfRegressor}7{nameOfRegressor}.pickle",
        f"{condition}_{typeOfRegressor}7{nameOfRegressor}.pickle",
        f"{condition}_{typeOfRegressor}8{nameOfRegressor}.pickle",
        f"{condition}_{typeOfRegressor}8{nameOfRegressor}.pickle",
        f"{condition}_{typeOfRegressor}9{nameOfRegressor}.pickle",
        f"{condition}_{typeOfRegressor}9{nameOfRegressor}.pickle",
        f"{condition}_{typeOfRegressor}10{nameOfRegressor}.pickle",
        f"{condition}_{typeOfRegressor}10{nameOfRegressor}.pickle",
        f"{condition}_{typeOfRegressor}11{nameOfRegressor}.pickle",
        f"{condition}_{typeOfRegressor}11{nameOfRegressor}.pickle",
        f"{condition}_{typeOfRegressor}12{nameOfRegressor}.pickle",
        f"{condition}_{typeOfRegressor}12{nameOfRegressor}.pickle",
        f"{condition}_{typeOfRegressor}13{nameOfRegressor}.pickle",
        f"{condition}_{typeOfRegressor}13{nameOfRegressor}.pickle",
        f"{condition}_{typeOfRegressor}14{nameOfRegressor}.pickle",
        f"{condition}_{typeOfRegressor}14{nameOfRegressor}.pickle",
        f"{condition}_{typeOfRegressor}15{nameOfRegressor}.pickle",
        f"{condition}_{typeOfRegressor}15{nameOfRegressor}.pickle",
        f"{condition}_{typeOfRegressor}16{nameOfRegressor}.pickle",
        f"{condition}_{typeOfRegressor}16{nameOfRegressor}.pickle",
    ]

    debugCorrection = (
        0  # Positive or negative time duration in seconds to add to all epoch lengths
    )

    denoiseMatlab = False

    edgePad = 0.001  # Edge padding in seconds; compute the TRFs with this padding included, but don't plot/save/analyze those edges, because of (mostly time-domain) edge artifacts

    ##################
    # Make sure to change the regressor below between AN model and impulses if needed!
    ##################

    # Low and high cutoff frequencies for main MNE bandpass filter
    # These are ok for looking for the FFR

    if nameOfRegressor[0] == "_":
        l_freq = 20
        h_freq = 1000
    else:
        l_freq = 2
        h_freq = None

    # l_freq=1
    # h_freq=40

    exgLabels = ["EXG1", "EXG2", "EXG3", "EXG4", "EXG5", "EXG6", "EXG7", "EXG8"]

    if nameOfRegressor[0] == "_":
        # fs = 5000  # This is the fs to actually be used in analysis below. This should be the same fs as the predictors/regressors that are being read in
        fs = 16384  # This is the fs to actually be used in analysis below. This should be the same fs as the predictors/regressors that are being read in
    else:
        fs = 500  # This is the fs to actually be used in analysis below. This should be the same fs as the predictors/regressors that are being read in

    if nameOfRegressor[0] == "~":
        eps = 1e7
        # eps = 1e3
        # eps = 1e0
    else:
        eps = 0

    if nameOfRegressor[0] == "_":
        windowStart = (
            -0.01
        )  # Time to analyze previous to event onset, in seconds. Will be converted to sample time for deconvolution below
        windowEnd = 0.075

    # windowStart=-.01  # Time to analyze previous to event onset, in seconds. Will be converted to sample time for deconvolution below
    # windowEnd=.35
    else:
        windowStart = (
            -0.1
        )  # Time to analyze previous to event onset, in seconds. Will be converted to sample time for deconvolution below
        windowEnd = 0.75

    printAllEvents = False

    lenToAnalyze = 60  # Length of time of regressors and responses to extract and analyze through deconvolution, in seconds

    # soundDelay=.006  # Speed of sound delay in ear tubes, in seconds. Convert this to samples below and add it to all trigger times
    soundDelay = (
        3.1 / 343 - 0.00275
    )  # Speed of sound delay in ear tubes, in seconds, minus the AN model delay compensation (2.75 ms according to Shan et al.). Convert this to samples below and add it to all trigger times
    presentationDelay = 0.0017  # Length of time that Presentation triggers are early, relative to Triggy triggers, on average

    nChannels = 32 + 157

    # unitCoefficient=1e9  # MNE stores EEG in units of volts, so convert to nanovolts for clarity and also for use in faster predictors
    unitCoefficient = 1

    # skipDist=np.array([[2,5,10,13],
    #                    [2,5,8,11],
    #                    [2,5,10,15],
    #                    [2,5,7,12]])  # Skip these distractors, and probably targets as well, because these trials have no distractor

    # Above are 1-indexed, out of 16, for reference. The numbers we actually need (trials without distractors) are zero-indexed, out of 32 with trial doubling, below
    skipDist = np.array(
        [
            [2, 3, 8, 9, 18, 19, 24, 25],
            [2, 3, 8, 9, 14, 15, 20, 21],
            [2, 3, 8, 9, 18, 19, 28, 29],
            [2, 3, 8, 9, 12, 13, 22, 23],
        ]
    )  # Skip these distractors, and probably targets as well, because these trials have no distractor
    if condition == "A":
        conditionNum = 0
    elif condition == "B":
        conditionNum = 1
    elif condition == "C":
        conditionNum = 2
    elif condition == "D":
        conditionNum = 3

    if not denoiseMatlab:
        refs = ["EXG3", "EXG4"]
        notchFreqs = np.arange(60, 8192, 60)
        notchFreqsMeg = np.arange(60, 1000, 60)

    # %%
    t0 = time.time()
    justDidIt = False
    justDidItMeg = False

    if denoiseMatlab:

        if os.path.exists(eegLocation + "denoiseMatlab-raw.fif"):
            denoised = mne.io.read_raw_fif(
                eegLocation + "denoiseMatlab-raw.fif", preload=True
            )
        else:
            raw = mne.io.read_raw_fif(eegLocation + fifFilename, preload=True)
            print("\n")
            print("Loading preprocessed data matrix from mat file")
            preprocMat = mat73.loadmat(eegLocation + matFilename)["full_output"]
            t1 = time.time()
            print(f"Took {t1-t0} seconds to load")
            print("\n")
            # print(preprocMat.shape)
            # print(preprocMat[32,1000000])
            # plt.plot(preprocMat[31,0:7300])
            print("Copying just EXG channels to new, temporary matrix")
            exgs = raw.copy().pick_channels(exgLabels)[:][
                0
            ]  # The first colon makes this return a tuple of two things, the first one is the data matrix, and the second is the time vector
            print(f"EXGs object is type {type(exgs)}")
            print(f"EXGs object is shape {exgs.shape}")
            t2 = time.time()
            print(f"Took {t2-t1} seconds")
            print("\n")
            # [eeg channels, 8 exgs, trigger] -- this should match the raw info
            # don't use the 8 exgs anywhere; shape just needs to match the raw info
            print(
                "Concatenating preprocessing matrix and EXG channels into a new, temporary matrix"
            )
            data = np.concatenate(
                [preprocMat[:nChannels], exgs, preprocMat[nChannels : nChannels + 1]],
                axis=0,
            )
            t3 = time.time()
            print(f"Took {t3-t2} seconds")
            print(f"New temporary data matrix is shape {data.shape}")
            print("\n")
            print("Creating new raw object with new data array and original raw info")
            denoised = mne.io.RawArray(data, raw.info, raw.first_samp)
            t4 = time.time()
            print(f"Took {t4-t3} seconds")
            print("\n")
            print(
                "Deleting old raw object, temporary data matrix, temporary EXGs matrix, and loaded preprocessing matrix from mat file"
            )
            del raw, data, exgs, preprocMat
            t5 = time.time()
            print(f"Took {t5-t4} seconds")
            print("\n")
            print("Finally saving denoised raw object")
            denoised.save(eegLocation + "denoiseMatlab-raw.fif")
            print(f"Took {time.time()-t5} seconds")

    else:

        print(f"Set to NOT load tsPCA/SNS denoised dataset output from MATLAB")
        if not os.path.exists(eegLocation + "noDenoiseMatlab-raw.fif"):
            denoised = mne.io.read_raw_fif(eegLocation + fifFilename, preload=True)
            print("\n")
            print(f"Took {time.time()-t0} seconds to load raw fif")
            print("\n")
            print("Now referencing and notching")
            t1 = time.time()
            denoised.set_eeg_reference(ref_channels=refs)
            denoised.notch_filter(notchFreqs)
            print("\n")
            t2 = time.time()
            print(f"Took {t2-t1} seconds")
            print("\n")
            print(
                "Finally saving non-denoised, but referenced and notched, EEG raw fif file"
            )
            denoised.save(eegLocation + "noDenoiseMatlab-raw.fif")
            print(f"Took {time.time()-t2} seconds")

            justDidIt = True

        if not os.path.exists(megLocation + "noDenoiseMatlab-raw.fif"):
            # For now, just going to do the comparable same thing for the MEG
            denoisedMeg = mne.io.read_raw_fif(
                megLocation + fifFilenameMeg, preload=True
            )
            print("\n")
            print(f"Took {time.time()-t0} seconds to load raw fif")
            print("\n")
            print("Now notching")
            t1 = time.time()
            denoisedMeg.notch_filter(notchFreqsMeg)
            print("\n")
            t2 = time.time()
            print(f"Took {t2-t1} seconds")
            print("\n")
            print("Finally saving non-denoised, but notched, MEG raw fif file")
            denoisedMeg.save(megLocation + "noDenoiseMatlab-raw.fif")
            print(f"Took {time.time()-t2} seconds")

            justDidItMeg = True

    # %%
    if denoiseMatlab:
        if os.path.exists(eegLocation + f"denoiseMatlab-fs{fs}.pickle"):
            print(f"Unpickling saved resampled raw file and events matrix for fs {fs}")
            t0 = time.time()
            denoised, events = eb.load.unpickle(
                eegLocation + f"denoiseMatlab-fs{fs}.pickle"
            )
            print("\n")
            t1 = time.time()
            print(f"Took {t1-t0} seconds")
            print("\n")
        else:
            print(f"Resampling raw file and events matrix for fs {fs}")
            t0 = time.time()
            denoised, events = denoised.resample(sfreq=fs, events=events, verbose=True)
            t1 = time.time()
            print("\n")
            print(f"Took {t1-t0} seconds to calculate")
            print("\n")
            print("Now pickling and saving resampled raw file and events matrix")
            eb.save.pickle(
                (denoised, events), eegLocation + f"denoiseMatlab-fs{fs}.pickle"
            )
            t2 = time.time()
            print(f"Took {t2-t1} seconds")
            print("\n")
    else:
        if os.path.exists(eegLocation + f"noDenoiseMatlab-fs{fs}.pickle"):
            print(f"Unpickling saved resampled raw file and events matrix for fs {fs}")
            t0 = time.time()
            if fs != 16384:
                denoised, events = eb.load.unpickle(
                    eegLocation + f"noDenoiseMatlab-fs{fs}.pickle"
                )
            else:
                denoised = mne.io.read_raw_fif(
                    eegLocation + "noDenoiseMatlab-raw.fif", preload=True
                )
                events = eb.load.unpickle(
                    eegLocation + f"noDenoiseMatlab-fs{fs}.pickle"
                )
            print("\n")
            t1 = time.time()
            print(f"Took {t1-t0} seconds")
            print("\n")
        else:
            print(
                f"Loading raw file if necessary, and resampling raw data and events matrix for fs {fs}"
            )
            t0 = time.time()
            if not justDidIt:
                denoised = mne.io.read_raw_fif(
                    eegLocation + "noDenoiseMatlab-raw.fif", preload=True
                )
            events = mne.find_events(denoised, shortest_event=0)
            t1 = time.time()
            print("\n")
            print(
                f"Took {t1-t0} seconds to load raw fif files (if not already loaded) and calculate events matrix"
            )
            print("\n")
            print("Now resampling raw data and events matrix")
            if fs == denoised.info["sfreq"]:
                denoised = denoised.resample(sfreq=fs, events=events, verbose=True)
            else:
                denoised, events = denoised.resample(
                    sfreq=fs, events=events, verbose=True
                )
            t2 = time.time()
            print("\n")
            print(f"Took {t2-t1} seconds to calculate")
            print("\n")
            print("Now pickling and saving resampled raw file and events matrix")
            if fs != 16384:
                eb.save.pickle(
                    (denoised, events), eegLocation + f"noDenoiseMatlab-fs{fs}.pickle"
                )
            else:
                eb.save.pickle(events, eegLocation + f"noDenoiseMatlab-fs{fs}.pickle")
            t2 = time.time()
            print(f"Took {t2-t1} seconds")
            print("\n")

        if os.path.exists(megLocation + f"noDenoiseMatlab-fs{fs}.pickle"):
            print(f"Unpickling saved resampled raw file and events matrix for fs {fs}")
            t0 = time.time()
            if fs != 16384:
                denoisedMeg, eventsMeg = eb.load.unpickle(
                    megLocation + f"noDenoiseMatlab-fs{fs}.pickle"
                )
            else:
                denoised = mne.io.read_raw_fif(
                    eegLocation + "noDenoiseMatlab-raw.fif", preload=True
                )
                events = eb.load.unpickle(
                    eegLocation + f"noDenoiseMatlab-fs{fs}.pickle"
                )
            print("\n")
            t1 = time.time()
            print(f"Took {t1-t0} seconds")
            print("\n")
        else:
            print(
                f"Loading raw file if necessary, and resampling raw data and events matrix for fs {fs}"
            )
            t0 = time.time()
            if not justDidItMeg:
                denoisedMeg = mne.io.read_raw_fif(
                    megLocation + "noDenoiseMatlab-raw.fif", preload=True
                )
            eventsMeg = mne.find_events(denoisedMeg, shortest_event=0)
            t1 = time.time()
            print("\n")
            print(
                f"Took {t1-t0} seconds to load raw fif files (if not already loaded) and calculate events matrix"
            )
            print("\n")
            print("Now resampling raw data and events matrix")
            if fs == denoisedMeg.info["sfreq"]:
                denoisedMeg = denoisedMeg.resample(
                    sfreq=fs, events=eventsMeg, verbose=True
                )
            else:
                denoisedMeg, eventsMeg = denoisedMeg.resample(
                    sfreq=fs, events=eventsMeg, verbose=True
                )
            t2 = time.time()
            print("\n")
            print(f"Took {t2-t1} seconds to calculate")
            print("\n")
            print("Now pickling and saving resampled raw file and events matrix")
            if fs != 16384:
                eb.save.pickle(
                    (denoisedMeg, eventsMeg),
                    megLocation + f"noDenoiseMatlab-fs{fs}.pickle",
                )
            else:
                eb.save.pickle(events, eegLocation + f"noDenoiseMatlab-fs{fs}.pickle")
            t2 = time.time()
            print(f"Took {t2-t1} seconds")
            print("\n")

    # %%
    if printAllEvents:
        print(events)
        print("\n")

    events[:, 2] = events[:, 2] - events[:, 1]
    events[:, 0] = events[:, 0] + int(
        round(soundDelay * fs)
    )  # Add speed of sound delay to all events at the very beginning

    if printAllEvents:
        print(events)

    # count=0  # Start a Triggy event counter using zero indexing
    # for i in range(events.shape[0]):  # For each event,
    #     if events[i,2]==trigStartCode:  # first see if it is a Triggy event;
    #         if bool(count%2):  # if the counter is odd-numbered (it is the second of a pair),
    #             events[i,2]=trigStopCode  # then change its number to be a stop trigger for use later
    #         count+=1  # and increment the Triggy event counter either way

    # print('\n')
    # print(events)

    # %%
    # For these event tables to make sense, there needs to be exactly one stop code for every previous start code, and they need to be in the
    # same order. Don't know why they ever wouldn't be.

    # presStartEvents=events[events[:,2]==presStartCode,:]
    # presStopEvents =events[events[:,2]==presStopCode,:]
    # trigStartEvents=events[events[:,2]==trigStartCode,:]
    # trigStopEvents =events[events[:,2]==trigStopCode,:]

    presStartEvents = np.zeros(len(regressorNames))
    presStopEvents = np.zeros(len(regressorNames))
    trigStartEvents = np.zeros(len(regressorNames))
    trigStopEvents = np.zeros(len(regressorNames))

    presStartEventCodes = np.zeros(len(regressorNames))
    presStopEventCodes = np.zeros(len(regressorNames))
    trigStartEventCodes = np.zeros(len(regressorNames))
    trigStopEventCodes = np.zeros(len(regressorNames))

    presStartEventsMat = np.zeros((len(regressorNames), 3))
    presStopEventsMat = np.zeros((len(regressorNames), 3))
    trigStartEventsMat = np.zeros((len(regressorNames), 3))
    trigStopEventsMat = np.zeros((len(regressorNames), 3))

    # Get the start of the experiment for both EEG and MEG as the first event time. The first event marker has to be the actual first event start time for both datasets
    startSampleEEG = events[0, 0]
    startTimeEEG = startSampleEEG / fs
    stopTimeEEG = events[-1, 0] / fs
    startTimeMEG = eventsMeg[0, 0] / fs
    # Get the experiment duration from the EEG as the last minus the first event, and add 1 second just so we know we're not exactly on the ending sample
    totalExpDuration = stopTimeEEG - startTimeEEG + 1
    # And crop them both the same way. This should work because we have already established that there is next to no drift between the EEG and MEG recording computers
    # print(startTimeEEG)
    # print(stopTimeEEG)
    # print(startTimeMEG)
    # print(eventsMeg[-1, 0] / fs)
    # print(totalExpDuration)
    # print(events.shape)
    # print(np.diff(events, axis=0))
    # print(eventsMeg.shape)
    # print(np.diff(eventsMeg, axis=0))
    # denoised.crop(tmin=startTimeEEG, tmax=startTimeEEG + totalExpDuration)
    # denoisedMeg.crop(tmin=startTimeMEG, tmax=startTimeMEG + totalExpDuration)
    # # Subtract the first event sample time from all event sample times to start at 0 for the EEG events matrix, but don't bother with the MEG events matrix because we won't use it anymore now
    # events[:, 0] = events[:, 0] - startSampleEEG

    count = 0
    for i in range(events.shape[0]):
        if np.any(events[i, 2] == presStartCodes):
            events[i, 0] = events[i, 0] + round(
                presentationDelay * fs
            )  # Add Presentation delay to all Presentation start and stop events
            presStartEvents[count] = events[i, 0]
            presStartEventCodes[count] = events[i, 2]
            presStartEventsMat[count, :] = events[i, :]
            count = count + 1

    print(f"Found {count} events for Presentation starts:")
    if printAllEvents:
        print(
            np.concatenate(
                (presStartEvents[:, None], presStartEventCodes[:, None]), axis=1
            )
        )
    print("\n")

    count = 0
    for i in range(events.shape[0]):
        if np.any(events[i, 2] == presStopCodes):
            events[i, 0] = events[i, 0] + round(
                presentationDelay * fs
            )  # Add Presentation delay to all Presentation start and stop events
            events[i, 0] = events[i, 0] - round(
                presStopCorrection * fs
            )  # Here we subtract off the correction for the triggering click duration at the end of the wav files
            presStopEvents[count] = events[i, 0]
            presStopEventCodes[count] = events[i, 2]
            presStopEventsMat[count, :] = events[i, :]
            count = count + 1

    print(f"Found {count} events for Presentation stops:")
    if printAllEvents:
        print(
            np.concatenate(
                (presStopEvents[:, None], presStopEventCodes[:, None]), axis=1
            )
        )
    print("\n")

    count = 0
    for i in range(events.shape[0]):
        if np.any(events[i, 2] == trigStartCodes):
            if (
                not np.any(
                    events[i - 1, 2] == np.concatenate((presStopCodes, trigStopCodes))
                )
                or events[i - 1, 0] + fs > events[i, 0]
            ):
                if (
                    not np.any(
                        events[i + 1, 2]
                        == np.concatenate((presStopCodes, trigStopCodes))
                    )
                    or events[i + 1, 0] - fs > events[i, 0]
                ):
                    trigStartEvents[count] = events[i, 0]
                    trigStartEventCodes[count] = events[i, 2]
                    trigStartEventsMat[count, :] = events[i, :]
                    count = count + 1

    print(f"Found {count} events for Triggy starts:")
    if printAllEvents:
        print(
            np.concatenate(
                (trigStartEvents[:, None], trigStartEventCodes[:, None]), axis=1
            )
        )
    print("\n")

    count = 0
    for i in range(events.shape[0]):
        if np.any(events[i, 2] == trigStopCodes):
            trigStopEvents[count] = events[i, 0]
            trigStopEventCodes[count] = events[i, 2]
            trigStopEventsMat[count, :] = events[i, :]
            count = count + 1

    print(f"Found {count} events for Triggy stops:")
    if printAllEvents:
        print(
            np.concatenate(
                (trigStopEvents[:, None], trigStopEventCodes[:, None]), axis=1
            )
        )
    print("\n")

    # %%
    # easycap_montage = mne.channels.make_standard_montage("biosemi32")
    # listChTypes=denoised.get_channel_types()
    # #listChTypes[32:40]=['eog','eog','eog','eog','eog','eog','eog','eog']
    # print(listChTypes)
    # changeChTypes={'EXG1':'eog','EXG2':'eog','EXG3':'eog','EXG4':'eog','EXG5':'eog','EXG6':'eog','EXG7':'eog','EXG8':'eog'}
    # denoised.set_channel_types(changeChTypes)
    # denoised.set_montage(easycap_montage)

    # denoised.compute_psd(fmax=8192).plot(picks="data", exclude="bads")

    # %%

    # %%
    if l_freq is not None or h_freq is not None:
        # denoised.filter(l_freq=.1,h_freq=20)
        # denoised.filter(l_freq=10,h_freq=50)
        denoised.filter(l_freq=l_freq, h_freq=h_freq)
        denoisedMeg.filter(l_freq=l_freq, h_freq=h_freq)
        print("\n")
    print(denoised.info)
    print(denoisedMeg.info)

    # %%

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

    denoised.set_channel_types(changeChTypes)
    denoised.set_montage(digMontage)
    # denoised.set_montage(easycap_montage)

    if badChanList is not None:
        denoised.info["bads"].extend(badChanList)
        denoised.interpolate_bads()

    for MEG_bad_channel in MEG_bad_channels:
        denoisedMeg.info["bads"].append(MEG_bad_channel)

    # %%
    # event_dict = {"Presentation": eventIDpres, "Triggy": eventIDtrig}

    # %%
    # epochs = mne.Epochs(
    #    denoised,
    #    events,
    #    event_id=event_dict,

    #    tmin=-0.1,  # Try these for FFRish ERP?
    #    tmax=.45,

    #    preload=True
    # )

    # fs=denoised.info['sfreq']

    allPresEpochs = []
    allTrigEpochs = []
    allRegressors = []
    # epochLengths=np.array([10,300,300,300,300])  # This is in general NOT to be used, just to experiment with
    # invVarProps=[]

    force_update_info = True

    if doPresentation:

        for i in range(presStartEvents.shape[0]):
            tmax = presStopEvents[i] / fs - presStartEvents[i] / fs
            # tmax=epochLengths[i]
            epoch = mne.Epochs(
                denoised,
                presStartEventsMat[i : i + 1, :].astype(int),
                event_id=int(presStartCodes[0]),
                tmin=0,
                tmax=tmax + debugCorrection,
                baseline=None,
                picks="eeg",
                preload=True,
            )
            print("\n")

            # tmax = eventsMeg[i * 2 + 1, 0] / fs - eventsMeg[i * 2, 0] / fs
            # tmax=epochLengths[i]
            epochMEG = mne.Epochs(
                denoisedMeg,
                eventsMeg[i * 2 : i * 2 + 1, :].astype(int),
                event_id=int(eventsMeg[0, 2]),
                tmin=0,
                tmax=tmax + debugCorrection,
                baseline=None,
                picks="meg",
                preload=True,
            )
            print("\n")
            # epoch.info["dev_head_t"] = epochMEG.info["dev_head_t"]
            epochMEG.add_channels([epoch], force_update_info=force_update_info)
            epoch = epochMEG

            allPresEpochs.append(epoch)

        print("\n")

    if doTriggy:

        for i in range(trigStartEvents.shape[0]):
            tmax = trigStopEvents[i] / fs - trigStartEvents[i] / fs
            # tmax=epochLengths[i]
            epoch = mne.Epochs(
                denoised,
                trigStartEventsMat[i : i + 1, :].astype(int),
                event_id=int(trigStartCodes[0]),
                tmin=0,
                tmax=tmax + debugCorrection,
                baseline=None,
                picks="eeg",
                preload=True,
            )
            print("\n")

            # tmax = eventsMeg[i * 2 + 1, 0] / fs - eventsMeg[i * 2, 0] / fs
            # tmax=epochLengths[i]
            epochMEG = mne.Epochs(
                denoisedMeg,
                eventsMeg[i * 2 : i * 2 + 1, :].astype(int),
                event_id=int(eventsMeg[0, 2]),
                tmin=0,
                tmax=tmax + debugCorrection,
                baseline=None,
                picks="meg",
                preload=True,
            )
            print("\n")
            # epoch.info["dev_head_t"] = epochMEG.info["dev_head_t"]
            epochMEG.add_channels([epoch], force_update_info=force_update_info)
            epoch = epochMEG

            allTrigEpochs.append(epoch)

    if doPresentation and not doTriggy:
        for i in range(presStartEvents.shape[0]):
            # stimTimes=sp.io.loadmat(f'{regressorDir}longTrialStimTimes_{i+1}.mat')['currentStimTimes'].squeeze()
            # invVars=np.zeros(len(stimTimes))

            # print(f'Loaded {len(stimTimes)} stimulus delivery timepoints for epoch number {i}')
            # print(f'Calculating inverse of local variance around each stimulus timepoint for individual trial weight')
            # print('\n')

            # for j, stimTime in enumerate(stimTimes):
            #    thisVariance=epoch.copy().crop(tmin=stimTime-.05, tmax=stimTime+.3).get_data().squeeze().T.var(axis=0).mean()
            #    invVars[j]=1/thisVariance

            # allInvVars=invVars.sum()
            # invVarProp=invVars/allInvVars
            # epochLength=sp.io.loadmat(f'{regressorDir}longTrialStimTimes_{i+1}.mat')['longTrialLength']
            # regressor=createPredictorTimeseries(stimTimes, fs, int(epochLength*fs), values=invVarProp)
            # regressor=createPredictorTimeseries(stimTimes, fs, int(epochLength*fs))

            if fs == 16384:  # If fs is 5000 it is probably the AN model...
                if (
                    not np.any(skipDist[conditionNum] == i) or typeOfRegressor == "mix"
                ):  # If it's not a trial without a distractor, or if we're getting mixes anyway
                    regressor = eb.load.unpickle(f"{regressorDir}{regressorNames[i]}")
                else:  # otherwise save it as None, and just add NaNs to the TRF matrix below
                    regressor = None
            elif (
                fs == 500
            ):  # Else it is probably a regressor from Eelbrain, so it's an NDVar so we need .x
                if (
                    not np.any(skipDist[conditionNum] == i) or typeOfRegressor == "mix"
                ):  # If it's not a trial without a distractor, or if we're getting mixes anyway
                    regressor = eb.load.unpickle(f"{regressorDir}{regressorNames[i]}")
                    if isinstance(regressor, eb._data_obj.NDVar):
                        regressor = regressor.x
                        lenRegressor = len(regressor)
                        regressor = sp.signal.resample(
                            regressor, int(lenRegressor / 4)
                        )  # The Eelbrain regressors are made with fs=2000, so use a magic number for now and resample them for fs=500
                else:  # otherwise save it as None, and just add NaNs to the TRF matrix below
                    regressor = None  # and in that case, save it as None, and just add zeros to the TRF matrix below

            # invVarProps.append(invVarProp)
            allRegressors.append(regressor)
            # allANRegressors.append(ANregressor)

    else:
        for i in range(trigStartEvents.shape[0]):
            # stimTimes=sp.io.loadmat(f'{regressorDir}longTrialStimTimes_{i+1}.mat')['currentStimTimes'].squeeze()
            # invVars=np.zeros(len(stimTimes))

            # print(f'Loaded {len(stimTimes)} stimulus delivery timepoints for epoch number {i}')
            # print(f'Calculating inverse of local variance around each stimulus timepoint for individual trial weight')
            # print('\n')

            # for j, stimTime in enumerate(stimTimes):
            #    thisVariance=epoch.copy().crop(tmin=stimTime-.05, tmax=stimTime+.3).get_data().squeeze().T.var(axis=0).mean()
            #    invVars[j]=1/thisVariance

            # allInvVars=invVars.sum()
            # invVarProp=invVars/allInvVars
            # epochLength=sp.io.loadmat(f'{regressorDir}longTrialStimTimes_{i+1}.mat')['longTrialLength']
            # regressor=createPredictorTimeseries(stimTimes, fs, int(epochLength*fs), values=invVarProp)
            # regressor=createPredictorTimeseries(stimTimes, fs, int(epochLength*fs))

            if fs == 16384:  # If fs is 5000 it is probably the AN model...
                if (
                    not np.any(skipDist[conditionNum] == i) or typeOfRegressor == "mix"
                ):  # If it's not a trial without a distractor, or if we're getting mixes anyway
                    regressor = eb.load.unpickle(f"{regressorDir}{regressorNames[i]}")
                else:  # otherwise save it as None, and just add NaNs to the TRF matrix below
                    regressor = None
            elif (
                fs == 500
            ):  # Else it is probably a regressor from Eelbrain, so it's an NDVar so we need .x
                if (
                    not np.any(skipDist[conditionNum] == i) or typeOfRegressor == "mix"
                ):  # If it's not a trial without a distractor, or if we're getting mixes anyway
                    regressor = eb.load.unpickle(f"{regressorDir}{regressorNames[i]}")
                    if isinstance(regressor, eb._data_obj.NDVar):
                        regressor = regressor.x
                        lenRegressor = len(regressor)
                        regressor = sp.signal.resample(
                            regressor, int(lenRegressor / 4)
                        )  # The Eelbrain regressors are made with fs=2000, so use a magic number for now and resample them for fs=500
                else:  # otherwise save it as None, and just add NaNs to the TRF matrix below
                    regressor = None  # and in that case, save it as None, and just add zeros to the TRF matrix below

            # invVarProps.append(invVarProp)
            allRegressors.append(regressor)
            # allANRegressors.append(ANregressor)

    print("\n")
    print("\n")

    # for i in range(trigStartEvents.shape[0]):
    #    stimTimes=sp.io.loadmat(f'{regressorDir}longTrialStimTimes_{i+1}.mat')['currentStimTimes'].squeeze()
    #    invVars=np.zeros(len(stimTimes))
    #
    #    print(f'Loaded {len(stimTimes)} stimulus delivery timepoints for epoch number {i}')
    #    print(f'Calculating inverse of local variance around each stimulus timepoint for individual trial weight')
    #    print('\n')
    #
    #    for j, stimTime in enumerate(stimTimes):
    #        thisVariance=epoch.copy().crop(tmin=stimTime-.05, tmax=stimTime+.3).get_data().squeeze().T.var(axis=0).mean()
    #        invVars[j]=1/thisVariance
    #
    #    allInvVars=invVars.sum()
    #    invVarProp=invVars/allInvVars
    #    epochLength=sp.io.loadmat(f'{regressorDir}longTrialStimTimes_{i+1}.mat')['longTrialLength']
    #    regressor=createPredictorTimeseries(stimTimes, fs, int(epochLength*fs), values=invVarProp)
    #    invVarProps.append(invVarProp)
    #    allRegressors.append(regressor)

    # %%
    if doPresentation:
        print(allPresEpochs[5].get_data().shape[2] / fs)
    if doTriggy:
        print(allTrigEpochs[5].get_data().shape[2] / fs)
    print(allRegressors[5].shape[0] / fs)
    print(allRegressors[5].shape)

    # %%
    # epochsPres=epochs["Presentation"]
    # epochsTrig=epochs["Triggy"]

    # %%
    # print(allRegressors[1].shape)
    # plt.plot(allRegressors[1])

    # %%
    tDeconv = time.time()

    if trialsToAnalyze is None:
        trialsToAnalyze = np.arange(32)  # Zero indexed

    regressorsToDo = np.zeros((lenToAnalyze * fs, len(trialsToAnalyze)))

    if doPresentation:

        # TRFsFreqPres = np.zeros(
        #     [
        #         len(allPresEpochs),
        #         int(2 * (lenToAnalyze * fs - windowStart * fs)),
        #         nChannels,
        #     ]
        # )
        # TRFsTimePres = np.zeros(
        #     [len(allPresEpochs), int(windowEnd * fs - windowStart * fs), nChannels]
        # )

        # def doAllPresentationEpochs(
        #     i,
        #     allPresEpochs,
        #     allRegressors,
        #     regressorNames,
        #     fs,
        #     lenToAnalyze,
        #     eps,
        #     windowStart,
        #     windowEnd,
        #     edgePad,
        # ):

        presEpochsToDo = np.zeros((lenToAnalyze * fs, len(trialsToAnalyze), nChannels))

        for count, i in enumerate(trialsToAnalyze):

            epoch = allPresEpochs[i]

            # if allRegressors[i] is not None:
            #     regressor = allRegressors[i]
            # else:
            #     regressor = allRegressors[i - 2]

            regressor = allRegressors[i]

            currentEpochMat = (
                epoch.get_data().squeeze().T
            )  # Time is now the first dimension (dimension 0)

            print(
                f"Regressor {regressorNames[i]} is shape {regressor.shape}, which is {regressor.shape[0]/fs} seconds, while"
            )
            print(
                f"M/EEG response matrix {i} for Presentation triggers is shape {currentEpochMat.shape}, which corresponds to {currentEpochMat.shape[0]/fs} seconds;"
            )
            print(
                f"Now resampling M/EEG response matrix to be length {regressor.shape[0]}"
            )
            print("\n")
            response = sp.signal.resample(currentEpochMat, regressor.shape[0], axis=0)
            print(f"And then truncating both to be exactly length {lenToAnalyze*fs}")
            print("\n")
            # response=epoch.get_data().squeeze().T

            regressorsToDo[:, count] = regressor[: lenToAnalyze * fs]
            presEpochsToDo[:, count, :] = response[: lenToAnalyze * fs, :]

            # TRFfreq, TRFtime = deconvMain(
            #     regressor[: lenToAnalyze * fs],
            #     response[: lenToAnalyze * fs, :],
            #     eps,
            #     windowStart=(windowStart - edgePad) * fs,
            #     windowEnd=(windowEnd + edgePad) * fs,
            # )

            # if regressor.shape[0]>response.shape[0]:
            #    TRF=deconvMain(regressor[:response.shape[0]], response, eps, windowStart=windowStart*fs)
            # else:
            #    TRF=deconvMain(regressor, response[:regressor.shape[0]], eps, windowStart=windowStart*fs)
            # TRFsFreqTrig[i,:TRFfreq.shape[0],:]=TRFfreq
            # TRFsTimeTrig[i,:TRFtime.shape[0],:]=TRFtime

            # TRFfreq = TRFfreq[int(2 * edgePad * fs) :, :]
            # TRFtime = TRFtime[int(edgePad * fs) : int(-edgePad * fs - 1), :]
            # if allRegressors[i] is not None:
            #     return TRFfreq, TRFtime
            # else:
            #     return TRFfreq * np.nan, TRFtime * np.nan

        print("Now z-scoring regressor matrix...")
        regressorsToDo = sp.stats.zscore(regressorsToDo)
        print("\n")
        print("Done")
        print("\n")
        print("Now z-scoring response matrix...")
        presEpochsToDo = sp.stats.zscore(presEpochsToDo)
        print("\n")
        print("Done")
        print("\n")

        rfPres = mne.decoding.ReceptiveField(
            windowStart - edgePad, windowEnd + edgePad, fs, estimator=eps, n_jobs=n_jobs
        )
        rfPres.fit(regressorsToDo[:, :, np.newaxis], presEpochsToDo)
        TRFsTimePres = (
            rfPres.coef_[:, 0, int(edgePad * fs) : int(-edgePad * fs - 1)].squeeze().T
        )

        # if doParallel:

        #     results = joblib.Parallel(
        #         n_jobs=n_jobs, backend=parallelBackend, verbose=49
        #     )(
        #         joblib.delayed(doAllPresentationEpochs)(
        #             i,
        #             allPresEpochs,
        #             allRegressors,
        #             regressorNames,
        #             fs,
        #             lenToAnalyze,
        #             eps,
        #             windowStart,
        #             windowEnd,
        #             edgePad,
        #         )
        #         for i in range(len(allPresEpochs))
        #     )

        #     for i in range(len(results)):
        #         TRFsFreqPres[i, : results[i][0].shape[0], :] = results[i][0]
        #         TRFsTimePres[i, : results[i][1].shape[0], :] = results[i][1]

        #     del results

        # else:

        #     for i in range(len(allPresEpochs)):
        #         TRFfreq, TRFtime = doAllPresentationEpochs(
        #             i,
        #             allPresEpochs,
        #             allRegressors,
        #             regressorNames,
        #             fs,
        #             lenToAnalyze,
        #             eps,
        #             windowStart,
        #             windowEnd,
        #             edgePad,
        #         )
        #         TRFsFreqPres[i, : TRFfreq.shape[0], :] = TRFfreq
        #         TRFsTimePres[i, : TRFtime.shape[0], :] = TRFtime

    if doTriggy:

        # TRFsFreqTrig = np.zeros(
        #     [
        #         len(allTrigEpochs),
        #         int(2 * (lenToAnalyze * fs - windowStart * fs)),
        #         nChannels,
        #     ]
        # )
        # TRFsTimeTrig = np.zeros(
        #     [len(allTrigEpochs), int(windowEnd * fs - windowStart * fs), nChannels]
        # )

        # def doAllTriggyEpochs(
        #     i,
        #     allTrigEpochs,
        #     allRegressors,
        #     regressorNames,
        #     fs,
        #     lenToAnalyze,
        #     eps,
        #     windowStart,
        #     windowEnd,
        #     edgePad,
        # ):
        trigEpochsToDo = np.zeros((lenToAnalyze * fs, len(trialsToAnalyze), nChannels))

        for count, i in enumerate(trialsToAnalyze):

            epoch = allTrigEpochs[i]

            # if allRegressors[i] is not None:
            #     regressor = allRegressors[i]
            # else:
            #     regressor = allRegressors[i - 2]

            regressor = allRegressors[i]

            currentEpochMat = epoch.get_data().squeeze().T
            print(
                f"Regressor {regressorNames[i]} is shape {regressor.shape}, which is {regressor.shape[0]/fs} seconds, while"
            )
            print(
                f"M/EEG response matrix {i} for Triggy triggers is shape {currentEpochMat.shape}, which corresponds to {currentEpochMat.shape[0]/fs} seconds;"
            )
            print(
                f"Now resampling M/EEG response matrix to be length {regressor.shape[0]}"
            )
            print("\n")
            response = sp.signal.resample(currentEpochMat, regressor.shape[0], axis=0)
            print(f"And then truncating both to be exactly length {lenToAnalyze*fs}")
            print("\n")
            # response=epoch.get_data().squeeze().T

            regressorsToDo[:, count] = regressor[: lenToAnalyze * fs]
            trigEpochsToDo[:, count, :] = response[: lenToAnalyze * fs, :]

            # TRFfreq, TRFtime = deconvMain(
            #     regressor[: lenToAnalyze * fs],
            #     response[: lenToAnalyze * fs, :],
            #     eps,
            #     windowStart=(windowStart - edgePad) * fs,
            #     windowEnd=(windowEnd + edgePad) * fs,
            # )
            # if regressor.shape[0]>response.shape[0]:
            #    TRF=deconvMain(regressor[:response.shape[0]], response, eps, windowStart=windowStart*fs)
            # else:
            #    TRF=deconvMain(regressor, response[:regressor.shape[0]], eps, windowStart=windowStart*fs)
            # TRFsFreqTrig[i,:TRFfreq.shape[0],:]=TRFfreq
            # TRFsTimeTrig[i,:TRFtime.shape[0],:]=TRFtime
            # TRFfreq = TRFfreq[int(2 * edgePad * fs) :, :]
            # TRFtime = TRFtime[int(edgePad * fs) : int(-edgePad * fs - 1), :]
            # if allRegressors[i] is not None:
            #     return TRFfreq, TRFtime
            # else:
            #     return TRFfreq * np.nan, TRFtime * np.nan

        print("Now z-scoring regressor matrix...")
        regressorsToDo = sp.stats.zscore(regressorsToDo)
        print("\n")
        print("Done")
        print("\n")
        print("Now z-scoring response matrix...")
        trigEpochsToDo = sp.stats.zscore(trigEpochsToDo)
        print("\n")
        print("Done")
        print("\n")

        rfTrig = mne.decoding.ReceptiveField(
            windowStart - edgePad, windowEnd + edgePad, fs, estimator=eps, n_jobs=n_jobs
        )
        rfTrig.fit(regressorsToDo[:, :, np.newaxis], trigEpochsToDo)
        TRFsTimeTrig = (
            rfTrig.coef_[:, 0, int(edgePad * fs) : int(-edgePad * fs - 1)].squeeze().T
        )

        # if doParallel:

        #     results = joblib.Parallel(
        #         n_jobs=n_jobs, backend=parallelBackend, verbose=49
        #     )(
        #         joblib.delayed(doAllTriggyEpochs)(
        #             i,
        #             allTrigEpochs,
        #             allRegressors,
        #             regressorNames,
        #             fs,
        #             lenToAnalyze,
        #             eps,
        #             windowStart,
        #             windowEnd,
        #             edgePad,
        #         )
        #         for i in range(len(allTrigEpochs))
        #     )

        #     for i in range(len(results)):
        #         TRFsFreqTrig[i, : results[i][0].shape[0], :] = results[i][0]
        #         TRFsTimeTrig[i, : results[i][1].shape[0], :] = results[i][1]

        #     del results

        # else:

        #     for i in range(len(allTrigEpochs)):
        #         TRFfreq, TRFtime = doAllTriggyEpochs(
        #             i,
        #             allTrigEpochs,
        #             allRegressors,
        #             regressorNames,
        #             fs,
        #             lenToAnalyze,
        #             eps,
        #             windowStart,
        #             windowEnd,
        #             edgePad,
        #         )
        #         TRFsFreqTrig[i, : TRFfreq.shape[0], :] = TRFfreq
        #         TRFsTimeTrig[i, : TRFtime.shape[0], :] = TRFtime

    if doPresentation:

        # print(
        #     f"Shape of frequency-domain Presentation TRF matrix is now {TRFsFreqPres.shape}"
        # )
        # print("\n")
        print(
            f"Shape of time-domain Presentation TRF matrix is now {TRFsTimePres.shape}"
        )
        print("\n")

    if doTriggy:

        # print(
        #     f"Shape of frequency-domain Triggy TRF matrix is now {TRFsFreqTrig.shape}"
        # )
        # print("\n")
        print(f"Shape of time-domain Triggy TRF matrix is now {TRFsTimeTrig.shape}")
        print("\n")

    print(f"Deconvolution took {time.time()-tDeconv} seconds")

    # %%
    # cmap='viridis'
    # cmap='plasma'
    cmapName = "hsv"

    # cmap = matplotlib.colormaps.get_cmap(cmapName).resampled(TRFsPres.shape[2]).colors
    cmap = matplotlib.colormaps.get_cmap(cmapName).resampled(nChannels)
    colors = cmap(np.arange(0, cmap.N))

    duration = windowEnd - windowStart

    sampleStart = 0
    sampleEnd = int(fs * duration)

    figsize = [15, 9]

    # This is C
    # trialsToAnalyze=[2,3,6,7,10,11,14,15,18,19,22,23,26,27,30,31]  # Male target trials
    # trialsToAnalyze=[2,3,10,11,14,15,18,19,22,23,26,27]  # Male only trials
    # trialsToAnalyze=[0,1,4,5,8,9,12,13,16,17,20,21,24,25,28,29]  # Female target trials
    # trialsToAnalyze=[4,5,8,9,12,13,20,21,24,25,28,29]  # Female only trials

    # This is D
    # trialsToAnalyze=[0,1,4,5,8,9,12,13,16,17,20,21,24,25,28,29]  # Male target trials
    # trialsToAnalyze=[4,5,8,9,12,13,16,17,24,25,28,29]  # Male only trials
    # trialsToAnalyze=[2,3,6,7,10,11,14,15,18,19,22,23,26,27,30,31]  # Female target trials
    # trialsToAnalyze=[2,3,10,11,14,15,18,19,22,23,26,27]  # Female only trials

    if doPresentation:

        # TRFsFreqPresMean = np.nanmean(TRFsFreqPres[trialsToAnalyze, :, :], axis=0)
        # TRFsTimePresMean = np.nanmean(TRFsTimePres[trialsToAnalyze, :, :], axis=0)

        # freqPres_portion = TRFsFreqPresMean[sampleStart:sampleEnd, :] * unitCoefficient
        # timePres_portion = TRFsTimePresMean * unitCoefficient
        timePres_portion = TRFsTimePres * unitCoefficient

    if doTriggy:

        # TRFsFreqTrigMean = np.nanmean(TRFsFreqTrig[trialsToAnalyze, :, :], axis=0)
        # TRFsTimeTrigMean = np.nanmean(TRFsTimeTrig[trialsToAnalyze, :, :], axis=0)

        # freqTrig_portion = TRFsFreqTrigMean[sampleStart:sampleEnd, :] * unitCoefficient
        # timeTrig_portion = TRFsTimeTrigMean * unitCoefficient
        timeTrig_portion = TRFsTimeTrig * unitCoefficient

    # %%

    if doPlotting:

        if doPresentation:

            plt.figure(figsize=figsize)

            plt.gca().set_prop_cycle(plt.cycler("color", colors))
            # ws_portion = filtered_ws[int(args.fs_resamp*delay):int(args.fs_resamp*duration)]
            plt.plot(
                np.linspace(windowStart, windowEnd, freqPres_portion.shape[0]),
                freqPres_portion,
            )
            plt.title(
                "TRFs from regularized frequency domain deconvolution: Presentation",
                fontsize=17,
            )
            plt.xlabel("Time (s)", fontsize=17)
            plt.ylabel("Potential (nV)", fontsize=17)
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
            plt.legend(epoch.ch_names)
            plt.show()

            plt.figure(figsize=figsize)

            plt.gca().set_prop_cycle(plt.cycler("color", colors))
            # ws_portion = filtered_ws[int(args.fs_resamp*delay):int(args.fs_resamp*duration)]
            plt.plot(
                np.linspace(windowStart, windowEnd, timePres_portion.shape[0]),
                timePres_portion,
            )
            plt.title(
                "TRFs from regularized time domain deconvolution: Presentation",
                fontsize=17,
            )
            plt.xlabel("Time (s)", fontsize=17)
            plt.ylabel("Potential (nV)", fontsize=17)
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
            plt.legend(epoch.ch_names)
            plt.show()

        # %%
        if doTriggy:

            plt.figure(figsize=figsize)

            plt.gca().set_prop_cycle(plt.cycler("color", colors))
            # ws_portion = filtered_ws[int(args.fs_resamp*delay):int(args.fs_resamp*duration)]
            plt.plot(
                np.linspace(windowStart, windowEnd, freqTrig_portion.shape[0]),
                freqTrig_portion,
            )
            plt.title(
                "TRFs from regularized frequency domain deconvolution: Triggy",
                fontsize=17,
            )
            plt.xlabel("Time (s)", fontsize=17)
            plt.ylabel("Potential (nV)", fontsize=17)
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
            plt.legend(epoch.ch_names)
            plt.show()

            plt.figure(figsize=figsize)

            plt.gca().set_prop_cycle(plt.cycler("color", colors))
            # ws_portion = filtered_ws[int(args.fs_resamp*delay):int(args.fs_resamp*duration)]
            plt.plot(
                np.linspace(windowStart, windowEnd, timeTrig_portion.shape[0]),
                timeTrig_portion,
            )
            plt.title(
                "TRFs from regularized time domain deconvolution: Triggy", fontsize=17
            )
            plt.xlabel("Time (s)", fontsize=17)
            plt.ylabel("Potential (nV)", fontsize=17)
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
            plt.legend(epoch.ch_names)
            plt.show()

    # %%
    if doPresentation:

        # evokedFreqPres = mne.EvokedArray(
        #     freqPres_portion.T, epoch.info, tmin=windowStart
        # )
        evokedTimePres = mne.EvokedArray(
            timePres_portion.T, epoch.info, tmin=windowStart
        )

        if doSaving:

            evokedTimePres.save(
                f"{megLocation}recField{nameOfRegressor}_{typeOfRegressor}{filenameSuffix}-ave.fif",
                overwrite=True,
            )

            eb.save.pickle(
                rfPres,
                f"{megLocation}recField{nameOfRegressor}_{typeOfRegressor}{filenameSuffix}.pickle",
            )

    if doTriggy:

        # evokedFreqTrig = mne.EvokedArray(
        #     freqTrig_portion.T, epoch.info, tmin=windowStart
        # )
        evokedTimeTrig = mne.EvokedArray(
            timeTrig_portion.T, epoch.info, tmin=windowStart
        )

        if doSaving:

            evokedTimeTrig.save(
                f"{megLocation}recField{nameOfRegressor}_{typeOfRegressor}{filenameSuffix}-ave.fif",
                overwrite=True,
            )

            eb.save.pickle(
                rfTrig,
                f"{megLocation}recField{nameOfRegressor}_{typeOfRegressor}{filenameSuffix}.pickle",
            )

    # %%

    if doPlotting:

        if doPresentation:

            # plt.figure()
            evokedFreqPres.pick_types(eeg=True).plot_topo(color="r", legend=False)

            # plt.figure()
            evokedTimePres.pick_types(eeg=True).plot_topo(color="r", legend=False)

        # %%
        if doTriggy:

            # plt.figure()
            evokedFreqTrig.pick_types(eeg=True).plot_topo(color="r", legend=False)

            # plt.figure()
            evokedTimeTrig.pick_types(eeg=True).plot_topo(color="r", legend=False)

    # %%
    print(
        f"Everything took {time.time()-t0overall} seconds; deconvolution itself took {time.time()-tDeconv} seconds"
    )

    # gc.collect()

    # %%
    # subjectsToAverage=['R3089','R3095','R3151','R2774','R3152','R2877']
    # avgMat=np.zeros((426,nChannels,len(subjectsToAverage)))
    # for i, subject in enumerate(subjectsToAverage):
    #     avgMat[:,:,i]=eb.load.unpickle(f'/Users/karl/map/{subject}/eegAndMeg/eeg/abrMat.pickle')

    # evokedAvg=mne.EvokedArray(avgMat.mean(axis=2).T, epoch.info, tmin=windowStart)

    # evokedAvg.pick_types(eeg=True).plot_topo(color="r", legend=False)

    # %%
    # channelToAnalyze=31

    # #from localCode.freqAnalysis import *
    # #reload(specAndAutocorr)

    # #signalToView=stim[24000*5:24000*10]
    # #signalToView=wsMean[:10000,27]
    # #signalToView=wsMean[:10000,30]
    # #signalToView=wsMean[:10000,16]
    # #signalToView=stim[:24000]
    # #signalToView=audio[:6000]
    # signalToView=erpMatrixPres[channelToAnalyze,:]

    # lowestFreq=25
    # #lowestFreq=10

    # meanTimeBounds=[0,.3]

    # #fsForSpec=fs
    # fsForSpec=evokedPres.info['sfreq']

    # tLim=[evokedPres.times[0],evokedPres.times[-1]]

    # specAndAutocorr(signalToView, fs=fsForSpec, NFFT=8192, specFreqPortion=[0,50], autoFreqPortion=[0,100],
    #                 specWindowLength=500, windowStep=3, dynRangePortion=[50,100], autoWindowLength=int(fsForSpec/lowestFreq),
    #                 meanTimeBounds=meanTimeBounds, tLim=tLim)

    # plt.title('Presentation triggers')

    # #from localCode.freqAnalysis import *
    # #reload(specAndAutocorr)

    # #signalToView=stim[24000*5:24000*10]
    # #signalToView=wsMean[:10000,27]
    # #signalToView=wsMean[:10000,30]
    # #signalToView=wsMean[:10000,16]
    # #signalToView=stim[:24000]
    # #signalToView=audio[:6000]
    # signalToView=erpMatrixTrig[channelToAnalyze,:]

    # #fsForSpec=fs
    # fsForSpec=evokedTrig.info['sfreq']

    # tLim=[evokedTrig.times[0],evokedTrig.times[-1]]

    # specAndAutocorr(signalToView, fs=fsForSpec, NFFT=8192, specFreqPortion=[0,50], autoFreqPortion=[0,100],
    #                 specWindowLength=500, windowStep=3, dynRangePortion=[50,100], autoWindowLength=int(fsForSpec/lowestFreq),
    #                 meanTimeBounds=meanTimeBounds, tLim=tLim)

    # plt.title('Triggy triggers')

    # %%
    # plt.plot(invVars)

    # %%
    # plt.figure()
    # plt.plot(invVarProps[1])
    # plt.figure()
    # plt.plot(invVarProps[2])
    # plt.figure()
    # plt.plot(invVarProps[3])
    # plt.figure()
    # plt.plot(invVarProps[4])

    # %%
    # plt.plot(regressor)

    # %%


def computeSources(
    parentDir,
    subject,
    useAvgBrain,
    badChanList,
    typeOfRegressor,
    nameOfRegressor,
    bandpassFreqs,
    filenameSuffix,
):
    # # %%
    # useAvgBrain = False

    # # subject = "R3045";badChanList=None;condition="A";presStopCorrection=0
    # # subject='R3089';badChanList=['P7','CP6','C4','T7','CP5','P3','P4','O2','Oz','PO4'];condition='B';presStopCorrection=0
    # # subject='R3093';badChanList=['Oz','P8','CP6','Fp2'];condition='C';presStopCorrection=0
    # # subject='R3095';badChanList=['P7','T8','O2','PO4'];condition='D';presStopCorrection=0  # and possibly O2 and PO4

    # # subject='R3151';badChanList=['C3','FC5','P4'];condition='A';useAvgBrain=True
    # # subject='R2774';badChanList=['Fp1','AF3','F7','F3','Fz','F4','FC6','C3','CP5','Pz','CP6','P8','FC1','FC5','T7','AF4'];condition='B'
    # # subject='R3152';badChanList=['T7','C3','P7','Pz','O1','P8','CP6'];condition='C'
    # subject = "R2877"
    # # badChanList = ["CP5", "P7", "F3", "FC5", "C3", "P4", "FC6", "FC1", "P8"]
    # badChanList = None
    # condition = "D";useAvgBrain=True

    # # subject='R3157';badChanList=['CP6'];condition='A'  # Keep an eye on CP2 and possibly others
    # # subject='R3157';badChanList=None;condition='A'  # Keep an eye on CP2 and possibly others
    # # subject='R2783';badChanList=['T7','P7','CP5'];condition='B'
    # # subject='R2783';badChanList=None;condition='B'

    badChanListMEG = ["MEG 056", "MEG 086"]  # Hard code this here for now

    if useAvgBrain:
        mriSubject = "fsaverage"
    else:
        mriSubject = subject

    subjects_dir = os.path.expandvars("$SUBJECTS_DIR")

    subDirs = "/eegAndMeg/eeg/"
    subDirsMEG = "/eegAndMeg/meg/"

    eegLocation = parentDir + subject + subDirs
    megLocation = parentDir + subject + subDirsMEG

    doPlotting = False

    # nameOfRegressor = "_ANmodel_correctedLevels";initial_time = 0.0085
    # # nameOfRegressor='~gammatone-1';initial_time = 0.05
    # # nameOfRegressor='~gammatone-on-1';initial_time = 0.05

    # typeOfRegressor='mix'
    # # typeOfRegressor = "target"
    # # typeOfRegressor='distractor'

    temp = sorted(glob.glob(f"{eegLocation}*baseline*bdf"))
    eegBaselineFile = temp[-1]
    temp = sorted(
        glob.glob(f"{megLocation}*baseline*fif")
    )  # Read both of these the same way, even though there really should never be more than one of either named this way
    megBaselineFile = temp[-1]

    # These are bandpasses for the noise covariance matrix, and they matter! Probably should be the same as the bandpasses on the TRFs
    # # l_freq = 2
    # # h_freq = None
    # l_freq = 20
    # h_freq = 1000
    l_freq = bandpassFreqs[0]
    h_freq = bandpassFreqs[1]

    refs = ["EXG3", "EXG4"]
    # notchFreqs = np.arange(60, 8192, 60)
    notchFreqs = np.arange(60, 250, 60)

    # labels_vol = ["Left-Thalamus-Proper", "Right-Thalamus-Proper", "Brain-Stem"]
    labels_vol = None

    evoked = mne.read_evokeds(
        f"{megLocation}recField{nameOfRegressor}_{typeOfRegressor}{filenameSuffix}-ave.fif"
    )
    evoked = evoked[0]
    evoked.set_eeg_reference(projection=True)
    fs = evoked.info["sfreq"]

    # %%
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

    if badChanList is not None:
        eegBaselineRaw.info["bads"].extend(badChanList)
        # baselineRaw.interpolate_bads()

    if badChanListMEG is not None:
        megBaselineRaw.info["bads"].extend(badChanListMEG)
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
    baselineRaw.notch_filter(notchFreqs)

    # %%
    if l_freq is not None or h_freq is not None:
        # denoised.filter(l_freq=.1,h_freq=20)
        # denoised.filter(l_freq=10,h_freq=50)
        baselineRaw.filter(l_freq=l_freq, h_freq=h_freq)
        print("\n")

    # %%
    baselineCov = mne.compute_raw_covariance(baselineRaw, picks=["meg", "eeg"])

    # %%
    if os.path.exists(subjects_dir + "/" + mriSubject + "/bem/conductorModel.pickle"):
        bem = eb.load.unpickle(
            subjects_dir + "/" + mriSubject + "/bem/conductorModel.pickle"
        )
    else:
        surfaces = mne.make_bem_model(mriSubject)
        bem = mne.make_bem_solution(surfaces)
        eb.save.pickle(
            bem, subjects_dir + "/" + mriSubject + "/bem/conductorModel.pickle"
        )

    trans = eegLocation + "eeg-trans.fif"
    mriVol = subjects_dir + "/" + mriSubject + "/mri/aparc+aseg.mgz"

    # src_surf = mne.setup_source_space(
    #     mriSubject, spacing="oct5", add_dist=False, subjects_dir=subjects_dir
    # )

    src_vol = mne.setup_volume_source_space(
        mriSubject,
        mri=mriVol,
        pos=5.0,
        bem=bem,
        volume_label=labels_vol,
        subjects_dir=subjects_dir,
        add_interpolator=True,  # just for speed, usually this should be True
        verbose=True,
    )

    # # Generate the mixed source space
    # src_mix = src_surf + src_vol
    # print(
    #     f"The mixed source space contains {len(src_mix)} spaces and "
    #     f"{sum(s['nuse'] for s in src_mix)} vertices"
    # )

    # src.plot(subjects_dir=subjects_dir)

    # Setup volumn source space (This is the part how to create volSourceEstimate)
    # vol_src = mne.setup_volume_source_space(
    #     subject, mri=mri, pos=10.0, bem=bem,
    #     subjects_dir=subjects_dir,
    #     add_interpolator=True,
    #     verbose=True)

    # %%

    # fwd_surf = mne.make_forward_solution(
    #     evoked.info,
    #     trans,
    #     src_surf,
    #     bem,
    #     mindist=5.0,  # ignore sources<=5mm from innerskull
    #     meg=True,
    #     eeg=True,
    #     n_jobs=None,
    # )
    fwd_vol = mne.make_forward_solution(
        evoked.info,
        trans,
        src_vol,
        bem,
        mindist=5.0,  # ignore sources<=5mm from innerskull
        meg=True,
        eeg=True,
        n_jobs=None,
    )
    # fwd_mix = mne.make_forward_solution(
    #     evoked.info,
    #     trans,
    #     src_mix,
    #     bem,
    #     mindist=5.0,  # ignore sources<=5mm from innerskull
    #     meg=True,
    #     eeg=True,
    #     n_jobs=None,
    # )

    # del src  # save memory

    # leadfield = fwd_mix["sol"]["data"]
    leadfield = fwd_vol["sol"]["data"]
    print("Leadfield size : %d sensors x %d dipoles" % leadfield.shape)
    print(
        f"The fwd source space contains {len(fwd_vol['src'])} spaces and "
        f"{sum(s['nuse'] for s in fwd_vol['src'])} vertices"
    )

    # %%
    snr = 3.0  # use smaller SNR for raw data
    inv_method = "dSPM"  # sLORETA, MNE, dSPM
    parc = "aparc"  # the parcellation to use, e.g., 'aparc' 'aparc.a2009s'
    # loose_surf = dict(surface=0.2)
    loose_vol = dict(volume=1.0)
    # loose_mix = dict(surface=0.2, volume=1.0)
    depth = 0.8
    # depth = 10.0

    lambda2 = 1.0 / snr**2

    # inverse_operator_surf = mne.minimum_norm.make_inverse_operator(
    #     evoked.info, fwd_surf, baselineCov, depth=depth, loose=loose_surf, verbose=True
    # )

    inverse_operator_vol = mne.minimum_norm.make_inverse_operator(
        evoked.info, fwd_vol, baselineCov, depth=depth, loose=loose_vol, verbose=True
    )

    # inverse_operator_mix = mne.minimum_norm.make_inverse_operator(
    #     evoked.info, fwd_mix, baselineCov, depth=depth, loose=loose_mix, verbose=True
    # )

    # del fwd

    # stc = mne.minimum_norm.apply_inverse(
    #     evoked, inverse_operator_surf, lambda2, inv_method, pick_ori="normal"
    # )
    # stc = mne.minimum_norm.apply_inverse(
    #     evoked, inverse_operator, lambda2, inv_method, pick_ori=None
    # )

    # src = inverse_operator["src"]

    # %%
    stc_vec = mne.minimum_norm.apply_inverse(
        evoked, inverse_operator_vol, lambda2, inv_method, pick_ori="vector"
    )

    if doPlotting:

        # brain = stc_vec.plot(
        #     hemi="both",
        #     src=inverse_operator_mix["src"],
        #     views="coronal",
        #     initial_time=initial_time,
        #     subjects_dir=subjects_dir,
        #     brain_kwargs=dict(silhouette=True),
        #     smoothing_steps=7,
        #     show_traces=True,
        # )

        brain = stc_vec.plot(
            hemi="both",
            src=inverse_operator_vol["src"],
            views="coronal",
            initial_time=initial_time,
            subjects_dir=subjects_dir,
            brain_kwargs=dict(silhouette=True),
            smoothing_steps=7,
            show_traces=True,
        )

    # %%
    # brain2 = stc.surface().plot(
    #     initial_time=initial_time, subjects_dir=subjects_dir, smoothing_steps=7
    # )

    # %%
    # fig = stc.volume().plot(initial_time=initial_time, src=src, subjects_dir=subjects_dir)

    # %%

    # Plot electrode locations on scalp

    if doPlotting:

        fig = mne.viz.plot_alignment(
            evoked.info,
            trans,
            subject=mriSubject,
            dig=True,
            eeg=["original", "projected"],
            meg=[],
            coord_frame="head",
            subjects_dir=subjects_dir,
            surfaces=dict(brain=0.4, outer_skull=0.6, inner_skull=0.4, head=None),
        )

        # Set viewing angle
        mne.viz.set_3d_view(figure=fig, azimuth=135, elevation=80)

    # eb.save.pickle(
    #     (
    #         subject,
    #         mriSubject,
    #         labels_vol,
    #         src_surf,
    #         src_vol,
    #         src_mix,
    #         fwd_surf,
    #         fwd_vol,
    #         fwd_mix,
    #         inverse_operator_surf,
    #         inverse_operator_vol,
    #         inverse_operator_mix,
    #         stc,
    #         stc_vec,
    #     ),
    #     f"{eegLocation}sources{nameOfRegressor}_{typeOfRegressor}{filenameSuffix}.pickle",
    # )

    eb.save.pickle(
        (
            subject,
            mriSubject,
            labels_vol,
            src_vol,
            fwd_vol,
            inverse_operator_vol,
            stc_vec,
        ),
        f"{megLocation}sources{nameOfRegressor}_{typeOfRegressor}{filenameSuffix}.pickle",
    )
