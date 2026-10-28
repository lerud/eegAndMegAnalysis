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
import pathlib
import sys
import pickle
import glob
import time
import gc
import hashlib

from neuroAndSignalTools.freqAnalysis import *
from neuroAndSignalTools.deconvGeneralized import *
from neuroAndSignalTools.utilityFuncs import *

# matplotlib.use("QtAgg")
# plt.ion()

# %load_ext autoreload
# %autoreload 2

def safelyUnpickle(path):
    while os.path.exists(path + ".tmp"):
        timeToSleep = 5 + np.random.random() * 3
        print("\n")
        print(f"Waiting for {path + ".tmp"} to be deleted for {timeToSleep} seconds...")
        print("\n")
        time.sleep(timeToSleep)
    thing = eb.load.unpickle(path)
    return thing

def safelyPickle(thing, path):
    if not os.path.exists(path + ".tmp"):
        pathlib.Path.touch(path + ".tmp")
        eb.save.pickle(thing, path)
        if os.path.exists(path + ".tmp"):
            os.remove(path + ".tmp")
    else:
        while os.path.exists(path + ".tmp"):
            timeToSleep = 5 + np.random.random() * 3
            print("\n")
            print(f"Waiting for {path + ".tmp"} to be deleted for {timeToSleep} seconds...")
            print("\n")
            time.sleep(timeToSleep)

    

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
    filenamePrefix="concurRecField",
    filenameSuffix="",
    lambdaInd=0,
    thisLambda=0,
):
    # %%

    if trialsToAnalyze is None:
        trialsToAnalyze = np.arange(32)  # Zero indexed

    # addNoiseToRegressors = None
    addNoiseToRegressors = 2.5
        
    t0overall = time.time()

    doPlotting = False
    saveEvoked = False
    saveRf = True
    # filenamePrefix = "recField"
    # filenamePrefix = "concurRecField"

    doParallel = True
    # n_jobs = 16
    # n_jobs = 8
    # n_jobs = 2
    # n_jobs = None
    # n_jobs = -1
    parallelBackend = "loky"
    # parallelBackend = "sequential"
    # parallelBackend = "threading"
    # parallelBackend = "multiprocessing"
    verbose = 49

    scoring = "corrcoef"

    # filterPhase = "zero"
    filterPhase = "minimum"
    
    if presStopCorrection is None:
        doPresentation = False
        doTriggy = True
    else:
        doPresentation = True
        doTriggy = False

    if subject == "R3265" or subject == "R3275" or subject == "R3247" or subject == "R3304":
        doPresentation = True
        doTriggy = False

    eegTriggerRepair = None
    if subject == "R3265":
        eegTriggerRepair = np.array([5, 21, 22])

    subDirs = "/eegAndMeg/eeg/"
    subDirsMeg = "/eegAndMeg/meg/"
    subDirsHash = "/eegAndMeg/intermediate/"

    runName = "maintask"

    presStartCodes = np.array([142])
    presStopCodes = np.array([148, 404, 660, 916])
    trigStartCodes = np.array([256])
    trigStopCodes = np.array([512, 660, 768, 916])

    if presStopCorrection is None:
        presStopCorrection = (
            1 / 600
        )  # This is the extra time in seconds that the Presentation durations will end up having, because of the triggering click added to the actual wav files

    eegLocation = parentDir + subject + subDirs
    megLocation = parentDir + subject + subDirsMeg
    hashedLocation = parentDir + subject + subDirsHash

    fifFilename = subject + "_" + runName + ".fif"
    fifFilenameMeg = subject + "_" + runName + "-raw.fif"

    badChanListMEG = badChanList[1]
    badChanListEEG = badChanList[0]

    matFilename = runName + "SnsTspcaOutput.mat"

    regressorNames = []
    for i in range(1, 17):  # These filenames are 1-indexed and there are 16 of them
        # This way of organizing the variable assumes that nameOfRegressor is a list of regressor name strings, even if only one regressor name
        # Thus regressorNames will be a list of lists, even if each element of the regressorNames list is a list of just one string
        regressorNames.append(
            [
                f"{condition}_{typeOfRegressor}{i}{nameOfRegressor[j]}.pickle"
                for j in range(len(nameOfRegressor))
            ]
        )
        # and do this twice because all our conditions are doubled up consecutively
        regressorNames.append(
            [
                f"{condition}_{typeOfRegressor}{i}{nameOfRegressor[j]}.pickle"
                for j in range(len(nameOfRegressor))
            ]
        )

    debugCorrection = (
        0  # Positive or negative time duration in seconds to add to all epoch lengths
    )

    for i in range(len(nameOfRegressor)):  # Now that we have the correct filenames in the variable regressorNames, take the dash and noise instance number out of the nameOfRegressor variable (if they exist) so that we can still use these strings generally later
        if "ANmodel_fs5000" in nameOfRegressor[i]:
            nameOfRegressor[i] = "_ANmodel_fs5000"
        elif "ANmodel_maxFs" in nameOfRegressor[i]:
            nameOfRegressor[i] = "_ANmodel_maxFs"
    
    denoisePipeline = "noDenoiseMatlab"
    # denoisePipeline = "maxwellAndNotch"

    newMat = np.array([[1,     0,      -0.05, 0   ],
                       [0,     1,      0.325, 0   ],
                       [0.05,  -0.325, 1,     0.03],
                       [0,     0,      0,     1   ]])
    commonDestination = mne.transforms.Transform("meg", "head", newMat)

    # Low and high cutoff frequencies for main MNE bandpass filter
    print(nameOfRegressor)
    print(nameOfRegressor[0])
    print(nameOfRegressor[0][0])
    if nameOfRegressor[0] == "_ANmodel_maxFs":
        l_freq = 20
        h_freq = 1000
    elif nameOfRegressor[0] == "_ANmodel_fs5000":
        l_freq = 20
        h_freq = 300
    else:
        l_freq = 2
        h_freq = 40

    exgLabels = ["EXG1", "EXG2", "EXG3", "EXG4", "EXG5", "EXG6", "EXG7", "EXG8"]

    if nameOfRegressor[0] == "_ANmodel_maxFs":
        fs = 16384  # This is the fs to actually be used in analysis below. This should be the same fs as the predictors/regressors that are being read in
        eegOnly = True
    elif nameOfRegressor[0] == "_ANmodel_fs5000":
        fs = 5000  # This is the fs to actually be used in analysis below. This should be the same fs as the predictors/regressors that are being read in
        eegOnly = False
    else:
        fs = 500  # This is the fs to actually be used in analysis below. This should be the same fs as the predictors/regressors that are being read in
        eegOnly = False

    # if nameOfRegressor[0][0] == "~":
    #     # eps = 1e7
    #     eps = thisLambda
    #     # eps = 1e3
    #     # eps = 1e0
    # else:
    #     eps = 0

    eps = thisLambda

    if nameOfRegressor[0] == "_ANmodel_maxFs":
        windowStart = (
            -0.025
        )  # Time to analyze previous to event onset, in seconds. Will be converted to sample time for deconvolution below
        windowEnd = 0.075
        edgePad = 0.01  # Edge padding in seconds; compute the TRFs with this padding included, but don't plot/save/analyze those edges, because of (mostly time-domain) edge artifacts
    elif nameOfRegressor[0] == "_ANmodel_fs5000":
        windowStart = (
            -0.025
        )  # Time to analyze previous to event onset, in seconds. Will be converted to sample time for deconvolution below
        windowEnd = 0.075
        edgePad = 0.01  # Edge padding in seconds; compute the TRFs with this padding included, but don't plot/save/analyze those edges, because of (mostly time-domain) edge artifacts
    else:
        windowStart = (
            -0.1
        )  # Time to analyze previous to event onset, in seconds. Will be converted to sample time for deconvolution below
        windowEnd = 0.75
        edgePad = 0.1  # Edge padding in seconds; compute the TRFs with this padding included, but don't plot/save/analyze those edges, because of (mostly time-domain) edge artifacts

    print(l_freq)
    print(h_freq)
    print(fs)
    print(eps)
    print(windowStart)
    print(windowEnd)
    print("\n")

    printAllEvents = False

    lenToAnalyze = 60  # Length of time of regressors and responses to extract and analyze through deconvolution, in seconds

    soundDelay = (
        3.1 / 343 - 0.00275
    )  # Speed of sound delay in ear tubes, in seconds, minus the AN model delay compensation (2.75 ms according to Shan et al.). Convert this to samples below and add it to all trigger times
    presentationDelay = 0.0017  # Length of time that Presentation triggers are early, relative to Triggy triggers, on average

    if eegOnly:
        nChannels = 32
    else:
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

    refs = ["EXG3", "EXG4"]
    notchFreqs = np.arange(60, 8192, 60)
    notchFreqsMeg = np.arange(60, 1000, 60)

    # %%
    justDidIt = False
    justDidItMeg = False
    
    if doPresentation:  # This hash includes doTriggy for both on purpose; it will be False if using Presentation triggers, and True if using Triggy triggers
        presPipelineHashList = [subject.encode("utf-8"), denoisePipeline.encode("utf-8"), str(eegTriggerRepair).encode("utf-8"), str(doTriggy).encode("utf-8"), np.array(fs), str(l_freq).encode("utf-8"), str(h_freq).encode("utf-8")]
    
        if badChanListEEG is not None:
            presPipelineHashList.append("".join(badChanListEEG).encode("utf-8"))

        if not eegOnly:
            if badChanListMEG is not None:
                presPipelineHashList.append("".join(badChanListMEG).encode("utf-8"))


        presHasher = hashlib.blake2b(digest_size=20)
        [presHasher.update(i) for i in presPipelineHashList]

        print("\n")
        print("*************Debugging******************")
        print("\n")
        print("Presentation pipeline hash list is")
        print(presPipelineHashList)
        print("\n")
        print("*************Debugging******************")
        print("\n")
        
        presHash = presHasher.hexdigest()

    if doTriggy:
        trigPipelineHashList = [subject.encode("utf-8"), denoisePipeline.encode("utf-8"), str(eegTriggerRepair).encode("utf-8"), str(doTriggy).encode("utf-8"), np.array(fs), str(l_freq).encode("utf-8"), str(h_freq).encode("utf-8")]
    
        if badChanListEEG is not None:
            trigPipelineHashList.append("".join(badChanListEEG).encode("utf-8"))

        if not eegOnly:
            if badChanListMEG is not None:
                trigPipelineHashList.append("".join(badChanListMEG).encode("utf-8"))


        trigHasher = hashlib.blake2b(digest_size=20)
        [trigHasher.update(i) for i in trigPipelineHashList]

        print("\n")
        print("*************Debugging******************")
        print("\n")
        print("Triggy pipeline hash list is")
        print(trigPipelineHashList)
        print("\n")
        print("*************Debugging******************")
        print("\n")
        
        trigHash = trigHasher.hexdigest()

    passThrough = False


    print(f"Set to use denoising pipeline {denoisePipeline}")
    print("\n")

    thisPath = eegLocation + f"{denoisePipeline}-raw.fif"
    if not os.path.exists(thisPath) and not os.path.exists(thisPath + ".tmp"):
        pathlib.Path.touch(thisPath + ".tmp")

        t0 = time.time()
        print(f"Raw EEG fif files for denoising pipeline {denoisePipeline} do not exist, so opening original EEG fif files to create and save them, and keep MNE raw object in kernel")
        print("\n")
        denoised = mne.io.read_raw_fif(eegLocation + fifFilename, preload=True)

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

        
        if badChanListEEG is not None:
            denoised.info["bads"].extend(badChanListEEG)
            denoised.interpolate_bads(reset_bads=False)

        print("\n")
        print(f"Took {time.time()-t0} seconds to load raw fif and interpolate bad channels")
        print("\n")
        
        print("Now referencing and denoising")
        t1 = time.time()
        denoised.set_eeg_reference(ref_channels=refs)
        if denoisePipeline == "noDenoiseMatlab" or denoisePipeline == "maxwellAndNotch":
            denoised.notch_filter(notchFreqs, phase=filterPhase)
        print("\n")
        t2 = time.time()
        print(f"Took {t2-t1} seconds")
        print("\n")
        print(
            "Finally saving denoised EEG raw fif file"
        )
        denoised.save(eegLocation + f"{denoisePipeline}-raw.fif")
        print(f"Took {time.time()-t2} seconds")


        tNoise = time.time()
        print(f"Making temporary copy of EEG and filtering it for creating highpass noise covariance matrix")
        eegCopyForNoise = denoised.copy().filter(l_freq=1500, h_freq=None, phase=filterPhase)
        t1 = time.time()
        print(f"Took {t1-tNoise} seconds to copy and filter raw EEG")

        print(f"Now calculating and saving full sample rate EEG-only highpass noise covariance matrix")
        print("\n")
        eegNoiseCov = mne.compute_raw_covariance(eegCopyForNoise)
        eb.save.pickle(eegNoiseCov, eegLocation + "highpassNoiseCovariance.pickle")
        if eegOnly and eegTriggerRepair is None:
            print(f"and deleting raw EEG copy")
            del eegCopyForNoise
        else:
            print(f"and downsampling raw EEG copy for creating M/EEG covariance matrix")
            eegCopyForNoise.resample(sfreq=2000)  # Magic number 2000 here is hopefully the MEG sampling rate
        t2 = time.time()
        print(f"Took {t2-t1} seconds")
        print("\n")
            
        justDidIt = True

        os.remove(thisPath + ".tmp")

    elif os.path.exists(thisPath + ".tmp"):
        while os.path.exists(thisPath + ".tmp"):
            timeToSleep = 5 + np.random.random() * 3
            print("\n")
            print(f"Waiting for {thisPath + ".tmp"} to be deleted for {timeToSleep} seconds...")
            print("\n")
            time.sleep(timeToSleep)


    thisPath = megLocation + f"{denoisePipeline}-raw.fif"
    if (not eegOnly or eegTriggerRepair is not None) and not os.path.exists(thisPath) and not os.path.exists(thisPath + ".tmp"):
        pathlib.Path.touch(thisPath + ".tmp")

        t0 = time.time()
        print(f"Raw MEG fif files for denoising pipeline {denoisePipeline} do not exist, so opening original MEG fif files to create and save them, and keep MNE raw object in kernel")
        print("\n")
        # For now, just going to do the comparable same thing for the MEG
        denoisedMeg = mne.io.read_raw_fif(
            megLocation + fifFilenameMeg, preload=True
        )

        if badChanListMEG is not None:
            for MEG_bad_channel in badChanListMEG:
                denoisedMeg.info["bads"].append(MEG_bad_channel)
            denoisedMeg.interpolate_bads(reset_bads=False)
            
        print("\n")
        print(f"Took {time.time()-t0} seconds to load raw fif and interpolate bad channels")
        print("\n")
        t0 = time.time()
        if denoisePipeline == "maxwellAndNotch":
            print("Now implementing Maxwell filter")
            print("\n")
            denoisedMeg = mne.preprocessing.maxwell_filter(denoisedMeg, st_duration=10, st_correlation=.9, ignore_ref=True)
            print(f"Took {time.time()-t0} seconds")
            print("Now notching")
            t1 = time.time()
            print("\n")
            denoisedMeg.notch_filter(notchFreqsMeg, phase=filterPhase)
            print("\n")
            t2 = time.time()
            print(f"Took {t2-t1} seconds")
        if denoisePipeline == "noDenoiseMatlab":
            print("Now notching")
            t1 = time.time()
            print("\n")
            denoisedMeg.notch_filter(notchFreqsMeg, phase=filterPhase)
            print("\n")
            t2 = time.time()
            print(f"Took {t2-t1} seconds")
        print("\n")
        print("Finally saving denoised MEG raw fif file")
        denoisedMeg.save(megLocation + f"{denoisePipeline}-raw.fif")
        print(f"Took {time.time()-t2} seconds")


        tNoise = time.time()
        print(f"Making temporary copy of MEG and filtering it for creating highpass noise covariance matrix")
        megCopyForNoise = denoisedMeg.copy().filter(l_freq=350, h_freq=None, phase=filterPhase)
        t1 = time.time()
        print(f"Took {t1-tNoise} seconds to copy and filter raw MEG")
        print("Now combining EEG and MEG, creating noise covariance matrix, saving, and deleting copies")
        print("\n")

        if not justDidIt:  # If the EEG has been done previously, then we already have the EEG noise covariance, but we still have to load it
            print("Loading and resampling full-sampling-rate EEG, just for the purpose of merging with the MEG and making noise covariance object with both modalities")
            print("\n")
            eegCopyForNoise = mne.io.read_raw_fif(
                eegLocation + f"{denoisePipeline}-raw.fif", preload=True
            ).resample(sfreq=2000)
            eegNoiseCov = eb.load.unpickle(eegLocation + "highpassNoiseCovariance.pickle")

        
        lenEEG = eegCopyForNoise.times[-1]
        lenMEG = megCopyForNoise.times[-1]
        if lenEEG > lenMEG:
            eegCopyForNoise.crop(tmax=lenMEG)
        else:
            megCopyForNoise.crop(tmax=lenEEG)
        megCopyForNoise.add_channels([eegCopyForNoise], force_update_info=True)
        megNoiseCov = mne.compute_raw_covariance(megCopyForNoise)
        eegSize = eegNoiseCov.data.shape[0]
        megNoiseCov.data[-eegSize:, -eegSize:] = eegNoiseCov.data  # replace the EEG covariances with what we calculated above at the full sampling rate
        eb.save.pickle(megNoiseCov, megLocation + "highpassNoiseCovariance.pickle")
        del eegCopyForNoise
        del megCopyForNoise
        t2 = time.time()
        print(f"Took {t2-t1} seconds")
        
        justDidItMeg = True

        os.remove(thisPath + ".tmp")

    elif os.path.exists(thisPath + ".tmp"):
        while os.path.exists(thisPath + ".tmp"):
            timeToSleep = 5 + np.random.random() * 3
            print("\n")
            print(f"Waiting for {thisPath + ".tmp"} to be deleted for {timeToSleep} seconds...")
            print("\n")
            time.sleep(timeToSleep)

    # %%
    alreadyFiltered = False

    thisPath = eegLocation + f"{denoisePipeline}-fs{fs}.pickle"
    if os.path.exists(thisPath) or os.path.exists(thisPath + ".tmp"):
        while os.path.exists(thisPath + ".tmp"):
            timeToSleep = 5 + np.random.random() * 3
            print("\n")
            print(f"Waiting for {thisPath + ".tmp"} to be deleted for {timeToSleep} seconds...")
            print("\n")
            time.sleep(timeToSleep)
        t0 = time.time()
        if fs != 16384:
            thisPath = eegLocation + f"{denoisePipeline}-fs{fs}-filt{l_freq}And{h_freq}.pickle"
            if os.path.exists(thisPath) or os.path.exists(thisPath + ".tmp"):
                while os.path.exists(thisPath + ".tmp"):
                    timeToSleep = 5 + np.random.random() * 3
                    print("\n")
                    print(f"Waiting for {thisPath + ".tmp"} to be deleted for {timeToSleep} seconds...")
                    print("\n")
                    time.sleep(timeToSleep)
                if doPresentation:
                    if not os.path.exists(hashedLocation + presHash + ".pickle"):
                        print(f"Unpickling saved resampled and filtered EEG raw file and events matrix for fs {fs}, l_freq {l_freq}, and h_freq {h_freq}")
                        denoised, events = eb.load.unpickle(thisPath)
                        t1 = time.time()
                        print(f"Took {t1-t0} seconds to load")
                        print("\n")
                    else:
                        print(f"M/EEG epochs object exists for fs {fs}, l_freq {l_freq}, and h_freq {h_freq}, so unpickling it")
                        passThrough = True
                if doTriggy:
                    if not os.path.exists(hashedLocation + trigHash + ".pickle"):
                        print(f"Unpickling saved resampled and filtered EEG raw file and events matrix for fs {fs}, l_freq {l_freq}, and h_freq {h_freq}")
                        denoised, events = eb.load.unpickle(thisPath)
                        t1 = time.time()
                        print(f"Took {t1-t0} seconds to load")
                        print("\n")
                    else:
                        print(f"M/EEG epochs object exists for fs {fs}, l_freq {l_freq}, and h_freq {h_freq}, so unpickling it")
                        passThrough = True
            else:
                pathlib.Path.touch(thisPath + ".tmp")
                print(f"Unpickling saved resampled EEG raw file and events matrix for fs {fs}")
                denoised, events = eb.load.unpickle(eegLocation + f"{denoisePipeline}-fs{fs}.pickle")
                t1 = time.time()
                print(f"Took {t1-t0} seconds to load; now filtering for l_freq {l_freq} and h_freq {h_freq} and saving")
                print("\n")
                if l_freq is not None or h_freq is not None:
                    denoised.filter(l_freq=l_freq, h_freq=h_freq, phase=filterPhase)
                    eb.save.pickle((denoised, events), thisPath)
                t2 = time.time()
                print(f"Took {t2-t1} seconds to filter and save")
                print("\n")
                os.remove(thisPath + ".tmp")
        else:
            thisPath = eegLocation + f"{denoisePipeline}-raw-filt{l_freq}And{h_freq}.fif"
            if os.path.exists(thisPath) or os.path.exists(thisPath + ".tmp"):
                while os.path.exists(thisPath + ".tmp"):
                    timeToSleep = 5 + np.random.random() * 3
                    print("\n")
                    print(f"Waiting for {thisPath + ".tmp"} to be deleted for {timeToSleep} seconds...")
                    print("\n")
                    time.sleep(timeToSleep)
                if doPresentation:
                    if not os.path.exists(hashedLocation + presHash + ".pickle"):
                        print(f"Unpickling saved filtered EEG raw file and events matrix for fs {fs}, l_freq {l_freq}, and h_freq {h_freq}")
                        denoised = mne.io.read_raw_fif(thisPath, preload=True)
                    else:
                        print(f"M/EEG epochs object exists for fs {fs}, l_freq {l_freq}, and h_freq {h_freq}, so unpickling it")
                        passThrough = True
                if doTriggy:
                    if not os.path.exists(hashedLocation + trigHash + ".pickle"):
                        print(f"Unpickling saved filtered EEG raw file and events matrix for fs {fs}, l_freq {l_freq}, and h_freq {h_freq}")
                        denoised = mne.io.read_raw_fif(thisPath, preload=True)
                    else:
                        print(f"M/EEG epochs object exists for fs {fs}, l_freq {l_freq}, and h_freq {h_freq}, so unpickling it")
                        passThrough = True
            else:
                pathlib.Path.touch(thisPath + ".tmp")
                t0 = time.time()
                print(f"Loading EEG raw fif file for denoising pipeline {denoisePipeline}")
                denoised = mne.io.read_raw_fif(eegLocation + f"{denoisePipeline}-raw.fif", preload=True)
                t1 = time.time()
                print(f"Took {t1-t0} seconds to load; now filtering for l_freq {l_freq} and h_freq {h_freq} and saving")
                denoised.filter(l_freq=l_freq, h_freq=h_freq, phase=filterPhase)
                denoised.save(thisPath)
                t2 = time.time()
                print(f"Took {t2-t1} seconds to filter and save")
                print("\n")
                os.remove(thisPath + ".tmp")
            print(f"Loading events matrix for fs {fs}")
            events = eb.load.unpickle(eegLocation + f"{denoisePipeline}-fs{fs}.pickle")
        print("\n")
        alreadyFiltered = True
        
    else:
        
        pathlib.Path.touch(thisPath + ".tmp")
        
        t0 = time.time()
        if not justDidIt:
            print(f"Loading denoised EEG raw file, finding all events, resampling raw data and events matrix for fs {fs}, and saving pickle")
            denoised = mne.io.read_raw_fif(
                eegLocation + f"{denoisePipeline}-raw.fif", preload=True
            )
        else:
            print(f"Finding all events, resampling raw data and events matrix for fs {fs}, and saving pickle")
        events = mne.find_events(denoised, shortest_event=0)
        t1 = time.time()
        print("\n")
        print(
            f"Took {t1-t0} seconds to load raw fif files (if not already loaded in kernel) and calculate events matrix"
        )
        print("\n")
        print("Now resampling raw data and events matrix")
        if fs == denoised.info["sfreq"]:
            denoised, _ = denoised.resample(sfreq=fs, events=events, verbose=True)
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
                (denoised, events), eegLocation + f"{denoisePipeline}-fs{fs}.pickle"
            )
        else:
            eb.save.pickle(events, eegLocation + f"{denoisePipeline}-fs{fs}.pickle")
        t2 = time.time()
        print(f"Took {t2-t1} seconds")
        print("\n")
        os.remove(thisPath + ".tmp")

    if not eegOnly or eegTriggerRepair is not None:

        thisPath = megLocation + f"{denoisePipeline}-fs{fs}.pickle"
        if os.path.exists(thisPath) or os.path.exists(thisPath + ".tmp"):
            while os.path.exists(thisPath + ".tmp"):
                timeToSleep = 5 + np.random.random() * 3
                print("\n")
                print(f"Waiting for {thisPath + ".tmp"} to be deleted for {timeToSleep} seconds...")
                print("\n")
                time.sleep(timeToSleep)
            t0 = time.time()
            thisPath = megLocation + f"{denoisePipeline}-fs{fs}-filt{l_freq}And{h_freq}.pickle"
            if os.path.exists(thisPath) or os.path.exists(thisPath + ".tmp"):
                while os.path.exists(thisPath + ".tmp"):
                    timeToSleep = 5 + np.random.random() * 3
                    print("\n")
                    print(f"Waiting for {thisPath + ".tmp"} to be deleted for {timeToSleep} seconds...")
                    print("\n")
                    time.sleep(timeToSleep)
                if doPresentation:
                    if not os.path.exists(hashedLocation + presHash + ".pickle"):
                        print(f"Unpickling saved resampled and filtered MEG raw file and events matrix for fs {fs}, l_freq {l_freq}, and h_freq {h_freq}")
                        denoisedMeg, eventsMeg = eb.load.unpickle(thisPath)
                        t1 = time.time()
                        print(f"Took {t1-t0} seconds to load")
                        print("\n")
                    else:
                        print(f"M/EEG epochs object exists for fs {fs}, l_freq {l_freq}, and h_freq {h_freq}, so unpickling it")
                        passThrough = True
                if doTriggy:
                    if not os.path.exists(hashedLocation + trigHash + ".pickle"):
                        print(f"Unpickling saved resampled and filtered MEG raw file and events matrix for fs {fs}, l_freq {l_freq}, and h_freq {h_freq}")
                        denoisedMeg, eventsMeg = eb.load.unpickle(thisPath)
                        t1 = time.time()
                        print(f"Took {t1-t0} seconds to load")
                        print("\n")
                    else:
                        print(f"M/EEG epochs object exists for fs {fs}, l_freq {l_freq}, and h_freq {h_freq}, so unpickling it")
                        passThrough = True
            else:
                pathlib.Path.touch(thisPath + ".tmp")
                print(f"Unpickling saved resampled MEG raw file and events matrix for fs {fs}")
                denoisedMeg, eventsMeg = eb.load.unpickle(megLocation + f"{denoisePipeline}-fs{fs}.pickle")
                t1 = time.time()
                print(f"Took {t1-t0} seconds to load; now filtering for l_freq {l_freq} and h_freq {h_freq} and saving pickle")
                if l_freq is not None or h_freq is not None:
                    denoisedMeg.filter(l_freq=l_freq, h_freq=h_freq, phase=filterPhase)
                    eb.save.pickle((denoisedMeg, eventsMeg), thisPath)
                t2 = time.time()
                print(f"Took {t2-t1} seconds to filter and save")
                os.remove(thisPath + ".tmp")
            alreadyFiltered = True
            
        else:
            
            pathlib.Path.touch(thisPath + ".tmp")

            t0 = time.time()
            if not justDidItMeg:
                print(f"Loading denoised MEG raw file, finding all events, resampling raw data and events matrix for fs {fs}, and saving pickle")
                denoisedMeg = mne.io.read_raw_fif(megLocation + f"{denoisePipeline}-raw.fif", preload=True)
            else:
                print(f"Finding all events, resampling raw data and events matrix for fs {fs}, and saving pickle")
            eventsMeg = mne.find_events(denoisedMeg, shortest_event=0)
            t1 = time.time()
            print("\n")
            print(
                f"Took {t1-t0} seconds to load raw fif files (if not already loaded) and calculate events matrix"
            )
            print("\n")

            if fs != 16384:
                print("Now resampling raw data and events matrix")
                if fs == denoisedMeg.info["sfreq"]:
                    denoisedMeg, _ = denoisedMeg.resample(
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
                print(
                    "Now pickling and saving resampled raw file and events matrix"
                )
                eb.save.pickle((denoisedMeg, eventsMeg), thisPath)
                t3 = time.time()
                print(f"Took {t3-t2} seconds")
                print("\n")
            else:
                denoisedMegSfreq = denoisedMeg.info["sfreq"]
                del denoisedMeg

            os.remove(thisPath + ".tmp")

    # %%

    if not passThrough:
    
        if printAllEvents:
            print(events)
            print("\n")

        events[:, 2] = events[:, 2] - events[:, 1]
        events[:, 0] = events[:, 0] + int(
            round(soundDelay * fs)
        )  # Add speed of sound delay to all events at the very beginning

        if not eegOnly or eegTriggerRepair is not None:
            eventsMeg[:, 0] = (
                eventsMeg[:, 0]
                + int(round(soundDelay * fs))
                + int(round(presentationDelay * fs))
            )  # Add speed of sound delay and Presentation delay to all MEG events at the very beginning

            eventsMeg[eventsMeg[:, 2] == 180, 0] = eventsMeg[
                eventsMeg[:, 2] == 180, 0
            ] - int(
                round(presStopCorrection * fs)
            )  # Subtract triggering click duration from end of wav audio file for all MEG Presentation stop triggers

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

        # # Get the start of the experiment for both EEG and MEG as the first event time. The first event marker has to be the actual first event start time for both datasets
        # startSampleEEG = events[0, 0]
        # startTimeEEG = startSampleEEG / fs
        # stopTimeEEG = events[-1, 0] / fs
        # startTimeMEG = eventsMeg[0, 0] / fs
        # # Get the experiment duration from the EEG as the last minus the first event, and add 1 second just so we know we're not exactly on the ending sample
        # totalExpDuration = stopTimeEEG - startTimeEEG + 1
        # # And crop them both the same way. This should work because we have already established that there is next to no drift between the EEG and MEG recording computers
        # # print(startTimeEEG)
        # # print(stopTimeEEG)
        # # print(startTimeMEG)
        # # print(eventsMeg[-1, 0] / fs)
        # # print(totalExpDuration)
        # # print(events.shape)
        # # print(np.diff(events, axis=0))
        # # print(eventsMeg.shape)
        # # print(np.diff(eventsMeg, axis=0))
        # # denoised.crop(tmin=startTimeEEG, tmax=startTimeEEG + totalExpDuration)
        # # denoisedMeg.crop(tmin=startTimeMEG, tmax=startTimeMEG + totalExpDuration)
        # # # Subtract the first event sample time from all event sample times to start at 0 for the EEG events matrix, but don't bother with the MEG events matrix because we won't use it anymore now
        # # events[:, 0] = events[:, 0] - startSampleEEG

        if (
            eegTriggerRepair is not None
        ):  # Have to do something custom here that would not work generally
            if not eegOnly:  # Have to get this here if we didn't get it above
                denoisedMegSfreq = denoisedMeg.info["sfreq"]

            eventsEeg = events
            numStarts = sum(eventsEeg[:, 2] == 142)
            numStops = sum(eventsEeg[:, 2] == 148)

            newEventsEeg = np.zeros((numStarts + numStops, 3)).astype(int)
            count = 0
            for i in range(eventsEeg.shape[0]):
                if eventsEeg[i, 2] == 142:
                    newEventsEeg[count] = eventsEeg[i, :]
                    count += 1
                elif eventsEeg[i, 2] == 148:
                    newEventsEeg[count] = eventsEeg[i, :]
                    count += 1

            finalEventsEeg = np.zeros((64, 3)).astype(int)
            count = 0
            for i in range(finalEventsEeg.shape[0]):
                if not np.any(eegTriggerRepair == i):
                    finalEventsEeg[i, :] = newEventsEeg[count, :]
                    count += 1
                else:
                    prevEegTime = (
                        finalEventsEeg[i - 1, 0] / denoised.info["sfreq"]
                    )  # Get the time of the last eeg trigger in seconds
                    megDuration = (
                        eventsMeg[i, 0] - eventsMeg[i - 1, 0]
                    ) / denoisedMegSfreq  # Get the inter-trigger interval from the meg in seconds
                    newTriggerTime = (prevEegTime + megDuration) * denoised.info[
                        "sfreq"
                    ]  # Calculate the new trigger time in samples for the eeg
                    if eventsMeg[i, 2] == 174:
                        newTrigger = 142
                    elif eventsMeg[i, 2] == 180:
                        newTrigger = 148
                    finalEventsEeg[i, :] = np.array([newTriggerTime, 0, newTrigger]).astype(
                        int
                    )
            events = finalEventsEeg
            # print("\n")
            # print("****************************")
            # print("******Debugging*************")
            # print("****************************")
            # print("EEG:")
            # print(events)
            # print("\n")
            # print("MEG:")
            # print(eventsMeg)
            # print("****************************")
            # print("******Debugging*************")
            # print("****************************")
            # print("\n")

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
        if not alreadyFiltered:
            t0 = time.time()
            print(f"Not already filtered, so filtering EEG and MEG now for l_freq {l_freq} and h_freq {h_freq}")
            print("\n")
            if l_freq is not None or h_freq is not None:
                denoised.filter(l_freq=l_freq, h_freq=h_freq, phase=filterPhase)
                t1 = time.time()
                print(f"Took {t1-t0} seconds to filter EEG")
                print("\n")
                if not eegOnly:
                    denoisedMeg.filter(l_freq=l_freq, h_freq=h_freq, phase=filterPhase)
                    t2 = time.time()
                    print(f"Then took {t2-t1} seconds to filter MEG")
                print("\n")
            print(f"EEG info is")
            print("\n")
            print(denoised.info)
            if not eegOnly:
                print("\n")
                print(f"MEG info is")
                print("\n")
                print(denoisedMeg.info)
                print("\n")

        
        force_update_info = True

        if doPresentation:

            allPresEpochs = []
            allPresEpochsForCov = []

            for i in range(presStartEvents.shape[0]):
                tmax = presStopEvents[i] / fs - presStartEvents[i] / fs
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

                if not eegOnly:
                    # print("debugging")
                    # print(subject)
                    # print(i)
                    # print(mne.find_events(denoisedMeg))
                    # print(eventsMeg)
                    # print("done debugging")
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

                    epochMEG.add_channels([epoch], force_update_info=force_update_info)
                    epoch = epochMEG

                allPresEpochs.append(epoch)
                allPresEpochsForCov.append(epoch.copy().crop(tmin=0, tmax=60))

            print(f"Done creating Presentation epochs object, now calculating and saving data covariance matrix, and saving info object and epochs object")
            t0 = time.time()
            info = epoch.info
            dataCov = mne.compute_covariance(allPresEpochsForCov)
            del allPresEpochsForCov
            t1 = time.time()
            print(f"Data covariance calculated; time taken was {t1-t0} seconds")
            print("Now saving data covariance matrix")
            if not eegOnly:
                eb.save.pickle(dataCov, f"{megLocation}dataCovariance_fs{fs}")
            else:
                eb.save.pickle(dataCov, f"{eegLocation}dataCovariance_fs{fs}")
            t2 = time.time()
            print(f"Time taken was {t2-t1} seconds; now saving info object and epochs object")
            safelyPickle(allPresEpochs, hashedLocation + presHash + ".pickle")
            tempHasher = presHasher.copy()
            tempHasher.update("info".encode("utf-8"))
            tempHash = tempHasher.hexdigest()
            safelyPickle(info, hashedLocation + tempHash + ".pickle")
            t3 = time.time()
            print(f"Took {t3-t2} seconds to save")
            print("\n")

        if doTriggy:

            allTrigEpochs = []
            allTrigEpochsForCov = []

            for i in range(trigStartEvents.shape[0]):
                tmax = trigStopEvents[i] / fs - trigStartEvents[i] / fs
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

                if not eegOnly:
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

                    epochMEG.add_channels([epoch], force_update_info=force_update_info)
                    epoch = epochMEG

                allTrigEpochs.append(epoch)
                allTrigEpochsForCov.append(epoch.copy().crop(tmin=0, tmax=60))

            print(f"Done creating Triggy epochs object, now calculating and saving data covariance matrix, and saving info object and epochs object")
            t0 = time.time()
            info = epoch.info
            dataCov = mne.compute_covariance(allTrigEpochsForCov)
            del allTrigEpochsForCov
            t1 = time.time()
            print(f"Data covariance calculated; time taken was {t1-t0} seconds")
            print("Now saving data covariance matrix")
            if not eegOnly:
                eb.save.pickle(dataCov, f"{megLocation}dataCovariance_fs{fs}")
            else:
                eb.save.pickle(dataCov, f"{eegLocation}dataCovariance_fs{fs}")
            t2 = time.time()
            print(f"Time taken was {t2-t1} seconds; now saving info object and epochs object")
            safelyPickle(allTrigEpochs, hashedLocation + trigHash + ".pickle")
            tempHasher = trigHasher.copy()
            tempHasher.update("info".encode("utf-8"))
            tempHash = tempHasher.hexdigest()
            safelyPickle(info, hashedLocation + tempHash + ".pickle")
            t3 = time.time()
            print(f"Took {t3-t2} seconds to save")
            print("\n")
            
        del denoised
        if not eegOnly:
            del denoisedMeg

    else:

        t0 = time.time()
        if doPresentation:
            tempHasher = presHasher.copy()
            tempHasher.update("info".encode("utf-8"))
            tempHash = tempHasher.hexdigest()
            info = safelyUnpickle(hashedLocation + tempHash + ".pickle")
            del tempHasher

            [presHasher.update(i.encode("utf-8")) for i in nameOfRegressor]
            existingInds = np.ones(trialsToAnalyze.shape[0])
            for count, i in enumerate(trialsToAnalyze):
                tempHasher = presHasher.copy()
                tempHasher.update(np.array(i))
                tempHash = tempHasher.hexdigest()
                del tempHasher
                thisPath = hashedLocation + tempHash + ".pickle"
                if not os.path.exists(thisPath) or os.path.exists(thisPath + ".tmp"):
                    existingInds[count] = 0
            if sum(existingInds) != trialsToAnalyze.shape[0]:
                print(f"Only {sum(existingInds)} out of a total of {trialsToAnalyze.shape[0]} individual resampled epochs have been saved, therefore loading the pickle with MNE object with all epochs")
                allPresEpochs = safelyUnpickle(hashedLocation + presHash + ".pickle")
            else:
                print(f"All {sum(existingInds)} individual resampled epochs have been saved, so only loading those instead of loading MNE object with all epochs")
        if doTriggy:
            tempHasher = trigHasher.copy()
            tempHasher.update("info".encode("utf-8"))
            tempHash = tempHasher.hexdigest()
            info = eb.load.unpickle(hashedLocation + tempHash + ".pickle")
            del tempHasher

            [trigHasher.update(i.encode("utf-8")) for i in nameOfRegressor]
            existingInds = np.ones(trialsToAnalyze.shape[0])
            for count, i in enumerate(trialsToAnalyze):
                tempHasher = trigHasher.copy()
                tempHasher.update(np.array(i))
                tempHash = tempHasher.hexdigest()
                del tempHasher
                thisPath = hashedLocation + tempHash + ".pickle"
                if not os.path.exists(thisPath) or os.path.exists(thisPath + ".tmp"):
                    existingInds[count] = 0
            if sum(existingInds) != trialsToAnalyze.shape[0]:
                print(f"Only {sum(existingInds)} out of a total of {trialsToAnalyze.shape[0]} individual resampled epochs have been saved, therefore loading the pickle with MNE object with all epochs")
                allTrigEpochs = safelyUnpickle(hashedLocation + trigHash + ".pickle")
            else:
                print(f"All {sum(existingInds)} individual resampled epochs have been saved, so only loading those instead of loading MNE object with all epochs")


    regressorHashList = [subject.encode("utf-8"), denoisePipeline.encode("utf-8"), np.array(fs), str(l_freq).encode("utf-8"), str(h_freq).encode("utf-8"), filterPhase.encode("utf-8")]
    regressorHashList.append(typeOfRegressor.encode("utf-8"))
    regressorHashList.append(condition.encode("utf-8"))
    [regressorHashList.append(i.encode("utf-8")) for i in nameOfRegressor]
    regressorHasher = hashlib.blake2b(digest_size=20)
    [regressorHasher.update(i) for i in regressorHashList]

    print("\n")
    print("*************Debugging******************")
    print("\n")
    print(f"Regressor hash list is {regressorHashList}")
    print("\n")
    print("*************Debugging******************")
    print("\n")

    regressorHash = regressorHasher.hexdigest()

    if os.path.exists(regressorDir + "hashFiles/" + regressorHash + ".pickle"):

        print(f"Regressor object has been created and saved already, so loading it")
        allRegressors = safelyUnpickle(regressorDir + "hashFiles/" + regressorHash + ".pickle")

    else:

        t0 = time.time()
        print(f"Regressor object has not been saved yet, so loading all regressors, filtering them, and saving them")
        allRegressors = []

        for i in range(32):
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

            tempRegressorList = []
            tempLengthList = []

            for j in range(len(regressorNames[i])):

                if fs == 16384 or fs == 5000:  # If fs is 5000 it is probably the AN model...
                    if (
                        not np.any(skipDist[conditionNum] == i)
                        or typeOfRegressor == "mix"
                    ):  # If it's not a trial without a distractor, or if we're getting mixes anyway
                        regressor = eb.load.unpickle(
                            f"{regressorDir}{regressorNames[i][j]}"
                        )
                    else:  # otherwise save it as numpy array with 0; this won't error out, and will not get used below anyway
                        regressor = np.array([0])
                elif (
                    fs == 500
                ):  # Else it is probably a regressor from Eelbrain, so it's an NDVar so we need .x
                    if (
                        not np.any(skipDist[conditionNum] == i)
                        or typeOfRegressor == "mix"
                    ):  # If it's not a trial without a distractor, or if we're getting mixes anyway
                        regressor = eb.load.unpickle(
                            f"{regressorDir}{regressorNames[i][j]}"
                        )
                        if isinstance(regressor, eb._data_obj.NDVar):
                            regressor = regressor.x
                            lenRegressor = len(regressor)
                            regressor = sp.signal.resample(
                                regressor, int(lenRegressor / 4)
                            )  # The Eelbrain regressors are made with fs=2000, so use a magic number for now and resample them for fs=500
                    else:  # otherwise save it as numpy array with 0; this won't error out, and will not get used below anyway
                        regressor = np.array([0])

                tempRegressorList.append(regressor)
                tempLengthList.append(regressor.shape[0])

            # The reason for zero padding out slightly shorter regressors here is that the ones from Eelbrain/TRF-Tools end up
            # 10 ms shorter than the input because of a default 10 ms integration window time. So it is appropriate to just
            # add zeros to the end of that to bring it within a sample or two of being exactly "correct", meaning the length of
            # the other regressors.
            maxLen = np.array(tempLengthList).max()
            regressors = np.zeros((maxLen, len(regressorNames[i])))
            for j in range(len(regressorNames[i])):
                regressors[: len(tempRegressorList[j]), j] = tempRegressorList[j]

            # invVarProps.append(invVarProp)

            t1 = time.time()
            print(f"Took {t1-t0} seconds to load and organize regressors")
            print(f"Regressor matrix is shape {regressors.shape}")
            regressors = regressors.T  # Time is now the second dimension, for making MNE RawArray
            print(f"Regressor matrix is now shape {regressors.shape} in order to be the data of an MNE raw object for filtering")
            print("\n")
            regressorInfo = mne.create_info(ch_names=regressorNames[i], sfreq=fs, ch_types="stim")
            regressorRaw = mne.io.RawArray(data=regressors, info=regressorInfo)
            regressorRaw.notch_filter(np.arange(60, fs/2, 60), picks="all", phase=filterPhase)
            regressorRaw.filter(l_freq=l_freq, h_freq=h_freq, picks="all", phase=filterPhase)

            # And then append this to the main list of regressors for each trial of the session
            allRegressors.append(regressorRaw)

        safelyPickle(allRegressors, regressorDir + "hashFiles/" + regressorHash + ".pickle")
        t2 = time.time()
        print(f"Then took {t2-t1} seconds to filter and save regressors")

            
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
    tDeconv = time.time()

    # Initialize this with the third dimension being the length of the input parameter nameOfRegressor, which is
    # a list of strings, one string for each regressor filename, even if it's only one regressor
    regressorsToDo = np.zeros(
        (lenToAnalyze * fs, len(trialsToAnalyze), len(nameOfRegressor))
    )

    # trainSubset = np.arange(0, len(trialsToAnalyze))
    # testSubset = np.arange(0, len(trialsToAnalyze))

    # if condition == "A" or condition == "B":
    #     trainSubset = np.arange(0, len(trialsToAnalyze))[::2]
    #     testSubset = np.arange(0, len(trialsToAnalyze))[1::2]
    # elif condition == "C" or condition == "D":
    #     trainSubset = np.arange(0, len(trialsToAnalyze))[1::2]
    #     testSubset = np.arange(0, len(trialsToAnalyze))[::2]

    # nFolds = 4
    nFolds = len(trialsToAnalyze)
    indsToDo = np.arange(len(trialsToAnalyze))
    # allTestInds = np.random.choice(indsToDo, nFolds, replace=False)
    allTestInds = indsToDo

    if doPresentation:

        if not passThrough:  # passThrough means we already have the epochs object saved, and if we did, we already updated the hashers above
            [presHasher.update(i.encode("utf-8")) for i in nameOfRegressor]
        
        presEpochsToDo = np.zeros((lenToAnalyze * fs, len(trialsToAnalyze), nChannels))

        for count, i in enumerate(trialsToAnalyze):

            regressor = allRegressors[i].get_data().T

            tempHasher = presHasher.copy()
            tempHasher.update(np.array(i))
            presHash = tempHasher.hexdigest()
            del tempHasher

            if not os.path.exists(hashedLocation + presHash + ".pickle"):

                epoch = allPresEpochs[i]

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

                safelyPickle(response, hashedLocation + presHash + ".pickle")
                
            else:

                response = safelyUnpickle(hashedLocation + presHash + ".pickle")

            regressorsToDo[:, count, :] = regressor[: lenToAnalyze * fs, :]
            presEpochsToDo[:, count, :] = response[: lenToAnalyze * fs, :]


        if np.any(np.isinf(regressorsToDo)):
            print(f"There are {np.sum(np.isinf(regressorsToDo))} infs in regressor matrix for subject {subject} before z-scoring")
        if np.any(np.isinf(presEpochsToDo)):
            print(f"There are {np.sum(np.isinf(presEpochsToDo))} infs in trig response matrix for subject {subject} before z-scoring")
        if np.any(np.isnan(regressorsToDo)):
            print(f"There are {np.sum(np.isnan(regressorsToDo))} NaNs in regressor matrix for subject {subject} before z-scoring")
        if np.any(np.isnan(presEpochsToDo)):
            print(f"There are {np.sum(np.isnan(presEpochsToDo))} NaNs in trig response matrix for subject {subject} before z-scoring")

        if addNoiseToRegressors is not None:
            regressorSTDs = np.std(regressorsToDo, axis=0)
            noise = np.random.random(regressorsToDo.shape) * 2 - 1
            noise *= addNoiseToRegressors * regressorSTDs
            regressorsToDo += noise
            
        regressorSTDs = np.std(regressorsToDo, axis=0)
        presEpochsSTDs = np.std(presEpochsToDo, axis=0)
            
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

        def doOnePresFold(
            indsToDo,
            allTestInds,
            fold,
            windowStart,
            edgePad,
            windowEnd,
            fs,
            eps,
            scoring,
            regressorsToDo,
            presEpochsToDo,
        ):

            testSubset = allTestInds[fold]
            trainSubset = np.delete(indsToDo, testSubset)

            rfPres = mne.decoding.ReceptiveField(
                windowStart - edgePad,
                windowEnd + edgePad,
                fs,
                estimator=eps,
                scoring=scoring,
            )
            if np.any(np.isinf(regressorsToDo)):
                print(f"There are {np.sum(np.isinf(regressorsToDo))} infs in regressor matrix for subject {subject} after z-scoring")
            if np.any(np.isinf(presEpochsToDo)):
                print(f"There are {np.sum(np.isinf(presEpochsToDo))} infs in trig response matrix for subject {subject} after z-scoring")
            if np.any(np.isnan(regressorsToDo)):
                print(f"There are {np.sum(np.isnan(regressorsToDo))} NaNs in regressor matrix for subject {subject} after z-scoring")
            if np.any(np.isnan(presEpochsToDo)):
                print(f"There are {np.sum(np.isnan(presEpochsToDo))} NaNs in trig response matrix for subject {subject} after z-scoring")


            # Now there is no need for any indexing or adding dimensions etc. here, because regressors
            # matrix is already time x epochs x regressors, and M/EEG is already time x epochs x channels
            rfPres.fit(
                regressorsToDo[:, trainSubset, :], presEpochsToDo[:, trainSubset, :]
            )

            score = rfPres.score(
                regressorsToDo[:, testSubset, :], presEpochsToDo[:, testSubset, :]
            )

            # The coef_ matrix in here is channels x regressors x time/n_delays, so move
            # the axes so that time is the first dimension, and then channels and then regressors
            TRFsTimePres = np.moveaxis(
                rfPres.coef_[:, :, int(edgePad * fs) : int(-edgePad * fs - 1)], -1, 0
            )

            return rfPres, score, TRFsTimePres

        if doParallel:

            if nFolds > 12:
                n_jobs = nFolds / 2
            else:
                n_jobs = nFolds
                
            results = joblib.Parallel(
                n_jobs=n_jobs, backend=parallelBackend, verbose=verbose
            )(
                joblib.delayed(doOnePresFold)(
                    indsToDo,
                    allTestInds,
                    fold,
                    windowStart,
                    edgePad,
                    windowEnd,
                    fs,
                    eps,
                    scoring,
                    regressorsToDo,
                    presEpochsToDo,
                )
                for fold in range(nFolds)
            )
            rfPresAll = []
            scoresAll = []
            TRFsTimePresAll = []
            for loopInd in range(len(results)):
                rfPresAll.append(results[loopInd][0])
                scoresAll.append(results[loopInd][1])
                TRFsTimePresAll.append(results[loopInd][2])
        else:
            rfPresAll = []
            scoresAll = []
            TRFsTimePresAll = []
            for fold in range(nFolds):
                rfPres, score, TRFsTimePres = doOnePresFold(
                    indsToDo,
                    allTestInds,
                    fold,
                    windowStart,
                    edgePad,
                    windowEnd,
                    fs,
                    eps,
                    scoring,
                    regressorsToDo,
                    presEpochsToDo,
                )
                rfPresAll.append(rfPres)
                scoresAll.append(score)
                TRFsTimePresAll.append(TRFsTimePres)

        rfPres = rfPresAll[
            -1
        ]  # Here just get the last receptive field object to save, but not sure whether we would need any at all
        score = np.array(scoresAll).mean(axis=0)
        TRFsTimePres = np.array(TRFsTimePresAll).mean(axis=0)

        print("Scoring done")
        print(f"Shape of score is {score.shape}")
        print(f"Max of score is {score.max()}")
        print(f"Median of score is {np.median(score)}")
        print(f"Min of score is {score.min()}")
        print(score)
        print("\n")
        print("\n")

    if doTriggy:

        if not passThrough:  # passThrough means we already have the epochs object saved, and if we did, we already updated the hashers above
            [trigHasher.update(i.encode("utf-8")) for i in nameOfRegressor]

        trigEpochsToDo = np.zeros((lenToAnalyze * fs, len(trialsToAnalyze), nChannels))

        for count, i in enumerate(trialsToAnalyze):

            regressor = allRegressors[i].get_data().T
            
            tempHasher = trigHasher.copy()
            tempHasher.update(np.array(i))
            trigHash = tempHasher.hexdigest()
            del tempHasher

            if not os.path.exists(hashedLocation + trigHash + ".pickle"):

                epoch = allTrigEpochs[i]

                currentEpochMat = (
                    epoch.get_data().squeeze().T
                )  # Time is now the first dimension (dimension 0)

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

                safelyPickle(response, hashedLocation + trigHash + ".pickle")

            else:

                response = safelyUnpickle(hashedLocation + trigHash + ".pickle")

            regressorsToDo[:, count, :] = regressor[: lenToAnalyze * fs, :]
            trigEpochsToDo[:, count, :] = response[: lenToAnalyze * fs, :]

            
        if np.any(np.isinf(regressorsToDo)):
            print(f"There are {np.sum(np.isinf(regressorsToDo))} infs in regressor matrix for subject {subject} before z-scoring")
        if np.any(np.isinf(trigEpochsToDo)):
            print(f"There are {np.sum(np.isinf(trigEpochsToDo))} infs in trig response matrix for subject {subject} before z-scoring")
        if np.any(np.isnan(regressorsToDo)):
            print(f"There are {np.sum(np.isnan(regressorsToDo))} NaNs in regressor matrix for subject {subject} before z-scoring")
        if np.any(np.isnan(trigEpochsToDo)):
            print(f"There are {np.sum(np.isnan(trigEpochsToDo))} NaNs in trig response matrix for subject {subject} before z-scoring")

        if addNoiseToRegressors is not None:
            regressorSTDs = np.std(regressorsToDo, axis=0)
            noise = np.random.random(regressorsToDo.shape) * 2 - 1
            noise *= addNoiseToRegressors * regressorSTDs
            regressorsToDo += noise
            
        regressorSTDs = np.std(regressorsToDo, axis=0)
        trigEpochsSTDs = np.std(trigEpochsToDo, axis=0)

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

        def doOneTrigFold(
            indsToDo,
            allTestInds,
            fold,
            windowStart,
            edgePad,
            windowEnd,
            fs,
            eps,
            scoring,
            regressorsToDo,
            trigEpochsToDo,
        ):

            testSubset = allTestInds[fold]
            trainSubset = np.delete(indsToDo, testSubset)

            rfTrig = mne.decoding.ReceptiveField(
                windowStart - edgePad,
                windowEnd + edgePad,
                fs,
                estimator=eps,
                scoring=scoring,
            )
            if np.any(np.isinf(regressorsToDo)):
                print(f"There are {np.sum(np.isinf(regressorsToDo))} infs in regressor matrix for subject {subject} after z-scoring")
            if np.any(np.isinf(trigEpochsToDo)):
                print(f"There are {np.sum(np.isinf(trigEpochsToDo))} infs in trig response matrix for subject {subject} after z-scoring")
            if np.any(np.isnan(regressorsToDo)):
                print(f"There are {np.sum(np.isnan(regressorsToDo))} NaNs in regressor matrix for subject {subject} after z-scoring")
            if np.any(np.isnan(trigEpochsToDo)):
                print(f"There are {np.sum(np.isnan(trigEpochsToDo))} NaNs in trig response matrix for subject {subject} after z-scoring")

            # Now there is no need for any indexing or adding dimensions etc. here, because regressors
            # matrix is already time x epochs x regressors, and M/EEG is already time x epochs x channels
            rfTrig.fit(
                regressorsToDo[:, trainSubset, :], trigEpochsToDo[:, trainSubset, :]
            )

            score = rfTrig.score(
                regressorsToDo[:, testSubset, :], trigEpochsToDo[:, testSubset, :]
            )

            # The coef_ matrix in here is channels x regressors x time/n_delays, so move
            # the axes so that time is the first dimension, and then channels and then regressors
            TRFsTimeTrig = np.moveaxis(
                rfTrig.coef_[:, :, int(edgePad * fs) : int(-edgePad * fs - 1)], -1, 0
            )

            return rfTrig, score, TRFsTimeTrig

        if doParallel:

            if nFolds > 12:
                n_jobs = nFolds / 2
            else:
                n_jobs = nFolds
                
            results = joblib.Parallel(
                n_jobs=n_jobs, backend=parallelBackend, verbose=verbose
            )(
                joblib.delayed(doOneTrigFold)(
                    indsToDo,
                    allTestInds,
                    fold,
                    windowStart,
                    edgePad,
                    windowEnd,
                    fs,
                    eps,
                    scoring,
                    regressorsToDo,
                    trigEpochsToDo,
                )
                for fold in range(nFolds)
            )
            rfTrigAll = []
            scoresAll = []
            TRFsTimeTrigAll = []
            for loopInd in range(len(results)):
                rfTrigAll.append(results[loopInd][0])
                scoresAll.append(results[loopInd][1])
                TRFsTimeTrigAll.append(results[loopInd][2])
        else:
            rfTrigAll = []
            scoresAll = []
            TRFsTimeTrigAll = []
            for fold in range(nFolds):
                rfTrig, score, TRFsTimeTrig = doOneTrigFold(
                    indsToDo,
                    allTestInds,
                    fold,
                    windowStart,
                    edgePad,
                    windowEnd,
                    fs,
                    eps,
                    scoring,
                    regressorsToDo,
                    trigEpochsToDo,
                )
                rfTrigAll.append(rfTrig)
                scoresAll.append(score)
                TRFsTimeTrigAll.append(TRFsTimeTrig)

        rfTrig = rfTrigAll[
            -1
        ]  # Here just get the last receptive field object to save, but not sure whether we would need any at all
        score = np.array(scoresAll).mean(axis=0)
        TRFsTimeTrig = np.array(TRFsTimeTrigAll).mean(axis=0)

        print("Scoring done")
        print(f"Shape of score is {score.shape}")
        print(f"Max of score is {score.max()}")
        print(f"Median of score is {np.median(score)}")
        print(f"Min of score is {score.min()}")
        print(score)
        print("\n")
        print("\n")

    if doPresentation:

        print(
            f"Shape of time-domain Presentation TRF matrix is now {TRFsTimePres.shape} which should be time/n_delays X channels X regressors"
        )
        print("\n")

    if doTriggy:

        print(
            f"Shape of time-domain Triggy TRF matrix is now {TRFsTimeTrig.shape} which should be time/n_delays X channels X regressors"
        )
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

    if doPresentation:

        timePres_portion = TRFsTimePres * unitCoefficient

    if doTriggy:

        timeTrig_portion = TRFsTimeTrig * unitCoefficient

    # %%
    if doPlotting:

        if doPresentation:

            plt.figure(figsize=figsize)

            plt.gca().set_prop_cycle(plt.cycler("color", colors))
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

    if eegOnly:
        saveLocation = eegLocation
    else:
        saveLocation = megLocation

    if doPresentation:

        evokedTimePres = []
        for i in range(len(nameOfRegressor)):
            evokedTimePres.append(
                mne.EvokedArray(
                    timePres_portion[:, :, i].T, info, tmin=windowStart
                )
            )

        if saveEvoked:

            for i in range(len(nameOfRegressor)):
                evokedTimePres[i].save(
                    f"{saveLocation}{filenamePrefix}{nameOfRegressor[i]}_{typeOfRegressor}{filenameSuffix}-ave.fif",
                    overwrite=True,
                )

        if saveRf:

            # eb.save.pickle(
            #     (rfPres, evokedTimePres, score, eps),
            #     f"{saveLocation}{filenamePrefix}_{typeOfRegressor}{filenameSuffix}_lambda{lambdaInd}.pickle",
            # )

            eb.save.pickle(
                (None, evokedTimePres, score, eps, regressorSTDs, presEpochsSTDs),
                f"{saveLocation}{filenamePrefix}_{typeOfRegressor}{filenameSuffix}_lambda{lambdaInd}.pickle",
            )

    if doTriggy:

        evokedTimeTrig = []
        for i in range(len(nameOfRegressor)):
            evokedTimeTrig.append(
                mne.EvokedArray(
                    timeTrig_portion[:, :, i].T, info, tmin=windowStart
                )
            )

        if saveEvoked:

            for i in range(len(nameOfRegressor)):
                evokedTimeTrig[i].save(
                    f"{saveLocation}{filenamePrefix}{nameOfRegressor[i]}_{typeOfRegressor}{filenameSuffix}-ave.fif",
                    overwrite=True,
                )

        if saveRf:

            # eb.save.pickle(
            #     (rfTrig, evokedTimeTrig, score, eps),
            #     f"{saveLocation}{filenamePrefix}_{typeOfRegressor}{filenameSuffix}_lambda{lambdaInd}.pickle",
            # )

            eb.save.pickle(
                (None, evokedTimeTrig, score, eps, regressorSTDs, trigEpochsSTDs),
                f"{saveLocation}{filenamePrefix}_{typeOfRegressor}{filenameSuffix}_lambda{lambdaInd}.pickle",
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
    doSaving = True

    # evokedFilenamePrefix = "recField"
    evokedFilenamePrefix = "concurRecField"
    # savedFilenamePrefix = "sources"
    savedFilenamePrefix = "concurSources"

    temp = sorted(glob.glob(f"{eegLocation}*baseline*bdf"))
    eegBaselineFile = temp[-1]
    temp = sorted(
        glob.glob(f"{megLocation}*baseline*fif")
    )  # Read both of these the same way, even though there really should never be more than one of either named this way
    megBaselineFile = temp[-1]

    l_freq = bandpassFreqs[0]
    h_freq = bandpassFreqs[1]

    refs = ["EXG3", "EXG4"]
    # notchFreqs = np.arange(60, 8192, 60)
    notchFreqs = np.arange(60, 250, 60)

    # labels_vol = ["Left-Thalamus-Proper", "Right-Thalamus-Proper", "Brain-Stem"]
    labels_vol = None

    evoked = mne.read_evokeds(
        f"{megLocation}{evokedFilenamePrefix}{nameOfRegressor}_{typeOfRegressor}{filenameSuffix}-ave.fif"
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
    loose_vol = dict(volume=1.0)
    depth = 0.8
    # depth = 10.0

    lambda2 = 1.0 / snr**2

    inverse_operator_vol = mne.minimum_norm.make_inverse_operator(
        evoked.info, fwd_vol, baselineCov, depth=depth, loose=loose_vol, verbose=True
    )

    # %%
    stc_vec = mne.minimum_norm.apply_inverse(
        evoked, inverse_operator_vol, lambda2, inv_method, pick_ori="vector"
    )

    if doPlotting:

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

    if doSaving:

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
            f"{megLocation}{savedFilenamePrefix}{nameOfRegressor}_{typeOfRegressor}{filenameSuffix}.pickle",
        )
