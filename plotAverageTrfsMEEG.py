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

from neuroAndSignalTools.freqAnalysis import *
from neuroAndSignalTools.deconvGeneralized import *

matplotlib.use("QtAgg")
plt.ion()

# %%
# parentDir = "/Users/karl/map/"
parentDir = "/Volumes/Seagate/map/"

subDirs = "/eegAndMeg/meg/"
nChannels = 32 + 157

# filePrefix = "evoked"
# filePrefix = "recField"
# filePrefix = "concurRecField"
# filePrefix = "acousticConcurRecField"
# filePrefix = "acousticRecFieldAllFolds"
# filePrefix = "linguisticConcurRecField"
# filePrefix = "linguisticRecFieldAllFolds"
filePrefix = "combinedRecFieldAllFolds"

edgePad = .001


# %%
# nameOfRegressor = "_ANmodel_correctedLevels"
# lenResponse = 425
# lenToRamp = .008
# # lenToRamp = 0
# expToRamp = 2

# nameOfRegressor = "_ANmodel_maxFs"
# lenResponse = 1393
# lenToRamp = .008
# # lenToRamp = 0
# expToRamp = 2

# nameOfRegressor = "~gammatone-1"
# lenResponse = 426
# lenToRamp = 0
# expToRamp = 2

nameOfRegressor = "~gammatone-on-1"
lenResponse = 426
lenToRamp = 0
expToRamp = 2

# nameOfRegressor = "~wordOnsets_gaussian15msSD"
# lenResponse = 426
# lenToRamp = 0
# expToRamp = 2

# nameOfRegressor = "~phoneOnsets_gaussian15msSD"
# lenResponse = 426
# lenToRamp = 0
# expToRamp = 2

# nameOfRegressor = "~phsurp"
# lenResponse = 426
# lenToRamp = 0
# expToRamp = 2

# nameOfRegressor = "~cohtent"
# lenResponse = 426
# lenToRamp = 0
# expToRamp = 2

# nameOfRegressor = "~wordprob"
# lenResponse = 426
# lenToRamp = 0
# expToRamp = 2

# nameOfRegressor = "~wordsurp"
# lenResponse = 426
# lenToRamp = 0
# expToRamp = 2


# nameOfRegressors = [
#     "~gammatone-1",
#     "~gammatone-on-1",
#     "~wordOnsets_gaussian15msSD",
#     "~phoneOnsets_gaussian15msSD",
#     "~phsurp",
#     "~cohtent",
#     "~wordprob",
#     "~wordsurp",
# ]

nameOfRegressors = [
    "~gammatone-1",
    "~gammatone-on-1",
    "~wordOnsets_gaussian15msSD",
    "~phoneOnsets_gaussian15msSD",
    "~phsurp",
    "~cohtent",
    "~wordsurp",
]

# nameOfRegressors = [
#     "~gammatone-1",
#     "~gammatone-on-1",
# ]

# nameOfRegressors = [
#     "~wordOnsets_gaussian15msSD",
#     "~phoneOnsets_gaussian15msSD",
#     "~phsurp",
#     "~cohtent",
#     "~wordsurp",
# ]

# nameOfRegressors = [
#     "~wordOnsets_gaussian15msSD",
#     "~phoneOnsets_gaussian15msSD",
#     "~phsurp",
#     "~cohtent",
#     "~wordprob",
#     "~wordsurp",
# ]

lambdas = np.logspace(-1, 9, 15)
lambdaInd = 14  # This indexes the lambdas that have been run, defined above

timeToAddToStart = 0
# timeToAddToStart = 0.008

complexSubtypeNames = [" (real)", " (imag)", " (abs)", " (phase)"]

ci = 0.95  # confidence interval for shading


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
]

## List not including the 4 lab members as subjects
# subjectsToAverage = [
#     "R3045",
#     "R3089",
#     "R3093",
#     "R3095",
#     "R3157",
#     "R2783",
#     "R3170",
#     "R3172",
#     "R3193",
#     "R3184",
#     "R3214",
# ]

# subjectsToAverage=["R3045", "R3089", "R3093", "R3095", "R3151", "R2774", "R3152", "R2877", "R3157"]
# subjectsToAverage=["R3045", "R3089", "R3093", "R3095", "R3151", "R2774", "R3152", "R2877"]
# subjectsToAverage=["R3095", "R3151", "R2774", "R3152", "R2877", "R3157", "R2783"]
# subjectsToAverage=["R2877", "R3151", "R3152"]
# subjectsToAverage = ["R3151"]


# %%
def readRecField(parentDir, subject, subDirs, filePrefix, typeOfRegressor, filenameSuffix, lambdaInd, nameOfRegressor, nameOfRegressors):
    megLocation = parentDir + subject + subDirs
    regressorInd = nameOfRegressors.index(nameOfRegressor)
    
    recField, evokedAll, score, eps = eb.load.unpickle(f"{megLocation}{filePrefix}_{typeOfRegressor}{filenameSuffix}_lambda{lambdaInd}.pickle")
    evoked = evokedAll[regressorInd]

    # # Temp version
    # recField, score, eps = eb.load.unpickle(f"{megLocation}{filePrefix}_{typeOfRegressor}{filenameSuffix}_lambda{lambdaInd}.pickle")
    # tempEvoked = mne.read_evokeds(
    #     f"{megLocation}{filePrefix}{nameOfRegressor}_{typeOfRegressor}{filenameSuffix}-ave.fif"
    # )[0]

    # fs = tempEvoked.info["sfreq"]
    # tempMat = recField.coef_[:, regressorInd, int(edgePad * fs) : int(-edgePad * fs - 1)]

    # evoked = mne.EvokedArray(tempMat, tempEvoked.info, tmin=tempEvoked.times[0])


    
    return evoked, score
    

def makeAnalytic(mat):
    analyticMat = sp.signal.hilbert(mat, axis=0)
    return analyticMat


def constructComplexList(rampFunction, avgMat, evoked, scores, comment, complexSubtypeNames):

    evokedAvgList = []
    evokedAvgList.append(
        mne.EvokedArray(
            rampFunction * np.real(makeAnalytic(avgMat).mean(axis=2)).T,
            evoked.info,
            tmin=evoked.times[0],
            comment=comment + complexSubtypeNames[0],
        )
    )
    evokedAvgList.append(
        mne.EvokedArray(
            rampFunction * np.imag(makeAnalytic(avgMat).mean(axis=2)).T,
            evoked.info,
            tmin=evoked.times[0],
            comment=comment + complexSubtypeNames[1],
        )
    )
    evokedAvgList.append(
        mne.EvokedArray(
            rampFunction * np.abs(makeAnalytic(avgMat).mean(axis=2)).T,
            evoked.info,
            tmin=evoked.times[0],
            comment=comment + complexSubtypeNames[2],
        )
    )
    evokedAvgList.append(
        mne.EvokedArray(
            rampFunction * np.angle(makeAnalytic(avgMat).mean(axis=2)).T,
            evoked.info,
            tmin=evoked.times[0],
            comment=comment + complexSubtypeNames[3],
        )
    )

    realList = []
    imagList = []
    absList = []
    angleList = []
    for i in range(avgMat.shape[2]):
        realList.append(
            mne.EvokedArray(
                rampFunction * np.real(makeAnalytic(avgMat[:, :, i])).T,
                evoked.info,
                tmin=evoked.times[0],
                comment=comment + complexSubtypeNames[0],
            )
        )
        imagList.append(
            mne.EvokedArray(
                rampFunction * np.imag(makeAnalytic(avgMat[:, :, i])).T,
                evoked.info,
                tmin=evoked.times[0],
                comment=comment + complexSubtypeNames[1],
            )
        )
        absList.append(
            mne.EvokedArray(
                rampFunction * np.abs(makeAnalytic(avgMat[:, :, i])).T,
                evoked.info,
                tmin=evoked.times[0],
                comment=comment + complexSubtypeNames[2],
            )
        )
        angleList.append(
            mne.EvokedArray(
                rampFunction * np.angle(makeAnalytic(avgMat[:, :, i])).T,
                evoked.info,
                tmin=evoked.times[0],
                comment=comment + complexSubtypeNames[3],
            )
        )

    evokedList = [realList, imagList, absList, angleList]

    return evokedAvgList, evokedList, scores


# %%
typeOfRegressor = "mix"

filenameSuffix = "_quiet"


avgMat = np.zeros((lenResponse, nChannels, len(subjectsToAverage)))

scores = np.zeros(nChannels)

for i, subject in enumerate(subjectsToAverage):
    megLocation = parentDir + subject + subDirs
    evoked, score = readRecField(parentDir, subject, subDirs, filePrefix, typeOfRegressor, filenameSuffix, lambdaInd, nameOfRegressor, nameOfRegressors)
    evokedMat = evoked.get_data().T[int(timeToAddToStart * evoked.info["sfreq"]) :, :]
    avgMat[int(timeToAddToStart * evoked.info["sfreq"]) :, :, i] = evokedMat
    scores += score

scores /= i+1

fs = evoked.info["sfreq"]
lenResponse = avgMat.shape[0]
rampFunction = np.concatenate(
    (
        np.linspace(0, 1, int(lenToRamp * fs)) ** expToRamp,
        np.ones(lenResponse - int(lenToRamp * fs)),
    )
)
rampFunction = rampFunction[None, :]


# evokedAvg = mne.EvokedArray(
#     avgMat.mean(axis=2).T, evoked.info, tmin=evoked.times[0], comment="Target"
# )
evokedAvgQuiet = constructComplexList(
    rampFunction,
    avgMat,
    evoked,
    scores,
    comment="T: Quiet",
    complexSubtypeNames=complexSubtypeNames,
)

# evokedAvgQuiet.append(mne.EvokedArray(
#     rampFunction * np.real(makeAnalytic(avgMat).mean(axis=2)).T, evoked.info, tmin=evoked.times[0], comment="T: Quiet (real)"
# ))
# evokedAvgQuiet.append(mne.EvokedArray(
#     rampFunction * np.imag(makeAnalytic(avgMat).mean(axis=2)).T, evoked.info, tmin=evoked.times[0], comment="T: Quiet (imag)"
# ))
# evokedAvgQuiet.append(mne.EvokedArray(
#     rampFunction * np.abs(makeAnalytic(avgMat).mean(axis=2)).T, evoked.info, tmin=evoked.times[0], comment="T: Quiet (abs)"
# ))
# evokedAvgQuiet.append(mne.EvokedArray(
#     rampFunction * np.angle(makeAnalytic(avgMat).mean(axis=2)).T, evoked.info, tmin=evoked.times[0], comment="T: Quiet (phase)"
# ))

# %%
# evokedAvg.pick_types(eeg=True).plot_topo(color="r", legend=False)

# %%
typeOfRegressor = "target"

filenameSuffix = "_easy"

avgMat = np.zeros((lenResponse, nChannels, len(subjectsToAverage)))

scores = np.zeros(nChannels)

for i, subject in enumerate(subjectsToAverage):
    megLocation = parentDir + subject + subDirs
    evoked, score = readRecField(parentDir, subject, subDirs, filePrefix, typeOfRegressor, filenameSuffix, lambdaInd, nameOfRegressor, nameOfRegressors)
    evokedMat = evoked.get_data().T[int(timeToAddToStart * evoked.info["sfreq"]) :, :]
    avgMat[int(timeToAddToStart * evoked.info["sfreq"]) :, :, i] = evokedMat
    scores += score

scores /= i+1

evokedAvgTargetEasy = constructComplexList(
    rampFunction,
    avgMat,
    evoked,
    scores,
    comment="T: Easy",
    complexSubtypeNames=complexSubtypeNames,
)

# %%
typeOfRegressor = "target"

filenameSuffix = "_hard"

avgMat = np.zeros((lenResponse, nChannels, len(subjectsToAverage)))

scores = np.zeros(nChannels)

for i, subject in enumerate(subjectsToAverage):
    megLocation = parentDir + subject + subDirs
    evoked, score = readRecField(parentDir, subject, subDirs, filePrefix, typeOfRegressor, filenameSuffix, lambdaInd, nameOfRegressor, nameOfRegressors)
    evokedMat = evoked.get_data().T[int(timeToAddToStart * evoked.info["sfreq"]) :, :]
    avgMat[int(timeToAddToStart * evoked.info["sfreq"]) :, :, i] = evokedMat
    scores += score

scores /= i+1

evokedAvgTargetHard = constructComplexList(
    rampFunction,
    avgMat,
    evoked,
    scores,
    comment="T: Hard",
    complexSubtypeNames=complexSubtypeNames,
)

# %%
typeOfRegressor = "distractor"

filenameSuffix = "_easy"

avgMat = np.zeros((lenResponse, nChannels, len(subjectsToAverage)))

scores = np.zeros(nChannels)

for i, subject in enumerate(subjectsToAverage):
    megLocation = parentDir + subject + subDirs
    evoked, score = readRecField(parentDir, subject, subDirs, filePrefix, typeOfRegressor, filenameSuffix, lambdaInd, nameOfRegressor, nameOfRegressors)
    evokedMat = evoked.get_data().T[int(timeToAddToStart * evoked.info["sfreq"]) :, :]
    avgMat[int(timeToAddToStart * evoked.info["sfreq"]) :, :, i] = evokedMat
    scores += score

scores /= i+1

evokedAvgDistEasy = constructComplexList(
    rampFunction,
    avgMat,
    evoked,
    scores,
    comment="D: Easy",
    complexSubtypeNames=complexSubtypeNames,
)

# %%
typeOfRegressor = "distractor"

filenameSuffix = "_hard"

avgMat = np.zeros((lenResponse, nChannels, len(subjectsToAverage)))

scores = np.zeros(nChannels)

for i, subject in enumerate(subjectsToAverage):
    megLocation = parentDir + subject + subDirs
    evoked, score = readRecField(parentDir, subject, subDirs, filePrefix, typeOfRegressor, filenameSuffix, lambdaInd, nameOfRegressor, nameOfRegressors)
    evokedMat = evoked.get_data().T[int(timeToAddToStart * evoked.info["sfreq"]) :, :]
    avgMat[int(timeToAddToStart * evoked.info["sfreq"]) :, :, i] = evokedMat
    scores += score

scores /= i+1

evokedAvgDistHard = constructComplexList(
    rampFunction,
    avgMat,
    evoked,
    scores,
    comment="D: Hard",
    complexSubtypeNames=complexSubtypeNames,
)

# %%
# mne.viz.plot_evoked_topo([evokedAvg, evokedAvg2, evokedAvg3], color=["blue", "red", "green"], legend=True)
# mne.viz.plot_evoked_topo([evokedAvg, evokedAvg2], color=["blue", "red"], legend=True)
# mne.viz.plot_evoked_topo([evokedAvg, evokedAvg2, evokedAvg3], legend=True)
# mne.viz.plot_evoked_topo([evokedAvg, evokedAvg2, evokedAvg3, evokedAvg4, evokedAvg5], color=["blue", "cyan", "green", "red", "brown"], legend=True)
# mne.viz.plot_evoked_topo([evokedAvg, evokedAvg2], legend=True)


# %%
# typeOfRegressor = "target"

# filenameSuffix = "_male"

# avgMat = np.zeros((lenResponse, nChannels, len(subjectsToAverage)))

# scores = np.zeros(nChannels)

# for i, subject in enumerate(subjectsToAverage):
#     megLocation = parentDir + subject + subDirs
#     evoked, score = readRecField(parentDir, subject, subDirs, filePrefix, typeOfRegressor, filenameSuffix, lambdaInd, nameOfRegressor, nameOfRegressors)
#     evokedMat = evoked.get_data().T[int(timeToAddToStart * evoked.info["sfreq"]) :, :]
#     avgMat[int(timeToAddToStart * evoked.info["sfreq"]) :, :, i] = evokedMat
#     scores += score

# scores /= i+1

# evokedAvgTargetMale = constructComplexList(
#     rampFunction,
#     avgMat,
#     evoked,
#     scores,
#     comment="T: Male",
#     complexSubtypeNames=complexSubtypeNames,
# )

# %%
# typeOfRegressor = "target"

# filenameSuffix = "_female"

# avgMat = np.zeros((lenResponse, nChannels, len(subjectsToAverage)))

# scores = np.zeros(nChannels)

# for i, subject in enumerate(subjectsToAverage):
#     megLocation = parentDir + subject + subDirs
#     evoked, score = readRecField(parentDir, subject, subDirs, filePrefix, typeOfRegressor, filenameSuffix, lambdaInd, nameOfRegressor, nameOfRegressors)
#     evokedMat = evoked.get_data().T[int(timeToAddToStart * evoked.info["sfreq"]) :, :]
#     avgMat[int(timeToAddToStart * evoked.info["sfreq"]) :, :, i] = evokedMat
#     scores += score

# scores /= i+1

# evokedAvgTargetFemale = constructComplexList(
#     rampFunction,
#     avgMat,
#     evoked,
#     scores,
#     comment="T: Female",
#     complexSubtypeNames=complexSubtypeNames,
# )

# %%
# typeOfRegressor = "mix"

# filenameSuffix = "_quiet_male"

# avgMat = np.zeros((lenResponse, nChannels, len(subjectsToAverage)))

# scores = np.zeros(nChannels)

# for i, subject in enumerate(subjectsToAverage):
#     megLocation = parentDir + subject + subDirs
#     evoked, score = readRecField(parentDir, subject, subDirs, filePrefix, typeOfRegressor, filenameSuffix, lambdaInd, nameOfRegressor, nameOfRegressors)
#     evokedMat = evoked.get_data().T[int(timeToAddToStart * evoked.info["sfreq"]) :, :]
#     avgMat[int(timeToAddToStart * evoked.info["sfreq"]) :, :, i] = evokedMat
#     scores += score

# scores /= i+1

# evokedAvgQuietMale = constructComplexList(
#     rampFunction,
#     avgMat,
#     evoked,
#     scores,
#     comment="T: Quiet male",
#     complexSubtypeNames=complexSubtypeNames,
# )

# %%
# typeOfRegressor = "mix"

# filenameSuffix = "_quiet_female"

# avgMat = np.zeros((lenResponse, nChannels, len(subjectsToAverage)))

# scores = np.zeros(nChannels)

# for i, subject in enumerate(subjectsToAverage):
#     megLocation = parentDir + subject + subDirs
#     evoked, score = readRecField(parentDir, subject, subDirs, filePrefix, typeOfRegressor, filenameSuffix, lambdaInd, nameOfRegressor, nameOfRegressors)
#     evokedMat = evoked.get_data().T[int(timeToAddToStart * evoked.info["sfreq"]) :, :]
#     avgMat[int(timeToAddToStart * evoked.info["sfreq"]) :, :, i] = evokedMat
#     scores += score

# scores /= i+1

# evokedAvgQuietFemale = constructComplexList(
#     rampFunction,
#     avgMat,
#     evoked,
#     scores,
#     comment="T: Quiet female",
#     complexSubtypeNames=complexSubtypeNames,
# )

# %%
# typeOfRegressor = "distractor"

# filenameSuffix = "_male"

# avgMat = np.zeros((lenResponse, nChannels, len(subjectsToAverage)))

# scores = np.zeros(nChannels)

# for i, subject in enumerate(subjectsToAverage):
#     megLocation = parentDir + subject + subDirs
#     evoked, score = readRecField(parentDir, subject, subDirs, filePrefix, typeOfRegressor, filenameSuffix, lambdaInd, nameOfRegressor, nameOfRegressors)
#     evokedMat = evoked.get_data().T[int(timeToAddToStart * evoked.info["sfreq"]) :, :]
#     avgMat[int(timeToAddToStart * evoked.info["sfreq"]) :, :, i] = evokedMat
#     scores += score

# scores /= i+1

# evokedAvgDistMale = constructComplexList(
#     rampFunction,
#     avgMat,
#     evoked,
#     scores,
#     comment="D: Male",
#     complexSubtypeNames=complexSubtypeNames,
# )

# %%
# typeOfRegressor = "distractor"

# filenameSuffix = "_female"

# avgMat = np.zeros((lenResponse, nChannels, len(subjectsToAverage)))

# scores = np.zeros(nChannels)

# for i, subject in enumerate(subjectsToAverage):
#     megLocation = parentDir + subject + subDirs
#     evoked, score = readRecField(parentDir, subject, subDirs, filePrefix, typeOfRegressor, filenameSuffix, lambdaInd, nameOfRegressor, nameOfRegressors)
#     evokedMat = evoked.get_data().T[int(timeToAddToStart * evoked.info["sfreq"]) :, :]
#     avgMat[int(timeToAddToStart * evoked.info["sfreq"]) :, :, i] = evokedMat
#     scores += score

# scores /= i+1

# evokedAvgDistFemale = constructComplexList(
#     rampFunction,
#     avgMat,
#     evoked,
#     scores,
#     comment="D: Female",
#     complexSubtypeNames=complexSubtypeNames,
# )

# %%
# typeOfRegressor = "mix"

# filenameSuffix = "_4dB"

# avgMat = np.zeros((lenResponse, nChannels, len(subjectsToAverage)))

# scores = np.zeros(nChannels)

# for i, subject in enumerate(subjectsToAverage):
#     megLocation = parentDir + subject + subDirs
#     evoked, score = readRecField(parentDir, subject, subDirs, filePrefix, typeOfRegressor, filenameSuffix, lambdaInd, nameOfRegressor, nameOfRegressors)
#     evokedMat = evoked.get_data().T[int(timeToAddToStart * evoked.info["sfreq"]) :, :]
#     avgMat[int(timeToAddToStart * evoked.info["sfreq"]) :, :, i] = evokedMat
#     scores += score

# scores /= i+1

# evokedAvgMix4dB = constructComplexList(
#     rampFunction,
#     avgMat,
#     evoked,
#     scores,
#     comment="M: 4 dB",
#     complexSubtypeNames=complexSubtypeNames,
# )

# %%
# typeOfRegressor = "mix"

# filenameSuffix = "_0dB"

# avgMat = np.zeros((lenResponse, nChannels, len(subjectsToAverage)))

# scores = np.zeros(nChannels)

# for i, subject in enumerate(subjectsToAverage):
#     megLocation = parentDir + subject + subDirs
#     evoked, score = readRecField(parentDir, subject, subDirs, filePrefix, typeOfRegressor, filenameSuffix, lambdaInd, nameOfRegressor, nameOfRegressors)
#     evokedMat = evoked.get_data().T[int(timeToAddToStart * evoked.info["sfreq"]) :, :]
#     avgMat[int(timeToAddToStart * evoked.info["sfreq"]) :, :, i] = evokedMat
#     scores += score

# scores /= i+1

# evokedAvgMix0dB = constructComplexList(
#     rampFunction,
#     avgMat,
#     evoked,
#     scores,
#     comment="M: 0 dB",
#     complexSubtypeNames=complexSubtypeNames,
# )

# %%
# typeOfRegressor = "mix"

# filenameSuffix = "_-4dB"

# avgMat = np.zeros((lenResponse, nChannels, len(subjectsToAverage)))

# scores = np.zeros(nChannels)

# for i, subject in enumerate(subjectsToAverage):
#     megLocation = parentDir + subject + subDirs
#     evoked, score = readRecField(parentDir, subject, subDirs, filePrefix, typeOfRegressor, filenameSuffix, lambdaInd, nameOfRegressor, nameOfRegressors)
#     evokedMat = evoked.get_data().T[int(timeToAddToStart * evoked.info["sfreq"]) :, :]
#     avgMat[int(timeToAddToStart * evoked.info["sfreq"]) :, :, i] = evokedMat
#     scores += score

# scores /= i+1

# evokedAvgMixNeg4dB = constructComplexList(
#     rampFunction,
#     avgMat,
#     evoked,
#     scores,
#     comment="M: -4 dB",
#     complexSubtypeNames=complexSubtypeNames,
# )

# %%
# typeOfRegressor = "target"

# filenameSuffix = "_4dB"

# avgMat = np.zeros((lenResponse, nChannels, len(subjectsToAverage)))

# scores = np.zeros(nChannels)

# for i, subject in enumerate(subjectsToAverage):
#     megLocation = parentDir + subject + subDirs
#     evoked, score = readRecField(parentDir, subject, subDirs, filePrefix, typeOfRegressor, filenameSuffix, lambdaInd, nameOfRegressor, nameOfRegressors)
#     evokedMat = evoked.get_data().T[int(timeToAddToStart * evoked.info["sfreq"]) :, :]
#     avgMat[int(timeToAddToStart * evoked.info["sfreq"]) :, :, i] = evokedMat
#     scores += score

# scores /= i+1

# evokedAvgTarget4dB = constructComplexList(
#     rampFunction,
#     avgMat,
#     evoked,
#     scores,
#     comment="T: 4 dB",
#     complexSubtypeNames=complexSubtypeNames,
# )

# %%
# typeOfRegressor = "target"

# filenameSuffix = "_0dB"

# avgMat = np.zeros((lenResponse, nChannels, len(subjectsToAverage)))

# scores = np.zeros(nChannels)

# for i, subject in enumerate(subjectsToAverage):
#     megLocation = parentDir + subject + subDirs
#     evoked, score = readRecField(parentDir, subject, subDirs, filePrefix, typeOfRegressor, filenameSuffix, lambdaInd, nameOfRegressor, nameOfRegressors)
#     evokedMat = evoked.get_data().T[int(timeToAddToStart * evoked.info["sfreq"]) :, :]
#     avgMat[int(timeToAddToStart * evoked.info["sfreq"]) :, :, i] = evokedMat
#     scores += score

# scores /= i+1

# evokedAvgTarget0dB = constructComplexList(
#     rampFunction,
#     avgMat,
#     evoked,
#     scores,
#     comment="T: 0 dB",
#     complexSubtypeNames=complexSubtypeNames,
# )

# %%
# typeOfRegressor = "target"

# filenameSuffix = "_-4dB"

# avgMat = np.zeros((lenResponse, nChannels, len(subjectsToAverage)))

# scores = np.zeros(nChannels)

# for i, subject in enumerate(subjectsToAverage):
#     megLocation = parentDir + subject + subDirs
#     evoked, score = readRecField(parentDir, subject, subDirs, filePrefix, typeOfRegressor, filenameSuffix, lambdaInd, nameOfRegressor, nameOfRegressors)
#     evokedMat = evoked.get_data().T[int(timeToAddToStart * evoked.info["sfreq"]) :, :]
#     avgMat[int(timeToAddToStart * evoked.info["sfreq"]) :, :, i] = evokedMat
#     scores += score

# scores /= i+1

# evokedAvgTargetNeg4dB = constructComplexList(
#     rampFunction,
#     avgMat,
#     evoked,
#     scores,
#     comment="T: -4 dB",
#     complexSubtypeNames=complexSubtypeNames,
# )

# %%
# typeOfRegressor = "distractor"

# filenameSuffix = "_4dB"

# avgMat = np.zeros((lenResponse, nChannels, len(subjectsToAverage)))

# scores = np.zeros(nChannels)

# for i, subject in enumerate(subjectsToAverage):
#     megLocation = parentDir + subject + subDirs
#     evoked, score = readRecField(parentDir, subject, subDirs, filePrefix, typeOfRegressor, filenameSuffix, lambdaInd, nameOfRegressor, nameOfRegressors)
#     evokedMat = evoked.get_data().T[int(timeToAddToStart * evoked.info["sfreq"]) :, :]
#     avgMat[int(timeToAddToStart * evoked.info["sfreq"]) :, :, i] = evokedMat
#     scores += score

# scores /= i+1

# evokedAvgDist4dB = constructComplexList(
#     rampFunction,
#     avgMat,
#     evoked,
#     scores,
#     comment="D: 4 dB",
#     complexSubtypeNames=complexSubtypeNames,
# )

# %%
# typeOfRegressor = "distractor"

# filenameSuffix = "_0dB"

# avgMat = np.zeros((lenResponse, nChannels, len(subjectsToAverage)))

# scores = np.zeros(nChannels)

# for i, subject in enumerate(subjectsToAverage):
#     megLocation = parentDir + subject + subDirs
#     evoked, score = readRecField(parentDir, subject, subDirs, filePrefix, typeOfRegressor, filenameSuffix, lambdaInd, nameOfRegressor, nameOfRegressors)
#     evokedMat = evoked.get_data().T[int(timeToAddToStart * evoked.info["sfreq"]) :, :]
#     avgMat[int(timeToAddToStart * evoked.info["sfreq"]) :, :, i] = evokedMat
#     scores += score

# scores /= i+1

# evokedAvgDist0dB = constructComplexList(
#     rampFunction,
#     avgMat,
#     evoked,
#     scores,
#     comment="D: 0 dB",
#     complexSubtypeNames=complexSubtypeNames,
# )

# %%
# typeOfRegressor = "distractor"

# filenameSuffix = "_-4dB"

# avgMat = np.zeros((lenResponse, nChannels, len(subjectsToAverage)))

# scores = np.zeros(nChannels)

# for i, subject in enumerate(subjectsToAverage):
#     megLocation = parentDir + subject + subDirs
#     evoked, score = readRecField(parentDir, subject, subDirs, filePrefix, typeOfRegressor, filenameSuffix, lambdaInd, nameOfRegressor, nameOfRegressors)
#     evokedMat = evoked.get_data().T[int(timeToAddToStart * evoked.info["sfreq"]) :, :]
#     avgMat[int(timeToAddToStart * evoked.info["sfreq"]) :, :, i] = evokedMat
#     scores += score

# scores /= i+1

# evokedAvgDistNeg4dB = constructComplexList(
#     rampFunction,
#     avgMat,
#     evoked,
#     scores,
#     comment="D: -4 dB",
#     complexSubtypeNames=complexSubtypeNames,
# )

# %%
# typeOfRegressor = "mix"

# filenameSuffix = "_easy"

# avgMat = np.zeros((lenResponse, nChannels, len(subjectsToAverage)))

# scores = np.zeros(nChannels)

# for i, subject in enumerate(subjectsToAverage):
#     megLocation = parentDir + subject + subDirs
#     evoked, score = readRecField(parentDir, subject, subDirs, filePrefix, typeOfRegressor, filenameSuffix, lambdaInd, nameOfRegressor, nameOfRegressors)
#     evokedMat = evoked.get_data().T[int(timeToAddToStart * evoked.info["sfreq"]) :, :]
#     avgMat[int(timeToAddToStart * evoked.info["sfreq"]) :, :, i] = evokedMat
#     scores += score

# scores /= i+1

# evokedAvgMixEasy = constructComplexList(
#     rampFunction,
#     avgMat,
#     evoked,
#     scores,
#     comment="M: Easy",
#     complexSubtypeNames=complexSubtypeNames,
# )

# %%
# typeOfRegressor = "mix"

# filenameSuffix = "_hard"

# avgMat = np.zeros((lenResponse, nChannels, len(subjectsToAverage)))

# scores = np.zeros(nChannels)

# for i, subject in enumerate(subjectsToAverage):
#     megLocation = parentDir + subject + subDirs
#     evoked, score = readRecField(parentDir, subject, subDirs, filePrefix, typeOfRegressor, filenameSuffix, lambdaInd, nameOfRegressor, nameOfRegressors)
#     evokedMat = evoked.get_data().T[int(timeToAddToStart * evoked.info["sfreq"]) :, :]
#     avgMat[int(timeToAddToStart * evoked.info["sfreq"]) :, :, i] = evokedMat
#     scores += score

# scores /= i+1

# evokedAvgMixHard = constructComplexList(
#     rampFunction,
#     avgMat,
#     evoked,
#     scores,
#     comment="M: Hard",
#     complexSubtypeNames=complexSubtypeNames,
# )

# %%
# mne.viz.plot_compare_evokeds([evokedAvgQuiet[0], evokedAvgQuiet[1], evokedAvgQuiet[2]],
#                              legend=True, axes="topo", colors=["blue", "blue", "red"],
#                              styles={"T: Quiet (real)": {"linewidth": 2, "alpha": .75},
#                                      "T: Quiet (imag)": {"linewidth": 2, "alpha": .75, "linestyle": "dashed"},
#                                      "T: Quiet (abs)": {"linewidth": 1, "alpha": .75}})

# %%
# mne.viz.plot_compare_evokeds([evokedAvgQuiet[3]], legend=True, axes="topo", colors=["blue"], styles={"T: Quiet (phase)": {"linewidth": 2, "alpha": .75}})

# %%
# mne.viz.plot_compare_evokeds([evokedAvgQuiet, evokedAvgTargetEasy, evokedAvgTargetHard, evokedAvgDistEasy, evokedAvgDistHard],
#                              legend=True, axes="topo", colors=["blue", "cyan", "green", "red", "brown"])

# %%
# mne.viz.plot_compare_evokeds([evokedAvgQuiet, evokedAvgTargetEasy, evokedAvgDistHard],
#                              legend=True, axes="topo", colors=["blue", "green", "red"],
#                              styles={"T: Quiet": {"linewidth": 3, "alpha": .75}, "T: Easy": {"linewidth": 3, "alpha": .75}, "D: Hard": {"linewidth": 3, "alpha": .75}})

# %%
# linewidth=3
# alpha=.75
# complexSubtype = 0
# mne.viz.plot_compare_evokeds([evokedAvgTargetEasy[complexSubtype], evokedAvgDistHard[complexSubtype], evokedAvgQuiet[complexSubtype]],
#                              legend=True, axes="topo", colors=["green", "red", "blue"],
#                              styles={"T: Quiet" + complexSubtypeNames[complexSubtype]: {"linewidth": linewidth, "alpha": alpha},
#                                      "T: Easy" + complexSubtypeNames[complexSubtype]: {"linewidth": linewidth, "alpha": alpha},
#                                      "D: Hard" + complexSubtypeNames[complexSubtype]: {"linewidth": linewidth, "alpha": alpha}})
# linewidth=3
# alpha=.75
# complexSubtype = 2
# mne.viz.plot_compare_evokeds([evokedAvgTargetEasy[complexSubtype], evokedAvgDistHard[complexSubtype], evokedAvgQuiet[complexSubtype]],
#                              legend=True, axes="topo", colors=["green", "red", "blue"],
#                              styles={"T: Quiet" + complexSubtypeNames[complexSubtype]: {"linewidth": linewidth, "alpha": alpha},
#                                      "T: Easy" + complexSubtypeNames[complexSubtype]: {"linewidth": linewidth, "alpha": alpha},
#                                      "D: Hard" + complexSubtypeNames[complexSubtype]: {"linewidth": linewidth, "alpha": alpha}})
# linewidth=3
# alpha=.75
# complexSubtype = 3
# mne.viz.plot_compare_evokeds([evokedAvgTargetEasy[complexSubtype], evokedAvgDistHard[complexSubtype], evokedAvgQuiet[complexSubtype]],
#                              legend=True, axes="topo", colors=["green", "red", "blue"],
#                              styles={"T: Quiet" + complexSubtypeNames[complexSubtype]: {"linewidth": linewidth, "alpha": alpha},
#                                      "T: Easy" + complexSubtypeNames[complexSubtype]: {"linewidth": linewidth, "alpha": alpha},
#                                      "D: Hard" + complexSubtypeNames[complexSubtype]: {"linewidth": linewidth, "alpha": alpha}})


# %%
linewidth = 3
alpha = 0.75
avgOrAll = 0  # 0 for pre-averaged TRFs, 1 for lists of evoked objects for each subject

complexSubtype = 0

condsToPlot = [
    evokedAvgQuiet,
    evokedAvgDistHard,
    evokedAvgTargetEasy,
    evokedAvgDistEasy,
    evokedAvgTargetHard,
]
colors = ["blue", "brown", "cyan", "red", "green"]

# condsToPlot = [evokedAvgDistHard, evokedAvgTargetEasy, evokedAvgDistEasy, evokedAvgTargetHard]
# colors = ["brown", "cyan", "red", "green"]

# condsToPlot = [evokedAvgQuiet, evokedAvgTargetEasy, evokedAvgTargetHard]
# colors = ["blue", "cyan", "green"]

# condsToPlot = [evokedAvgQuiet, evokedAvgMixHard, evokedAvgMixEasy]
# colors = ["blue", "red", "green"]

# condsToPlot = [evokedAvgQuiet, evokedAvgMix4dB, evokedAvgMix0dB, evokedAvgMixNeg4dB]
# colors = ["blue", "cyan", "xkcd:light green", "green"]

# condsToPlot = [evokedAvgMix4dB, evokedAvgMix0dB, evokedAvgMixNeg4dB]
# colors = ["cyan", "xkcd:light green", "green"]

# condsToPlot = [evokedAvgMixEasy, evokedAvgMixHard]
# colors = ["cyan", "red"]


# condsToPlot = [evokedAvgTarget4dB, evokedAvgTargetNeg4dB, evokedAvgDist4dB, evokedAvgDistNeg4dB]
# colors = ["blue", "green", "red", "brown"]

# condsToPlot = [evokedAvgQuiet, evokedAvgTarget4dB, evokedAvgTarget0dB, evokedAvgTargetNeg4dB]
# colors = ["blue", "cyan", "xkcd:light green", "green"]

# condsToPlot = [evokedAvgQuiet, evokedAvgDist4dB, evokedAvgDist0dB, evokedAvgDistNeg4dB]
# colors = ["blue", "pink", "red", "brown"]


# condsToPlot = [evokedAvgQuietMale, evokedAvgQuietFemale]
# colors = ["blue", "green"]

# condsToPlot = [evokedAvgTargetMale, evokedAvgTargetFemale]
# colors = ["blue", "green"]

# condsToPlot = [evokedAvgTargetMale, evokedAvgTargetFemale, evokedAvgDistMale, evokedAvgDistFemale]
# colors = ["blue", "green", "red", "brown"]

# condsToPlot = [evokedAvgQuietMale, evokedAvgTargetMale, evokedAvgQuietFemale, evokedAvgTargetFemale]
# colors = ["xkcd:blue", "xkcd:light blue", "xkcd:green", "xkcd:light green"]


keys = [cond[0][complexSubtype].comment for cond in condsToPlot]
evokedValues = [cond[avgOrAll][complexSubtype] for cond in condsToPlot]
evokeds = {k: v for (k, v) in zip(keys, evokedValues)}

styleValues = [
    {"linewidth": linewidth, "alpha": alpha, "color": colors[i]}
    for i in range(len(condsToPlot))
]
styles = {k: v for (k, v) in zip(keys, styleValues)}

mne.viz.plot_compare_evokeds(
    evokeds=evokeds, legend=True, axes="topo", styles=styles, ci=ci, picks="eeg"
)


# %%
mne.viz.plot_compare_evokeds(
    evokeds=evokeds, legend=True, axes="topo", styles=styles, ci=ci, picks="meg"
)

# %%
for i in range(len(evokedValues)):
    evokedValues[i].plot_joint(picks="meg", title=evokedValues[i].comment)

# %%
# fig1 = plt.figure(figsize=[18,5])

cmap="RdBu_r"
# cmap="Reds"

# for i in range(1, 1+len(condsToPlot)):

#     ax = fig1.add_subplot(1, len(condsToPlot), i)
    
#     im, cn = mne.viz.plot_topomap(condsToPlot[2][2][0:157], mne.pick_info(evokedValues[0].info, sel=np.arange(157)), cmap=cmap, axes=ax)
#     ax.colorbar(im)

maxAllEEG = 0
minAllEEG = 0
for i in range(len(condsToPlot)):
    maxAllEEG = np.array([maxAllEEG, condsToPlot[i][2][157:189].max()]).max()
    minAllEEG = np.array([minAllEEG, condsToPlot[i][2][157:189].min()]).min()
    

fig2, (ax5, ax6, ax7, ax8, ax9) = plt.subplots(ncols=5, figsize=[21, 6])

im, cn = mne.viz.plot_topomap(condsToPlot[0][2][157:189], mne.pick_info(evokedValues[0].info, sel=np.arange(157, 189)), cmap=cmap, axes=ax5, vlim=(minAllEEG, maxAllEEG))
ax5.set_title(condsToPlot[0][0][0].comment)
im, cn = mne.viz.plot_topomap(condsToPlot[2][2][157:189], mne.pick_info(evokedValues[0].info, sel=np.arange(157, 189)), cmap=cmap, axes=ax6, vlim=(minAllEEG, maxAllEEG))
ax6.set_title(condsToPlot[2][0][0].comment)
im, cn = mne.viz.plot_topomap(condsToPlot[4][2][157:189], mne.pick_info(evokedValues[0].info, sel=np.arange(157, 189)), cmap=cmap, axes=ax7, vlim=(minAllEEG, maxAllEEG))
ax7.set_title(condsToPlot[4][0][0].comment)
im, cn = mne.viz.plot_topomap(condsToPlot[1][2][157:189], mne.pick_info(evokedValues[0].info, sel=np.arange(157, 189)), cmap=cmap, axes=ax8, vlim=(minAllEEG, maxAllEEG))
ax8.set_title(condsToPlot[1][0][0].comment)
im, cn = mne.viz.plot_topomap(condsToPlot[3][2][157:189], mne.pick_info(evokedValues[0].info, sel=np.arange(157, 189)), cmap=cmap, axes=ax9, vlim=(minAllEEG, maxAllEEG))
ax9.set_title(condsToPlot[3][0][0].comment)
# im,cm   = mne.viz.plot_topomap(x2, info, axes=ax2,show=False,vmin=vmin,vmax=vmax)   
# manually fiddle the position of colorbar
# ax_x_start = 0.95
# ax_x_width = 0.04
# ax_y_start = 0.1
# ax_y_height = 0.9
# cbar_ax = fig.add_axes([ax_x_start, ax_y_start, ax_x_width, ax_y_height])
# clb = fig.colorbar(im, cax=cbar_ax)
# clb2 = fig2.colorbar(im)

# Create a new axes for the colorbar
cax2 = fig2.add_axes([0.92, 0.1, 0.03, 0.8]) # [left, bottom, width, height]


# Add the colorbar to the new axes
fig2.colorbar(im, cax=cax2)


fig2.suptitle(f"Lambda of {lambdas[lambdaInd]}, max Pearson r of {maxAllEEG}")

print(f"Max for EEG is {maxAllEEG}")

# clb.ax.set_title(unit_label,fontsize=fontsize) # title on top of colorbar

# %%
# fig1 = plt.figure(figsize=[18,5])

cmap="RdBu_r"
# cmap="Reds"

# for i in range(1, 1+len(condsToPlot)):

#     ax = fig1.add_subplot(1, len(condsToPlot), i)
    
#     im, cn = mne.viz.plot_topomap(condsToPlot[2][2][0:157], mne.pick_info(evokedValues[0].info, sel=np.arange(157)), cmap=cmap, axes=ax)
#     ax.colorbar(im)

maxAllMEG = 0
minAllMEG = 0
for i in range(len(condsToPlot)):
    maxAllMEG = np.array([maxAllMEG, condsToPlot[i][2][0:157].max()]).max()
    minAllMEG = np.array([minAllMEG, condsToPlot[i][2][0:157].min()]).min()
    

fig1, (ax0, ax1, ax2, ax3, ax4) = plt.subplots(ncols=5, figsize=[21, 6])

im, cn = mne.viz.plot_topomap(condsToPlot[0][2][0:157], mne.pick_info(evokedValues[0].info, sel=np.arange(157)), cmap=cmap, axes=ax0, vlim=(minAllMEG, maxAllMEG))
ax0.set_title(condsToPlot[0][0][0].comment)
im, cn = mne.viz.plot_topomap(condsToPlot[2][2][0:157], mne.pick_info(evokedValues[0].info, sel=np.arange(157)), cmap=cmap, axes=ax1, vlim=(minAllMEG, maxAllMEG))
ax1.set_title(condsToPlot[2][0][0].comment)
im, cn = mne.viz.plot_topomap(condsToPlot[4][2][0:157], mne.pick_info(evokedValues[0].info, sel=np.arange(157)), cmap=cmap, axes=ax2, vlim=(minAllMEG, maxAllMEG))
ax2.set_title(condsToPlot[4][0][0].comment)
im, cn = mne.viz.plot_topomap(condsToPlot[1][2][0:157], mne.pick_info(evokedValues[0].info, sel=np.arange(157)), cmap=cmap, axes=ax3, vlim=(minAllMEG, maxAllMEG))
ax3.set_title(condsToPlot[1][0][0].comment)
im, cn = mne.viz.plot_topomap(condsToPlot[3][2][0:157], mne.pick_info(evokedValues[0].info, sel=np.arange(157)), cmap=cmap, axes=ax4, vlim=(minAllMEG, maxAllMEG))
ax4.set_title(condsToPlot[3][0][0].comment)
# im,cm   = mne.viz.plot_topomap(x2, info, axes=ax2,show=False,vmin=vmin,vmax=vmax)   
# manually fiddle the position of colorbar
# ax_x_start = 0.95
# ax_x_width = 0.04
# ax_y_start = 0.1
# ax_y_height = 0.9
# cbar_ax = fig.add_axes([ax_x_start, ax_y_start, ax_x_width, ax_y_height])
# clb = fig.colorbar(im, cax=cbar_ax)



# clb1 = fig1.colorbar(im)

# Create a new axes for the colorbar
cax1 = fig1.add_axes([0.92, 0.1, 0.03, 0.8]) # [left, bottom, width, height]

# Add the colorbar to the new axes
fig1.colorbar(im, cax=cax1)

fig1.suptitle(f"Lambda of {lambdas[lambdaInd]}, max Pearson r of {maxAllMEG}")

print(f"Max for MEG is {maxAllMEG}")

# clb.ax.set_title(unit_label,fontsize=fontsize) # title on top of colorbar

# %%
# mne.viz.plot_evoked_topo([evokedValues[i].copy().pick("mag") for i in range(len(evokedValues))])



# %%
# for i in range(len(evokedValues)):
#     evokedValues[i].plot_joint(picks="eeg", title=evokedValues[i].comment)

# %%

# %%
# timeToZoom = .1
# samplesToZoom = int(timeToZoom * evokedAvgQuiet[0].info["sfreq"])
# elecInd = evokedAvgQuiet[0].ch_names.index("Fz")
# plt.figure(figsize=[10,10])
# plt.polar(evokedAvgQuiet[3].get_data()[elecInd,samplesToZoom:-samplesToZoom].T,evokedAvgQuiet[2].get_data()[elecInd,samplesToZoom:-samplesToZoom].T, color="blue", linewidth=3)
# plt.polar(evokedAvgTargetEasy[3].get_data()[elecInd,samplesToZoom:-samplesToZoom].T,evokedAvgTargetEasy[2].get_data()[elecInd,samplesToZoom:-samplesToZoom].T, color="cyan", linewidth=3)
# plt.polar(evokedAvgTargetHard[3].get_data()[elecInd,samplesToZoom:-samplesToZoom].T,evokedAvgTargetHard[2].get_data()[elecInd,samplesToZoom:-samplesToZoom].T, color="green", linewidth=3)
# plt.polar(evokedAvgDistEasy[3].get_data()[elecInd,samplesToZoom:-samplesToZoom].T,evokedAvgTargetEasy[2].get_data()[elecInd,samplesToZoom:-samplesToZoom].T, color="red", linewidth=3)
# plt.polar(evokedAvgDistHard[3].get_data()[elecInd,samplesToZoom:-samplesToZoom].T,evokedAvgTargetHard[2].get_data()[elecInd,samplesToZoom:-samplesToZoom].T, color="brown", linewidth=3)
# plt.plot(evokedAvgQuiet[0].get_data()[25:32,samplesToZoom:-samplesToZoom].T, evokedAvgQuiet[1].get_data()[25:32,samplesToZoom:-samplesToZoom].T)
